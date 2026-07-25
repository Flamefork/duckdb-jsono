#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_extension.hpp"
#include "jsono_memory.hpp"
#include "jsono_merge_core.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_reconstruct.hpp"
#include "jsono_row_read.hpp"
#include "jsono_scalar_write.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// ===== Order-independent last-write-wins group merge (jsono_group_merge_max / _min) =====
//
// jsono_group_merge(value ORDER BY key) is order_dependent, so DuckDB drives it through the
// ordered-aggregate path that buffers and sorts EVERY input row before folding — O(rows) memory.
// These keyed variants take the ordering key as an ordinary second argument and resolve per-leaf
// conflicts themselves, so the aggregate is commutative and associative (registered WITHOUT
// order_dependent): DuckDB streams rows straight into Update with no buffer and state stays
// O(distinct leaves per group).
//
// Drop-in semantics (identical to the ORDER BY form on structurally-stable data):
//   _max(value, key)  ==  jsono_group_merge(value ORDER BY key)        — greatest key wins per leaf
//   _min(value, key)  ==  jsono_group_merge(value ORDER BY key DESC)   — smallest key wins per leaf
// Per leaf the value from the row with the winning key wins; null object members never overwrite
// (RFC 7396 IGNORE NULLS, exactly like jsono_group_merge). The key is any comparable type —
// composite keys via ROW(...)/STRUCT compare lexicographically — encoded once per row into a
// sort-key blob (CreateSortKey, NULLS FIRST so a NULL key never wins) so per-leaf comparison is a
// memcmp and arbitrary types/null ordering are handled by DuckDB's own sort encoding. Exact ties
// on the key fall back to a deterministic comparison of the value bytes so the result is fully
// order-independent; supply a unique (composite) key when ties must resolve a specific way.
//
// State is a heap-owned mutable tree. Object nodes keep sorted JSON-key children; leaf nodes keep
// the winning sort-key bytes plus a standalone JSONO blob for the winning non-object value. An
// object<->leaf change at the same path fails loud: a commutative per-leaf aggregate cannot
// reproduce the order-dependent RFC 7396 fold for structurally-inconsistent rows.

struct GroupMergeLWWBindData : public FunctionData {
	OrderModifiers modifiers;
	// Same sticky-shredded contract as jsono_group_merge: the result stays shredded in this type
	// (empty for a plain input). Drives the direct fold plan; only when that path declines does
	// Finalize reshred a plain result back into this type.
	vector<std::pair<string, LogicalType>> shreds;
	// Carried into Update/Combine to account the LWW tree/lane growth; re-captured on plan round-trips
	// (no serialize callback, so deserialize re-runs the bind).
	BufferManager &buffer_manager;

	GroupMergeLWWBindData(OrderModifiers modifiers, BufferManager &buffer_manager)
	    : modifiers(modifiers), buffer_manager(buffer_manager) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<GroupMergeLWWBindData>(modifiers, buffer_manager);
		result->shreds = shreds;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<GroupMergeLWWBindData>();
		return modifiers == other.modifiers && shreds == other.shreds;
	}
};

enum class LWWTreeKind : uint8_t { Object, Leaf };

struct LWWTreeNode;

struct LWWScalarValue {
	uint64_t slot = 0;
	uint64_t num = 0;
	uint32_t length = 0;
};

constexpr uint32_t LWW_INVALID_SORT_KEY_ID = std::numeric_limits<uint32_t>::max();

struct LWWScalarLane {
	bool has_value = false;
	uint32_t sort_key_id = LWW_INVALID_SORT_KEY_ID;
	uint64_t value_bits = 0;
};

struct LWWListElement {
	LWWScalarValue value;
	string text;
};

struct LWWListValue {
	vector<LWWListElement> elements;
	OwnedJsonoBlob skeleton;
};

struct LWWListLane {
	bool has_value = false;
	uint32_t sort_key_id = LWW_INVALID_SORT_KEY_ID;
	LWWListValue value;
};

struct GroupMergeLWWState {
	LWWTreeNode *root;
	LWWScalarLane *scalar_lanes;
	idx_t scalar_lane_count;
	string *scalar_texts;
	idx_t scalar_text_lane_count;
	LWWListLane *list_lanes;
	idx_t list_lane_count;
	vector<string> *lane_sort_keys;
	bool has_input;
	JsonoMemoryReservation mem;
	// Additions-only accounting for keyed Update: every fold that grows the state adds the exact growth to
	// `mem_pending`; the account step reserves that (throwing OOM) and periodically trues up. `mem_unmeasured`
	// is the bytes reserved since the last exact footprint walk, and drives the geometric re-measure. Combine
	// stays walk-based and resets both after its exact Resize.
	idx_t mem_pending;
	idx_t mem_unmeasured;
};

struct GroupMergeLWWFunction {
	static void Initialize(GroupMergeLWWState &state) {
		state.root = nullptr;
		state.scalar_lanes = nullptr;
		state.scalar_lane_count = 0;
		state.scalar_texts = nullptr;
		state.scalar_text_lane_count = 0;
		state.list_lanes = nullptr;
		state.list_lane_count = 0;
		state.lane_sort_keys = nullptr;
		state.has_input = false;
		state.mem.reserved = 0;
		state.mem.buffer_manager = nullptr;
		state.mem_pending = 0;
		state.mem_unmeasured = 0;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		state.mem.Release();
		delete state.root;
		delete[] state.scalar_lanes;
		delete[] state.scalar_texts;
		delete[] state.list_lanes;
		delete state.lane_sort_keys;
		state.root = nullptr;
		state.scalar_lanes = nullptr;
		state.scalar_lane_count = 0;
		state.scalar_texts = nullptr;
		state.scalar_text_lane_count = 0;
		state.list_lanes = nullptr;
		state.list_lane_count = 0;
		state.lane_sort_keys = nullptr;
		state.has_input = false;
		state.mem_pending = 0;
		state.mem_unmeasured = 0;
	}
};

struct LWWObjectChild {
	string key;
	unique_ptr<LWWTreeNode> node;

	LWWObjectChild(string key, unique_ptr<LWWTreeNode> node) : key(std::move(key)), node(std::move(node)) {
	}

	LWWObjectChild(LWWObjectChild &&) = default;
	LWWObjectChild &operator=(LWWObjectChild &&) = default;
	LWWObjectChild(const LWWObjectChild &) = delete;
	LWWObjectChild &operator=(const LWWObjectChild &) = delete;
};

struct LWWTreeNode {
	LWWTreeKind kind = LWWTreeKind::Object;
	vector<LWWObjectChild> children;
	string sort_key;
	OwnedJsonoBlob value;
};

struct LWWTreeScratch {
	JsonoBuilder builder;
	OwnedJsonoBlob candidate;
	// Owned-heap growth accumulated by one MergeIncomingLWWNode call, so a per-row fold can add it to the
	// group's mem_pending without re-walking the tree. Reset before each top-level merge (see FoldRowLWW).
	idx_t bytes_added = 0;
};

// Owned-heap bytes of one LWW tree node (its winning value blob, sort key, and object children,
// recursively). Walks the whole subtree, so keyed group_merge accounts per Update/Combine call rather
// than per row (see AccountDistinctStates).
idx_t LWWTreeNodeBytes(const LWWTreeNode &node) {
	idx_t bytes = sizeof(LWWTreeNode) + node.sort_key.capacity() + BlobBytes(node.value);
	bytes += node.children.capacity() * sizeof(LWWObjectChild);
	for (auto &child : node.children) {
		bytes += child.key.capacity();
		if (child.node) {
			bytes += LWWTreeNodeBytes(*child.node);
		}
	}
	return bytes;
}

// Owned-heap bytes of one direct list-shred lane value (its skeleton blob and element texts). Shared by
// the footprint walk and the incremental MergeLWWListLane delta so both measure the lane identically.
idx_t LWWListValueBytes(const LWWListValue &value) {
	idx_t bytes = BlobBytes(value.skeleton);
	bytes += value.elements.capacity() * sizeof(LWWListElement);
	for (auto &element : value.elements) {
		bytes += element.text.capacity();
	}
	return bytes;
}

idx_t GroupMergeLWWFootprint(const GroupMergeLWWState &state) {
	idx_t bytes = 0;
	if (state.root) {
		bytes += LWWTreeNodeBytes(*state.root);
	}
	bytes += state.scalar_lane_count * sizeof(LWWScalarLane);
	if (state.scalar_texts) {
		for (idx_t i = 0; i < state.scalar_text_lane_count; i++) {
			bytes += state.scalar_texts[i].capacity();
		}
	}
	bytes += state.list_lane_count * sizeof(LWWListLane);
	if (state.list_lanes) {
		for (idx_t i = 0; i < state.list_lane_count; i++) {
			bytes += LWWListValueBytes(state.list_lanes[i].value);
		}
	}
	if (state.lane_sort_keys) {
		bytes += state.lane_sort_keys->capacity() * sizeof(string);
		for (auto &key : *state.lane_sort_keys) {
			bytes += key.capacity();
		}
	}
	return bytes;
}

// Re-walk the exact footprint only after this many bytes of additions-only drift have been reserved (or
// after the state has doubled, whichever is larger). Keeps the O(tree) footprint walk amortized O(1) per
// growing chunk while bounding how far the additions-only over-estimate can drift above the true size.
constexpr idx_t kRemeasureMinBytes = 64 * 1024;

// Incremental accounting for keyed group_merge Update. Each fold pushes its owned-heap growth into
// state.mem_pending; here we Reserve that (which throws the engine OOM if it breaches max_memory), then
// periodically Resize to the exact footprint to shed the additions-only over-estimate. The pending == 0
// fast path makes steady-state LWW (rows that lose the merge and grow nothing) skip the footprint walk
// entirely -- that is what removes the per-row O(tree) cost regression. Dedupe across rows sharing a state
// is free: the first row reserves its pending and zeroes it, so later rows on the same state short-circuit.
template <class StateFor>
void AccountLWWUpdateStates(idx_t count, BufferManager &bm, StateFor state_for) {
	for (idx_t row = 0; row < count; row++) {
		auto &s = state_for(row);
		if (s.mem_pending == 0) {
			continue;
		}
		// Additions-only over-bound: reserved + pending must never be below the true footprint. A breach
		// means a growth site forgot to add to mem_pending -- fix the site, do not relax this.
		D_ASSERT(GroupMergeLWWFootprint(s) <= s.mem.reserved + s.mem_pending);
		s.mem.Reserve(bm, s.mem_pending);
		s.mem_unmeasured += s.mem_pending;
		s.mem_pending = 0;
		idx_t baseline = s.mem.reserved - s.mem_unmeasured;
		if (s.mem_unmeasured >= MaxValue<idx_t>(kRemeasureMinBytes, baseline)) {
			s.mem.Resize(bm, GroupMergeLWWFootprint(s));
			s.mem_unmeasured = 0;
		}
	}
}

int CompareRawBytes(const char *a, size_t na, const char *b, size_t nb) {
	auto n = std::min(na, nb);
	if (n > 0) {
		int c = std::memcmp(a, b, n);
		if (c != 0) {
			return c;
		}
	}
	return na < nb ? -1 : (na > nb ? 1 : 0);
}

// std::string binds here via the implicit string_view conversion, yielding the same pointer+size
// the explicit per-type overloads built by hand. One wrapper covers every (string|string_view) pair.
int CompareRawBytes(nonstd::string_view a, nonstd::string_view b) {
	return CompareRawBytes(a.data(), a.size(), b.data(), b.size());
}

void AssignBytes(string &out, nonstd::string_view bytes) {
	out.resize(bytes.size());
	if (bytes.size() > 0) {
		std::memcpy(&out[0], bytes.data(), bytes.size());
	}
}

// Sort-key ids are references to stored bytes, not canonical dictionary ids; all equality/order checks
// compare bytes, so non-adjacent duplicate keys are correct and avoid quadratic global dedupe.
uint32_t StoreLWWLaneSortKey(GroupMergeLWWState &state, nonstd::string_view K) {
	if (!state.lane_sort_keys) {
		state.lane_sort_keys = new vector<string>();
	}
	auto &keys = *state.lane_sort_keys;
	if (!keys.empty() && CompareRawBytes(keys.back(), K) == 0) {
		return uint32_t(keys.size() - 1);
	}
	if (keys.size() >= LWW_INVALID_SORT_KEY_ID) {
		throw InternalException("jsono_group_merge: too many distinct lane sort keys in one aggregate state");
	}
	string copy;
	AssignBytes(copy, K);
	idx_t cap_before = keys.capacity();
	keys.push_back(std::move(copy));
	state.mem_pending += keys.back().capacity() + (keys.capacity() - cap_before) * sizeof(string);
	return uint32_t(keys.size() - 1);
}

nonstd::string_view LWWLaneSortKey(const GroupMergeLWWState &state, uint32_t sort_key_id) {
	if (!state.lane_sort_keys || sort_key_id >= state.lane_sort_keys->size()) {
		throw InternalException("jsono_group_merge: invalid lane sort-key reference");
	}
	auto &key = (*state.lane_sort_keys)[sort_key_id];
	return nonstd::string_view(key.data(), key.size());
}

void WriteScalarLeafHeader(OwnedJsonoBlob &out, uint64_t slot) {
	out.slots.resize(JSONO_HEADER_SIZE + sizeof(uint64_t));
	WriteJsonoHeaderInto(reinterpret_cast<uint8_t *>(&out.slots[0]), flags::SORTED_KEYS);
	std::memcpy(&out.slots[JSONO_HEADER_SIZE], &slot, sizeof(uint64_t));
	out.key_heap.clear();
	out.string_heap.clear();
	out.lengths.clear();
	out.nums.clear();
	out.skips.resize(sizeof(ContainerMetadataHeader));
	WriteEmptyMetadataInto(reinterpret_cast<uint8_t *>(&out.skips[0]));
}

void StoreScalarLeafBlob(const JsonoView &view, JsonoCursor &cursor, OwnedJsonoBlob &out) {
	auto slot = view.SlotAt(cursor.pos);
	WriteScalarLeafHeader(out, slot);
	switch (ClassifyRawScalarSlot(slot)) {
	case RawScalarValueKind::LengthHeap: {
		auto len = view.LengthAt(cursor.length_cursor);
		auto s = view.StringAt(cursor.string_cursor, len);
		out.string_heap.assign(s.data(), s.size());
		out.lengths.assign(reinterpret_cast<const char *>(&len), sizeof(uint32_t));
		cursor.string_cursor += len;
		cursor.length_cursor++;
		cursor.pos++;
		return;
	}
	case RawScalarValueKind::Number: {
		auto num = view.NumAt(cursor.num_cursor);
		out.nums.assign(reinterpret_cast<const char *>(&num), sizeof(uint64_t));
		cursor.num_cursor++;
		cursor.pos++;
		return;
	}
	case RawScalarValueKind::Literal:
		cursor.pos++;
		return;
	}
}

// Deterministic total order over two standalone JSONO leaf blobs, used only to break an exact key
// tie so the fold stays order-independent. This matches the old builder-vector order: slots
// without JSONO header, then string_heap, nums, key_heap, lengths.
int CompareBlobValueTie(const OwnedJsonoBlob &a, const OwnedJsonoBlob &b) {
	if (a.slots.size() < JSONO_HEADER_SIZE || b.slots.size() < JSONO_HEADER_SIZE) {
		throw InternalException("jsono_group_merge: malformed leaf blob");
	}
	if (int c = CompareRawBytes(a.slots.data() + JSONO_HEADER_SIZE, a.slots.size() - JSONO_HEADER_SIZE,
	                            b.slots.data() + JSONO_HEADER_SIZE, b.slots.size() - JSONO_HEADER_SIZE)) {
		return c;
	}
	if (int c = CompareRawBytes(a.string_heap, b.string_heap)) {
		return c;
	}
	if (int c = CompareRawBytes(a.nums, b.nums)) {
		return c;
	}
	if (int c = CompareRawBytes(a.key_heap, b.key_heap)) {
		return c;
	}
	return CompareRawBytes(a.lengths, b.lengths);
}

bool LWWScalarValueHasNum(const LWWScalarValue &value) {
	auto slot_tag = SlotTag(value.slot);
	if (slot_tag == tag::VAL_INT60 || slot_tag == tag::VAL_DEC60) {
		return true;
	}
	if (slot_tag != tag::VAL_EXT) {
		return false;
	}
	auto subtype = ExtSubtype(value.slot);
	return subtype == ext_subtype::INT64 || subtype == ext_subtype::UINT64 || subtype == ext_subtype::DOUBLE;
}

bool LWWScalarValueHasLength(const LWWScalarValue &value) {
	auto slot_tag = SlotTag(value.slot);
	if (slot_tag == tag::VAL_STR_HEAP) {
		return true;
	}
	return slot_tag == tag::VAL_EXT && ExtSubtype(value.slot) == ext_subtype::NUMBER;
}

void SerializeLWWScalarValueToBlob(const LWWScalarValue &value, nonstd::string_view text, OwnedJsonoBlob &out) {
	WriteScalarLeafHeader(out, value.slot);
	if (LWWScalarValueHasLength(value)) {
		out.string_heap.assign(text.data(), text.size());
		out.lengths.assign(reinterpret_cast<const char *>(&value.length), sizeof(uint32_t));
	}
	if (LWWScalarValueHasNum(value)) {
		out.nums.assign(reinterpret_cast<const char *>(&value.num), sizeof(uint64_t));
	}
}

void EmitLWWScalarValue(const LWWScalarValue &value, nonstd::string_view text, const LogicalType &type,
                        JsonoBuilder &builder) {
	auto primitive = JsonoScalarPrimitiveFromType(type, "jsono_group_merge direct list lane");
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar:
		builder.EmitString(text);
		return;
	case JsonoScalarPrimitive::Bigint: {
		int64_t v;
		std::memcpy(&v, &value.num, sizeof(v));
		builder.EmitInt(v);
		return;
	}
	case JsonoScalarPrimitive::Ubigint:
		builder.EmitUInt(value.num);
		return;
	case JsonoScalarPrimitive::Double: {
		double v;
		std::memcpy(&v, &value.num, sizeof(v));
		builder.EmitDouble(v);
		return;
	}
	case JsonoScalarPrimitive::Boolean:
		builder.EmitBool(SlotTag(value.slot) == tag::VAL_TRUE);
		return;
	}
}

void SerializeLWWListValueToBlob(const LWWListValue &value, const LogicalType &type, OwnedJsonoBlob &out,
                                 JsonoBuilder &builder) {
	builder.Reset();
	builder.EmitArrayStart();
	auto &element_type = ListType::GetChildType(type);
	for (auto &element : value.elements) {
		EmitLWWScalarValue(element.value, nonstd::string_view(element.text.data(), element.text.size()), element_type,
		                   builder);
	}
	builder.EmitArrayEnd();
	SerializeBuilderToBlob(builder, out);
}

int CompareLWWListValueTie(const LWWListValue &a, const LWWListValue &b, const LogicalType &type) {
	static thread_local JsonoBuilder builder_a;
	static thread_local JsonoBuilder builder_b;
	static thread_local OwnedJsonoBlob blob_a;
	static thread_local OwnedJsonoBlob blob_b;
	SerializeLWWListValueToBlob(a, type, blob_a, builder_a);
	SerializeLWWListValueToBlob(b, type, blob_b, builder_b);
	return CompareBlobValueTie(blob_a, blob_b);
}

int CompareLWWListValueTie(const LWWListValue &a, const LogicalType &type, const OwnedJsonoBlob &b) {
	static thread_local JsonoBuilder builder;
	static thread_local OwnedJsonoBlob blob;
	SerializeLWWListValueToBlob(a, type, blob, builder);
	return CompareBlobValueTie(blob, b);
}

// Per-leaf last-write-wins is well-defined only when every path has a consistent kind across rows
// (always an object, or always a non-object leaf). When a path is a non-empty object in one row and
// a scalar/array in another, the sequential RFC 7396 fold's result depends on row order, so a
// commutative per-leaf rule cannot reproduce it — failing loud beats a silently order-dependent
// answer. Empty object members are stripped before touching a child, so a nested OBJ_START conflict
// is a real structural object (the document root can still be an empty object input).
[[noreturn]] void ThrowMixedKindConflict() {
	throw InvalidInputException(
	    "jsono_group_merge_max/min: a path is an object in one row and a scalar/array in another; "
	    "last-write-wins per leaf is order-dependent for such structurally-inconsistent data. "
	    "Use jsono_group_merge(value ORDER BY key) instead for paths that change kind across rows.");
}

size_t FindLWWChildIndex(const vector<LWWObjectChild> &children, nonstd::string_view key) {
	size_t lo = 0;
	size_t hi = children.size();
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto &stored = children[mid].key;
		if (CompareJsonoKeys(nonstd::string_view(stored.data(), stored.size()), key) < 0) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	return lo;
}

void ResetLWWNodeToObject(LWWTreeNode &node) {
	node.kind = LWWTreeKind::Object;
	node.children.clear();
	node.sort_key.clear();
	node.value = OwnedJsonoBlob();
}

bool LWWRootKeySkipped(const vector<nonstd::string_view> *root_skip_keys, size_t depth, nonstd::string_view key) {
	if (!root_skip_keys || depth != 0) {
		return false;
	}
	for (auto skip_key : *root_skip_keys) {
		if (skip_key == key) {
			return true;
		}
	}
	return false;
}

void SerializeIncomingLWWLeaf(const JsonoView &view, JsonoCursor &cursor, OwnedJsonoBlob &out, LWWTreeScratch &scratch,
                              size_t depth) {
	if (SlotTag(view.SlotAt(cursor.pos)) != tag::ARR_START) {
		StoreScalarLeafBlob(view, cursor, out);
		return;
	}
	scratch.builder.Reset();
	EmitValueVerbatim(view, cursor, scratch.builder, depth);
	SerializeBuilderToBlob(scratch.builder, out);
}

void StoreLWWLeafFromIncoming(LWWTreeNode &node, const JsonoView &view, JsonoCursor &cursor, nonstd::string_view K,
                              LWWTreeScratch &scratch, size_t depth) {
	// Metadata before the value emit: the keyed tie-break reads node.sort_key.
	node.kind = LWWTreeKind::Leaf;
	node.children.clear();
	AssignBytes(node.sort_key, K);
	SerializeIncomingLWWLeaf(view, cursor, node.value, scratch, depth);
}

void BuildIncomingLWWNode(LWWTreeNode &node, const JsonoView &V, JsonoCursor &cursor, nonstd::string_view K,
                          LWWTreeScratch &scratch, size_t depth,
                          const vector<nonstd::string_view> *root_skip_keys = nullptr);

void BuildIncomingLWWObject(LWWTreeNode &node, const JsonoView &V, JsonoCursor &cursor, nonstd::string_view K,
                            LWWTreeScratch &scratch, size_t depth,
                            const vector<nonstd::string_view> *root_skip_keys = nullptr) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	ResetLWWNodeToObject(node);
	auto layout = ReadObjectLayout(V, cursor.pos);
	JsonoCursor vc = cursor;
	vc.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = V.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		if (!MemberSurvivesStrip(V, vc.pos, MergeMode::IgnoreNulls)) {
			SkipValueFast(V, vc);
			continue;
		}
		auto key = V.KeyAt(SlotPayload(key_slot));
		if (LWWRootKeySkipped(root_skip_keys, depth, key)) {
			SkipValueFast(V, vc);
			continue;
		}
		string key_copy;
		AssignBytes(key_copy, key);
		auto child = make_uniq<LWWTreeNode>();
		BuildIncomingLWWNode(*child, V, vc, K, scratch, depth + 1, root_skip_keys);
		node.children.emplace_back(std::move(key_copy), std::move(child));
	}
	if (vc.pos >= V.Slots() || SlotTag(V.SlotAt(vc.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	vc.pos++;
	cursor = vc;
}

void BuildIncomingLWWNode(LWWTreeNode &node, const JsonoView &V, JsonoCursor &cursor, nonstd::string_view K,
                          LWWTreeScratch &scratch, size_t depth, const vector<nonstd::string_view> *root_skip_keys) {
	auto slot_tag = SlotTag(V.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		BuildIncomingLWWObject(node, V, cursor, K, scratch, depth, root_skip_keys);
		return;
	}
	// Arrays are a single leaf (replaced wholesale), exactly like jsono_group_merge. Scalars,
	// including a top-level JSON null, are leaves too.
	StoreLWWLeafFromIncoming(node, V, cursor, K, scratch, depth);
}

void MergeIncomingLWWNode(LWWTreeNode &node, const JsonoView &V, JsonoCursor &cursor, nonstd::string_view K,
                          LWWTreeScratch &scratch, size_t depth,
                          const vector<nonstd::string_view> *root_skip_keys = nullptr) {
	auto slot_tag = SlotTag(V.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		if (node.kind != LWWTreeKind::Object) {
			ThrowMixedKindConflict();
		}
		if (depth > JSONO_MAX_NESTING_DEPTH) {
			throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
			                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
		}
		auto layout = ReadObjectLayout(V, cursor.pos);
		JsonoCursor vc = cursor;
		vc.pos = layout.value_start;
		idx_t child_pos = 0;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = V.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			auto key = V.KeyAt(SlotPayload(key_slot));
			if (!MemberSurvivesStrip(V, vc.pos, MergeMode::IgnoreNulls)) {
				SkipValueFast(V, vc);
				continue;
			}
			if (LWWRootKeySkipped(root_skip_keys, depth, key)) {
				SkipValueFast(V, vc);
				continue;
			}
			while (child_pos < node.children.size() &&
			       CompareJsonoKeys(
			           nonstd::string_view(node.children[child_pos].key.data(), node.children[child_pos].key.size()),
			           key) < 0) {
				child_pos++;
			}
			auto child_idx = child_pos;
			bool found = child_idx < node.children.size() &&
			             CompareJsonoKeys(nonstd::string_view(node.children[child_idx].key.data(),
			                                                  node.children[child_idx].key.size()),
			                              key) == 0;
			if (found) {
				MergeIncomingLWWNode(*node.children[child_idx].node, V, vc, K, scratch, depth + 1, root_skip_keys);
				child_pos = child_idx + 1;
			} else {
				string key_copy;
				AssignBytes(key_copy, key);
				auto child = make_uniq<LWWTreeNode>();
				BuildIncomingLWWNode(*child, V, vc, K, scratch, depth + 1, root_skip_keys);
				// Whole fresh subtree counted once here (its interior grows are BuildIncoming-owned and left
				// uninstrumented); plus the inserted key bytes and any children-vector reallocation growth.
				idx_t cap_before = node.children.capacity();
				node.children.insert(node.children.begin() + child_idx,
				                     LWWObjectChild(std::move(key_copy), std::move(child)));
				auto &inserted = node.children[child_idx];
				scratch.bytes_added += LWWTreeNodeBytes(*inserted.node) + inserted.key.capacity() +
				                       (node.children.capacity() - cap_before) * sizeof(LWWObjectChild);
				child_pos = child_idx + 1;
			}
		}
		if (vc.pos >= V.Slots() || SlotTag(V.SlotAt(vc.pos)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		vc.pos++;
		cursor = vc;
		return;
	}
	if (node.kind != LWWTreeKind::Leaf) {
		ThrowMixedKindConflict();
	}
	int key_cmp = CompareRawBytes(node.sort_key, K);
	if (key_cmp > 0) {
		SkipValueFast(V, cursor);
		return;
	}
	if (key_cmp < 0) {
		idx_t before = node.sort_key.capacity() + BlobBytes(node.value);
		StoreLWWLeafFromIncoming(node, V, cursor, K, scratch, depth);
		idx_t after = node.sort_key.capacity() + BlobBytes(node.value);
		scratch.bytes_added += (after > before ? after - before : 0);
		return;
	}
	SerializeIncomingLWWLeaf(V, cursor, scratch.candidate, scratch, depth);
	if (CompareBlobValueTie(node.value, scratch.candidate) < 0) {
		idx_t before = BlobBytes(node.value);
		node.value = scratch.candidate;
		idx_t after = BlobBytes(node.value);
		scratch.bytes_added += (after > before ? after - before : 0);
	}
}

unique_ptr<LWWTreeNode> CloneLWWTreeNode(const LWWTreeNode &source) {
	auto result = make_uniq<LWWTreeNode>();
	result->kind = source.kind;
	result->sort_key = source.sort_key;
	result->value = source.value;
	result->children.reserve(source.children.size());
	for (auto &child : source.children) {
		result->children.emplace_back(child.key, CloneLWWTreeNode(*child.node));
	}
	return result;
}

// The keyed combine transfers a source group's state into the target either by deep copy (a shared
// source merged into several targets) or by move (the source is consumed). `destructive` is the
// aggregate's combine_type, constant for the whole Combine call, so every copy/move twin below is
// one body branching on it: the non-destructive combine path also holds a mutable source, so the
// copy branch simply reads it without resetting.
void TransferLWWTreeNode(LWWTreeNode &target, LWWTreeNode &source, bool destructive) {
	if (destructive) {
		target.kind = source.kind;
		target.children = std::move(source.children);
		target.sort_key = std::move(source.sort_key);
		target.value = std::move(source.value);
		source.children.clear();
		source.sort_key.clear();
		source.value = OwnedJsonoBlob();
	} else {
		target = std::move(*CloneLWWTreeNode(source));
	}
}

void EnsureLWWScalarLanes(GroupMergeLWWState &state, idx_t count, idx_t text_count);
void EnsureLWWListLanes(GroupMergeLWWState &state, idx_t count);

LWWScalarLane *FindLWWScalarLane(GroupMergeLWWState &state, idx_t lane_idx) {
	if (!state.scalar_lanes || lane_idx >= state.scalar_lane_count) {
		return nullptr;
	}
	auto &lane = state.scalar_lanes[lane_idx];
	return lane.has_value ? &lane : nullptr;
}

const LWWScalarLane *FindLWWScalarLane(const GroupMergeLWWState &state, idx_t lane_idx) {
	if (!state.scalar_lanes || lane_idx >= state.scalar_lane_count) {
		return nullptr;
	}
	auto &lane = state.scalar_lanes[lane_idx];
	return lane.has_value ? &lane : nullptr;
}

LWWScalarLane &FindOrCreateLWWScalarLane(GroupMergeLWWState &state, idx_t lane_idx) {
	if (!state.scalar_lanes || lane_idx >= state.scalar_lane_count) {
		throw InternalException("jsono_group_merge: scalar lane storage is not initialized");
	}
	return state.scalar_lanes[lane_idx];
}

bool LWWScalarLaneHasText(const vector<idx_t> &text_indices, idx_t lane_idx) {
	return text_indices[lane_idx] != DConstants::INVALID_INDEX;
}

string *FindLWWScalarLaneText(GroupMergeLWWState &state, const vector<idx_t> &text_indices, idx_t lane_idx) {
	if (!LWWScalarLaneHasText(text_indices, lane_idx) || !state.scalar_texts) {
		return nullptr;
	}
	return &state.scalar_texts[text_indices[lane_idx]];
}

string *EnsureLWWScalarLaneText(GroupMergeLWWState &state, const vector<idx_t> &text_indices, idx_t lane_idx) {
	if (!LWWScalarLaneHasText(text_indices, lane_idx)) {
		return nullptr;
	}
	if (!state.scalar_texts) {
		throw InternalException("jsono_group_merge: scalar text lane storage is not initialized");
	}
	return &state.scalar_texts[text_indices[lane_idx]];
}

string *RequireLWWScalarLaneText(GroupMergeLWWState &state, const vector<idx_t> &text_indices, idx_t lane_idx) {
	if (!LWWScalarLaneHasText(text_indices, lane_idx)) {
		return nullptr;
	}
	auto *text = FindLWWScalarLaneText(state, text_indices, lane_idx);
	if (!text) {
		throw InternalException("jsono_group_merge: missing scalar text lane");
	}
	return text;
}

nonstd::string_view LWWScalarTextView(const string *text) {
	if (!text) {
		return nonstd::string_view();
	}
	return nonstd::string_view(text->data(), text->size());
}

LWWScalarValue LWWScalarValueFromShredBits(const LogicalType &type, uint64_t value_bits, nonstd::string_view text) {
	auto kind = JsonoScalarPrimitiveFromType(type, "jsono_group_merge direct shredded update");
	LWWScalarValue result;
	switch (kind) {
	case JsonoScalarPrimitive::Varchar:
		if (text.size() > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: string value exceeds storage limits");
		}
		result.slot = MakeSlot(tag::VAL_STR_HEAP, 0);
		result.length = uint32_t(text.size());
		return result;
	case JsonoScalarPrimitive::Bigint: {
		auto value = int64_t(value_bits);
		result.slot = FitsInt60(value) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64);
		result.num = value_bits;
		return result;
	}
	case JsonoScalarPrimitive::Ubigint:
		if (value_bits <= uint64_t(std::numeric_limits<int64_t>::max())) {
			auto signed_value = int64_t(value_bits);
			result.slot = FitsInt60(signed_value) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64);
		} else {
			result.slot = MakeExtSlot(ext_subtype::UINT64);
		}
		result.num = value_bits;
		return result;
	case JsonoScalarPrimitive::Double:
		result.slot = MakeExtSlot(ext_subtype::DOUBLE);
		result.num = value_bits;
		return result;
	case JsonoScalarPrimitive::Boolean:
		result.slot = MakeSlot(value_bits != 0 ? tag::VAL_TRUE : tag::VAL_FALSE, 0);
		return result;
	}
	throw InternalException("jsono_group_merge: unhandled scalar shred primitive");
}

void SerializeLWWScalarLaneToBlob(const LogicalType &type, uint64_t value_bits, nonstd::string_view text,
                                  OwnedJsonoBlob &out) {
	SerializeLWWScalarValueToBlob(LWWScalarValueFromShredBits(type, value_bits, text), text, out);
}

// All keyed tie-breaks route through CompareBlobValueTie on standalone leaf blobs: serialize the
// scalar lane value once and compare bytes. Ties (equal sort keys) are rare, so the per-tie
// serialize is cheaper than carrying a representation-specific comparator per lane shape.
int CompareLWWScalarLaneValueTie(const LWWScalarLane &lane, nonstd::string_view lane_text, const LogicalType &type,
                                 uint64_t candidate_bits, nonstd::string_view candidate_text) {
	static thread_local OwnedJsonoBlob lane_blob;
	static thread_local OwnedJsonoBlob candidate_blob;
	SerializeLWWScalarLaneToBlob(type, lane.value_bits, lane_text, lane_blob);
	SerializeLWWScalarLaneToBlob(type, candidate_bits, candidate_text, candidate_blob);
	return CompareBlobValueTie(lane_blob, candidate_blob);
}

// Same copy/move split as the tree above (see TransferLWWTreeNode): `destructive` is the combine's
// constant combine_type, so the lane copy/move twins become one body. The source lane and its text
// are mutable in both modes (the non-destructive combine holds a mutable source); only move resets
// them. Re-storing the sort key into the target is identical for copy and move.
void TransferLWWScalarLane(GroupMergeLWWState &target_state, LWWScalarLane &target, string *target_text,
                           const GroupMergeLWWState &source_state, LWWScalarLane &source, string *source_text,
                           bool destructive) {
	target.has_value = source.has_value;
	target.sort_key_id = StoreLWWLaneSortKey(target_state, LWWLaneSortKey(source_state, source.sort_key_id));
	target.value_bits = source.value_bits;
	if (target_text) {
		if (!source_text) {
			throw InternalException("jsono_group_merge: missing source scalar text lane");
		}
		if (destructive) {
			*target_text = std::move(*source_text);
			source_text->clear();
		} else {
			*target_text = *source_text;
		}
	}
	if (destructive) {
		source.has_value = false;
		source.sort_key_id = LWW_INVALID_SORT_KEY_ID;
		source.value_bits = 0;
	}
}

void MergeLWWScalarLaneState(GroupMergeLWWState &target_state, GroupMergeLWWState &source_state, idx_t lane_idx,
                             LWWScalarLane &source, const vector<ReconShred> &shreds, const vector<idx_t> &text_indices,
                             bool destructive) {
	if (!source.has_value) {
		return;
	}
	auto *source_text = RequireLWWScalarLaneText(source_state, text_indices, lane_idx);
	auto *target = FindLWWScalarLane(target_state, lane_idx);
	if (!target) {
		target = &FindOrCreateLWWScalarLane(target_state, lane_idx);
		TransferLWWScalarLane(target_state, *target, EnsureLWWScalarLaneText(target_state, text_indices, lane_idx),
		                      source_state, source, source_text, destructive);
		return;
	}
	auto *target_text = RequireLWWScalarLaneText(target_state, text_indices, lane_idx);
	int key_cmp = CompareRawBytes(LWWLaneSortKey(target_state, target->sort_key_id),
	                              LWWLaneSortKey(source_state, source.sort_key_id));
	if (key_cmp < 0 ||
	    (key_cmp == 0 && CompareLWWScalarLaneValueTie(*target, LWWScalarTextView(target_text), shreds[lane_idx].type,
	                                                  source.value_bits, LWWScalarTextView(source_text)) < 0)) {
		TransferLWWScalarLane(target_state, *target, target_text, source_state, source, source_text, destructive);
	}
}

void MergeLWWScalarLanes(GroupMergeLWWState &target, GroupMergeLWWState &source, const vector<ReconShred> &shreds,
                         const vector<idx_t> &text_indices, bool destructive) {
	if (!source.scalar_lanes) {
		return;
	}
	EnsureLWWScalarLanes(target, source.scalar_lane_count, source.scalar_text_lane_count);
	for (idx_t i = 0; i < source.scalar_lane_count; i++) {
		MergeLWWScalarLaneState(target, source, i, source.scalar_lanes[i], shreds, text_indices, destructive);
	}
	if (destructive) {
		delete[] source.scalar_lanes;
		delete[] source.scalar_texts;
		source.scalar_lanes = nullptr;
		source.scalar_lane_count = 0;
		source.scalar_texts = nullptr;
		source.scalar_text_lane_count = 0;
	}
}

void TransferLWWListLane(GroupMergeLWWState &target_state, LWWListLane &target, const GroupMergeLWWState &source_state,
                         LWWListLane &source, bool destructive) {
	target.has_value = source.has_value;
	target.sort_key_id = StoreLWWLaneSortKey(target_state, LWWLaneSortKey(source_state, source.sort_key_id));
	if (destructive) {
		target.value = std::move(source.value);
		source.has_value = false;
		source.sort_key_id = LWW_INVALID_SORT_KEY_ID;
		source.value = LWWListValue();
	} else {
		target.value = source.value;
	}
}

void MergeLWWListLaneState(GroupMergeLWWState &target_state, LWWListLane &target,
                           const GroupMergeLWWState &source_state, LWWListLane &source, const LogicalType &type,
                           bool destructive) {
	if (!source.has_value) {
		return;
	}
	if (!target.has_value) {
		TransferLWWListLane(target_state, target, source_state, source, destructive);
		return;
	}
	int key_cmp = CompareRawBytes(LWWLaneSortKey(target_state, target.sort_key_id),
	                              LWWLaneSortKey(source_state, source.sort_key_id));
	if (key_cmp < 0 || (key_cmp == 0 && CompareLWWListValueTie(target.value, source.value, type) < 0)) {
		TransferLWWListLane(target_state, target, source_state, source, destructive);
	}
}

void MergeLWWListLanes(GroupMergeLWWState &target, GroupMergeLWWState &source, const vector<ReconShred> &shreds,
                       bool destructive) {
	if (!source.list_lanes) {
		return;
	}
	EnsureLWWListLanes(target, source.list_lane_count);
	for (idx_t i = 0; i < source.list_lane_count; i++) {
		MergeLWWListLaneState(target, target.list_lanes[i], source, source.list_lanes[i], shreds[i].type, destructive);
	}
	if (destructive) {
		delete[] source.list_lanes;
		source.list_lanes = nullptr;
		source.list_lane_count = 0;
	}
}

void MergeLWWTreeInto(LWWTreeNode &target, LWWTreeNode &source, bool destructive) {
	if (target.kind == LWWTreeKind::Object && source.kind == LWWTreeKind::Object) {
		for (auto &source_child : source.children) {
			if (destructive && !source_child.node) {
				continue;
			}
			auto key = nonstd::string_view(source_child.key.data(), source_child.key.size());
			auto child_idx = FindLWWChildIndex(target.children, key);
			bool found = child_idx < target.children.size() &&
			             CompareJsonoKeys(nonstd::string_view(target.children[child_idx].key.data(),
			                                                  target.children[child_idx].key.size()),
			                              key) == 0;
			if (found) {
				MergeLWWTreeInto(*target.children[child_idx].node, *source_child.node, destructive);
			} else if (destructive) {
				target.children.insert(target.children.begin() + child_idx, std::move(source_child));
			} else {
				target.children.insert(target.children.begin() + child_idx,
				                       LWWObjectChild(source_child.key, CloneLWWTreeNode(*source_child.node)));
			}
		}
		if (destructive) {
			source.children.clear();
		}
		return;
	}
	if (target.kind == LWWTreeKind::Object || source.kind == LWWTreeKind::Object) {
		ThrowMixedKindConflict();
	}
	int key_cmp = CompareRawBytes(target.sort_key, source.sort_key);
	if (key_cmp < 0 || (key_cmp == 0 && CompareBlobValueTie(target.value, source.value) < 0)) {
		TransferLWWTreeNode(target, source, destructive);
	}
}

void EmitLWWTreeNode(const LWWTreeNode &node, JsonoBuilder &builder, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	if (node.kind == LWWTreeKind::Object) {
		builder.EmitObjectStart(node.children.size());
		for (auto &child : node.children) {
			builder.EmitKeySlot(nonstd::string_view(child.key.data(), child.key.size()));
		}
		for (auto &child : node.children) {
			builder.EmitObjectChildStart();
			EmitLWWTreeNode(*child.node, builder, depth + 1);
		}
		builder.EmitObjectEnd();
		return;
	}
	JsonoView value_view = ViewOfBlob(node.value);
	if (!value_view.ParseHeader()) {
		throw InternalException("jsono_group_merge: malformed leaf blob");
	}
	JsonoCursor cursor;
	EmitValueVerbatim(value_view, cursor, builder, depth);
}

// Fold the incoming row into the mutable per-group tree. Object rows update only surviving
// IGNORE NULLS leaves; arrays/scalars are standalone leaves with a copied sort key.
void FoldRowLWW(GroupMergeLWWState &state, const JsonoView &V, nonstd::string_view K,
                const vector<nonstd::string_view> *root_skip_keys = nullptr) {
	static thread_local LWWTreeScratch scratch;
	JsonoCursor cursor;
	if (!state.has_input) {
		state.root = new LWWTreeNode();
		BuildIncomingLWWNode(*state.root, V, cursor, K, scratch, 0, root_skip_keys);
		state.has_input = true;
		state.mem_pending += LWWTreeNodeBytes(*state.root);
		return;
	}
	if (!state.root) {
		throw InternalException("jsono_group_merge: missing keyed aggregate root");
	}
	scratch.bytes_added = 0;
	MergeIncomingLWWNode(*state.root, V, cursor, K, scratch, 0, root_skip_keys);
	state.mem_pending += scratch.bytes_added;
}

vector<idx_t> LWWScalarTextLaneIndices(const vector<ReconShred> &shreds) {
	vector<idx_t> result;
	result.reserve(shreds.size());
	idx_t text_count = 0;
	for (auto &shred : shreds) {
		auto primitive = JsonoScalarPrimitiveFromType(shred.type, "jsono_group_merge direct lanes");
		if (primitive == JsonoScalarPrimitive::Varchar) {
			result.push_back(text_count++);
		} else {
			result.push_back(DConstants::INVALID_INDEX);
		}
	}
	return result;
}

idx_t LWWScalarTextLaneCount(const vector<idx_t> &text_indices) {
	idx_t count = 0;
	for (auto text_idx : text_indices) {
		if (text_idx != DConstants::INVALID_INDEX) {
			count++;
		}
	}
	return count;
}

void EnsureLWWScalarLanes(GroupMergeLWWState &state, idx_t count, idx_t text_count) {
	if (count == 0) {
		return;
	}
	if (state.scalar_lanes) {
		if (state.scalar_lane_count != count || state.scalar_text_lane_count != text_count) {
			throw InternalException("jsono_group_merge: scalar lane count changed inside one aggregate state");
		}
		return;
	}
	if (state.scalar_lane_count != 0 || state.scalar_text_lane_count != 0) {
		throw InternalException("jsono_group_merge: scalar lane count set without lane storage");
	}
	state.scalar_lanes = new LWWScalarLane[count];
	state.scalar_lane_count = count;
	state.mem_pending += count * sizeof(LWWScalarLane);
	if (text_count > 0) {
		state.scalar_texts = new string[text_count];
		state.scalar_text_lane_count = text_count;
		// Footprint counts each text string's capacity(), which is non-zero even when empty (SSO); account the
		// fresh strings' capacity here so per-write deltas only add growth beyond it.
		for (idx_t i = 0; i < text_count; i++) {
			state.mem_pending += state.scalar_texts[i].capacity();
		}
	}
}

void EnsureLWWListLanes(GroupMergeLWWState &state, idx_t count) {
	if (count == 0) {
		return;
	}
	if (state.list_lanes) {
		if (state.list_lane_count != count) {
			throw InternalException("jsono_group_merge: list lane count changed inside one aggregate state");
		}
		return;
	}
	if (state.list_lane_count != 0) {
		throw InternalException("jsono_group_merge: list lane count set without lane storage");
	}
	state.list_lanes = new LWWListLane[count];
	state.list_lane_count = count;
	state.mem_pending += count * sizeof(LWWListLane);
}

void StorePrimitiveLaneValueBits(const LogicalType &type, const UnifiedVectorFormat &fmt, idx_t idx, uint64_t &out,
                                 string *text_out) {
	auto kind = JsonoScalarPrimitiveFromType(type, "jsono_group_merge direct shredded update");
	out = JsonoPrimitiveVectorValueBits(kind, fmt, idx, text_out);
}

void StorePrimitiveLaneValue(const LogicalType &type, const UnifiedVectorFormat &fmt, idx_t idx, LWWScalarValue &out,
                             string *text_out) {
	uint64_t bits;
	StorePrimitiveLaneValueBits(type, fmt, idx, bits, text_out);
	out = LWWScalarValueFromShredBits(type, bits, LWWScalarTextView(text_out));
}

void StoreScalarShredLaneValue(const ReconShred &shred, const UnifiedVectorFormat &fmt, idx_t idx, uint64_t &out,
                               string &text_out) {
	text_out.clear();
	StorePrimitiveLaneValueBits(shred.type, fmt, idx, out, &text_out);
}

uint32_t ResolveLWWRowSortKey(GroupMergeLWWState &state, nonstd::string_view K, uint32_t &sort_key_id) {
	if (sort_key_id == LWW_INVALID_SORT_KEY_ID) {
		sort_key_id = StoreLWWLaneSortKey(state, K);
	}
	return sort_key_id;
}

void StoreLWWScalarLane(LWWScalarLane &lane, string *lane_text, uint64_t value_bits, nonstd::string_view text,
                        uint32_t sort_key_id) {
	lane.has_value = true;
	lane.sort_key_id = sort_key_id;
	lane.value_bits = value_bits;
	if (lane_text) {
		lane_text->assign(text.data(), text.size());
	}
}

void MergeLWWScalarLane(GroupMergeLWWState &state, idx_t lane_idx, const vector<idx_t> &text_indices,
                        const LogicalType &type, uint64_t candidate_bits, nonstd::string_view candidate_text,
                        nonstd::string_view K, uint32_t &sort_key_id) {
	auto *lane = FindLWWScalarLane(state, lane_idx);
	if (!lane) {
		lane = &FindOrCreateLWWScalarLane(state, lane_idx);
		auto *lane_text = EnsureLWWScalarLaneText(state, text_indices, lane_idx);
		idx_t before = lane_text ? lane_text->capacity() : 0;
		StoreLWWScalarLane(*lane, lane_text, candidate_bits, candidate_text,
		                   ResolveLWWRowSortKey(state, K, sort_key_id));
		idx_t after = lane_text ? lane_text->capacity() : 0;
		state.mem_pending += (after > before ? after - before : 0);
		return;
	}
	auto *lane_text = RequireLWWScalarLaneText(state, text_indices, lane_idx);
	int key_cmp = CompareRawBytes(LWWLaneSortKey(state, lane->sort_key_id), K);
	if (key_cmp > 0) {
		return;
	}
	if (key_cmp < 0 ||
	    CompareLWWScalarLaneValueTie(*lane, LWWScalarTextView(lane_text), type, candidate_bits, candidate_text) < 0) {
		idx_t before = lane_text ? lane_text->capacity() : 0;
		StoreLWWScalarLane(*lane, lane_text, candidate_bits, candidate_text,
		                   ResolveLWWRowSortKey(state, K, sort_key_id));
		idx_t after = lane_text ? lane_text->capacity() : 0;
		state.mem_pending += (after > before ? after - before : 0);
	}
}

bool LocateTopLevelLWWValue(const JsonoView &view, nonstd::string_view key, JsonoCursor &out) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return false;
	}
	auto layout = ReadObjectLayout(view, 0);
	JsonoCursor cursor;
	cursor.pos = layout.value_start;
	for (idx_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		auto current = view.KeyAt(SlotPayload(key_slot));
		if (current == key) {
			out = cursor;
			return true;
		}
		SkipValueFast(view, cursor);
	}
	return false;
}

bool StoreCompleteListShredLaneValue(const ReconShred &shred, const JsonoView &base_view,
                                     const UnifiedVectorFormat &list_fmt, const UnifiedVectorFormat &element_fmt,
                                     idx_t row, LWWTreeScratch &scratch, LWWListValue &out) {
	if (!RowIsValid(list_fmt, row)) {
		return false;
	}
	JsonoCursor skeleton_cursor;
	auto key = nonstd::string_view(shred.steps[0].key.data(), shred.steps[0].key.size());
	if (!LocateTopLevelLWWValue(base_view, key, skeleton_cursor)) {
		throw InternalException("jsono_group_merge direct list update: missing array skeleton");
	}
	auto &element_type = ListType::GetChildType(shred.type);
	auto list_entry = UnifiedVectorFormat::GetData<list_entry_t>(list_fmt)[RowIndex(list_fmt, row)];
	for (idx_t i = 0; i < list_entry.length; i++) {
		if (!RowIsValid(element_fmt, list_entry.offset + i)) {
			return false;
		}
	}
	out.elements.clear();
	out.elements.resize(list_entry.length);
	for (idx_t i = 0; i < list_entry.length; i++) {
		StorePrimitiveLaneValue(element_type, element_fmt, RowIndex(element_fmt, list_entry.offset + i),
		                        out.elements[i].value, &out.elements[i].text);
	}
	SerializeIncomingLWWLeaf(base_view, skeleton_cursor, out.skeleton, scratch, 1);
	return true;
}

void StoreLWWListLane(LWWListLane &lane, const LWWListValue &value, uint32_t sort_key_id) {
	lane.has_value = true;
	lane.sort_key_id = sort_key_id;
	lane.value = value;
}

void MergeLWWListLane(GroupMergeLWWState &state, LWWListLane &lane, const LWWListValue &candidate,
                      const LogicalType &type, nonstd::string_view K, uint32_t &sort_key_id) {
	if (!lane.has_value) {
		idx_t before = LWWListValueBytes(lane.value);
		StoreLWWListLane(lane, candidate, ResolveLWWRowSortKey(state, K, sort_key_id));
		idx_t after = LWWListValueBytes(lane.value);
		state.mem_pending += (after > before ? after - before : 0);
		return;
	}
	int key_cmp = CompareRawBytes(LWWLaneSortKey(state, lane.sort_key_id), K);
	if (key_cmp > 0) {
		return;
	}
	if (key_cmp < 0 || CompareLWWListValueTie(lane.value, candidate, type) < 0) {
		idx_t before = LWWListValueBytes(lane.value);
		StoreLWWListLane(lane, candidate, ResolveLWWRowSortKey(state, K, sort_key_id));
		idx_t after = LWWListValueBytes(lane.value);
		state.mem_pending += (after > before ? after - before : 0);
	}
}

bool PrepareDirectLWWShreddedInput(const vector<std::pair<string, LogicalType>> &bind_shreds,
                                   vector<ReconShred> &scalar_shreds, vector<ReconShred> &list_shreds,
                                   vector<idx_t> &overlay_shreds) {
	for (idx_t i = 0; i < bind_shreds.size(); i++) {
		auto &name = bind_shreds[i].first;
		vector<PathStep> steps = ShredNamePath(name, "jsono_group_merge direct shredded update");
		for (auto &step : steps) {
			if (step.kind != PathStepKind::Key) {
				return false;
			}
		}
		auto &type = bind_shreds[i].second;
		if (IsShredListType(type)) {
			if (IsShredScalarArrayType(type) && steps.size() == 1) {
				ReconShred shred {i, type, std::move(steps)};
				shred.manifest_path = name;
				list_shreds.push_back(std::move(shred));
				continue;
			}
			overlay_shreds.push_back(i);
			continue;
		}
		ReconShred shred {i, type, std::move(steps)};
		shred.manifest_path = name;
		scalar_shreds.push_back(std::move(shred));
	}
	return true;
}

// The four manifest walkers below each run the same two-pointer advance (manifest and shreds are
// both sorted by path, so one rising `shred_idx` matches each entry to its shred). A shared functor
// helper was tried but regressed the keyed group_merge hot path ~6%: capturing the loop-carried
// accumulators (sort_key_id, candidate_bits) by reference forces them to the stack and defeats
// register allocation in the inner loop. The walk is deliberately inlined per function instead.
void FoldManifestedScalarShredsLWW(GroupMergeLWWState &state, const vector<ReconShred> &shreds,
                                   const vector<idx_t> &text_indices, idx_t text_count,
                                   vector<UnifiedVectorFormat> &shred_fmt,
                                   const std::vector<ShredManifestEntry> &manifest, idx_t row, nonstd::string_view K) {
	if (shreds.empty()) {
		return;
	}
	EnsureLWWScalarLanes(state, shreds.size(), text_count);
	uint32_t sort_key_id = LWW_INVALID_SORT_KEY_ID;
	uint64_t candidate_bits;
	string candidate_text;
	idx_t shred_idx = 0;
	for (auto &entry : manifest) {
		while (shred_idx < shreds.size() && nonstd::string_view(shreds[shred_idx].manifest_path.data(),
		                                                        shreds[shred_idx].manifest_path.size()) < entry.path) {
			shred_idx++;
		}
		if (shred_idx >= shreds.size()) {
			break;
		}
		auto &shred = shreds[shred_idx];
		if (entry.path != nonstd::string_view(shred.manifest_path.data(), shred.manifest_path.size())) {
			continue;
		}
		if (!RowIsValid(shred_fmt[shred_idx], row)) {
			continue;
		}
		StoreScalarShredLaneValue(shred, shred_fmt[shred_idx], RowIndex(shred_fmt[shred_idx], row), candidate_bits,
		                          candidate_text);
		MergeLWWScalarLane(state, shred_idx, text_indices, shred.type, candidate_bits,
		                   nonstd::string_view(candidate_text.data(), candidate_text.size()), K, sort_key_id);
	}
}

bool CanSkipManifestedScalarShredsLWW(const GroupMergeLWWState &state, const vector<ReconShred> &shreds,
                                      vector<UnifiedVectorFormat> &shred_fmt,
                                      const std::vector<ShredManifestEntry> &manifest, idx_t row,
                                      nonstd::string_view K) {
	if (!state.scalar_lanes) {
		return false;
	}
	if (state.scalar_lane_count != shreds.size()) {
		throw InternalException("jsono_group_merge: scalar lane count changed inside one aggregate state");
	}
	bool saw_scalar = false;
	idx_t shred_idx = 0;
	for (auto &entry : manifest) {
		while (shred_idx < shreds.size() && nonstd::string_view(shreds[shred_idx].manifest_path.data(),
		                                                        shreds[shred_idx].manifest_path.size()) < entry.path) {
			shred_idx++;
		}
		if (shred_idx >= shreds.size()) {
			break;
		}
		auto &shred = shreds[shred_idx];
		if (entry.path != nonstd::string_view(shred.manifest_path.data(), shred.manifest_path.size())) {
			continue;
		}
		saw_scalar = true;
		if (!RowIsValid(shred_fmt[shred_idx], row)) {
			return false;
		}
		auto *lane = FindLWWScalarLane(state, shred_idx);
		if (!lane || CompareRawBytes(LWWLaneSortKey(state, lane->sort_key_id), K) <= 0) {
			return false;
		}
	}
	return saw_scalar;
}

bool DirectListShredManifestRowComplete(const vector<ReconShred> &shreds, vector<UnifiedVectorFormat> &list_fmt,
                                        vector<UnifiedVectorFormat> &element_fmt,
                                        const std::vector<ShredManifestEntry> &manifest, idx_t row) {
	idx_t shred_idx = 0;
	for (auto &entry : manifest) {
		while (shred_idx < shreds.size() && nonstd::string_view(shreds[shred_idx].manifest_path.data(),
		                                                        shreds[shred_idx].manifest_path.size()) < entry.path) {
			shred_idx++;
		}
		if (shred_idx >= shreds.size()) {
			break;
		}
		auto &shred = shreds[shred_idx];
		if (entry.path != nonstd::string_view(shred.manifest_path.data(), shred.manifest_path.size())) {
			continue;
		}
		if (!RowIsValid(list_fmt[shred_idx], row)) {
			return false;
		}
		auto list_entry =
		    UnifiedVectorFormat::GetData<list_entry_t>(list_fmt[shred_idx])[RowIndex(list_fmt[shred_idx], row)];
		for (idx_t i = 0; i < list_entry.length; i++) {
			if (!RowIsValid(element_fmt[shred_idx], list_entry.offset + i)) {
				return false;
			}
		}
	}
	return true;
}

bool CanUseDirectListShreds(Vector &input, idx_t count, const vector<ReconShred> &shreds,
                            vector<UnifiedVectorFormat> &list_fmt, vector<UnifiedVectorFormat> &element_fmt) {
	if (shreds.empty()) {
		return true;
	}
	JsonoRowReader reader;
	reader.Init(input, count);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		if (!DirectListShredManifestRowComplete(shreds, list_fmt, element_fmt, reader.RowManifest(view), row)) {
			return false;
		}
	}
	return true;
}

void FoldManifestedListShredsLWW(GroupMergeLWWState &state, const vector<ReconShred> &shreds,
                                 vector<UnifiedVectorFormat> &list_fmt, vector<UnifiedVectorFormat> &element_fmt,
                                 const std::vector<ShredManifestEntry> &manifest, const JsonoView &base_view, idx_t row,
                                 nonstd::string_view K, vector<nonstd::string_view> &skip_root_keys) {
	if (shreds.empty()) {
		return;
	}
	static thread_local LWWTreeScratch scratch;
	EnsureLWWListLanes(state, shreds.size());
	uint32_t sort_key_id = LWW_INVALID_SORT_KEY_ID;
	LWWListValue candidate;
	idx_t shred_idx = 0;
	for (auto &entry : manifest) {
		while (shred_idx < shreds.size() && nonstd::string_view(shreds[shred_idx].manifest_path.data(),
		                                                        shreds[shred_idx].manifest_path.size()) < entry.path) {
			shred_idx++;
		}
		if (shred_idx >= shreds.size()) {
			break;
		}
		auto &shred = shreds[shred_idx];
		if (entry.path != nonstd::string_view(shred.manifest_path.data(), shred.manifest_path.size())) {
			continue;
		}
		if (!StoreCompleteListShredLaneValue(shred, base_view, list_fmt[shred_idx], element_fmt[shred_idx], row,
		                                     scratch, candidate)) {
			throw InternalException("jsono_group_merge direct list update: manifested list shred is incomplete");
		}
		MergeLWWListLane(state, state.list_lanes[shred_idx], candidate, shred.type, K, sort_key_id);
		skip_root_keys.push_back(nonstd::string_view(shred.steps[0].key.data(), shred.steps[0].key.size()));
	}
}

// The grouped (per-row state) and ungrouped (single state) direct-shredded updates differ only in
// where the row's accumulator comes from, so `state_for(row)` is the only varying piece: the
// SimpleUpdate wrapper returns the one state for every row, the grouped Update indexes state_data.
template <class StateFor>
bool JsonoGroupMergeLWWUpdateDirectShreddedImpl(Vector inputs[], const GroupMergeLWWBindData &bind_data, idx_t count,
                                                UnifiedVectorFormat &sk_fmt, const string_t *sk_data,
                                                StateFor state_for) {
	if (!IsShreddedJsonoType(inputs[0].GetType()) || bind_data.shreds.empty()) {
		return false;
	}
	vector<ReconShred> scalar_shreds;
	vector<ReconShred> list_shreds;
	vector<idx_t> overlay_shreds;
	if (!PrepareDirectLWWShreddedInput(bind_data.shreds, scalar_shreds, list_shreds, overlay_shreds)) {
		return false;
	}
	auto scalar_text_indices = LWWScalarTextLaneIndices(scalar_shreds);
	auto scalar_text_count = LWWScalarTextLaneCount(scalar_text_indices);
	if (!list_shreds.empty() && !overlay_shreds.empty()) {
		return false;
	}
	Vector overlay(JsonoType(), count);
	Vector *base = &inputs[0];
	if (!overlay_shreds.empty()) {
		JsonoOverlayShredsToPlain(inputs[0], count, overlay_shreds, overlay);
		base = &overlay;
	}
	JsonoRowReader base_reader;
	base_reader.Init(*base, count);
	JsonoRowReader manifest_reader;
	if (base != &inputs[0]) {
		manifest_reader.Init(inputs[0], count);
	}
	vector<UnifiedVectorFormat> shred_fmt(scalar_shreds.size());
	for (idx_t k = 0; k < scalar_shreds.size(); k++) {
		JsonoShredVector(inputs[0], scalar_shreds[k].child).ToUnifiedFormat(count, shred_fmt[k]);
	}
	vector<UnifiedVectorFormat> list_fmt(list_shreds.size());
	vector<UnifiedVectorFormat> element_fmt(list_shreds.size());
	for (idx_t k = 0; k < list_shreds.size(); k++) {
		auto &list_vec = JsonoShredVector(inputs[0], list_shreds[k].child);
		list_vec.ToUnifiedFormat(count, list_fmt[k]);
		ListVector::GetEntry(list_vec).ToUnifiedFormat(ListVector::GetListSize(list_vec), element_fmt[k]);
	}
	if (!CanUseDirectListShreds(inputs[0], count, list_shreds, list_fmt, element_fmt)) {
		return false;
	}

	JsonoView base_view;
	JsonoView manifest_view;
	vector<nonstd::string_view> skip_root_keys;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow base_blob;
		if (base_reader.Read(row, base_blob, base_view) != JsonoRowState::Value) {
			continue;
		}
		const std::vector<ShredManifestEntry> *manifest = nullptr;
		if (base == &inputs[0]) {
			manifest = &base_reader.RowManifest(base_view);
		} else {
			JsonoBlobRow manifest_blob;
			if (manifest_reader.Read(row, manifest_blob, manifest_view) != JsonoRowState::Value) {
				continue;
			}
			manifest = &manifest_reader.RowManifest(manifest_view);
		}
		auto &k = sk_data[sk_fmt.sel->get_index(row)];
		auto K = nonstd::string_view(k.GetData(), k.GetSize());
		auto &state = state_for(row);
		skip_root_keys.clear();
		FoldManifestedListShredsLWW(state, list_shreds, list_fmt, element_fmt, *manifest, base_view, row, K,
		                            skip_root_keys);
		FoldRowLWW(state, base_view, K, &skip_root_keys);
		if (!CanSkipManifestedScalarShredsLWW(state, scalar_shreds, shred_fmt, *manifest, row, K)) {
			FoldManifestedScalarShredsLWW(state, scalar_shreds, scalar_text_indices, scalar_text_count, shred_fmt,
			                              *manifest, row, K);
		}
	}
	return true;
}

bool JsonoGroupMergeLWWSimpleUpdateDirectShredded(Vector inputs[], const GroupMergeLWWBindData &bind_data,
                                                  data_ptr_t state_ptr, idx_t count, UnifiedVectorFormat &sk_fmt,
                                                  const string_t *sk_data) {
	auto &state = *reinterpret_cast<GroupMergeLWWState *>(state_ptr);
	return JsonoGroupMergeLWWUpdateDirectShreddedImpl(inputs, bind_data, count, sk_fmt, sk_data,
	                                                  [&](idx_t) -> GroupMergeLWWState & { return state; });
}

bool JsonoGroupMergeLWWUpdateDirectShredded(Vector inputs[], const GroupMergeLWWBindData &bind_data, idx_t count,
                                            UnifiedVectorFormat &sk_fmt, const string_t *sk_data,
                                            UnifiedVectorFormat &state_fmt, GroupMergeLWWState *const *state_data) {
	return JsonoGroupMergeLWWUpdateDirectShreddedImpl(
	    inputs, bind_data, count, sk_fmt, sk_data,
	    [&](idx_t row) -> GroupMergeLWWState & { return *state_data[RowIndex(state_fmt, row)]; });
}

void EmitLWWTreeNodeStrippingPaths(const LWWTreeNode &node, const vector<const vector<PathStep> *> &strip_paths,
                                   JsonoBuilder &builder, size_t depth) {
	if (strip_paths.empty() || node.kind != LWWTreeKind::Object) {
		EmitLWWTreeNode(node, builder, depth);
		return;
	}
	idx_t child_count = 0;
	for (auto &child : node.children) {
		auto key = nonstd::string_view(child.key.data(), child.key.size());
		child_count += AnyPathTerminatesOnKey(strip_paths, depth, key) ? 0 : 1;
	}
	builder.EmitObjectStart(child_count);
	for (auto &child : node.children) {
		auto key = nonstd::string_view(child.key.data(), child.key.size());
		if (!AnyPathTerminatesOnKey(strip_paths, depth, key)) {
			builder.EmitKeySlot(key);
		}
	}
	vector<const vector<PathStep> *> deeper;
	for (auto &child : node.children) {
		auto key = nonstd::string_view(child.key.data(), child.key.size());
		if (AnyPathTerminatesOnKey(strip_paths, depth, key)) {
			continue;
		}
		builder.EmitObjectChildStart();
		CollectContinuingPaths(strip_paths, depth, key, deeper);
		EmitLWWTreeNodeStrippingPaths(*child.node, deeper, builder, depth + 1);
	}
	builder.EmitObjectEnd();
}

struct LWWRootEmitEntry {
	bool is_list_override;
	idx_t index;
	nonstd::string_view key;
};

void EmitLWWListSkeleton(const LWWListLane &lane, JsonoBuilder &builder, size_t depth) {
	JsonoView value_view = ViewOfBlob(lane.value.skeleton);
	if (!value_view.ParseHeader()) {
		throw InternalException("jsono_group_merge: malformed list skeleton blob");
	}
	JsonoCursor cursor;
	EmitValueVerbatim(value_view, cursor, builder, depth);
}

void EmitLWWTreeNodeWithListOverrides(const LWWTreeNode &node, JsonoBuilder &builder,
                                      const vector<const vector<PathStep> *> &scalar_strip_paths,
                                      const vector<idx_t> &skip_children, const vector<idx_t> &list_override_indices,
                                      const vector<ReconShred> &list_shreds, const LWWListLane *list_lanes,
                                      size_t depth) {
	if (list_override_indices.empty()) {
		EmitLWWTreeNodeStrippingPaths(node, scalar_strip_paths, builder, depth);
		return;
	}
	if (node.kind != LWWTreeKind::Object || depth != 0 || !list_lanes) {
		throw InternalException("jsono_group_merge: list skeleton override requires a root object");
	}
	vector<LWWRootEmitEntry> entries;
	entries.reserve(node.children.size() - skip_children.size() + list_override_indices.size());
	idx_t skip_pos = 0;
	for (idx_t i = 0; i < node.children.size(); i++) {
		if (skip_pos < skip_children.size() && skip_children[skip_pos] == i) {
			skip_pos++;
			continue;
		}
		auto key = nonstd::string_view(node.children[i].key.data(), node.children[i].key.size());
		if (!AnyPathTerminatesOnKey(scalar_strip_paths, depth, key)) {
			entries.push_back(LWWRootEmitEntry {false, i, key});
		}
	}
	for (auto list_idx : list_override_indices) {
		auto &key = list_shreds[list_idx].steps[0].key;
		entries.push_back(LWWRootEmitEntry {true, list_idx, nonstd::string_view(key.data(), key.size())});
	}
	std::sort(entries.begin(), entries.end(),
	          [](const LWWRootEmitEntry &a, const LWWRootEmitEntry &b) { return CompareJsonoKeys(a.key, b.key) < 0; });
	for (idx_t i = 1; i < entries.size(); i++) {
		if (CompareJsonoKeys(entries[i - 1].key, entries[i].key) == 0) {
			throw InternalException("jsono_group_merge: duplicate root key while emitting list skeleton override");
		}
	}
	builder.EmitObjectStart(entries.size());
	for (auto &entry : entries) {
		builder.EmitKeySlot(entry.key);
	}
	for (auto &entry : entries) {
		builder.EmitObjectChildStart();
		if (entry.is_list_override) {
			EmitLWWListSkeleton(list_lanes[entry.index], builder, depth + 1);
		} else {
			vector<const vector<PathStep> *> deeper;
			CollectContinuingPaths(scalar_strip_paths, depth, entry.key, deeper);
			EmitLWWTreeNodeStrippingPaths(*node.children[entry.index].node, deeper, builder, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

int CompareLWWScalarLaneToTreeLeaf(const GroupMergeLWWState &state, const LWWScalarLane &lane, const string *lane_text,
                                   const ReconShred &shred, const LWWTreeNode &node) {
	if (node.kind != LWWTreeKind::Leaf) {
		ThrowMixedKindConflict();
	}
	int key_cmp = CompareRawBytes(LWWLaneSortKey(state, lane.sort_key_id), node.sort_key);
	if (key_cmp != 0) {
		return key_cmp;
	}
	static thread_local OwnedJsonoBlob lane_blob;
	SerializeLWWScalarLaneToBlob(shred.type, lane.value_bits, LWWScalarTextView(lane_text), lane_blob);
	return CompareBlobValueTie(lane_blob, node.value);
}

int CompareLWWListLaneToTreeLeaf(const GroupMergeLWWState &state, const LWWListLane &lane, const ReconShred &shred,
                                 const LWWTreeNode &node) {
	if (node.kind != LWWTreeKind::Leaf) {
		ThrowMixedKindConflict();
	}
	int key_cmp = CompareRawBytes(LWWLaneSortKey(state, lane.sort_key_id), node.sort_key);
	if (key_cmp != 0) {
		return key_cmp;
	}
	return CompareLWWListValueTie(lane.value, shred.type, node.value);
}

void WriteLWWScalarLaneValue(const LWWScalarValue &value, nonstd::string_view text, const LogicalType &type,
                             Vector &out, idx_t rid) {
	auto primitive = JsonoScalarPrimitiveFromType(type, "jsono_group_merge direct finalize");
	FlatVector::Validity(out).SetValid(rid);
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar:
		FlatVector::GetData<string_t>(out)[rid] = StringVector::AddString(out, text.data(), text.size());
		return;
	case JsonoScalarPrimitive::Bigint: {
		int64_t v;
		std::memcpy(&v, &value.num, sizeof(v));
		FlatVector::GetData<int64_t>(out)[rid] = v;
		return;
	}
	case JsonoScalarPrimitive::Ubigint:
		FlatVector::GetData<uint64_t>(out)[rid] = value.num;
		return;
	case JsonoScalarPrimitive::Double: {
		double v;
		std::memcpy(&v, &value.num, sizeof(v));
		FlatVector::GetData<double>(out)[rid] = v;
		return;
	}
	case JsonoScalarPrimitive::Boolean:
		FlatVector::GetData<bool>(out)[rid] = SlotTag(value.slot) == tag::VAL_TRUE;
		return;
	}
}

void WriteLWWScalarLaneValue(const LWWScalarLane &lane, const string *lane_text, const ReconShred &shred, Vector &out,
                             idx_t rid) {
	auto text = LWWScalarTextView(lane_text);
	WriteLWWScalarLaneValue(LWWScalarValueFromShredBits(shred.type, lane.value_bits, text), text, shred.type, out, rid);
}

void WriteLWWListLaneValue(const LWWListLane &lane, const ReconShred &shred, Vector &out, idx_t rid) {
	FlatVector::Validity(out).SetValid(rid);
	auto start = ListVector::GetListSize(out);
	EnsureListCapacity(out, start + lane.value.elements.size());
	auto &child = ListVector::GetEntry(out);
	child.SetVectorType(VectorType::FLAT_VECTOR);
	auto &element_type = ListType::GetChildType(shred.type);
	for (idx_t i = 0; i < lane.value.elements.size(); i++) {
		auto &element = lane.value.elements[i];
		WriteLWWScalarLaneValue(element.value, nonstd::string_view(element.text.data(), element.text.size()),
		                        element_type, child, start + i);
	}
	FinishListRow(out, rid, start, lane.value.elements.size());
}

// `diverted` (out) is the per-shred completeness signal for the keyed finalize: true when a PRESENT
// value at this scalar-shred path stays in the residual (a bare lane read would miss its `->>`), so
// the finalize must mark the shred not-complete. A container (object/array subtree), a present scalar
// that does not fit the typed/VARCHAR lane — all divert. Only an explicit JSON null leaves it false:
// a bare NULL read equals `->>` there. This function never runs for an absent path (the caller gates
// on `found`), so every NULL-lane return except explicit-null is a divert.
bool WriteLWWScalarShredValue(const LWWTreeNode &node, const ReconShred &shred, Vector &out, idx_t rid,
                              bool &diverted) {
	diverted = false;
	if (node.kind != LWWTreeKind::Leaf) {
		// A merged object/array subtree at a scalar-shred path: it stays in the residual skeleton.
		diverted = true;
		return false;
	}
	JsonoView value_view = ViewOfBlob(node.value);
	if (!value_view.ParseHeader() || value_view.Slots() == 0) {
		diverted = true;
		return false;
	}
	auto slot_tag = SlotTag(value_view.SlotAt(0));
	if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
		// A container value: it cannot fit a scalar lane and stays in the residual.
		diverted = true;
		return false;
	}
	JsonoCursor cursor;
	auto scalar = DecodeScalarAt(value_view, cursor);
	auto primitive = JsonoScalarPrimitiveFromType(shred.type, "jsono_group_merge direct finalize");
	if (primitive == JsonoScalarPrimitive::Varchar) {
		if (scalar.kind != JsonoScalarKind::String) {
			// A non-string scalar (number/bool) renders to `->>` text but is not captured here, so it
			// stays in the residual; an explicit JSON null reads NULL either way.
			diverted = scalar.kind != JsonoScalarKind::Null;
			return false;
		}
		FlatVector::Validity(out).SetValid(rid);
		FlatVector::GetData<string_t>(out)[rid] = StringVector::AddString(out, scalar.text.data(), scalar.text.size());
		return true;
	}
	if (!JsonoScalarFitsPrimitive(scalar, primitive)) {
		// A present scalar that does not fit the typed lane stays in the residual; an explicit JSON
		// null reads NULL either way, so a bare read of the NULL lane is correct.
		diverted = scalar.kind != JsonoScalarKind::Null;
		return false;
	}
	FlatVector::Validity(out).SetValid(rid);
	switch (primitive) {
	case JsonoScalarPrimitive::Bigint:
		FlatVector::GetData<int64_t>(out)[rid] = scalar.int_value;
		return true;
	case JsonoScalarPrimitive::Ubigint:
		FlatVector::GetData<uint64_t>(out)[rid] =
		    scalar.kind == JsonoScalarKind::UInt64 ? scalar.uint_value : uint64_t(scalar.int_value);
		return true;
	case JsonoScalarPrimitive::Double:
		FlatVector::GetData<double>(out)[rid] = scalar.double_value;
		return true;
	case JsonoScalarPrimitive::Boolean:
		FlatVector::GetData<bool>(out)[rid] = scalar.bool_value;
		return true;
	case JsonoScalarPrimitive::Varchar:
		break;
	}
	return false;
}

enum class LWWPathLookupKind : uint8_t { Missing, Found, PrefixLeaf };

struct LWWPathLookup {
	LWWPathLookup() {
	}
	LWWPathLookup(LWWPathLookupKind kind_p, LWWTreeNode *node_p) : kind(kind_p), node(node_p) {
	}
	LWWPathLookupKind kind = LWWPathLookupKind::Missing;
	LWWTreeNode *node = nullptr;
};

LWWPathLookup FindLWWPathNode(LWWTreeNode &root, const vector<PathStep> &steps) {
	LWWTreeNode *node = &root;
	for (idx_t depth = 0; depth < steps.size(); depth++) {
		if (node->kind != LWWTreeKind::Object) {
			return {LWWPathLookupKind::PrefixLeaf, node};
		}
		auto key = nonstd::string_view(steps[depth].key.data(), steps[depth].key.size());
		auto child_idx = FindLWWChildIndex(node->children, key);
		bool found = child_idx < node->children.size() &&
		             CompareJsonoKeys(nonstd::string_view(node->children[child_idx].key.data(),
		                                                  node->children[child_idx].key.size()),
		                              key) == 0;
		if (!found) {
			return {};
		}
		node = node->children[child_idx].node.get();
	}
	return {LWWPathLookupKind::Found, node};
}

bool JsonoGroupMergeLWWFinalizeDirectShredded(Vector &result, UnifiedVectorFormat &state_fmt,
                                              GroupMergeLWWState *const *state_data,
                                              const GroupMergeLWWBindData &bind_data, idx_t count, idx_t offset) {
	vector<ReconShred> scalar_shreds;
	vector<ReconShred> list_shreds;
	vector<idx_t> ignored_list_shreds;
	if (!PrepareDirectLWWShreddedInput(bind_data.shreds, scalar_shreds, list_shreds, ignored_list_shreds)) {
		return false;
	}
	auto scalar_text_indices = LWWScalarTextLaneIndices(scalar_shreds);
	if (!list_shreds.empty() && !ignored_list_shreds.empty()) {
		return false;
	}

	JsonoBodyWriter writer;
	writer.Init(result);
	vector<Vector *> shred_out(bind_data.shreds.size());
	for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}
	JsonoSpillStamp stamp;
	stamp.Init(result);
	vector<string> shred_names;
	shred_names.reserve(bind_data.shreds.size());
	for (auto &shred : bind_data.shreds) {
		shred_names.push_back(shred.first);
	}
	auto spill_ranks = JsonoSpillRanksOfNames(shred_names);

	vector<JsonoShredManifestEntryBytes> manifest_entries(bind_data.shreds.size());
	for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(bind_data.shreds[f].first, bind_data.shreds[f].second);
	}

	JsonoBuilder builder;
	vector<const vector<PathStep> *> scalar_strip_paths;
	vector<idx_t> stripped_child_indices;
	vector<idx_t> stripped_shred_indices;
	vector<idx_t> list_override_indices;
	std::string manifest;
	for (idx_t i = 0; i < count; i++) {
		auto rid = i + offset;
		for (auto *shred : shred_out) {
			FlatVector::SetNull(*shred, rid, true);
		}
		// The clean/dirty marker + spill stamp lands after the scalar loop below has collected this
		// row's divert bits (a no-input group emits SQL NULL instead).
		stamp.ResetRow();

		auto &state = *state_data[RowIndex(state_fmt, i)];
		builder.Reset();
		scalar_strip_paths.clear();
		stripped_child_indices.clear();
		stripped_shred_indices.clear();
		list_override_indices.clear();
		if (!state.has_input) {
			// Zero non-NULL inputs -> SQL NULL (DEFAULT_NULL_HANDLING): null the body and the whole
			// shreds subtree (set marker + every shred field) so struct Verify sees a fully-NULL row.
			writer.SetRowNull(rid);
			JsonoSetRowMarkerNull(result, rid);
			continue;
		}
		if (!state.root) {
			throw InternalException("jsono_group_merge: missing keyed aggregate root");
		}
		if (state.scalar_lanes && state.scalar_lane_count != scalar_shreds.size()) {
			throw InternalException("jsono_group_merge: scalar lane count changed inside one aggregate state");
		}
		if (state.list_lanes && state.list_lane_count != list_shreds.size()) {
			throw InternalException("jsono_group_merge: list lane count changed inside one aggregate state");
		}
		if ((state.scalar_lanes || state.list_lanes) && state.root->kind != LWWTreeKind::Object) {
			ThrowMixedKindConflict();
		}
		if (state.root->kind == LWWTreeKind::Object) {
			for (idx_t s = 0; s < scalar_shreds.size(); s++) {
				auto &shred = scalar_shreds[s];
				auto lookup = FindLWWPathNode(*state.root, shred.steps);
				auto *lane = FindLWWScalarLane(state, s);
				auto *lane_text = lane ? RequireLWWScalarLaneText(state, scalar_text_indices, s) : nullptr;
				if (lane && lookup.kind == LWWPathLookupKind::PrefixLeaf) {
					ThrowMixedKindConflict();
				}
				bool found = lookup.kind == LWWPathLookupKind::Found;
				if (lane &&
				    (!found || CompareLWWScalarLaneToTreeLeaf(state, *lane, lane_text, shred, *lookup.node) >= 0)) {
					WriteLWWScalarLaneValue(*lane, lane_text, shred, *shred_out[shred.child], rid);
					if (found) {
						scalar_strip_paths.push_back(&shred.steps);
					}
					stripped_shred_indices.push_back(shred.child);
					continue;
				}
				if (!found) {
					continue;
				}
				bool diverted = false;
				if (WriteLWWScalarShredValue(*lookup.node, shred, *shred_out[shred.child], rid, diverted)) {
					scalar_strip_paths.push_back(&shred.steps);
					stripped_shred_indices.push_back(shred.child);
				}
				if (diverted) {
					stamp.SetBit(spill_ranks[shred.child]);
				}
			}
			for (idx_t s = 0; s < list_shreds.size(); s++) {
				auto &shred = list_shreds[s];
				auto key = nonstd::string_view(shred.steps[0].key.data(), shred.steps[0].key.size());
				auto child_idx = FindLWWChildIndex(state.root->children, key);
				bool found = child_idx < state.root->children.size() &&
				             CompareJsonoKeys(nonstd::string_view(state.root->children[child_idx].key.data(),
				                                                  state.root->children[child_idx].key.size()),
				                              key) == 0;
				auto *lane = state.list_lanes && state.list_lanes[s].has_value ? &state.list_lanes[s] : nullptr;
				if (!lane) {
					continue;
				}
				if (!found ||
				    CompareLWWListLaneToTreeLeaf(state, *lane, shred, *state.root->children[child_idx].node) >= 0) {
					if (found) {
						stripped_child_indices.push_back(child_idx);
					}
					WriteLWWListLaneValue(*lane, shred, *shred_out[shred.child], rid);
					stripped_shred_indices.push_back(shred.child);
					list_override_indices.push_back(s);
				}
			}
		}
		std::sort(stripped_child_indices.begin(), stripped_child_indices.end());
		EmitLWWTreeNodeWithListOverrides(*state.root, builder, scalar_strip_paths, stripped_child_indices,
		                                 list_override_indices, list_shreds, state.list_lanes, 0);
		stamp.StampRow(rid);
		const std::string *manifest_ptr = nullptr;
		if (!stripped_shred_indices.empty()) {
			std::sort(stripped_shred_indices.begin(), stripped_shred_indices.end());
			manifest.clear();
			JsonoAppendShredManifest(manifest, manifest_entries, stripped_shred_indices);
			manifest_ptr = &manifest;
		}
		writer.WriteRow(rid, builder, manifest_ptr);
	}
	return true;
}

unique_ptr<FunctionData> JsonoGroupMergeLWWBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments, OrderType direction) {
	if (arguments.size() != 2) {
		throw BinderException("%s requires a JSONO value and an order key argument", function.name);
	}
	if (arguments[0]->HasParameter() || arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto &type = arguments[0]->return_type;
	JsonoRequireExtensionOptimizerForShredded(context, type, function.name);
	auto bind_data = make_uniq<GroupMergeLWWBindData>(OrderModifiers(direction, OrderByNullType::NULLS_FIRST),
	                                                  BufferManager::GetBufferManager(context));
	if (IsShreddedJsonoType(type)) {
		// Sticky shredding, same as jsono_group_merge: keep the shredded argument native and capture
		// the shreds. Update folds shredded rows directly and Finalize writes the native lanes out;
		// the reconstruct-to-plain / reshred pair is the fallback taken only when those decline.
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(type, layout);
		for (auto &shred : layout.shreds) {
			bind_data->shreds.emplace_back(shred.first, shred.second);
		}
		function.arguments[0] = type;
		function.return_type = type;
	} else if (type.id() == LogicalTypeId::SQLNULL || IsJsonoType(type)) {
		function.arguments[0] = JsonoType();
	} else {
		JsonoRejectForeignLayout(type, function.name);
		throw BinderException("%s value argument must be JSONO", function.name);
	}
	// arguments[1] (order key) stays its own type (function.arguments[1] is ANY → no cast) so
	// CreateSortKey sees the real values in Update.
	return std::move(bind_data);
}

unique_ptr<FunctionData> JsonoGroupMergeMaxBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
	return JsonoGroupMergeLWWBind(context, function, arguments, OrderType::ASCENDING);
}

unique_ptr<FunctionData> JsonoGroupMergeMinBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
	return JsonoGroupMergeLWWBind(context, function, arguments, OrderType::DESCENDING);
}

// Encode each row's order key into a memcmp-comparable sort-key blob, with the direction baked in
// at bind (ASC → greatest wins, DESC → smallest wins; NULLS FIRST so a NULL key never wins).
void LWWMakeSortKeys(Vector &order_key, idx_t count, const GroupMergeLWWBindData &bind_data, Vector &sort_keys) {
	CreateSortKeyHelpers::CreateSortKey(order_key, count, bind_data.modifiers, sort_keys);
}

// Reconstruct a shredded input to plain for the keyed-LWW fallback below (the non-keyed direct fold
// in jsono_group_merge.cpp consumes the residual and lanes natively and never stages this). Plain
// input passes through unchanged.
Vector *GroupMergeReadInput(Vector &input, idx_t count, Vector &reconstructed) {
	if (IsShreddedJsonoType(input.GetType())) {
		JsonoReconstructToPlain(input, count, reconstructed);
		return &reconstructed;
	}
	return &input;
}

// Fallback for inputs the direct-shredded path declined (plain JSONO, or shreds it can't fold
// directly): reconstruct each row to plain and fold it. Same `state_for(row)` indirection as the
// direct path, so the ungrouped and grouped updates share this loop.
template <class StateFor>
void JsonoGroupMergeLWWReconstructFold(Vector inputs[], idx_t count, UnifiedVectorFormat &sk_fmt,
                                       const string_t *sk_data, StateFor state_for) {
	Vector reconstructed(JsonoType(), count);
	JsonoRowReader reader;
	reader.Init(*GroupMergeReadInput(inputs[0], count, reconstructed), count);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto &k = sk_data[sk_fmt.sel->get_index(row)];
		auto K = nonstd::string_view(k.GetData(), k.GetSize());
		FoldRowLWW(state_for(row), view, K);
	}
}

void JsonoGroupMergeLWWSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                    data_ptr_t state_ptr, idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeLWWBindData>();
	Vector sort_keys(LogicalType::BLOB, count);
	LWWMakeSortKeys(inputs[1], count, bind_data, sort_keys);
	UnifiedVectorFormat sk_fmt;
	sort_keys.ToUnifiedFormat(count, sk_fmt);
	auto sk_data = UnifiedVectorFormat::GetData<string_t>(sk_fmt);
	auto &state = *reinterpret_cast<GroupMergeLWWState *>(state_ptr);
	auto account = [&]() {
		AccountLWWUpdateStates(count, bind_data.buffer_manager, [&](idx_t) -> GroupMergeLWWState & { return state; });
	};
	if (JsonoGroupMergeLWWSimpleUpdateDirectShredded(inputs, bind_data, state_ptr, count, sk_fmt, sk_data)) {
		account();
		return;
	}
	JsonoGroupMergeLWWReconstructFold(inputs, count, sk_fmt, sk_data,
	                                  [&](idx_t) -> GroupMergeLWWState & { return state; });
	account();
}

void JsonoGroupMergeLWWUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                              idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeLWWBindData>();
	Vector sort_keys(LogicalType::BLOB, count);
	LWWMakeSortKeys(inputs[1], count, bind_data, sort_keys);
	UnifiedVectorFormat sk_fmt;
	sort_keys.ToUnifiedFormat(count, sk_fmt);
	auto sk_data = UnifiedVectorFormat::GetData<string_t>(sk_fmt);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeLWWState *>(state_fmt);
	// Reserve each state's additions-only pending bytes after the chunk folds; the pending == 0 fast path
	// skips the O(tree) footprint walk for states no incoming row grew (the common steady-state LWW case).
	auto account = [&]() {
		AccountLWWUpdateStates(count, bind_data.buffer_manager, [&](idx_t row) -> GroupMergeLWWState & {
			return *state_data[RowIndex(state_fmt, row)];
		});
	};
	if (JsonoGroupMergeLWWUpdateDirectShredded(inputs, bind_data, count, sk_fmt, sk_data, state_fmt, state_data)) {
		account();
		return;
	}
	JsonoGroupMergeLWWReconstructFold(inputs, count, sk_fmt, sk_data, [&](idx_t row) -> GroupMergeLWWState & {
		return *state_data[RowIndex(state_fmt, row)];
	});
	account();
}

void JsonoGroupMergeLWWCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeLWWBindData>();
	vector<ReconShred> scalar_shreds;
	vector<ReconShred> list_shreds;
	vector<idx_t> ignored_list_shreds;
	if (!PrepareDirectLWWShreddedInput(bind_data.shreds, scalar_shreds, list_shreds, ignored_list_shreds)) {
		list_shreds.clear();
	}
	auto scalar_text_indices = LWWScalarTextLaneIndices(scalar_shreds);

	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<GroupMergeLWWState *>(source_fmt);
	auto target_data = FlatVector::GetData<GroupMergeLWWState *>(target);
	auto destructive = aggr_input_data.combine_type == AggregateCombineType::ALLOW_DESTRUCTIVE;

	for (idx_t row = 0; row < count; row++) {
		auto &src = *source_data[RowIndex(source_fmt, row)];
		if (!src.has_input) {
			continue;
		}
		if (!src.root) {
			throw InternalException("jsono_group_merge: missing keyed aggregate root");
		}
		auto &tgt = *target_data[row];
		if (!tgt.has_input) {
			if (destructive) {
				tgt.root = src.root;
				tgt.scalar_lanes = src.scalar_lanes;
				tgt.scalar_lane_count = src.scalar_lane_count;
				tgt.scalar_texts = src.scalar_texts;
				tgt.scalar_text_lane_count = src.scalar_text_lane_count;
				tgt.list_lanes = src.list_lanes;
				tgt.list_lane_count = src.list_lane_count;
				tgt.lane_sort_keys = src.lane_sort_keys;
				src.root = nullptr;
				src.scalar_lanes = nullptr;
				src.scalar_lane_count = 0;
				src.scalar_texts = nullptr;
				src.scalar_text_lane_count = 0;
				src.list_lanes = nullptr;
				src.list_lane_count = 0;
				src.lane_sort_keys = nullptr;
				src.has_input = false;
			} else {
				tgt.root = CloneLWWTreeNode(*src.root).release();
				MergeLWWScalarLanes(tgt, src, scalar_shreds, scalar_text_indices, false);
				MergeLWWListLanes(tgt, src, list_shreds, false);
			}
			tgt.has_input = true;
			continue;
		}
		if (!tgt.root) {
			throw InternalException("jsono_group_merge: missing keyed aggregate root");
		}
		if (destructive) {
			MergeLWWTreeInto(*tgt.root, *src.root, true);
			MergeLWWScalarLanes(tgt, src, scalar_shreds, scalar_text_indices, true);
			MergeLWWListLanes(tgt, src, list_shreds, true);
			delete src.root;
			src.root = nullptr;
			src.has_input = false;
		} else {
			MergeLWWTreeInto(*tgt.root, *src.root, false);
			MergeLWWScalarLanes(tgt, src, scalar_shreds, scalar_text_indices, false);
			MergeLWWListLanes(tgt, src, list_shreds, false);
		}
	}
	auto &bm = bind_data.buffer_manager;
	// Targets grow; a destructive combine also shrinks sources by moving their trees/lanes out. Account
	// sources (freeing the transferred bytes) only in the destructive case, then targets (reserving them),
	// so the transfer nets to zero and a PRESERVE_INPUT window combine only reserves the copy.
	if (destructive) {
		AccountDistinctStates<GroupMergeLWWState>(
		    count, bm, [&](idx_t row) -> GroupMergeLWWState & { return *source_data[RowIndex(source_fmt, row)]; },
		    [](const GroupMergeLWWState &s) { return GroupMergeLWWFootprint(s); });
	}
	AccountDistinctStates<GroupMergeLWWState>(
	    count, bm, [&](idx_t row) -> GroupMergeLWWState & { return *target_data[row]; },
	    [](const GroupMergeLWWState &s) { return GroupMergeLWWFootprint(s); });
	// The exact-footprint Resize above is authoritative: reserved now equals the true size, so the pending
	// deltas the transfer bumped (lane sort keys) are already counted and the additions-only drift is zero.
	// Reset both counters on every accounted state -- sources only in the destructive case, mirroring the
	// walk above (a PRESERVE_INPUT window combine leaves the untouched source's own counters intact).
	for (idx_t row = 0; row < count; row++) {
		auto &tgt = *target_data[row];
		tgt.mem_pending = 0;
		tgt.mem_unmeasured = 0;
	}
	if (destructive) {
		for (idx_t row = 0; row < count; row++) {
			auto &src = *source_data[RowIndex(source_fmt, row)];
			src.mem_pending = 0;
			src.mem_unmeasured = 0;
		}
	}
}

// Serialize each group's tree (or an empty object for an empty group) into `out`.
void FinalizeLWWPlainGroups(Vector &out, UnifiedVectorFormat &state_fmt, GroupMergeLWWState *const *state_data,
                            idx_t count, idx_t base) {
	JsonoBodyWriter writer;
	writer.Init(out);
	JsonoBuilder builder;
	for (idx_t i = 0; i < count; i++) {
		auto rid = i + base;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		builder.Reset();
		if (!state.has_input) {
			// Zero non-NULL inputs -> SQL NULL (DEFAULT_NULL_HANDLING). `out` is plain here (shreds
			// empty, or the reshred-fallback's plain temp, where JsonoShredFromSpec carries the NULL
			// row through into the shredded result).
			writer.SetRowNull(rid);
			continue;
		}
		if (!state.root) {
			throw InternalException("jsono_group_merge: missing keyed aggregate root");
		}
		EmitLWWTreeNode(*state.root, builder, 0);
		writer.WriteRow(rid, builder);
	}
}

void JsonoGroupMergeLWWFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeLWWState *>(state_fmt);

	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeLWWBindData>();
	if (bind_data.shreds.empty()) {
		FinalizeLWWPlainGroups(result, state_fmt, state_data, count, offset);
		return;
	}
	if (JsonoGroupMergeLWWFinalizeDirectShredded(result, state_fmt, state_data, bind_data, count, offset)) {
		return;
	}
	Vector plain(JsonoType(), count);
	FinalizeLWWPlainGroups(plain, state_fmt, state_data, count, 0);
	Vector shredded(result.GetType(), count);
	JsonoShredFromSpec(plain, count, bind_data.shreds, shredded);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	VectorOperations::Copy(shredded, result, count, 0, offset);
}

AggregateFunction JsonoGroupMergeKeyedFunction(const char *name, bind_aggregate_function_t bind) {
	AggregateFunction fun(name, {LogicalType::ANY, LogicalType::ANY}, JsonoType(),
	                      AggregateFunction::StateSize<GroupMergeLWWState>,
	                      AggregateFunction::StateInitialize<GroupMergeLWWState, GroupMergeLWWFunction>,
	                      JsonoGroupMergeLWWUpdate, JsonoGroupMergeLWWCombine, JsonoGroupMergeLWWFinalize,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, JsonoGroupMergeLWWSimpleUpdate, bind,
	                      AggregateFunction::StateDestroy<GroupMergeLWWState, GroupMergeLWWFunction>);
	// Per-leaf conflict resolution by the key argument is commutative and associative: declaring it
	// not-order-dependent lets DuckDB stream rows into Update (no ordered-aggregate row buffer).
	fun.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
	return fun;
}

} // namespace

void RegisterJsonoGroupMergeKeyed(ExtensionLoader &loader) {
	// Order-independent keyed variants: the order key is an argument, conflicts resolve per leaf,
	// so these are commutative/associative (no ORDER BY, no O(rows) buffering — memory O(groups)).
	loader.RegisterFunction(JsonoGroupMergeKeyedFunction("jsono_group_merge_max", JsonoGroupMergeMaxBind));
	loader.RegisterFunction(JsonoGroupMergeKeyedFunction("jsono_group_merge_min", JsonoGroupMergeMinBind));
}

} // namespace duckdb
