#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// jsono_diff(prev, cur, arrays := 'atomic'|'counts'|'elements') -> structural diff of prev -> cur.
// The rendering of a changed value follows the CUR side's type (the diff describes how to reach cur).
// `arrays` is value-selected at bind (a closed set, fail-loud on a typo — not a behaviour flag):
//   atomic   (default): an array change emits the whole new cur array (reverse merge-patch; round-trips
//                       with jsono_merge_patch). Order-sensitive.
//   counts            : {added: N, removed: M} multiset cardinalities. Order-insensitive report.
//   elements          : {added: [...], removed: [...]} multiset elements. Order-insensitive report.
// Object/scalar handling is identical across modes; only array rendering differs.
enum class DiffArrayMode : uint8_t { Atomic, Counts, Elements };

struct JsonoDiffBindData : public FunctionData {
	explicit JsonoDiffBindData(DiffArrayMode mode_p) : mode(mode_p) {
	}
	DiffArrayMode mode;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoDiffBindData>(mode);
	}
	bool Equals(const FunctionData &other_p) const override {
		return mode == other_p.Cast<JsonoDiffBindData>().mode;
	}
};

// One object child captured in a single value-block walk: its key and the value's cursor (slot
// position plus the three stream cursors) and slot tag. Mirrors the merge family's MergeChild.
struct DiffChild {
	nonstd::string_view key;
	JsonoCursor cursor;
	uint64_t tag;
};

// A planned diff child of an object, decided before the object header is emitted (the builder needs
// the child count up front). Removed -> key:null; Added -> render cur[k] against an empty baseline;
// Both -> render the recursive sub-diff. The cursors index the prev/cur input views.
enum class DiffKind : uint8_t { Removed, Added, Both };
struct PlanEntry {
	nonstd::string_view key;
	DiffKind kind;
	JsonoCursor prev_cursor;
	JsonoCursor cur_cursor;
};

// Per-thread scratch. The object recursion reuses depth-indexed child/plan vectors so a nested
// object node allocates nothing (the pool slot for its depth is cleared and refilled); the array
// machinery is flat because an array diff never re-enters the diff recursion (its elements are
// whole-value multiset members emitted verbatim, never themselves diffed), so one array call is the
// only one active at a time.
struct DiffScratch {
	std::vector<std::vector<DiffChild>> cur_children_pool;
	std::vector<std::vector<DiffChild>> prev_children_pool;
	std::vector<std::vector<PlanEntry>> plan_pool;

	std::vector<JsonoCursor> arr_cur_cursors;
	std::vector<JsonoCursor> arr_prev_cursors;
	std::vector<std::string> arr_cur_keys;
	std::vector<std::string> arr_prev_keys;
	std::vector<idx_t> added_idx;
	std::vector<idx_t> removed_idx;
	std::unordered_map<std::string, int> count_prev;
	std::unordered_map<std::string, int> count_cur;
	std::unordered_map<std::string, int> seen;
	JsonoBuilder key_builder;

	// The pools are reserved to the maximum nesting depth so the on-demand resize below never
	// reallocates the outer vector — a parent frame holds a reference to its pool[depth] slot across
	// the recursive calls that grow the pool for deeper levels, and a reallocation there would dangle
	// it. With the capacity reserved, resize only constructs new tail slots, leaving existing ones put.
	template <class T>
	static std::vector<T> &PoolSlot(std::vector<std::vector<T>> &pool, size_t depth) {
		if (pool.capacity() <= depth) {
			pool.reserve(JSONO_MAX_NESTING_DEPTH + 2);
		}
		if (pool.size() <= depth) {
			pool.resize(depth + 1);
		}
		return pool[depth];
	}
	std::vector<DiffChild> &CurChildren(size_t depth) {
		return PoolSlot(cur_children_pool, depth);
	}
	std::vector<DiffChild> &PrevChildren(size_t depth) {
		return PoolSlot(prev_children_pool, depth);
	}
	std::vector<PlanEntry> &Plan(size_t depth) {
		return PoolSlot(plan_pool, depth);
	}
};

struct JsonoDiffLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	DiffScratch scratch;
	// The empty-document identity (jsono('{}')), materialized once per execute and reused for every
	// SQL-NULL cur argument (§7: NULL is the empty-document, not a propagated NULL). The backing
	// strings outlive the row loop. prev-NULL needs no view — it is modeled by prev_present = false.
	std::string empty_slots, empty_key_heap, empty_string_heap, empty_skips, empty_lengths, empty_nums;
	JsonoView empty_view;
	bool empty_built = false;

	const JsonoView &EmptyObjectView() {
		if (empty_built) {
			return empty_view;
		}
		JsonoBuilder eb;
		eb.EmitObjectStart(0);
		eb.EmitObjectEnd();
		auto slots_bytes = eb.slots.size() * sizeof(uint64_t);
		empty_slots.resize(JSONO_HEADER_SIZE + slots_bytes);
		WriteJsonoHeaderInto(reinterpret_cast<uint8_t *>(&empty_slots[0]), flags::SORTED_KEYS);
		std::memcpy(&empty_slots[JSONO_HEADER_SIZE], eb.slots.data(), slots_bytes);
		// No spans/checkpoints for an empty object: the skips blob is the bare metadata header.
		empty_skips.resize(sizeof(ContainerMetadataHeader));
		WriteEmptyMetadataInto(reinterpret_cast<uint8_t *>(&empty_skips[0]));
		empty_view =
		    JsonoView(empty_slots.data(), empty_slots.size(), empty_key_heap.data(), empty_key_heap.size(),
		              empty_string_heap.data(), empty_string_heap.size(), empty_skips.data(), empty_skips.size(),
		              empty_lengths.data(), empty_lengths.size(), empty_nums.data(), empty_nums.size());
		empty_view.ParseHeader();
		empty_built = true;
		return empty_view;
	}

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoDiffLocalState>();
	}
};

int CompareKeys(nonstd::string_view a, nonstd::string_view b) {
	auto n = std::min(a.size(), b.size());
	if (n > 0) {
		auto c = std::memcmp(a.data(), b.data(), n);
		if (c != 0) {
			return c;
		}
	}
	return a.size() < b.size() ? -1 : (a.size() > b.size() ? 1 : 0);
}

// Capture every child of the object at `obj_cursor` (key bytes, the value's cursor, and its slot
// tag) in one value-block walk. KEY slots consume no stream entries, so the value block's stream
// cursors equal the object cursor's — value cursors come from skipping each value in turn.
void CollectChildren(const JsonoView &view, const JsonoCursor &obj_cursor, std::vector<DiffChild> &out) {
	out.clear();
	auto layout = ReadObjectLayout(view, obj_cursor.pos);
	out.reserve(layout.key_count);
	JsonoCursor vc = obj_cursor;
	vc.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		if (vc.pos >= view.Slots()) {
			throw InvalidInputException("malformed JSONO: value position out of bounds");
		}
		auto value_slot = view.SlotAt(vc.pos);
		out.push_back(DiffChild {key, vc, SlotTag(value_slot)});
		SkipValueFastFromSlot(view, value_slot, vc);
	}
}

// Record the start cursor of each element of the array at `arr_cursor` (cursor at ARR_START). The
// ARR_START slot consumes no stream entries, so the first element's cursor is arr_cursor advanced by
// one slot; each subsequent element follows a SkipValueFast.
void CollectArrayElements(const JsonoView &view, const JsonoCursor &arr_cursor, std::vector<JsonoCursor> &out) {
	out.clear();
	auto end_pos = ReadArrayEndPos(view, arr_cursor.pos);
	JsonoCursor c = arr_cursor;
	c.pos++;
	while (c.pos < end_pos) {
		out.push_back(c);
		SkipValueFast(view, c);
	}
}

// Canonical scalar equality. The caller guarantees `cur` is a scalar; `prev` may be any value. Slot
// words are shape-constant in the v4 format (the value lives in the streams), so equal slot words
// mean equal tag/ext-subtype — and a container's slot word (carrying a container id/child count)
// never equals a scalar's, so this also rejects a type change to a scalar. The stream payload is then
// compared per the scalar's storage kind. Number representation is part of identity (1 INT60, 1.0
// DEC60 and 1e0 NUMBER-text carry distinct slot words), so this reports them unequal — by design.
bool ScalarEqual(const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv, const JsonoCursor &cc) {
	auto cur_slot = cv.SlotAt(cc.pos);
	auto prev_slot = pv.SlotAt(pc.pos);
	if (prev_slot != cur_slot) {
		return false;
	}
	switch (ClassifyRawScalarSlot(cur_slot)) {
	case RawScalarValueKind::Literal:
		return true;
	case RawScalarValueKind::Number:
		return pv.NumAt(pc.num_cursor) == cv.NumAt(cc.num_cursor);
	case RawScalarValueKind::LengthHeap: {
		auto prev_len = pv.LengthAt(pc.length_cursor);
		auto cur_len = cv.LengthAt(cc.length_cursor);
		if (prev_len != cur_len) {
			return false;
		}
		return pv.StringAt(pc.string_cursor, prev_len) == cv.StringAt(cc.string_cursor, cur_len);
	}
	}
	return false;
}

// A canonical comparison key for one array element: its value re-emitted verbatim into a fresh
// builder, whose five value streams (slots, key_heap, string_heap, lengths, nums) are then
// length-prefix concatenated. A fresh builder normalizes container ids and heap offsets from zero,
// so two canonically-equal values (sorted keys are an input invariant) produce byte-identical keys
// regardless of their source layout, and unequal values produce distinct keys. The skips metadata is
// excluded — it is derived navigation data, not part of the value's identity.
void AppendStreamKey(const JsonoBuilder &b, std::string &out) {
	auto put = [&](const void *p, size_t n) {
		uint64_t len = n;
		out.append(reinterpret_cast<const char *>(&len), sizeof(len));
		out.append(reinterpret_cast<const char *>(p), n);
	};
	out.clear();
	put(b.slots.data(), b.slots.size() * sizeof(uint64_t));
	put(b.key_heap.data(), b.key_heap.size());
	put(b.string_heap.data(), b.string_heap.size());
	put(b.lengths.data(), b.lengths.size() * sizeof(uint32_t));
	put(b.nums.data(), b.nums.size() * sizeof(uint64_t));
}

void ElementKey(const JsonoView &view, JsonoCursor cursor, JsonoBuilder &key_builder, std::string &out, size_t depth) {
	key_builder.Reset();
	EmitValueVerbatim(view, cursor, key_builder, depth);
	AppendStreamKey(key_builder, out);
}

// Materialize both arrays' element cursors and (when needed) their canonical keys into scratch.
// `prev` contributes its elements only if it is itself an array; otherwise the baseline is empty.
bool BuildArraySides(DiffScratch &s, bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                     const JsonoCursor &cc, bool need_keys, size_t depth) {
	CollectArrayElements(cv, cc, s.arr_cur_cursors);
	bool prev_is_array = prev_present && SlotTag(pv.SlotAt(pc.pos)) == tag::ARR_START;
	if (prev_is_array) {
		CollectArrayElements(pv, pc, s.arr_prev_cursors);
	} else {
		s.arr_prev_cursors.clear();
	}
	if (need_keys) {
		s.arr_cur_keys.resize(s.arr_cur_cursors.size());
		for (size_t i = 0; i < s.arr_cur_cursors.size(); i++) {
			ElementKey(cv, s.arr_cur_cursors[i], s.key_builder, s.arr_cur_keys[i], depth + 1);
		}
		s.arr_prev_keys.resize(s.arr_prev_cursors.size());
		for (size_t i = 0; i < s.arr_prev_cursors.size(); i++) {
			ElementKey(pv, s.arr_prev_cursors[i], s.key_builder, s.arr_prev_keys[i], depth + 1);
		}
	}
	return prev_is_array;
}

// By-value multiset difference over the canonical element keys. `added_idx` holds the cur element
// indices whose cur multiplicity exceeds their prev multiplicity (in cur iteration order);
// `removed_idx` the symmetric prev indices (in prev iteration order). The first min(prev,cur)
// occurrences of a value match; the surplus is the diff. Deterministic and order-insensitive.
void ComputeMultiset(DiffScratch &s) {
	s.count_prev.clear();
	for (auto &k : s.arr_prev_keys) {
		s.count_prev[k]++;
	}
	s.count_cur.clear();
	for (auto &k : s.arr_cur_keys) {
		s.count_cur[k]++;
	}
	s.added_idx.clear();
	s.seen.clear();
	for (idx_t i = 0; i < s.arr_cur_keys.size(); i++) {
		auto it = s.count_prev.find(s.arr_cur_keys[i]);
		int prev_count = it == s.count_prev.end() ? 0 : it->second;
		if (s.seen[s.arr_cur_keys[i]]++ >= prev_count) {
			s.added_idx.push_back(i);
		}
	}
	s.removed_idx.clear();
	s.seen.clear();
	for (idx_t i = 0; i < s.arr_prev_keys.size(); i++) {
		auto it = s.count_cur.find(s.arr_prev_keys[i]);
		int cur_count = it == s.count_cur.end() ? 0 : it->second;
		if (s.seen[s.arr_prev_keys[i]]++ >= cur_count) {
			s.removed_idx.push_back(i);
		}
	}
}

bool DiffChanged(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                 const JsonoCursor &cc, DiffArrayMode mode, size_t depth, DiffScratch &s);

// Whether the array at `cc` differs from its prev baseline under the mode's notion of change. Atomic
// is order-sensitive byte equality; counts/elements omit only when the multiset is unchanged
// (including a pure reorder). The empty baseline (prev not an array) means every cur element is new.
bool ArrayDiffers(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                  const JsonoCursor &cc, DiffArrayMode mode, size_t depth, DiffScratch &s) {
	bool prev_is_array = BuildArraySides(s, prev_present, pv, pc, cv, cc, true, depth);
	if (mode == DiffArrayMode::Atomic) {
		if (!prev_is_array || s.arr_cur_keys.size() != s.arr_prev_keys.size()) {
			return true;
		}
		for (size_t i = 0; i < s.arr_cur_keys.size(); i++) {
			if (s.arr_cur_keys[i] != s.arr_prev_keys[i]) {
				return true;
			}
		}
		return false;
	}
	ComputeMultiset(s);
	return !s.added_idx.empty() || !s.removed_idx.empty();
}

// Whether the (prev -> cur) diff is a real change (emit) or empty (omit at the parent key; jsono('{}')
// at the top). Mirrors the emit walk below; the two are kept structurally parallel. A separate
// predicate is unavoidable because the builder needs an object's changed-child count before the
// header is written.
bool DiffObjectChanged(const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv, const JsonoCursor &cc,
                       DiffArrayMode mode, size_t depth, DiffScratch &s) {
	auto &cur_children = s.CurChildren(depth);
	CollectChildren(cv, cc, cur_children);
	auto &prev_children = s.PrevChildren(depth);
	CollectChildren(pv, pc, prev_children);
	size_t ia = 0;
	size_t ib = 0;
	while (ia < prev_children.size() || ib < cur_children.size()) {
		int cmp;
		if (ib >= cur_children.size()) {
			cmp = -1;
		} else if (ia >= prev_children.size()) {
			cmp = 1;
		} else {
			cmp = CompareKeys(prev_children[ia].key, cur_children[ib].key);
		}
		if (cmp < 0) {
			return true; // prev-only key removed
		}
		if (cmp > 0) {
			// cur-only key added: a change only if the value rendered against an empty baseline is one.
			if (DiffChanged(false, pv, JsonoCursor(), cv, cur_children[ib].cursor, mode, depth + 1, s)) {
				return true;
			}
			ib++;
		} else {
			if (DiffChanged(true, pv, prev_children[ia].cursor, cv, cur_children[ib].cursor, mode, depth + 1, s)) {
				return true;
			}
			ia++;
			ib++;
		}
	}
	return false;
}

bool DiffChanged(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                 const JsonoCursor &cc, DiffArrayMode mode, size_t depth, DiffScratch &s) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto cur_tag = SlotTag(cv.SlotAt(cc.pos));
	if (cur_tag == tag::ARR_START) {
		return ArrayDiffers(prev_present, pv, pc, cv, cc, mode, depth, s);
	}
	if (cur_tag == tag::OBJ_START) {
		bool prev_is_object = prev_present && SlotTag(pv.SlotAt(pc.pos)) == tag::OBJ_START;
		if (!prev_is_object) {
			// Type changed to object (or an added object): always a change, even an empty {} — the
			// merge-patch inverse needs key:{} to reach cur, and an added key newly exists.
			return true;
		}
		return DiffObjectChanged(pv, pc, cv, cc, mode, depth, s);
	}
	return !(prev_present && ScalarEqual(pv, pc, cv, cc));
}

void EmitDiffValue(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                   const JsonoCursor &cc, JsonoBuilder &builder, DiffArrayMode mode, size_t depth, DiffScratch &s);

// Emit the array diff into `builder`. Called only when ArrayDiffers returned true. Atomic emits the
// whole cur array verbatim; counts emits {added: N, removed: M}; elements emits the multiplicity-
// expanded {added: [...], removed: [...]}. Both report objects always carry both keys (in ascending
// "added" < "removed" order, the JSONO sorted-key invariant).
void EmitDiffArray(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                   const JsonoCursor &cc, JsonoBuilder &builder, DiffArrayMode mode, size_t depth, DiffScratch &s) {
	if (mode == DiffArrayMode::Atomic) {
		JsonoCursor c = cc;
		EmitValueVerbatim(cv, c, builder, depth);
		return;
	}
	BuildArraySides(s, prev_present, pv, pc, cv, cc, true, depth);
	ComputeMultiset(s);
	builder.EmitObjectStart(2);
	builder.EmitKeySlot(nonstd::string_view("added", 5));
	builder.EmitKeySlot(nonstd::string_view("removed", 7));
	if (mode == DiffArrayMode::Counts) {
		builder.EmitObjectChildStart();
		builder.EmitInt(int64_t(s.added_idx.size()));
		builder.EmitObjectChildStart();
		builder.EmitInt(int64_t(s.removed_idx.size()));
		builder.EmitObjectEnd();
		return;
	}
	// elements: copy each surplus element verbatim from its source. The scratch cursor/index vectors
	// stay put through the emit (EmitValueVerbatim touches no scratch and an array element is never
	// itself diffed, so no nested array call overwrites them).
	builder.EmitObjectChildStart();
	builder.EmitArrayStart();
	for (auto i : s.added_idx) {
		JsonoCursor c = s.arr_cur_cursors[i];
		EmitValueVerbatim(cv, c, builder, depth + 1);
	}
	builder.EmitArrayEnd();
	builder.EmitObjectChildStart();
	builder.EmitArrayStart();
	for (auto i : s.removed_idx) {
		JsonoCursor c = s.arr_prev_cursors[i];
		EmitValueVerbatim(pv, c, builder, depth + 1);
	}
	builder.EmitArrayEnd();
	builder.EmitObjectEnd();
}

// Emit the object diff: the minimal object of only changed/added/removed keys. A removed key is
// key:null; an added key renders cur[k] against an empty baseline; a both-present key renders the
// recursive sub-diff (included only when it is a real change). The two-pointer over the two sorted
// key sequences yields the planned keys already ascending.
void EmitDiffObject(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                    const JsonoCursor &cc, JsonoBuilder &builder, DiffArrayMode mode, size_t depth, DiffScratch &s) {
	bool prev_is_object = prev_present && SlotTag(pv.SlotAt(pc.pos)) == tag::OBJ_START;
	auto &cur_children = s.CurChildren(depth);
	CollectChildren(cv, cc, cur_children);
	auto &plan = s.Plan(depth);
	plan.clear();
	if (prev_is_object) {
		auto &prev_children = s.PrevChildren(depth);
		CollectChildren(pv, pc, prev_children);
		size_t ia = 0;
		size_t ib = 0;
		while (ia < prev_children.size() || ib < cur_children.size()) {
			int cmp;
			if (ib >= cur_children.size()) {
				cmp = -1;
			} else if (ia >= prev_children.size()) {
				cmp = 1;
			} else {
				cmp = CompareKeys(prev_children[ia].key, cur_children[ib].key);
			}
			if (cmp < 0) {
				plan.push_back(PlanEntry {prev_children[ia].key, DiffKind::Removed, prev_children[ia].cursor, {}});
				ia++;
			} else if (cmp > 0) {
				if (DiffChanged(false, pv, JsonoCursor(), cv, cur_children[ib].cursor, mode, depth + 1, s)) {
					plan.push_back(PlanEntry {cur_children[ib].key, DiffKind::Added, {}, cur_children[ib].cursor});
				}
				ib++;
			} else {
				if (DiffChanged(true, pv, prev_children[ia].cursor, cv, cur_children[ib].cursor, mode, depth + 1, s)) {
					plan.push_back(PlanEntry {cur_children[ib].key, DiffKind::Both, prev_children[ia].cursor,
					                          cur_children[ib].cursor});
				}
				ia++;
				ib++;
			}
		}
	} else {
		// prev is not an object (type change / added): every cur key renders against an empty baseline.
		for (auto &child : cur_children) {
			if (DiffChanged(false, pv, JsonoCursor(), cv, child.cursor, mode, depth + 1, s)) {
				plan.push_back(PlanEntry {child.key, DiffKind::Added, {}, child.cursor});
			}
		}
	}
	builder.EmitObjectStart(plan.size());
	for (auto &entry : plan) {
		builder.EmitKeySlot(entry.key);
	}
	for (auto &entry : plan) {
		builder.EmitObjectChildStart();
		switch (entry.kind) {
		case DiffKind::Removed:
			builder.EmitNull();
			break;
		case DiffKind::Added:
			EmitDiffValue(false, pv, entry.prev_cursor, cv, entry.cur_cursor, builder, mode, depth + 1, s);
			break;
		case DiffKind::Both:
			EmitDiffValue(true, pv, entry.prev_cursor, cv, entry.cur_cursor, builder, mode, depth + 1, s);
			break;
		}
	}
	builder.EmitObjectEnd();
}

// Emit the (prev -> cur) diff value into `builder`, assuming the caller verified it is a change
// (DiffChanged). A scalar is copied verbatim from cur (its native jsono type — §9); objects and
// arrays delegate. The plan-build inside EmitDiffObject re-runs DiffChanged per child to decide
// inclusion, so the changed-decision logic lives in exactly one place.
void EmitDiffValue(bool prev_present, const JsonoView &pv, const JsonoCursor &pc, const JsonoView &cv,
                   const JsonoCursor &cc, JsonoBuilder &builder, DiffArrayMode mode, size_t depth, DiffScratch &s) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto cur_tag = SlotTag(cv.SlotAt(cc.pos));
	if (cur_tag == tag::ARR_START) {
		EmitDiffArray(prev_present, pv, pc, cv, cc, builder, mode, depth, s);
		return;
	}
	if (cur_tag == tag::OBJ_START) {
		EmitDiffObject(prev_present, pv, pc, cv, cc, builder, mode, depth, s);
		return;
	}
	JsonoCursor c = cc;
	EmitScalarVerbatim(cv, c, builder);
}

unique_ptr<FunctionData> JsonoDiffBind(ClientContext &context, ScalarFunction &bound_function,
                                       vector<unique_ptr<Expression>> &arguments) {
	auto mode = DiffArrayMode::Atomic;
	if (arguments.size() >= 3) {
		auto &arrays_arg = arguments[2];
		if (arrays_arg->GetAlias() != "arrays") {
			throw BinderException("jsono_diff: unknown argument '%s' (pass arrays := 'atomic' | 'counts' | 'elements')",
			                      arrays_arg->GetAlias());
		}
		if (arrays_arg->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arrays_arg->IsFoldable()) {
			throw BinderException("jsono_diff: arrays must be constant");
		}
		auto arrays_value = ExpressionExecutor::EvaluateScalar(context, *arrays_arg);
		if (arrays_value.IsNull()) {
			throw BinderException("jsono_diff: arrays must not be NULL");
		}
		auto name = StringValue::Get(arrays_value);
		StringUtil::Trim(name);
		name = StringUtil::Lower(name);
		if (name == "atomic") {
			mode = DiffArrayMode::Atomic;
		} else if (name == "counts") {
			mode = DiffArrayMode::Counts;
		} else if (name == "elements") {
			mode = DiffArrayMode::Elements;
		} else {
			throw BinderException("jsono_diff: arrays must be 'atomic', 'counts' or 'elements'");
		}
	}
	// Both document arguments are reconstructed to plain JSONO at bind (the binder inserts the
	// lossless reconstruct cast for a shredded input), so the executor reads whole logical values and
	// never touches shred lanes — sidestepping the shredded-array read path entirely.
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], "jsono_diff", true);
	bound_function.arguments[1] = JsonoResolveJsonoArgument(context, *arguments[1], "jsono_diff", true);
	return make_uniq<JsonoDiffBindData>(mode);
}

void JsonoDiffExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoDiffLocalState>();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<JsonoDiffBindData>();
	auto mode = bind_data.mode;
	auto count = args.size();

	// Whole-document readers (manifest-verifying). After the bind cast both inputs are plain JSONO,
	// so a manifest entry would be a row narrowed by a raw struct cast and fails loud.
	JsonoRowReader prev_reader;
	prev_reader.Init(args.data[0], count);
	JsonoRowReader cur_reader;
	cur_reader.Init(args.data[1], count);

	JsonoBodyWriter writer;
	writer.Init(result);
	auto &builder = lstate.builder;
	auto &scratch = lstate.scratch;
	const JsonoView &empty_object = lstate.EmptyObjectView();

	JsonoView prev_view;
	JsonoView cur_view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow prev_blob;
		JsonoBlobRow cur_blob;
		// A SQL-NULL (or empty) argument is the empty-document identity (§7): prev-NULL is "everything
		// new" (prev_present = false models the absent baseline); cur-NULL is "everything removed"
		// (diff against the empty object). The result is total — never SQL NULL, at minimum jsono('{}').
		bool prev_present = prev_reader.Read(row, prev_blob, prev_view) == JsonoRowState::Value;
		bool cur_present = cur_reader.Read(row, cur_blob, cur_view) == JsonoRowState::Value;
		const JsonoView &cur = cur_present ? cur_view : empty_object;
		JsonoCursor prev_cursor;
		JsonoCursor cur_cursor;
		builder.Reset();
		if (DiffChanged(prev_present, prev_view, prev_cursor, cur, cur_cursor, mode, 0, scratch)) {
			EmitDiffValue(prev_present, prev_view, prev_cursor, cur, cur_cursor, builder, mode, 0, scratch);
		} else {
			builder.EmitObjectStart(0);
			builder.EmitObjectEnd();
		}
		writer.WriteRow(row, builder);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

ScalarFunction MakeJsonoDiffFunction(const vector<LogicalType> &arguments) {
	ScalarFunction fun("jsono_diff", arguments, JsonoType(), JsonoDiffExecute, JsonoDiffBind, nullptr, nullptr,
	                   JsonoDiffLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

} // namespace

void RegisterJsonoDiff(ExtensionLoader &loader) {
	// The inputs are ANY so a shredded jsono struct (whose type varies by shred set) also binds;
	// JsonoDiffBind validates each is JSONO or shredded and reconstructs it to plain. The optional
	// third VARCHAR carries the `arrays := ...` named parameter (read by GetAlias at bind).
	ScalarFunctionSet set("jsono_diff");
	set.AddFunction(MakeJsonoDiffFunction({LogicalType::ANY, LogicalType::ANY}));
	set.AddFunction(MakeJsonoDiffFunction({LogicalType::ANY, LogicalType::ANY, LogicalType::VARCHAR}));
	loader.RegisterFunction(set);
}

} // namespace duckdb
