#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// Patch: RFC 7396 merge_patch (B wins, B-null deletes). IgnoreNulls: jsono_group_merge
// (B wins, null members dropped). Overlay: A (the first argument) is authoritative —
// B only fills keys absent from A, B-null is a no-op, A's own nulls are kept. Overlay
// powers shredded reconstruction: the residual (A) wins, shreds (B) fill stripped paths.
enum class MergeMode : uint8_t { Patch, IgnoreNulls, Overlay };

// One object child captured in a single pass: its key, the value's cursor (slot
// position plus all three stream cursors), and the value's slot tag. Capturing the
// tag here lets the merge loop classify null/object children without re-reading slots.
struct MergeChild {
	nonstd::string_view key;
	JsonoCursor cursor;
	uint64_t tag;
};

enum class MergeSrc : uint8_t { A, B, Recurse };

struct MergePlanEntry {
	nonstd::string_view key;
	MergeSrc src;
	size_t ra;
	size_t rb;
};

struct JsonoOpsLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	std::vector<MergeChild> merge_children_a;
	std::vector<MergeChild> merge_children_b;
	std::vector<MergePlanEntry> merge_plan;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoOpsLocalState>();
	}
};

// ---- Cursor-based merge (the engine for jsono_merge_patch and jsono_group_merge) ----
// Walks source views directly into the builder with no intermediate DOM (no
// std::map, no per-key/value heap allocation). Patch implements RFC 7396
// merge_patch: the A side is the target (its nulls are kept), a B-side patch null
// deletes a key, and a patch object merged onto a non-object/absent value has its
// own nulls stripped. IgnoreNulls implements jsono_group_merge (null members
// dropped). Array values (and everything nested inside an array) are verbatim.

// Push a scalar value's slot verbatim WITHOUT copying its stream entries. Used by
// the merge value phase to emit a run of scalars whose string bytes and lengths/
// nums entries are copied in bulk (contiguous slices of the source streams):
// per-child it only touches `slots` (cheap), and the streams are filled once.
// Caller must have excluded containers.
void PushScalarValueSlots(const JsonoView &view, size_t pos, JsonoBuilder &builder) {
	auto slot = view.SlotAt(pos);
	switch (SlotTag(slot)) {
	case tag::VAL_STR_HEAP:
	case tag::VAL_EXT:
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		builder.slots.push_back(slot);
		return;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

void EmitValueStrip(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder, MergeMode merge_mode,
                    size_t depth);

// True if the value at `pos` keeps any content under IGNORE NULLS stripping: a non-null
// scalar or array survives; an object survives iff some member survives (recursively).
// Used to omit keys whose object value strips to empty — jsono_group_merge drops
// `key:{}`, unlike jsono_merge_patch Patch mode which keeps it. Read-only navigation
// with a position-only cursor (the stream cursors start at zero: slot advancement does
// not depend on stream values, and zero-based prefix sums stay within bounds);
// early-exits on the first surviving leaf so non-empty objects cost ~one probe.
bool ValueSurvivesIgnoreNulls(const JsonoView &view, size_t pos) {
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::VAL_NULL) {
		return false;
	}
	if (slot_tag != tag::OBJ_START) {
		return true;
	}
	auto layout = ReadObjectLayout(view, pos);
	JsonoCursor probe;
	probe.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		if (ValueSurvivesIgnoreNulls(view, probe.pos)) {
			return true;
		}
		SkipValueFast(view, probe);
	}
	return false;
}

// A member is kept if non-null and, under IGNORE NULLS, not an object that strips to
// empty. Patch mode keeps every non-null member (RFC 7396). The survival probe
// early-exits; EmitObjectStrip caches the result so it runs at most once per member.
inline bool MemberSurvivesStrip(const JsonoView &view, size_t vp, MergeMode merge_mode) {
	if (SlotTag(view.SlotAt(vp)) == tag::VAL_NULL) {
		return false;
	}
	return merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(view, vp);
}

void EmitObjectStrip(const JsonoView &view, const JsonoCursor &obj_cursor, JsonoBuilder &builder, MergeMode merge_mode,
                     size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout = ReadObjectLayout(view, obj_cursor.pos);
	JsonoCursor value_start_cursor = obj_cursor;
	value_start_cursor.pos = layout.value_start;
	// Cache the per-member survival mask once. The IGNORE NULLS survival probe is
	// recursive, so recomputing it across the count/key/value passes is wasteful — and
	// with the mask the count and key passes need no slot walk at all (keys are addressed
	// by key_start + i). A stack mask avoids a per-object heap alloc; objects wider than
	// it (rare) fall back to recomputing the cheap predicate per pass.
	static constexpr size_t MASK_STACK = 128;
	char keep[MASK_STACK];
	bool cached = layout.key_count <= MASK_STACK;
	size_t count = 0;
	{
		JsonoCursor vc = value_start_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = MemberSurvivesStrip(view, vc.pos, merge_mode);
			if (cached) {
				keep[i] = survives ? 1 : 0;
			}
			count += survives ? 1 : 0;
			SkipValueFast(view, vc);
		}
	}
	builder.EmitObjectStart(count);
	if (cached) {
		for (size_t i = 0; i < layout.key_count; i++) {
			if (!keep[i]) {
				continue;
			}
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
	} else {
		JsonoCursor vc = value_start_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			if (MemberSurvivesStrip(view, vc.pos, merge_mode)) {
				auto key_slot = view.SlotAt(layout.key_start + i);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
			}
			SkipValueFast(view, vc);
		}
	}
	{
		JsonoCursor vc = value_start_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = cached ? (keep[i] != 0) : MemberSurvivesStrip(view, vc.pos, merge_mode);
			if (!survives) {
				SkipValueFast(view, vc);
				continue;
			}
			builder.EmitObjectChildStart();
			EmitValueStrip(view, vc, builder, merge_mode, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

void EmitValueStrip(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder, MergeMode merge_mode,
                    size_t depth) {
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		EmitObjectStrip(view, cursor, builder, merge_mode, depth);
		SkipValueFast(view, cursor);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		EmitValueVerbatim(view, cursor, builder, depth);
		return;
	}
	EmitScalarVerbatim(view, cursor, builder);
}

int CompareJsonoKeys(nonstd::string_view a, nonstd::string_view b) {
	auto n = std::min(a.size(), b.size());
	if (n > 0) {
		auto c = std::memcmp(a.data(), b.data(), n);
		if (c != 0) {
			return c;
		}
	}
	if (a.size() < b.size()) {
		return -1;
	}
	if (a.size() > b.size()) {
		return 1;
	}
	return 0;
}

// Captures every child and returns the cursor just past the last child, so the merge
// value phase can read child r's per-stream span as the gap between consecutive
// children's cursors (out[r+1].cursor, or this end for the last child) without
// re-decoding the value.
JsonoCursor CollectObjectChildren(const JsonoView &view, const ObjectLayout &layout, const JsonoCursor &obj_cursor,
                                  std::vector<MergeChild> &out) {
	out.clear();
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
		out.push_back(MergeChild {key, vc, SlotTag(value_slot)});
		SkipValueFastFromSlot(view, value_slot, vc);
	}
	return vc;
}

void MergeTwoObjects(const JsonoView &va, const JsonoCursor &cursor_a, const JsonoView &vb, const JsonoCursor &cursor_b,
                     JsonoBuilder &builder, MergeMode merge_mode, size_t depth);

// Merge two object views into the builder (B patches A), sorted-key linear merge.
void MergeTwoObjectsWithScratch(const JsonoView &va, const JsonoCursor &cursor_a, const JsonoView &vb,
                                const JsonoCursor &cursor_b, JsonoBuilder &builder, MergeMode merge_mode, size_t depth,
                                std::vector<MergeChild> &children_a, std::vector<MergeChild> &children_b,
                                std::vector<MergePlanEntry> &plan) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout_a = ReadObjectLayout(va, cursor_a.pos);
	auto layout_b = ReadObjectLayout(vb, cursor_b.pos);
	JsonoCursor children_a_end = CollectObjectChildren(va, layout_a, cursor_a, children_a);
	JsonoCursor children_b_end = CollectObjectChildren(vb, layout_b, cursor_b, children_b);

	plan.clear();
	plan.reserve(children_a.size() + children_b.size());

	size_t ia = 0;
	size_t ib = 0;
	auto na = children_a.size();
	auto nb = children_b.size();
	while (ia < na || ib < nb) {
		int cmp;
		if (ib >= nb) {
			cmp = -1;
		} else if (ia >= na) {
			cmp = 1;
		} else {
			cmp = CompareJsonoKeys(children_a[ia].key, children_b[ib].key);
		}
		if (cmp < 0) {
			// A-only key. Patch and Overlay copy A verbatim — its explicit nulls are
			// values, not deletions. Under IGNORE NULLS a null member is dropped.
			if (merge_mode != MergeMode::IgnoreNulls || children_a[ia].tag != tag::VAL_NULL) {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, 0});
			}
			ia++;
		} else if (cmp > 0) {
			// B-only key. Patch keeps every non-null member (RFC 7396 may leave `{}`);
			// Overlay and IGNORE NULLS drop a member whose object value strips to empty.
			if (children_b[ib].tag != tag::VAL_NULL &&
			    (merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(vb, children_b[ib].cursor.pos))) {
				plan.push_back(MergePlanEntry {children_b[ib].key, MergeSrc::B, 0, ib});
			}
			ib++;
		} else if (merge_mode == MergeMode::Overlay) {
			// Both sides have the key. A (the residual) is authoritative. If both are
			// objects, recurse so B can still fill nested keys A dropped (A wins every
			// leaf conflict); otherwise keep A verbatim, including an explicit null.
			if (children_a[ia].tag == tag::OBJ_START && children_b[ib].tag == tag::OBJ_START) {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::Recurse, ia, ib});
			} else {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, ib});
			}
			ia++;
			ib++;
		} else {
			if (children_b[ib].tag != tag::VAL_NULL) {
				if (children_a[ia].tag == tag::OBJ_START && children_b[ib].tag == tag::OBJ_START) {
					// Recurse keeps A's surviving members, so the merged object is non-empty.
					plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::Recurse, ia, ib});
				} else if (merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(vb, children_b[ib].cursor.pos)) {
					plan.push_back(MergePlanEntry {children_b[ib].key, MergeSrc::B, ia, ib});
				}
			} else if (merge_mode == MergeMode::IgnoreNulls && children_a[ia].tag != tag::VAL_NULL) {
				// IGNORE NULLS: B's null must not overwrite A's accumulated value.
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, ib});
			}
			ia++;
			ib++;
		}
	}

	builder.EmitObjectStart(plan.size());
	// Key block. Output keys are a subset of one source's sorted keys, so a run of
	// plan keys that stay adjacent in that source's key_heap (no stripped key fell
	// between them) is one contiguous byte slice — copy it once and stamp each KEY
	// slot's relocated offset, instead of one key_heap insert per key. A run stays
	// in one key_heap: B-sourced keys live in vb's, the rest in va's, so the offset
	// arithmetic below stays within a single buffer.
	size_t key_plan_size = plan.size();
	size_t kk = 0;
	while (kk < key_plan_size) {
		size_t key_run_end = kk + 1;
		while (key_run_end < key_plan_size &&
		       (plan[key_run_end - 1].src == MergeSrc::B) == (plan[key_run_end].src == MergeSrc::B) &&
		       plan[key_run_end - 1].key.data() + plan[key_run_end - 1].key.size() == plan[key_run_end].key.data()) {
			key_run_end++;
		}
		const char *first_key = plan[kk].key.data();
		size_t key_base = builder.key_heap.size();
		size_t key_total = size_t(plan[key_run_end - 1].key.data() + plan[key_run_end - 1].key.size() - first_key);
		builder.key_heap.insert(builder.key_heap.end(), first_key, first_key + key_total);
		for (size_t i = kk; i < key_run_end; i++) {
			builder.PushKeySlot(key_base + size_t(plan[i].key.data() - first_key), plan[i].key.size());
		}
		kk = key_run_end;
	}
	// Value block. Consecutive scalar children from one source have their string
	// bytes and lengths/nums entries in contiguous slices of that source's streams
	// (null-stripping removes keys/slots but never stream entries of kept members),
	// so a run is copied with one bulk insert per stream instead of one insert per
	// value. A run stops at a source change, a container child (emitted per-leaf), a
	// stream-cursor gap (a stripped member carried stream entries), or a
	// checkpoint-stride boundary. The stride stop keeps the builder's per-child
	// EmitObjectChildStart snapshots correct: within a run only its first index can
	// be a stride multiple, and that snapshot reads the sizes the already-emitted
	// prior run left behind.
	bool has_checkpoints;
	size_t stride;
	{
		// Capture by value: recursive child emission reallocates open_containers.
		auto &open = builder.open_containers.back();
		has_checkpoints = open.checkpoint_offset != NO_OBJECT_CHECKPOINTS;
		stride = has_checkpoints ? size_t(open.checkpoint_stride) : 0;
	}
	size_t plan_size = plan.size();
	size_t k = 0;
	while (k < plan_size) {
		auto src = plan[k].src;
		if (src == MergeSrc::Recurse) {
			builder.EmitObjectChildStart();
			MergeTwoObjects(va, children_a[plan[k].ra].cursor, vb, children_b[plan[k].rb].cursor, builder, merge_mode,
			                depth + 1);
			k++;
			continue;
		}
		const JsonoView &vsrc = src == MergeSrc::A ? va : vb;
		auto &csrc = src == MergeSrc::A ? children_a : children_b;
		const JsonoCursor &csrc_end = src == MergeSrc::A ? children_a_end : children_b_end;
		size_t rank = src == MergeSrc::A ? plan[k].ra : plan[k].rb;
		auto value_tag = csrc[rank].tag;
		if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
			builder.EmitObjectChildStart();
			JsonoCursor c = csrc[rank].cursor;
			// A-side under Patch/Overlay is the verbatim target (A authoritative, keep
			// its nested nulls); the B-side under any mode strips null members so an
			// Overlay shred refill never re-introduces a null the residual dropped.
			if (src == MergeSrc::A && (merge_mode == MergeMode::Overlay || merge_mode == MergeMode::Patch)) {
				EmitValueVerbatim(vsrc, c, builder, depth + 1);
			} else {
				EmitValueStrip(vsrc, c, builder, merge_mode, depth + 1);
			}
			k++;
			continue;
		}
		// Extend the scalar run: same source, scalar value, before the next stride
		// boundary, and stream-contiguous in the source. A child's per-stream span is
		// the gap between its cursor and the next child's (csrc_end for the last), so
		// member `r` is contiguous with the run iff its cursor equals the position
		// just past the previous member on EVERY stream. A base member stripped for
		// being null leaves no stream entries, so the slices span it; a member
		// stripped by a patch null can carry entries (string/number/object), which
		// break the run there.
		size_t boundary = has_checkpoints ? (k / stride + 1) * stride : plan_size;
		JsonoCursor first_cursor = csrc[rank].cursor;
		size_t last_rank = rank;
		size_t run_end = k + 1;
		while (run_end < boundary && run_end < plan_size && plan[run_end].src == src) {
			size_t run_rank = src == MergeSrc::A ? plan[run_end].ra : plan[run_end].rb;
			auto run_tag = csrc[run_rank].tag;
			if (run_tag == tag::OBJ_START || run_tag == tag::ARR_START) {
				break;
			}
			const JsonoCursor &after_last = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].cursor : csrc_end;
			auto &run_cursor = csrc[run_rank].cursor;
			if (run_cursor.string_cursor != after_last.string_cursor ||
			    run_cursor.length_cursor != after_last.length_cursor ||
			    run_cursor.num_cursor != after_last.num_cursor) {
				break;
			}
			last_rank = run_rank;
			run_end++;
		}
		// Advance the per-child cursor for the whole run first (only index k can hit
		// a stride boundary, and its snapshot reads the prior run's emitted sizes).
		for (size_t i = k; i < run_end; i++) {
			builder.EmitObjectChildStart();
		}
		for (size_t i = k; i < run_end; i++) {
			size_t run_rank = src == MergeSrc::A ? plan[i].ra : plan[i].rb;
			PushScalarValueSlots(vsrc, csrc[run_rank].cursor.pos, builder);
		}
		const JsonoCursor &run_end_cursor = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].cursor : csrc_end;
		if (run_end_cursor.string_cursor > first_cursor.string_cursor) {
			auto slice =
			    vsrc.StringAt(first_cursor.string_cursor, run_end_cursor.string_cursor - first_cursor.string_cursor);
			builder.string_heap.insert(builder.string_heap.end(), slice.begin(), slice.end());
		}
		if (run_end_cursor.length_cursor > first_cursor.length_cursor) {
			auto entry_count = run_end_cursor.length_cursor - first_cursor.length_cursor;
			auto bytes = vsrc.LengthsBytes(first_cursor.length_cursor, entry_count);
			auto old_size = builder.lengths.size();
			builder.lengths.resize(old_size + entry_count);
			std::memcpy(builder.lengths.data() + old_size, bytes, entry_count * sizeof(uint32_t));
		}
		if (run_end_cursor.num_cursor > first_cursor.num_cursor) {
			auto entry_count = run_end_cursor.num_cursor - first_cursor.num_cursor;
			auto bytes = vsrc.NumsBytes(first_cursor.num_cursor, entry_count);
			auto old_size = builder.nums.size();
			builder.nums.resize(old_size + entry_count);
			std::memcpy(builder.nums.data() + old_size, bytes, entry_count * sizeof(uint64_t));
		}
		k = run_end;
	}
	builder.EmitObjectEnd();
}

void MergeTwoObjects(const JsonoView &va, const JsonoCursor &cursor_a, const JsonoView &vb, const JsonoCursor &cursor_b,
                     JsonoBuilder &builder, MergeMode merge_mode, size_t depth) {
	std::vector<MergeChild> children_a;
	std::vector<MergeChild> children_b;
	std::vector<MergePlanEntry> plan;
	MergeTwoObjectsWithScratch(va, cursor_a, vb, cursor_b, builder, merge_mode, depth, children_a, children_b, plan);
}

// A self-owned JSONO blob: the in-flight accumulator for the fold-based scalar
// jsono_merge_patch and the jsono_group_merge aggregate. Each fold runs the
// cursor-based MergeTwoObjects over contiguous bytes (no per-key std::string, no
// per-node heap alloc, keys compared in cache order) and re-serializes the merged
// builder back into the blob so the next fold can view it.
struct OwnedJsonoBlob {
	std::string slots;
	std::string key_heap;
	std::string string_heap;
	std::string skips;
	std::string lengths;
	std::string nums;
};

void SerializeBuilderToBlob(const JsonoBuilder &builder, OwnedJsonoBlob &out) {
	auto slots_bytes = builder.slots.size() * sizeof(uint64_t);
	out.slots.resize(JSONO_HEADER_SIZE + slots_bytes);
	WriteJsonoHeaderInto(reinterpret_cast<uint8_t *>(&out.slots[0]), flags::SORTED_KEYS);
	if (slots_bytes > 0) {
		std::memcpy(&out.slots[JSONO_HEADER_SIZE], builder.slots.data(), slots_bytes);
	}
	out.key_heap.assign(builder.key_heap.data(), builder.key_heap.size());
	out.string_heap.assign(builder.string_heap.data(), builder.string_heap.size());
	out.lengths.assign(reinterpret_cast<const char *>(builder.lengths.data()),
	                   builder.lengths.size() * sizeof(uint32_t));
	out.nums.assign(reinterpret_cast<const char *>(builder.nums.data()), builder.nums.size() * sizeof(uint64_t));

	auto span_ids_bytes = builder.span_ids.size() * sizeof(uint32_t);
	auto spans_bytes = builder.skips.size() * sizeof(ContainerSpan);
	auto index_bytes = builder.object_checkpoint_index.size() * sizeof(ObjectCheckpointIndex);
	auto checkpoints_bytes = builder.object_checkpoints.size() * sizeof(ObjectCursorCheckpoint);
	out.skips.resize(sizeof(ContainerMetadataHeader) + span_ids_bytes + spans_bytes + index_bytes + checkpoints_bytes);
	ContainerMetadataHeader meta;
	meta.span_count = uint32_t(builder.skips.size());
	meta.checkpoint_index_count = uint32_t(builder.object_checkpoint_index.size());
	meta.checkpoint_count = uint32_t(builder.object_checkpoints.size());
	size_t off = 0;
	std::memcpy(&out.skips[off], &meta, sizeof(meta));
	off += sizeof(meta);
	if (span_ids_bytes > 0) {
		std::memcpy(&out.skips[off], builder.span_ids.data(), span_ids_bytes);
		off += span_ids_bytes;
	}
	if (spans_bytes > 0) {
		std::memcpy(&out.skips[off], builder.skips.data(), spans_bytes);
		off += spans_bytes;
	}
	if (index_bytes > 0) {
		std::memcpy(&out.skips[off], builder.object_checkpoint_index.data(), index_bytes);
		off += index_bytes;
	}
	if (checkpoints_bytes > 0) {
		std::memcpy(&out.skips[off], builder.object_checkpoints.data(), checkpoints_bytes);
	}
}

JsonoView ViewOfBlob(const OwnedJsonoBlob &b) {
	return JsonoView(b.slots.data(), b.slots.size(), b.key_heap.data(), b.key_heap.size(), b.string_heap.data(),
	                 b.string_heap.size(), b.skips.data(), b.skips.size(), b.lengths.data(), b.lengths.size(),
	                 b.nums.data(), b.nums.size());
}

// One RFC 7396 fold step. Returns true when the result is SQL NULL (a SQL NULL
// patch argument replaces the result with SQL NULL). Otherwise the merged value is
// written into `builder`: an object patch deep-merges onto an object accumulator
// (the accumulator is the target — its nulls are kept; the patch's nulls delete
// keys), while onto a non-object or SQL NULL accumulator it is merged onto an empty
// object (its own nulls stripped). A non-object patch replaces the accumulator.
bool MergeFoldStep(MergeMode mode, bool acc_is_sqlnull, const JsonoView &acc_view, bool patch_present,
                   const JsonoView &patch_view, JsonoBuilder &builder, std::vector<MergeChild> &children_a,
                   std::vector<MergeChild> &children_b, std::vector<MergePlanEntry> &plan) {
	if (!patch_present) {
		// Patch: a SQL NULL patch nullifies the result. Overlay: A is authoritative, so
		// a missing B leaves the accumulator untouched.
		return mode == MergeMode::Overlay ? acc_is_sqlnull : true;
	}
	builder.Reset();
	JsonoCursor cursor;
	bool acc_is_object = !acc_is_sqlnull && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START;
	bool patch_is_object = SlotTag(patch_view.SlotAt(0)) == tag::OBJ_START;
	if (acc_is_object && patch_is_object) {
		MergeTwoObjectsWithScratch(acc_view, JsonoCursor(), patch_view, JsonoCursor(), builder, mode, 0, children_a,
		                           children_b, plan);
		return false;
	}
	if (mode == MergeMode::Overlay) {
		// A authoritative: keep the accumulator if present, otherwise fall to the patch.
		const JsonoView &src = acc_is_sqlnull ? patch_view : acc_view;
		JsonoCursor src_cursor;
		EmitValueVerbatim(src, src_cursor, builder, 0);
		return false;
	}
	if (!patch_is_object) {
		EmitValueVerbatim(patch_view, cursor, builder, 0);
	} else {
		EmitValueStrip(patch_view, cursor, builder, MergeMode::Patch, 0);
	}
	return false;
}

// A shred carried through a shred-aware merge: its name, the result struct child index it
// lands in, and its type. The merge folds the six-BLOB residuals (existing logic) and
// copies each union shred from the last input that declares it. Shred keys are assumed
// disjoint from residual keys (true when shreds are computed fields lifted into columns).
struct MergeShred {
	string name;
	idx_t result_child_index;
	LogicalType type;
};

struct JsonoMergeBindData : public FunctionData {
	vector<MergeShred> shreds;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoMergeBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoMergeBindData>();
		if (shreds.size() != other.shreds.size()) {
			return false;
		}
		for (idx_t i = 0; i < shreds.size(); i++) {
			if (shreds[i].name != other.shreds[i].name || shreds[i].type != other.shreds[i].type) {
				return false;
			}
		}
		return true;
	}
};

unique_ptr<FunctionData> JsonoMergePatchBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	if (arguments.empty()) {
		throw BinderException("jsono_merge_patch() requires at least one argument");
	}
	// Union the shreds of any shredded inputs (later inputs win a name conflict, matching
	// the patch-wins fold). A shredded input keeps its type so the executor can read its
	// shred columns; a plain input is bound as JSONO and contributes only its residual.
	auto bind_data = make_uniq<JsonoMergeBindData>();
	auto &shreds = bind_data->shreds;
	for (auto &argument : arguments) {
		if (argument->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		auto &type = argument->return_type;
		JsonoRequireExtensionOptimizerForShredded(context, type, "jsono_merge_patch");
		if (type.id() == LogicalTypeId::SQLNULL) {
			bound_function.arguments.push_back(JsonoType());
			continue;
		}
		if (IsShreddedJsonoType(type)) {
			JsonoLayoutType layout;
			TryParseJsonoLayoutType(type, layout);
			for (auto &layout_shred : layout.shreds) {
				// Union by the shred path (the inputs may carry different shred sets).
				auto &shred_name = layout_shred.first;
				bool found = false;
				for (auto &merge_shred : shreds) {
					if (merge_shred.name == shred_name) {
						merge_shred.type = layout_shred.second;
						found = true;
						break;
					}
				}
				if (!found) {
					shreds.push_back(MergeShred {shred_name, 0, layout_shred.second});
				}
			}
			bound_function.arguments.push_back(type);
			continue;
		}
		if (IsJsonoType(type)) {
			bound_function.arguments.push_back(JsonoType());
			continue;
		}
		throw BinderException("jsono_merge_patch() arguments must be JSONO");
	}
	if (shreds.empty()) {
		bound_function.return_type = JsonoType();
		return std::move(bind_data);
	}
	// Canonical shred order (sorted by name) so the merged type is a pure function of the shred set,
	// not of argument order: the executor writes shreds by result_child_index, so the index order and
	// the type's shred order must agree.
	std::sort(shreds.begin(), shreds.end(), [](const MergeShred &a, const MergeShred &b) { return a.name < b.name; });
	child_list_t<LogicalType> shred_types;
	idx_t next = 0; // shred-relative index inside the result's `.shreds` struct
	for (auto &shred : shreds) {
		shred.result_child_index = next++;
		shred_types.emplace_back(shred.name, shred.type);
	}
	bound_function.return_type = JsonoShreddedStructType(shred_types);
	return std::move(bind_data);
}

struct ReconShred {
	idx_t child;
	LogicalType type;
	vector<PathStep> steps; // object-key path; size 1 is a top-level key
	string manifest_path;
};

void EmitReconShredScalar(JsonoBuilder &builder, const LogicalType &type, const UnifiedVectorFormat &fmt, idx_t idx) {
	ShredPrimitive kind;
	if (!TypeToShredPrimitive(type, kind)) {
		throw InternalException("jsono reconstruct: non-scalar shred type '%s'", type.ToString());
	}
	// Exhaustive over the closed scalar shred set, so a new scalar shred type fails to compile here.
	switch (kind) {
	case ShredPrimitive::Varchar: {
		auto value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
		return;
	}
	case ShredPrimitive::Bigint:
		builder.EmitInt(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx]);
		return;
	case ShredPrimitive::Ubigint:
		builder.EmitUInt(UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx]);
		return;
	case ShredPrimitive::Double:
		builder.EmitDouble(UnifiedVectorFormat::GetData<double>(fmt)[idx]);
		return;
	case ShredPrimitive::Boolean:
		builder.EmitBool(UnifiedVectorFormat::GetData<bool>(fmt)[idx]);
		return;
	}
}

// True if a pure object-key path ends at / continues past `key` when matched at `depth`. The
// size check precedes the [depth] read, so a path shorter than depth+1 never indexes OOB. The
// path ref is std::vector so duckdb::vector<PathStep> binds too (derived-to-base). Shared by the
// array read overlay, the residual skeleton emit, and the LWW tree strip below.
static bool PathTerminatesOnKey(const std::vector<PathStep> &path, size_t depth, nonstd::string_view key) {
	return path.size() == depth + 1 && nonstd::string_view(path[depth].key.data(), path[depth].key.size()) == key;
}

static bool PathContinuesPastKey(const std::vector<PathStep> &path, size_t depth, nonstd::string_view key) {
	return path.size() > depth + 1 && nonstd::string_view(path[depth].key.data(), path[depth].key.size()) == key;
}

// Multi-path variants: "does any active path terminate on `key`" and "collect the active paths
// that continue past `key`". Templated over the path-pointer container so both the std::vector
// skeleton side and the duckdb::vector LWW side share them; both delegate to the predicates above.
template <class Paths>
static bool AnyPathTerminatesOnKey(const Paths &paths, size_t depth, nonstd::string_view key) {
	for (auto *path : paths) {
		if (PathTerminatesOnKey(*path, depth, key)) {
			return true;
		}
	}
	return false;
}

template <class Paths>
static void CollectContinuingPaths(const Paths &paths, size_t depth, nonstd::string_view key, Paths &out) {
	out.clear();
	for (auto *path : paths) {
		if (PathContinuesPastKey(*path, depth, key)) {
			out.push_back(path);
		}
	}
}

// ---- Array shred overlay (read side) ----
// A LIST<STRUCT> shred column holds, per row, one struct per element of the skeleton array, in
// lockstep. Reconstruct rebuilds the array: each object element merges its present shred subfields
// (residual-authoritative) back over its skeleton tail; a non-object/null element or a NULL shred
// struct is emitted verbatim. The skeleton array length is authoritative — a non-NULL shred list of
// a different length is a corrupt row (e.g. a struct cast that truncated the LIST) and fails loud.
struct ArrayReconShred {
	idx_t child; // shred index over the shred set
	ShredKind kind = ShredKind::Array;
	vector<PathStep> path;        // pure object-key chain to the array
	UnifiedVectorFormat list_fmt; // list_entry_t + per-row validity (both kinds)
	const list_entry_t *list_entries = nullptr;
	// kind == Array: the element struct's subfields lifted into a LIST<STRUCT> column.
	vector<std::pair<string, LogicalType>> subfields; // element subfield (name, scalar type), struct order
	// Subfield indices in sorted-key order: the overlay patch object must emit keys ascending
	// (the JSONO object invariant and the sorted-key two-pointer MergeTwoObjects both require it).
	vector<idx_t> sorted_subfields;
	UnifiedVectorFormat struct_fmt;      // per-element struct validity
	vector<UnifiedVectorFormat> sub_fmt; // per subfield: value + validity
	// kind == ScalarArray: each whole element lifted into a LIST<element_type> column.
	LogicalType element_type;
	UnifiedVectorFormat element_fmt; // the list child scalar vector: value + per-element validity
};

struct ArrayOverlayScratch {
	JsonoBuilder patch_builder;
	OwnedJsonoBlob patch_storage;
};

// Rebuild one array (cursor at ARR_START), overlaying the row's shred subfields onto each skeleton
// element, advancing `cursor` past ARR_END.
void OverlayArray(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder, const ArrayReconShred &shred,
                  idx_t row, ArrayOverlayScratch &scratch, size_t depth) {
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	builder.EmitArrayStart();
	cursor.pos++; // first element; ARR_START consumes no stream entries
	auto list_idx = shred.list_fmt.sel->get_index(row);
	if (!shred.list_fmt.validity.RowIsValid(list_idx)) {
		// NULL shred list: the array stayed whole in the residual (nothing lifted) — emit verbatim.
		while (cursor.pos < end_pos) {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
		builder.EmitArrayEnd();
		cursor.pos = end_pos + 1;
		return;
	}
	auto entry = shred.list_entries[list_idx];
	idx_t i = 0;
	while (cursor.pos < end_pos) {
		if (i >= entry.length) {
			throw InvalidInputException("malformed JSONO: array shred list is shorter than the residual array "
			                            "(a struct cast truncated it)");
		}
		idx_t child = entry.offset + i;
		if (shred.kind == ShredKind::ScalarArray) {
			// A non-NULL element slot is a lifted scalar (emit it; the skeleton placeholder is skipped);
			// a NULL slot is a kept element (a non-conforming scalar / null / object / array), emitted
			// verbatim from the skeleton. Validity alone disambiguates the lifted from the kept.
			auto elem_idx = shred.element_fmt.sel->get_index(child);
			if (shred.element_fmt.validity.RowIsValid(elem_idx)) {
				EmitReconShredScalar(builder, shred.element_type, shred.element_fmt, elem_idx);
				SkipValueFast(view, cursor);
			} else {
				EmitValueVerbatim(view, cursor, builder, depth + 1);
			}
			i++;
			continue;
		}
		bool is_object = SlotTag(view.SlotAt(cursor.pos)) == tag::OBJ_START;
		bool struct_present = is_object && shred.struct_fmt.validity.RowIsValid(shred.struct_fmt.sel->get_index(child));
		size_t present = 0;
		if (struct_present) {
			for (size_t j = 0; j < shred.subfields.size(); j++) {
				if (shred.sub_fmt[j].validity.RowIsValid(shred.sub_fmt[j].sel->get_index(child))) {
					present++;
				}
			}
		}
		if (present > 0) {
			// The patch object must list its keys ascending (sorted_subfields), so the merge's
			// sorted-key two-pointer walk and the JSONO object invariant both hold.
			scratch.patch_builder.Reset();
			scratch.patch_builder.EmitObjectStart(present);
			for (auto j : shred.sorted_subfields) {
				if (shred.sub_fmt[j].validity.RowIsValid(shred.sub_fmt[j].sel->get_index(child))) {
					scratch.patch_builder.EmitKeySlot(
					    nonstd::string_view(shred.subfields[j].first.data(), shred.subfields[j].first.size()));
				}
			}
			for (auto j : shred.sorted_subfields) {
				auto sub_idx = shred.sub_fmt[j].sel->get_index(child);
				if (shred.sub_fmt[j].validity.RowIsValid(sub_idx)) {
					scratch.patch_builder.EmitObjectChildStart();
					EmitReconShredScalar(scratch.patch_builder, shred.subfields[j].second, shred.sub_fmt[j], sub_idx);
				}
			}
			scratch.patch_builder.EmitObjectEnd();
			SerializeBuilderToBlob(scratch.patch_builder, scratch.patch_storage);
			JsonoView patch_view = ViewOfBlob(scratch.patch_storage);
			patch_view.ParseHeader();
			// A authoritative (the skeleton element keeps its tail and any kept null/diverted subfield);
			// B fills the lifted subfields that were stripped from the skeleton.
			MergeTwoObjects(view, cursor, patch_view, JsonoCursor(), builder, MergeMode::Overlay, depth + 1);
			SkipValueFast(view, cursor);
		} else {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
		i++;
	}
	if (i != entry.length) {
		throw InvalidInputException("malformed JSONO: array shred list is longer than the residual array "
		                            "(a struct cast altered it)");
	}
	builder.EmitArrayEnd();
	cursor.pos = end_pos + 1;
}

// Walk the scalar-overlaid value, replacing each array-shred path's skeleton array with the
// overlaid array; advances `cursor` past the emitted value (mirrors EmitValueVerbatim).
void EmitArrayOverlay(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder,
                      const std::vector<const ArrayReconShred *> &array_shreds, idx_t row, ArrayOverlayScratch &scratch,
                      size_t depth) {
	if (SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
		EmitValueVerbatim(view, cursor, builder, depth);
		return;
	}
	auto layout = ReadObjectLayout(view, cursor.pos);
	auto key_at = [&](size_t i) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		return view.KeyAt(SlotPayload(key_slot));
	};
	builder.EmitObjectStart(layout.key_count);
	for (size_t i = 0; i < layout.key_count; i++) {
		builder.EmitKeySlot(key_at(i));
	}
	cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		builder.EmitObjectChildStart();
		const ArrayReconShred *terminal = nullptr;
		std::vector<const ArrayReconShred *> deeper;
		for (auto *shred : array_shreds) {
			if (PathTerminatesOnKey(shred->path, depth, key)) {
				terminal = shred;
			} else if (PathContinuesPastKey(shred->path, depth, key)) {
				deeper.push_back(shred);
			}
		}
		auto value_tag = SlotTag(view.SlotAt(cursor.pos));
		if (terminal && value_tag == tag::ARR_START) {
			OverlayArray(view, cursor, builder, *terminal, row, scratch, depth);
		} else if (!deeper.empty() && value_tag == tag::OBJ_START) {
			EmitArrayOverlay(view, cursor, builder, deeper, row, scratch, depth + 1);
		} else {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
	}
	builder.EmitObjectEnd();
	if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	cursor.pos++;
}

// Reconstruct a shredded JSONO `input` (six-BLOB residual + shred columns) into the plain
// JSONO `result` by overlaying each row's shred values back onto its residual object. Scalar
// shreds are disjoint from residual keys (the shred invariant), so the overlay re-inserts them:
// top-level shreds as one flat patch (the common jsono({...}) case), nested-path shreds (`$.a.b`)
// as a one-path chain overlay each. An array shred (LIST<STRUCT>) instead carries a skeleton array
// in the residual; after the scalar overlay, EmitArrayOverlay rebuilds each such array in lockstep.
// A valid row with a NULL residual blob never comes from the writer (a NULL value nulls the
// whole struct): it throws — the row is the trace of a struct cast NULL-filling the body.
// `shred_filter` (shred indices over the shred set) restricts the overlay to those shreds — the
// single-pass narrowing reshred folds only the shreds the target drops back into the residual;
// the manifest is still verified against ALL of the type's shreds.
void ReconstructShreddedToPlainImpl(Vector &input, idx_t count, Vector &result,
                                    const vector<idx_t> *shred_filter = nullptr) {
	JsonoLayoutType layout;
	TryParseJsonoLayoutType(input.GetType(), layout);
	vector<ReconShred> shreds;
	vector<ArrayReconShred> array_shreds;
	bool all_top_level = true;
	for (idx_t i = 0; i < layout.shreds.size(); i++) {
		if (shred_filter && std::find(shred_filter->begin(), shred_filter->end(), i) == shred_filter->end()) {
			continue;
		}
		// Shred field names ARE the clean path (no fingerprint suffix in the nested layout).
		auto &name = layout.shreds[i].first;
		vector<PathStep> steps;
		if (name.size() >= 2 && name[0] == '$' && name[1] == '.') {
			steps = ParseJsonoPath(name, "jsono reconstruct");
		} else {
			steps.push_back(PathStep {PathStepKind::Key, name, 0});
		}
		for (auto &step : steps) {
			if (step.kind != PathStepKind::Key) {
				throw InvalidInputException("jsono reconstruct: shred path '%s' is not an object-key path", name);
			}
		}
		if (IsShredArrayType(layout.shreds[i].second)) {
			ArrayReconShred ars;
			ars.child = i;
			ars.kind = ShredKind::Array;
			ars.path = std::move(steps);
			auto &element = ListType::GetChildType(layout.shreds[i].second);
			for (auto &sub : StructType::GetChildTypes(element)) {
				ars.subfields.emplace_back(sub.first, sub.second);
			}
			array_shreds.push_back(std::move(ars));
			continue;
		}
		if (IsShredScalarArrayType(layout.shreds[i].second)) {
			ArrayReconShred ars;
			ars.child = i;
			ars.kind = ShredKind::ScalarArray;
			ars.path = std::move(steps);
			ars.element_type = ListType::GetChildType(layout.shreds[i].second);
			array_shreds.push_back(std::move(ars));
			continue;
		}
		if (steps.size() != 1) {
			all_top_level = false;
		}
		shreds.push_back(ReconShred {i, layout.shreds[i].second, std::move(steps)});
	}
	std::sort(shreds.begin(), shreds.end(), [](const ReconShred &a, const ReconShred &b) {
		auto n = std::min(a.steps.size(), b.steps.size());
		for (size_t i = 0; i < n; i++) {
			if (a.steps[i].key != b.steps[i].key) {
				return a.steps[i].key < b.steps[i].key;
			}
		}
		return a.steps.size() < b.steps.size();
	});

	// The row reader verifies each row's shred manifest against the shreds this type carries
	// (the manifest is checked against ALL of them even under a shred_filter).
	JsonoRowReader residual_reader;
	residual_reader.Init(input, count);
	vector<UnifiedVectorFormat> shred_fmt(shreds.size());
	for (idx_t k = 0; k < shreds.size(); k++) {
		JsonoShredVector(input, shreds[k].child).ToUnifiedFormat(count, shred_fmt[k]);
	}
	// Array shred read handles: the per-row list entries plus, per kind, the element lanes —
	// a LIST<STRUCT>'s per-element struct validity and per-subfield value+validity, or a
	// LIST<scalar>'s element value+validity — set up once the array_shreds vector is stable.
	for (auto &ars : array_shreds) {
		auto &list_vec = JsonoShredVector(input, ars.child);
		list_vec.ToUnifiedFormat(count, ars.list_fmt);
		ars.list_entries = UnifiedVectorFormat::GetData<list_entry_t>(ars.list_fmt);
		auto child_size = ListVector::GetListSize(list_vec);
		if (ars.kind == ShredKind::ScalarArray) {
			ListVector::GetEntry(list_vec).ToUnifiedFormat(child_size, ars.element_fmt);
			continue;
		}
		auto &struct_vec = ListVector::GetEntry(list_vec);
		struct_vec.ToUnifiedFormat(child_size, ars.struct_fmt);
		auto &subs = StructVector::GetEntries(struct_vec);
		ars.sub_fmt.resize(subs.size());
		for (idx_t j = 0; j < subs.size(); j++) {
			subs[j]->ToUnifiedFormat(child_size, ars.sub_fmt[j]);
		}
		ars.sorted_subfields.resize(ars.subfields.size());
		for (idx_t j = 0; j < ars.subfields.size(); j++) {
			ars.sorted_subfields[j] = j;
		}
		std::sort(ars.sorted_subfields.begin(), ars.sorted_subfields.end(),
		          [&](idx_t a, idx_t b) { return ars.subfields[a].first < ars.subfields[b].first; });
	}
	std::vector<const ArrayReconShred *> array_shred_ptrs;
	for (auto &ars : array_shreds) {
		array_shred_ptrs.push_back(&ars);
	}

	JsonoBodyWriter writer;
	writer.Init(result);

	JsonoBuilder patch_builder;
	JsonoBuilder out_builder;
	JsonoBuilder final_builder;
	OwnedJsonoBlob patch_storage;
	OwnedJsonoBlob acc_storage;
	OwnedJsonoBlob scalar_storage;
	ArrayOverlayScratch overlay_scratch;
	JsonoView residual_view;
	// Finish a row: with array shreds, rebuild each skeleton array over the scalar-overlaid value;
	// otherwise write the scalar overlay directly.
	auto finish_row = [&](idx_t row, JsonoBuilder &value_builder) {
		if (array_shred_ptrs.empty()) {
			writer.WriteRow(row, value_builder);
			return;
		}
		SerializeBuilderToBlob(value_builder, scalar_storage);
		JsonoView scalar_view = ViewOfBlob(scalar_storage);
		scalar_view.ParseHeader();
		final_builder.Reset();
		JsonoCursor cursor;
		EmitArrayOverlay(scalar_view, cursor, final_builder, array_shred_ptrs, row, overlay_scratch, 0);
		writer.WriteRow(row, final_builder);
	};
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob {};
		if (residual_reader.Read(row, blob, residual_view) != JsonoRowState::Value) {
			writer.SetRowNull(row);
			continue;
		}
		if (SlotTag(residual_view.SlotAt(0)) != tag::OBJ_START) {
			// A non-object residual (scalar/array) is the whole value: object-key shreds (scalar or
			// array) cannot match it, so emit it verbatim — no overlay applies.
			out_builder.Reset();
			JsonoCursor cursor;
			EmitValueVerbatim(residual_view, cursor, out_builder, 0);
			writer.WriteRow(row, out_builder);
			continue;
		}
		if (shreds.empty()) {
			// Array-only shred set: no scalar overlay, the residual is the skeleton to rebuild.
			out_builder.Reset();
			JsonoCursor cursor;
			EmitValueVerbatim(residual_view, cursor, out_builder, 0);
			finish_row(row, out_builder);
			continue;
		}
		if (all_top_level) {
			// One flat patch object of the present shreds, overlaid in a single pass.
			patch_builder.Reset();
			idx_t present = 0;
			for (idx_t k = 0; k < shreds.size(); k++) {
				present += RowIsValid(shred_fmt[k], row) ? 1 : 0;
			}
			patch_builder.EmitObjectStart(present);
			for (idx_t k = 0; k < shreds.size(); k++) {
				if (RowIsValid(shred_fmt[k], row)) {
					patch_builder.EmitKeySlot(shreds[k].steps[0].key);
				}
			}
			for (idx_t k = 0; k < shreds.size(); k++) {
				if (!RowIsValid(shred_fmt[k], row)) {
					continue;
				}
				patch_builder.EmitObjectChildStart();
				EmitReconShredScalar(patch_builder, shreds[k].type, shred_fmt[k], RowIndex(shred_fmt[k], row));
			}
			patch_builder.EmitObjectEnd();
			SerializeBuilderToBlob(patch_builder, patch_storage);
			JsonoView patch_view = ViewOfBlob(patch_storage);
			patch_view.ParseHeader();
			out_builder.Reset();
			MergeTwoObjects(residual_view, JsonoCursor(), patch_view, JsonoCursor(), out_builder, MergeMode::Overlay,
			                0);
			finish_row(row, out_builder);
			continue;
		}
		// Nested shreds: overlay one single-path chain at a time onto the accumulator. The
		// residual is authoritative; each disjoint shred path fills in where it was stripped.
		const JsonoView *acc = &residual_view;
		JsonoView acc_view = residual_view;
		bool any = false;
		for (idx_t k = 0; k < shreds.size(); k++) {
			if (!RowIsValid(shred_fmt[k], row)) {
				continue;
			}
			patch_builder.Reset();
			for (auto &step : shreds[k].steps) {
				patch_builder.EmitObjectStart(1);
				patch_builder.EmitKeySlot(step.key);
				patch_builder.EmitObjectChildStart();
			}
			EmitReconShredScalar(patch_builder, shreds[k].type, shred_fmt[k], RowIndex(shred_fmt[k], row));
			for (idx_t s = 0; s < shreds[k].steps.size(); s++) {
				patch_builder.EmitObjectEnd();
			}
			SerializeBuilderToBlob(patch_builder, patch_storage);
			JsonoView patch_view = ViewOfBlob(patch_storage);
			patch_view.ParseHeader();
			out_builder.Reset();
			MergeTwoObjects(*acc, JsonoCursor(), patch_view, JsonoCursor(), out_builder, MergeMode::Overlay, 0);
			SerializeBuilderToBlob(out_builder, acc_storage);
			acc_view = ViewOfBlob(acc_storage);
			acc_view.ParseHeader();
			acc = &acc_view;
			any = true;
		}
		out_builder.Reset();
		JsonoCursor cursor;
		EmitValueVerbatim(any ? *acc : residual_view, cursor, out_builder, 0);
		finish_row(row, out_builder);
	}
}

// Fold the residual views in `inputs` left-to-right under `mode` and write the merged plain
// JSONO into `out` (the six-BLOB residual). Shared by the shred-aware fast path (raw residuals)
// and the reshred fallback (reconstructed values). Each input reader carries its own manifest
// policy: the fold rebuilds the residual without the inputs' manifests, so an unverified
// narrowed input would launder the loss into a permanently undetectable one — the readers throw
// before rebuilding (except jsono_overlay's trusted residual, see JsonoFoldExecute).
void RunResidualFold(MergeMode mode, vector<JsonoRowReader> &inputs, idx_t ncols, idx_t count, Vector &out,
                     JsonoOpsLocalState &lstate) {
	JsonoBodyWriter writer;
	writer.Init(out);
	// jsono_merge_patch folds left to right (RFC 7396): the first argument is the
	// target document (kept verbatim — its nulls are values), the rest are patches.
	static thread_local OwnedJsonoBlob acc_storage;
	JsonoView acc_view;
	JsonoView patch_view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow target_blob {};
		bool acc_is_sqlnull = inputs[0].Read(row, target_blob, acc_view) != JsonoRowState::Value;
		if (ncols == 1) {
			// A lone target is returned verbatim (its nulls survive).
			if (acc_is_sqlnull) {
				writer.SetRowNull(row);
			} else {
				lstate.builder.Reset();
				JsonoCursor cursor;
				EmitValueVerbatim(acc_view, cursor, lstate.builder, 0);
				writer.WriteRow(row, lstate.builder);
			}
			continue;
		}
		for (idx_t i = 1; i < ncols; i++) {
			JsonoBlobRow patch_blob {};
			bool patch_present = inputs[i].Read(row, patch_blob, patch_view) == JsonoRowState::Value;
			bool result_sqlnull =
			    MergeFoldStep(mode, acc_is_sqlnull, acc_view, patch_present, patch_view, lstate.builder,
			                  lstate.merge_children_a, lstate.merge_children_b, lstate.merge_plan);
			if (i + 1 == ncols) {
				// Last step writes straight to the output builder — no serialize round-trip.
				if (result_sqlnull) {
					writer.SetRowNull(row);
				} else {
					writer.WriteRow(row, lstate.builder);
				}
			} else if (result_sqlnull) {
				acc_is_sqlnull = true;
			} else {
				SerializeBuilderToBlob(lstate.builder, acc_storage);
				acc_view = ViewOfBlob(acc_storage);
				acc_view.ParseHeader();
				acc_is_sqlnull = false;
			}
		}
	}
}

// True if the residual object has `key` at top level (binary search over the sorted keys).
bool ResidualHasTopLevelKey(const JsonoView &view, const string &key) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return false;
	}
	auto layout = ReadObjectLayout(view, 0);
	nonstd::string_view target(key);
	size_t lo = 0;
	size_t hi = layout.key_count;
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto ks = view.SlotAt(layout.key_start + mid);
		if (SlotTag(ks) != tag::KEY) {
			return false;
		}
		if (view.KeyAt(SlotPayload(ks)) < target) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	if (lo >= layout.key_count) {
		return false;
	}
	auto ks = view.SlotAt(layout.key_start + lo);
	return SlotTag(ks) == tag::KEY && view.KeyAt(SlotPayload(ks)) == target;
}

// True if the object has any top-level key that matches a merged shred name. `shreds`
// is sorted by name (bind canonicalizes it), so each of the object's (few) keys is
// binary-searched. Used to gate the fast path against a PLAIN input whose top-level
// key names a shred: a value there collides into the residual (also caught by the
// per-row residual probe), but an RFC 7396 null at that key DELETES the lane and
// leaves no trace in the folded residual — the lane copy-through would wrongly keep
// it, so any such key forces the reshred fallback.
bool ObjectKeyInShredSet(const JsonoView &view, const vector<MergeShred> &shreds) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return false;
	}
	auto layout = ReadObjectLayout(view, 0);
	for (size_t k = 0; k < layout.key_count; k++) {
		auto ks = view.SlotAt(layout.key_start + k);
		if (SlotTag(ks) != tag::KEY) {
			break;
		}
		auto key = view.KeyAt(SlotPayload(ks));
		size_t lo = 0;
		size_t hi = shreds.size();
		while (lo < hi) {
			auto mid = lo + (hi - lo) / 2;
			if (nonstd::string_view(shreds[mid].name) < key) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		if (lo < shreds.size() && nonstd::string_view(shreds[lo].name) == key) {
			return true;
		}
	}
	return false;
}

// Conflict between a value (folded residual or a plain input) and a NESTED object-key shred
// path: the lane copy-through is valid only when the value neither carries the path's leaf nor
// breaks the path's object skeleton. True when the path resolves to a PRESENT value (the value
// shadows the lane, or an RFC 7396 null at the path would delete it) OR a node along the path
// before the leaf is a present non-object (the lane cannot overlay into a scalar/array). An
// absent key short-circuits to no-conflict — the lane safely owns/creates that subtree.
bool ResidualConflictsWithShredPath(const JsonoView &view, const vector<PathStep> &steps) {
	JsonoCursor cursor;
	for (idx_t s = 0; s < steps.size(); s++) {
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			return true; // need to descend into an object but found a non-object/empty
		}
		if (!LocatePathStep(nullptr, s, view, steps[s], cursor)) {
			return false; // key absent at this level (the node is an object) → no conflict
		}
	}
	return true; // full path resolved to a present value
}

// A scalar-array shred deliberately leaves an array skeleton at the shred path in the residual.
// That terminal array is not a conflict: the copied LIST lane overlays it element-for-element on
// reconstruct. Anything else present at that path is a real replacement/collision and must fall
// back to the full reshred path.
bool ResidualConflictsWithScalarArrayShredPath(const JsonoView &view, const vector<PathStep> &steps) {
	JsonoCursor cursor;
	for (idx_t s = 0; s < steps.size(); s++) {
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			return true;
		}
		if (!LocatePathStep(nullptr, s, view, steps[s], cursor)) {
			return false;
		}
	}
	return cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::ARR_START;
}

// True if two object-key shred paths overlap structurally: they agree on the full length of the
// shorter one, so one is a prefix of (or equal to) the other (e.g. `a` and `$.a.a`, or `a` and
// `$.a`). Such a pair cannot be two independent lanes — one names a scalar leaf, the other lives
// under it — so only the actual merge (the reshred fallback) can decide which structure wins.
// Sibling paths (`$.a.b` vs `$.a.c`) diverge before the shorter ends and do not conflict.
bool ShredPathsStructurallyConflict(const vector<PathStep> &a, const vector<PathStep> &b) {
	idx_t common = std::min(a.size(), b.size());
	for (idx_t i = 0; i < common; i++) {
		if (a[i].key != b[i].key) {
			return false;
		}
	}
	return true;
}

// Copy a shred column from the winning input into the result shred, then null it where the
// merged value is SQL NULL. Patch takes the last input that declares the shred (RFC 7396
// last-wins); Overlay takes the first (base-authoritative).
void FastCopyShred(DataChunk &args, idx_t ncols, idx_t count, MergeMode mode, const MergeShred &shred, Vector &dst,
                   const ValidityMask &result_validity) {
	idx_t src_arg = DConstants::INVALID_INDEX;
	idx_t src_child = 0;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (!IsShreddedJsonoType(t)) {
			continue; // only shredded inputs declare shreds to copy from
		}
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(t, layout);
		for (idx_t c = 0; c < layout.shreds.size(); c++) {
			if (layout.shreds[c].first == shred.name &&
			    (mode != MergeMode::Overlay || src_arg == DConstants::INVALID_INDEX)) {
				src_arg = i;
				src_child = c;
			}
		}
	}
	dst.SetVectorType(VectorType::FLAT_VECTOR);
	if (src_arg == DConstants::INVALID_INDEX) {
		auto &dv = FlatVector::Validity(dst);
		for (idx_t row = 0; row < count; row++) {
			dv.SetInvalid(row);
		}
		return;
	}
	args.data[src_arg].Flatten(count);
	VectorOperations::Copy(JsonoShredVector(args.data[src_arg], src_child), dst, count, 0, 0);
	if (!result_validity.AllValid()) {
		dst.Flatten(count);
		auto &dv = FlatVector::Validity(dst);
		for (idx_t row = 0; row < count; row++) {
			if (!result_validity.RowIsValid(row)) {
				dv.SetInvalid(row);
			}
		}
	}
}

void JsonoFoldExecute(DataChunk &args, ExpressionState &state, Vector &result, MergeMode mode) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoOpsLocalState>();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<JsonoMergeBindData>();
	auto count = args.size();
	idx_t ncols = args.ColumnCount();
	bool has_shreds = !bind_data.shreds.empty();

	// Each input reader verifies its rows' manifests against that input's own shreds (a plain
	// input carries none, so any manifest entry on it fails loud). jsono_overlay is exempt: it
	// is the optimizer's reconstruction primitive and its residual argument legitimately
	// carries a manifest already verified by __jsono_internal_checked_residual.
	auto init_input = [&](JsonoRowReader &reader, Vector &input) {
		if (mode == MergeMode::Overlay) {
			reader.InitTrusted(input, count);
		} else {
			reader.Init(input, count);
		}
	};

	if (!has_shreds) {
		// Plain merge: fold straight into the result.
		vector<JsonoRowReader> inputs(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			init_input(inputs[i], args.data[i]);
		}
		RunResidualFold(mode, inputs, ncols, count, result, lstate);
		if (args.AllConstant()) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Fast path: fold the raw residuals and copy shred lanes through verbatim, skipping the
	// per-row reconstruct+reshred. Both shredded AND plain inputs ride it: a plain input folds
	// its whole value into the residual and declares no lanes (FastCopyShred skips it). Every
	// shred is an object-key lane (top-level `K` or nested `$.p.q`) the read overlay re-inserts
	// at its path; a scalar-array shred copies through with its residual skeleton. LIST<STRUCT>
	// array shreds still fall back because each element merges multiple subfield lanes over a
	// per-element tail. The per-row gate diverts to the fallback whenever an input could make the
	// lane copy-through wrong (a non-object replace, a key/path naming a lane).
	bool fast_viable = true;
	vector<idx_t> plain_inputs;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (t.id() != LogicalTypeId::SQLNULL && !IsShreddedJsonoType(t)) {
			plain_inputs.push_back(i);
		}
	}
	// Parse each shred's object-key path once (top-level shreds are a single Key step). An array
	// shred with element objects, or any non-Key step (only reachable through an array path),
	// disqualifies the fast path.
	vector<vector<PathStep>> shred_paths(bind_data.shreds.size());
	vector<ShredKind> shred_kinds(bind_data.shreds.size());
	bool has_jsonpath_shred = false;
	bool has_nested_shred = false;
	for (idx_t k = 0; k < bind_data.shreds.size() && fast_viable; k++) {
		auto &shred = bind_data.shreds[k];
		shred_kinds[k] = ClassifyShredKind(shred.type);
		if (shred_kinds[k] == ShredKind::Array) {
			fast_viable = false;
			break;
		}
		if (shred.name.size() >= 2 && shred.name[0] == '$' && shred.name[1] == '.') {
			has_jsonpath_shred = true;
			shred_paths[k] = ParseJsonoPath(shred.name, "jsono merge fast path");
			for (auto &step : shred_paths[k]) {
				if (step.kind != PathStepKind::Key) {
					fast_viable = false;
					break;
				}
			}
			if (shred_paths[k].size() > 1) {
				has_nested_shred = true;
			}
		} else {
			shred_paths[k].push_back(PathStep {PathStepKind::Key, shred.name, 0});
		}
	}
	// A nested shred whose path prefixes (or equals) another shred's path is structurally
	// contradictory — only the reshred fallback can resolve which wins. Two distinct top-level
	// keys never prefix each other, so this only matters once a nested shred is present.
	if (fast_viable && has_nested_shred) {
		for (idx_t a = 0; a < shred_paths.size() && fast_viable; a++) {
			for (idx_t b = a + 1; b < shred_paths.size(); b++) {
				if (ShredPathsStructurallyConflict(shred_paths[a], shred_paths[b])) {
					fast_viable = false;
					break;
				}
			}
		}
	}

	if (fast_viable) {
		vector<JsonoRowReader> raw(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			init_input(raw[i], args.data[i]);
		}
		Vector fast_residual(JsonoType(), count);
		RunResidualFold(mode, raw, ncols, count, fast_residual, lstate);
		JsonoVectorData fr;
		InitJsonoVectorData(fast_residual, count, fr);
		bool conflict = false;
		JsonoView pview;
		JsonoBlobRow pblob {};
		for (idx_t row = 0; row < count && !conflict; row++) {
			JsonoBlobRow blob {};
			if (!ReadJsonoRowStrict(fr, row, blob)) {
				continue; // SQL NULL folded residual: the row is NULL and its lanes are nulled below
			}
			JsonoView v = MakeJsonoView(blob);
			if (!v.ParseHeader()) {
				continue;
			}
			// A non-object merged value (a non-object plain patch replaced the document) carries no
			// lanes; copying them would attach stale lanes to a scalar/array. The all-shredded case
			// never folds to a non-object with non-null lanes, so gate this on plain inputs to keep
			// that case on the fast path unchanged.
			if (!plain_inputs.empty() && SlotTag(v.SlotAt(0)) != tag::OBJ_START) {
				conflict = true;
				break;
			}
			for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
				// A top-level shred conflicts only if its key sits in the folded residual (a
				// non-object residual is handled by the plain-input gate above, preserving the
				// all-shredded behaviour). A nested shred additionally conflicts if a node along
				// its path is a present non-object (its lane cannot overlay into a scalar).
				bool hit = shred_kinds[k] == ShredKind::ScalarArray
				               ? ResidualConflictsWithScalarArrayShredPath(v, shred_paths[k])
				               : (shred_paths[k].size() == 1 ? ResidualHasTopLevelKey(v, bind_data.shreds[k].name)
				                                             : ResidualConflictsWithShredPath(v, shred_paths[k]));
				if (hit) {
					conflict = true;
					break;
				}
			}
			if (conflict) {
				break;
			}
			for (idx_t pi : plain_inputs) {
				if (raw[pi].Read(row, pblob, pview) != JsonoRowState::Value) {
					continue;
				}
				// A non-object plain value replaces the whole document mid-fold (RFC 7396),
				// discarding the base's lanes — but the lane copy-through still copies them, and a
				// later object patch can rebuild an object residual so the non-object check on the
				// folded residual alone misses it. Divert to the fallback on any non-object plain input.
				if (pview.Slots() == 0 || SlotTag(pview.SlotAt(0)) != tag::OBJ_START) {
					conflict = true;
					break;
				}
				// A plain input naming a shred also forces the fallback: a value there is caught by
				// the residual probe above, but an RFC 7396 null-delete of a lane leaves no trace in
				// the folded residual and the lane copy-through would wrongly keep it. ObjectKeyInShredSet
				// covers the top-level shred names in one pass; a plain input touching a NESTED shred path
				// (its leaf or a non-object along it) needs the per-path descent.
				if (ObjectKeyInShredSet(pview, bind_data.shreds)) {
					conflict = true;
					break;
				}
				if (has_jsonpath_shred) {
					for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
						auto &name = bind_data.shreds[k].name;
						if (name.size() >= 2 && name[0] == '$' && name[1] == '.' &&
						    ResidualConflictsWithShredPath(pview, shred_paths[k])) {
							conflict = true;
							break;
						}
					}
					if (conflict) {
						break;
					}
				}
			}
		}
		if (!conflict) {
			// Capture constness before FastCopyShred flattens any input (which would otherwise
			// make AllConstant() spuriously false and trip the constant-fold assertion).
			bool all_constant = args.AllConstant();
			fast_residual.Flatten(count);
			// Copy the folded plain residual's body blobs into the shredded result's body —
			// except `skips`, which is re-emitted per row below with the output's shred manifest.
			JsonoBodyWriter writer;
			writer.Init(result);
			// The fast path is reached only when scalar shred keys do not appear in the folded
			// residual. Scalar-array keys may appear as their normal skeleton arrays, but they do not
			// participate in the per-shred completeness typed-scalar shortcut. A diverted scalar (case B)
			// would sit in that residual at the scalar shred's key and trip the gate to the fallback,
			// so every NULL scalar shred here is an absent path (case A), never a diversion.
			JsonoFillShredsComplete(result, count);
			auto &fr_blobs = StructVector::GetEntries(JsonoBodyVector(fast_residual));
			FlatVector::Validity(result) = FlatVector::Validity(fast_residual);
			for (idx_t b = 0; b < BODY_BLOB_COUNT; b++) {
				if (b == BODY_SKIPS) {
					continue;
				}
				VectorOperations::Copy(*fr_blobs[b], *writer.vec[b], count, 0, 0);
			}
			auto &result_validity = FlatVector::Validity(result);
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row)) {
					JsonoSetRowMarkerNull(result, row);
				}
			}
			for (auto &shred : bind_data.shreds) {
				FastCopyShred(args, ncols, count, mode, shred, JsonoShredVector(result, shred.result_child_index),
				              result_validity);
			}
			// The folded residual was rebuilt without the inputs' manifests, but a copied shred
			// that carries a value has no copy in the residual (the no-conflict gate above) —
			// exactly what the manifest must record, or a later raw narrowing cast would lose it
			// silently. Re-emit each row's skips with the manifest of its non-NULL shreds.
			vector<JsonoShredManifestEntryBytes> manifest_entries(bind_data.shreds.size());
			vector<UnifiedVectorFormat> shred_fmt(bind_data.shreds.size());
			for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
				auto &shred = bind_data.shreds[k];
				manifest_entries[k] = JsonoShredManifestEntry(shred.name, shred.type);
				JsonoShredVector(result, shred.result_child_index).ToUnifiedFormat(count, shred_fmt[k]);
			}
			auto fr_skips = FlatVector::GetData<string_t>(*fr_blobs[BODY_SKIPS]);
			auto &fr_skips_validity = FlatVector::Validity(*fr_blobs[BODY_SKIPS]);
			auto &r_skips = writer.Skips();
			auto skips_out = writer.data[BODY_SKIPS];
			std::string skips_buf;
			vector<idx_t> stripped_fields;
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row) || !fr_skips_validity.RowIsValid(row)) {
					FlatVector::SetNull(r_skips, row, true);
					continue;
				}
				skips_buf.clear();
				skips_buf.append(fr_skips[row].GetData(), fr_skips[row].GetSize());
				stripped_fields.clear();
				for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
					if (RowIsValid(shred_fmt[k], row)) {
						stripped_fields.push_back(k);
					}
				}
				if (!stripped_fields.empty()) {
					JsonoAppendShredManifest(skips_buf, manifest_entries, stripped_fields);
				}
				skips_out[row] = WriteBlobInto(r_skips, skips_buf.data(), skips_buf.size());
			}
			if (all_constant) {
				result.SetVectorType(VectorType::CONSTANT_VECTOR);
			}
			return;
		}
		// A shred key landed in the residual (conflict) — fall through to the correct reshred path.
	}

	// Reshred fallback: reconstruct shredded inputs to plain, fold, reshred. Correct for any
	// shred/residual key overlap and for plain inputs that can shadow a shred.
	vector<unique_ptr<Vector>> reconstructed;
	vector<JsonoRowReader> inputs(ncols);
	for (idx_t i = 0; i < ncols; i++) {
		if (IsShreddedJsonoType(args.data[i].GetType())) {
			// The reconstruction verifies the shredded input's manifest itself; its plain
			// output never carries one.
			auto plain = make_uniq<Vector>(JsonoType(), count);
			ReconstructShreddedToPlainImpl(args.data[i], count, *plain);
			init_input(inputs[i], *plain);
			reconstructed.push_back(std::move(plain));
		} else {
			init_input(inputs[i], args.data[i]);
		}
	}
	Vector fold_out(JsonoType(), count);
	RunResidualFold(mode, inputs, ncols, count, fold_out, lstate);
	vector<std::pair<string, LogicalType>> shred_specs;
	shred_specs.reserve(bind_data.shreds.size());
	for (auto &shred : bind_data.shreds) {
		shred_specs.emplace_back(shred.name, shred.type);
	}
	JsonoShredFromSpec(fold_out, count, shred_specs, result);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoMergePatchExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	JsonoFoldExecute(args, state, result, MergeMode::Patch);
}

void JsonoOverlayExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	JsonoFoldExecute(args, state, result, MergeMode::Overlay);
}

struct GroupMergeBindData : public FunctionData {
	MergeMode merge_mode;
	// Empty for a plain input (Finalize writes the plain accumulator verbatim). For a shredded
	// input the captured shreds (clean names, canonical order) reshred the merged accumulator back
	// to the input's shredded type so the shreds "stick" across the aggregation.
	vector<std::pair<string, LogicalType>> shreds;

	explicit GroupMergeBindData(MergeMode merge_mode) : merge_mode(merge_mode) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<GroupMergeBindData>(merge_mode);
		result->shreds = shreds;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<GroupMergeBindData>();
		return merge_mode == other.merge_mode && shreds == other.shreds;
	}
};

struct GroupMergeState {
	OwnedJsonoBlob *acc;
	bool has_input;
};

struct GroupMergeFunction {
	static void Initialize(GroupMergeState &state) {
		state.acc = nullptr;
		state.has_input = false;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.acc;
		state.acc = nullptr;
		state.has_input = false;
	}
};

unique_ptr<FunctionData> JsonoGroupMergeBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw BinderException("jsono_group_merge() requires a single JSONO argument");
	}
	auto &arg = *arguments[0];
	if (arg.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto &type = arg.return_type;
	JsonoRequireExtensionOptimizerForShredded(context, type, "jsono_group_merge");
	auto bind_data = make_uniq<GroupMergeBindData>(MergeMode::IgnoreNulls);
	if (IsShreddedJsonoType(type)) {
		// Capture the shreds (clean names, the type's canonical order) so Finalize reshreds the
		// merged accumulator back to this exact shredded type — sticky shredding, like
		// jsono_merge_patch. Redeclaring the argument as plain JSONO makes the binder insert the
		// lossless reconstruct cast, so Update/Combine see plain JSONO and the accumulator stays plain.
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(type, layout);
		for (auto &shred : layout.shreds) {
			bind_data->shreds.emplace_back(shred.first, shred.second);
		}
		// Keep the shredded argument type (Update reconstructs it to plain itself) rather than
		// forcing a plain argument with a binder cast. With a plain argument the bound aggregate's
		// argument type and its sticky shredded return type disagree, so a plan serialize→deserialize
		// re-bind (debug verification) would re-derive a plain return and fail to reconcile it back to
		// the shredded type. Arg type == return type makes the re-bind reproduce the same shredded type.
		function.arguments[0] = type;
		function.return_type = type;
	} else if (type.id() == LogicalTypeId::SQLNULL || IsJsonoType(type)) {
		function.arguments[0] = JsonoType();
	} else {
		throw BinderException("jsono_group_merge() input must be JSONO");
	}
	return std::move(bind_data);
}

// Fold one incoming value (a row blob, or a partial's accumulated blob) into the
// group state, matching jsono_group_merge IGNORE NULLS semantics: object+object →
// cursor merge (incoming wins per key, its nulls ignored, A kept on B-null, keys whose
// object value strips to empty omitted); a fresh object (no prior object accumulator)
// is seeded with nulls/empty-objects stripped; a non-object incoming replaces the
// accumulator wholesale (verbatim, including a top-level null) — the same rule
// jsono_merge_patch applies to a non-object patch. The merged builder is
// re-serialized so the next fold can view the accumulator.
void FoldIntoGroupState(GroupMergeState &state, const JsonoView &incoming) {
	static thread_local JsonoBuilder scratch;
	scratch.Reset();
	JsonoCursor cursor;
	bool incoming_is_object = SlotTag(incoming.SlotAt(0)) == tag::OBJ_START;
	bool merged_objects = false;
	if (incoming_is_object && state.acc) {
		JsonoView acc_view = ViewOfBlob(*state.acc);
		if (acc_view.ParseHeader() && acc_view.Slots() > 0 && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START) {
			MergeTwoObjects(acc_view, JsonoCursor(), incoming, JsonoCursor(), scratch, MergeMode::IgnoreNulls, 0);
			merged_objects = true;
		}
	}
	if (!merged_objects) {
		if (incoming_is_object) {
			EmitValueStrip(incoming, cursor, scratch, MergeMode::IgnoreNulls, 0);
		} else {
			EmitValueVerbatim(incoming, cursor, scratch, 0);
		}
	}
	if (!state.acc) {
		state.acc = new OwnedJsonoBlob();
	}
	SerializeBuilderToBlob(scratch, *state.acc);
	state.has_input = true;
}

// Shredded group_merge input is kept native at bind (so the aggregate's arg and sticky
// return types agree across a debug serialize re-bind); reconstruct it to the full plain value here,
// because the fold reads the whole document and the residual body alone would drop the shred values.
Vector *GroupMergeReadInput(Vector &input, idx_t count, Vector &reconstructed) {
	if (IsShreddedJsonoType(input.GetType())) {
		JsonoReconstructToPlain(input, count, reconstructed);
		return &reconstructed;
	}
	return &input;
}

void JsonoGroupMergeSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 data_ptr_t state_ptr, idx_t count) {
	(void)aggr_input_data;
	(void)input_count;
	// Update inputs are plain (a shredded argument was reconstructed — and verified — by
	// GroupMergeReadInput), so any manifest entry is a narrowed row: the reader fails loud
	// before the fold rebuilds the accumulator without it.
	Vector reconstructed(JsonoType(), count);
	JsonoRowReader reader;
	reader.Init(*GroupMergeReadInput(inputs[0], count, reconstructed), count);
	auto &state = *reinterpret_cast<GroupMergeState *>(state_ptr);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		FoldIntoGroupState(state, view);
	}
}

void JsonoGroupMergeUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                           idx_t count) {
	(void)aggr_input_data;
	(void)input_count;
	// Same per-row policy as the simple update: plain inputs, any manifest entry fails loud.
	Vector reconstructed(JsonoType(), count);
	JsonoRowReader reader;
	reader.Init(*GroupMergeReadInput(inputs[0], count, reconstructed), count);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto state_idx = RowIndex(state_fmt, row);
		FoldIntoGroupState(*state_data[state_idx], view);
	}
}

void JsonoGroupMergeCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	(void)aggr_input_data;
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<GroupMergeState *>(source_fmt);
	auto target_data = FlatVector::GetData<GroupMergeState *>(target);

	for (idx_t row = 0; row < count; row++) {
		auto &source_state = *source_data[RowIndex(source_fmt, row)];
		if (!source_state.has_input || !source_state.acc) {
			continue;
		}
		JsonoView source_view = ViewOfBlob(*source_state.acc);
		if (!source_view.ParseHeader() || source_view.Slots() == 0) {
			continue;
		}
		FoldIntoGroupState(*target_data[row], source_view);
	}
}

// Copy a serialized accumulator blob's six streams into the body writer at row `rid`.
void WriteOwnedBlobRow(JsonoBodyWriter &writer, idx_t rid, const OwnedJsonoBlob &blob) {
	writer.data[BODY_SLOTS][rid] = WriteBlobInto(writer.Slots(), blob.slots.data(), blob.slots.size());
	writer.data[BODY_KEY_HEAP][rid] = WriteBlobInto(writer.KeyHeap(), blob.key_heap.data(), blob.key_heap.size());
	writer.data[BODY_STRING_HEAP][rid] =
	    WriteBlobInto(writer.StringHeap(), blob.string_heap.data(), blob.string_heap.size());
	writer.data[BODY_SKIPS][rid] = WriteBlobInto(writer.Skips(), blob.skips.data(), blob.skips.size());
	writer.data[BODY_LENGTHS][rid] = WriteBlobInto(writer.Lengths(), blob.lengths.data(), blob.lengths.size());
	writer.data[BODY_NUMS][rid] = WriteBlobInto(writer.Nums(), blob.nums.data(), blob.nums.size());
}

// Write each group's accumulated plain blob (or an empty object for an empty/NULL group) into a
// plain JSONO struct vector `out` at row base+i. Shared by the plain finalize (writes straight
// into the result at the chunk offset) and the shredded finalize (writes a temp at base 0).
void FinalizePlainGroups(Vector &out, UnifiedVectorFormat &state_fmt, GroupMergeState *const *state_data, idx_t count,
                         idx_t base) {
	// Writes at rows [base, base+count) and produces no SQL NULLs (an empty group becomes an
	// empty object). JsonoBodyWriter only flattens and resolves data pointers — it touches no
	// validity — so writing at a base offset is safe.
	JsonoBodyWriter writer;
	writer.Init(out);
	JsonoBuilder empty_builder;

	for (idx_t i = 0; i < count; i++) {
		auto rid = i + base;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.has_input || !state.acc) {
			empty_builder.Reset();
			empty_builder.EmitObjectStart(0);
			empty_builder.EmitObjectEnd();
			writer.WriteRow(rid, empty_builder);
			continue;
		}
		// The accumulator is already a serialized blob; copy each component verbatim.
		WriteOwnedBlobRow(writer, rid, *state.acc);
	}
}

void JsonoGroupMergeFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeBindData>();
	if (bind_data.shreds.empty()) {
		FinalizePlainGroups(result, state_fmt, state_data, count, offset);
		return;
	}
	// Shredded return: assemble the merged plain accumulators in a temp vector, then reshred into
	// the result's shredded type and copy into place at the chunk offset.
	Vector plain(JsonoType(), count);
	FinalizePlainGroups(plain, state_fmt, state_data, count, 0);
	Vector shredded(result.GetType(), count);
	JsonoShredFromSpec(plain, count, bind_data.shreds, shredded);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	VectorOperations::Copy(shredded, result, count, 0, offset);
}

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
	// Same sticky-shredded contract as jsono_group_merge: a shredded input reshreds back to its
	// own type in Finalize (empty for a plain input).
	vector<std::pair<string, LogicalType>> shreds;

	explicit GroupMergeLWWBindData(OrderModifiers modifiers) : modifiers(modifiers) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<GroupMergeLWWBindData>(modifiers);
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
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
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
};

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
	keys.push_back(std::move(copy));
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
	ShredPrimitive primitive;
	if (!TypeToShredPrimitive(type, primitive)) {
		throw InternalException("jsono_group_merge direct list lane: non-scalar element type '%s'", type.ToString());
	}
	switch (primitive) {
	case ShredPrimitive::Varchar:
		builder.EmitString(text);
		return;
	case ShredPrimitive::Bigint: {
		int64_t v;
		std::memcpy(&v, &value.num, sizeof(v));
		builder.EmitInt(v);
		return;
	}
	case ShredPrimitive::Ubigint:
		builder.EmitUInt(value.num);
		return;
	case ShredPrimitive::Double: {
		double v;
		std::memcpy(&v, &value.num, sizeof(v));
		builder.EmitDouble(v);
		return;
	}
	case ShredPrimitive::Boolean:
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
				node.children.insert(node.children.begin() + child_idx,
				                     LWWObjectChild(std::move(key_copy), std::move(child)));
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
		StoreLWWLeafFromIncoming(node, V, cursor, K, scratch, depth);
		return;
	}
	SerializeIncomingLWWLeaf(V, cursor, scratch.candidate, scratch, depth);
	if (CompareBlobValueTie(node.value, scratch.candidate) < 0) {
		node.value = scratch.candidate;
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
	ShredPrimitive kind;
	if (!TypeToShredPrimitive(type, kind)) {
		throw InternalException("jsono_group_merge direct shredded update: non-scalar shred type '%s'",
		                        type.ToString());
	}
	LWWScalarValue result;
	switch (kind) {
	case ShredPrimitive::Varchar:
		if (text.size() > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: string value exceeds storage limits");
		}
		result.slot = MakeSlot(tag::VAL_STR_HEAP, 0);
		result.length = uint32_t(text.size());
		return result;
	case ShredPrimitive::Bigint: {
		auto value = int64_t(value_bits);
		result.slot = FitsInt60(value) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64);
		result.num = value_bits;
		return result;
	}
	case ShredPrimitive::Ubigint:
		if (value_bits <= uint64_t(std::numeric_limits<int64_t>::max())) {
			auto signed_value = int64_t(value_bits);
			result.slot = FitsInt60(signed_value) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64);
		} else {
			result.slot = MakeExtSlot(ext_subtype::UINT64);
		}
		result.num = value_bits;
		return result;
	case ShredPrimitive::Double:
		result.slot = MakeExtSlot(ext_subtype::DOUBLE);
		result.num = value_bits;
		return result;
	case ShredPrimitive::Boolean:
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
		return;
	}
	if (!state.root) {
		throw InternalException("jsono_group_merge: missing keyed aggregate root");
	}
	MergeIncomingLWWNode(*state.root, V, cursor, K, scratch, 0, root_skip_keys);
}

vector<idx_t> LWWScalarTextLaneIndices(const vector<ReconShred> &shreds) {
	vector<idx_t> result;
	result.reserve(shreds.size());
	idx_t text_count = 0;
	for (auto &shred : shreds) {
		ShredPrimitive primitive;
		if (!TypeToShredPrimitive(shred.type, primitive)) {
			throw InternalException("jsono_group_merge direct lanes: non-scalar shred type '%s'",
			                        shred.type.ToString());
		}
		if (primitive == ShredPrimitive::Varchar) {
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
	if (text_count > 0) {
		state.scalar_texts = new string[text_count];
		state.scalar_text_lane_count = text_count;
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
}

void StorePrimitiveLaneValueBits(const LogicalType &type, const UnifiedVectorFormat &fmt, idx_t idx, uint64_t &out,
                                 string *text_out) {
	ShredPrimitive kind;
	if (!TypeToShredPrimitive(type, kind)) {
		throw InternalException("jsono_group_merge direct shredded update: non-scalar shred type '%s'",
		                        type.ToString());
	}
	out = 0;
	switch (kind) {
	case ShredPrimitive::Varchar: {
		auto value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		if (value.GetSize() > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: string value exceeds storage limits");
		}
		text_out->assign(value.GetData(), value.GetSize());
		return;
	}
	case ShredPrimitive::Bigint:
		out = uint64_t(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx]);
		return;
	case ShredPrimitive::Ubigint:
		out = UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx];
		return;
	case ShredPrimitive::Double: {
		auto value = UnifiedVectorFormat::GetData<double>(fmt)[idx];
		if (!std::isfinite(value)) {
			throw InvalidInputException("jsono: cannot store non-finite double value (NaN/Infinity)");
		}
		std::memcpy(&out, &value, sizeof(out));
		return;
	}
	case ShredPrimitive::Boolean:
		out = UnifiedVectorFormat::GetData<bool>(fmt)[idx] ? 1 : 0;
		return;
	}
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
		StoreLWWScalarLane(*lane, EnsureLWWScalarLaneText(state, text_indices, lane_idx), candidate_bits,
		                   candidate_text, ResolveLWWRowSortKey(state, K, sort_key_id));
		return;
	}
	auto *lane_text = RequireLWWScalarLaneText(state, text_indices, lane_idx);
	int key_cmp = CompareRawBytes(LWWLaneSortKey(state, lane->sort_key_id), K);
	if (key_cmp > 0) {
		return;
	}
	if (key_cmp < 0 ||
	    CompareLWWScalarLaneValueTie(*lane, LWWScalarTextView(lane_text), type, candidate_bits, candidate_text) < 0) {
		StoreLWWScalarLane(*lane, lane_text, candidate_bits, candidate_text,
		                   ResolveLWWRowSortKey(state, K, sort_key_id));
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
		StoreLWWListLane(lane, candidate, ResolveLWWRowSortKey(state, K, sort_key_id));
		return;
	}
	int key_cmp = CompareRawBytes(LWWLaneSortKey(state, lane.sort_key_id), K);
	if (key_cmp > 0) {
		return;
	}
	if (key_cmp < 0 || CompareLWWListValueTie(lane.value, candidate, type) < 0) {
		StoreLWWListLane(lane, candidate, ResolveLWWRowSortKey(state, K, sort_key_id));
	}
}

bool PrepareDirectLWWShreddedInput(const vector<std::pair<string, LogicalType>> &bind_shreds,
                                   vector<ReconShred> &scalar_shreds, vector<ReconShred> &list_shreds,
                                   vector<idx_t> &overlay_shreds) {
	for (idx_t i = 0; i < bind_shreds.size(); i++) {
		auto &name = bind_shreds[i].first;
		vector<PathStep> steps;
		if (name.size() >= 2 && name[0] == '$' && name[1] == '.') {
			steps = ParseJsonoPath(name, "jsono_group_merge direct shredded update");
		} else {
			steps.push_back(PathStep {PathStepKind::Key, name, 0});
		}
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
	ShredPrimitive primitive;
	if (!TypeToShredPrimitive(type, primitive)) {
		throw InternalException("jsono_group_merge direct finalize: non-scalar type '%s'", type.ToString());
	}
	FlatVector::Validity(out).SetValid(rid);
	switch (primitive) {
	case ShredPrimitive::Varchar:
		FlatVector::GetData<string_t>(out)[rid] = StringVector::AddString(out, text.data(), text.size());
		return;
	case ShredPrimitive::Bigint: {
		int64_t v;
		std::memcpy(&v, &value.num, sizeof(v));
		FlatVector::GetData<int64_t>(out)[rid] = v;
		return;
	}
	case ShredPrimitive::Ubigint:
		FlatVector::GetData<uint64_t>(out)[rid] = value.num;
		return;
	case ShredPrimitive::Double: {
		double v;
		std::memcpy(&v, &value.num, sizeof(v));
		FlatVector::GetData<double>(out)[rid] = v;
		return;
	}
	case ShredPrimitive::Boolean:
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
	ShredPrimitive primitive;
	if (!TypeToShredPrimitive(shred.type, primitive)) {
		throw InternalException("jsono_group_merge direct finalize: non-scalar shred type '%s'", shred.type.ToString());
	}
	if (primitive == ShredPrimitive::Varchar) {
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
	if (!JsonoScalarFitsShredType(scalar, shred.type)) {
		// A present scalar that does not fit the typed lane stays in the residual; an explicit JSON
		// null reads NULL either way, so a bare read of the NULL lane is correct.
		diverted = scalar.kind != JsonoScalarKind::Null;
		return false;
	}
	FlatVector::Validity(out).SetValid(rid);
	switch (primitive) {
	case ShredPrimitive::Bigint:
		FlatVector::GetData<int64_t>(out)[rid] = scalar.int_value;
		return true;
	case ShredPrimitive::Ubigint:
		FlatVector::GetData<uint64_t>(out)[rid] =
		    scalar.kind == JsonoScalarKind::UInt64 ? scalar.uint_value : uint64_t(scalar.int_value);
		return true;
	case ShredPrimitive::Double:
		FlatVector::GetData<double>(out)[rid] = scalar.double_value;
		return true;
	case ShredPrimitive::Boolean:
		FlatVector::GetData<bool>(out)[rid] = scalar.bool_value;
		return true;
	case ShredPrimitive::Varchar:
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
	vector<Vector *> complete_out(bind_data.shreds.size(), nullptr); // scalar shreds only
	for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
		if (!IsShredListType(bind_data.shreds[f].second)) {
			complete_out[f] = &JsonoShredCompleteVector(result, f);
			complete_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
		}
	}
	Vector *set_out = &JsonoShredSetVector(result);
	set_out->SetVectorType(VectorType::FLAT_VECTOR);

	// The shred-set marker value, order-independent over the shred set (see JsonoLayoutHashOf): the
	// optimizer recomputes it from the read type, which a set-op/reorder cast may present in a
	// different field order than this finalize built.
	auto manifest_layout_hash = JsonoLayoutHashOf(result.GetType());
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
		// Shred-set marker on every value row (group_merge never emits a SQL-NULL row — an empty
		// state emits `{}`); each scalar shred's completeness starts true and a divert clears it below.
		FlatVector::Validity(*set_out).SetValid(rid);
		FlatVector::GetData<int64_t>(*set_out)[rid] = int64_t(manifest_layout_hash);
		for (auto *complete : complete_out) {
			if (complete) {
				FlatVector::GetData<int8_t>(*complete)[rid] = 1;
			}
		}

		auto &state = *state_data[RowIndex(state_fmt, i)];
		builder.Reset();
		scalar_strip_paths.clear();
		stripped_child_indices.clear();
		stripped_shred_indices.clear();
		list_override_indices.clear();
		if (!state.has_input) {
			builder.EmitObjectStart(0);
			builder.EmitObjectEnd();
			writer.WriteRow(rid, builder);
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
				if (diverted && complete_out[shred.child]) {
					FlatVector::GetData<int8_t>(*complete_out[shred.child])[rid] = 0;
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
	auto bind_data = make_uniq<GroupMergeLWWBindData>(OrderModifiers(direction, OrderByNullType::NULLS_FIRST));
	if (IsShreddedJsonoType(type)) {
		// Sticky shredding, same as jsono_group_merge: keep the shredded argument native (Update
		// reconstructs it to plain) and capture the shreds so Finalize reshreds back to this type.
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
	if (JsonoGroupMergeLWWSimpleUpdateDirectShredded(inputs, bind_data, state_ptr, count, sk_fmt, sk_data)) {
		return;
	}
	auto &state = *reinterpret_cast<GroupMergeLWWState *>(state_ptr);
	JsonoGroupMergeLWWReconstructFold(inputs, count, sk_fmt, sk_data,
	                                  [&](idx_t) -> GroupMergeLWWState & { return state; });
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
	if (JsonoGroupMergeLWWUpdateDirectShredded(inputs, bind_data, count, sk_fmt, sk_data, state_fmt, state_data)) {
		return;
	}
	JsonoGroupMergeLWWReconstructFold(inputs, count, sk_fmt, sk_data, [&](idx_t row) -> GroupMergeLWWState & {
		return *state_data[RowIndex(state_fmt, row)];
	});
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
			builder.EmitObjectStart(0);
			builder.EmitObjectEnd();
			writer.WriteRow(rid, builder);
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

// Public entry (declared in jsono.hpp): forwards to the in-TU implementation, which reuses
// the merge overlay machinery. Lets the struct-constructor cast reconstruct a shredded value
// losslessly without duplicating the overlay logic.
void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result) {
	ReconstructShreddedToPlainImpl(input, count, result);
}

// Public entry (declared in jsono.hpp): overlay only `shreds` of the shredded `input` onto its
// residual, producing plain JSONO. The single-pass narrowing reshred uses it to fold the shreds
// the target type drops back into the residual without materializing the full document.
void JsonoOverlayShredsToPlain(Vector &input, idx_t count, const vector<idx_t> &shreds, Vector &result) {
	ReconstructShreddedToPlainImpl(input, count, result, &shreds);
}

// The residual-emit core (defined below, after the array-skeleton helpers it dispatches to):
// copy `view`'s object minus the leaves named by `scalar_paths`, and replace each terminal
// array-shred key with its skeleton array. An empty `array_specs` reduces it to the plain leaf
// strip — the shredded scalar writer's residual emit.
static void EmitSkeletonObject(const JsonoView &view, const JsonoCursor &obj_cursor, JsonoBuilder &builder,
                               const std::vector<const std::vector<PathStep> *> &scalar_paths,
                               const std::vector<const JsonoArrayShredSpec *> &array_specs, size_t depth);

// Emit `view` into `builder` with the located leaf of each path removed, used by the
// shredded writer to build the residual once it knows which paths were losslessly lifted
// into shreds. Paths are object-key sequences: a top-level path drops a root key, a nested
// path drops only its leaf and keeps the surrounding object. An empty path set or a
// non-object value is copied verbatim. Reuses the cursor-walk and EmitValueVerbatim so the
// residual's blocks/checkpoints are rebuilt correctly.
void JsonoEmitObjectStrippingPaths(const JsonoView &view, const std::vector<const std::vector<PathStep> *> &paths,
                                   JsonoBuilder &builder) {
	if (paths.empty() || view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		jsono::JsonoCursor cursor;
		EmitValueVerbatim(view, cursor, builder, 0);
		return;
	}
	EmitSkeletonObject(view, jsono::JsonoCursor(), builder, paths, {}, 0);
}

// ---- Residual skeleton emit for array shreds (write side) ----
// The array stays in the residual as a skeleton: each object element keeps its tail keys but loses
// the subfields lifted into the LIST<STRUCT> shred column. Length, element order, and non-object/
// null elements carry the lossless reconstruct (the parallel shred LIST overlays back). The strip
// gate is JsonoScalarFitsShredType — the same one WriteArrayShred uses — so write and skeleton agree.

// True if `name` is present in the element object at `element_cursor` as a scalar that fits `type`
// byte-for-byte (so its shred captured it and the skeleton drops it). Mirrors WriteShred's gate.
static bool ElementSubfieldStrippable(const JsonoView &view, const JsonoCursor &element_cursor, const string &name,
                                      const LogicalType &type) {
	JsonoCursor probe = element_cursor;
	if (!LocatePathStep(nullptr, 0, view, PathStep {PathStepKind::Key, name, 0}, probe)) {
		return false;
	}
	auto tag = SlotTag(view.SlotAt(probe.pos));
	if (tag == tag::OBJ_START || tag == tag::ARR_START) {
		return false;
	}
	auto scalar = DecodeScalarAt(view, probe);
	return JsonoScalarFitsShredType(scalar, type);
}

// Emit the skeleton array (cursor at ARR_START) for either array shred kind. `spec.kind` is fixed
// per array, so the lane dispatch hoists out of the element loop.
//
// ScalarArray lane: each element the lane lifts (a scalar fitting `spec.element_type`) becomes a
// VAL_NULL placeholder — it carries no value, the parallel LIST<element_type> shred does — every
// other element (a non-conforming scalar, a null, an object or array) stays verbatim. The lift gate
// is JsonoScalarFitsShredType, the same one WriteScalarArrayShred uses, so the placeholder positions
// match the shred's non-NULL slots exactly. On reconstruct the placeholder is never read: a non-NULL
// shred slot overrides it, a NULL slot means the element was kept (and a kept element is never a
// placeholder), so a placeholder and an explicit JSON null never collide.
//
// Array lane: each object element loses its strippable `spec.subfields` (lifted into the LIST<STRUCT>
// shred), every other element verbatim.
static void EmitSkeletonArray(const JsonoView &view, const JsonoCursor &array_cursor, JsonoBuilder &builder,
                              const JsonoArrayShredSpec &spec, size_t depth) {
	auto end_pos = ReadArrayEndPos(view, array_cursor.pos);
	builder.EmitArrayStart();
	JsonoCursor cursor = array_cursor;
	cursor.pos++;                                     // first element; ARR_START consumes no stream entries
	std::vector<std::vector<PathStep>> strip_storage; // Array lane: per-element strip paths
	std::vector<const std::vector<PathStep> *> strip;
	while (cursor.pos < end_pos) {
		if (spec.kind == ShredKind::ScalarArray) {
			auto value_tag = SlotTag(view.SlotAt(cursor.pos));
			if (value_tag != tag::OBJ_START && value_tag != tag::ARR_START) {
				JsonoCursor probe = cursor;
				auto scalar = DecodeScalarAt(view, probe);
				if (JsonoScalarFitsShredType(scalar, spec.element_type)) {
					builder.EmitNull(); // lifted: placeholder; the shred holds the value
					SkipValueFast(view, cursor);
					continue;
				}
			}
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		} else if (SlotTag(view.SlotAt(cursor.pos)) == tag::OBJ_START) {
			strip_storage.clear();
			for (auto &sub : spec.subfields) {
				if (ElementSubfieldStrippable(view, cursor, sub.first, sub.second)) {
					strip_storage.push_back({PathStep {PathStepKind::Key, sub.first, 0}});
				}
			}
			strip.clear();
			for (auto &path : strip_storage) {
				strip.push_back(&path);
			}
			EmitSkeletonObject(view, cursor, builder, strip, {}, 0);
			SkipValueFast(view, cursor);
		} else {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
	}
	builder.EmitArrayEnd();
}

// One object level of the residual emit (the strip core forward-declared above): strip the scalar
// leaves (`scalar_paths`) and replace the array at each terminal array-shred key with its skeleton,
// recursing for deeper paths. With an empty `array_specs` this is the plain leaf strip a top-level
// path drops a root key, a nested path drops only its leaf and keeps the surrounding object;
// surrounding keys are emitted verbatim. Every scalar path has length > depth by construction.
static void EmitSkeletonObject(const JsonoView &view, const JsonoCursor &obj_cursor, JsonoBuilder &builder,
                               const std::vector<const std::vector<PathStep> *> &scalar_paths,
                               const std::vector<const JsonoArrayShredSpec *> &array_specs, size_t depth) {
	auto layout = ReadObjectLayout(view, obj_cursor.pos);
	auto key_at = [&](size_t i) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		return view.KeyAt(SlotPayload(key_slot));
	};
	size_t surviving = 0;
	for (size_t i = 0; i < layout.key_count; i++) {
		if (!AnyPathTerminatesOnKey(scalar_paths, depth, key_at(i))) {
			surviving++;
		}
	}
	builder.EmitObjectStart(surviving);
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (!AnyPathTerminatesOnKey(scalar_paths, depth, key)) {
			builder.EmitKeySlot(key);
		}
	}
	JsonoCursor value_cursor = obj_cursor;
	value_cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (AnyPathTerminatesOnKey(scalar_paths, depth, key)) {
			SkipValueFast(view, value_cursor);
			continue;
		}
		builder.EmitObjectChildStart();
		std::vector<const std::vector<PathStep> *> deeper_scalar;
		for (auto *path : scalar_paths) {
			auto &step = (*path)[depth];
			if (depth + 1 < path->size() && nonstd::string_view(step.key.data(), step.key.size()) == key) {
				deeper_scalar.push_back(path);
			}
		}
		const JsonoArrayShredSpec *terminal_array = nullptr;
		std::vector<const JsonoArrayShredSpec *> deeper_array;
		for (auto *spec : array_specs) {
			if (PathTerminatesOnKey(spec->path, depth, key)) {
				terminal_array = spec;
			} else if (PathContinuesPastKey(spec->path, depth, key)) {
				deeper_array.push_back(spec);
			}
		}
		auto value_tag = SlotTag(view.SlotAt(value_cursor.pos));
		if (terminal_array && value_tag == tag::ARR_START) {
			EmitSkeletonArray(view, value_cursor, builder, *terminal_array, depth);
			SkipValueFast(view, value_cursor);
		} else if ((!deeper_scalar.empty() || !deeper_array.empty()) && value_tag == tag::OBJ_START) {
			EmitSkeletonObject(view, value_cursor, builder, deeper_scalar, deeper_array, depth + 1);
			SkipValueFast(view, value_cursor);
		} else {
			EmitValueVerbatim(view, value_cursor, builder, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

void JsonoEmitResidualSkeleton(const JsonoView &view, const std::vector<const std::vector<PathStep> *> &scalar_paths,
                               const std::vector<const JsonoArrayShredSpec *> &array_specs, JsonoBuilder &builder) {
	if ((scalar_paths.empty() && array_specs.empty()) || view.Slots() == 0 ||
	    SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		jsono::JsonoCursor cursor;
		EmitValueVerbatim(view, cursor, builder, 0);
		return;
	}
	EmitSkeletonObject(view, jsono::JsonoCursor(), builder, scalar_paths, array_specs, 0);
}

// The jsono_merge_patch function, exposed so the optimizer's shredded
// reconstruction can fold the shred patch into the residual without a catalog lookup.
ScalarFunction JsonoMergePatchFunction() {
	ScalarFunction fun("jsono_merge_patch", {}, JsonoType(), JsonoMergePatchExecute, JsonoMergePatchBind, nullptr,
	                   nullptr, JsonoOpsLocalState::Init);
	fun.varargs = LogicalType::ANY;
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

// jsono_overlay(base, patch...): the base is authoritative, each patch fills only the
// keys the base lacks (B-null is a no-op). This is the shredded-reconstruction primitive
// (residual wins, shreds refill stripped paths), not a headline user operation — the
// optimizer binds it directly via this factory. It is registered in the catalog (see
// RegisterJsonoMerge) only so the optimizer-injected reconstruction expression survives
// plan (de)serialization via a name lookup; it is intentionally left out of the docs.
ScalarFunction JsonoOverlayFunction() {
	ScalarFunction fun("jsono_overlay", {}, JsonoType(), JsonoOverlayExecute, JsonoMergePatchBind, nullptr, nullptr,
	                   JsonoOpsLocalState::Init);
	fun.varargs = LogicalType::ANY;
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

namespace {

// __jsono_internal_checked_residual(residual, shreds): pass the plain residual through after
// verifying its shred manifest against `shreds`, the shred signatures ("path\x01type") of the
// shredded type the optimizer read the residual out of. The optimizer wraps every residual
// reinterpret with this check, so a row narrowed by a raw struct cast (its manifest lists a
// shred the type no longer carries) fails loud on every optimizer read path — extract fallback,
// overlay reconstruction, introspection — instead of silently reading an incomplete residual.
void JsonoCheckedResidualExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	auto &shreds_value = args.data[1];
	if (shreds_value.GetVectorType() != VectorType::CONSTANT_VECTOR) {
		throw InvalidInputException("__jsono_internal_checked_residual: shreds must be constant");
	}
	vector<std::pair<std::string, std::string>> shred_signatures;
	auto shreds = ListValue::GetChildren(shreds_value.GetValue(0));
	shred_signatures.reserve(shreds.size());
	for (auto &shred : shreds) {
		auto signature = StringValue::Get(shred);
		auto sep = signature.find('\x01');
		if (sep == string::npos) {
			throw InvalidInputException("__jsono_internal_checked_residual: malformed shred signature");
		}
		shred_signatures.emplace_back(signature.substr(0, sep), signature.substr(sep + 1));
	}

	JsonoRowReader reader;
	reader.Init(args.data[0], count, std::move(shred_signatures));
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		reader.ReadPermissive(row, blob, view);
	}
	result.Reference(args.data[0]);
}

// __jsono_internal_strip_manifest(skips): drop the shred-manifest tail from a residual's skips blob,
// so a reader sees it as manifest-free. The soft residual reinterpret (the COALESCE fallback arm of a
// typed shred read) wraps its skips with this: a read of a shred-lane path absent from the residual
// then yields plain NULL — the lane holds the value — instead of the point-read manifest guard's
// narrowing throw. That throw is only correct when the read context lacks the lane (a genuinely
// narrowed row), which reads through the checked residual, never this fallback. Letting the manifest
// survive made any EAGER evaluation of the fallback (the projector fuse, any future hoist out of the
// COALESCE) throw on every row whose value lives in its lane.
void JsonoStripManifestExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t skips) {
		auto offset = JsonoSkipsManifestOffset(reinterpret_cast<const uint8_t *>(skips.GetData()), skips.GetSize());
		return StringVector::AddStringOrBlob(result, skips.GetData(), offset);
	});
}

} // namespace

ScalarFunction JsonoCheckedResidualFunction() {
	ScalarFunction fun("__jsono_internal_checked_residual", {JsonoType(), LogicalType::LIST(LogicalType::VARCHAR)},
	                   JsonoType(), JsonoCheckedResidualExecute);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

ScalarFunction JsonoStripManifestFunction() {
	ScalarFunction fun("__jsono_internal_strip_manifest", {LogicalType::BLOB}, LogicalType::BLOB,
	                   JsonoStripManifestExecute);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

void RegisterJsonoMerge(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{ loader.RegisterFunction(JsonoMergePatchFunction()); }
	// Internal reconstruction helpers, registered for serialization (see JsonoOverlayFunction).
	{ loader.RegisterFunction(JsonoOverlayFunction()); }
	{ loader.RegisterFunction(JsonoCheckedResidualFunction()); }
	{ loader.RegisterFunction(JsonoStripManifestFunction()); }
	{
		AggregateFunction fun(
		    "jsono_group_merge", {LogicalType::ANY}, jsono_type, AggregateFunction::StateSize<GroupMergeState>,
		    AggregateFunction::StateInitialize<GroupMergeState, GroupMergeFunction>, JsonoGroupMergeUpdate,
		    JsonoGroupMergeCombine, JsonoGroupMergeFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
		    JsonoGroupMergeSimpleUpdate, JsonoGroupMergeBind,
		    AggregateFunction::StateDestroy<GroupMergeState, GroupMergeFunction>);
		fun.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		loader.RegisterFunction(std::move(fun));
	}
	// Order-independent keyed variants: the order key is an argument, conflicts resolve per leaf,
	// so these are commutative/associative (no ORDER BY, no O(rows) buffering — memory O(groups)).
	{ loader.RegisterFunction(JsonoGroupMergeKeyedFunction("jsono_group_merge_max", JsonoGroupMergeMaxBind)); }
	{ loader.RegisterFunction(JsonoGroupMergeKeyedFunction("jsono_group_merge_min", JsonoGroupMergeMinBind)); }
}

} // namespace duckdb
