#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// Patch: RFC 7396 merge_patch (B wins, B-null deletes). IgnoreNulls: jsono_group_merge
// (B wins, null members dropped). Overlay: A (the first argument) is authoritative —
// B only fills keys absent from A, B-null is a no-op, A's own nulls are kept. Overlay
// powers shredded reconstruction: the residual (A) wins, lanes (B) fill stripped paths.
enum class MergeMode : uint8_t { Patch, IgnoreNulls, Overlay };

// One object child captured in a single pass: its key, the value's slot/string
// position, and the value's slot tag. Capturing the tag here lets the merge loop
// classify null/object children without re-reading slots.
struct MergeChild {
	nonstd::string_view key;
	size_t pos;
	size_t string_cursor;
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

// Copy a scalar value at (pos, string_cursor) byte-for-byte into the builder,
// advancing pos/string_cursor. Equivalent to decoding the scalar and re-emitting
// it, but without reconstructing the value: every scalar encoding round-trips to
// the identical slot(s) (INT60/DEC60 re-encode to themselves; ext INT64/UINT64/
// DOUBLE re-store the same raw bits; STR_HEAP/NUMBER re-append the same heap
// bytes), so copying the source slot(s) and heap slice is the same result with
// none of the decode/re-classify work. Caller must have excluded container slots.
void EmitScalarVerbatim(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder) {
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	switch (slot_tag) {
	case tag::VAL_STR_HEAP: {
		auto len = size_t(StringLen(SlotPayload(slot)));
		auto s = view.StringAt(string_cursor, len);
		builder.string_heap.insert(builder.string_heap.end(), s.begin(), s.end());
		builder.slots.push_back(slot);
		string_cursor += len;
		pos++;
		return;
	}
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		auto slot_count = ExtSlotCount(subtype);
		if (slot_count > view.Slots() - pos) {
			throw InvalidInputException("malformed JSONO: extended scalar payload out of bounds");
		}
		if (subtype == ext_subtype::NUMBER) {
			auto len = size_t(NumberExtLen(slot));
			auto s = view.StringAt(string_cursor, len);
			builder.string_heap.insert(builder.string_heap.end(), s.begin(), s.end());
			builder.slots.push_back(slot);
			string_cursor += len;
			pos++;
			return;
		}
		builder.slots.push_back(slot);
		builder.slots.push_back(view.SlotAt(pos + 1));
		pos += slot_count;
		return;
	}
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		builder.slots.push_back(slot);
		pos++;
		return;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

// Push a scalar value's slot(s) verbatim WITHOUT copying its string bytes. Used by
// the merge value phase to emit a run of scalars whose string bytes are copied in
// one bulk insert (a contiguous slice of the source string_heap): per-child it
// only touches `slots` (cheap), and the heap is filled once. Caller must have
// excluded containers.
void PushScalarValueSlots(const JsonoView &view, size_t pos, JsonoBuilder &builder) {
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	switch (slot_tag) {
	case tag::VAL_STR_HEAP:
		builder.slots.push_back(slot);
		return;
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		auto slot_count = ExtSlotCount(subtype);
		if (slot_count > view.Slots() - pos) {
			throw InvalidInputException("malformed JSONO: extended scalar payload out of bounds");
		}
		builder.slots.push_back(slot);
		if (subtype != ext_subtype::NUMBER) {
			builder.slots.push_back(view.SlotAt(pos + 1));
		}
		return;
	}
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

void EmitValueVerbatim(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		auto layout = ReadObjectLayout(view, pos);
		builder.EmitObjectStart(layout.key_count);
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
		size_t val_pos = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			builder.EmitObjectChildStart();
			EmitValueVerbatim(view, val_pos, string_cursor, builder, depth + 1);
		}
		builder.EmitObjectEnd();
		pos = layout.after_pos;
		return;
	}
	if (slot_tag == tag::ARR_START) {
		auto end_pos = ReadArrayEndPos(view, pos);
		builder.EmitArrayStart();
		pos++;
		while (pos < end_pos) {
			EmitValueVerbatim(view, pos, string_cursor, builder, depth + 1);
		}
		builder.EmitArrayEnd();
		pos = end_pos + 1;
		return;
	}
	EmitScalarVerbatim(view, pos, string_cursor, builder);
}

void EmitValueStrip(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder,
                    MergeMode merge_mode, size_t depth);

// True if the value at `pos` keeps any content under IGNORE NULLS stripping: a non-null
// scalar or array survives; an object survives iff some member survives (recursively).
// Used to omit keys whose object value strips to empty — jsono_group_merge drops
// `key:{}`, unlike jsono_merge_patch Patch mode which keeps it. Read-only navigation;
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
	size_t vp = layout.value_start;
	size_t sc = 0;
	for (size_t i = 0; i < layout.key_count; i++) {
		if (ValueSurvivesIgnoreNulls(view, vp)) {
			return true;
		}
		SkipValueFast(view, vp, sc);
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

void EmitObjectStrip(const JsonoView &view, size_t obj_pos, size_t obj_string_cursor, JsonoBuilder &builder,
                     MergeMode merge_mode, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout = ReadObjectLayout(view, obj_pos);
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
		size_t vp = layout.value_start;
		size_t sc = obj_string_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = MemberSurvivesStrip(view, vp, merge_mode);
			if (cached) {
				keep[i] = survives ? 1 : 0;
			}
			count += survives ? 1 : 0;
			SkipValueFast(view, vp, sc);
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
		size_t vp = layout.value_start;
		size_t sc = obj_string_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			if (MemberSurvivesStrip(view, vp, merge_mode)) {
				auto key_slot = view.SlotAt(layout.key_start + i);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
			}
			SkipValueFast(view, vp, sc);
		}
	}
	{
		size_t vp = layout.value_start;
		size_t sc = obj_string_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = cached ? (keep[i] != 0) : MemberSurvivesStrip(view, vp, merge_mode);
			if (!survives) {
				SkipValueFast(view, vp, sc);
				continue;
			}
			builder.EmitObjectChildStart();
			EmitValueStrip(view, vp, sc, builder, merge_mode, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

void EmitValueStrip(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder,
                    MergeMode merge_mode, size_t depth) {
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		EmitObjectStrip(view, pos, string_cursor, builder, merge_mode, depth);
		auto span = CheckedContainerSpan(view, pos, tag::OBJ_START);
		pos += size_t(span.slot_span);
		string_cursor += size_t(span.string_byte_span);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		EmitValueVerbatim(view, pos, string_cursor, builder, depth);
		return;
	}
	EmitScalarVerbatim(view, pos, string_cursor, builder);
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

// Captures every child and returns the string cursor just past the last child, so
// the merge value phase can read child r's string-heap span as the gap between
// consecutive children's cursors (out[r+1].string_cursor, or this end for the last
// child) without re-decoding the value.
size_t CollectObjectChildren(const JsonoView &view, const ObjectLayout &layout, size_t obj_string_cursor,
                             std::vector<MergeChild> &out) {
	out.clear();
	out.reserve(layout.key_count);
	size_t vp = layout.value_start;
	size_t sc = obj_string_cursor;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		auto value_slot = view.SlotAt(vp);
		out.push_back(MergeChild {key, vp, sc, SlotTag(value_slot)});
		SkipValueFastFromSlot(view, value_slot, vp, sc);
	}
	return sc;
}

void MergeTwoObjects(const JsonoView &va, size_t pos_a, size_t sc_a, const JsonoView &vb, size_t pos_b, size_t sc_b,
                     JsonoBuilder &builder, MergeMode merge_mode, size_t depth);

// Merge two object views into the builder (B patches A), sorted-key linear merge.
void MergeTwoObjectsWithScratch(const JsonoView &va, size_t pos_a, size_t sc_a, const JsonoView &vb, size_t pos_b,
                                size_t sc_b, JsonoBuilder &builder, MergeMode merge_mode, size_t depth,
                                std::vector<MergeChild> &children_a, std::vector<MergeChild> &children_b,
                                std::vector<MergePlanEntry> &plan) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout_a = ReadObjectLayout(va, pos_a);
	auto layout_b = ReadObjectLayout(vb, pos_b);
	size_t children_a_end = CollectObjectChildren(va, layout_a, sc_a, children_a);
	size_t children_b_end = CollectObjectChildren(vb, layout_b, sc_b, children_b);

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
			    (merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(vb, children_b[ib].pos))) {
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
				} else if (merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(vb, children_b[ib].pos)) {
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
	// bytes in a contiguous slice of that source's string_heap (null-stripping
	// removes keys/slots but never string bytes), so a run is copied with one bulk
	// insert instead of one insert per value. A run stops at a source change, a
	// container child (emitted per-leaf), or a checkpoint-stride boundary. The
	// stride stop keeps the builder's per-child EmitObjectChildStart snapshots
	// correct: within a run only its first index can be a stride multiple, and that
	// snapshot reads the sizes the already-emitted prior run left behind.
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
			MergeTwoObjects(va, children_a[plan[k].ra].pos, children_a[plan[k].ra].string_cursor, vb,
			                children_b[plan[k].rb].pos, children_b[plan[k].rb].string_cursor, builder, merge_mode,
			                depth + 1);
			k++;
			continue;
		}
		const JsonoView &vsrc = src == MergeSrc::A ? va : vb;
		auto &csrc = src == MergeSrc::A ? children_a : children_b;
		size_t csrc_end = src == MergeSrc::A ? children_a_end : children_b_end;
		size_t rank = src == MergeSrc::A ? plan[k].ra : plan[k].rb;
		auto value_tag = csrc[rank].tag;
		if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
			builder.EmitObjectChildStart();
			size_t p = csrc[rank].pos;
			size_t sc = csrc[rank].string_cursor;
			// A-side under Patch/Overlay is the verbatim target (A authoritative, keep
			// its nested nulls); the B-side under any mode strips null members so an
			// Overlay lane refill never re-introduces a null the residual dropped.
			if (src == MergeSrc::A && (merge_mode == MergeMode::Overlay || merge_mode == MergeMode::Patch)) {
				EmitValueVerbatim(vsrc, p, sc, builder, depth + 1);
			} else {
				EmitValueStrip(vsrc, p, sc, builder, merge_mode, depth + 1);
			}
			k++;
			continue;
		}
		// Extend the scalar run: same source, scalar value, before the next stride
		// boundary, and string-contiguous in the source heap. A child's string span
		// is the gap between its cursor and the next child's (csrc_end for the last),
		// so member `r` is contiguous with the run iff its cursor equals the position
		// just past the previous member. A base member stripped for being null leaves
		// no string bytes, so the slice spans it; a member stripped by a patch null
		// can carry bytes (string/number/object), which break the run there.
		size_t boundary = has_checkpoints ? (k / stride + 1) * stride : plan_size;
		size_t first_string_cursor = csrc[rank].string_cursor;
		size_t last_rank = rank;
		size_t run_end = k + 1;
		while (run_end < boundary && run_end < plan_size && plan[run_end].src == src) {
			size_t run_rank = src == MergeSrc::A ? plan[run_end].ra : plan[run_end].rb;
			auto run_tag = csrc[run_rank].tag;
			if (run_tag == tag::OBJ_START || run_tag == tag::ARR_START) {
				break;
			}
			size_t after_last = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].string_cursor : csrc_end;
			if (csrc[run_rank].string_cursor != after_last) {
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
			PushScalarValueSlots(vsrc, csrc[run_rank].pos, builder);
		}
		size_t run_string_end = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].string_cursor : csrc_end;
		if (run_string_end > first_string_cursor) {
			auto slice = vsrc.StringAt(first_string_cursor, run_string_end - first_string_cursor);
			builder.string_heap.insert(builder.string_heap.end(), slice.begin(), slice.end());
		}
		k = run_end;
	}
	builder.EmitObjectEnd();
}

void MergeTwoObjects(const JsonoView &va, size_t pos_a, size_t sc_a, const JsonoView &vb, size_t pos_b, size_t sc_b,
                     JsonoBuilder &builder, MergeMode merge_mode, size_t depth) {
	std::vector<MergeChild> children_a;
	std::vector<MergeChild> children_b;
	std::vector<MergePlanEntry> plan;
	MergeTwoObjectsWithScratch(va, pos_a, sc_a, vb, pos_b, sc_b, builder, merge_mode, depth, children_a, children_b,
	                           plan);
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
};

void SerializeBuilderToBlob(const JsonoBuilder &builder, OwnedJsonoBlob &out) {
	auto slots_bytes = builder.slots.size() * sizeof(uint64_t);
	out.slots.resize(JSONO_HEADER_SIZE + slots_bytes);
	JsonoHeader header;
	header.magic = MAGIC;
	header.version = VERSION;
	header.flags = flags::SORTED_KEYS;
	header.reserved = 0;
	std::memcpy(&out.slots[0], &header, JSONO_HEADER_SIZE);
	if (slots_bytes > 0) {
		std::memcpy(&out.slots[JSONO_HEADER_SIZE], builder.slots.data(), slots_bytes);
	}
	out.key_heap.assign(builder.key_heap.data(), builder.key_heap.size());
	out.string_heap.assign(builder.string_heap.data(), builder.string_heap.size());

	auto spans_bytes = builder.skips.size() * sizeof(ContainerSpan);
	auto index_bytes = builder.object_checkpoint_index.size() * sizeof(ObjectCheckpointIndex);
	auto checkpoints_bytes = builder.object_checkpoints.size() * sizeof(ObjectCursorCheckpoint);
	out.skips.resize(sizeof(ContainerMetadataHeader) + spans_bytes + index_bytes + checkpoints_bytes);
	ContainerMetadataHeader meta;
	meta.container_count = uint32_t(builder.skips.size());
	meta.checkpoint_index_count = uint32_t(builder.object_checkpoint_index.size());
	meta.checkpoint_count = uint32_t(builder.object_checkpoints.size());
	size_t off = 0;
	std::memcpy(&out.skips[off], &meta, sizeof(meta));
	off += sizeof(meta);
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
	                 b.string_heap.size(), b.skips.data(), b.skips.size());
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
	size_t p = 0;
	size_t sc = 0;
	bool acc_is_object = !acc_is_sqlnull && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START;
	bool patch_is_object = SlotTag(patch_view.SlotAt(0)) == tag::OBJ_START;
	if (acc_is_object && patch_is_object) {
		MergeTwoObjectsWithScratch(acc_view, 0, 0, patch_view, 0, 0, builder, mode, 0, children_a, children_b, plan);
		return false;
	}
	if (mode == MergeMode::Overlay) {
		// A authoritative: keep the accumulator if present, otherwise fall to the patch.
		const JsonoView &src = acc_is_sqlnull ? patch_view : acc_view;
		size_t sp = 0;
		size_t ssc = 0;
		EmitValueVerbatim(src, sp, ssc, builder, 0);
		return false;
	}
	if (!patch_is_object) {
		EmitValueVerbatim(patch_view, p, sc, builder, 0);
	} else {
		EmitValueStrip(patch_view, p, sc, builder, MergeMode::Patch, 0);
	}
	return false;
}

// A lane carried through a lane-aware merge: its name, the result struct child index it
// lands in, and its type. The merge folds the four-BLOB residuals (existing logic) and
// copies each union lane from the last input that declares it. Lane keys are assumed
// disjoint from residual keys (true when lanes are computed fields lifted into columns).
struct MergeLane {
	string name;
	idx_t result_child_index;
	LogicalType type;
};

struct JsonoMergeBindData : public FunctionData {
	vector<MergeLane> lanes;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoMergeBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoMergeBindData>();
		if (lanes.size() != other.lanes.size()) {
			return false;
		}
		for (idx_t i = 0; i < lanes.size(); i++) {
			if (lanes[i].name != other.lanes[i].name || lanes[i].type != other.lanes[i].type) {
				return false;
			}
		}
		return true;
	}
};

unique_ptr<FunctionData> JsonoMergePatchBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	(void)context;
	if (arguments.empty()) {
		throw BinderException("jsono_merge_patch() requires at least one argument");
	}
	// Union the lanes of any shredded inputs (later inputs win a name conflict, matching
	// the patch-wins fold). A shredded input keeps its type so the executor can read its
	// lane columns; a plain input is bound as JSONO and contributes only its residual.
	auto bind_data = make_uniq<JsonoMergeBindData>();
	auto &lanes = bind_data->lanes;
	auto prefix = StructType::GetChildTypes(JsonoRawStructType()).size();
	for (auto &argument : arguments) {
		if (argument->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		auto &type = argument->return_type;
		if (type.id() == LogicalTypeId::SQLNULL) {
			bound_function.arguments.push_back(JsonoType());
			continue;
		}
		if (IsShreddedJsonoType(type)) {
			auto &children = StructType::GetChildTypes(type);
			for (idx_t i = prefix; i < children.size(); i++) {
				bool found = false;
				for (auto &lane : lanes) {
					if (lane.name == children[i].first) {
						lane.type = children[i].second;
						found = true;
						break;
					}
				}
				if (!found) {
					lanes.push_back(MergeLane {children[i].first, 0, children[i].second});
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
	if (lanes.empty()) {
		bound_function.return_type = JsonoType();
		return std::move(bind_data);
	}
	child_list_t<LogicalType> children = StructType::GetChildTypes(JsonoRawStructType());
	idx_t next = children.size();
	for (auto &lane : lanes) {
		lane.result_child_index = next++;
		children.emplace_back(lane.name, lane.type);
	}
	bound_function.return_type = LogicalType::STRUCT(std::move(children));
	return std::move(bind_data);
}

struct ReconLane {
	idx_t child;
	LogicalType type;
	vector<PathStep> steps; // object-key path; size 1 is a top-level key
};

void EmitReconLaneScalar(JsonoBuilder &builder, const ReconLane &lane, const UnifiedVectorFormat &fmt, idx_t idx) {
	switch (lane.type.id()) {
	case LogicalTypeId::VARCHAR: {
		auto value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
		return;
	}
	case LogicalTypeId::BIGINT:
		builder.EmitInt(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx]);
		return;
	case LogicalTypeId::UBIGINT:
		builder.EmitUInt(UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx]);
		return;
	case LogicalTypeId::DOUBLE:
		builder.EmitDouble(UnifiedVectorFormat::GetData<double>(fmt)[idx]);
		return;
	case LogicalTypeId::BOOLEAN:
		builder.EmitBool(UnifiedVectorFormat::GetData<bool>(fmt)[idx]);
		return;
	default:
		throw InternalException("jsono reconstruct: unexpected lane type '%s'", lane.type.ToString());
	}
}

// Reconstruct a shredded JSONO `input` (four-BLOB residual + lane columns) into the plain
// JSONO `result` by overlaying each row's lane values back onto its residual object. Lanes
// are disjoint from residual keys (the shred invariant), so the overlay re-inserts them.
// Top-level lanes are emitted as one flat patch (the common jsono({...}) case); nested-path
// lanes (`$.a.b`) accumulate a one-path chain overlay each so any depth round-trips.
void ReconstructShreddedToPlainImpl(Vector &input, idx_t count, Vector &result) {
	auto &input_types = StructType::GetChildTypes(input.GetType());
	auto prefix = StructType::GetChildTypes(JsonoRawStructType()).size();
	vector<ReconLane> lanes;
	bool all_top_level = true;
	for (idx_t i = prefix; i < input_types.size(); i++) {
		auto &name = input_types[i].first;
		vector<PathStep> steps;
		if (name.size() >= 2 && name[0] == '$' && name[1] == '.') {
			steps = ParseJsonoPath(name, "jsono reconstruct");
		} else {
			steps.push_back(PathStep {PathStepKind::Key, name, 0});
		}
		for (auto &step : steps) {
			if (step.kind != PathStepKind::Key) {
				throw InvalidInputException("jsono reconstruct: lane path '%s' is not an object-key path", name);
			}
		}
		if (steps.size() != 1) {
			all_top_level = false;
		}
		lanes.push_back(ReconLane {i, input_types[i].second, std::move(steps)});
	}
	std::sort(lanes.begin(), lanes.end(), [](const ReconLane &a, const ReconLane &b) {
		auto n = std::min(a.steps.size(), b.steps.size());
		for (size_t i = 0; i < n; i++) {
			if (a.steps[i].key != b.steps[i].key) {
				return a.steps[i].key < b.steps[i].key;
			}
		}
		return a.steps.size() < b.steps.size();
	});

	JsonoVectorData residual;
	InitJsonoVectorData(input, count, residual);
	vector<UnifiedVectorFormat> lane_fmt(lanes.size());
	auto &struct_children = StructVector::GetEntries(input);
	for (idx_t k = 0; k < lanes.size(); k++) {
		struct_children[lanes[k].child]->ToUnifiedFormat(count, lane_fmt[k]);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &out_children = StructVector::GetEntries(result);
	auto &slots_vec = *out_children[0];
	auto &key_heap_vec = *out_children[1];
	auto &string_heap_vec = *out_children[2];
	auto &skips_vec = *out_children[3];
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);

	JsonoBuilder patch_builder;
	JsonoBuilder out_builder;
	OwnedJsonoBlob patch_storage;
	OwnedJsonoBlob acc_storage;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob {};
		if (!ReadJsonoRow(residual, row, blob)) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			continue;
		}
		JsonoView residual_view = MakeJsonoView(blob);
		if (!residual_view.ParseHeader() || residual_view.Slots() == 0) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			continue;
		}
		if (all_top_level) {
			// One flat patch object of the present lanes, overlaid in a single pass.
			patch_builder.Reset();
			idx_t present = 0;
			for (idx_t k = 0; k < lanes.size(); k++) {
				present += RowIsValid(lane_fmt[k], row) ? 1 : 0;
			}
			patch_builder.EmitObjectStart(present);
			for (idx_t k = 0; k < lanes.size(); k++) {
				if (RowIsValid(lane_fmt[k], row)) {
					patch_builder.EmitKeySlot(lanes[k].steps[0].key);
				}
			}
			for (idx_t k = 0; k < lanes.size(); k++) {
				if (!RowIsValid(lane_fmt[k], row)) {
					continue;
				}
				patch_builder.EmitObjectChildStart();
				EmitReconLaneScalar(patch_builder, lanes[k], lane_fmt[k], RowIndex(lane_fmt[k], row));
			}
			patch_builder.EmitObjectEnd();
			SerializeBuilderToBlob(patch_builder, patch_storage);
			JsonoView patch_view = ViewOfBlob(patch_storage);
			patch_view.ParseHeader();
			out_builder.Reset();
			MergeTwoObjects(residual_view, 0, 0, patch_view, 0, 0, out_builder, MergeMode::Overlay, 0);
			WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
			                 string_heap_out, skips_out, row, out_builder);
			continue;
		}
		// Nested lanes: overlay one single-path chain at a time onto the accumulator. The
		// residual is authoritative; each disjoint lane path fills in where it was stripped.
		const JsonoView *acc = &residual_view;
		JsonoView acc_view = residual_view;
		bool any = false;
		for (idx_t k = 0; k < lanes.size(); k++) {
			if (!RowIsValid(lane_fmt[k], row)) {
				continue;
			}
			patch_builder.Reset();
			for (auto &step : lanes[k].steps) {
				patch_builder.EmitObjectStart(1);
				patch_builder.EmitKeySlot(step.key);
				patch_builder.EmitObjectChildStart();
			}
			EmitReconLaneScalar(patch_builder, lanes[k], lane_fmt[k], RowIndex(lane_fmt[k], row));
			for (idx_t s = 0; s < lanes[k].steps.size(); s++) {
				patch_builder.EmitObjectEnd();
			}
			SerializeBuilderToBlob(patch_builder, patch_storage);
			JsonoView patch_view = ViewOfBlob(patch_storage);
			patch_view.ParseHeader();
			out_builder.Reset();
			MergeTwoObjects(*acc, 0, 0, patch_view, 0, 0, out_builder, MergeMode::Overlay, 0);
			SerializeBuilderToBlob(out_builder, acc_storage);
			acc_view = ViewOfBlob(acc_storage);
			acc_view.ParseHeader();
			acc = &acc_view;
			any = true;
		}
		out_builder.Reset();
		if (any) {
			size_t p = 0;
			size_t sc = 0;
			EmitValueVerbatim(*acc, p, sc, out_builder, 0);
		} else {
			size_t p = 0;
			size_t sc = 0;
			EmitValueVerbatim(residual_view, p, sc, out_builder, 0);
		}
		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out, string_heap_out,
		                 skips_out, row, out_builder);
	}
}

// Fold the residual views in `inputs` left-to-right under `mode` and write the merged plain
// JSONO into `out` (the four-BLOB residual). Shared by the lane-aware fast path (raw residuals)
// and the reshred fallback (reconstructed values).
void RunResidualFold(MergeMode mode, vector<JsonoVectorData> &inputs, idx_t ncols, idx_t count, Vector &out,
                     JsonoOpsLocalState &lstate) {
	out.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(out);
	auto &slots_vec = *children[0];
	auto &key_heap_vec = *children[1];
	auto &string_heap_vec = *children[2];
	auto &skips_vec = *children[3];
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);
	// jsono_merge_patch folds left to right (RFC 7396): the first argument is the
	// target document (kept verbatim — its nulls are values), the rest are patches.
	static thread_local OwnedJsonoBlob acc_storage;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow target_blob {};
		bool acc_is_sqlnull = !ReadJsonoRow(inputs[0], row, target_blob);
		JsonoView acc_view = MakeJsonoView(target_blob);
		if (!acc_is_sqlnull && (!acc_view.ParseHeader() || acc_view.Slots() == 0)) {
			acc_is_sqlnull = true;
		}
		if (ncols == 1) {
			// A lone target is returned verbatim (its nulls survive).
			if (acc_is_sqlnull) {
				SetJsonoStructNull(out, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			} else {
				lstate.builder.Reset();
				size_t p = 0;
				size_t sc = 0;
				EmitValueVerbatim(acc_view, p, sc, lstate.builder, 0);
				WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
				                 string_heap_out, skips_out, row, lstate.builder);
			}
			continue;
		}
		for (idx_t i = 1; i < ncols; i++) {
			JsonoBlobRow patch_blob {};
			bool patch_present = ReadJsonoRow(inputs[i], row, patch_blob);
			JsonoView patch_view = MakeJsonoView(patch_blob);
			if (patch_present && (!patch_view.ParseHeader() || patch_view.Slots() == 0)) {
				patch_present = false;
			}
			bool result_sqlnull =
			    MergeFoldStep(mode, acc_is_sqlnull, acc_view, patch_present, patch_view, lstate.builder,
			                  lstate.merge_children_a, lstate.merge_children_b, lstate.merge_plan);
			if (i + 1 == ncols) {
				// Last step writes straight to the output builder — no serialize round-trip.
				if (result_sqlnull) {
					SetJsonoStructNull(out, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
				} else {
					WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
					                 string_heap_out, skips_out, row, lstate.builder);
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

// Copy a lane column from the winning input into the result lane, then null it where the
// merged value is SQL NULL. Patch takes the last input that declares the lane (RFC 7396
// last-wins); Overlay takes the first (base-authoritative).
void FastCopyLane(DataChunk &args, idx_t ncols, idx_t count, MergeMode mode, const MergeLane &lane, Vector &dst,
                  const ValidityMask &result_validity) {
	idx_t src_arg = DConstants::INVALID_INDEX;
	idx_t src_child = 0;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (t.id() != LogicalTypeId::STRUCT) {
			continue;
		}
		auto &ch = StructType::GetChildTypes(t);
		for (idx_t c = 0; c < ch.size(); c++) {
			if (ch[c].first == lane.name && (mode != MergeMode::Overlay || src_arg == DConstants::INVALID_INDEX)) {
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
	VectorOperations::Copy(*StructVector::GetEntries(args.data[src_arg])[src_child], dst, count, 0, 0);
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
	bool has_lanes = !bind_data.lanes.empty();

	if (!has_lanes) {
		// Plain merge: fold straight into the result.
		vector<JsonoVectorData> inputs(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			InitJsonoVectorData(args.data[i], count, inputs[i]);
		}
		RunResidualFold(mode, inputs, ncols, count, result, lstate);
		if (args.AllConstant()) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Fast path: when every input is shredded (no plain residual can shadow a lane key) and
	// every lane is a top-level key, fold the raw residuals and copy lanes directly — skipping
	// the reconstruct+reshred — as long as no lane key collides into the merged residual.
	bool fast_viable = true;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (t.id() != LogicalTypeId::SQLNULL && !IsShreddedJsonoType(t)) {
			fast_viable = false;
		}
	}
	for (auto &lane : bind_data.lanes) {
		if (!lane.name.empty() && lane.name[0] == '$') {
			fast_viable = false;
		}
	}

	if (fast_viable) {
		vector<JsonoVectorData> raw(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			InitJsonoVectorData(args.data[i], count, raw[i]);
		}
		Vector fast_residual(JsonoType(), count);
		RunResidualFold(mode, raw, ncols, count, fast_residual, lstate);
		JsonoVectorData fr;
		InitJsonoVectorData(fast_residual, count, fr);
		bool conflict = false;
		for (idx_t row = 0; row < count && !conflict; row++) {
			JsonoBlobRow blob {};
			if (!ReadJsonoRow(fr, row, blob)) {
				continue;
			}
			JsonoView v = MakeJsonoView(blob);
			if (!v.ParseHeader()) {
				continue;
			}
			for (auto &lane : bind_data.lanes) {
				if (ResidualHasTopLevelKey(v, lane.name)) {
					conflict = true;
					break;
				}
			}
		}
		if (!conflict) {
			// Capture constness before FastCopyLane flattens any input (which would otherwise
			// make AllConstant() spuriously false and trip the constant-fold assertion).
			bool all_constant = args.AllConstant();
			result.SetVectorType(VectorType::FLAT_VECTOR);
			fast_residual.Flatten(count);
			auto &rc = StructVector::GetEntries(result);
			auto &frc = StructVector::GetEntries(fast_residual);
			FlatVector::Validity(result) = FlatVector::Validity(fast_residual);
			for (idx_t b = 0; b < 4; b++) {
				rc[b]->SetVectorType(VectorType::FLAT_VECTOR);
				VectorOperations::Copy(*frc[b], *rc[b], count, 0, 0);
			}
			auto &result_validity = FlatVector::Validity(result);
			for (auto &lane : bind_data.lanes) {
				FastCopyLane(args, ncols, count, mode, lane, *rc[lane.result_child_index], result_validity);
			}
			if (all_constant) {
				result.SetVectorType(VectorType::CONSTANT_VECTOR);
			}
			return;
		}
		// A lane key landed in the residual (conflict) — fall through to the correct reshred path.
	}

	// Reshred fallback: reconstruct shredded inputs to plain, fold, reshred. Correct for any
	// lane/residual key overlap and for plain inputs that can shadow a lane.
	vector<unique_ptr<Vector>> reconstructed;
	vector<JsonoVectorData> inputs(ncols);
	for (idx_t i = 0; i < ncols; i++) {
		if (IsShreddedJsonoType(args.data[i].GetType())) {
			auto plain = make_uniq<Vector>(JsonoType(), count);
			ReconstructShreddedToPlainImpl(args.data[i], count, *plain);
			InitJsonoVectorData(*plain, count, inputs[i]);
			reconstructed.push_back(std::move(plain));
		} else {
			InitJsonoVectorData(args.data[i], count, inputs[i]);
		}
	}
	Vector fold_out(JsonoType(), count);
	RunResidualFold(mode, inputs, ncols, count, fold_out, lstate);
	vector<std::pair<string, LogicalType>> lane_specs;
	lane_specs.reserve(bind_data.lanes.size());
	for (auto &lane : bind_data.lanes) {
		lane_specs.emplace_back(lane.name, lane.type);
	}
	JsonoShredFromSpec(fold_out, count, lane_specs, result);
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

	explicit GroupMergeBindData(MergeMode merge_mode) : merge_mode(merge_mode) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<GroupMergeBindData>(merge_mode);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<GroupMergeBindData>();
		return merge_mode == other.merge_mode;
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

string NormalizeGroupMergeMode(string mode) {
	StringUtil::Trim(mode);
	return StringUtil::Lower(mode);
}

unique_ptr<FunctionData> JsonoGroupMergeBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	(void)function;
	if (arguments.size() != 2) {
		throw BinderException("jsono_group_merge() requires JSONO plus constant 'IGNORE NULLS'");
	}
	if (arguments[0]->return_type.id() != LogicalTypeId::SQLNULL && !IsJsonoType(arguments[0]->return_type)) {
		throw BinderException("jsono_group_merge() input must be JSONO");
	}
	auto &mode_arg = arguments[1];
	if (mode_arg->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!mode_arg->IsFoldable()) {
		throw BinderException("jsono_group_merge() null handling mode must be constant");
	}
	auto mode_value = ExpressionExecutor::EvaluateScalar(context, *mode_arg);
	if (mode_value.IsNull() || mode_value.type().id() != LogicalTypeId::VARCHAR ||
	    NormalizeGroupMergeMode(StringValue::Get(mode_value)) != "ignore nulls") {
		throw InvalidInputException("jsono_group_merge(): null handling mode must be IGNORE NULLS");
	}
	return make_uniq<GroupMergeBindData>(MergeMode::IgnoreNulls);
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
	size_t pos = 0;
	size_t sc = 0;
	bool incoming_is_object = SlotTag(incoming.SlotAt(0)) == tag::OBJ_START;
	bool merged_objects = false;
	if (incoming_is_object && state.acc) {
		JsonoView acc_view = ViewOfBlob(*state.acc);
		if (acc_view.ParseHeader() && acc_view.Slots() > 0 && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START) {
			MergeTwoObjects(acc_view, 0, 0, incoming, 0, 0, scratch, MergeMode::IgnoreNulls, 0);
			merged_objects = true;
		}
	}
	if (!merged_objects) {
		if (incoming_is_object) {
			EmitValueStrip(incoming, pos, sc, scratch, MergeMode::IgnoreNulls, 0);
		} else {
			EmitValueVerbatim(incoming, pos, sc, scratch, 0);
		}
	}
	if (!state.acc) {
		state.acc = new OwnedJsonoBlob();
	}
	SerializeBuilderToBlob(scratch, *state.acc);
	state.has_input = true;
}

void ApplyGroupMergeFromBlob(GroupMergeState &state, const JsonoBlobRow &blob) {
	JsonoView view = MakeJsonoView(blob);
	if (!view.ParseHeader() || view.Slots() == 0) {
		return;
	}
	FoldIntoGroupState(state, view);
}

void JsonoGroupMergeSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 data_ptr_t state_ptr, idx_t count) {
	(void)aggr_input_data;
	(void)input_count;
	JsonoVectorData input;
	InitJsonoVectorData(inputs[0], count, input);
	auto &state = *reinterpret_cast<GroupMergeState *>(state_ptr);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			continue;
		}
		ApplyGroupMergeFromBlob(state, blob);
	}
}

void JsonoGroupMergeUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                           idx_t count) {
	(void)aggr_input_data;
	(void)input_count;
	JsonoVectorData input;
	InitJsonoVectorData(inputs[0], count, input);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			continue;
		}
		auto state_idx = RowIndex(state_fmt, row);
		ApplyGroupMergeFromBlob(*state_data[state_idx], blob);
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

void JsonoGroupMergeFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	(void)aggr_input_data;
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	auto &slots_vec = *children[0];
	auto &key_heap_vec = *children[1];
	auto &string_heap_vec = *children[2];
	auto &skips_vec = *children[3];
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);
	JsonoBuilder empty_builder;

	for (idx_t i = 0; i < count; i++) {
		auto rid = i + offset;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.has_input || !state.acc) {
			empty_builder.Reset();
			empty_builder.EmitObjectStart(0);
			empty_builder.EmitObjectEnd();
			WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
			                 string_heap_out, skips_out, rid, empty_builder);
			continue;
		}
		// The accumulator is already a serialized blob; copy each component verbatim.
		slots_out[rid] = WriteBlobInto(slots_vec, state.acc->slots.data(), state.acc->slots.size());
		key_heap_out[rid] = WriteBlobInto(key_heap_vec, state.acc->key_heap.data(), state.acc->key_heap.size());
		string_heap_out[rid] =
		    WriteBlobInto(string_heap_vec, state.acc->string_heap.data(), state.acc->string_heap.size());
		skips_out[rid] = WriteBlobInto(skips_vec, state.acc->skips.data(), state.acc->skips.size());
	}
}

} // namespace

// Public entry (declared in jsono.hpp): forwards to the in-TU implementation, which reuses
// the merge overlay machinery. Lets the struct-constructor cast reconstruct a shredded value
// losslessly without duplicating the overlay logic.
void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result) {
	ReconstructShreddedToPlainImpl(input, count, result);
}

// True if some active path ends on `key` at this depth — its leaf is the value to strip.
static bool PathTerminatesOnKey(const std::vector<const std::vector<PathStep> *> &active, size_t depth,
                                nonstd::string_view key) {
	for (auto *path : active) {
		auto &step = (*path)[depth];
		if (depth + 1 == path->size() && nonstd::string_view(step.key.data(), step.key.size()) == key) {
			return true;
		}
	}
	return false;
}

// One object level of the residual emit: copy `view`'s object at `obj_pos` minus the
// leaves named by `active`. Each active path is a sequence of object keys, matched against
// this object's keys at `depth`: a path that terminates here strips that key's value, a
// longer path recurses into the child object so only its leaf is removed. Surrounding
// keys are emitted verbatim. Every active path has length > depth by construction.
static void EmitObjectStrippingPaths(const JsonoView &view, size_t obj_pos, size_t obj_string_cursor,
                                     JsonoBuilder &builder, const std::vector<const std::vector<PathStep> *> &active,
                                     size_t depth) {
	auto layout = ReadObjectLayout(view, obj_pos);
	auto key_at = [&](size_t i) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		return view.KeyAt(SlotPayload(key_slot));
	};
	size_t surviving = 0;
	for (size_t i = 0; i < layout.key_count; i++) {
		if (!PathTerminatesOnKey(active, depth, key_at(i))) {
			surviving++;
		}
	}
	builder.EmitObjectStart(surviving);
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (!PathTerminatesOnKey(active, depth, key)) {
			builder.EmitKeySlot(key);
		}
	}
	size_t value_pos = layout.value_start;
	size_t value_string_cursor = obj_string_cursor;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (PathTerminatesOnKey(active, depth, key)) {
			SkipValueFast(view, value_pos, value_string_cursor);
			continue;
		}
		builder.EmitObjectChildStart();
		// Paths that continue past this key descend into the child. A path can only have
		// located through an object, so recurse only into an object child; otherwise the
		// value is verbatim.
		std::vector<const std::vector<PathStep> *> deeper;
		for (auto *path : active) {
			auto &step = (*path)[depth];
			if (depth + 1 < path->size() && nonstd::string_view(step.key.data(), step.key.size()) == key) {
				deeper.push_back(path);
			}
		}
		if (!deeper.empty() && SlotTag(view.SlotAt(value_pos)) == tag::OBJ_START) {
			EmitObjectStrippingPaths(view, value_pos, value_string_cursor, builder, deeper, depth + 1);
			auto span = CheckedContainerSpan(view, value_pos, tag::OBJ_START);
			value_pos += size_t(span.slot_span);
			value_string_cursor += size_t(span.string_byte_span);
		} else {
			EmitValueVerbatim(view, value_pos, value_string_cursor, builder, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

// Emit `view` into `builder` with the located leaf of each path removed, used by the
// shredded writer to build the residual once it knows which paths were losslessly lifted
// into lanes. Paths are object-key sequences: a top-level path drops a root key, a nested
// path drops only its leaf and keeps the surrounding object. An empty path set or a
// non-object value is copied verbatim. Reuses the cursor-walk and EmitValueVerbatim so the
// residual's blocks/checkpoints are rebuilt correctly.
void JsonoEmitObjectStrippingPaths(const JsonoView &view, const std::vector<const std::vector<PathStep> *> &paths,
                                   JsonoBuilder &builder) {
	if (paths.empty() || view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		size_t pos = 0;
		size_t string_cursor = 0;
		EmitValueVerbatim(view, pos, string_cursor, builder, 0);
		return;
	}
	EmitObjectStrippingPaths(view, 0, 0, builder, paths, 0);
}

// The jsono_merge_patch function, exposed so the optimizer's shredded
// reconstruction can fold the lane patch into the residual without a catalog lookup.
ScalarFunction JsonoMergePatchFunction() {
	ScalarFunction fun("jsono_merge_patch", {}, JsonoType(), JsonoMergePatchExecute, JsonoMergePatchBind, nullptr,
	                   nullptr, JsonoOpsLocalState::Init);
	fun.varargs = LogicalType::ANY;
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

// jsono_overlay(base, patch...): the base is authoritative, each patch fills only the
// keys the base lacks (B-null is a no-op). Powers shredded reconstruction; exposed so
// the optimizer can fold the lanes back onto the residual without a catalog lookup.
ScalarFunction JsonoOverlayFunction() {
	ScalarFunction fun("jsono_overlay", {}, JsonoType(), JsonoOverlayExecute, JsonoMergePatchBind, nullptr, nullptr,
	                   JsonoOpsLocalState::Init);
	fun.varargs = LogicalType::ANY;
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

void RegisterJsonoMerge(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{ loader.RegisterFunction(JsonoMergePatchFunction()); }
	{ loader.RegisterFunction(JsonoOverlayFunction()); }
	{
		AggregateFunction fun("jsono_group_merge", {jsono_type, LogicalType::VARCHAR}, jsono_type,
		                      AggregateFunction::StateSize<GroupMergeState>,
		                      AggregateFunction::StateInitialize<GroupMergeState, GroupMergeFunction>,
		                      JsonoGroupMergeUpdate, JsonoGroupMergeCombine, JsonoGroupMergeFinalize,
		                      FunctionNullHandling::DEFAULT_NULL_HANDLING, JsonoGroupMergeSimpleUpdate,
		                      JsonoGroupMergeBind,
		                      AggregateFunction::StateDestroy<GroupMergeState, GroupMergeFunction>);
		fun.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		loader.RegisterFunction(std::move(fun));
	}
}

} // namespace duckdb
