#include "jsono_ops.hpp"
#include "jsono.hpp"
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

// Copy a scalar value at `cursor` byte-for-byte into the builder, advancing the
// cursor. Equivalent to decoding the scalar and re-emitting it, but without
// reconstructing the value: every scalar encoding round-trips to the identical
// slot + stream entries (INT60/DEC60 re-store the same nums word; ext INT64/
// UINT64/DOUBLE re-store the same raw bits; STR_HEAP/NUMBER re-append the same
// heap bytes and length entry), so copying the source slot and stream entries is
// the same result with none of the decode/re-classify work. Caller must have
// excluded container slots.
void EmitScalarVerbatim(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder) {
	auto slot = view.SlotAt(cursor.pos);
	auto slot_tag = SlotTag(slot);
	switch (slot_tag) {
	case tag::VAL_STR_HEAP: {
		auto len = view.LengthAt(cursor.length_cursor);
		auto s = view.StringAt(cursor.string_cursor, len);
		builder.string_heap.insert(builder.string_heap.end(), s.begin(), s.end());
		builder.lengths.push_back(len);
		builder.slots.push_back(slot);
		cursor.string_cursor += len;
		cursor.length_cursor++;
		cursor.pos++;
		return;
	}
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		if (subtype == ext_subtype::NUMBER) {
			auto len = view.LengthAt(cursor.length_cursor);
			auto s = view.StringAt(cursor.string_cursor, len);
			builder.string_heap.insert(builder.string_heap.end(), s.begin(), s.end());
			builder.lengths.push_back(len);
			builder.slots.push_back(slot);
			cursor.string_cursor += len;
			cursor.length_cursor++;
			cursor.pos++;
			return;
		}
		if (subtype >= ext_subtype::COUNT) {
			throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
		}
		builder.slots.push_back(slot);
		builder.nums.push_back(view.NumAt(cursor.num_cursor));
		cursor.num_cursor++;
		cursor.pos++;
		return;
	}
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
		builder.slots.push_back(slot);
		builder.nums.push_back(view.NumAt(cursor.num_cursor));
		cursor.num_cursor++;
		cursor.pos++;
		return;
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		builder.slots.push_back(slot);
		cursor.pos++;
		return;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

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

void EmitValueVerbatim(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		auto layout = ReadObjectLayout(view, cursor.pos);
		builder.EmitObjectStart(layout.key_count);
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
		cursor.pos = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			builder.EmitObjectChildStart();
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
		builder.EmitObjectEnd();
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		cursor.pos++;
		return;
	}
	if (slot_tag == tag::ARR_START) {
		auto end_pos = ReadArrayEndPos(view, cursor.pos);
		builder.EmitArrayStart();
		cursor.pos++;
		while (cursor.pos < end_pos) {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
		builder.EmitArrayEnd();
		cursor.pos = end_pos + 1;
		return;
	}
	EmitScalarVerbatim(view, cursor, builder);
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
// lands in, and its type. The merge folds the four-BLOB residuals (existing logic) and
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
};

void EmitReconShredScalar(JsonoBuilder &builder, const ReconShred &shred, const UnifiedVectorFormat &fmt, idx_t idx) {
	ShredPrimitive kind;
	if (!TypeToShredPrimitive(shred.type, kind)) {
		throw InternalException("jsono reconstruct: non-scalar shred type '%s'", shred.type.ToString());
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

// True if an array shred's pure object-key path ends at / continues past `key` when matched at
// `depth`. Shared by the read overlay and the write skeleton emit (which it precedes in the TU).
static bool ArrayPathTerminates(const vector<PathStep> &path, size_t depth, nonstd::string_view key) {
	return path.size() == depth + 1 && nonstd::string_view(path[depth].key.data(), path[depth].key.size()) == key;
}

static bool ArrayPathContinues(const vector<PathStep> &path, size_t depth, nonstd::string_view key) {
	return path.size() > depth + 1 && nonstd::string_view(path[depth].key.data(), path[depth].key.size()) == key;
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
				ReconShred element_shred {0, shred.element_type, {}};
				EmitReconShredScalar(builder, element_shred, shred.element_fmt, elem_idx);
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
					ReconShred sub_shred {0, shred.subfields[j].second, {}};
					EmitReconShredScalar(scratch.patch_builder, sub_shred, shred.sub_fmt[j], sub_idx);
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
			if (ArrayPathTerminates(shred->path, depth, key)) {
				terminal = shred;
			} else if (ArrayPathContinues(shred->path, depth, key)) {
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
				EmitReconShredScalar(patch_builder, shreds[k], shred_fmt[k], RowIndex(shred_fmt[k], row));
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
			EmitReconShredScalar(patch_builder, shreds[k], shred_fmt[k], RowIndex(shred_fmt[k], row));
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
// JSONO into `out` (the four-BLOB residual). Shared by the shred-aware fast path (raw residuals)
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
	// at its path; an ARRAY shred is not a whole-lane copy (its elements interleave the residual
	// skeleton) so it falls back. The per-row gate diverts to the fallback whenever an input could
	// make the lane copy-through wrong (a non-object replace, a key/path naming a lane).
	bool fast_viable = true;
	vector<idx_t> plain_inputs;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (t.id() != LogicalTypeId::SQLNULL && !IsShreddedJsonoType(t)) {
			plain_inputs.push_back(i);
		}
	}
	// Parse each shred's object-key path once (top-level shreds are a single Key step). An array
	// shred, or any non-Key step (only reachable through an array path), disqualifies the fast path.
	vector<vector<PathStep>> shred_paths(bind_data.shreds.size());
	bool has_nested_shred = false;
	for (idx_t k = 0; k < bind_data.shreds.size() && fast_viable; k++) {
		auto &shred = bind_data.shreds[k];
		if (IsShredArrayType(shred.type) || IsShredScalarArrayType(shred.type)) {
			fast_viable = false;
			break;
		}
		if (shred.name.size() >= 2 && shred.name[0] == '$' && shred.name[1] == '.') {
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
				bool hit = shred_paths[k].size() == 1 ? ResidualHasTopLevelKey(v, bind_data.shreds[k].name)
				                                      : ResidualConflictsWithShredPath(v, shred_paths[k]);
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
				if (has_nested_shred) {
					for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
						if (shred_paths[k].size() > 1 && ResidualConflictsWithShredPath(pview, shred_paths[k])) {
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
			// The fast path is reached only when no shred key appears in the folded residual (the
			// no-conflict gate above). A diverted scalar (case B) would sit in that residual at the
			// shred's key and trip the gate to the fallback — so every NULL shred here is an absent
			// path (case A), never a diversion: the result is value-complete by construction.
			JsonoFillValueCompleteAllTrue(result, count);
			auto &fr_blobs = StructVector::GetEntries(JsonoBodyVector(fast_residual));
			FlatVector::Validity(result) = FlatVector::Validity(fast_residual);
			for (idx_t b = 0; b < BODY_BLOB_COUNT; b++) {
				if (b == BODY_SKIPS) {
					continue;
				}
				VectorOperations::Copy(*fr_blobs[b], *writer.vec[b], count, 0, 0);
			}
			auto &result_validity = FlatVector::Validity(result);
			auto &vc = JsonoVcVector(result);
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row)) {
					FlatVector::SetNull(vc, row, true);
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
			vector<std::string> manifest_entries(bind_data.shreds.size());
			vector<UnifiedVectorFormat> shred_fmt(bind_data.shreds.size());
			for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
				auto &shred = bind_data.shreds[k];
				auto &entry = manifest_entries[k];
				auto append_lv = [&](const string &text) {
					if (text.size() > std::numeric_limits<uint16_t>::max()) {
						throw InvalidInputException("jsono merge: path or type name exceeds manifest limits");
					}
					uint16_t len = uint16_t(text.size());
					entry.append(reinterpret_cast<const char *>(&len), sizeof(len));
					entry.append(text);
				};
				append_lv(shred.name);
				append_lv(shred.type.ToString());
				JsonoShredVector(result, shred.result_child_index).ToUnifiedFormat(count, shred_fmt[k]);
			}
			auto fr_skips = FlatVector::GetData<string_t>(*fr_blobs[BODY_SKIPS]);
			auto &fr_skips_validity = FlatVector::Validity(*fr_blobs[BODY_SKIPS]);
			auto &r_skips = writer.Skips();
			auto skips_out = writer.data[BODY_SKIPS];
			std::string skips_buf;
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row) || !fr_skips_validity.RowIsValid(row)) {
					FlatVector::SetNull(r_skips, row, true);
					continue;
				}
				skips_buf.clear();
				skips_buf.append(fr_skips[row].GetData(), fr_skips[row].GetSize());
				uint32_t stripped = 0;
				for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
					stripped += RowIsValid(shred_fmt[k], row) ? 1 : 0;
				}
				if (stripped > 0) {
					skips_buf.append(reinterpret_cast<const char *>(&stripped), sizeof(stripped));
					for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
						if (RowIsValid(shred_fmt[k], row)) {
							skips_buf.append(manifest_entries[k]);
						}
					}
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
// State carries two parallel JSONO blobs: `values` is the merged document; `keys` mirrors its
// object structure with every leaf replaced by the winning row's sort-key bytes, stored as a
// string value (read/written only through the verbatim cursor path — never validated or rendered
// as JSON, so arbitrary sort-key bytes are fine). Object<->scalar type changes at the same path
// across rows are resolved wholesale by the larger subtree key; for such structurally-unstable
// paths the result stays order-independent but may differ from the sequential ORDER BY fold.

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

struct GroupMergeLWWState {
	OwnedJsonoBlob *values;
	OwnedJsonoBlob *keys;
	bool has_input;
};

struct GroupMergeLWWFunction {
	static void Initialize(GroupMergeLWWState &state) {
		state.values = nullptr;
		state.keys = nullptr;
		state.has_input = false;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.values;
		delete state.keys;
		state.values = nullptr;
		state.keys = nullptr;
		state.has_input = false;
	}
};

// Read a key-tree leaf's stored sort-key bytes (a VAL_STR_HEAP string value at `cursor`).
nonstd::string_view KeyLeafBytes(const JsonoView &kview, const JsonoCursor &cursor) {
	auto len = kview.LengthAt(cursor.length_cursor);
	return kview.StringAt(cursor.string_cursor, len);
}

// The representative key of a key-tree node: its own sort-key if it is a leaf, else the maximum
// over its subtree's leaves. Only the object branch (an object<->scalar transition) recurses; the
// common leaf-vs-leaf conflict reads one string directly with no allocation.
nonstd::string_view MaxKeyInSubtree(const JsonoView &kview, const JsonoCursor &cursor) {
	auto t = SlotTag(kview.SlotAt(cursor.pos));
	if (t != tag::OBJ_START) {
		if (t != tag::VAL_STR_HEAP) {
			throw InternalException("jsono_group_merge: malformed key tree leaf");
		}
		return KeyLeafBytes(kview, cursor);
	}
	auto layout = ReadObjectLayout(kview, cursor.pos);
	std::vector<MergeChild> kids;
	CollectObjectChildren(kview, layout, cursor, kids);
	nonstd::string_view best;
	bool have = false;
	for (auto &kid : kids) {
		auto cand = MaxKeyInSubtree(kview, kid.cursor);
		if (!have || CompareJsonoKeys(cand, best) > 0) {
			best = cand;
			have = true;
		}
	}
	if (!have) {
		throw InternalException("jsono_group_merge: empty key-tree object");
	}
	return best;
}

template <class T>
int CompareVecBytes(const std::vector<T> &a, const std::vector<T> &b) {
	size_t na = a.size() * sizeof(T);
	size_t nb = b.size() * sizeof(T);
	size_t n = std::min(na, nb);
	if (n > 0) {
		int c = std::memcmp(a.data(), b.data(), n);
		if (c != 0) {
			return c;
		}
	}
	return na < nb ? -1 : (na > nb ? 1 : 0);
}

// Deterministic total order over two JSONO values, used only to break an exact key tie so the
// fold stays order-independent. Serializes each value standalone (heaps start at 0, so equal
// values produce byte-identical streams) and compares the streams in a fixed order.
int CompareValueTie(const JsonoView &va, const JsonoCursor &ca, const JsonoView &vb, const JsonoCursor &cb) {
	static thread_local JsonoBuilder ta;
	static thread_local JsonoBuilder tb;
	ta.Reset();
	tb.Reset();
	JsonoCursor x = ca;
	EmitValueVerbatim(va, x, ta, 0);
	JsonoCursor y = cb;
	EmitValueVerbatim(vb, y, tb, 0);
	if (int c = CompareVecBytes(ta.slots, tb.slots)) {
		return c;
	}
	if (int c = CompareVecBytes(ta.string_heap, tb.string_heap)) {
		return c;
	}
	if (int c = CompareVecBytes(ta.nums, tb.nums)) {
		return c;
	}
	if (int c = CompareVecBytes(ta.key_heap, tb.key_heap)) {
		return c;
	}
	return CompareVecBytes(ta.lengths, tb.lengths);
}

// Per-leaf last-write-wins is well-defined only when every path has a consistent kind across rows
// (always an object, or always a non-object leaf). When a path is a non-empty object in one row and
// a scalar/array in another, the sequential RFC 7396 fold's result depends on row order, so a
// commutative per-leaf rule cannot reproduce it — failing loud beats a silently order-dependent
// answer. The accumulator never stores empty/stripped objects, so an OBJ_START here is always a real
// structural object.
[[noreturn]] void ThrowMixedKindConflict() {
	throw InvalidInputException(
	    "jsono_group_merge_max/min: a path is an object in one row and a scalar/array in another; "
	    "last-write-wins per leaf is order-dependent for such structurally-inconsistent data. "
	    "Use jsono_group_merge(value ORDER BY key) instead for paths that change kind across rows.");
}

// >0 if the A subtree's key wins the conflict, <0 if B wins, 0 on an exact key tie.
int CompareConflict(const JsonoView &va, const JsonoCursor &ca, const JsonoView &ka, const JsonoCursor &cka,
                    const JsonoView &vb, const JsonoCursor &cb, const JsonoView &kb, const JsonoCursor &ckb) {
	int c = CompareJsonoKeys(MaxKeyInSubtree(ka, cka), MaxKeyInSubtree(kb, ckb));
	if (c != 0) {
		return c;
	}
	return CompareValueTie(va, ca, vb, cb);
}

void MergeLWW(const JsonoView &va, const JsonoCursor &ca, const JsonoView &ka, const JsonoCursor &cka,
              const JsonoView &vb, const JsonoCursor &cb, const JsonoView &kb, const JsonoCursor &ckb,
              JsonoBuilder &out_v, JsonoBuilder &out_k, size_t depth);

// Merge two object nodes (with their mirrored key nodes) into out_v/out_k. Both sides are
// non-null, non-empty objects (the accumulator never stores nulls/empties, and incoming rows are
// null-stripped before merge), so this is a pure sorted-key two-pointer with no null bookkeeping.
void MergeLWWObjects(const JsonoView &va, const JsonoCursor &ca, const JsonoView &ka, const JsonoCursor &cka,
                     const JsonoView &vb, const JsonoCursor &cb, const JsonoView &kb, const JsonoCursor &ckb,
                     JsonoBuilder &out_v, JsonoBuilder &out_k, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	// The key children align with the value children by index (same keys, same order), so the
	// two-pointer compares value keys and indexes the key children with the same rank.
	std::vector<MergeChild> cva, cka_ch, cvb, ckb_ch;
	CollectObjectChildren(va, ReadObjectLayout(va, ca.pos), ca, cva);
	CollectObjectChildren(ka, ReadObjectLayout(ka, cka.pos), cka, cka_ch);
	CollectObjectChildren(vb, ReadObjectLayout(vb, cb.pos), cb, cvb);
	CollectObjectChildren(kb, ReadObjectLayout(kb, ckb.pos), ckb, ckb_ch);

	struct PlanEntry {
		nonstd::string_view key;
		MergeSrc src;
		size_t ra;
		size_t rb;
	};
	std::vector<PlanEntry> plan;
	plan.reserve(cva.size() + cvb.size());
	size_t ia = 0;
	size_t ib = 0;
	while (ia < cva.size() || ib < cvb.size()) {
		int cmp;
		if (ib >= cvb.size()) {
			cmp = -1;
		} else if (ia >= cva.size()) {
			cmp = 1;
		} else {
			cmp = CompareJsonoKeys(cva[ia].key, cvb[ib].key);
		}
		if (cmp < 0) {
			plan.push_back(PlanEntry {cva[ia].key, MergeSrc::A, ia, 0});
			ia++;
		} else if (cmp > 0) {
			plan.push_back(PlanEntry {cvb[ib].key, MergeSrc::B, 0, ib});
			ib++;
		} else {
			if (cva[ia].tag == tag::OBJ_START && cvb[ib].tag == tag::OBJ_START) {
				plan.push_back(PlanEntry {cva[ia].key, MergeSrc::Recurse, ia, ib});
			} else {
				if (cva[ia].tag == tag::OBJ_START || cvb[ib].tag == tag::OBJ_START) {
					ThrowMixedKindConflict();
				}
				int c = CompareConflict(va, cva[ia].cursor, ka, cka_ch[ia].cursor, vb, cvb[ib].cursor, kb,
				                        ckb_ch[ib].cursor);
				plan.push_back(PlanEntry {cva[ia].key, c >= 0 ? MergeSrc::A : MergeSrc::B, ia, ib});
			}
			ia++;
			ib++;
		}
	}

	out_v.EmitObjectStart(plan.size());
	for (auto &p : plan) {
		out_v.EmitKeySlot(p.key);
	}
	out_k.EmitObjectStart(plan.size());
	for (auto &p : plan) {
		out_k.EmitKeySlot(p.key);
	}
	for (auto &p : plan) {
		out_v.EmitObjectChildStart();
		out_k.EmitObjectChildStart();
		if (p.src == MergeSrc::Recurse) {
			MergeLWW(va, cva[p.ra].cursor, ka, cka_ch[p.ra].cursor, vb, cvb[p.rb].cursor, kb, ckb_ch[p.rb].cursor,
			         out_v, out_k, depth + 1);
		} else if (p.src == MergeSrc::A) {
			JsonoCursor vc = cva[p.ra].cursor;
			EmitValueVerbatim(va, vc, out_v, depth + 1);
			JsonoCursor kc = cka_ch[p.ra].cursor;
			EmitValueVerbatim(ka, kc, out_k, depth + 1);
		} else {
			JsonoCursor vc = cvb[p.rb].cursor;
			EmitValueVerbatim(vb, vc, out_v, depth + 1);
			JsonoCursor kc = ckb_ch[p.rb].cursor;
			EmitValueVerbatim(kb, kc, out_k, depth + 1);
		}
	}
	out_v.EmitObjectEnd();
	out_k.EmitObjectEnd();
}

void MergeLWW(const JsonoView &va, const JsonoCursor &ca, const JsonoView &ka, const JsonoCursor &cka,
              const JsonoView &vb, const JsonoCursor &cb, const JsonoView &kb, const JsonoCursor &ckb,
              JsonoBuilder &out_v, JsonoBuilder &out_k, size_t depth) {
	bool a_obj = SlotTag(va.SlotAt(ca.pos)) == tag::OBJ_START;
	bool b_obj = SlotTag(vb.SlotAt(cb.pos)) == tag::OBJ_START;
	if (a_obj && b_obj) {
		MergeLWWObjects(va, ca, ka, cka, vb, cb, kb, ckb, out_v, out_k, depth);
		return;
	}
	if (a_obj || b_obj) {
		ThrowMixedKindConflict();
	}
	// Both sides are non-object leaves at this path: the higher key wins the whole node.
	int c = CompareConflict(va, ca, ka, cka, vb, cb, kb, ckb);
	if (c >= 0) {
		JsonoCursor vc = ca;
		EmitValueVerbatim(va, vc, out_v, depth);
		JsonoCursor kc = cka;
		EmitValueVerbatim(ka, kc, out_k, depth);
	} else {
		JsonoCursor vc = cb;
		EmitValueVerbatim(vb, vc, out_v, depth);
		JsonoCursor kc = ckb;
		EmitValueVerbatim(kb, kc, out_k, depth);
	}
}

void BuildIncomingValue(const JsonoView &V, JsonoCursor &cursor, nonstd::string_view K, JsonoBuilder &out_v,
                        JsonoBuilder &out_k, size_t depth);

// Emit one incoming object into out_v (null members stripped, IGNORE NULLS) and the mirrored
// key tree into out_k (every surviving leaf gets the row's sort key K). Models EmitObjectStrip's
// survival mask but with the parallel key emission and recursion.
void BuildIncomingObject(const JsonoView &V, const JsonoCursor &obj_cursor, nonstd::string_view K, JsonoBuilder &out_v,
                         JsonoBuilder &out_k, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout = ReadObjectLayout(V, obj_cursor.pos);
	JsonoCursor value_start = obj_cursor;
	value_start.pos = layout.value_start;
	static constexpr size_t MASK_STACK = 128;
	char keep[MASK_STACK];
	bool cached = layout.key_count <= MASK_STACK;
	size_t count = 0;
	{
		JsonoCursor vc = value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = MemberSurvivesStrip(V, vc.pos, MergeMode::IgnoreNulls);
			if (cached) {
				keep[i] = survives ? 1 : 0;
			}
			count += survives ? 1 : 0;
			SkipValueFast(V, vc);
		}
	}
	out_v.EmitObjectStart(count);
	out_k.EmitObjectStart(count);
	{
		JsonoCursor vc = value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = cached ? (keep[i] != 0) : MemberSurvivesStrip(V, vc.pos, MergeMode::IgnoreNulls);
			if (survives) {
				auto key_slot = V.SlotAt(layout.key_start + i);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				auto key = V.KeyAt(SlotPayload(key_slot));
				out_v.EmitKeySlot(key);
				out_k.EmitKeySlot(key);
			}
			SkipValueFast(V, vc);
		}
	}
	{
		JsonoCursor vc = value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = cached ? (keep[i] != 0) : MemberSurvivesStrip(V, vc.pos, MergeMode::IgnoreNulls);
			if (!survives) {
				SkipValueFast(V, vc);
				continue;
			}
			out_v.EmitObjectChildStart();
			out_k.EmitObjectChildStart();
			BuildIncomingValue(V, vc, K, out_v, out_k, depth + 1);
		}
	}
	out_v.EmitObjectEnd();
	out_k.EmitObjectEnd();
}

void BuildIncomingValue(const JsonoView &V, JsonoCursor &cursor, nonstd::string_view K, JsonoBuilder &out_v,
                        JsonoBuilder &out_k, size_t depth) {
	auto slot_tag = SlotTag(V.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		BuildIncomingObject(V, cursor, K, out_v, out_k, depth);
		SkipValueFast(V, cursor);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		// Arrays are a single leaf (replaced wholesale), exactly like jsono_group_merge.
		EmitValueVerbatim(V, cursor, out_v, depth);
		out_k.EmitString(K);
		return;
	}
	// Any scalar, including a top-level JSON null (a non-object incoming replaces wholesale).
	EmitScalarVerbatim(V, cursor, out_v);
	out_k.EmitString(K);
}

// Serialize the incoming row into the value/key builders, then either seed an empty group or
// merge into the accumulator. Mirrors FoldIntoGroupState but maintains the parallel key tree.
void FoldRowLWW(GroupMergeLWWState &state, const JsonoView &V, nonstd::string_view K) {
	static thread_local JsonoBuilder inc_v;
	static thread_local JsonoBuilder inc_k;
	inc_v.Reset();
	inc_k.Reset();
	JsonoCursor cursor;
	BuildIncomingValue(V, cursor, K, inc_v, inc_k, 0);

	if (!state.has_input) {
		state.values = new OwnedJsonoBlob();
		state.keys = new OwnedJsonoBlob();
		SerializeBuilderToBlob(inc_v, *state.values);
		SerializeBuilderToBlob(inc_k, *state.keys);
		state.has_input = true;
		return;
	}

	static thread_local OwnedJsonoBlob inc_v_blob;
	static thread_local OwnedJsonoBlob inc_k_blob;
	SerializeBuilderToBlob(inc_v, inc_v_blob);
	SerializeBuilderToBlob(inc_k, inc_k_blob);

	static thread_local JsonoBuilder out_v;
	static thread_local JsonoBuilder out_k;
	out_v.Reset();
	out_k.Reset();
	JsonoView acc_v = ViewOfBlob(*state.values);
	JsonoView acc_k = ViewOfBlob(*state.keys);
	JsonoView iv = ViewOfBlob(inc_v_blob);
	JsonoView ik = ViewOfBlob(inc_k_blob);
	if (!acc_v.ParseHeader() || !acc_k.ParseHeader() || !iv.ParseHeader() || !ik.ParseHeader()) {
		throw InternalException("jsono_group_merge: malformed accumulator blob");
	}
	MergeLWW(acc_v, JsonoCursor(), acc_k, JsonoCursor(), iv, JsonoCursor(), ik, JsonoCursor(), out_v, out_k, 0);
	SerializeBuilderToBlob(out_v, *state.values);
	SerializeBuilderToBlob(out_k, *state.keys);
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

void JsonoGroupMergeLWWSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                    data_ptr_t state_ptr, idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeLWWBindData>();
	Vector reconstructed(JsonoType(), count);
	JsonoRowReader reader;
	reader.Init(*GroupMergeReadInput(inputs[0], count, reconstructed), count);
	Vector sort_keys(LogicalType::BLOB, count);
	LWWMakeSortKeys(inputs[1], count, bind_data, sort_keys);
	UnifiedVectorFormat sk_fmt;
	sort_keys.ToUnifiedFormat(count, sk_fmt);
	auto sk_data = UnifiedVectorFormat::GetData<string_t>(sk_fmt);

	auto &state = *reinterpret_cast<GroupMergeLWWState *>(state_ptr);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto &k = sk_data[sk_fmt.sel->get_index(row)];
		FoldRowLWW(state, view, nonstd::string_view(k.GetData(), k.GetSize()));
	}
}

void JsonoGroupMergeLWWUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                              idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeLWWBindData>();
	Vector reconstructed(JsonoType(), count);
	JsonoRowReader reader;
	reader.Init(*GroupMergeReadInput(inputs[0], count, reconstructed), count);
	Vector sort_keys(LogicalType::BLOB, count);
	LWWMakeSortKeys(inputs[1], count, bind_data, sort_keys);
	UnifiedVectorFormat sk_fmt;
	sort_keys.ToUnifiedFormat(count, sk_fmt);
	auto sk_data = UnifiedVectorFormat::GetData<string_t>(sk_fmt);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeLWWState *>(state_fmt);

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto &k = sk_data[sk_fmt.sel->get_index(row)];
		FoldRowLWW(*state_data[RowIndex(state_fmt, row)], view, nonstd::string_view(k.GetData(), k.GetSize()));
	}
}

void JsonoGroupMergeLWWCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	(void)aggr_input_data;
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<GroupMergeLWWState *>(source_fmt);
	auto target_data = FlatVector::GetData<GroupMergeLWWState *>(target);

	static thread_local JsonoBuilder out_v;
	static thread_local JsonoBuilder out_k;
	for (idx_t row = 0; row < count; row++) {
		auto &src = *source_data[RowIndex(source_fmt, row)];
		if (!src.has_input || !src.values || !src.keys) {
			continue;
		}
		auto &tgt = *target_data[row];
		if (!tgt.has_input) {
			tgt.values = new OwnedJsonoBlob(*src.values);
			tgt.keys = new OwnedJsonoBlob(*src.keys);
			tgt.has_input = true;
			continue;
		}
		out_v.Reset();
		out_k.Reset();
		JsonoView tv = ViewOfBlob(*tgt.values);
		JsonoView tk = ViewOfBlob(*tgt.keys);
		JsonoView sv = ViewOfBlob(*src.values);
		JsonoView sk = ViewOfBlob(*src.keys);
		if (!tv.ParseHeader() || !tk.ParseHeader() || !sv.ParseHeader() || !sk.ParseHeader()) {
			throw InternalException("jsono_group_merge: malformed accumulator blob");
		}
		MergeLWW(tv, JsonoCursor(), tk, JsonoCursor(), sv, JsonoCursor(), sk, JsonoCursor(), out_v, out_k, 0);
		SerializeBuilderToBlob(out_v, *tgt.values);
		SerializeBuilderToBlob(out_k, *tgt.keys);
	}
}

// Write each group's merged `values` blob (or an empty object for an empty group) into `out`.
void FinalizeLWWPlainGroups(Vector &out, UnifiedVectorFormat &state_fmt, GroupMergeLWWState *const *state_data,
                            idx_t count, idx_t base) {
	JsonoBodyWriter writer;
	writer.Init(out);
	JsonoBuilder empty_builder;
	for (idx_t i = 0; i < count; i++) {
		auto rid = i + base;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.has_input || !state.values) {
			empty_builder.Reset();
			empty_builder.EmitObjectStart(0);
			empty_builder.EmitObjectEnd();
			writer.WriteRow(rid, empty_builder);
			continue;
		}
		WriteOwnedBlobRow(writer, rid, *state.values);
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
static void EmitObjectStrippingPaths(const JsonoView &view, const JsonoCursor &obj_cursor, JsonoBuilder &builder,
                                     const std::vector<const std::vector<PathStep> *> &active, size_t depth) {
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
	JsonoCursor value_cursor = obj_cursor;
	value_cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (PathTerminatesOnKey(active, depth, key)) {
			SkipValueFast(view, value_cursor);
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
		if (!deeper.empty() && SlotTag(view.SlotAt(value_cursor.pos)) == tag::OBJ_START) {
			EmitObjectStrippingPaths(view, value_cursor, builder, deeper, depth + 1);
			SkipValueFast(view, value_cursor);
		} else {
			EmitValueVerbatim(view, value_cursor, builder, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

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
	EmitObjectStrippingPaths(view, jsono::JsonoCursor(), builder, paths, 0);
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

// Emit the skeleton scalar array (cursor at ARR_START): each element that the lane lifts (a scalar
// fitting `element_type`) becomes a VAL_NULL placeholder — it carries no value, the parallel
// LIST<element_type> shred does — every other element (a non-conforming scalar, a null, an object or
// array) stays verbatim. The lift gate is JsonoScalarFitsShredType, the same one WriteScalarArrayShred
// uses, so the placeholder positions match the shred's non-NULL slots exactly. On reconstruct the
// placeholder is never read: a non-NULL shred slot overrides it, a NULL slot means the element was
// kept (and a kept element is never a placeholder), so a placeholder and an explicit JSON null never
// collide.
static void EmitSkeletonScalarArray(const JsonoView &view, const JsonoCursor &array_cursor, JsonoBuilder &builder,
                                    const LogicalType &element_type, size_t depth) {
	auto end_pos = ReadArrayEndPos(view, array_cursor.pos);
	builder.EmitArrayStart();
	JsonoCursor cursor = array_cursor;
	cursor.pos++; // first element; ARR_START consumes no stream entries
	while (cursor.pos < end_pos) {
		auto value_tag = SlotTag(view.SlotAt(cursor.pos));
		if (value_tag != tag::OBJ_START && value_tag != tag::ARR_START) {
			JsonoCursor probe = cursor;
			auto scalar = DecodeScalarAt(view, probe);
			if (JsonoScalarFitsShredType(scalar, element_type)) {
				builder.EmitNull(); // lifted: placeholder; the shred holds the value
				SkipValueFast(view, cursor);
				continue;
			}
		}
		EmitValueVerbatim(view, cursor, builder, depth + 1);
	}
	builder.EmitArrayEnd();
}

// Emit the skeleton array (cursor at ARR_START): each object element minus its strippable
// subfields, every other element verbatim.
static void EmitSkeletonArray(const JsonoView &view, const JsonoCursor &array_cursor, JsonoBuilder &builder,
                              const vector<std::pair<string, LogicalType>> &subfields, size_t depth) {
	auto end_pos = ReadArrayEndPos(view, array_cursor.pos);
	builder.EmitArrayStart();
	JsonoCursor cursor = array_cursor;
	cursor.pos++; // first element; ARR_START consumes no stream entries
	std::vector<std::vector<PathStep>> strip_storage;
	std::vector<const std::vector<PathStep> *> strip;
	while (cursor.pos < end_pos) {
		if (SlotTag(view.SlotAt(cursor.pos)) == tag::OBJ_START) {
			strip_storage.clear();
			for (auto &sub : subfields) {
				if (ElementSubfieldStrippable(view, cursor, sub.first, sub.second)) {
					strip_storage.push_back({PathStep {PathStepKind::Key, sub.first, 0}});
				}
			}
			strip.clear();
			for (auto &path : strip_storage) {
				strip.push_back(&path);
			}
			EmitObjectStrippingPaths(view, cursor, builder, strip, 0);
			SkipValueFast(view, cursor);
		} else {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
	}
	builder.EmitArrayEnd();
}

// One object level of the residual skeleton: strip the scalar leaves (`scalar_paths`) and replace
// the array at each terminal array-shred key with its skeleton, recursing for deeper paths.
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
		if (!PathTerminatesOnKey(scalar_paths, depth, key_at(i))) {
			surviving++;
		}
	}
	builder.EmitObjectStart(surviving);
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (!PathTerminatesOnKey(scalar_paths, depth, key)) {
			builder.EmitKeySlot(key);
		}
	}
	JsonoCursor value_cursor = obj_cursor;
	value_cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		if (PathTerminatesOnKey(scalar_paths, depth, key)) {
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
			if (ArrayPathTerminates(spec->path, depth, key)) {
				terminal_array = spec;
			} else if (ArrayPathContinues(spec->path, depth, key)) {
				deeper_array.push_back(spec);
			}
		}
		auto value_tag = SlotTag(view.SlotAt(value_cursor.pos));
		if (terminal_array && value_tag == tag::ARR_START) {
			if (terminal_array->kind == ShredKind::ScalarArray) {
				EmitSkeletonScalarArray(view, value_cursor, builder, terminal_array->element_type, depth);
			} else {
				EmitSkeletonArray(view, value_cursor, builder, terminal_array->subfields, depth);
			}
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
