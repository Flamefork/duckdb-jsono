#pragma once

#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace duckdb {

// ---- Cursor-based merge core ----
// The walkers shared by jsono_merge_patch, jsono_group_merge (both variants), and the shredded
// reconstruct overlay. They walk the source views straight into the builder with no intermediate
// DOM (no std::map, no per-key/value heap allocation); array values, and everything nested inside
// an array, are copied verbatim. Header-only so every consumer TU can still inline them, but that
// is vague linkage rather than the internal linkage they had as single-TU statics — hence the
// explicit JSONO_ALWAYS_INLINE on the small hot helpers (see EmitScalarVerbatim in jsono_copy.hpp
// for the caveat itself).

// Patch: RFC 7396 merge_patch — B wins, a B-null deletes, A's own nulls are kept, and a patch
// object merged onto a non-object/absent value has its own nulls stripped. IgnoreNulls:
// jsono_group_merge (B wins, null members dropped). Overlay: A (the first argument) is
// authoritative — B only fills keys absent from A, B-null is a no-op. Overlay powers shredded
// reconstruction: the residual (A) wins, shreds (B) fill stripped paths.
enum class MergeMode : uint8_t { Patch, IgnoreNulls, Overlay };

// One object child captured in a single pass: its key, the value's cursor (slot
// position plus all three stream cursors), and the value's slot tag. Capturing the
// tag here lets the merge loop classify null/object children without re-reading slots.
struct MergeChild {
	nonstd::string_view key;
	jsono::JsonoCursor cursor;
	uint64_t tag;
};

enum class MergeSrc : uint8_t { A, B, Recurse };

struct MergePlanEntry {
	nonstd::string_view key;
	MergeSrc src;
	size_t ra;
	size_t rb;
};

// Push a scalar value's slot verbatim WITHOUT copying its stream entries. Used by
// the merge value phase to emit a run of scalars whose string bytes and lengths/
// nums entries are copied in bulk (contiguous slices of the source streams):
// per-child it only touches `slots` (cheap), and the streams are filled once.
// Caller must have excluded containers.
JSONO_ALWAYS_INLINE void PushScalarValueSlots(const jsono::JsonoView &view, size_t pos, jsono::JsonoBuilder &builder) {
	auto slot = view.SlotAt(pos);
	switch (jsono::SlotTag(slot)) {
	case jsono::tag::VAL_STR_HEAP:
	case jsono::tag::VAL_EXT:
	case jsono::tag::VAL_INT60:
	case jsono::tag::VAL_DEC60:
	case jsono::tag::VAL_TRUE:
	case jsono::tag::VAL_FALSE:
	case jsono::tag::VAL_NULL:
		builder.slots.push_back(slot);
		return;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

inline void EmitValueStrip(const jsono::JsonoView &view, jsono::JsonoCursor &cursor, jsono::JsonoBuilder &builder,
                           MergeMode merge_mode, size_t depth);

// True if the value at `pos` keeps any content under IGNORE NULLS stripping: a non-null
// scalar or array survives; an object survives iff some member survives (recursively).
// Used to omit keys whose object value strips to empty — jsono_group_merge drops
// `key:{}`, unlike jsono_merge_patch Patch mode which keeps it. Read-only navigation
// with a position-only cursor (the stream cursors start at zero: slot advancement does
// not depend on stream values, and zero-based prefix sums stay within bounds);
// early-exits on the first surviving leaf so non-empty objects cost ~one probe.
inline bool ValueSurvivesIgnoreNulls(const jsono::JsonoView &view, size_t pos) {
	auto slot_tag = jsono::SlotTag(view.SlotAt(pos));
	if (slot_tag == jsono::tag::VAL_NULL) {
		return false;
	}
	if (slot_tag != jsono::tag::OBJ_START) {
		return true;
	}
	auto layout = ReadObjectLayout(view, pos);
	jsono::JsonoCursor probe;
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
JSONO_ALWAYS_INLINE bool MemberSurvivesStrip(const jsono::JsonoView &view, size_t vp, MergeMode merge_mode) {
	if (jsono::SlotTag(view.SlotAt(vp)) == jsono::tag::VAL_NULL) {
		return false;
	}
	return merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(view, vp);
}

inline void EmitObjectStrip(const jsono::JsonoView &view, const jsono::JsonoCursor &obj_cursor,
                            jsono::JsonoBuilder &builder, MergeMode merge_mode, size_t depth) {
	if (depth > jsono::JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)jsono::JSONO_MAX_NESTING_DEPTH);
	}
	auto layout = ReadObjectLayout(view, obj_cursor.pos);
	jsono::JsonoCursor value_start_cursor = obj_cursor;
	value_start_cursor.pos = layout.value_start;
	// Cache the per-member survival mask during the count pass. The IGNORE NULLS survival probe is
	// recursive, so recomputing it across the count/key/value passes is wasteful — and with the mask
	// the key pass skips the value walk entirely (keys are addressed by key_start + i). A stack mask
	// avoids a per-object heap alloc; objects wider than it (rare) fall back to recomputing the cheap
	// predicate per pass.
	static constexpr size_t MASK_STACK = 128;
	char keep[MASK_STACK];
	bool cached = layout.key_count <= MASK_STACK;
	size_t count = 0;
	{
		jsono::JsonoCursor vc = value_start_cursor;
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
			if (jsono::SlotTag(key_slot) != jsono::tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(jsono::SlotPayload(key_slot)));
		}
	} else {
		jsono::JsonoCursor vc = value_start_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			if (MemberSurvivesStrip(view, vc.pos, merge_mode)) {
				auto key_slot = view.SlotAt(layout.key_start + i);
				if (jsono::SlotTag(key_slot) != jsono::tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				builder.EmitKeySlot(view.KeyAt(jsono::SlotPayload(key_slot)));
			}
			SkipValueFast(view, vc);
		}
	}
	{
		jsono::JsonoCursor vc = value_start_cursor;
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

inline void EmitValueStrip(const jsono::JsonoView &view, jsono::JsonoCursor &cursor, jsono::JsonoBuilder &builder,
                           MergeMode merge_mode, size_t depth) {
	auto slot_tag = jsono::SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag == jsono::tag::OBJ_START) {
		EmitObjectStrip(view, cursor, builder, merge_mode, depth);
		SkipValueFast(view, cursor);
		return;
	}
	if (slot_tag == jsono::tag::ARR_START) {
		EmitValueVerbatim(view, cursor, builder, depth);
		return;
	}
	EmitScalarVerbatim(view, cursor, builder);
}

JSONO_ALWAYS_INLINE int CompareJsonoKeys(nonstd::string_view a, nonstd::string_view b) {
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
inline jsono::JsonoCursor CollectObjectChildren(const jsono::JsonoView &view, const jsono::ObjectLayout &layout,
                                                const jsono::JsonoCursor &obj_cursor, std::vector<MergeChild> &out) {
	out.clear();
	out.reserve(layout.key_count);
	jsono::JsonoCursor vc = obj_cursor;
	vc.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (jsono::SlotTag(key_slot) != jsono::tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		auto key = view.KeyAt(jsono::SlotPayload(key_slot));
		if (vc.pos >= view.Slots()) {
			throw InvalidInputException("malformed JSONO: value position out of bounds");
		}
		auto value_slot = view.SlotAt(vc.pos);
		out.push_back(MergeChild {key, vc, jsono::SlotTag(value_slot)});
		SkipValueFastFromSlot(view, value_slot, vc);
	}
	return vc;
}

inline void MergeTwoObjects(const jsono::JsonoView &va, const jsono::JsonoCursor &cursor_a, const jsono::JsonoView &vb,
                            const jsono::JsonoCursor &cursor_b, jsono::JsonoBuilder &builder, MergeMode merge_mode,
                            size_t depth);

// Merge two object views into the builder (B patches A), sorted-key linear merge.
inline void MergeTwoObjectsWithScratch(const jsono::JsonoView &va, const jsono::JsonoCursor &cursor_a,
                                       const jsono::JsonoView &vb, const jsono::JsonoCursor &cursor_b,
                                       jsono::JsonoBuilder &builder, MergeMode merge_mode, size_t depth,
                                       std::vector<MergeChild> &children_a, std::vector<MergeChild> &children_b,
                                       std::vector<MergePlanEntry> &plan) {
	if (depth > jsono::JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)jsono::JSONO_MAX_NESTING_DEPTH);
	}
	auto layout_a = ReadObjectLayout(va, cursor_a.pos);
	auto layout_b = ReadObjectLayout(vb, cursor_b.pos);
	jsono::JsonoCursor children_a_end = CollectObjectChildren(va, layout_a, cursor_a, children_a);
	jsono::JsonoCursor children_b_end = CollectObjectChildren(vb, layout_b, cursor_b, children_b);

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
			if (merge_mode != MergeMode::IgnoreNulls || children_a[ia].tag != jsono::tag::VAL_NULL) {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, 0});
			}
			ia++;
		} else if (cmp > 0) {
			// B-only key. Patch keeps every non-null member (RFC 7396 may leave `{}`);
			// Overlay and IGNORE NULLS drop a member whose object value strips to empty.
			if (children_b[ib].tag != jsono::tag::VAL_NULL &&
			    (merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(vb, children_b[ib].cursor.pos))) {
				plan.push_back(MergePlanEntry {children_b[ib].key, MergeSrc::B, 0, ib});
			}
			ib++;
		} else if (merge_mode == MergeMode::Overlay) {
			// Both sides have the key. A (the residual) is authoritative. If both are
			// objects, recurse so B can still fill nested keys A dropped (A wins every
			// leaf conflict); otherwise keep A verbatim, including an explicit null.
			if (children_a[ia].tag == jsono::tag::OBJ_START && children_b[ib].tag == jsono::tag::OBJ_START) {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::Recurse, ia, ib});
			} else {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, ib});
			}
			ia++;
			ib++;
		} else {
			if (children_b[ib].tag != jsono::tag::VAL_NULL) {
				if (children_a[ia].tag == jsono::tag::OBJ_START && children_b[ib].tag == jsono::tag::OBJ_START) {
					// Recurse keeps A's surviving members, so the merged object is non-empty.
					plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::Recurse, ia, ib});
				} else if (merge_mode == MergeMode::Patch || ValueSurvivesIgnoreNulls(vb, children_b[ib].cursor.pos)) {
					plan.push_back(MergePlanEntry {children_b[ib].key, MergeSrc::B, ia, ib});
				}
			} else if (merge_mode == MergeMode::IgnoreNulls && children_a[ia].tag != jsono::tag::VAL_NULL) {
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
		has_checkpoints = open.checkpoint_offset != jsono::NO_OBJECT_CHECKPOINTS;
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
		const jsono::JsonoView &vsrc = src == MergeSrc::A ? va : vb;
		auto &csrc = src == MergeSrc::A ? children_a : children_b;
		const jsono::JsonoCursor &csrc_end = src == MergeSrc::A ? children_a_end : children_b_end;
		size_t rank = src == MergeSrc::A ? plan[k].ra : plan[k].rb;
		auto value_tag = csrc[rank].tag;
		if (value_tag == jsono::tag::OBJ_START || value_tag == jsono::tag::ARR_START) {
			builder.EmitObjectChildStart();
			jsono::JsonoCursor c = csrc[rank].cursor;
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
		jsono::JsonoCursor first_cursor = csrc[rank].cursor;
		size_t last_rank = rank;
		size_t run_end = k + 1;
		while (run_end < boundary && run_end < plan_size && plan[run_end].src == src) {
			size_t run_rank = src == MergeSrc::A ? plan[run_end].ra : plan[run_end].rb;
			auto run_tag = csrc[run_rank].tag;
			if (run_tag == jsono::tag::OBJ_START || run_tag == jsono::tag::ARR_START) {
				break;
			}
			const jsono::JsonoCursor &after_last = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].cursor : csrc_end;
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
		const jsono::JsonoCursor &run_end_cursor = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].cursor : csrc_end;
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

inline void MergeTwoObjects(const jsono::JsonoView &va, const jsono::JsonoCursor &cursor_a, const jsono::JsonoView &vb,
                            const jsono::JsonoCursor &cursor_b, jsono::JsonoBuilder &builder, MergeMode merge_mode,
                            size_t depth) {
	std::vector<MergeChild> children_a;
	std::vector<MergeChild> children_b;
	std::vector<MergePlanEntry> plan;
	MergeTwoObjectsWithScratch(va, cursor_a, vb, cursor_b, builder, merge_mode, depth, children_a, children_b, plan);
}

// One RFC 7396 fold step. Returns true when the result is SQL NULL (a SQL NULL
// patch argument replaces the result with SQL NULL). Otherwise the merged value is
// written into `builder`: an object patch deep-merges onto an object accumulator
// (the accumulator is the target — its nulls are kept; the patch's nulls delete
// keys), while onto a non-object or SQL NULL accumulator it is merged onto an empty
// object (its own nulls stripped). A non-object patch replaces the accumulator.
inline bool MergeFoldStep(MergeMode mode, bool acc_is_sqlnull, const jsono::JsonoView &acc_view, bool patch_present,
                          const jsono::JsonoView &patch_view, jsono::JsonoBuilder &builder,
                          std::vector<MergeChild> &children_a, std::vector<MergeChild> &children_b,
                          std::vector<MergePlanEntry> &plan) {
	if (!patch_present) {
		// Patch: a SQL NULL patch nullifies the result. Overlay: A is authoritative, so
		// a missing B leaves the accumulator untouched.
		return mode == MergeMode::Overlay ? acc_is_sqlnull : true;
	}
	builder.Reset();
	jsono::JsonoCursor cursor;
	bool acc_is_object = !acc_is_sqlnull && jsono::SlotTag(acc_view.SlotAt(0)) == jsono::tag::OBJ_START;
	bool patch_is_object = jsono::SlotTag(patch_view.SlotAt(0)) == jsono::tag::OBJ_START;
	if (acc_is_object && patch_is_object) {
		MergeTwoObjectsWithScratch(acc_view, jsono::JsonoCursor(), patch_view, jsono::JsonoCursor(), builder, mode, 0,
		                           children_a, children_b, plan);
		return false;
	}
	if (mode == MergeMode::Overlay) {
		// A authoritative: keep the accumulator if present, otherwise fall to the patch.
		const jsono::JsonoView &src = acc_is_sqlnull ? patch_view : acc_view;
		jsono::JsonoCursor src_cursor;
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

// Outcome of following an object-key shred path from the root of a residual value. Absent: a key
// along the path is missing from its (present) object parent — the shred safely owns/creates that
// subtree. Present: the full path resolved to a present value — the residual shadows the shred (an
// RFC 7396 null there would delete it). NonObjectPrefix: the root or a node before the leaf is a
// present non-object — the shred cannot overlay into a scalar/array. The three direct-fold sites
// (shadow-delete, combine adopt, finalize) plus the nested-conflict test all depend on this exact
// three-way split agreeing, so the walk lives here once.
enum class ResidualPathOutcome : uint8_t { Absent, Present, NonObjectPrefix };

// `depth_reached` is the number of leading steps that resolved through objects before the walk
// stopped: steps.size() on Present, and on NonObjectPrefix the count up to AND including the
// non-object node (so a caller can reuse it as the prefix length to delete/rebuild). It is
// meaningless on Absent (no caller reads it there). NonObjectPrefix with depth_reached == 0 means
// the ROOT is not an object, which yields no usable prefix — callers that rebuild from the prefix
// must already know their value is an object (the direct folds gate on their accumulator kind).
JSONO_ALWAYS_INLINE ResidualPathOutcome ClassifyResidualPath(const jsono::JsonoView &view,
                                                             const vector<PathStep> &steps, idx_t &depth_reached) {
	jsono::JsonoCursor cursor;
	for (idx_t s = 0; s < steps.size(); s++) {
		if (cursor.pos >= view.Slots() || jsono::SlotTag(view.SlotAt(cursor.pos)) != jsono::tag::OBJ_START) {
			depth_reached = s; // the non-object sits at step s-1 (or is the root when s == 0)
			return ResidualPathOutcome::NonObjectPrefix;
		}
		if (!LocatePathStep(nullptr, s, view, steps[s], cursor)) {
			depth_reached = s;
			return ResidualPathOutcome::Absent;
		}
	}
	depth_reached = steps.size();
	return ResidualPathOutcome::Present;
}

// Conflict between a value (folded residual or a plain input) and a NESTED object-key shred path:
// the lane copy-through is valid only when the value neither carries the path's leaf nor breaks the
// path's object skeleton — i.e. only an absent key is safe (Present shadows the lane, NonObjectPrefix
// cannot be overlaid into).
JSONO_ALWAYS_INLINE bool ResidualConflictsWithShredPath(const jsono::JsonoView &view, const vector<PathStep> &steps) {
	idx_t depth_reached;
	return ClassifyResidualPath(view, steps, depth_reached) != ResidualPathOutcome::Absent;
}

} // namespace duckdb
