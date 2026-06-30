#pragma once

#include "jsono.hpp"
#include "jsono_number.hpp"

#include "duckdb/common/types/vector.hpp"

#include "string_view.hpp"

#include <cstring>
#include <string>

namespace duckdb {
namespace jsono {

struct JsonoBlobRow {
	string_t slots;
	string_t key_heap;
	string_t string_heap;
	string_t skips;
	string_t lengths;
	string_t nums;
};

struct JsonoVectorData {
	UnifiedVectorFormat struct_fmt;
	UnifiedVectorFormat layout_fmt;
	UnifiedVectorFormat body_fmt;
	UnifiedVectorFormat slots_fmt;
	UnifiedVectorFormat key_heap_fmt;
	UnifiedVectorFormat string_heap_fmt;
	UnifiedVectorFormat skips_fmt;
	UnifiedVectorFormat lengths_fmt;
	UnifiedVectorFormat nums_fmt;
	const string_t *slots_data = nullptr;
	const string_t *key_heap_data = nullptr;
	const string_t *string_heap_data = nullptr;
	const string_t *skips_data = nullptr;
	const string_t *lengths_data = nullptr;
	const string_t *nums_data = nullptr;
	// The string_heap child vector itself (post-flatten): the source of every text value a
	// point read hands out. A VARCHAR reader references this vector's heap into its result so a
	// String/NumberText value can be a string_t pointing straight into these bytes (zero-copy)
	// instead of being copied row by row.
	Vector *string_heap_vec = nullptr;
	// All nine levels report AllValid(): per-row reads skip every validity check (the
	// dominant storage-scan case — six blob children make per-row checks measurable).
	bool all_valid = false;
};

// Layout of one object read off its OBJ_START slot. `after_pos`, `shape_hash` and the
// checkpoint fields are meaningful only when `has_span`: the layout consults the
// ContainerSpan table for objects with child_count > OBJECT_CHECKPOINT_STRIDE (the
// checkpointed ones). Smaller objects are laid out without it — their end position is
// discovered by the walk itself — even though those with a container child do store a
// span (it serves subtree skips, not key lookup).
struct ObjectLayout {
	size_t key_start;
	size_t key_count;
	size_t value_start;
	size_t after_pos;
	bool has_span;
	uint32_t checkpoint_offset;
	uint16_t checkpoint_stride;
	uint64_t shape_hash;
};

inline idx_t RowIndex(const UnifiedVectorFormat &fmt, idx_t row) {
	return fmt.sel->get_index(row);
}

inline bool RowIsValid(const UnifiedVectorFormat &fmt, idx_t row) {
	auto idx = RowIndex(fmt, row);
	return fmt.validity.RowIsValid(idx);
}

// Navigate a jsono value vector (plain or shredded) to its six body blob child vectors:
// top-level layout field [0] -> body [0] -> {slots, key_heap, string_heap, skips, lengths, nums}.
// The body is identical for plain and shredded, so the rest of the reader never needs to know
// which it is.
inline void InitJsonoVectorData(Vector &input, idx_t count, JsonoVectorData &data) {
	input.ToUnifiedFormat(count, data.struct_fmt);
	auto &layout = *StructVector::GetEntries(input)[0];
	layout.ToUnifiedFormat(count, data.layout_fmt);
	auto &body = *StructVector::GetEntries(layout)[0];
	body.ToUnifiedFormat(count, data.body_fmt);
	auto &blobs = StructVector::GetEntries(body);
	blobs[0]->ToUnifiedFormat(count, data.slots_fmt);
	blobs[1]->ToUnifiedFormat(count, data.key_heap_fmt);
	blobs[2]->ToUnifiedFormat(count, data.string_heap_fmt);
	blobs[3]->ToUnifiedFormat(count, data.skips_fmt);
	blobs[4]->ToUnifiedFormat(count, data.lengths_fmt);
	blobs[5]->ToUnifiedFormat(count, data.nums_fmt);
	data.string_heap_vec = blobs[2].get();
	data.slots_data = UnifiedVectorFormat::GetData<string_t>(data.slots_fmt);
	data.key_heap_data = UnifiedVectorFormat::GetData<string_t>(data.key_heap_fmt);
	data.string_heap_data = UnifiedVectorFormat::GetData<string_t>(data.string_heap_fmt);
	data.skips_data = UnifiedVectorFormat::GetData<string_t>(data.skips_fmt);
	data.lengths_data = UnifiedVectorFormat::GetData<string_t>(data.lengths_fmt);
	data.nums_data = UnifiedVectorFormat::GetData<string_t>(data.nums_fmt);
	data.all_valid = data.struct_fmt.validity.AllValid() && data.layout_fmt.validity.AllValid() &&
	                 data.body_fmt.validity.AllValid() && data.slots_fmt.validity.AllValid() &&
	                 data.key_heap_fmt.validity.AllValid() && data.string_heap_fmt.validity.AllValid() &&
	                 data.skips_fmt.validity.AllValid() && data.lengths_fmt.validity.AllValid() &&
	                 data.nums_fmt.validity.AllValid();
}

// The unchecked blob copy shared by both row readers once validity is settled.
inline void ReadJsonoRowBlobs(const JsonoVectorData &data, idx_t row, JsonoBlobRow &out) {
	out.slots = data.slots_data[RowIndex(data.slots_fmt, row)];
	out.key_heap = data.key_heap_data[RowIndex(data.key_heap_fmt, row)];
	out.string_heap = data.string_heap_data[RowIndex(data.string_heap_fmt, row)];
	out.skips = data.skips_data[RowIndex(data.skips_fmt, row)];
	out.lengths = data.lengths_data[RowIndex(data.lengths_fmt, row)];
	out.nums = data.nums_data[RowIndex(data.nums_fmt, row)];
}

// Issue prefetches for the first cache line of every stream a point read is about to walk.
// A row's blobs are cold (the working set is the whole column), and the read path touches
// them serially — slots in ParseHeader, then skips, key_heap, lengths, string_heap during
// the locate — so each first touch stalls in turn. Prefetching them together at row start
// overlaps those misses. Only the per-row point readers (extract / match / project) call
// this; bulk walkers touch the streams densely anyway.
inline void PrefetchJsonoRowStreams(const JsonoBlobRow &blob) {
#if defined(__GNUC__) || defined(__clang__)
	__builtin_prefetch(blob.slots.GetData());
	__builtin_prefetch(blob.key_heap.GetData());
	__builtin_prefetch(blob.string_heap.GetData());
	__builtin_prefetch(blob.skips.GetData());
	__builtin_prefetch(blob.lengths.GetData());
#endif
}

[[noreturn]] inline void ThrowCorruptJsonoRow(const char *field) {
	throw InvalidInputException(
	    "corrupt JSONO storage: top-level row is valid but %s is NULL; this can be produced by an unsafe "
	    "STRUCT cast when the extension optimizer is disabled",
	    field);
}

// Read one row's six body blobs once validity is unsettled. `Strict` selects the missing-child
// policy: the strict read (the default for value-decoding operators) treats a present top-level row
// with a NULL nested field as corruption and fails loud with the field name; the permissive read
// (optimizer-injected expressions that can see NULL body fields on rows the surrounding CASE
// discards) treats any missing nested field as an absent row. A NULL top-level struct is an absent
// value under both. The all_valid fast copy and the per-field handling are otherwise identical, so
// one body owns both.
template <bool Strict>
JSONO_ALWAYS_INLINE bool ReadJsonoRowImpl(const JsonoVectorData &data, idx_t row, JsonoBlobRow &out) {
	if (data.all_valid) {
		ReadJsonoRowBlobs(data, row, out);
		return true;
	}
	if (!RowIsValid(data.struct_fmt, row)) {
		return false;
	}
	if (!RowIsValid(data.layout_fmt, row)) {
		if (Strict) {
			ThrowCorruptJsonoRow("layout");
		}
		return false;
	}
	if (!RowIsValid(data.body_fmt, row)) {
		if (Strict) {
			ThrowCorruptJsonoRow("body");
		}
		return false;
	}
	// The six body blobs share one validity/copy shape; only the field name (strict throw) and the
	// destination member differ. The body and layout checks above gate this loop and stay explicit.
	const struct {
		const UnifiedVectorFormat &fmt;
		const string_t *blob_data;
		const char *field;
		string_t JsonoBlobRow::*member;
	} blobs[] = {
	    {data.slots_fmt, data.slots_data, "slots blob", &JsonoBlobRow::slots},
	    {data.key_heap_fmt, data.key_heap_data, "key_heap blob", &JsonoBlobRow::key_heap},
	    {data.string_heap_fmt, data.string_heap_data, "string_heap blob", &JsonoBlobRow::string_heap},
	    {data.skips_fmt, data.skips_data, "skips blob", &JsonoBlobRow::skips},
	    {data.lengths_fmt, data.lengths_data, "lengths blob", &JsonoBlobRow::lengths},
	    {data.nums_fmt, data.nums_data, "nums blob", &JsonoBlobRow::nums},
	};
	idx_t blob_idx[6];
	for (idx_t i = 0; i < 6; i++) {
		blob_idx[i] = RowIndex(blobs[i].fmt, row);
		if (!blobs[i].fmt.validity.RowIsValid(blob_idx[i])) {
			if (Strict) {
				ThrowCorruptJsonoRow(blobs[i].field);
			}
			return false;
		}
	}
	for (idx_t i = 0; i < 6; i++) {
		out.*blobs[i].member = blobs[i].blob_data[blob_idx[i]];
	}
	return true;
}

// Permissive row read for __jsono_internal_checked_residual: it eagerly verifies every row of the
// reinterpreted residual before the surrounding CASE (WHEN column IS NULL THEN NULL ELSE ...) selects,
// so it legitimately sees NULL body fields on the NULL-column rows the CASE then discards. Any missing
// nested field is treated as an absent row rather than corruption.
inline bool ReadJsonoRow(const JsonoVectorData &data, idx_t row, JsonoBlobRow &out) {
	return ReadJsonoRowImpl<false>(data, row, out);
}

inline bool ReadJsonoRowStrict(const JsonoVectorData &data, idx_t row, JsonoBlobRow &out) {
	return ReadJsonoRowImpl<true>(data, row, out);
}

// Emit `text` as a result string_t without copying its bytes. For a long value the string_t
// stores a pointer straight into `text` (which is a view into the row's string_heap); the
// caller must have referenced that heap into the result vector (StringVector::AddHeapReference
// over JsonoRowReader::StringHeapVector) so the bytes outlive the result. Short values inline
// into the string_t and need no reference. The string_t constructor copies the comparison
// prefix, so this is a drop-in for StringVector::AddString minus the heap copy.
JSONO_ALWAYS_INLINE string_t ZeroCopyHeapText(nonstd::string_view text) {
	return string_t(text.data(), UnsafeNumericCast<uint32_t>(text.size()));
}

inline JsonoView MakeJsonoView(const JsonoBlobRow &blob) {
	return JsonoView(blob.slots.GetData(), blob.slots.GetSize(), blob.key_heap.GetData(), blob.key_heap.GetSize(),
	                 blob.string_heap.GetData(), blob.string_heap.GetSize(), blob.skips.GetData(), blob.skips.GetSize(),
	                 blob.lengths.GetData(), blob.lengths.GetSize(), blob.nums.GetData(), blob.nums.GetSize());
}

inline ContainerSpan CheckedContainerSpan(const JsonoView &view, size_t pos, uint64_t expected_tag) {
	if (pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: container position out of bounds");
	}
	auto slot = view.SlotAt(pos);
	if (SlotTag(slot) != expected_tag) {
		throw InvalidInputException("malformed JSONO: unexpected container tag");
	}
	auto span = view.ContainerSpanAt(ContainerId(SlotPayload(slot)));
	if (span.slot_span < 2 || pos > view.Slots() || size_t(span.slot_span) > view.Slots() - pos) {
		throw InvalidInputException("malformed JSONO: container slot span out of bounds");
	}
	if (size_t(span.string_byte_span) > view.StringHeapSize()) {
		throw InvalidInputException("malformed JSONO: container string span out of bounds");
	}
	return span;
}

// Advance `cursor` past a spanned container in O(1): all four stream cursors jump by the
// span's per-stream counts, bounds-checked against the row's blobs.
inline void AdvanceCursorBySpan(const JsonoView &view, const ContainerSpan &span, JsonoCursor &cursor) {
	if (cursor.string_cursor > view.StringHeapSize() ||
	    size_t(span.string_byte_span) > view.StringHeapSize() - cursor.string_cursor) {
		throw InvalidInputException("malformed JSONO: container string span out of bounds");
	}
	if (cursor.length_cursor > view.LengthCount() ||
	    size_t(span.length_count) > view.LengthCount() - cursor.length_cursor) {
		throw InvalidInputException("malformed JSONO: container lengths span out of bounds");
	}
	if (cursor.num_cursor > view.NumCount() || size_t(span.num_count) > view.NumCount() - cursor.num_cursor) {
		throw InvalidInputException("malformed JSONO: container nums span out of bounds");
	}
	cursor.pos += size_t(span.slot_span);
	cursor.string_cursor += size_t(span.string_byte_span);
	cursor.length_cursor += size_t(span.length_count);
	cursor.num_cursor += size_t(span.num_count);
}

// `probe_small_object_span`: key-lookup callers (extract / optimizer match+project) pass true so a
// legacy small object that does store a span yields its shape_hash — that is what lets their
// per-object rank cache trust a cached key rank by an int compare instead of re-reading the key heap.
// Walk-everything callers (serializers, merge, transform) keep the default: they never consult
// shape_hash, so the sparse-index binary search would be pure waste.
inline ObjectLayout ReadObjectLayout(const JsonoView &view, size_t pos, bool probe_small_object_span = false) {
	auto slot = view.SlotAt(pos);
	auto payload = SlotPayload(slot);
	if (SlotTag(slot) != tag::OBJ_START) {
		throw InvalidInputException("malformed JSONO: unexpected container tag");
	}
	auto key_count = size_t(ContainerChildCount(payload));
	auto key_start = pos + 1;
	if (key_count >= view.Slots() || key_start > view.Slots() - key_count) {
		throw InvalidInputException("malformed JSONO: object child count out of bounds");
	}
	if (key_count <= OBJECT_CHECKPOINT_STRIDE) {
		ContainerSpan span;
		if (probe_small_object_span && view.TryContainerSpan(ContainerId(payload), span)) {
			if (span.slot_span < 2 || size_t(span.slot_span) > view.Slots() - pos) {
				throw InvalidInputException("malformed JSONO: container slot span out of bounds");
			}
			return ObjectLayout {key_start,
			                     key_count,
			                     key_start + key_count,
			                     pos + size_t(span.slot_span),
			                     true,
			                     NO_OBJECT_CHECKPOINTS,
			                     0,
			                     span.shape_hash};
		}
		// Small objects need no span for layout: the walk discovers OBJ_END itself.
		return ObjectLayout {key_start, key_count, key_start + key_count, 0, false, NO_OBJECT_CHECKPOINTS, 0, 0};
	}
	auto span = CheckedContainerSpan(view, pos, tag::OBJ_START);
	auto after_pos = pos + size_t(span.slot_span);
	auto end_pos = after_pos - 1;
	if (SlotTag(view.SlotAt(end_pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object span does not end with OBJ_END");
	}
	if (key_count > end_pos - key_start) {
		throw InvalidInputException("malformed JSONO: object child count out of bounds");
	}
	auto checkpoint_index = view.ObjectCheckpointIndexForContainer(uint32_t(ContainerId(payload)));
	return ObjectLayout {key_start,
	                     key_count,
	                     key_start + key_count,
	                     after_pos,
	                     true,
	                     checkpoint_index.checkpoint_offset,
	                     checkpoint_index.checkpoint_stride,
	                     span.shape_hash};
}

inline size_t ReadArrayEndPos(const JsonoView &view, size_t pos) {
	auto slot = view.SlotAt(pos);
	auto payload = SlotPayload(slot);
	if (SlotTag(slot) != tag::ARR_START) {
		throw InvalidInputException("malformed JSONO: unexpected container tag");
	}
	ContainerSpan span;
	if (view.TryContainerSpan(ContainerId(payload), span)) {
		if (span.slot_span < 2 || size_t(span.slot_span) > view.Slots() - pos) {
			throw InvalidInputException("malformed JSONO: container slot span out of bounds");
		}
		auto end_pos = pos + size_t(span.slot_span) - 1;
		if (SlotTag(view.SlotAt(end_pos)) != tag::ARR_END) {
			throw InvalidInputException("malformed JSONO: array span does not end with ARR_END");
		}
		return end_pos;
	}
	if (pos + 1 < view.Slots() && SlotTag(view.SlotAt(pos + 1)) == tag::ARR_END) {
		return pos + 1;
	}
	throw InvalidInputException("malformed JSONO: container span missing");
}

void SkipValueFast(const JsonoView &view, JsonoCursor &cursor, size_t depth);
void SkipContainerFromSlot(const JsonoView &view, uint64_t slot, JsonoCursor &cursor, size_t depth);

// The scalar tag/ext-subtype ladder and its stream-cursor accounting, shared by the decode
// (DecodeScalarAt) and skip (TrySkipScalarFromSlot) readers via a sink. The ladder is the single
// owner of how a scalar slot maps to its stream reads and cursor advance; the sink decides what to
// do with each decoded component — store it (ScalarDecodeSink) or discard it (ScalarSkipSink). The
// raw stream reads (LengthAt/StringAt/NumAt) stay in the ladder so the skip sink keeps their bounds
// checks even though it discards the bytes. Returns false for containers and non-value slots (the
// caller dispatches those); on an unknown VAL_EXT subtype it throws like both originals did.
// Always-inline: with an empty sink the compiler collapses this back to pure cursor bumps for the
// skip twin, and scalarizes the decoded value into DecodeScalarAt's caller.
template <class SINK>
JSONO_ALWAYS_INLINE bool DecodeScalarSlot(const JsonoView &view, uint64_t slot, JsonoCursor &cursor, SINK &sink) {
	switch (SlotTag(slot)) {
	case tag::VAL_STR_HEAP: {
		auto len = size_t(view.LengthAt(cursor.length_cursor));
		sink.OnString(view.StringAt(cursor.string_cursor, len));
		cursor.string_cursor += len;
		cursor.length_cursor++;
		cursor.pos++;
		return true;
	}
	case tag::VAL_INT60:
		sink.OnInt64(int64_t(view.NumAt(cursor.num_cursor)));
		cursor.num_cursor++;
		cursor.pos++;
		return true;
	case tag::VAL_DEC60:
		sink.OnDec60(view.NumAt(cursor.num_cursor));
		cursor.num_cursor++;
		cursor.pos++;
		return true;
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		if (subtype == ext_subtype::NUMBER) {
			auto len = size_t(view.LengthAt(cursor.length_cursor));
			sink.OnNumberText(view.StringAt(cursor.string_cursor, len));
			cursor.string_cursor += len;
			cursor.length_cursor++;
			cursor.pos++;
			return true;
		}
		auto raw = view.NumAt(cursor.num_cursor);
		cursor.num_cursor++;
		cursor.pos++;
		switch (subtype) {
		case ext_subtype::INT64:
			sink.OnInt64(int64_t(raw));
			return true;
		case ext_subtype::UINT64:
			sink.OnUInt64(raw);
			return true;
		case ext_subtype::DOUBLE: {
			double value;
			std::memcpy(&value, &raw, sizeof(value));
			sink.OnDouble(value);
			return true;
		}
		default:
			throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
		}
	}
	case tag::VAL_TRUE:
		sink.OnBool(true);
		cursor.pos++;
		return true;
	case tag::VAL_FALSE:
		sink.OnBool(false);
		cursor.pos++;
		return true;
	case tag::VAL_NULL:
		sink.OnNull();
		cursor.pos++;
		return true;
	default:
		return false;
	}
}

// Discarding sink: the skip twin keeps the ladder's stream reads (their bounds checks) but throws
// the decoded value away, so DecodeScalarSlot<ScalarSkipSink> compiles to pure cursor advances.
struct ScalarSkipSink {
	JSONO_ALWAYS_INLINE void OnString(nonstd::string_view) {
	}
	JSONO_ALWAYS_INLINE void OnNumberText(nonstd::string_view) {
	}
	JSONO_ALWAYS_INLINE void OnInt64(int64_t) {
	}
	JSONO_ALWAYS_INLINE void OnUInt64(uint64_t) {
	}
	JSONO_ALWAYS_INLINE void OnDouble(double) {
	}
	JSONO_ALWAYS_INLINE void OnDec60(uint64_t) {
	}
	JSONO_ALWAYS_INLINE void OnBool(bool) {
	}
	JSONO_ALWAYS_INLINE void OnNull() {
	}
};

// Skip a scalar value whose slot is already read at `cursor.pos`. Returns false for
// containers and non-value slots (the caller dispatches those to the out-of-line
// container path). Always-inline: per-value walk loops pay no call for the dominant
// scalar case.
JSONO_ALWAYS_INLINE bool TrySkipScalarFromSlot(const JsonoView &view, uint64_t slot, JsonoCursor &cursor) {
	ScalarSkipSink sink;
	return DecodeScalarSlot(view, slot, cursor, sink);
}

// Skip the value whose first slot is `slot`, already read at `cursor.pos` (a valid slot
// index). Lets callers that have just read the slot — e.g. to inspect its tag —
// skip without paying a second SlotAt for the common scalar cases. Spanned containers
// (non-empty arrays and large objects) jump via their ContainerSpan; small objects and
// empty arrays without spans are walked linearly, guarded by `depth`.
JSONO_ALWAYS_INLINE void SkipValueFastFromSlot(const JsonoView &view, uint64_t slot, JsonoCursor &cursor,
                                               size_t depth = 0) {
	if (TrySkipScalarFromSlot(view, slot, cursor)) {
		return;
	}
	SkipContainerFromSlot(view, slot, cursor, depth);
}

JSONO_ALWAYS_INLINE void SkipValueFast(const JsonoView &view, JsonoCursor &cursor, size_t depth = 0) {
	if (cursor.pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: value position out of bounds");
	}
	SkipValueFastFromSlot(view, view.SlotAt(cursor.pos), cursor, depth);
}

// The container (and malformed-slot) tail of SkipValueFastFromSlot, deliberately
// out-of-line: it recurses through SkipValueFast for small-object walks.
inline void SkipContainerFromSlot(const JsonoView &view, uint64_t slot, JsonoCursor &cursor, size_t depth) {
	auto slot_tag = SlotTag(slot);
	auto payload = SlotPayload(slot);
	switch (slot_tag) {
	case tag::OBJ_START: {
		ContainerSpan span;
		if (view.TryContainerSpan(ContainerId(payload), span)) {
			if (span.slot_span < 2 || size_t(span.slot_span) > view.Slots() - cursor.pos) {
				throw InvalidInputException("malformed JSONO: container slot span out of bounds");
			}
			if (SlotTag(view.SlotAt(cursor.pos + size_t(span.slot_span) - 1)) != tag::OBJ_END) {
				throw InvalidInputException("malformed JSONO: object span does not end with OBJ_END");
			}
			AdvanceCursorBySpan(view, span, cursor);
			return;
		}
		if (depth > JSONO_MAX_NESTING_DEPTH) {
			throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
			                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
		}
		auto key_count = size_t(ContainerChildCount(payload));
		if (key_count > OBJECT_CHECKPOINT_STRIDE) {
			throw InvalidInputException("malformed JSONO: container span missing");
		}
		if (key_count + 1 > view.Slots() - cursor.pos) {
			throw InvalidInputException("malformed JSONO: object child count out of bounds");
		}
		cursor.pos += 1 + key_count;
		for (size_t i = 0; i < key_count; i++) {
			SkipValueFast(view, cursor, depth + 1);
		}
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object walk does not end with OBJ_END");
		}
		cursor.pos++;
		return;
	}
	case tag::ARR_START: {
		ContainerSpan span;
		if (view.TryContainerSpan(ContainerId(payload), span)) {
			if (span.slot_span < 2 || size_t(span.slot_span) > view.Slots() - cursor.pos) {
				throw InvalidInputException("malformed JSONO: container slot span out of bounds");
			}
			if (SlotTag(view.SlotAt(cursor.pos + size_t(span.slot_span) - 1)) != tag::ARR_END) {
				throw InvalidInputException("malformed JSONO: array span does not end with ARR_END");
			}
			AdvanceCursorBySpan(view, span, cursor);
			return;
		}
		if (cursor.pos + 1 < view.Slots() && SlotTag(view.SlotAt(cursor.pos + 1)) == tag::ARR_END) {
			cursor.pos += 2;
			return;
		}
		throw InvalidInputException("malformed JSONO: container span missing");
	}
	case tag::KEY:
	case tag::OBJ_END:
	case tag::ARR_END:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	default:
		throw InvalidInputException("malformed JSONO: unknown slot tag");
	}
}

// Semantic kind of a decoded scalar slot, decoupled from the storage tag/subtype.
enum class JsonoScalarKind : uint8_t { Null, Bool, Int64, UInt64, Double, Dec60, NumberText, String };

// A decoded scalar value. For Dec60 the mantissa/scale components are carried so
// callers can both render byte-exact text and reconstruct the double; for
// NumberText/String, `text` references the source heap bytes (no copy).
struct JsonoScalar {
	JsonoScalarKind kind = JsonoScalarKind::Null;
	bool bool_value = false;
	int64_t int_value = 0;
	uint64_t uint_value = 0;
	double double_value = 0;
	bool dec_negative = false;
	uint64_t dec_mantissa = 0;
	uint64_t dec_scale = 0;
	nonstd::string_view text;
};

// Storing sink: projects each decoded component into a JsonoScalar. The VAL_DEC60 mantissa/
// scale unpack lives here (not in the ladder) so the skip twin pays nothing for it.
struct ScalarDecodeSink {
	JsonoScalar scalar;
	JSONO_ALWAYS_INLINE void OnString(nonstd::string_view text) {
		scalar.kind = JsonoScalarKind::String;
		scalar.text = text;
	}
	JSONO_ALWAYS_INLINE void OnNumberText(nonstd::string_view text) {
		scalar.kind = JsonoScalarKind::NumberText;
		scalar.text = text;
	}
	JSONO_ALWAYS_INLINE void OnInt64(int64_t value) {
		scalar.kind = JsonoScalarKind::Int64;
		scalar.int_value = value;
	}
	JSONO_ALWAYS_INLINE void OnUInt64(uint64_t value) {
		scalar.kind = JsonoScalarKind::UInt64;
		scalar.uint_value = value;
	}
	JSONO_ALWAYS_INLINE void OnDouble(double value) {
		scalar.kind = JsonoScalarKind::Double;
		scalar.double_value = value;
	}
	JSONO_ALWAYS_INLINE void OnDec60(uint64_t packed) {
		scalar.kind = JsonoScalarKind::Dec60;
		scalar.dec_negative = Dec60Negative(packed);
		scalar.dec_mantissa = Dec60Mantissa(packed);
		scalar.dec_scale = Dec60Scale(packed);
	}
	JSONO_ALWAYS_INLINE void OnBool(bool value) {
		scalar.kind = JsonoScalarKind::Bool;
		scalar.bool_value = value;
	}
	JSONO_ALWAYS_INLINE void OnNull() {
		scalar.kind = JsonoScalarKind::Null;
	}
};

// Out-of-line throw for a non-scalar slot reached by DecodeScalarAt: KEY/OBJ_END/ARR_END are a
// non-value slot, anything else (containers the caller failed to handle, unknown tags) is an
// unknown tag. Keeps DecodeScalarAt's two distinct error messages off the hot path.
[[noreturn]] JSONO_COLD inline void ThrowNonScalarSlot(uint64_t slot) {
	switch (SlotTag(slot)) {
	case tag::KEY:
	case tag::OBJ_END:
	case tag::ARR_END:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	default:
		throw InvalidInputException("malformed JSONO: unknown slot tag");
	}
}

// Decode the scalar value at `cursor`, advancing it past the value. The shared DecodeScalarSlot
// ladder is the single owner of the tag/ext-subtype dispatch and stream-cursor accounting; this
// projects it into a JsonoScalar. Throws on container or non-value slots (callers handle
// OBJ_START/ARR_START first).
JSONO_ALWAYS_INLINE JsonoScalar DecodeScalarAt(const JsonoView &view, JsonoCursor &cursor) {
	if (cursor.pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: value position out of bounds");
	}
	auto slot = view.SlotAt(cursor.pos);
	ScalarDecodeSink sink;
	if (!DecodeScalarSlot(view, slot, cursor, sink)) {
		ThrowNonScalarSlot(slot);
	}
	return sink.scalar;
}

// SQL type name for a scalar kind. Numbers (native double, DEC60, and NUMBER-text
// bignums/exponents) all report DOUBLE, matching DuckDB json_type.
inline const char *JsonoScalarTypeName(JsonoScalarKind kind) {
	switch (kind) {
	case JsonoScalarKind::Null:
		return "NULL";
	case JsonoScalarKind::Bool:
		return "BOOLEAN";
	case JsonoScalarKind::Int64:
		return "BIGINT";
	case JsonoScalarKind::UInt64:
		return "UBIGINT";
	case JsonoScalarKind::Double:
	case JsonoScalarKind::Dec60:
	case JsonoScalarKind::NumberText:
		return "DOUBLE";
	case JsonoScalarKind::String:
		return "VARCHAR";
	}
	return nullptr;
}

// Move `value_cursor` (positioned somewhere at or before `target_rank` inside the
// object's value block) forward to the value at `target_rank`. `value_block_base` is
// the cursor at the start of the value block (rank 0) — checkpoint deltas are relative
// to it. The object cursor only moves forward.
inline void MoveObjectValueCursorToRank(const JsonoView &view, const ObjectLayout &layout,
                                        const JsonoCursor &value_block_base, size_t target_rank, size_t &cursor_rank,
                                        JsonoCursor &value_cursor) {
	if (target_rank < cursor_rank) {
		throw InternalException("JSONO object cursor cannot move backwards");
	}
	if (layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && target_rank >= layout.checkpoint_stride) {
		auto checkpoint_rank = target_rank - target_rank % layout.checkpoint_stride;
		if (checkpoint_rank > cursor_rank) {
			auto checkpoint_index = layout.checkpoint_offset + uint32_t(checkpoint_rank / layout.checkpoint_stride - 1);
			auto checkpoint = view.ObjectCheckpointAt(checkpoint_index);
			if (size_t(checkpoint.slot_delta) > layout.after_pos - 1 - layout.value_start) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint slot out of bounds");
			}
			if (value_block_base.string_cursor > view.StringHeapSize() ||
			    size_t(checkpoint.string_delta) > view.StringHeapSize() - value_block_base.string_cursor) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint heap out of bounds");
			}
			if (value_block_base.length_cursor > view.LengthCount() ||
			    size_t(checkpoint.length_delta) > view.LengthCount() - value_block_base.length_cursor) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint lengths out of bounds");
			}
			if (value_block_base.num_cursor > view.NumCount() ||
			    size_t(checkpoint.num_delta) > view.NumCount() - value_block_base.num_cursor) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint nums out of bounds");
			}
			cursor_rank = checkpoint_rank;
			value_cursor.pos = layout.value_start + size_t(checkpoint.slot_delta);
			value_cursor.string_cursor = value_block_base.string_cursor + size_t(checkpoint.string_delta);
			value_cursor.length_cursor = value_block_base.length_cursor + size_t(checkpoint.length_delta);
			value_cursor.num_cursor = value_block_base.num_cursor + size_t(checkpoint.num_delta);
		}
	}
	while (cursor_rank < target_rank) {
		SkipValueFast(view, value_cursor);
		cursor_rank++;
	}
}

} // namespace jsono
} // namespace duckdb
