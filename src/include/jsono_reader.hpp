#pragma once

#include "jsono.hpp"

#include "duckdb/common/types/vector.hpp"

#include "string_view.hpp"

#include <cstring>

namespace duckdb {
namespace jsono {

struct JsonoBlobRow {
	string_t slots;
	string_t key_heap;
	string_t string_heap;
	string_t skips;
};

struct JsonoVectorData {
	UnifiedVectorFormat struct_fmt;
	UnifiedVectorFormat layout_fmt;
	UnifiedVectorFormat body_fmt;
	UnifiedVectorFormat slots_fmt;
	UnifiedVectorFormat key_heap_fmt;
	UnifiedVectorFormat string_heap_fmt;
	UnifiedVectorFormat skips_fmt;
	const string_t *slots_data = nullptr;
	const string_t *key_heap_data = nullptr;
	const string_t *string_heap_data = nullptr;
	const string_t *skips_data = nullptr;
};

struct ObjectLayout {
	size_t key_start;
	size_t key_count;
	size_t value_start;
	size_t after_pos;
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

// Navigate a jsono value vector (plain or shredded) to its four body blob child vectors:
// top-level layout field [0] -> body [0] -> {slots, key_heap, string_heap, skips}. The body is
// identical for plain and shredded, so the rest of the reader never needs to know which it is.
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
	data.slots_data = UnifiedVectorFormat::GetData<string_t>(data.slots_fmt);
	data.key_heap_data = UnifiedVectorFormat::GetData<string_t>(data.key_heap_fmt);
	data.string_heap_data = UnifiedVectorFormat::GetData<string_t>(data.string_heap_fmt);
	data.skips_data = UnifiedVectorFormat::GetData<string_t>(data.skips_fmt);
}

// Permissive row read for union-merged reconstruction: a dead layout group is expected to be NULL,
// so any missing nested field is treated as an absent row.
inline bool ReadJsonoRow(const JsonoVectorData &data, idx_t row, JsonoBlobRow &out) {
	if (!RowIsValid(data.struct_fmt, row) || !RowIsValid(data.layout_fmt, row) || !RowIsValid(data.body_fmt, row)) {
		return false;
	}
	auto slots_idx = RowIndex(data.slots_fmt, row);
	auto key_heap_idx = RowIndex(data.key_heap_fmt, row);
	auto string_heap_idx = RowIndex(data.string_heap_fmt, row);
	auto skips_idx = RowIndex(data.skips_fmt, row);
	if (!data.slots_fmt.validity.RowIsValid(slots_idx) || !data.key_heap_fmt.validity.RowIsValid(key_heap_idx) ||
	    !data.string_heap_fmt.validity.RowIsValid(string_heap_idx) || !data.skips_fmt.validity.RowIsValid(skips_idx)) {
		return false;
	}
	out.slots = data.slots_data[slots_idx];
	out.key_heap = data.key_heap_data[key_heap_idx];
	out.string_heap = data.string_heap_data[string_heap_idx];
	out.skips = data.skips_data[skips_idx];
	return true;
}

inline void ThrowCorruptJsonoRow(const char *field) {
	throw InvalidInputException(
	    "corrupt JSONO storage: top-level row is valid but %s is NULL; this can be produced by an unsafe "
	    "STRUCT cast when the extension optimizer is disabled",
	    field);
}

inline bool ReadJsonoRowStrict(const JsonoVectorData &data, idx_t row, JsonoBlobRow &out) {
	if (!RowIsValid(data.struct_fmt, row)) {
		return false;
	}
	if (!RowIsValid(data.layout_fmt, row)) {
		ThrowCorruptJsonoRow("layout");
	}
	if (!RowIsValid(data.body_fmt, row)) {
		ThrowCorruptJsonoRow("body");
	}
	auto slots_idx = RowIndex(data.slots_fmt, row);
	auto key_heap_idx = RowIndex(data.key_heap_fmt, row);
	auto string_heap_idx = RowIndex(data.string_heap_fmt, row);
	auto skips_idx = RowIndex(data.skips_fmt, row);
	if (!data.slots_fmt.validity.RowIsValid(slots_idx)) {
		ThrowCorruptJsonoRow("slots blob");
	}
	if (!data.key_heap_fmt.validity.RowIsValid(key_heap_idx)) {
		ThrowCorruptJsonoRow("key_heap blob");
	}
	if (!data.string_heap_fmt.validity.RowIsValid(string_heap_idx)) {
		ThrowCorruptJsonoRow("string_heap blob");
	}
	if (!data.skips_fmt.validity.RowIsValid(skips_idx)) {
		ThrowCorruptJsonoRow("skips blob");
	}
	out.slots = data.slots_data[slots_idx];
	out.key_heap = data.key_heap_data[key_heap_idx];
	out.string_heap = data.string_heap_data[string_heap_idx];
	out.skips = data.skips_data[skips_idx];
	return true;
}

inline JsonoView MakeJsonoView(const JsonoBlobRow &blob) {
	return JsonoView(blob.slots.GetData(), blob.slots.GetSize(), blob.key_heap.GetData(), blob.key_heap.GetSize(),
	                 blob.string_heap.GetData(), blob.string_heap.GetSize(), blob.skips.GetData(),
	                 blob.skips.GetSize());
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

inline ObjectLayout ReadObjectLayout(const JsonoView &view, size_t pos) {
	auto slot = view.SlotAt(pos);
	auto payload = SlotPayload(slot);
	auto key_count = size_t(ContainerChildCount(payload));
	auto span = CheckedContainerSpan(view, pos, tag::OBJ_START);
	auto after_pos = pos + size_t(span.slot_span);
	auto end_pos = after_pos - 1;
	if (SlotTag(view.SlotAt(end_pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object span does not end with OBJ_END");
	}
	auto key_start = pos + 1;
	if (key_count > end_pos - key_start) {
		throw InvalidInputException("malformed JSONO: object child count out of bounds");
	}
	ObjectCheckpointIndex checkpoint_index {uint32_t(ContainerId(payload)), NO_OBJECT_CHECKPOINTS, 0, 0};
	if (key_count > OBJECT_CHECKPOINT_STRIDE) {
		checkpoint_index = view.ObjectCheckpointIndexForContainer(uint32_t(ContainerId(payload)));
	}
	return ObjectLayout {key_start,
	                     key_count,
	                     key_start + key_count,
	                     after_pos,
	                     checkpoint_index.checkpoint_offset,
	                     checkpoint_index.checkpoint_stride,
	                     span.shape_hash};
}

inline size_t ReadArrayEndPos(const JsonoView &view, size_t pos) {
	auto span = CheckedContainerSpan(view, pos, tag::ARR_START);
	auto end_pos = pos + size_t(span.slot_span) - 1;
	if (SlotTag(view.SlotAt(end_pos)) != tag::ARR_END) {
		throw InvalidInputException("malformed JSONO: array span does not end with ARR_END");
	}
	return end_pos;
}

// Skip the value whose first slot is `slot`, already read at `pos` (a valid slot
// index). Lets callers that have just read the slot — e.g. to inspect its tag —
// skip without paying a second SlotAt for the common scalar cases.
inline void SkipValueFastFromSlot(const JsonoView &view, uint64_t slot, size_t &pos, size_t &string_cursor) {
	auto slot_tag = SlotTag(slot);
	auto payload = SlotPayload(slot);
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto span = CheckedContainerSpan(view, pos, tag::OBJ_START);
		if (SlotTag(view.SlotAt(pos + size_t(span.slot_span) - 1)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object span does not end with OBJ_END");
		}
		if (string_cursor > view.StringHeapSize() ||
		    size_t(span.string_byte_span) > view.StringHeapSize() - string_cursor) {
			throw InvalidInputException("malformed JSONO: container string span out of bounds");
		}
		pos += size_t(span.slot_span);
		string_cursor += size_t(span.string_byte_span);
		return;
	}
	case tag::ARR_START: {
		auto span = CheckedContainerSpan(view, pos, tag::ARR_START);
		if (SlotTag(view.SlotAt(pos + size_t(span.slot_span) - 1)) != tag::ARR_END) {
			throw InvalidInputException("malformed JSONO: array span does not end with ARR_END");
		}
		if (string_cursor > view.StringHeapSize() ||
		    size_t(span.string_byte_span) > view.StringHeapSize() - string_cursor) {
			throw InvalidInputException("malformed JSONO: container string span out of bounds");
		}
		pos += size_t(span.slot_span);
		string_cursor += size_t(span.string_byte_span);
		return;
	}
	case tag::VAL_STR_HEAP: {
		auto len = size_t(StringLen(payload));
		(void)view.StringAt(string_cursor, len);
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
			(void)view.StringAt(string_cursor, len);
			string_cursor += len;
		}
		pos += slot_count;
		return;
	}
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		pos++;
		return;
	case tag::KEY:
	case tag::OBJ_END:
	case tag::ARR_END:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	default:
		throw InvalidInputException("malformed JSONO: unknown slot tag");
	}
}

inline void SkipValueFast(const JsonoView &view, size_t &pos, size_t &string_cursor) {
	if (pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: value position out of bounds");
	}
	SkipValueFastFromSlot(view, view.SlotAt(pos), pos, string_cursor);
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

// Decode the scalar value at `pos`, advancing pos and string_cursor past it. This
// is the single owner of the scalar tag/ext-subtype ladder and the trailing-slot /
// string-heap accounting; every reader projects the returned scalar to its sink.
// Throws on container or non-value slots (callers handle OBJ_START/ARR_START first).
JSONO_ALWAYS_INLINE JsonoScalar DecodeScalarAt(const JsonoView &view, size_t &pos, size_t &string_cursor) {
	if (pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: value position out of bounds");
	}
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	auto payload = SlotPayload(slot);
	JsonoScalar scalar;
	switch (slot_tag) {
	case tag::VAL_STR_HEAP: {
		auto len = size_t(StringLen(payload));
		scalar.kind = JsonoScalarKind::String;
		scalar.text = view.StringAt(string_cursor, len);
		string_cursor += len;
		pos++;
		return scalar;
	}
	case tag::VAL_INT60:
		scalar.kind = JsonoScalarKind::Int64;
		scalar.int_value = DecodeInt60(payload);
		pos++;
		return scalar;
	case tag::VAL_DEC60:
		scalar.kind = JsonoScalarKind::Dec60;
		scalar.dec_negative = Dec60Negative(payload);
		scalar.dec_mantissa = Dec60Mantissa(payload);
		scalar.dec_scale = Dec60Scale(payload);
		pos++;
		return scalar;
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		auto slot_count = ExtSlotCount(subtype);
		if (slot_count > view.Slots() - pos) {
			throw InvalidInputException("malformed JSONO: extended scalar payload out of bounds");
		}
		if (subtype == ext_subtype::NUMBER) {
			auto len = size_t(NumberExtLen(slot));
			scalar.kind = JsonoScalarKind::NumberText;
			scalar.text = view.StringAt(string_cursor, len);
			string_cursor += len;
			pos += slot_count;
			return scalar;
		}
		auto raw = view.SlotAt(pos + 1);
		pos += slot_count;
		switch (subtype) {
		case ext_subtype::INT64:
			scalar.kind = JsonoScalarKind::Int64;
			scalar.int_value = int64_t(raw);
			return scalar;
		case ext_subtype::UINT64:
			scalar.kind = JsonoScalarKind::UInt64;
			scalar.uint_value = raw;
			return scalar;
		case ext_subtype::DOUBLE:
			scalar.kind = JsonoScalarKind::Double;
			std::memcpy(&scalar.double_value, &raw, sizeof(scalar.double_value));
			return scalar;
		default:
			throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
		}
	}
	case tag::VAL_TRUE:
		scalar.kind = JsonoScalarKind::Bool;
		scalar.bool_value = true;
		pos++;
		return scalar;
	case tag::VAL_FALSE:
		scalar.kind = JsonoScalarKind::Bool;
		scalar.bool_value = false;
		pos++;
		return scalar;
	case tag::VAL_NULL:
		scalar.kind = JsonoScalarKind::Null;
		pos++;
		return scalar;
	case tag::KEY:
	case tag::OBJ_END:
	case tag::ARR_END:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	default:
		throw InvalidInputException("malformed JSONO: unknown slot tag");
	}
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

inline void MoveObjectValueCursorToRank(const JsonoView &view, const ObjectLayout &layout, size_t object_string_cursor,
                                        size_t target_rank, size_t &cursor_rank, size_t &value_pos,
                                        size_t &value_string_cursor) {
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
			if (object_string_cursor > view.StringHeapSize() ||
			    size_t(checkpoint.string_delta) > view.StringHeapSize() - object_string_cursor) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint heap out of bounds");
			}
			cursor_rank = checkpoint_rank;
			value_pos = layout.value_start + size_t(checkpoint.slot_delta);
			value_string_cursor = object_string_cursor + size_t(checkpoint.string_delta);
		}
	}
	while (cursor_rank < target_rank) {
		SkipValueFast(view, value_pos, value_string_cursor);
		cursor_rank++;
	}
}

} // namespace jsono
} // namespace duckdb
