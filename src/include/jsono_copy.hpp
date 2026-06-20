#pragma once

#include "jsono_reader.hpp"
#include "jsono_writer.hpp"

namespace duckdb {

using namespace jsono;

// How a raw scalar slot's value bytes are encoded, derived from the slot tag and the
// EXT subtype. The three outcomes drive both the verbatim builder copy and the
// standalone-blob store: a num word, a length+heap pair, or nothing (literal). The two
// throws are the fail-loud guard against forged/corrupt raw slots — they must NOT be
// dropped (unlike the trusted LWWScalarValue accessors which skip validation).
enum class RawScalarValueKind { Number, LengthHeap, Literal };

JSONO_ALWAYS_INLINE RawScalarValueKind ClassifyRawScalarSlot(uint64_t slot) {
	switch (SlotTag(slot)) {
	case tag::VAL_STR_HEAP:
		return RawScalarValueKind::LengthHeap;
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		if (subtype == ext_subtype::NUMBER) {
			return RawScalarValueKind::LengthHeap;
		}
		if (subtype >= ext_subtype::COUNT) {
			throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
		}
		return RawScalarValueKind::Number;
	}
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
		return RawScalarValueKind::Number;
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		return RawScalarValueKind::Literal;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

// Copy a scalar value at `cursor` byte-for-byte into the builder, advancing the
// cursor. Equivalent to decoding the scalar and re-emitting it, but without
// reconstructing the value: every scalar encoding round-trips to the identical
// slot + stream entries (INT60/DEC60 re-store the same nums word; ext INT64/
// UINT64/DOUBLE re-store the same raw bits; STR_HEAP/NUMBER re-append the same
// heap bytes and length entry), so copying the source slot and stream entries is
// the same result with none of the decode/re-classify work. Caller must have
// excluded container slots. JSONO_ALWAYS_INLINE keeps the scalar leaf folded into
// EmitValueVerbatim — without it the move to header (vague linkage) makes the inliner
// tail-branch instead, regressing the hot merge value loop.
JSONO_ALWAYS_INLINE void EmitScalarVerbatim(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder) {
	auto slot = view.SlotAt(cursor.pos);
	switch (ClassifyRawScalarSlot(slot)) {
	case RawScalarValueKind::LengthHeap: {
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
	case RawScalarValueKind::Number:
		builder.slots.push_back(slot);
		builder.nums.push_back(view.NumAt(cursor.num_cursor));
		cursor.num_cursor++;
		cursor.pos++;
		return;
	case RawScalarValueKind::Literal:
		builder.slots.push_back(slot);
		cursor.pos++;
		return;
	}
}

inline void EmitValueVerbatim(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder, size_t depth) {
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

} // namespace duckdb
