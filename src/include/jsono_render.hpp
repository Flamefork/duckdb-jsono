#pragma once

#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/cast_helpers.hpp"

#include "string_view.hpp"

#include <cstdio>
#include <string>

namespace duckdb {
namespace jsono {

// Shared render engine: the bare-text scalar render `->>` produces, the JSON-text
// serializer (escape + scalar + container walk), and the DFS walk skeleton it runs on.
// One implementation so quoting, number formatting and the walk's malformed-input checks
// cannot drift between to_json, extract, the optimizer's match/project runtime, the
// shred/transform leaf renders and jsono_entries.

// Escape map for JSON string bytes: 0 = byte passes through verbatim, 'u' = generic
// \u00XX control escape, anything else = second byte of a two-char escape ("\X").
// Entries beyond 0x5c ('\\') are zero-initialized.
static constexpr char JSON_ESCAPES[256] = {
    'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'b', 't', 'n', 'u', 'f', 'r', 'u', 'u', // 0x00-0x0f
    'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', // 0x10-0x1f
    0,   0,   '"', 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   // 0x20-0x2f
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   // 0x30-0x3f
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   // 0x40-0x4f
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   '\\'};              // 0x50-0x5c

// Append `value` as a quoted JSON string. Scans for the next byte that needs escaping and
// bulk-appends the clean run in between, so escape-free strings cost one append.
inline void AppendJsonString(nonstd::string_view value, std::string &out) {
	out.push_back('"');
	auto data = value.data();
	auto size = value.size();
	size_t run_start = 0;
	for (size_t i = 0; i < size; i++) {
		auto escape = JSON_ESCAPES[static_cast<unsigned char>(data[i])];
		if (!escape) {
			continue;
		}
		out.append(data + run_start, i - run_start);
		if (escape == 'u') {
			char buf[8];
			std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(data[i])));
			out.append(buf, 6);
		} else {
			char pair[2] = {'\\', escape};
			out.append(pair, 2);
		}
		run_start = i + 1;
	}
	out.append(data + run_start, size - run_start);
	out.push_back('"');
}

inline void AppendUnsignedText(uint64_t value, std::string &out) {
	char buf[20];
	auto end = buf + sizeof(buf);
	auto begin = NumericHelper::FormatUnsigned(value, end);
	out.append(begin, size_t(end - begin));
}

inline void AppendSignedText(int64_t value, std::string &out) {
	char buf[21];
	auto end = buf + sizeof(buf);
	auto unsigned_value = uint64_t(value);
	if (value < 0) {
		unsigned_value = 0 - unsigned_value;
	}
	auto begin = NumericHelper::FormatUnsigned(unsigned_value, end);
	if (value < 0) {
		*--begin = '-';
	}
	out.append(begin, size_t(end - begin));
}

// Render a leaf scalar as the bare text `->>` produces (no JSON quoting), appended to `out`.
// Returns true for JSON null, where the caller emits SQL NULL rather than the text "null".
inline bool RenderExtractText(const JsonoScalar &scalar, std::string &out) {
	switch (scalar.kind) {
	case JsonoScalarKind::Null:
		return true;
	case JsonoScalarKind::String:
	case JsonoScalarKind::NumberText:
		out.append(scalar.text.data(), scalar.text.size());
		return false;
	case JsonoScalarKind::Int64:
		AppendSignedText(scalar.int_value, out);
		return false;
	case JsonoScalarKind::UInt64:
		AppendUnsignedText(scalar.uint_value, out);
		return false;
	case JsonoScalarKind::Double:
		EmitDouble(scalar.double_value, out);
		return false;
	case JsonoScalarKind::Dec60:
		AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
		return false;
	case JsonoScalarKind::Bool:
		out.append(scalar.bool_value ? "true" : "false");
		return false;
	}
	return true;
}

// Render a scalar as JSON text: strings are quoted and escaped, JSON null is "null".
inline void AppendScalarJsonText(const JsonoScalar &scalar, std::string &out) {
	if (scalar.kind == JsonoScalarKind::String) {
		AppendJsonString(scalar.text, out);
		return;
	}
	if (RenderExtractText(scalar, out)) {
		out.append("null");
	}
}

// DFS walk over the value at `cursor`, advancing it past the value. SINK receives
// structural events; Begin{Member,Element} return a caller-saved token handed back to the
// matching End*, so sinks that track per-child state (e.g. a path string) keep it in the
// recursion frame without an explicit stack. Merge's builder emitters stay separate: they
// interleave survival bookkeeping with the walk and are not plain sinks.
template <class SINK>
void WalkJsonoValue(const JsonoView &view, JsonoCursor &cursor, SINK &sink, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, cursor.pos);
		cursor.pos = layout.value_start;
		sink.BeginObject();
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: expected KEY slot");
			}
			auto member = sink.BeginMember(view.KeyAt(SlotPayload(key_slot)), i);
			WalkJsonoValue(view, cursor, sink, depth + 1);
			sink.EndMember(member);
		}
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		cursor.pos++;
		sink.EndObject();
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, cursor.pos);
		cursor.pos++;
		sink.BeginArray();
		idx_t index = 0;
		while (cursor.pos < end_pos) {
			auto element = sink.BeginElement(index);
			WalkJsonoValue(view, cursor, sink, depth + 1);
			sink.EndElement(element);
			index++;
		}
		cursor.pos = end_pos + 1;
		sink.EndArray();
		return;
	}
	default:
		sink.Scalar(DecodeScalarAt(view, cursor));
		return;
	}
}

// Walk sink that renders the value as JSON text.
struct JsonTextSink {
	explicit JsonTextSink(std::string &out_p) : out(out_p) {
	}

	std::string &out;

	void BeginObject() {
		out.push_back('{');
	}
	void EndObject() {
		out.push_back('}');
	}
	void BeginArray() {
		out.push_back('[');
	}
	void EndArray() {
		out.push_back(']');
	}
	bool BeginMember(nonstd::string_view key, size_t index) {
		if (index) {
			out.push_back(',');
		}
		AppendJsonString(key, out);
		out.push_back(':');
		return false;
	}
	void EndMember(bool) {
	}
	bool BeginElement(idx_t index) {
		if (index) {
			out.push_back(',');
		}
		return false;
	}
	void EndElement(bool) {
	}
	void Scalar(const JsonoScalar &scalar) {
		AppendScalarJsonText(scalar, out);
	}
};

// Serialize the value at `cursor` as JSON text appended to `out`.
inline void AppendJsonValueText(const JsonoView &view, JsonoCursor &cursor, std::string &out, size_t depth) {
	JsonTextSink sink(out);
	WalkJsonoValue(view, cursor, sink, depth);
}

} // namespace jsono
} // namespace duckdb
