#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "string_view.hpp"

#include <cstdio>
#include <string>

namespace duckdb {

namespace {

using namespace jsono;

//===--------------------------------------------------------------------===//
// Reader: slots blob → JSON text (round-trip). Walks JsonoView and emits JSON
// text into a std::string. Spec-driven extraction (jsono_transform) walks
// JsonoView directly without the intermediate string.
//===--------------------------------------------------------------------===//

void EmitJsonValue(const JsonoView &r, size_t &pos, size_t &string_cursor, std::string &out, size_t depth);

void EscapeJsonString(nonstd::string_view s, std::string &out) {
	out.push_back('"');
	for (char c : s) {
		switch (c) {
		case '"':
			out.append("\\\"");
			break;
		case '\\':
			out.append("\\\\");
			break;
		case '\b':
			out.append("\\b");
			break;
		case '\f':
			out.append("\\f");
			break;
		case '\n':
			out.append("\\n");
			break;
		case '\r':
			out.append("\\r");
			break;
		case '\t':
			out.append("\\t");
			break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
				out.append(buf);
			} else {
				out.push_back(c);
			}
		}
	}
	out.push_back('"');
}

void EmitJsonValue(const JsonoView &r, size_t &pos, size_t &string_cursor, std::string &out, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto t = SlotTag(r.SlotAt(pos));

	switch (t) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(r, pos);
		auto key_start = layout.key_start;
		auto key_count = layout.key_count;
		auto val_cursor = layout.value_start;
		out.push_back('{');
		for (size_t i = 0; i < key_count; i++) {
			if (i) {
				out.push_back(',');
			}
			auto key_slot = r.SlotAt(key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			EscapeJsonString(r.KeyAt(SlotPayload(key_slot)), out);
			out.push_back(':');
			EmitJsonValue(r, val_cursor, string_cursor, out, depth + 1);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		out.push_back('}');
		break;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(r, pos);
		pos++;
		out.push_back('[');
		bool first = true;
		while (pos < end_pos) {
			if (!first) {
				out.push_back(',');
			}
			first = false;
			EmitJsonValue(r, pos, string_cursor, out, depth + 1);
		}
		pos = end_pos + 1;
		out.push_back(']');
		break;
	}
	default: {
		auto scalar = DecodeScalarAt(r, pos, string_cursor);
		switch (scalar.kind) {
		case JsonoScalarKind::String:
			EscapeJsonString(scalar.text, out);
			break;
		case JsonoScalarKind::Int64:
			out.append(std::to_string(scalar.int_value));
			break;
		case JsonoScalarKind::UInt64:
			out.append(std::to_string(scalar.uint_value));
			break;
		case JsonoScalarKind::Double:
			EmitDouble(scalar.double_value, out);
			break;
		case JsonoScalarKind::Dec60:
			AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			break;
		case JsonoScalarKind::NumberText:
			out.append(scalar.text.data(), scalar.text.size());
			break;
		case JsonoScalarKind::Bool:
			out.append(scalar.bool_value ? "true" : "false");
			break;
		case JsonoScalarKind::Null:
			out.append("null");
			break;
		}
		break;
	}
	}
}

void WriteJsonoAsJson(Vector &input, idx_t count, Vector &result) {
	JsonoVectorData input_data;
	InitJsonoVectorData(input, count, input_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);

	std::string buf;

	for (idx_t i = 0; i < count; i++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRowStrict(input_data, i, blob)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		JsonoView reader = MakeJsonoView(blob);
		if (!reader.ParseHeader() || reader.Slots() == 0) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		buf.clear();
		size_t pos = 0;
		size_t string_cursor = 0;
		EmitJsonValue(reader, pos, string_cursor, buf, 0);

		result_data[i] = StringVector::AddString(result, buf.data(), buf.size());
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoToJsonExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	WriteJsonoAsJson(args.data[0], args.size(), result);
}

bool JsonoCastToJson(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	WriteJsonoAsJson(source, count, result);
	return true;
}

// Cast a shredded JSONO value to its JSON text: reconstruct the plain document first, then serialize
// (same output as the optimizer's jsono_overlay + JsonoCastToJson path). Without this, a shredded
// struct->VARCHAR would fall to the default struct text and leak the physical "#fp" field names.
bool JsonoShreddedToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	Vector plain(JsonoType(), count);
	JsonoReconstructToPlain(source, count, plain);
	WriteJsonoAsJson(plain, count, result);
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

// Wildcard STRUCT(any)->VARCHAR bind: a shredded JSONO value serializes to JSON text; any other
// struct declines (a null-function BoundCastInfo), so the cast set falls through to the default
// struct->VARCHAR text cast — byte-identical to stock DuckDB for non-jsono structs. The exact
// plain-JSONO->VARCHAR entry (registered below) is matched before this wildcard, so a plain value
// is unaffected. There is no equivalent for ->JSON: core json owns STRUCT(any)->JSON and the cast
// registry keeps the first registration, so shredded ::JSON / to_json stay the optimizer's job.
BoundCastInfo JsonoStructToVarcharCastBind(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	(void)input;
	(void)target;
	if (IsShreddedJsonoType(source)) {
		return BoundCastInfo(JsonoShreddedToVarcharCast);
	}
	return BoundCastInfo(nullptr);
}

} // namespace

void RegisterJsonoToJson(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{
		ScalarFunction f("to_json", {jsono_type}, LogicalType::JSON(), JsonoToJsonExecute);
		f.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		loader.RegisterFunction(f);
	}
	loader.RegisterCastFunction(jsono_type, LogicalType::JSON(), BoundCastInfo(JsonoCastToJson), -1);
	loader.RegisterCastFunction(jsono_type, LogicalType::VARCHAR, BoundCastInfo(JsonoCastToJson), -1);
	// Bind-correct shredded->VARCHAR (declines for non-jsono structs). Explicit-only (-1): struct is
	// not implicitly castable to VARCHAR, and this entry must not make it so.
	loader.RegisterCastFunction(LogicalType::STRUCT({{"any", LogicalType::ANY}}), LogicalType::VARCHAR,
	                            JsonoStructToVarcharCastBind, -1);
}

} // namespace duckdb
