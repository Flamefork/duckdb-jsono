#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "string_view.hpp"

#include <string>

namespace duckdb {

namespace {

using namespace jsono;

//===--------------------------------------------------------------------===//
// Reader: slots blob → JSON text (round-trip) via the shared render engine
// (jsono_render.hpp). Spec-driven extraction (jsono_transform) walks JsonoView
// directly without the intermediate string.
//===--------------------------------------------------------------------===//

void WriteJsonoAsJson(Vector &input, idx_t count, Vector &result) {
	// `input` is plain JSONO, which never writes a shred manifest: any manifest entry here is
	// the trace of a raw struct cast that dropped the shreds, and the serialization would
	// silently omit the stripped values — the reader fails loud instead.
	JsonoRowReader row_reader;
	row_reader.Init(input, count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);

	std::string buf;

	JsonoView reader;
	for (idx_t i = 0; i < count; i++) {
		JsonoBlobRow blob;
		if (row_reader.Read(i, blob, reader) != JsonoRowState::Value) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		buf.clear();
		JsonoCursor cursor;
		AppendJsonValueText(reader, cursor, buf, 0);

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
	for (auto &target : {LogicalType::JSON(), LogicalType(LogicalType::VARCHAR)}) {
		loader.RegisterCastFunction(jsono_type, target, BoundCastInfo(JsonoCastToJson), -1);
	}
	// Bind-correct shredded->VARCHAR (declines for non-jsono structs). Explicit-only (-1): struct is
	// not implicitly castable to VARCHAR, and this entry must not make it so.
	loader.RegisterCastFunction(LogicalType::STRUCT({{"any", LogicalType::ANY}}), LogicalType::VARCHAR,
	                            JsonoStructToVarcharCastBind, -1);
}

} // namespace duckdb
