#pragma once

#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"

#include "string_view.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace duckdb {
namespace jsono {

enum class JsonoScalarPrimitive : uint8_t { Varchar, Bigint, Ubigint, Double, Boolean };

inline bool TypeToJsonoScalarPrimitive(const LogicalType &type, JsonoScalarPrimitive &out) {
	switch (type.id()) {
	case LogicalTypeId::VARCHAR:
		// A JSON-aliased VARCHAR carries a JSON document, not a scalar string value.
		if (type.HasAlias() && type.GetAlias() == LogicalType::JSON_TYPE_NAME) {
			return false;
		}
		out = JsonoScalarPrimitive::Varchar;
		return true;
	case LogicalTypeId::BIGINT:
		out = JsonoScalarPrimitive::Bigint;
		return true;
	case LogicalTypeId::UBIGINT:
		out = JsonoScalarPrimitive::Ubigint;
		return true;
	case LogicalTypeId::DOUBLE:
		out = JsonoScalarPrimitive::Double;
		return true;
	case LogicalTypeId::BOOLEAN:
		out = JsonoScalarPrimitive::Boolean;
		return true;
	default:
		return false;
	}
}

inline JsonoScalarPrimitive JsonoScalarPrimitiveFromType(const LogicalType &type, const char *context) {
	JsonoScalarPrimitive primitive;
	if (TypeToJsonoScalarPrimitive(type, primitive)) {
		return primitive;
	}
	throw InternalException("%s: non-scalar primitive type '%s'", context, type.ToString());
}

inline LogicalType JsonoScalarPrimitiveLogicalType(JsonoScalarPrimitive primitive) {
	switch (primitive) {
	case JsonoScalarPrimitive::Bigint:
		return LogicalType::BIGINT;
	case JsonoScalarPrimitive::Ubigint:
		return LogicalType::UBIGINT;
	case JsonoScalarPrimitive::Double:
		return LogicalType::DOUBLE;
	case JsonoScalarPrimitive::Varchar:
		return LogicalType::VARCHAR;
	case JsonoScalarPrimitive::Boolean:
		return LogicalType::BOOLEAN;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline const char *JsonoScalarPrimitiveTypeName(JsonoScalarPrimitive primitive) {
	switch (primitive) {
	case JsonoScalarPrimitive::Bigint:
		return "BIGINT";
	case JsonoScalarPrimitive::Ubigint:
		return "UBIGINT";
	case JsonoScalarPrimitive::Double:
		return "DOUBLE";
	case JsonoScalarPrimitive::Varchar:
		return "VARCHAR";
	case JsonoScalarPrimitive::Boolean:
		return "BOOLEAN";
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline bool JsonoScalarPrimitiveStoresReadCopy(JsonoScalarPrimitive primitive) {
	// Only VARCHAR renders the `->>` text of non-string scalars while leaving them in the residual.
	return primitive == JsonoScalarPrimitive::Varchar;
}

inline bool JsonoScalarFitsPrimitive(const JsonoScalar &scalar, JsonoScalarPrimitive primitive) {
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar:
		return scalar.kind == JsonoScalarKind::String;
	case JsonoScalarPrimitive::Bigint:
		return scalar.kind == JsonoScalarKind::Int64;
	case JsonoScalarPrimitive::Ubigint:
		return scalar.kind == JsonoScalarKind::UInt64 ||
		       (scalar.kind == JsonoScalarKind::Int64 && scalar.int_value >= 0);
	case JsonoScalarPrimitive::Double:
		return scalar.kind == JsonoScalarKind::Double;
	case JsonoScalarPrimitive::Boolean:
		return scalar.kind == JsonoScalarKind::Bool;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline JsonoScalar JsonoScalarFromPrimitiveVector(JsonoScalarPrimitive primitive, const UnifiedVectorFormat &fmt,
                                                  idx_t idx) {
	JsonoScalar scalar;
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar: {
		auto &value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		scalar.kind = JsonoScalarKind::String;
		scalar.text = nonstd::string_view(value.GetData(), value.GetSize());
		return scalar;
	}
	case JsonoScalarPrimitive::Bigint:
		scalar.kind = JsonoScalarKind::Int64;
		scalar.int_value = UnifiedVectorFormat::GetData<int64_t>(fmt)[idx];
		return scalar;
	case JsonoScalarPrimitive::Ubigint:
		scalar.kind = JsonoScalarKind::UInt64;
		scalar.uint_value = UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx];
		return scalar;
	case JsonoScalarPrimitive::Double:
		scalar.kind = JsonoScalarKind::Double;
		scalar.double_value = UnifiedVectorFormat::GetData<double>(fmt)[idx];
		return scalar;
	case JsonoScalarPrimitive::Boolean:
		scalar.kind = JsonoScalarKind::Bool;
		scalar.bool_value = UnifiedVectorFormat::GetData<bool>(fmt)[idx];
		return scalar;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline uint64_t JsonoPrimitiveVectorPayloadBytes(JsonoScalarPrimitive primitive, const UnifiedVectorFormat &fmt,
                                                 idx_t idx) {
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar:
		return UnifiedVectorFormat::GetData<string_t>(fmt)[idx].GetSize();
	case JsonoScalarPrimitive::Bigint:
	case JsonoScalarPrimitive::Ubigint:
	case JsonoScalarPrimitive::Double:
		return 8;
	case JsonoScalarPrimitive::Boolean:
		return 1;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline nonstd::string_view JsonoPrimitiveVectorText(JsonoScalarPrimitive primitive, const UnifiedVectorFormat &fmt,
                                                    idx_t idx, std::string &scratch) {
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar: {
		auto &value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		return nonstd::string_view(value.GetData(), value.GetSize());
	}
	case JsonoScalarPrimitive::Bigint:
		scratch.clear();
		AppendSignedText(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx], scratch);
		return scratch;
	case JsonoScalarPrimitive::Ubigint:
		scratch.clear();
		AppendUnsignedText(UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx], scratch);
		return scratch;
	case JsonoScalarPrimitive::Double:
		scratch.clear();
		EmitDouble(UnifiedVectorFormat::GetData<double>(fmt)[idx], scratch);
		return scratch;
	case JsonoScalarPrimitive::Boolean:
		scratch = UnifiedVectorFormat::GetData<bool>(fmt)[idx] ? "true" : "false";
		return scratch;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline void EmitJsonoPrimitiveVectorValue(JsonoBuilder &builder, JsonoScalarPrimitive primitive,
                                          const UnifiedVectorFormat &fmt, idx_t idx) {
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar: {
		auto value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
		return;
	}
	case JsonoScalarPrimitive::Bigint:
		builder.EmitInt(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx]);
		return;
	case JsonoScalarPrimitive::Ubigint:
		builder.EmitUInt(UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx]);
		return;
	case JsonoScalarPrimitive::Double:
		builder.EmitDouble(UnifiedVectorFormat::GetData<double>(fmt)[idx]);
		return;
	case JsonoScalarPrimitive::Boolean:
		builder.EmitBool(UnifiedVectorFormat::GetData<bool>(fmt)[idx]);
		return;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline uint64_t JsonoPrimitiveVectorValueBits(JsonoScalarPrimitive primitive, const UnifiedVectorFormat &fmt, idx_t idx,
                                              std::string *text_out) {
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar: {
		if (!text_out) {
			throw InternalException("jsono: missing text output for VARCHAR primitive bits");
		}
		auto value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		if (value.GetSize() > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: string value exceeds storage limits");
		}
		text_out->assign(value.GetData(), value.GetSize());
		return 0;
	}
	case JsonoScalarPrimitive::Bigint:
		return uint64_t(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx]);
	case JsonoScalarPrimitive::Ubigint:
		return UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx];
	case JsonoScalarPrimitive::Double: {
		auto value = UnifiedVectorFormat::GetData<double>(fmt)[idx];
		if (!std::isfinite(value)) {
			throw InvalidInputException("jsono: cannot store non-finite double value (NaN/Infinity)");
		}
		uint64_t out;
		std::memcpy(&out, &value, sizeof(out));
		return out;
	}
	case JsonoScalarPrimitive::Boolean:
		return UnifiedVectorFormat::GetData<bool>(fmt)[idx] ? 1 : 0;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

inline void WriteJsonoStringLane(Vector &out, idx_t row, nonstd::string_view value) {
	FlatVector::Validity(out).SetValid(row);
	FlatVector::GetData<string_t>(out)[row] = StringVector::AddString(out, value.data(), value.size());
}

inline void WriteJsonoBigintLane(Vector &out, idx_t row, int64_t value) {
	FlatVector::Validity(out).SetValid(row);
	FlatVector::GetData<int64_t>(out)[row] = value;
}

inline void WriteJsonoUbigintLane(Vector &out, idx_t row, uint64_t value) {
	FlatVector::Validity(out).SetValid(row);
	FlatVector::GetData<uint64_t>(out)[row] = value;
}

inline void WriteJsonoDoubleLane(Vector &out, idx_t row, double value) {
	FlatVector::Validity(out).SetValid(row);
	FlatVector::GetData<double>(out)[row] = value;
}

inline void WriteJsonoBooleanLane(Vector &out, idx_t row, bool value) {
	FlatVector::Validity(out).SetValid(row);
	FlatVector::GetData<bool>(out)[row] = value;
}

inline void WriteJsonoRenderedTextLane(Vector &out, idx_t row, const JsonoScalar &scalar, std::string &scratch) {
	scratch.clear();
	RenderExtractText(scalar, scratch);
	WriteJsonoStringLane(out, row, nonstd::string_view(scratch.data(), scratch.size()));
}

inline void WriteJsonoFittingScalarLane(Vector &out, idx_t row, const JsonoScalar &scalar,
                                        JsonoScalarPrimitive primitive, std::string &scratch) {
	switch (primitive) {
	case JsonoScalarPrimitive::Varchar:
		WriteJsonoRenderedTextLane(out, row, scalar, scratch);
		return;
	case JsonoScalarPrimitive::Bigint:
		WriteJsonoBigintLane(out, row, scalar.int_value);
		return;
	case JsonoScalarPrimitive::Ubigint:
		WriteJsonoUbigintLane(out, row,
		                      scalar.kind == JsonoScalarKind::UInt64 ? scalar.uint_value : uint64_t(scalar.int_value));
		return;
	case JsonoScalarPrimitive::Double:
		WriteJsonoDoubleLane(out, row, scalar.double_value);
		return;
	case JsonoScalarPrimitive::Boolean:
		WriteJsonoBooleanLane(out, row, scalar.bool_value);
		return;
	}
	throw InternalException("jsono: unhandled scalar primitive");
}

} // namespace jsono
} // namespace duckdb
