#include "jsono.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

void RegisterJsonoParse(ExtensionLoader &loader);
void RegisterJsonoStructConstructor(ExtensionLoader &loader);
void RegisterJsonoToJson(ExtensionLoader &loader);

namespace {

void JsonoNullExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

void JsonoIdentityExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	result.Reference(args.data[0]);
}

// jsono_storage_type() -> the DDL string of the physical STRUCT that a jsono value is.
// DuckLake rejects user-defined type aliases (the reason jsono carries none), so writers
// declare storage columns with this struct; exposing it here keeps the layout owned by the
// extension.
void JsonoStorageTypeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, Value(JsonoRawStructType().ToString()));
}

// jsono_storage_type(lanes) -> the shredded storage type: the 4-BLOB prefix plus the
// given lane columns, so a schema can declare a shredded jsono column from a readable lane
// spec (e.g. 'event_name VARCHAR, goals MAP(VARCHAR, BIGINT)') without naming the blobs.
void JsonoStorageTypeWithLanesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto base = JsonoRawStructType().ToString();
	auto prefix = base.substr(0, base.size() - 1); // drop the trailing ')'
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t lanes) {
		auto out = prefix + ", " + lanes.GetString() + ")";
		return StringVector::AddString(result, out);
	});
}

} // namespace

bool IsJsonoType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	// A jsono value is recognised purely by structure: the four jsono_* BLOB children.
	// There is no type alias to short-circuit on — a stored value (DuckDB-native, Parquet
	// or DuckLake) is just this STRUCT. JsonoRawStructType is the single field contract.
	return StructType::GetChildTypes(type) == StructType::GetChildTypes(JsonoRawStructType());
}

bool HasJsonoBlobPrefix(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto canonical = JsonoRawStructType();
	auto &children = StructType::GetChildTypes(type);
	auto &canonical_children = StructType::GetChildTypes(canonical);
	if (children.size() < canonical_children.size()) {
		return false;
	}
	for (idx_t i = 0; i < canonical_children.size(); i++) {
		if (children[i] != canonical_children[i]) {
			return false;
		}
	}
	return true;
}

bool IsShreddedJsonoType(const LogicalType &type) {
	return HasJsonoBlobPrefix(type) &&
	       StructType::GetChildTypes(type).size() > StructType::GetChildTypes(JsonoRawStructType()).size();
}

LogicalType JsonoRawStructType() {
	child_list_t<LogicalType> children;
	children.emplace_back("jsono_slots", LogicalType::BLOB);
	children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	children.emplace_back("jsono_skips", LogicalType::BLOB);
	return LogicalType::STRUCT(std::move(children));
}

LogicalType JsonoType() {
	// The JSONO type alias was dropped: a user-defined alias cannot round-trip through
	// Parquet (stripped on read) or DuckLake (rejected as an unsupported user-defined type),
	// so a stored jsono value is the physical STRUCT and nothing more. Function and cast
	// dispatch is structural (IsJsonoType). Kept as the accessor the registrations read
	// against; identical to JsonoRawStructType().
	return JsonoRawStructType();
}

void RegisterJsonoType(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();

	{
		ScalarFunctionSet set("jsono");
		set.AddFunction(ScalarFunction({LogicalType::SQLNULL}, jsono_type, JsonoNullExecute));
		set.AddFunction(ScalarFunction({jsono_type}, jsono_type, JsonoIdentityExecute));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("try_jsono");
		set.AddFunction(ScalarFunction({LogicalType::SQLNULL}, jsono_type, JsonoNullExecute));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_storage_type");
		set.AddFunction(ScalarFunction({}, LogicalType::VARCHAR, JsonoStorageTypeExecute));
		set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, JsonoStorageTypeWithLanesExecute));
		loader.RegisterFunction(set);
	}

	RegisterJsonoStructConstructor(loader);
	RegisterJsonoParse(loader);
	RegisterJsonoToJson(loader);
}

} // namespace duckdb
