#include "jsono.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
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

// Retype between the raw STRUCT(4 BLOB) and JSONO. The two are physically
// identical and differ only by the JSONO alias, so this is a no-op conversion.
// It cannot use DefaultCasts::ReinterpretCast: Vector::Reinterpret is defined
// only for non-nested types (it shares the auxiliary child buffer by pointer
// without retyping it), and a debug build asserts on a nested reinterpret.
// Reinterpreting the scalar BLOB children individually is allowed; the parent's
// vector type and validity carry across unchanged so a constant input (e.g. a
// folded literal cast) stays constant.
bool JsonoStructRetypeCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, ConstantVector::IsNull(source));
	} else {
		source.Flatten(count);
		result.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::Validity(result) = FlatVector::Validity(source);
	}
	auto &source_children = StructVector::GetEntries(source);
	auto &result_children = StructVector::GetEntries(result);
	for (idx_t i = 0; i < source_children.size(); i++) {
		result_children[i]->Reinterpret(*source_children[i]);
	}
	return true;
}

} // namespace

bool IsJsonoType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	if (type.HasAlias() && type.GetAlias() == JSONO_TYPE_NAME) {
		return true;
	}
	auto &children = StructType::GetChildTypes(type);
	return children.size() == 4 && children[0].first == "jsono_slots" && children[0].second == LogicalType::BLOB &&
	       children[1].first == "jsono_key_heap" && children[1].second == LogicalType::BLOB &&
	       children[2].first == "jsono_string_heap" && children[2].second == LogicalType::BLOB &&
	       children[3].first == "jsono_skips" && children[3].second == LogicalType::BLOB;
}

LogicalType JsonoType() {
	child_list_t<LogicalType> children;
	children.emplace_back("jsono_slots", LogicalType::BLOB);
	children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	children.emplace_back("jsono_skips", LogicalType::BLOB);
	auto t = LogicalType::STRUCT(std::move(children));
	t.SetAlias(JSONO_TYPE_NAME);
	return t;
}

void RegisterJsonoType(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	loader.RegisterType(JSONO_TYPE_NAME, jsono_type);

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

	RegisterJsonoStructConstructor(loader);
	RegisterJsonoParse(loader);
	RegisterJsonoToJson(loader);

	// JSONO is physically STRUCT(slots, key_heap, string_heap, skips) with an
	// alias. Reading the column back from a plain Parquet file drops the
	// alias — we still want transform/to_json on it. Register a no-op
	// reinterpret cast both ways so the structural shape implicitly converts.
	child_list_t<LogicalType> raw_children;
	raw_children.emplace_back("jsono_slots", LogicalType::BLOB);
	raw_children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	raw_children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	raw_children.emplace_back("jsono_skips", LogicalType::BLOB);
	auto raw_struct = LogicalType::STRUCT(raw_children);
	loader.RegisterCastFunction(raw_struct, jsono_type, JsonoStructRetypeCast, 1);
	loader.RegisterCastFunction(jsono_type, raw_struct, JsonoStructRetypeCast, 1);
}

} // namespace duckdb
