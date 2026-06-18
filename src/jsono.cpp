#include "jsono.hpp"
#include "jsono_shred.hpp"

#include "duckdb/common/enums/optimizer_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression.hpp"

#include <algorithm>

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

// jsono_version() -> the binary format version this extension reads/writes. A format bump
// leaves the physical STRUCT type unchanged, so a schema epoch derived from the type alone
// cannot see it; folding this into a schema hash triggers re-materialization on a bump
// before a version-mismatched read fails loud.
void JsonoVersionExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, Value::INTEGER(int32_t(jsono::VERSION)));
}

// jsono_storage_type(shreds) -> the shredded storage type's DDL: the 4-BLOB residual plus the
// given shred columns, so a schema can declare a shredded jsono column from a readable shred spec
// (e.g. 'event_name VARCHAR, n BIGINT'). The shred DDL is parsed into (name, type) and re-emitted
// through JsonoShreddedStructType, so the declared column is byte-identical to a value built from
// the same shreds.
void JsonoStorageTypeWithShredsExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	// Parser::ParseColumnList parses the DDL but leaves binder-resolved type aliases (UBIGINT and the
	// other unsigned ints, nested ones included) as unresolved USER types; bind each shred type so a
	// declared storage column matches a value the constructor builds from the same shreds.
	auto binder = Binder::CreateBinder(state.GetContext());
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t shreds) {
		auto columns = Parser::ParseColumnList(shreds.GetString());
		child_list_t<LogicalType> shred_types;
		for (auto &col : columns.Logical()) {
			auto type = col.Type();
			binder->BindLogicalType(type);
			shred_types.emplace_back(col.Name(), type);
		}
		// Canonical shred order (sorted by name) so the DDL matches a value built from the same shreds
		// regardless of the order they are written in here vs the shredding spec.
		std::sort(shred_types.begin(), shred_types.end(),
		          [](const std::pair<string, LogicalType> &a, const std::pair<string, LogicalType> &b) {
			          return a.first < b.first;
		          });
		return StringVector::AddString(result, JsonoShreddedStructType(shred_types).ToString());
	});
}

constexpr const char *JSONO_LAYOUT = "jsono";
// Reserved name of the optional trailing value-complete marker (see JsonoValueCompleteName). The
// double `$` cannot occur in a shred path (object-key paths are `$.key…`); the shred-spec parsers
// reject it outright so no shred can collide.
constexpr const char *JSONO_VALUE_COMPLETE = "$jsono$vc";

// Parse one layout field (top-level child name + its STRUCT type) into `out`. The body must be the
// six-BLOB body struct as field 0; a shredded layout adds its typed shreds as the remaining sibling
// fields. Shreds sit beside `body` (not in a nested struct) deliberately: `body` is then a field
// every two JSONO types share, so DuckDB's by-name struct cast binds between ANY two of them —
// including fully disjoint shred sets — and the optimizer can rewrite every such cast into the
// lossless reconstruct + reshred.
bool TryParseJsonoLayoutField(const string &name, const LogicalType &layout_type, JsonoLayoutType &out) {
	if (name != JSONO_LAYOUT) {
		return false;
	}
	if (layout_type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(layout_type)) {
		return false;
	}
	auto &fields = StructType::GetChildTypes(layout_type);
	if (fields.empty() || fields[0].first != "body" || fields[0].second != JsonoBodyStructType()) {
		return false;
	}
	if (fields.size() == 1) {
		out.kind = JsonoLayoutKind::Plain;
		out.layout_name = name;
		out.layout_type = layout_type;
		out.shreds.clear();
		return true;
	}
	// A field named JSONO_VALUE_COMPLETE (UBIGINT) is the value-complete marker, not a shred, and it
	// is identified by name at ANY position: a set-operation merged type interleaves it among the
	// shreds (CombineStructTypes iterates the left branch's fields — body, marker, shreds — then
	// appends the right branch's unique shreds after the marker), so a positional rule would misread
	// it as a shred. Every other sibling of `body` must be a shred value type; anything else means
	// this is not a JSONO struct (a user struct that merely carries a body-shaped field stays theirs).
	child_list_t<LogicalType> shreds;
	bool has_value_complete = false;
	idx_t value_complete_index = 0;
	for (idx_t i = 1; i < fields.size(); i++) {
		if (fields[i].first == JSONO_VALUE_COMPLETE && fields[i].second.id() == LogicalTypeId::UBIGINT) {
			has_value_complete = true;
			value_complete_index = i;
			continue;
		}
		if (!IsShredValueType(fields[i].second) && !IsShredListType(fields[i].second)) {
			return false;
		}
		shreds.push_back(fields[i]);
	}
	if (shreds.empty()) {
		return false; // body plus only the marker is not a valid shredded value
	}
	out.kind = JsonoLayoutKind::Shredded;
	out.layout_name = name;
	out.layout_type = layout_type;
	out.shreds = std::move(shreds);
	out.has_value_complete = has_value_complete;
	out.value_complete_index = value_complete_index;
	return true;
}

} // namespace

bool TryParseJsonoLayoutType(const LogicalType &type, JsonoLayoutType &out) {
	if (type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(type)) {
		return false;
	}
	auto &children = StructType::GetChildTypes(type);
	if (children.size() != 1) {
		return false; // an ordinary value has exactly one layout field
	}
	return TryParseJsonoLayoutField(children[0].first, children[0].second, out);
}

bool IsJsonoType(const LogicalType &type) {
	JsonoLayoutType layout;
	return TryParseJsonoLayoutType(type, layout) && layout.kind == JsonoLayoutKind::Plain;
}

bool IsShreddedJsonoType(const LogicalType &type) {
	JsonoLayoutType layout;
	return TryParseJsonoLayoutType(type, layout) && layout.kind == JsonoLayoutKind::Shredded;
}

void JsonoRequireExtensionOptimizerForShredded(ClientContext &context, const LogicalType &type,
                                               const string &function_name) {
	if (!IsShreddedJsonoType(type)) {
		return;
	}
	auto &config = DBConfig::GetConfig(context);
	if (config.options.disabled_optimizers.find(OptimizerType::EXTENSION) == config.options.disabled_optimizers.end()) {
		return;
	}
	throw BinderException("%s on shredded JSONO requires the extension optimizer; unsafe STRUCT casts may corrupt "
	                      "JSONO rows when it is disabled",
	                      function_name);
}

LogicalType JsonoResolveJsonoArgument(ClientContext &context, const Expression &arg, const string &function_name,
                                      bool reconstruct_shredded) {
	if (arg.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto &type = arg.return_type;
	JsonoRequireExtensionOptimizerForShredded(context, type, function_name);
	if (IsShreddedJsonoType(type)) {
		return reconstruct_shredded ? JsonoType() : type;
	}
	if (type.id() == LogicalTypeId::SQLNULL || IsJsonoType(type)) {
		return JsonoType();
	}
	throw BinderException("%s: argument must be JSONO", function_name);
}

LogicalType JsonoBodyStructType() {
	child_list_t<LogicalType> children;
	children.emplace_back("slots", LogicalType::BLOB);
	children.emplace_back("key_heap", LogicalType::BLOB);
	children.emplace_back("string_heap", LogicalType::BLOB);
	children.emplace_back("skips", LogicalType::BLOB);
	children.emplace_back("lengths", LogicalType::BLOB);
	children.emplace_back("nums", LogicalType::BLOB);
	return LogicalType::STRUCT(std::move(children));
}

string JsonoLayoutName() {
	return JSONO_LAYOUT;
}

string JsonoValueCompleteName() {
	return JSONO_VALUE_COMPLETE;
}

uint64_t JsonoLayoutHashOf(const LogicalType &type) {
	JsonoLayoutType layout;
	if (!TryParseJsonoLayoutType(type, layout) || layout.kind != JsonoLayoutKind::Shredded) {
		return 0;
	}
	vector<std::pair<std::string, std::string>> signatures;
	signatures.reserve(layout.shreds.size());
	for (auto &shred : layout.shreds) {
		signatures.emplace_back(shred.first, shred.second.ToString());
	}
	return jsono::HashShredManifestSignatures(signatures);
}

LogicalType JsonoShreddedStructType(const child_list_t<LogicalType> &shreds) {
	child_list_t<LogicalType> layout_children;
	layout_children.emplace_back("body", JsonoBodyStructType());
	// Every shredded layout carries the value-complete marker, uniformly. Its value is the canonical
	// layout hash (JsonoLayoutHashOf) of the shred set the row was written under when the row is
	// value-complete, NULL otherwise — so the optimizer can confirm via the marker's min/max zone-map
	// that every scanned row was written under EXACTLY the read type's shred set before dropping the
	// residual COALESCE fallback. A multi-file read that unions narrower shred sets leaves the marker
	// carrying a different hash per file, which keeps the fallback (see CollectShredTotality). It sits
	// right after `body`, BEFORE the shreds, on purpose: DuckDB's set-operation type reconciliation
	// (CombineStructTypes) iterates the left branch's fields then appends the right branch's unique
	// ones, so a marker placed after `body` stays at a fixed position with the shreds contiguous after
	// it in every branch and their merge — whereas a trailing marker would be interleaved (body,
	// shreds_left, marker, shreds_right), reordering the merged type away from the canonical one and
	// breaking the positional struct cast between them.
	layout_children.emplace_back(JSONO_VALUE_COMPLETE, LogicalType::UBIGINT);
	for (auto &shred : shreds) {
		layout_children.push_back(shred);
	}
	child_list_t<LogicalType> top;
	top.emplace_back(JsonoLayoutName(), LogicalType::STRUCT(std::move(layout_children)));
	return LogicalType::STRUCT(std::move(top));
}

LogicalType JsonoRawStructType() {
	child_list_t<LogicalType> layout_children;
	layout_children.emplace_back("body", JsonoBodyStructType());
	child_list_t<LogicalType> top;
	top.emplace_back(JsonoLayoutName(), LogicalType::STRUCT(std::move(layout_children)));
	return LogicalType::STRUCT(std::move(top));
}

LogicalType JsonoType() {
	// The JSONO type alias was dropped: a user-defined alias cannot round-trip through
	// Parquet (stripped on read) or DuckLake (rejected as an unsupported user-defined type),
	// so a stored jsono value is the physical nested STRUCT and nothing more. Function and cast
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
		set.AddFunction(
		    ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, JsonoStorageTypeWithShredsExecute));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_version");
		set.AddFunction(ScalarFunction({}, LogicalType::INTEGER, JsonoVersionExecute));
		loader.RegisterFunction(set);
	}

	RegisterJsonoStructConstructor(loader);
	RegisterJsonoParse(loader);
	RegisterJsonoToJson(loader);
}

} // namespace duckdb
