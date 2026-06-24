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

// jsono_storage_type(shreds) -> the shredded storage type's DDL: the 6-BLOB residual plus the
// given shred columns, so a schema can declare a shredded jsono column from a readable shred spec
// (e.g. 'event_name VARCHAR, n BIGINT'). The shred DDL is parsed into (name, type) and re-emitted
// through JsonoShreddedStructType, so the declared column is byte-identical to a value built from
// the same shreds. Shred specs are rejected by the same validation the constructor uses, so a declared
// column always matches some value the constructor can produce.
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
			JsonoValidateShredField(col.Name(), type);
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
// Name of the nested STRUCT holding the shred set: it sits beside `body` inside the layout struct.
// Shredded-ness is exactly the presence of this field.
constexpr const char *JSONO_SHREDS = "shreds";
// Reserved name of the shred-set marker, the first field inside `shreds`. It is a BIGINT hash of the
// shred set (paths + types) — schema identity across files, and the mandatory shared member that lets
// a by-name struct cast bind between any two shred sets (even disjoint ones). The double `$` cannot
// occur in a shred path (object-key paths are `$.key…`); the shred-spec parsers reject it so no shred
// can collide.
constexpr const char *JSONO_SHRED_SET = "$jsono$set";
// The two children of a scalar shred pair: the typed value lane and the per-shred completeness flag
// (TINYINT, 1 = the lane holds the full `->>` answer; 0 = the value spilled to the residual).
constexpr const char *JSONO_SHRED_VALUE = "value";
constexpr const char *JSONO_SHRED_COMPLETE = "complete";

// A scalar shred field is a pair STRUCT(value <scalar>, complete TINYINT); an array shred field is a
// LIST as-is. Unwrap either into its logical VALUE type (the scalar type / the LIST type) so the
// rest of the layout machinery (hash, casts, reconstruct) sees the shred's logical type, not the
// pair wrapper. Returns false when `type` is neither a scalar pair nor a list shred.
// `complete` is matched as IsIntegral (any integer width), not strictly TINYINT, for the same reason
// the shred-set marker is (see TryParseJsonoLayoutField): a value round-tripped through a generic
// value->SQL->value path (DuckLake inlined-data INSERT) re-parses the bare 1/0 flag as INTEGER/BIGINT
// instead of TINYINT, and a strict-TINYINT rule would then fail to recognise the pair — dropping the
// whole value to a plain user struct, so to_json leaks the physical layout and `->>` reads NULL. The
// flag is a metadata 1/0, so its width carries no semantics (unlike `value`, whose type IS the shred
// type); tolerating the width keeps recognition. No ambiguity with user data: `value` must be a scalar
// shred type, so a {value, complete} pair can only be the writer's wrapper, never a JSON object.
bool UnwrapShredFieldType(const LogicalType &type, LogicalType &value_type) {
	if (IsShredListType(type)) {
		value_type = type;
		return true;
	}
	if (type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(type)) {
		return false;
	}
	auto &pair = StructType::GetChildTypes(type);
	if (pair.size() != 2 || pair[0].first != JSONO_SHRED_VALUE || pair[1].first != JSONO_SHRED_COMPLETE ||
	    !pair[1].second.IsIntegral() || !IsShredValueType(pair[0].second)) {
		return false;
	}
	value_type = pair[0].second;
	return true;
}

// Parse one layout field (top-level child name + its STRUCT type) into `out`. The body must be the
// six-BLOB body struct as field 0; a shredded layout adds a `shreds` STRUCT sibling holding the
// shred-set marker plus one field per shred. The single layout name (`jsono` for plain and shredded)
// is deliberate: DuckDB reconciles set-operation branch types by field name (CombineStructTypes), so
// differently-shredded branches merge into one shredded type whose `shreds` is the union of the
// branches' shreds, and the marker stays the left branch's `shreds` field 0. The nested `shreds`
// struct keeps the marker contiguous with the shreds (a flat sibling layout interleaved it among
// the shreds under set-ops), and the marker is the mandatory shared member that lets a by-name cast
// bind between ANY two shred sets — including fully disjoint ones.
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
	// Shredded: exactly `body` plus a `shreds` STRUCT. The set-op merge keeps the layout struct at
	// these two by-name fields, so a third sibling means this is not a JSONO struct (a user struct
	// that merely carries a body-shaped field stays theirs).
	if (fields.size() != 2 || fields[1].first != JSONO_SHREDS) {
		return false;
	}
	auto &shreds_type = fields[1].second;
	if (shreds_type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(shreds_type)) {
		return false;
	}
	// Inside `shreds`: the marker is field 0 (any integer width — a value round-tripped through a
	// generic value->SQL->value path, e.g. DuckLake inlined-data INSERT, re-parses the bare-integer
	// marker as BIGINT/HUGEINT instead of BIGINT, but the reserved name still identifies it). It must
	// sit at field 0, not merely be present by name: every writer emits it first and every set-op
	// merge keeps it first (it is the left branch's `shreds` field 0 — the whole reason `shreds` is a
	// nested struct), and all accessors (JsonoShredVector / ShredExtract / CollectShredTotality) index
	// shred k as `shreds` field 1 + k. Accepting a marker at any other position would let a hand-built
	// reordered struct parse as JSONO and then crash on read (the accessor would read the marker as a
	// shred pair). Every following field is a shred — unwrap a scalar pair to its value type, keep a
	// list as-is.
	auto &shred_fields = StructType::GetChildTypes(shreds_type);
	if (shred_fields.empty() || shred_fields[0].first != JSONO_SHRED_SET || !shred_fields[0].second.IsIntegral()) {
		return false;
	}
	child_list_t<LogicalType> shreds;
	for (idx_t i = 1; i < shred_fields.size(); i++) {
		LogicalType value_type;
		if (!UnwrapShredFieldType(shred_fields[i].second, value_type)) {
			return false;
		}
		shreds.emplace_back(shred_fields[i].first, value_type);
	}
	if (shreds.empty()) {
		return false; // the marker alone is not a valid shredded value
	}
	out.kind = JsonoLayoutKind::Shredded;
	out.layout_name = name;
	out.layout_type = layout_type;
	out.shreds = std::move(shreds);
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

string JsonoShredsName() {
	return JSONO_SHREDS;
}

string JsonoShredSetName() {
	return JSONO_SHRED_SET;
}

string JsonoShredValueName() {
	return JSONO_SHRED_VALUE;
}

string JsonoShredCompleteName() {
	return JSONO_SHRED_COMPLETE;
}

void JsonoValidateShredFieldName(const string &name) {
	if (name == "body") {
		// `body` is the residual field beside the `shreds` struct in the layout. A bare 'body' shred name
		// is rejected for clarity and to keep the DDL path consistent with the constructor; the path form
		// '$.body' is a different name and is fine.
		throw BinderException("jsono shred: a shred cannot be named 'body' (the residual field); "
		                      "shred the JSON key through its path form '$.body'");
	}
	if (name == JSONO_SHRED_SET) {
		throw BinderException("jsono shred: '%s' is a reserved layout field name", name);
	}
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
	// Canonicalize order: the marker identifies the shred SET (paths + types), not the field order. A
	// set-operation merged type lists the left branch's shreds first then the right's unique ones
	// (CombineStructTypes), so a writer stamping the canonical-order hash and a reader recomputing it
	// over a reorder-cast result type would otherwise disagree and lose the schema-identity match.
	std::sort(signatures.begin(), signatures.end());
	return jsono::HashShredManifestSignatures(signatures);
}

LogicalType JsonoShreddedStructType(const child_list_t<LogicalType> &shreds) {
	// `shreds` STRUCT: the marker first, then one field per shred — a scalar shred wraps its value
	// type in a pair STRUCT(value, complete TINYINT); an array shred (LIST) is kept as-is (it carries
	// no per-shred completeness — array shreds are always reconstructed, never bare-read). The marker
	// (JsonoShredSetName) is the canonical layout hash of the shred set (its uint64 bits reinterpreted
	// as a signed BIGINT), so the optimizer can confirm via its min/max zone-map that every scanned row
	// was written under EXACTLY the read type's shred set; a multi-file read unioning narrower shred
	// sets carries a different hash per file (min!=max), which keeps the residual COALESCE fallback
	// (see CollectShredTotality). It is BIGINT, not UBIGINT, on purpose: DuckDB's Parquet writer omits
	// min/max stats for unsigned integers (the signed page-stat ordering would misrepresent them), so a
	// UBIGINT marker would carry no Parquet zone-map and never prove schema identity on a Parquet scan.
	child_list_t<LogicalType> shred_children;
	shred_children.emplace_back(JSONO_SHRED_SET, LogicalType::BIGINT);
	for (auto &shred : shreds) {
		if (IsShredListType(shred.second)) {
			shred_children.push_back(shred);
			continue;
		}
		child_list_t<LogicalType> pair;
		pair.emplace_back(JSONO_SHRED_VALUE, shred.second);
		// `complete` is a 1/0 flag, not a BOOLEAN: DuckDB's Parquet statistics reader does not
		// transform BOOLEAN page min/max into NumericStats (it has no BOOLEAN case), so a BOOLEAN flag
		// would carry no zone-map on a Parquet scan and never prove per-shred totality there. TINYINT
		// is transformed, so its min == 1 ("no spill") is provable. See CollectShredTotality.
		pair.emplace_back(JSONO_SHRED_COMPLETE, LogicalType::TINYINT);
		shred_children.emplace_back(shred.first, LogicalType::STRUCT(std::move(pair)));
	}
	child_list_t<LogicalType> layout_children;
	layout_children.emplace_back("body", JsonoBodyStructType());
	layout_children.emplace_back(JSONO_SHREDS, LogicalType::STRUCT(std::move(shred_children)));
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
		set.AddFunction(ScalarFunction({jsono_type}, jsono_type, JsonoIdentityExecute));
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
