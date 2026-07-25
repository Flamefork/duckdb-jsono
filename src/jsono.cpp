#include "jsono.hpp"
#include "jsono_extension.hpp"
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

namespace {

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

// The permanent anchor: the layout field name never carries a revision, so a value written under
// any revision stays recognizable as JSONO rather than decaying into an anonymous struct.
constexpr const char *JSONO_LAYOUT = "jsono";
// Name stems of the two revisioned layout fields. The residual holds the binary body; the nested
// `shreds` STRUCT beside it holds the shred set, and shredded-ness is exactly its presence.
constexpr const char *JSONO_BODY_STEM = "body";
constexpr const char *JSONO_SHREDS_STEM = "shreds";

string RevisionedName(const char *stem, idx_t revision) {
	return string(stem) + "$" + std::to_string(revision);
}

// Read a revisioned layout field name: "<stem>$<digits>" yields its revision, a bare "<stem>"
// yields revision 0 (every value written before layout revisions existed). Digits are canonical —
// no leading zeros, no sign, no overflow — so a near-miss name is not a layout field at all.
// Revision 0 is the only one with two spellings ("body" and "body$0"); the bare one is what every
// pre-revision value carries, and no writer emits the "$0" form.
bool TryReadRevisionedName(const string &name, const char *stem, idx_t &revision) {
	auto stem_size = strlen(stem);
	if (name.compare(0, stem_size, stem) != 0) {
		return false;
	}
	if (name.size() == stem_size) {
		revision = 0;
		return true;
	}
	if (name[stem_size] != '$') {
		return false;
	}
	auto digits = name.substr(stem_size + 1);
	if (digits.empty() || (digits.size() > 1 && digits[0] == '0')) {
		return false;
	}
	idx_t parsed = 0;
	for (auto c : digits) {
		if (c < '0' || c > '9') {
			return false;
		}
		if (parsed > (NumericLimits<idx_t>::Maximum() - idx_t(c - '0')) / 10) {
			return false;
		}
		parsed = parsed * 10 + idx_t(c - '0');
	}
	revision = parsed;
	return true;
}
// Reserved name of the shred-set marker, the first field inside `shreds`. It is a BIGINT hash of the
// shred set (paths + types) — schema identity across files, and the mandatory shared member that lets
// a by-name struct cast bind between any two shred sets (even disjoint ones). The `$jsono$` prefix
// cannot occur in a shred path (the shred-spec parsers reject the reserved prefix), so no shred can
// collide with either reserved field.
constexpr const char *JSONO_SHRED_SET = "$jsono$set";
// Reserved name stem of the per-row spill bitmap columns, the fields right after the marker inside
// `shreds`: "$jsono$spill$0", "$jsono$spill$1", … (see JsonoShredSpillName in jsono.hpp for the bit
// semantics). Every column carries its ordinal — column 0 is not special-cased — so the names read
// as a uniform indexed family.
constexpr const char *JSONO_SHRED_SPILL = "$jsono$spill";

string SpillColumnName(idx_t column) {
	return string(JSONO_SHRED_SPILL) + "$" + std::to_string(column);
}
// The reserved layout-field namespace inside `shreds`: every non-shred member is named under it.
constexpr const char *JSONO_RESERVED_PREFIX = "$jsono$";

bool HasJsonoReservedPrefix(const string &name) {
	return name.compare(0, strlen(JSONO_RESERVED_PREFIX), JSONO_RESERVED_PREFIX) == 0;
}

// A scalar shred field is its bare value type; an array shred field is a LIST as-is. Either way the
// field type IS the shred's logical value type. Returns false when `type` is neither.
bool UnwrapShredFieldType(const LogicalType &type, LogicalType &value_type) {
	if (IsShredListType(type) || IsShredValueType(type)) {
		value_type = type;
		return true;
	}
	return false;
}

// The anchor's shape requirement: a named STRUCT of nothing but BLOBs. Revision-neutral — every
// revision's residual is a set of blob columns, whatever their number or names — and it is what
// keeps an ordinary user struct that merely spells `{'jsono': {'body': ...}}` from being refused as
// a foreign JSONO value.
bool IsBlobStruct(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(type)) {
		return false;
	}
	auto &children = StructType::GetChildTypes(type);
	if (children.empty()) {
		return false;
	}
	for (auto &child : children) {
		if (child.second.id() != LogicalTypeId::BLOB) {
			return false;
		}
	}
	return true;
}

// Classify one layout field (top-level child name + its STRUCT type) into `out`. The anchor is the
// field name `jsono` plus a field 0 named `body` or `body$<digits>` whose type is a STRUCT of pure
// BLOBs; it is narrow enough that no user struct hits it by accident and permanent, so a value
// written under a DIFFERENT revision is classified Foreign rather than silently passed over. Past
// the anchor the current grammar is: the six-BLOB body struct at `body$1`, optionally a `shreds$1`
// STRUCT sibling holding the shred-set marker, the spill bitmap columns and one field per shred.
// The single, unrevisioned layout name (`jsono` for
// plain and shredded) is deliberate: DuckDB reconciles set-operation branch types by field name
// (CombineStructTypes), so differently-shredded branches merge into one shredded type whose
// `shreds` is the union of the branches' shreds, and the marker stays the left branch's `shreds`
// field 0. The nested `shreds` struct keeps the marker contiguous with the shreds (a flat sibling
// layout interleaved it among the shreds under set-ops), and the marker is the mandatory shared
// member that lets a by-name cast bind between ANY two shred sets — including fully disjoint ones.
JsonoLayoutMatch MatchJsonoLayoutField(const string &name, const LogicalType &layout_type, JsonoLayoutType &out) {
	if (name != JSONO_LAYOUT) {
		return JsonoLayoutMatch::NotJsono;
	}
	if (layout_type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(layout_type)) {
		return JsonoLayoutMatch::NotJsono;
	}
	auto &fields = StructType::GetChildTypes(layout_type);
	if (fields.empty() || !TryReadRevisionedName(fields[0].first, JSONO_BODY_STEM, out.body_revision) ||
	    !IsBlobStruct(fields[0].second)) {
		return JsonoLayoutMatch::NotJsono;
	}
	// Read every revisioned stem, not just the leading pair. Two stems of the same kind mean the type
	// is a MIXTURE of revisions — a multi-file scan (`union_by_name`) merges per-file schemas by name,
	// so an old file beside a current one yields `body$1, shreds$1, body, shreds` in whichever order
	// the files were listed. Without this the classification depended on that order: an old file first
	// was refused, a current file first passed the anchor, failed the grammar and went silently
	// NotJsono — the whole scan, current rows included, then read as NULL through core json. Naming a
	// FOREIGN revision (rather than the one that sorted first) makes the refusal order-independent.
	idx_t body_stems = 0;
	idx_t shreds_stems = 0;
	for (auto &field : fields) {
		idx_t revision;
		if (TryReadRevisionedName(field.first, JSONO_BODY_STEM, revision)) {
			body_stems++;
			if (revision != JSONO_BODY_REVISION) {
				out.body_revision = revision;
			}
		} else if (TryReadRevisionedName(field.first, JSONO_SHREDS_STEM, revision)) {
			shreds_stems++;
			if (out.shreds_revision == DConstants::INVALID_INDEX || revision != JSONO_SHREDS_REVISION) {
				out.shreds_revision = revision;
			}
		}
	}
	if (body_stems > 1 || shreds_stems > 1) {
		return JsonoLayoutMatch::Foreign;
	}
	// A KNOWN-BUT-DIFFERENT revision is the loud case: such a value was written by another build and
	// reading it would silently lose data. Everything that fails below stays silent (NotJsono) — the
	// behaviour from before revisions existed — because a type carrying THIS revision can fail the
	// grammar for reasons that are not stored data at all: a generic value->SQL->value round-trip
	// (the DuckLake inlined-data flush) narrows an all-NULL spill column to SQLNULL and a small lane
	// to INTEGER on its way to the declared column type, and a hand-built struct can put the reserved
	// fields anywhere. Refusing those would break a legal write path in order to catch a forgery.
	bool foreign_body = out.body_revision != JSONO_BODY_REVISION;
	bool foreign_shreds =
	    out.shreds_revision != DConstants::INVALID_INDEX && out.shreds_revision != JSONO_SHREDS_REVISION;
	if (foreign_body || foreign_shreds) {
		return JsonoLayoutMatch::Foreign;
	}
	if (fields[0].second != JsonoBodyStructType()) {
		return JsonoLayoutMatch::NotJsono;
	}
	if (fields.size() == 1) {
		out.kind = JsonoLayoutKind::Plain;
		out.shreds.clear();
		return JsonoLayoutMatch::Current;
	}
	// Shredded: exactly the residual plus a `shreds` STRUCT. The set-op merge keeps the layout struct
	// at these two by-name fields, so a third sibling is not a JSONO value.
	if (fields.size() != 2 || out.shreds_revision != JSONO_SHREDS_REVISION) {
		return JsonoLayoutMatch::NotJsono;
	}
	auto &shreds_type = fields[1].second;
	if (shreds_type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(shreds_type)) {
		return JsonoLayoutMatch::NotJsono;
	}
	// Inside `shreds`: the marker is field 0 and the spill bitmap columns fields 1..k (any integer
	// width — a value round-tripped through a generic value->SQL->value path, e.g. DuckLake
	// inlined-data INSERT, re-parses the bare-integer fields as BIGINT/HUGEINT, but the reserved
	// names still identify them). They must sit at those fixed positions, not merely be present by
	// name: every writer emits them first and every set-op merge keeps them first (they are the left
	// branch's `shreds` leading fields — the whole reason `shreds` is a nested struct), and all
	// accessors (JsonoShredVector / ShredExtract / CollectShredTotality) index shred k as `shreds`
	// field 1 + spill_columns + k. Accepting them at any other position would let a hand-built
	// reordered struct parse as JSONO and then crash on read (the accessor would read a reserved
	// field as a shred lane). Every following field is a shred — its type is the shred's value type
	// (bare scalar or LIST).
	auto &shred_fields = StructType::GetChildTypes(shreds_type);
	if (shred_fields.size() < 2 || shred_fields[0].first != JSONO_SHRED_SET || !shred_fields[0].second.IsIntegral()) {
		return JsonoLayoutMatch::NotJsono;
	}
	idx_t spill_columns = 0;
	while (1 + spill_columns < shred_fields.size() &&
	       shred_fields[1 + spill_columns].first == SpillColumnName(spill_columns)) {
		if (!shred_fields[1 + spill_columns].second.IsIntegral()) {
			return JsonoLayoutMatch::NotJsono;
		}
		spill_columns++;
	}
	if (spill_columns == 0) {
		return JsonoLayoutMatch::NotJsono;
	}
	child_list_t<LogicalType> shreds;
	for (idx_t i = 1 + spill_columns; i < shred_fields.size(); i++) {
		// A lane name that is not a non-empty pure object-key chain (an array-index or root '$' path)
		// cannot be a shred: no writer produces one, and every reconstruct-based reader would throw on
		// it. A reserved-prefix name cannot be one either. Rejecting them here keeps such a raw-cast /
		// stored struct from being recognized as JSONO.
		if (HasJsonoReservedPrefix(shred_fields[i].first) || !ShredNameIsObjectKeyPath(shred_fields[i].first)) {
			return JsonoLayoutMatch::NotJsono;
		}
		LogicalType value_type;
		if (!UnwrapShredFieldType(shred_fields[i].second, value_type)) {
			return JsonoLayoutMatch::NotJsono;
		}
		shreds.emplace_back(shred_fields[i].first, value_type);
	}
	if (shreds.empty()) {
		return JsonoLayoutMatch::NotJsono; // the reserved fields alone are not a valid shredded value
	}
	// A set-op merged type can carry FEWER spill columns than the crossing union's shred count
	// needs (see JsonoSpillColumnCount) — readable, never provable past its columns — but never
	// more: no writer over-provisions, so extra columns mean a hand-built struct.
	if (spill_columns > JsonoSpillColumnCount(shreds.size())) {
		return JsonoLayoutMatch::NotJsono;
	}
	out.kind = JsonoLayoutKind::Shredded;
	out.shreds = std::move(shreds);
	out.spill_columns = spill_columns;
	return JsonoLayoutMatch::Current;
}

} // namespace

JsonoLayoutMatch MatchJsonoLayoutType(const LogicalType &type, JsonoLayoutType &out) {
	out = JsonoLayoutType(); // never inherit a previous parse: callers reuse one `out` across types
	if (type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(type)) {
		return JsonoLayoutMatch::NotJsono;
	}
	auto &children = StructType::GetChildTypes(type);
	if (children.size() != 1) {
		return JsonoLayoutMatch::NotJsono; // an ordinary value has exactly one layout field
	}
	return MatchJsonoLayoutField(children[0].first, children[0].second, out);
}

void JsonoRejectForeignLayout(const LogicalType &type, const string &context) {
	JsonoLayoutType layout;
	if (MatchJsonoLayoutType(type, layout) != JsonoLayoutMatch::Foreign) {
		return;
	}
	// The revisions are named machine-readably on purpose: they are the only thing that tells a user
	// which build wrote the data, and the key into the revision map in docs/jsono_format.md, which
	// turns them into a commit range and an upgrade recipe. The body revision is always known here
	// (the anchor parsed it); the shreds one only for a value carrying shreds.
	string read = StringUtil::Format("body=%llu", (unsigned long long)layout.body_revision);
	if (layout.shreds_revision != DConstants::INVALID_INDEX) {
		read += StringUtil::Format(" shreds=%llu", (unsigned long long)layout.shreds_revision);
	}
	throw InvalidInputException(
	    "%s: JSONO value carries layout revision %s, this build reads body=%llu shreds=%llu. Reading it would "
	    "silently lose data, so it is refused. This build can still MOVE the value (SELECT *, INSERT ... SELECT, "
	    "COPY ... TO a Parquet file) but not read or render it: carry it to a build that reads that revision and "
	    "rewrite it there with jsono(...), or drop the data. Type: %s",
	    context, read, (unsigned long long)JSONO_BODY_REVISION, (unsigned long long)JSONO_SHREDS_REVISION,
	    type.ToString());
}

bool TryParseJsonoLayoutType(const LogicalType &type, JsonoLayoutType &out) {
	return MatchJsonoLayoutType(type, out) == JsonoLayoutMatch::Current;
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
	JsonoRejectForeignLayout(type, function_name);
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

string JsonoBodyName() {
	return RevisionedName(JSONO_BODY_STEM, JSONO_BODY_REVISION);
}

string JsonoShredsName() {
	return RevisionedName(JSONO_SHREDS_STEM, JSONO_SHREDS_REVISION);
}

string JsonoShredSetName() {
	return JSONO_SHRED_SET;
}

string JsonoShredSpillName(idx_t column) {
	return SpillColumnName(column);
}

void JsonoValidateShredFieldName(const string &name) {
	if (name == JSONO_BODY_STEM) {
		// The residual field beside the `shreds` struct is named after this stem. A bare 'body' shred name
		// is rejected for clarity and to keep the DDL path consistent with the constructor; the path form
		// '$.body' is a different name and is fine.
		throw BinderException("jsono shred: a shred cannot be named 'body' (the residual field); "
		                      "shred the JSON key through its path form '$.body'");
	}
	if (HasJsonoReservedPrefix(name)) {
		throw BinderException("jsono shred: '%s' collides with the reserved '%s' layout field namespace", name,
		                      JSONO_RESERVED_PREFIX);
	}
	if (!ShredNameIsObjectKeyPath(name)) {
		throw BinderException("jsono shred: shred path '%s' must be a non-empty object-key path "
		                      "(array-index and root '$' paths cannot be shredded)",
		                      name);
	}
}

vector<idx_t> JsonoSpillRanksOfNames(const vector<string> &names) {
	vector<idx_t> order(names.size());
	for (idx_t i = 0; i < names.size(); i++) {
		order[i] = i;
	}
	std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return names[a] < names[b]; });
	vector<idx_t> ranks(names.size());
	for (idx_t r = 0; r < order.size(); r++) {
		ranks[order[r]] = r;
	}
	return ranks;
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
	// `shreds` STRUCT: the marker, the ⌈N/63⌉ spill bitmap columns, then one field per shred — a
	// scalar shred is its bare value type, an array shred (LIST) is kept as-is. The marker
	// (JsonoShredSetName) is the canonical layout hash of the shred set (its uint64 bits
	// reinterpreted as a signed BIGINT), clean or bit-flipped dirty per row, so the optimizer can
	// confirm via its min/max zone-map that every scanned row was written under EXACTLY the read
	// type's shred set (and whether any was dirty); a multi-file read unioning narrower shred sets
	// carries an unrelated hash per file, which keeps the residual COALESCE fallback (see
	// CollectShredTotality). The spill columns (JsonoShredSpillName) carry the per-row
	// diverted-shred bits of the dirty rows; a min==max==M uniform mask decodes per-shred exactly.
	// All reserved fields are BIGINT, not UBIGINT, on purpose: DuckDB's Parquet writer omits
	// min/max stats for unsigned integers (the signed page-stat ordering would misrepresent them),
	// so a UBIGINT field would carry no Parquet zone-map and never prove anything on a Parquet scan.
	child_list_t<LogicalType> shred_children;
	shred_children.emplace_back(JSONO_SHRED_SET, LogicalType::BIGINT);
	for (idx_t column = 0; column < JsonoSpillColumnCount(shreds.size()); column++) {
		shred_children.emplace_back(SpillColumnName(column), LogicalType::BIGINT);
	}
	for (auto &shred : shreds) {
		shred_children.push_back(shred);
	}
	child_list_t<LogicalType> layout_children;
	layout_children.emplace_back(JsonoBodyName(), JsonoBodyStructType());
	layout_children.emplace_back(JsonoShredsName(), LogicalType::STRUCT(std::move(shred_children)));
	child_list_t<LogicalType> top;
	top.emplace_back(JsonoLayoutName(), LogicalType::STRUCT(std::move(layout_children)));
	return LogicalType::STRUCT(std::move(top));
}

LogicalType JsonoRawStructType() {
	child_list_t<LogicalType> layout_children;
	layout_children.emplace_back(JsonoBodyName(), JsonoBodyStructType());
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
}

} // namespace duckdb
