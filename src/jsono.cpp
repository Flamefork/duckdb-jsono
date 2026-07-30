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
#include "duckdb/planner/expression/bound_function_expression.hpp"

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

// jsono_layout_diagnose(value) -> what the extension sees `value` as, and why not JSONO when it is
// not. Purely a question about the argument's TYPE, so it is answered at bind and returned as a
// constant; the value's bytes are never read. It exists because the "current revision, fails the
// grammar" case is deliberately silent (refusing it would break the legal DuckLake write path, see
// MatchJsonoLayoutField) — and silence is indistinguishable from an ordinary NULL until something
// downstream fails. This is the one place to ask.
struct JsonoLayoutDiagnoseBindData : public FunctionData {
	explicit JsonoLayoutDiagnoseBindData(Value diagnosis_p) : diagnosis(std::move(diagnosis_p)) {
	}
	Value diagnosis;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoLayoutDiagnoseBindData>(diagnosis);
	}
	bool Equals(const FunctionData &other) const override {
		return diagnosis == other.Cast<JsonoLayoutDiagnoseBindData>().diagnosis;
	}
};

unique_ptr<FunctionData> JsonoLayoutDiagnoseBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	(void)context;
	(void)bound_function;
	if (arguments[0]->HasParameter()) {
		// The diagnosis is a bind-time fact frozen into the bind data, so binding against an unresolved
		// parameter would answer about UNKNOWN and keep answering that after the parameter arrives.
		throw ParameterNotResolvedException();
	}
	return make_uniq<JsonoLayoutDiagnoseBindData>(JsonoDiagnoseLayoutMatch(arguments[0]->return_type));
}

void JsonoLayoutDiagnoseExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	auto &info = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<JsonoLayoutDiagnoseBindData>();
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, info.diagnosis);
}

// jsono_layout_lanes(value) -> the shred lanes of `value`'s type as (logical path, type), ordered
// as the type carries them. A lane's STRUCT field name is the base32hex encoding of its path (see
// the lane-name boundary in jsono_path.hpp), so reading the schema by eye no longer answers "which
// paths are shredded here" — this does, and it is the reason that trade is affordable. Like
// jsono_layout_diagnose it is a question about the TYPE, answered at bind, reading no bytes.
//
// The empty list is reserved for the one type that honestly HAS no lanes: a plain JSONO value. A
// foreign revision does have lanes — this build just cannot read their naming — and a non-JSONO
// struct has no lanes to speak of, so answering `[]` for either would be the same silent substitution
// the refusal machinery exists to prevent, and indistinguishable from a plain value. Both refuse at
// bind, which is also why this function needs no exemption from the plan walk: it is already loud by
// the time the walk would reach it. jsono_layout_diagnose stays the one never-throwing answer to
// "what is this".
struct JsonoLayoutLanesBindData : public FunctionData {
	explicit JsonoLayoutLanesBindData(Value lanes_p) : lanes(std::move(lanes_p)) {
	}
	Value lanes;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoLayoutLanesBindData>(lanes);
	}
	bool Equals(const FunctionData &other) const override {
		return lanes == other.Cast<JsonoLayoutLanesBindData>().lanes;
	}
};

LogicalType JsonoLayoutLanesResultType() {
	child_list_t<LogicalType> lane;
	lane.emplace_back("path", LogicalType::VARCHAR);
	lane.emplace_back("type", LogicalType::VARCHAR);
	return LogicalType::LIST(LogicalType::STRUCT(std::move(lane)));
}

unique_ptr<FunctionData> JsonoLayoutLanesBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	(void)context;
	(void)bound_function;
	if (arguments[0]->HasParameter()) {
		// The answer is a bind-time fact about the argument's TYPE, so an unresolved parameter has to
		// re-bind once it has one rather than be classified as "not JSONO".
		throw ParameterNotResolvedException();
	}
	auto &argument_type = arguments[0]->return_type;
	JsonoLayoutType layout;
	string reason;
	auto match = MatchJsonoLayoutType(argument_type, layout, &reason);
	JsonoRejectForeignLayout(argument_type, "jsono_layout_lanes()");
	if (match != JsonoLayoutMatch::Current) {
		throw BinderException("jsono_layout_lanes(): argument is not a JSONO value: %s", reason);
	}
	vector<Value> lanes;
	for (auto &shred : layout.shreds) {
		child_list_t<Value> lane;
		lane.emplace_back("path", Value(JsonoLaneLogicalPath(shred.first)));
		// The lane type is reported logically too: an object-array lane's element subfields are
		// encoded one-step paths in the stored type, and printing those would both hide the JSON
		// keys and break pasting the answer back into a shredding spec.
		lane.emplace_back("type", Value(JsonoLaneLogicalType(shred.second).ToString()));
		lanes.push_back(Value::STRUCT(std::move(lane)));
	}
	auto element_type = ListType::GetChildType(JsonoLayoutLanesResultType());
	return make_uniq<JsonoLayoutLanesBindData>(Value::LIST(element_type, std::move(lanes)));
}

void JsonoLayoutLanesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	auto &info = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<JsonoLayoutLanesBindData>();
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, info.lanes);
}

// jsono_storage_type(spec) -> the shredded storage type's DDL: the 6-BLOB residual plus the
// given shred lanes, so a schema can declare a shredded jsono column from the SAME spec string the
// constructor takes (see JsonoShredSpecEntries). The paths are parsed and re-emitted through
// JsonoShreddedStructType, so the declared column is byte-identical to a value built with the same
// `shredding :=` spec.
void JsonoStorageTypeWithShredsExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t shreds) {
		auto entries = JsonoShredSpecEntries(Value(shreds.GetString()), "jsono_storage_type");
		vector<JsonoLaneSpec> lanes;
		vector<string> lane_names;
		vector<string> spec_names;
		for (auto &entry : entries) {
			auto type = JsonoParseShredSpecType(entry.second, context, "jsono_storage_type");
			auto lane = JsonoParseShredSpecField(entry.first, type);
			auto lane_name = JsonoEncodeLaneName(lane.path);
			for (idx_t i = 0; i < lane_names.size(); i++) {
				if (lane_names[i] == lane_name) {
					// Two spec entries naming one path would be two fields of one name — impossible in a
					// STRUCT. Both spellings are named; a byte-exact duplicate reaches here too
					// (JsonoShredSpecEntries keeps it: a spec is a declaration, not a last-wins document).
					throw BinderException("jsono_storage_type: '%s' and '%s' name the same shred path", spec_names[i],
					                      entry.first);
				}
			}
			lane_names.push_back(std::move(lane_name));
			spec_names.push_back(entry.first);
			lanes.push_back(std::move(lane));
		}
		// Canonical shred order (sorted by encoded lane name) so the DDL matches a value built from the
		// same shreds regardless of the order they are written in here vs the shredding spec.
		vector<idx_t> order(lanes.size());
		for (idx_t i = 0; i < order.size(); i++) {
			order[i] = i;
		}
		std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return lane_names[a] < lane_names[b]; });
		vector<JsonoLaneSpec> sorted;
		sorted.reserve(lanes.size());
		for (auto i : order) {
			sorted.push_back(std::move(lanes[i]));
		}
		return StringVector::AddString(result, JsonoShreddedStructType(sorted).ToString());
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

// Read the canonical decimal index a `$`-suffixed layout name carries: no leading zeros, no sign,
// no overflow, so a near-miss spelling is not that layout field at all rather than an alias of it.
// Shared by both `$`-suffixed name families (the revisioned stems and the spill columns) so the two
// cannot drift into accepting different spellings of the same number.
bool TryReadCanonicalIndex(const string &digits, idx_t &index) {
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
	index = parsed;
	return true;
}

// Read a revisioned layout field name: "<stem>$<digits>" yields its revision, a bare "<stem>"
// yields revision 0 (every value written before layout revisions existed).
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
	return TryReadCanonicalIndex(name.substr(stem_size + 1), revision);
}
// Reserved name of the shred-set marker, the field every writer emits first inside `shreds`. It is a BIGINT hash of the
// shred set (lane names + types) — schema identity across files, and the mandatory shared member that lets
// a by-name struct cast bind between any two shred sets (even disjoint ones). A lane name is drawn
// from the base32hex alphabet `0-9a-v` (see the lane-name boundary in jsono_path.hpp), so it can
// start with neither `$` nor anything else outside that alphabet: no shred can collide with a
// reserved field, structurally rather than by a check.
constexpr const char *JSONO_SHRED_SET = "$jsono$set";
// Reserved name stem of the per-row spill bitmap columns, the fields every writer emits right after the marker inside
// `shreds`: "$jsono$spill$0", "$jsono$spill$1", … (see JsonoShredSpillName in jsono.hpp for the bit
// semantics). Every column carries its ordinal — column 0 is not special-cased — so the names read
// as a uniform indexed family.
constexpr const char *JSONO_SHRED_SPILL = "$jsono$spill";

string SpillColumnName(idx_t column) {
	return string(JSONO_SHRED_SPILL) + "$" + std::to_string(column);
}

// Parse a spill column name ("$jsono$spill$<digits>") back into its column number. Unlike
// TryReadRevisionedName there is no bare-stem spelling: every writer numbers every spill column
// (column 0 included), so a name without a "$<digits>" suffix is not one.
bool TryParseSpillColumnName(const string &name, idx_t &column) {
	auto stem_size = strlen(JSONO_SHRED_SPILL);
	if (name.size() <= stem_size + 1 || name.compare(0, stem_size, JSONO_SHRED_SPILL) != 0 || name[stem_size] != '$') {
		return false;
	}
	return TryReadCanonicalIndex(name.substr(stem_size + 1), column);
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

// An object-array lane's element STRUCT names its subfields by the same codec one level down (a
// subfield is a one-step path), so the grammar must gate them too: CombineStructTypes recurses into
// element structs, and an unencoded subfield name would collapse case-insensitively exactly like a
// lane name. A non-canonical subfield is a hand-built or foreign struct, not a lane any writer
// emitted, and every reader that decodes one would throw mid-read.
bool ShredSubfieldNamesAreCanonical(const LogicalType &type) {
	if (!IsShredArrayType(type)) {
		return true;
	}
	for (auto &sub : StructType::GetChildTypes(ListType::GetChildType(type))) {
		vector<PathStep> steps;
		if (!JsonoTryDecodeLaneName(sub.first, steps) || steps.size() != 1) {
			return false;
		}
	}
	return true;
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

// Record why the grammar refused and return the refusal. `reason == nullptr` is the hot path —
// every bind classifies ordinary types through the grammar, and almost all of them are refused on
// the first check — so the message is only formatted when someone asked for it. Carrying the
// explanation out of the grammar itself is the point: a separate "why not" routine would be a
// second copy of the rules and would drift from the one that decides.
template <class... ARGS>
JsonoLayoutMatch RejectNotJsono(string *reason, const char *format, ARGS... args) {
	if (reason) {
		*reason = StringUtil::Format(format, args...);
	}
	return JsonoLayoutMatch::NotJsono;
}

// Classify one layout field (top-level child name + its STRUCT type) into `out`. The anchor is the
// field name `jsono` plus a field 0 named `body` or `body$<digits>` whose type is a STRUCT of pure
// BLOBs; it is narrow enough that no user struct hits it by accident and permanent, so a value
// written under a DIFFERENT revision is classified Foreign rather than silently passed over. Past
// the anchor the current grammar is: the six-BLOB body struct at `body$2`, optionally a `shreds$2`
// STRUCT sibling holding the shred-set marker, the spill bitmap columns and one field per shred.
// The single, unrevisioned layout name (`jsono` for
// plain and shredded) is deliberate: DuckDB reconciles set-operation branch types by field name
// (CombineStructTypes), so differently-shredded branches merge into one shredded type whose
// `shreds` is the union of the branches' shreds, and the marker stays the left branch's `shreds`
// field 0. The nested `shreds` struct keeps the marker contiguous with the shreds (a flat sibling
// layout interleaved it among the shreds under set-ops), and the marker is the mandatory shared
// member that lets a by-name cast bind between ANY two shred sets — including fully disjoint ones.
JsonoLayoutMatch MatchJsonoLayoutField(const string &name, const LogicalType &layout_type, JsonoLayoutType &out,
                                       string *reason) {
	if (name != JSONO_LAYOUT) {
		return RejectNotJsono(reason, "the value's single field is named '%s', not the layout anchor '%s'", name,
		                      JSONO_LAYOUT);
	}
	if (layout_type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(layout_type)) {
		return RejectNotJsono(reason, "the '%s' field is %s, not a named STRUCT", JSONO_LAYOUT, layout_type.ToString());
	}
	auto &fields = StructType::GetChildTypes(layout_type);
	if (fields.empty() || !TryReadRevisionedName(fields[0].first, JSONO_BODY_STEM, out.body_revision) ||
	    !IsBlobStruct(fields[0].second)) {
		return RejectNotJsono(reason,
		                      "field 0 of '%s' is not the residual anchor: it must be named '%s' (optionally "
		                      "'%s$<revision>') and be a STRUCT of nothing but BLOBs",
		                      JSONO_LAYOUT, JSONO_BODY_STEM, JSONO_BODY_STEM);
	}
	// Read every revisioned stem, not just the leading pair. Two stems of the same kind mean the type
	// is a MIXTURE of revisions — a multi-file scan (`union_by_name`) merges per-file schemas by name,
	// so an old file beside a current one yields `body$2, shreds$2, body, shreds` in whichever order
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
	// reading it would silently lose data. Everything that fails below CLASSIFIES silent (NotJsono)
	// — recognition runs over every struct in every plan and never throws — because a type carrying
	// THIS revision can fail the grammar for reasons that are not stored data at all: a generic
	// value->SQL->value round-trip (the DuckLake inlined-data flush) narrows an all-NULL spill
	// column to SQLNULL and a small lane to INTEGER on its way to the declared column type, and a
	// hand-built struct can put the reserved fields anywhere. Refusing those would break a legal
	// write path in order to catch a forgery. One class is separately marked for the optimizer to
	// refuse at the READ sites: a lane NAME that does not decode inside a fully-anchored type, which
	// no legal narrowing produces (see JsonoLayoutType::lane_name_malformed).
	bool foreign_body = out.body_revision != JSONO_BODY_REVISION;
	bool foreign_shreds =
	    out.shreds_revision != DConstants::INVALID_INDEX && out.shreds_revision != JSONO_SHREDS_REVISION;
	if (foreign_body || foreign_shreds) {
		return JsonoLayoutMatch::Foreign;
	}
	if (fields[0].second != JsonoBodyStructType()) {
		return RejectNotJsono(reason, "'%s' carries revision %llu but its type is %s, not this build's %s",
		                      fields[0].first, (unsigned long long)out.body_revision, fields[0].second.ToString(),
		                      JsonoBodyStructType().ToString());
	}
	if (fields.size() == 1) {
		out.kind = JsonoLayoutKind::Plain;
		out.shreds.clear();
		return JsonoLayoutMatch::Current;
	}
	// Shredded: exactly the residual plus a `shreds` STRUCT. The set-op merge keeps the layout struct
	// at these two by-name fields, so a third sibling is not a JSONO value.
	if (fields.size() != 2 || out.shreds_revision != JSONO_SHREDS_REVISION) {
		return RejectNotJsono(reason,
		                      "'%s' has %llu fields; a shredded value has exactly 2 (the residual and a '%s$%llu' "
		                      "STRUCT), a plain value exactly 1",
		                      JSONO_LAYOUT, (unsigned long long)fields.size(), JSONO_SHREDS_STEM,
		                      (unsigned long long)JSONO_SHREDS_REVISION);
	}
	auto &shreds_type = fields[1].second;
	if (shreds_type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(shreds_type)) {
		return RejectNotJsono(reason, "'%s' is %s, not a named STRUCT", fields[1].first, shreds_type.ToString());
	}
	// Inside `shreds`: the marker and the spill bitmap columns (any integer width — a value
	// round-tripped through a generic value->SQL->value path, e.g. DuckLake inlined-data INSERT,
	// re-parses the bare-integer fields as BIGINT/HUGEINT, but the reserved names still identify
	// them) are found BY NAME, not by fixed position. A set-op merge of two shredded branches
	// (CombineStructTypes) keeps each branch's own field order but appends fields unique to the
	// right branch at the END — so a union of a narrow-shred-set branch with a wide one (needing a
	// second spill column) produces `$jsono$set, $jsono$spill$0, <narrow's shreds...>,
	// $jsono$spill$1, <wide-only shreds...>`: the reserved fields are no longer contiguous. Every
	// accessor (JsonoShredVector / JsonoShredSpillVector / JsonoSpillColumnsOf) matches the same way,
	// so recognition and reads agree regardless of which fields a merge displaced.
	auto &shred_fields = StructType::GetChildTypes(shreds_type);
	if (shred_fields.empty()) {
		return RejectNotJsono(reason, "'%s' is an empty STRUCT", fields[1].first);
	}
	idx_t marker_index = shred_fields.size();
	idx_t spill_columns = 0;
	for (idx_t i = 0; i < shred_fields.size(); i++) {
		if (shred_fields[i].first == JSONO_SHRED_SET) {
			if (marker_index != shred_fields.size()) {
				// duplicate marker: not a value any writer or merge produced
				return RejectNotJsono(reason, "'%s' carries more than one '%s' marker field", fields[1].first,
				                      JSONO_SHRED_SET);
			}
			marker_index = i;
			continue;
		}
		idx_t column;
		if (TryParseSpillColumnName(shred_fields[i].first, column)) {
			spill_columns++;
		}
	}
	if (marker_index == shred_fields.size()) {
		return RejectNotJsono(reason, "'%s' has no '%s' marker field", fields[1].first, JSONO_SHRED_SET);
	}
	if (!shred_fields[marker_index].second.IsIntegral()) {
		return RejectNotJsono(reason, "the '%s' marker is %s, not an integer", JSONO_SHRED_SET,
		                      shred_fields[marker_index].second.ToString());
	}
	if (spill_columns == 0) {
		return RejectNotJsono(reason, "'%s' has no '%s$<n>' spill bitmap column", fields[1].first, JSONO_SHRED_SPILL);
	}
	// Spill columns are a dense 0..spill_columns-1 family (see SpillColumnName) — a gap or a
	// duplicate column number means a hand-built struct, not one any writer or merge produced.
	vector<bool> spill_seen(spill_columns, false);
	child_list_t<LogicalType> shreds;
	for (idx_t i = 0; i < shred_fields.size(); i++) {
		if (i == marker_index) {
			continue;
		}
		idx_t column;
		if (TryParseSpillColumnName(shred_fields[i].first, column)) {
			if (column >= spill_columns || spill_seen[column]) {
				return RejectNotJsono(reason, "spill column '%s' breaks the dense 0..%llu numbering every writer emits",
				                      shred_fields[i].first, (unsigned long long)(spill_columns - 1));
			}
			if (!shred_fields[i].second.IsIntegral()) {
				return RejectNotJsono(reason, "spill column '%s' is %s, not an integer", shred_fields[i].first,
				                      shred_fields[i].second.ToString());
			}
			spill_seen[column] = true;
			continue;
		}
		// Everything left is a lane, and a lane is named by the encoding of its path: a name that does
		// not decode canonically is one no writer could have minted, so the struct is a raw-cast or
		// hand-built value rather than a JSONO one. Canonicity subsumes the old object-key-path check
		// (a decode yields nothing but a non-empty chain of keys) and the old reserved-prefix check
		// (the alphabet `0-9a-v` cannot spell `$jsono$…`).
		if (!JsonoLaneNameIsCanonical(shred_fields[i].first)) {
			out.lane_name_malformed = true;
			return RejectNotJsono(reason,
			                      "field '%s' is not a shred lane: a lane is named by the base32hex encoding of "
			                      "its object-key path, and this name does not decode canonically",
			                      shred_fields[i].first);
		}
		LogicalType value_type;
		if (!UnwrapShredFieldType(shred_fields[i].second, value_type)) {
			return RejectNotJsono(reason, "shred '%s' is %s, which is not a shred value type or LIST of one",
			                      shred_fields[i].first, shred_fields[i].second.ToString());
		}
		// An element subfield is a lane name one level down, minted by the same codec
		// (JsonoEncodeLaneSubfieldName) — so a non-decoding subfield name marks the same
		// NAME-level class as a non-decoding lane name: a legal generic write narrows subfield
		// TYPES and keeps subfield NAMES.
		if (!ShredSubfieldNamesAreCanonical(value_type)) {
			out.lane_name_malformed = true;
			return RejectNotJsono(reason,
			                      "array shred '%s' has an element subfield whose name is not the base32hex "
			                      "encoding of a single object key",
			                      shred_fields[i].first);
		}
		shreds.emplace_back(shred_fields[i].first, value_type);
	}
	if (shreds.empty()) {
		// the reserved fields alone are not a valid shredded value
		return RejectNotJsono(reason, "'%s' carries only reserved layout fields and no shred", fields[1].first);
	}
	// A set-op merged type can carry FEWER spill columns than the crossing union's shred count
	// needs (see JsonoSpillColumnCount) — readable, never provable past its columns — but never
	// more: no writer over-provisions, so extra columns mean a hand-built struct.
	if (spill_columns > JsonoSpillColumnCount(shreds.size())) {
		return RejectNotJsono(reason, "%llu spill columns for %llu shreds; at most %llu can be provisioned",
		                      (unsigned long long)spill_columns, (unsigned long long)shreds.size(),
		                      (unsigned long long)JsonoSpillColumnCount(shreds.size()));
	}
	out.kind = JsonoLayoutKind::Shredded;
	out.shreds = std::move(shreds);
	out.spill_columns = spill_columns;
	return JsonoLayoutMatch::Current;
}

} // namespace

JsonoLayoutMatch MatchJsonoLayoutType(const LogicalType &type, JsonoLayoutType &out, string *reason) {
	out = JsonoLayoutType(); // never inherit a previous parse: callers reuse one `out` across types
	if (type.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(type)) {
		return RejectNotJsono(reason, "the value is %s, not a named STRUCT", type.ToString());
	}
	auto &children = StructType::GetChildTypes(type);
	if (children.size() != 1) {
		return RejectNotJsono(reason,
		                      "the value's STRUCT has %llu fields; a JSONO value has exactly one (the '%s' "
		                      "layout field)",
		                      (unsigned long long)children.size(), JSONO_LAYOUT);
	}
	return MatchJsonoLayoutField(children[0].first, children[0].second, out, reason);
}

LogicalType JsonoLayoutDiagnoseResultType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("kind", LogicalType::VARCHAR);
	fields.emplace_back("reason", LogicalType::VARCHAR);
	fields.emplace_back("body_revision", LogicalType::UBIGINT);
	fields.emplace_back("shreds_revision", LogicalType::UBIGINT);
	fields.emplace_back("spill_columns", LogicalType::UBIGINT);
	return LogicalType::STRUCT(std::move(fields));
}

Value JsonoDiagnoseLayoutMatch(const LogicalType &type) {
	JsonoLayoutType layout;
	string reason;
	auto match = MatchJsonoLayoutType(type, layout, &reason);
	// Unknown is NULL, never zero: a refused type reporting `0 spill columns` would read as a fact
	// about the value rather than as "the grammar never got that far". A revision is known exactly
	// when the anchor parsed one, which the parse already records as INVALID_INDEX-or-not; the counts
	// only once the grammar finished reading the shred set, which is only the Current answer.
	auto revision = [](idx_t value) {
		return value == DConstants::INVALID_INDEX ? Value(LogicalType::UBIGINT) : Value::UBIGINT(value);
	};
	string kind;
	auto reason_value = Value(LogicalType::VARCHAR);
	auto spill_columns = Value(LogicalType::UBIGINT);
	switch (match) {
	case JsonoLayoutMatch::Current:
		kind = layout.kind == JsonoLayoutKind::Plain ? "plain" : "shredded";
		spill_columns = Value::UBIGINT(layout.spill_columns);
		break;
	case JsonoLayoutMatch::Foreign:
		// Same wording source as the refusal itself, so the diagnosis and the error a read raises
		// cannot describe the same value differently.
		kind = "foreign";
		reason_value = Value(JsonoDescribeForeignLayout(layout));
		break;
	default:
		// A value of the CURRENT revision that fails the grammar is deliberately NOT refused at read
		// time (it stays silent NotJsono — see MatchJsonoLayoutField), which is exactly the case this
		// function exists to make visible: the read path hands the value to core json, where an
		// extract reads NULL and to_json serializes the physical struct. The one exception is the
		// malformed-NAME class, whose reads and conversions JsonoRejectMalformedAnchoredRead refuses
		// with this same reason.
		kind = "not jsono";
		reason_value = Value(reason);
		break;
	}
	child_list_t<Value> fields;
	fields.emplace_back("kind", Value(kind));
	fields.emplace_back("reason", std::move(reason_value));
	fields.emplace_back("body_revision", revision(layout.body_revision));
	fields.emplace_back("shreds_revision", revision(layout.shreds_revision));
	fields.emplace_back("spill_columns", std::move(spill_columns));
	return Value::STRUCT(std::move(fields));
}

string JsonoDescribeForeignLayout(const JsonoLayoutType &layout) {
	// The revisions are named machine-readably on purpose: they are the only thing that tells a user
	// which build wrote the data, and the key into the revision map in docs/jsono_format.md, which
	// turns them into a commit range and an upgrade recipe. The body revision is always known here
	// (the anchor parsed it); the shreds one only for a value carrying shreds.
	string read = StringUtil::Format("body=%llu", (unsigned long long)layout.body_revision);
	if (layout.shreds_revision != DConstants::INVALID_INDEX) {
		read += StringUtil::Format(" shreds=%llu", (unsigned long long)layout.shreds_revision);
	}
	return StringUtil::Format("layout revision %s, this build reads body=%llu shreds=%llu", read,
	                          (unsigned long long)JSONO_BODY_REVISION, (unsigned long long)JSONO_SHREDS_REVISION);
}

void JsonoRejectMalformedAnchoredRead(const LogicalType &type) {
	JsonoLayoutType layout;
	string reason;
	if (MatchJsonoLayoutType(type, layout, &reason) != JsonoLayoutMatch::NotJsono || !layout.lane_name_malformed) {
		return;
	}
	throw BinderException("jsono: this column carries the current jsono layout anchor, but %s. Reading it would "
	                      "silently treat it as an ordinary struct: a JSON read answers NULL for every path, and a "
	                      "JSONO conversion (jsono(), a cast, an INSERT into a JSONO column) rebuilds the document "
	                      "out of the raw layout fields — so it is refused. The value bytes are intact: RENAME the "
	                      "field back to the encoded lane name jsono_storage_type(<spec>) prints (ALTER TABLE ... "
	                      "RENAME COLUMN ...) and every read recovers in place. DROP COLUMN recovers only a lane "
	                      "that held no stripped values — after a drop, rows whose values were stripped into it "
	                      "are refused by the shred manifest. jsono_layout_diagnose(...) gives this diagnosis in "
	                      "SQL",
	                      reason);
}

void JsonoRejectForeignLayout(const LogicalType &type, const string &context) {
	JsonoLayoutType layout;
	if (MatchJsonoLayoutType(type, layout) != JsonoLayoutMatch::Foreign) {
		return;
	}
	throw InvalidInputException(
	    "%s: JSONO value carries %s. Reading it would "
	    "silently lose data, so it is refused. This build can still MOVE the value (SELECT *, INSERT ... SELECT, "
	    "COPY ... TO a Parquet file) but not read or render it: carry it to a build that reads that revision and "
	    "rewrite it there with jsono(...), or drop the data. Type: %s",
	    context, JsonoDescribeForeignLayout(layout), type.ToString());
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

bool JsonoIsShredSpillName(const string &name) {
	idx_t column;
	return TryParseSpillColumnName(name, column);
}

idx_t JsonoFindShredsFieldIndex(const LogicalType &shreds_type, const string &name) {
	auto &fields = StructType::GetChildTypes(shreds_type);
	for (idx_t i = 0; i < fields.size(); i++) {
		if (fields[i].first == name) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

// The rank of each string: its position in the byte-wise sorted list. Which order that IS depends on
// what the caller ranks, and the two framings of a lane genuinely differ. Ranked by ENCODED name it
// is structural path order — both stages of the codec are order-preserving, so sorting names IS
// sorting paths (see the lane-name boundary in jsono_path.hpp), and the spill bit numbering, the
// type's canonical field order and the sort-merges against the residual's byte-sorted document keys
// all rest on that. Ranked by logical path TEXT it is text order, which is what the manifest emits
// in and is not the same permutation (`$.a-c` precedes `$.a.b` as text, follows it structurally).
// The structural equality is a property of THAT serialization, not of encoding in general: re-verify
// it first if the codec ever changes.
vector<idx_t> JsonoRanksInByteOrder(const vector<string> &names) {
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
	vector<std::pair<std::string, std::string>> lanes;
	lanes.reserve(layout.shreds.size());
	for (auto &shred : layout.shreds) {
		lanes.emplace_back(shred.first, shred.second.ToString());
	}
	// Canonicalize order: the marker identifies the shred SET (names + types), not the field order. A
	// set-operation merged type lists the left branch's shreds first then the right's unique ones
	// (CombineStructTypes), so a writer stamping the canonical-order hash and a reader recomputing it
	// over a reorder-cast result type would otherwise disagree and lose the schema-identity match.
	std::sort(lanes.begin(), lanes.end());
	return jsono::HashShredSetIdentity(lanes);
}

LogicalType JsonoShreddedStructType(const vector<JsonoLaneSpec> &shreds) {
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
	// The one place a lane's physical field name is minted.
	for (auto &shred : shreds) {
		shred_children.emplace_back(JsonoEncodeLaneName(shred.path), shred.type);
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
		// ANY, and no jsono check in the bind: the whole point is answering for a value the grammar
		// REFUSED, which a JSONO-typed parameter could never receive. SPECIAL_HANDLING because the
		// answer is about the TYPE and reads no bytes: a NULL value has a type like any other, and
		// letting the executor short-circuit it would blank the diagnostic exactly on the all-NULL
		// column someone is trying to explain.
		ScalarFunctionSet set("jsono_layout_diagnose");
		ScalarFunction diagnose({LogicalType::ANY}, JsonoLayoutDiagnoseResultType(), JsonoLayoutDiagnoseExecute,
		                        JsonoLayoutDiagnoseBind);
		diagnose.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		set.AddFunction(std::move(diagnose));
		loader.RegisterFunction(set);
	}
	{
		// ANY and SPECIAL_HANDLING for the same reasons as jsono_layout_diagnose: the answer is about
		// the type, and a type the grammar refused — or a NULL value — must still be answerable.
		ScalarFunctionSet set("jsono_layout_lanes");
		ScalarFunction lanes({LogicalType::ANY}, JsonoLayoutLanesResultType(), JsonoLayoutLanesExecute,
		                     JsonoLayoutLanesBind);
		lanes.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		set.AddFunction(std::move(lanes));
		loader.RegisterFunction(set);
	}
}

} // namespace duckdb
