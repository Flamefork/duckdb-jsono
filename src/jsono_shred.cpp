#include "jsono_shred.hpp"
#include "jsono.hpp"
#include "jsono_dom.hpp"
#include "jsono_locate.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace duckdb {

// Build the residual: `view` minus the located leaf of each path (defined in jsono_merge.cpp).
void JsonoEmitObjectStrippingPaths(const jsono::JsonoView &view,
                                   const std::vector<const std::vector<PathStep> *> &paths,
                                   jsono::JsonoBuilder &builder);

namespace {

using namespace jsono;
using namespace duckdb_yyjson;

enum class ShredPrimitive : uint8_t { Varchar, Bigint, Ubigint, Double, Boolean };

struct ShredField {
	string path_name; // canonical path; also the shred struct field name
	vector<PathStep> steps;
	ShredPrimitive primitive;
	LogicalType logical_type;
};

struct ShredBindData : public FunctionData {
	vector<ShredField> fields;
	LogicalType return_type;
	// Single-pass reshred over a shredded input: each target field either copies its source shred
	// (keep_src holds the source shred index) or extracts a new path from the residual
	// (INVALID_INDEX). Source shreds the target drops or retypes (return_src) are folded back
	// into the residual by a vectorized partial overlay before the per-row pass.
	bool reshred_active = false;
	vector<idx_t> keep_src;
	vector<idx_t> return_src;
	// One-pass text write (JsonoShredFromTextExecute): a trie over the object-key shred paths
	// lets pass 1 of the direct DOM writer capture and strip the leaves while sizing — the
	// throwaway plain materialization and the per-row locate/strip/re-emit disappear. Fields
	// whose path is not a pure key sequence never strip; they are filled from the written
	// residual per row (residual_fill_fields). Duplicate paths in the spec (e.g. '$.a' and
	// 'a') fall back to the two-pass path.
	bool one_pass_text = false;
	std::vector<jsono_dom::DomShredTrieNode> trie;
	vector<idx_t> residual_fill_fields;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ShredBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		// return_type encodes every shred's name (the path) and type, so it fully identifies the
		// spec; the reshred plan is derived from the argument type, compared via keep_src.
		auto &other = other_p.Cast<ShredBindData>();
		return return_type == other.return_type && reshred_active == other.reshred_active &&
		       keep_src == other.keep_src && return_src == other.return_src;
	}
};

struct ShredLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	std::string text;
	std::vector<const std::vector<PathStep> *> strip_paths;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<ShredLocalState>();
	}
};

ShredPrimitive ParseShredType(const string &type) {
	auto upper = StringUtil::Upper(type);
	if (upper == "VARCHAR") {
		return ShredPrimitive::Varchar;
	}
	if (upper == "BIGINT") {
		return ShredPrimitive::Bigint;
	}
	if (upper == "UBIGINT") {
		return ShredPrimitive::Ubigint;
	}
	if (upper == "DOUBLE") {
		return ShredPrimitive::Double;
	}
	if (upper == "BOOLEAN") {
		return ShredPrimitive::Boolean;
	}
	throw BinderException("jsono shred:unsupported shred type '%s'", type);
}

bool TypeToShredPrimitive(const LogicalType &type, ShredPrimitive &out) {
	switch (type.id()) {
	case LogicalTypeId::VARCHAR:
		// A JSON-aliased VARCHAR carries a JSON document (object/array/number), not a string
		// scalar; it embeds as a sub-tree in the residual and must not be lifted into a shred.
		if (type.HasAlias() && type.GetAlias() == LogicalType::JSON_TYPE_NAME) {
			return false;
		}
		out = ShredPrimitive::Varchar;
		return true;
	case LogicalTypeId::BIGINT:
		out = ShredPrimitive::Bigint;
		return true;
	case LogicalTypeId::UBIGINT:
		out = ShredPrimitive::Ubigint;
		return true;
	case LogicalTypeId::DOUBLE:
		out = ShredPrimitive::Double;
		return true;
	case LogicalTypeId::BOOLEAN:
		out = ShredPrimitive::Boolean;
		return true;
	default:
		return false;
	}
}

LogicalType ShredLogicalType(ShredPrimitive primitive) {
	switch (primitive) {
	case ShredPrimitive::Varchar:
		return LogicalType::VARCHAR;
	case ShredPrimitive::Bigint:
		return LogicalType::BIGINT;
	case ShredPrimitive::Ubigint:
		return LogicalType::UBIGINT;
	case ShredPrimitive::Double:
		return LogicalType::DOUBLE;
	case ShredPrimitive::Boolean:
		return LogicalType::BOOLEAN;
	}
	throw InternalException("jsono shred:unhandled shred type");
}

// A bare key names a literal top-level object key, not a JSONPath expression.
vector<PathStep> LiteralKeyPath(const string &name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, name, 0});
	return path;
}

vector<PathStep> ParseShredFieldPath(const string &path) {
	auto steps = path.size() > 0 && path[0] == '$' ? ParseJsonoPath(path, "jsono shred") : LiteralKeyPath(path);
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("jsono shred: wildcard paths are not supported ('%s')", path);
		}
	}
	return steps;
}

// A path may be stripped from the residual only if it is a pure object-key sequence: the
// residual emit removes a key at each step, and an array element cannot be re-filled by
// the object overlay on reconstruction. Index/wildcard paths stay preserve-only.
bool IsObjectKeyPath(const vector<PathStep> &steps) {
	for (auto &step : steps) {
		if (step.kind != PathStepKind::Key) {
			return false;
		}
	}
	return !steps.empty();
}

struct YyjsonDoc {
	explicit YyjsonDoc(yyjson_doc *doc_p) : doc(doc_p) {
	}
	~YyjsonDoc() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	YyjsonDoc(const YyjsonDoc &) = delete;
	YyjsonDoc &operator=(const YyjsonDoc &) = delete;

	yyjson_doc *doc;
};

// The shredding spec is a constant STRUCT mapping each path to its shred type string, e.g.
// {'$.kind': 'VARCHAR', '$.commit.seq': 'BIGINT'} — the field name is the path, the field
// value is the stringified shred type (same idea as read_json's `columns` and Parquet's
// SHREDDING option).
unique_ptr<ShredBindData> ParseShredSpec(const Value &spec) {
	if (spec.IsNull() || spec.type().id() != LogicalTypeId::STRUCT) {
		throw BinderException(
		    "jsono shred: shredding spec must be a non-NULL STRUCT mapping path to type, e.g. {'$.kind': 'VARCHAR'}");
	}
	auto &child_types = StructType::GetChildTypes(spec.type());
	auto &child_values = StructValue::GetChildren(spec);

	auto bind_data = make_uniq<ShredBindData>();

	for (idx_t i = 0; i < child_types.size(); i++) {
		auto &path = child_types[i].first;
		auto type_value = child_values[i];
		if (type_value.IsNull() || !type_value.DefaultTryCastAs(LogicalType::VARCHAR)) {
			throw BinderException("jsono shred: type for '%s' must be a type string", path);
		}
		auto type_name = StringValue::Get(type_value);
		if (path == "body") {
			// The shreds sit beside the `body` field inside the layout struct, so a bare 'body'
			// shred name would duplicate it. The path form is a different field name and is fine.
			throw BinderException("jsono shred: a shred cannot be named 'body' (the residual field); "
			                      "shred the JSON key through its path form '$.body'");
		}
		if (path == JsonoValueCompleteName()) {
			throw BinderException("jsono shred: '%s' is a reserved layout field name", path);
		}
		ShredField field;
		field.path_name = path;
		field.steps = ParseShredFieldPath(path);
		field.primitive = ParseShredType(type_name);
		field.logical_type = ShredLogicalType(field.primitive);
		bind_data->fields.push_back(std::move(field));
	}
	if (bind_data->fields.empty()) {
		throw BinderException("jsono shred: empty shredding spec");
	}
	// Canonical shred order (sorted by path) makes the shredded type a pure function of the shred set,
	// not of spec order: the execution writes shreds by index, so the field write order and the type's
	// shred order must agree — sort both here. The fingerprint suffix is already order-independent.
	std::sort(bind_data->fields.begin(), bind_data->fields.end(),
	          [](const ShredField &a, const ShredField &b) { return a.path_name < b.path_name; });
	child_list_t<LogicalType> shreds;
	for (auto &field : bind_data->fields) {
		shreds.emplace_back(field.path_name, field.logical_type);
	}
	// JsonoShreddedStructType names every field (blobs included) with the shred-set fingerprint, so a
	// column declared from the same shreds is byte-identical and a mismatched cast fails loud.
	bind_data->return_type = JsonoShreddedStructType(shreds);

	bind_data->one_pass_text = true;
	bind_data->trie.emplace_back();
	for (idx_t f = 0; f < bind_data->fields.size(); f++) {
		auto &field = bind_data->fields[f];
		if (!IsObjectKeyPath(field.steps)) {
			bind_data->residual_fill_fields.push_back(f);
			continue;
		}
		uint32_t node = 0;
		for (auto &step : field.steps) {
			uint32_t next = std::numeric_limits<uint32_t>::max();
			for (auto &edge : bind_data->trie[node].children) {
				if (edge.first == step.key) {
					next = edge.second;
					break;
				}
			}
			if (next == std::numeric_limits<uint32_t>::max()) {
				bind_data->trie.emplace_back();
				next = uint32_t(bind_data->trie.size() - 1);
				bind_data->trie[node].children.emplace_back(step.key, next);
			}
			node = next;
		}
		if (bind_data->trie[node].field >= 0) {
			// Two spec entries name the same key path (e.g. '$.a' and 'a'); the per-row
			// locate-and-strip path handles them with the documented semantics.
			bind_data->one_pass_text = false;
			break;
		}
		bind_data->trie[node].field = int64_t(f);
	}
	return bind_data;
}

unique_ptr<FunctionData> JsonoShredBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->GetAlias() != "shredding") {
		throw BinderException("jsono(): unknown argument '%s' (pass shredding := {'<path>': '<type>', ...})",
		                      arguments[1]->GetAlias());
	}
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("jsono shred: shredding spec must be constant");
	}
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto spec_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	auto bind_data = ParseShredSpec(spec_value);
	bound_function.return_type = bind_data->return_type;
	auto &arg_type = arguments[0]->return_type;
	if (arg_type.id() == LogicalTypeId::STRUCT) {
		JsonoLayoutType src_layout;
		if (!TryParseJsonoLayoutType(arg_type, src_layout)) {
			throw BinderException("jsono shred: value must be JSON text or a jsono value");
		}
		// Reshred a shredded value in one pass: surviving shreds copy through, dropped/retyped
		// shreds (return_src) fold back into the residual via a partial overlay, and new target
		// paths extract from that residual. A plain value takes the plain shred path.
		if (src_layout.kind == JsonoLayoutKind::Shredded) {
			bind_data->keep_src.assign(bind_data->fields.size(), DConstants::INVALID_INDEX);
			vector<bool> src_kept(src_layout.shreds.size(), false);
			for (idx_t f = 0; f < bind_data->fields.size(); f++) {
				for (idx_t k = 0; k < src_layout.shreds.size(); k++) {
					if (src_layout.shreds[k].first == bind_data->fields[f].path_name &&
					    src_layout.shreds[k].second == bind_data->fields[f].logical_type) {
						bind_data->keep_src[f] = k;
						src_kept[k] = true;
						break;
					}
				}
			}
			for (idx_t k = 0; k < src_layout.shreds.size(); k++) {
				if (!src_kept[k]) {
					bind_data->return_src.push_back(k);
				}
			}
			bind_data->reshred_active = true;
			bound_function.arguments[0] = arg_type;
		} else {
			bound_function.arguments[0] = JsonoType();
		}
	}
	return std::move(bind_data);
}

struct Location {
	Location() = default;
	Location(const JsonoCursor &cursor_p, bool found_p) : cursor(cursor_p), found(found_p) {
	}

	JsonoCursor cursor;
	bool found = false;
};

Location LocatePath(const JsonoView &view, const vector<PathStep> &steps) {
	JsonoCursor cursor;
	if (!LocatePathSteps(nullptr, 0, steps, view, cursor)) {
		return {};
	}
	return Location {cursor, true};
}

// Write the shred value for a field and return whether the original value is losslessly
// captured by the shred type (so the path may be stripped from the residual). A VARCHAR
// shred stores the `->>` text for any scalar but only a real JSON string round-trips; a
// typed shred stores the value only when its kind matches exactly.
// `diverted` (optional out) is set true only when a PRESENT scalar fails a typed shred's kind gate
// and is kept in the residual instead — the case-B diversion that makes a bare struct_extract read
// NULL where the residual `->>`+CAST would yield the value. Absence, a present container, and a
// VARCHAR shred (which renders any scalar) all leave it false: a bare read is then correct.
bool WriteShred(const ShredField &field, const JsonoView &view, const Location &location, Vector &shred, idx_t row,
                std::string &scratch, bool *diverted = nullptr) {
	if (diverted) {
		*diverted = false;
	}
	if (!location.found) {
		FlatVector::SetNull(shred, row, true);
		return false;
	}
	auto value_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		FlatVector::SetNull(shred, row, true);
		return false;
	}
	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	switch (field.primitive) {
	case ShredPrimitive::Varchar: {
		scratch.clear();
		if (RenderExtractText(scalar, scratch)) {
			FlatVector::SetNull(shred, row, true);
			return false;
		}
		FlatVector::Validity(shred).SetValid(row);
		FlatVector::GetData<string_t>(shred)[row] = StringVector::AddString(shred, scratch.data(), scratch.size());
		return scalar.kind == JsonoScalarKind::String;
	}
	case ShredPrimitive::Bigint:
		if (scalar.kind == JsonoScalarKind::Int64) {
			FlatVector::Validity(shred).SetValid(row);
			FlatVector::GetData<int64_t>(shred)[row] = scalar.int_value;
			return true;
		}
		break;
	case ShredPrimitive::Ubigint:
		if (scalar.kind == JsonoScalarKind::UInt64) {
			FlatVector::Validity(shred).SetValid(row);
			FlatVector::GetData<uint64_t>(shred)[row] = scalar.uint_value;
			return true;
		}
		if (scalar.kind == JsonoScalarKind::Int64 && scalar.int_value >= 0) {
			FlatVector::Validity(shred).SetValid(row);
			FlatVector::GetData<uint64_t>(shred)[row] = uint64_t(scalar.int_value);
			return true;
		}
		break;
	case ShredPrimitive::Double:
		if (scalar.kind == JsonoScalarKind::Double) {
			FlatVector::Validity(shred).SetValid(row);
			FlatVector::GetData<double>(shred)[row] = scalar.double_value;
			return true;
		}
		break;
	case ShredPrimitive::Boolean:
		if (scalar.kind == JsonoScalarKind::Bool) {
			FlatVector::Validity(shred).SetValid(row);
			FlatVector::GetData<bool>(shred)[row] = scalar.bool_value;
			return true;
		}
		break;
	}
	// Reached only by a typed shred whose present scalar did not match its kind gate: the value
	// stays in the residual, so this row is not value-complete for that shred.
	if (diverted) {
		*diverted = true;
	}
	FlatVector::SetNull(shred, row, true);
	return false;
}

void ApplyShredFields(Vector &input_vec, idx_t count, const vector<ShredField> &fields, Vector &result,
                      ShredLocalState &lstate) {
	// The input is plain JSONO, which never carries a manifest of its own: the reader fails
	// loud on a narrowed row, whose re-emit below would otherwise replace the old manifest
	// with a fresh one and launder the loss into permanent undetectability.
	JsonoRowReader reader;
	reader.Init(input_vec, count);

	JsonoBodyWriter writer;
	writer.Init(result);
	// Value-complete marker, computed per row: NULL when a present scalar diverted into the residual
	// (case B), non-NULL otherwise. WriteShred reports the diversion below.
	Vector *vc_out = &JsonoVcVector(result);
	vc_out->SetVectorType(VectorType::FLAT_VECTOR);
	vector<Vector *> shred_children;
	shred_children.reserve(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_children.push_back(&JsonoShredVector(result, f));
		shred_children.back()->SetVectorType(VectorType::FLAT_VECTOR);
	}

	auto null_shred_row = [&](idx_t row) {
		writer.SetRowNull(row);
		for (auto &shred : shred_children) {
			FlatVector::SetNull(*shred, row, true);
		}
		// A NULL jsono row diverts nothing: a bare struct_extract reads the correct NULL.
		FlatVector::GetData<bool>(*vc_out)[row] = true;
	};

	// Per-field manifest entry bytes, built once: the per-row manifest concatenates the entries
	// of the paths actually stripped from that row's residual (fields are already in canonical
	// sorted order, so the manifest is too).
	vector<std::string> manifest_entries(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(fields[f].path_name, fields[f].logical_type);
	}
	std::string manifest;
	vector<idx_t> stripped_fields;

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			null_shred_row(row);
			continue;
		}

		FlatVector::Validity(result).SetValid(row);
		lstate.strip_paths.clear();
		stripped_fields.clear();
		bool diverted_row = false;
		for (idx_t f = 0; f < fields.size(); f++) {
			auto &field = fields[f];
			auto location = LocatePath(view, field.steps);
			bool diverted = false;
			bool strippable = WriteShred(field, view, location, *shred_children[f], row, lstate.text, &diverted);
			diverted_row = diverted_row || diverted;
			if (strippable && IsObjectKeyPath(field.steps)) {
				lstate.strip_paths.push_back(&field.steps);
				stripped_fields.push_back(f);
			}
		}
		if (diverted_row) {
			FlatVector::SetNull(*vc_out, row, true);
		} else {
			FlatVector::GetData<bool>(*vc_out)[row] = true;
		}

		lstate.builder.Reset();
		JsonoEmitObjectStrippingPaths(view, lstate.strip_paths, lstate.builder);
		const std::string *manifest_ptr = nullptr;
		if (!stripped_fields.empty()) {
			manifest.clear();
			uint32_t entry_count = uint32_t(stripped_fields.size());
			manifest.append(reinterpret_cast<const char *>(&entry_count), sizeof(entry_count));
			for (auto f : stripped_fields) {
				manifest.append(manifest_entries[f]);
			}
			manifest_ptr = &manifest;
		}
		writer.WriteRow(row, lstate.builder, manifest_ptr);
	}
}

// Single-pass reshred of a shredded input whose shreds all survive into the target (see
// ShredBindData::reshred_active): kept shreds are copied vector-at-a-time, only the target's new
// paths are located in and stripped from the residual, and the manifest carries over the kept
// entries plus the newly stripped ones — no reconstruct-to-plain materialization, no re-locating
// of paths that were already shredded.
void ApplyReshredShredded(Vector &input_vec, idx_t count, const ShredBindData &bind_data, Vector &result,
                          ShredLocalState &lstate) {
	auto &fields = bind_data.fields;

	input_vec.Flatten(count);

	// Shreds the target drops (or retypes) fold back into the residual first, as one vectorized
	// partial overlay; the per-row pass then locates/strips against that merged residual, while
	// the manifest still reads from the ORIGINAL residual (the overlay output carries none).
	// Reading the original residual, the reader verifies each row's manifest against the
	// source's shreds — a narrowed input row fails loud here too; the merged residual is plain
	// and unmanifested by construction (the overlay verified the source itself).
	Vector returned(JsonoType(), count);
	JsonoRowReader reader;
	bool reads_merged = !bind_data.return_src.empty();
	if (reads_merged) {
		JsonoOverlayShredsToPlain(input_vec, count, bind_data.return_src, returned);
		reader.Init(returned, count);
	} else {
		reader.Init(input_vec, count);
	}

	JsonoBodyWriter writer;
	writer.Init(result);
	// Value-complete marker, recomputed per row for the new shred set: NULL when a present scalar
	// stays in the residual for one of this target's typed shreds (case B), non-NULL otherwise.
	Vector *vc_out = &JsonoVcVector(result);
	vc_out->SetVectorType(VectorType::FLAT_VECTOR);
	vector<Vector *> shred_out(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}

	for (idx_t f = 0; f < fields.size(); f++) {
		if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
			VectorOperations::Copy(JsonoShredVector(input_vec, bind_data.keep_src[f]), *shred_out[f], count, 0, 0);
		}
	}

	vector<std::string> manifest_entries(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(fields[f].path_name, fields[f].logical_type);
	}

	const std::vector<ShredManifestEntry> *old_manifest = nullptr;
	std::vector<ShredManifestEntry> src_manifest;
	auto manifest_has_path = [&](const string &path) {
		for (auto &entry : *old_manifest) {
			if (entry.path == nonstd::string_view(path.data(), path.size())) {
				return true;
			}
		}
		return false;
	};

	auto null_row = [&](idx_t row) {
		writer.SetRowNull(row);
		for (auto *shred : shred_out) {
			FlatVector::SetNull(*shred, row, true);
		}
		// A NULL jsono row diverts nothing: a bare struct_extract reads the correct NULL.
		FlatVector::GetData<bool>(*vc_out)[row] = true;
	};

	JsonoVectorData input;
	InitJsonoVectorData(input_vec, count, input);
	std::string manifest;
	vector<idx_t> stripped_fields;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			null_row(row);
			continue;
		}
		if (!reads_merged) {
			old_manifest = &reader.RowManifest(view);
		} else {
			// The merged residual carries no manifest; the kept shreds' stripped state lives in
			// the original row's manifest. The partial overlay already verified it.
			JsonoBlobRow src_blob;
			src_manifest.clear();
			if (ReadJsonoRowStrict(input, row, src_blob)) {
				JsonoView src_view = MakeJsonoView(src_blob);
				if (src_view.ParseHeader()) {
					src_view.ReadShredManifest(src_manifest);
				}
			}
			old_manifest = &src_manifest;
		}

		FlatVector::Validity(result).SetValid(row);
		lstate.strip_paths.clear();
		stripped_fields.clear();
		bool diverted_row = false;
		for (idx_t f = 0; f < fields.size(); f++) {
			if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
				// The kept shred's stripped-or-not state carries over from the input row's manifest.
				if (manifest_has_path(fields[f].path_name)) {
					stripped_fields.push_back(f);
				}
				// A kept TYPED shred that is NULL this row is case A (absent) unless a present scalar
				// sits at its path in the residual — which can only be a value the source diverted
				// (case B), since a captured value would have been stripped. That makes the row not
				// value-complete: a bare struct_extract would miss the residual value.
				if (fields[f].primitive != ShredPrimitive::Varchar &&
				    !FlatVector::Validity(*shred_out[f]).RowIsValid(row)) {
					auto location = LocatePath(view, fields[f].steps);
					if (location.found) {
						auto value_tag = SlotTag(view.SlotAt(location.cursor.pos));
						if (value_tag != tag::OBJ_START && value_tag != tag::ARR_START) {
							diverted_row = true;
						}
					}
				}
				continue;
			}
			auto &field = fields[f];
			auto location = LocatePath(view, field.steps);
			bool diverted = false;
			bool strippable = WriteShred(field, view, location, *shred_out[f], row, lstate.text, &diverted);
			diverted_row = diverted_row || diverted;
			if (strippable && IsObjectKeyPath(field.steps)) {
				lstate.strip_paths.push_back(&field.steps);
				stripped_fields.push_back(f);
			}
		}
		if (diverted_row) {
			FlatVector::SetNull(*vc_out, row, true);
		} else {
			FlatVector::GetData<bool>(*vc_out)[row] = true;
		}

		if (lstate.strip_paths.empty()) {
			// No path was stripped from this row's residual, so its blobs are unchanged. Copy them
			// verbatim instead of re-emitting through the builder. The original residual already
			// carries the right manifest (the kept entries are exactly the input row's manifest);
			// a merged residual carries none, so the kept entries are appended to its skips.
			writer.data[BODY_SLOTS][row] = WriteBlobInto(writer.Slots(), blob.slots.GetData(), blob.slots.GetSize());
			writer.data[BODY_KEY_HEAP][row] =
			    WriteBlobInto(writer.KeyHeap(), blob.key_heap.GetData(), blob.key_heap.GetSize());
			writer.data[BODY_STRING_HEAP][row] =
			    WriteBlobInto(writer.StringHeap(), blob.string_heap.GetData(), blob.string_heap.GetSize());
			writer.data[BODY_LENGTHS][row] =
			    WriteBlobInto(writer.Lengths(), blob.lengths.GetData(), blob.lengths.GetSize());
			writer.data[BODY_NUMS][row] = WriteBlobInto(writer.Nums(), blob.nums.GetData(), blob.nums.GetSize());
			if (reads_merged && !stripped_fields.empty()) {
				manifest.clear();
				manifest.append(blob.skips.GetData(), blob.skips.GetSize());
				uint32_t entry_count = uint32_t(stripped_fields.size());
				manifest.append(reinterpret_cast<const char *>(&entry_count), sizeof(entry_count));
				for (auto f : stripped_fields) {
					manifest.append(manifest_entries[f]);
				}
				writer.data[BODY_SKIPS][row] = WriteBlobInto(writer.Skips(), manifest.data(), manifest.size());
			} else {
				writer.data[BODY_SKIPS][row] =
				    WriteBlobInto(writer.Skips(), blob.skips.GetData(), blob.skips.GetSize());
			}
			continue;
		}

		lstate.builder.Reset();
		JsonoEmitObjectStrippingPaths(view, lstate.strip_paths, lstate.builder);
		const std::string *manifest_ptr = nullptr;
		if (!stripped_fields.empty()) {
			manifest.clear();
			uint32_t entry_count = uint32_t(stripped_fields.size());
			manifest.append(reinterpret_cast<const char *>(&entry_count), sizeof(entry_count));
			for (auto f : stripped_fields) {
				manifest.append(manifest_entries[f]);
			}
			manifest_ptr = &manifest;
		}
		writer.WriteRow(row, lstate.builder, manifest_ptr);
	}
}

void JsonoShredExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ShredBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ShredLocalState>();
	bool input_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR || expr.children[0]->IsFoldable();
	if (bind_data.reshred_active) {
		ApplyReshredShredded(args.data[0], args.size(), bind_data, result, lstate);
	} else {
		ApplyShredFields(args.data[0], args.size(), bind_data.fields, result, lstate);
	}
	if (input_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// jsono(text, shredding := spec): parse the JSON text and shred it in one pass — the direct
// DOM writer's sizing pass captures the shred leaves and strips the lossless ones from the
// emit plan, so the full plain value is never materialized and no path is located twice.
// The rare present-but-lossy VARCHAR shred (`->>` text of a number/bool) is filled from the
// written residual with the exact two-pass semantics, as are non-object-key paths.
void JsonoShredFromTextExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ShredBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ShredLocalState>();
	auto count = args.size();
	bool input_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR || expr.children[0]->IsFoldable();
	if (!bind_data.one_pass_text) {
		Vector plain(JsonoType(), count);
		JsonoParseTextVector(args.data[0], count, plain);
		ApplyShredFields(plain, count, bind_data.fields, result, lstate);
		if (input_constant) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	auto &fields = bind_data.fields;
	UnifiedVectorFormat input_fmt;
	args.data[0].ToUnifiedFormat(count, input_fmt);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_fmt);

	JsonoBodyWriter writer;
	writer.Init(result);
	vector<Vector *> shred_out(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}
	// Value-complete marker (every shredded result carries one): NULL on a row that kept a present
	// scalar in the residual (case B), non-NULL otherwise. A column whose marker carries no NULL has
	// no such diversion, so the optimizer reads its typed shreds as a bare struct_extract. For a
	// VARCHAR-only shred set it stays uniformly non-NULL (a VARCHAR shred never diverts).
	Vector *vc_out = &JsonoVcVector(result);
	vc_out->SetVectorType(VectorType::FLAT_VECTOR);

	vector<std::string> manifest_entries(fields.size());
	jsono_dom::DomShredContext ctx;
	ctx.nodes = &bind_data.trie;
	ctx.manifest_entries = &manifest_entries;
	ctx.kinds.resize(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(fields[f].path_name, fields[f].logical_type);
		switch (fields[f].primitive) {
		case ShredPrimitive::Varchar:
			ctx.kinds[f] = jsono_dom::DomShredKind::Varchar;
			break;
		case ShredPrimitive::Bigint:
			ctx.kinds[f] = jsono_dom::DomShredKind::Int64;
			break;
		case ShredPrimitive::Ubigint:
			ctx.kinds[f] = jsono_dom::DomShredKind::Uint64;
			break;
		case ShredPrimitive::Double:
			ctx.kinds[f] = jsono_dom::DomShredKind::Double;
			break;
		case ShredPrimitive::Boolean:
			ctx.kinds[f] = jsono_dom::DomShredKind::Boolean;
			break;
		}
	}

	jsono_dom::YyjsonAllocator parser;
	jsono_dom::DomDirectState dom;

	// Fill a shred from the row's written residual: the value was kept there, so the
	// two-pass locate + render path applies verbatim.
	auto fill_from_residual = [&](const JsonoView &view, idx_t f, idx_t row) {
		auto location = LocatePath(view, fields[f].steps);
		WriteShred(fields[f], view, location, *shred_out[f], row, lstate.text);
	};

	for (idx_t row = 0; row < count; row++) {
		auto idx = input_fmt.sel->get_index(row);
		if (!input_fmt.validity.RowIsValid(idx)) {
			writer.SetRowNull(row);
			for (idx_t f = 0; f < fields.size(); f++) {
				FlatVector::SetNull(*shred_out[f], row, true);
			}
			// A NULL jsono row has no value to divert: its shreds read NULL via a bare struct_extract
			// (struct-null propagates), which is the correct result, so the marker stays non-NULL so a
			// table of value-complete rows plus NULL rows is still value-complete.
			if (vc_out) {
				FlatVector::GetData<bool>(*vc_out)[row] = true;
			}
			continue;
		}
		auto input = inputs[idx];
		if (input.GetSize() == 0) {
			throw InvalidInputException("jsono: input is empty");
		}
		yyjson_read_err err;
		auto doc = jsono_dom::ReadJsonoDoc(input, parser.alc, err);
		if (!doc) {
			throw InvalidInputException("jsono: invalid JSON - %s", err.msg);
		}
		bool needs_residual_fill = !bind_data.residual_fill_fields.empty();
		try {
			jsono_dom::EmitDomRowDirect(yyjson_doc_get_root(doc), dom, writer, row, &ctx);
			FlatVector::Validity(result).SetValid(row);
			bool diverted = false;
			for (idx_t f = 0; f < fields.size(); f++) {
				auto &cap = ctx.captures[f];
				diverted = diverted || cap.diverted_scalar;
				switch (cap.state) {
				case jsono_dom::DomShredCapture::State::Missing:
					FlatVector::SetNull(*shred_out[f], row, true);
					break;
				case jsono_dom::DomShredCapture::State::String:
					FlatVector::GetData<string_t>(*shred_out[f])[row] =
					    StringVector::AddString(*shred_out[f], cap.text.data(), cap.text.size());
					break;
				case jsono_dom::DomShredCapture::State::Int:
					FlatVector::GetData<int64_t>(*shred_out[f])[row] = cap.int_value;
					break;
				case jsono_dom::DomShredCapture::State::Uint:
					FlatVector::GetData<uint64_t>(*shred_out[f])[row] = cap.uint_value;
					break;
				case jsono_dom::DomShredCapture::State::Bool:
					FlatVector::GetData<bool>(*shred_out[f])[row] = cap.bool_value;
					break;
				case jsono_dom::DomShredCapture::State::ResidualFill:
					needs_residual_fill = true;
					break;
				}
			}
			if (vc_out) {
				if (diverted) {
					FlatVector::SetNull(*vc_out, row, true);
				} else {
					FlatVector::GetData<bool>(*vc_out)[row] = true;
				}
			}
		} catch (...) {
			yyjson_doc_free(doc);
			throw;
		}
		yyjson_doc_free(doc);
		if (needs_residual_fill) {
			JsonoBlobRow blob {writer.data[BODY_SLOTS][row],       writer.data[BODY_KEY_HEAP][row],
			                   writer.data[BODY_STRING_HEAP][row], writer.data[BODY_SKIPS][row],
			                   writer.data[BODY_LENGTHS][row],     writer.data[BODY_NUMS][row]};
			JsonoView view = MakeJsonoView(blob);
			if (view.ParseHeader()) {
				for (auto f : bind_data.residual_fill_fields) {
					fill_from_residual(view, f, row);
				}
				for (idx_t f = 0; f < fields.size(); f++) {
					if (ctx.captures[f].state == jsono_dom::DomShredCapture::State::ResidualFill) {
						fill_from_residual(view, f, row);
					}
				}
			}
		}
	}
	if (input_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

bool IsShredValueType(const LogicalType &type) {
	ShredPrimitive primitive;
	return TypeToShredPrimitive(type, primitive);
}

std::string JsonoShredManifestEntry(const string &path, const LogicalType &type) {
	std::string entry;
	auto append_lv = [&](const string &text) {
		if (text.size() > std::numeric_limits<uint16_t>::max()) {
			throw InvalidInputException("jsono shred: path or type name exceeds manifest limits");
		}
		uint16_t len = uint16_t(text.size());
		entry.append(reinterpret_cast<const char *>(&len), sizeof(len));
		entry.append(text);
	};
	append_lv(path);
	append_lv(type.ToString());
	return entry;
}

void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &shreds,
                        Vector &result) {
	vector<ShredField> fields;
	fields.reserve(shreds.size());
	for (auto &shred : shreds) {
		ShredField field;
		field.path_name = shred.first;
		field.steps = ParseShredFieldPath(shred.first);
		ShredPrimitive primitive;
		if (!TypeToShredPrimitive(shred.second, primitive)) {
			throw InternalException("jsono shred field '%s' has a non-shred type", shred.first);
		}
		field.primitive = primitive;
		field.logical_type = shred.second;
		fields.push_back(std::move(field));
	}
	ShredLocalState lstate;
	ApplyShredFields(input, count, fields, result, lstate);
}

// The jsono(jsono, shredding := spec) overload, exposed so the optimizer's cast normalization
// can reshred a value to a target shred set without a catalog lookup. Declared on a generic
// STRUCT so a shredded value reaches the bind natively: when every source shred survives into
// the target the bind plans the single-pass reshred; otherwise it redeclares the argument as
// plain JSONO and the binder's reconstruct cast feeds the plain shred path. The injected
// expression binds through the regular JsonoShredBind (a constant STRUCT spec), so it re-binds
// identically on plan deserialization.
ScalarFunction JsonoShredFromJsonoFunction() {
	ScalarFunction from_jsono("jsono", {LogicalTypeId::STRUCT, LogicalType::ANY}, LogicalType::ANY, JsonoShredExecute,
	                          JsonoShredBind, nullptr, nullptr, ShredLocalState::Init);
	from_jsono.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	from_jsono.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return from_jsono;
}

// Shredding is exposed as a named argument on the jsono() constructor:
// `shredding := {'<path>': '<type>', ...}` (a constant STRUCT, like Parquet's SHREDDING).
// The primary form parses text and shreds in one pass; the secondary form shreds an existing
// plain jsono value. Both reuse the same bind (spec parse + return type) and shred writer. The
// spec slot is ANY so the STRUCT literal binds; the bind validates it is a STRUCT.
void RegisterJsonoShred(ExtensionLoader &loader) {
	ScalarFunctionSet set("jsono");

	// jsono(text, shredding := spec) — the primary entry point: parse + shred.
	ScalarFunction from_text({LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::ANY, JsonoShredFromTextExecute,
	                         JsonoShredBind, nullptr, nullptr, ShredLocalState::Init);
	from_text.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	from_text.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	set.AddFunction(from_text);

	// jsono(jsono, shredding := spec) — shred a value that is already plain jsono.
	set.AddFunction(JsonoShredFromJsonoFunction());

	loader.RegisterFunction(set);
}

} // namespace duckdb
