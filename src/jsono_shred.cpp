#include "jsono_shred.hpp"
#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
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

bool FindObjectKeyRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t &rank) {
	size_t lo = 0;
	size_t hi = layout.key_count;
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto key_slot = view.SlotAt(layout.key_start + mid);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		if (view.KeyAt(SlotPayload(key_slot)) < key) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	rank = lo;
	if (lo >= layout.key_count) {
		return false;
	}
	auto key_slot = view.SlotAt(layout.key_start + lo);
	return SlotTag(key_slot) == tag::KEY && view.KeyAt(SlotPayload(key_slot)) == key;
}

struct Location {
	Location() = default;
	Location(size_t pos_p, size_t string_cursor_p, bool found_p)
	    : pos(pos_p), string_cursor(string_cursor_p), found(found_p) {
	}

	size_t pos = 0;
	size_t string_cursor = 0;
	bool found = false;
};

Location LocatePath(const JsonoView &view, const vector<PathStep> &steps) {
	size_t pos = 0;
	size_t string_cursor = 0;
	for (auto &step : steps) {
		if (pos >= view.Slots()) {
			return {};
		}
		auto slot_tag = SlotTag(view.SlotAt(pos));
		if (step.kind == PathStepKind::Key) {
			if (slot_tag != tag::OBJ_START) {
				return {};
			}
			auto layout = ReadObjectLayout(view, pos);
			size_t rank = 0;
			if (!FindObjectKeyRank(view, layout, step.key, rank)) {
				return {};
			}
			size_t value_rank = 0;
			size_t value_pos = layout.value_start;
			size_t value_string_cursor = string_cursor;
			MoveObjectValueCursorToRank(view, layout, string_cursor, rank, value_rank, value_pos, value_string_cursor);
			pos = value_pos;
			string_cursor = value_string_cursor;
		} else if (step.kind == PathStepKind::Index) {
			if (slot_tag != tag::ARR_START) {
				return {};
			}
			auto end_pos = ReadArrayEndPos(view, pos);
			size_t element_pos = pos + 1;
			idx_t current = 0;
			while (element_pos < end_pos && current < step.index) {
				SkipValueFast(view, element_pos, string_cursor);
				current++;
			}
			if (element_pos >= end_pos || current < step.index) {
				return {};
			}
			pos = element_pos;
		} else {
			return {};
		}
	}
	return Location {pos, string_cursor, true};
}

// Render a scalar as the text `->>` would produce. Returns true for JSON null (the
// extract yields SQL NULL), false otherwise (text appended to `out`).
bool RenderExtractText(const JsonoScalar &scalar, std::string &out) {
	switch (scalar.kind) {
	case JsonoScalarKind::Null:
		return true;
	case JsonoScalarKind::String:
	case JsonoScalarKind::NumberText:
		out.append(scalar.text.data(), scalar.text.size());
		return false;
	case JsonoScalarKind::Int64:
		out.append(std::to_string(scalar.int_value));
		return false;
	case JsonoScalarKind::UInt64:
		out.append(std::to_string(scalar.uint_value));
		return false;
	case JsonoScalarKind::Double:
		EmitDouble(scalar.double_value, out);
		return false;
	case JsonoScalarKind::Dec60:
		AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
		return false;
	case JsonoScalarKind::Bool:
		out.append(scalar.bool_value ? "true" : "false");
		return false;
	}
	return true;
}

// Write the shred value for a field and return whether the original value is losslessly
// captured by the shred type (so the path may be stripped from the residual). A VARCHAR
// shred stores the `->>` text for any scalar but only a real JSON string round-trips; a
// typed shred stores the value only when its kind matches exactly.
bool WriteShred(const ShredField &field, const JsonoView &view, const Location &location, Vector &shred, idx_t row,
                std::string &scratch) {
	if (!location.found) {
		FlatVector::SetNull(shred, row, true);
		return false;
	}
	auto value_tag = SlotTag(view.SlotAt(location.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		FlatVector::SetNull(shred, row, true);
		return false;
	}
	size_t pos = location.pos;
	size_t string_cursor = location.string_cursor;
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
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
	FlatVector::SetNull(shred, row, true);
	return false;
}

void ApplyShredFields(Vector &input_vec, idx_t count, const vector<ShredField> &fields, Vector &result,
                      ShredLocalState &lstate) {
	JsonoVectorData input;
	InitJsonoVectorData(input_vec, count, input);

	Vector *slots_p, *key_heap_p, *string_heap_p, *skips_p;
	InitJsonoBodyWrite(result, slots_p, key_heap_p, string_heap_p, skips_p);
	auto &slots_vec = *slots_p;
	auto &key_heap_vec = *key_heap_p;
	auto &string_heap_vec = *string_heap_p;
	auto &skips_vec = *skips_p;
	vector<Vector *> shred_children;
	shred_children.reserve(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_children.push_back(&JsonoShredVector(result, f));
		shred_children.back()->SetVectorType(VectorType::FLAT_VECTOR);
	}
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);

	auto null_shred_row = [&](idx_t row) {
		SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
		for (auto &shred : shred_children) {
			FlatVector::SetNull(*shred, row, true);
		}
	};

	// Per-field manifest entry bytes (u16 path_len, path, u16 type_len, type), built once: the
	// per-row manifest concatenates the entries of the paths actually stripped from that row's
	// residual (fields are already in canonical sorted order, so the manifest is too).
	vector<std::string> manifest_entries(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		auto &entry = manifest_entries[f];
		auto append_lv = [&](const string &text) {
			if (text.size() > std::numeric_limits<uint16_t>::max()) {
				throw InvalidInputException("jsono shred: path or type name exceeds manifest limits");
			}
			uint16_t len = uint16_t(text.size());
			entry.append(reinterpret_cast<const char *>(&len), sizeof(len));
			entry.append(text);
		};
		append_lv(fields[f].path_name);
		append_lv(fields[f].logical_type.ToString());
	}
	std::string manifest;
	vector<idx_t> stripped_fields;

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRowStrict(input, row, blob)) {
			null_shred_row(row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			null_shred_row(row);
			continue;
		}

		FlatVector::Validity(result).SetValid(row);
		lstate.strip_paths.clear();
		stripped_fields.clear();
		for (idx_t f = 0; f < fields.size(); f++) {
			auto &field = fields[f];
			auto location = LocatePath(view, field.steps);
			bool strippable = WriteShred(field, view, location, *shred_children[f], row, lstate.text);
			if (strippable && IsObjectKeyPath(field.steps)) {
				lstate.strip_paths.push_back(&field.steps);
				stripped_fields.push_back(f);
			}
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
		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out, string_heap_out,
		                 skips_out, row, lstate.builder, manifest_ptr);
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
	JsonoLayoutType src_layout;
	TryParseJsonoLayoutType(input_vec.GetType(), src_layout);

	input_vec.Flatten(count);
	JsonoVectorData input;
	InitJsonoVectorData(input_vec, count, input);

	// Shreds the target drops (or retypes) fold back into the residual first, as one vectorized
	// partial overlay; the per-row pass then locates/strips against that merged residual, while
	// the manifest still reads from the ORIGINAL residual (the overlay output carries none).
	Vector returned(JsonoType(), count);
	JsonoVectorData merged;
	JsonoVectorData *residual_data = &input;
	if (!bind_data.return_src.empty()) {
		JsonoOverlayShredsToPlain(input_vec, count, bind_data.return_src, returned);
		InitJsonoVectorData(returned, count, merged);
		residual_data = &merged;
	}

	Vector *slots_p, *key_heap_p, *string_heap_p, *skips_p;
	InitJsonoBodyWrite(result, slots_p, key_heap_p, string_heap_p, skips_p);
	auto &slots_vec = *slots_p;
	auto &key_heap_vec = *key_heap_p;
	auto &string_heap_vec = *string_heap_p;
	auto &skips_vec = *skips_p;
	vector<Vector *> shred_out(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);

	for (idx_t f = 0; f < fields.size(); f++) {
		if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
			VectorOperations::Copy(JsonoShredVector(input_vec, bind_data.keep_src[f]), *shred_out[f], count, 0, 0);
		}
	}

	vector<std::string> manifest_entries(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		auto &entry = manifest_entries[f];
		auto append_lv = [&](const string &text) {
			if (text.size() > std::numeric_limits<uint16_t>::max()) {
				throw InvalidInputException("jsono shred: path or type name exceeds manifest limits");
			}
			uint16_t len = uint16_t(text.size());
			entry.append(reinterpret_cast<const char *>(&len), sizeof(len));
			entry.append(text);
		};
		append_lv(fields[f].path_name);
		append_lv(fields[f].logical_type.ToString());
	}

	// The source's shreds, verified against each row's manifest before reading the residual: a
	// narrowed input row (its manifest lists a shred the source type lost) must fail loud here too.
	vector<std::pair<std::string, std::string>> shred_signatures;
	shred_signatures.reserve(src_layout.shreds.size());
	for (auto &shred : src_layout.shreds) {
		shred_signatures.emplace_back(shred.first, shred.second.ToString());
	}
	std::vector<ShredManifestEntry> old_manifest;
	auto manifest_has_path = [&](const string &path) {
		for (auto &entry : old_manifest) {
			if (entry.path == nonstd::string_view(path.data(), path.size())) {
				return true;
			}
		}
		return false;
	};

	auto null_row = [&](idx_t row) {
		SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
		for (auto *shred : shred_out) {
			FlatVector::SetNull(*shred, row, true);
		}
	};

	std::string manifest;
	vector<idx_t> stripped_fields;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRowStrict(*residual_data, row, blob)) {
			null_row(row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			null_row(row);
			continue;
		}
		if (residual_data == &input) {
			VerifyShredManifest(view, shred_signatures, old_manifest);
		} else {
			// The merged residual carries no manifest; the kept shreds' stripped state lives in
			// the original row's manifest. The partial overlay already verified it.
			JsonoBlobRow src_blob;
			old_manifest.clear();
			if (ReadJsonoRowStrict(input, row, src_blob)) {
				JsonoView src_view = MakeJsonoView(src_blob);
				if (src_view.ParseHeader()) {
					src_view.ReadShredManifest(old_manifest);
				}
			}
		}

		FlatVector::Validity(result).SetValid(row);
		lstate.strip_paths.clear();
		stripped_fields.clear();
		for (idx_t f = 0; f < fields.size(); f++) {
			if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
				// The kept shred's stripped-or-not state carries over from the input row's manifest.
				if (manifest_has_path(fields[f].path_name)) {
					stripped_fields.push_back(f);
				}
				continue;
			}
			auto &field = fields[f];
			auto location = LocatePath(view, field.steps);
			bool strippable = WriteShred(field, view, location, *shred_out[f], row, lstate.text);
			if (strippable && IsObjectKeyPath(field.steps)) {
				lstate.strip_paths.push_back(&field.steps);
				stripped_fields.push_back(f);
			}
		}

		if (lstate.strip_paths.empty()) {
			// No path was stripped from this row's residual, so its blobs are unchanged. Copy them
			// verbatim instead of re-emitting through the builder. The original residual already
			// carries the right manifest (the kept entries are exactly the input row's manifest);
			// a merged residual carries none, so the kept entries are appended to its skips.
			slots_out[row] = WriteBlobInto(slots_vec, blob.slots.GetData(), blob.slots.GetSize());
			key_heap_out[row] = WriteBlobInto(key_heap_vec, blob.key_heap.GetData(), blob.key_heap.GetSize());
			string_heap_out[row] =
			    WriteBlobInto(string_heap_vec, blob.string_heap.GetData(), blob.string_heap.GetSize());
			if (residual_data != &input && !stripped_fields.empty()) {
				manifest.clear();
				manifest.append(blob.skips.GetData(), blob.skips.GetSize());
				uint32_t entry_count = uint32_t(stripped_fields.size());
				manifest.append(reinterpret_cast<const char *>(&entry_count), sizeof(entry_count));
				for (auto f : stripped_fields) {
					manifest.append(manifest_entries[f]);
				}
				skips_out[row] = WriteBlobInto(skips_vec, manifest.data(), manifest.size());
			} else {
				skips_out[row] = WriteBlobInto(skips_vec, blob.skips.GetData(), blob.skips.GetSize());
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
		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out, string_heap_out,
		                 skips_out, row, lstate.builder, manifest_ptr);
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

// jsono(text, shredding := spec): parse the JSON text into a plain jsono value, then shred it
// against the spec in the same call. The parse output is a throwaway plain vector; only the
// shredded result is materialized.
void JsonoShredFromTextExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ShredBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ShredLocalState>();
	auto count = args.size();
	bool input_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR || expr.children[0]->IsFoldable();
	Vector plain(JsonoType(), count);
	JsonoParseTextVector(args.data[0], count, plain);
	ApplyShredFields(plain, count, bind_data.fields, result, lstate);
	if (input_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

bool IsShredValueType(const LogicalType &type) {
	ShredPrimitive primitive;
	return TypeToShredPrimitive(type, primitive);
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
