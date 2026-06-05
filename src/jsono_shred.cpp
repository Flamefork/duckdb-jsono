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
	string path_name; // canonical path; also the lane struct field name
	vector<PathStep> steps;
	ShredPrimitive primitive;
	LogicalType logical_type;
};

struct ShredBindData : public FunctionData {
	string spec;
	vector<ShredField> fields;
	LogicalType return_type;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ShredBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		return spec == other_p.Cast<ShredBindData>().spec;
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
	throw BinderException("jsono_shred: unsupported lane type '%s'", type);
}

bool TypeToShredPrimitive(const LogicalType &type, ShredPrimitive &out) {
	switch (type.id()) {
	case LogicalTypeId::VARCHAR:
		// A JSON-aliased VARCHAR carries a JSON document (object/array/number), not a string
		// scalar; it embeds as a sub-tree in the residual and must not be lifted into a lane.
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
	throw InternalException("jsono_shred: unhandled lane type");
}

// A bare key names a literal top-level object key, not a JSONPath expression.
vector<PathStep> LiteralKeyPath(const string &name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, name, 0});
	return path;
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

unique_ptr<ShredBindData> ParseShredSpec(const string &spec) {
	yyjson_read_err read_err;
	YyjsonDoc holder(
	    yyjson_read_opts(const_cast<char *>(spec.data()), spec.size(), YYJSON_READ_NOFLAG, nullptr, &read_err));
	if (!holder.doc) {
		throw BinderException("jsono_shred: invalid spec JSON: %s", read_err.msg);
	}
	auto root = yyjson_doc_get_root(holder.doc);
	if (!yyjson_is_obj(root)) {
		throw BinderException("jsono_shred: spec must be a JSON object mapping path to type");
	}

	auto bind_data = make_uniq<ShredBindData>();
	child_list_t<LogicalType> children;
	children.emplace_back("jsono_slots", LogicalType::BLOB);
	children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	children.emplace_back("jsono_skips", LogicalType::BLOB);

	yyjson_obj_iter iter = yyjson_obj_iter_with(root);
	yyjson_val *key;
	while ((key = yyjson_obj_iter_next(&iter))) {
		auto value = yyjson_obj_iter_get_val(key);
		string path(yyjson_get_str(key), yyjson_get_len(key));
		if (!yyjson_is_str(value)) {
			throw BinderException("jsono_shred: lane type for '%s' must be a type string", path);
		}
		string type_name(yyjson_get_str(value), yyjson_get_len(value));
		ShredField field;
		field.path_name = path;
		field.steps = path.size() > 0 && path[0] == '$' ? ParseJsonoPath(path, "jsono_shred") : LiteralKeyPath(path);
		for (auto &step : field.steps) {
			if (step.kind == PathStepKind::Wildcard) {
				throw BinderException("jsono_shred: wildcard paths are not supported ('%s')", path);
			}
		}
		field.primitive = ParseShredType(type_name);
		field.logical_type = ShredLogicalType(field.primitive);
		children.emplace_back(path, field.logical_type);
		bind_data->fields.push_back(std::move(field));
	}
	if (bind_data->fields.empty()) {
		throw BinderException("jsono_shred: empty spec");
	}
	bind_data->return_type = LogicalType::STRUCT(std::move(children));
	return bind_data;
}

unique_ptr<FunctionData> JsonoShredBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("jsono_shred: spec must be constant");
	}
	auto spec_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (spec_value.IsNull() || !spec_value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		throw BinderException("jsono_shred: spec must be a non-NULL string");
	}
	auto spec = StringValue::Get(spec_value);
	auto bind_data = ParseShredSpec(spec);
	bind_data->spec = std::move(spec);
	bound_function.return_type = bind_data->return_type;
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

// Write the lane value for a field and return whether the original value is losslessly
// captured by the lane type (so the path may be stripped from the residual). A VARCHAR
// lane stores the `->>` text for any scalar but only a real JSON string round-trips; a
// typed lane stores the value only when its kind matches exactly.
bool WriteLane(const ShredField &field, const JsonoView &view, const Location &location, Vector &lane, idx_t row,
               std::string &scratch) {
	if (!location.found) {
		FlatVector::SetNull(lane, row, true);
		return false;
	}
	auto value_tag = SlotTag(view.SlotAt(location.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		FlatVector::SetNull(lane, row, true);
		return false;
	}
	size_t pos = location.pos;
	size_t string_cursor = location.string_cursor;
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	switch (field.primitive) {
	case ShredPrimitive::Varchar: {
		scratch.clear();
		if (RenderExtractText(scalar, scratch)) {
			FlatVector::SetNull(lane, row, true);
			return false;
		}
		FlatVector::Validity(lane).SetValid(row);
		FlatVector::GetData<string_t>(lane)[row] = StringVector::AddString(lane, scratch.data(), scratch.size());
		return scalar.kind == JsonoScalarKind::String;
	}
	case ShredPrimitive::Bigint:
		if (scalar.kind == JsonoScalarKind::Int64) {
			FlatVector::Validity(lane).SetValid(row);
			FlatVector::GetData<int64_t>(lane)[row] = scalar.int_value;
			return true;
		}
		break;
	case ShredPrimitive::Ubigint:
		if (scalar.kind == JsonoScalarKind::UInt64) {
			FlatVector::Validity(lane).SetValid(row);
			FlatVector::GetData<uint64_t>(lane)[row] = scalar.uint_value;
			return true;
		}
		if (scalar.kind == JsonoScalarKind::Int64 && scalar.int_value >= 0) {
			FlatVector::Validity(lane).SetValid(row);
			FlatVector::GetData<uint64_t>(lane)[row] = uint64_t(scalar.int_value);
			return true;
		}
		break;
	case ShredPrimitive::Double:
		if (scalar.kind == JsonoScalarKind::Double) {
			FlatVector::Validity(lane).SetValid(row);
			FlatVector::GetData<double>(lane)[row] = scalar.double_value;
			return true;
		}
		break;
	case ShredPrimitive::Boolean:
		if (scalar.kind == JsonoScalarKind::Bool) {
			FlatVector::Validity(lane).SetValid(row);
			FlatVector::GetData<bool>(lane)[row] = scalar.bool_value;
			return true;
		}
		break;
	}
	FlatVector::SetNull(lane, row, true);
	return false;
}

void ApplyShredFields(Vector &input_vec, idx_t count, const vector<ShredField> &fields, Vector &result,
                      ShredLocalState &lstate) {
	JsonoVectorData input;
	InitJsonoVectorData(input_vec, count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	auto &slots_vec = *children[0];
	auto &key_heap_vec = *children[1];
	auto &string_heap_vec = *children[2];
	auto &skips_vec = *children[3];
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			for (idx_t f = 0; f < fields.size(); f++) {
				FlatVector::SetNull(*children[4 + f], row, true);
			}
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			for (idx_t f = 0; f < fields.size(); f++) {
				FlatVector::SetNull(*children[4 + f], row, true);
			}
			continue;
		}

		FlatVector::Validity(result).SetValid(row);
		lstate.strip_paths.clear();
		for (idx_t f = 0; f < fields.size(); f++) {
			auto &field = fields[f];
			auto location = LocatePath(view, field.steps);
			bool strippable = WriteLane(field, view, location, *children[4 + f], row, lstate.text);
			if (strippable && IsObjectKeyPath(field.steps)) {
				lstate.strip_paths.push_back(&field.steps);
			}
		}

		lstate.builder.Reset();
		JsonoEmitObjectStrippingPaths(view, lstate.strip_paths, lstate.builder);
		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out, string_heap_out,
		                 skips_out, row, lstate.builder);
	}
}

void JsonoShredExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ShredBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ShredLocalState>();
	ApplyShredFields(args.data[0], args.size(), bind_data.fields, result, lstate);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

bool IsShredLaneType(const LogicalType &type) {
	ShredPrimitive primitive;
	return TypeToShredPrimitive(type, primitive);
}

void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &lanes,
                        Vector &result) {
	vector<ShredField> fields;
	fields.reserve(lanes.size());
	for (auto &lane : lanes) {
		ShredField field;
		field.path_name = lane.first;
		field.steps = LiteralKeyPath(lane.first);
		ShredPrimitive primitive;
		if (!TypeToShredPrimitive(lane.second, primitive)) {
			throw InternalException("jsono shred lane '%s' has a non-lane type", lane.first);
		}
		field.primitive = primitive;
		field.logical_type = lane.second;
		fields.push_back(std::move(field));
	}
	ShredLocalState lstate;
	ApplyShredFields(input, count, fields, result, lstate);
}

void RegisterJsonoShred(ExtensionLoader &loader) {
	ScalarFunction fun("jsono_shred", {JsonoType(), LogicalType::VARCHAR}, LogicalType::ANY, JsonoShredExecute,
	                   JsonoShredBind, nullptr, nullptr, ShredLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	loader.RegisterFunction(fun);
}

} // namespace duckdb
