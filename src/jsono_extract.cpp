#include "jsono_extract.hpp"
#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace duckdb {
namespace {

using namespace jsono;

struct ExtractPathBindData : public FunctionData {
	ExtractPathBindData(string path_p, vector<PathStep> steps_p) : path(std::move(path_p)), steps(std::move(steps_p)) {
	}

	string path;
	vector<PathStep> steps;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ExtractPathBindData>(path, steps);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ExtractPathBindData>();
		if (path != other.path || steps.size() != other.steps.size()) {
			return false;
		}
		for (idx_t i = 0; i < steps.size(); i++) {
			if (steps[i].kind != other.steps[i].kind || steps[i].index != other.steps[i].index ||
			    steps[i].key != other.steps[i].key) {
				return false;
			}
		}
		return true;
	}
};

struct ExtractRankCacheEntry {
	bool valid = false;
	idx_t step_index = DConstants::INVALID_INDEX;
	size_t key_count = 0;
	string key;
	size_t rank = 0;
	bool found = false;
};

constexpr idx_t RANK_CACHE_SIZE = 4096;

struct ExtractLocalState : public FunctionLocalState {
	ExtractLocalState() : rank_cache(RANK_CACHE_SIZE) {
	}

	vector<ExtractRankCacheEntry> rank_cache;
	JsonoBuilder builder;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<ExtractLocalState>();
	}
};

struct Location {
	size_t pos = 0;
	size_t string_cursor = 0;
};

vector<PathStep> LiteralKeyPath(nonstd::string_view name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, string(name), 0});
	return path;
}

vector<PathStep> ArrayIndexPath(idx_t index) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Index, string(), index});
	return path;
}

unique_ptr<FunctionData> JsonoExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto function_name = bound_function.name;
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("%s: path must be constant", function_name);
	}
	auto path_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (path_value.IsNull()) {
		throw BinderException("%s: path must not be NULL", function_name);
	}
	if (arguments[1]->return_type.IsIntegral()) {
		if (!path_value.DefaultTryCastAs(LogicalType::BIGINT)) {
			throw BinderException("%s: path must be BIGINT", function_name);
		}
		auto index = path_value.GetValue<int64_t>();
		if (index < 0) {
			throw BinderException("%s: array index must be non-negative", function_name);
		}
		return make_uniq<ExtractPathBindData>(std::to_string(index), ArrayIndexPath(idx_t(index)));
	}
	if (!path_value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		throw BinderException("%s: path must be VARCHAR", function_name);
	}
	auto path = StringValue::Get(path_value);
	if (path.empty()) {
		ThrowInvalidPath(function_name.c_str(), path);
	}
	vector<PathStep> steps;
	if (path[0] == '$') {
		steps = ParseJsonoPath(path, function_name.c_str());
	} else {
		steps = LiteralKeyPath(path);
	}
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("%s: wildcard paths are not supported", function_name);
		}
	}
	return make_uniq<ExtractPathBindData>(std::move(path), std::move(steps));
}

nonstd::string_view ObjectKeyAtRank(const JsonoView &view, const ObjectLayout &layout, size_t rank) {
	if (rank >= layout.key_count) {
		throw InternalException("JSONO object key rank out of bounds");
	}
	auto key_slot = view.SlotAt(layout.key_start + rank);
	if (SlotTag(key_slot) != tag::KEY) {
		throw InvalidInputException("malformed JSONO: expected KEY slot");
	}
	return view.KeyAt(SlotPayload(key_slot));
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
	if (SlotTag(key_slot) != tag::KEY) {
		throw InvalidInputException("malformed JSONO: expected KEY slot");
	}
	if (view.KeyAt(SlotPayload(key_slot)) != key) {
		return false;
	}
	return true;
}

bool ValidateCachedObjectRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t rank,
                              bool found) {
	if (rank > layout.key_count) {
		return false;
	}
	if (found) {
		return rank < layout.key_count && ObjectKeyAtRank(view, layout, rank) == key;
	}
	if (rank > 0 && ObjectKeyAtRank(view, layout, rank - 1) >= key) {
		return false;
	}
	if (rank < layout.key_count && ObjectKeyAtRank(view, layout, rank) <= key) {
		return false;
	}
	return true;
}

ExtractRankCacheEntry &GetRankCacheEntry(vector<ExtractRankCacheEntry> &rank_cache, idx_t step_index,
                                         size_t key_count) {
	auto cache_index = idx_t(((uint64_t(step_index) * 0x9E3779B97F4A7C15ULL) ^ key_count) & (RANK_CACHE_SIZE - 1));
	return rank_cache[cache_index];
}

bool FindPathObjectRank(ExtractLocalState &lstate, idx_t step_index, const JsonoView &view, const ObjectLayout &layout,
                        const string &key, size_t &rank) {
	auto &entry = GetRankCacheEntry(lstate.rank_cache, step_index, layout.key_count);
	auto cache_valid = entry.valid && entry.step_index == step_index && entry.key_count == layout.key_count &&
	                   entry.key == key && ValidateCachedObjectRank(view, layout, key, entry.rank, entry.found);
	if (!cache_valid) {
		bool found = FindObjectKeyRank(view, layout, key, rank);
		entry.valid = true;
		entry.step_index = step_index;
		entry.key_count = layout.key_count;
		entry.key = key;
		entry.rank = rank;
		entry.found = found;
		return found;
	}
	rank = entry.rank;
	return entry.found;
}

bool LocatePath(ExtractLocalState &lstate, const vector<PathStep> &steps, const JsonoView &view, Location &location) {
	size_t pos = 0;
	size_t string_cursor = 0;
	for (idx_t step_index = 0; step_index < steps.size(); step_index++) {
		if (pos >= view.Slots()) {
			return false;
		}
		auto slot_tag = SlotTag(view.SlotAt(pos));
		auto &step = steps[step_index];
		switch (step.kind) {
		case PathStepKind::Key: {
			if (slot_tag != tag::OBJ_START) {
				return false;
			}
			auto layout = ReadObjectLayout(view, pos);
			size_t rank = 0;
			if (!FindPathObjectRank(lstate, step_index, view, layout, step.key, rank)) {
				return false;
			}
			size_t value_rank = 0;
			size_t value_pos = layout.value_start;
			size_t value_string_cursor = string_cursor;
			MoveObjectValueCursorToRank(view, layout, string_cursor, rank, value_rank, value_pos, value_string_cursor);
			pos = value_pos;
			string_cursor = value_string_cursor;
			break;
		}
		case PathStepKind::Index: {
			if (slot_tag != tag::ARR_START) {
				return false;
			}
			auto end_pos = ReadArrayEndPos(view, pos);
			size_t elem_pos = pos + 1;
			idx_t current = 0;
			while (elem_pos < end_pos && current < step.index) {
				SkipValueFast(view, elem_pos, string_cursor);
				current++;
			}
			if (elem_pos >= end_pos || current < step.index) {
				return false;
			}
			pos = elem_pos;
			break;
		}
		case PathStepKind::Wildcard:
			return false;
		}
	}
	location.pos = pos;
	location.string_cursor = string_cursor;
	return true;
}

void AppendJsonString(nonstd::string_view value, std::string &out) {
	out.push_back('"');
	for (char c : value) {
		switch (c) {
		case '"':
			out.append("\\\"");
			break;
		case '\\':
			out.append("\\\\");
			break;
		case '\b':
			out.append("\\b");
			break;
		case '\f':
			out.append("\\f");
			break;
		case '\n':
			out.append("\\n");
			break;
		case '\r':
			out.append("\\r");
			break;
		case '\t':
			out.append("\\t");
			break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
				out.append(buf);
			} else {
				out.push_back(c);
			}
		}
	}
	out.push_back('"');
}

bool AppendExtractStringScalar(const JsonoScalar &scalar, std::string &out) {
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

void AppendJsonValueText(const JsonoView &view, size_t &pos, size_t &string_cursor, std::string &out, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, pos);
		auto val_cursor = layout.value_start;
		out.push_back('{');
		for (size_t i = 0; i < layout.key_count; i++) {
			if (i) {
				out.push_back(',');
			}
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: expected KEY slot");
			}
			AppendJsonString(view.KeyAt(SlotPayload(key_slot)), out);
			out.push_back(':');
			AppendJsonValueText(view, val_cursor, string_cursor, out, depth + 1);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		out.push_back('}');
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, pos);
		pos++;
		out.push_back('[');
		bool first = true;
		while (pos < end_pos) {
			if (!first) {
				out.push_back(',');
			}
			first = false;
			AppendJsonValueText(view, pos, string_cursor, out, depth + 1);
		}
		pos = end_pos + 1;
		out.push_back(']');
		return;
	}
	default: {
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		if (scalar.kind == JsonoScalarKind::String) {
			AppendJsonString(scalar.text, out);
			return;
		}
		if (AppendExtractStringScalar(scalar, out)) {
			out.append("null");
		}
		return;
	}
	}
}

void EmitJsonoSubtree(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, pos);
		auto val_cursor = layout.value_start;
		builder.EmitObjectStart(layout.key_count);
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: expected KEY slot");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
		for (size_t i = 0; i < layout.key_count; i++) {
			builder.EmitObjectChildStart();
			EmitJsonoSubtree(view, val_cursor, string_cursor, builder, depth + 1);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		builder.EmitObjectEnd();
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, pos);
		builder.EmitArrayStart();
		pos++;
		while (pos < end_pos) {
			EmitJsonoSubtree(view, pos, string_cursor, builder, depth + 1);
		}
		pos = end_pos + 1;
		builder.EmitArrayEnd();
		return;
	}
	default: {
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		switch (scalar.kind) {
		case JsonoScalarKind::String:
			builder.EmitString(scalar.text);
			return;
		case JsonoScalarKind::Int64:
			builder.EmitInt(scalar.int_value);
			return;
		case JsonoScalarKind::UInt64:
			builder.EmitUInt(scalar.uint_value);
			return;
		case JsonoScalarKind::Double:
			builder.EmitDouble(scalar.double_value);
			return;
		case JsonoScalarKind::Dec60:
			builder.EmitDec60(scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			return;
		case JsonoScalarKind::NumberText:
			builder.EmitNumberText(scalar.text);
			return;
		case JsonoScalarKind::Bool:
			builder.EmitBool(scalar.bool_value);
			return;
		case JsonoScalarKind::Null:
			builder.EmitNull();
			return;
		}
		return;
	}
	}
}

void JsonoExtractExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ExtractPathBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ExtractLocalState>();
	auto count = args.size();

	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
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
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			continue;
		}
		Location location;
		if (!LocatePath(lstate, bind_data.steps, view, location)) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			continue;
		}
		size_t pos = location.pos;
		size_t string_cursor = location.string_cursor;
		lstate.builder.Reset();
		EmitJsonoSubtree(view, pos, string_cursor, lstate.builder, 0);
		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out, string_heap_out,
		                 skips_out, row, lstate.builder);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoExtractStringExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ExtractPathBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ExtractLocalState>();
	auto count = args.size();

	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	std::string scratch;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		Location location;
		if (!LocatePath(lstate, bind_data.steps, view, location)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		size_t pos = location.pos;
		size_t string_cursor = location.string_cursor;
		auto slot_tag = SlotTag(view.SlotAt(pos));
		scratch.clear();
		if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
			AppendJsonValueText(view, pos, string_cursor, scratch, 0);
			result_data[row] = StringVector::AddString(result, scratch.data(), scratch.size());
			continue;
		}
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
			result_data[row] = StringVector::AddString(result, scalar.text.data(), scalar.text.size());
		} else if (AppendExtractStringScalar(scalar, scratch)) {
			FlatVector::SetNull(result, row, true);
		} else {
			result_data[row] = StringVector::AddString(result, scratch.data(), scratch.size());
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

ScalarFunction MakeExtractFunction(string name, LogicalType path_type) {
	ScalarFunction fun(std::move(name), {JsonoType(), std::move(path_type)}, JsonoType(), JsonoExtractExecute,
	                   JsonoExtractBind, nullptr, nullptr, ExtractLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

ScalarFunction MakeExtractStringFunction(string name, LogicalType path_type) {
	ScalarFunction fun(std::move(name), {JsonoType(), std::move(path_type)}, LogicalType::VARCHAR,
	                   JsonoExtractStringExecute, JsonoExtractBind, nullptr, nullptr, ExtractLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

} // namespace

void RegisterJsonoExtract(ExtensionLoader &loader) {
	ScalarFunctionSet extract("jsono_extract");
	extract.AddFunction(MakeExtractFunction("jsono_extract", LogicalType::VARCHAR));
	extract.AddFunction(MakeExtractFunction("jsono_extract", LogicalType::BIGINT));
	loader.RegisterFunction(std::move(extract));

	ScalarFunctionSet core_extract("json_extract");
	core_extract.AddFunction(MakeExtractFunction("json_extract", LogicalType::VARCHAR));
	core_extract.AddFunction(MakeExtractFunction("json_extract", LogicalType::BIGINT));
	loader.RegisterFunction(std::move(core_extract));

	ScalarFunctionSet extract_string("jsono_extract_string");
	extract_string.AddFunction(MakeExtractStringFunction("jsono_extract_string", LogicalType::VARCHAR));
	extract_string.AddFunction(MakeExtractStringFunction("jsono_extract_string", LogicalType::BIGINT));
	loader.RegisterFunction(std::move(extract_string));

	ScalarFunctionSet arrow_string("->>");
	arrow_string.AddFunction(MakeExtractStringFunction("->>", LogicalType::VARCHAR));
	arrow_string.AddFunction(MakeExtractStringFunction("->>", LogicalType::BIGINT));
	loader.RegisterFunction(std::move(arrow_string));
}

} // namespace duckdb
