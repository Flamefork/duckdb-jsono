#include "jsono.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"

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

#include <algorithm>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

void EnsureListCapacity(Vector &result, idx_t needed) {
	if (needed <= ListVector::GetListCapacity(result)) {
		return;
	}
	auto next = std::max<idx_t>(needed, std::max<idx_t>(ListVector::GetListCapacity(result) * 2, 1));
	ListVector::Reserve(result, next);
}

void AppendListString(Vector &result, idx_t &length, const char *data, size_t size) {
	auto child_row = ListVector::GetListSize(result) + length;
	EnsureListCapacity(result, child_row + 1);
	auto &child = ListVector::GetEntry(result);
	FlatVector::GetData<string_t>(child)[child_row] = StringVector::AddString(child, data, size);
	length++;
}

void FinishListRow(Vector &result, idx_t row, idx_t start_offset, idx_t length) {
	ListVector::GetData(result)[row] = {start_offset, length};
	ListVector::SetListSize(result, start_offset + length);
}

void SetListRowNull(Vector &result, idx_t row) {
	FlatVector::SetNull(result, row, true);
	ListVector::GetData(result)[row] = {0, 0};
}

// Walk from the document root to the slot addressed by `steps`, tracking the
// string heap cursor so nested string payloads resolve correctly. Returns false
// when a step misses (absent key, out-of-range index, or stepping into a
// non-container). Wildcard steps are rejected at bind, so they never reach here.
bool NavigateToPath(const JsonoView &view, const vector<PathStep> &steps, size_t &pos, size_t &string_cursor) {
	pos = 0;
	string_cursor = 0;
	for (auto &step : steps) {
		if (pos >= view.Slots()) {
			return false;
		}
		auto slot_tag = SlotTag(view.SlotAt(pos));
		if (step.kind == PathStepKind::Key) {
			if (slot_tag != tag::OBJ_START) {
				return false;
			}
			auto layout = ReadObjectLayout(view, pos);
			nonstd::string_view target(step.key);
			size_t lo = 0;
			size_t hi = layout.key_count;
			while (lo < hi) {
				auto mid = lo + (hi - lo) / 2;
				auto key_slot = view.SlotAt(layout.key_start + mid);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				if (view.KeyAt(SlotPayload(key_slot)) < target) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			if (lo >= layout.key_count || view.KeyAt(SlotPayload(view.SlotAt(layout.key_start + lo))) != target) {
				return false;
			}
			size_t value_pos = layout.key_start + layout.key_count;
			size_t value_string_cursor = string_cursor;
			for (size_t i = 0; i < lo; i++) {
				SkipValueFast(view, value_pos, value_string_cursor);
			}
			pos = value_pos;
			string_cursor = value_string_cursor;
		} else if (step.kind == PathStepKind::Index) {
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
		} else {
			return false;
		}
	}
	return true;
}

struct JsonoPathBindData : public FunctionData {
	explicit JsonoPathBindData(vector<PathStep> steps_p) : steps(std::move(steps_p)) {
	}
	vector<PathStep> steps;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoPathBindData>(steps);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoPathBindData>();
		if (steps.size() != other.steps.size()) {
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

unique_ptr<FunctionData> JsonoPathBind(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                       const char *function_name) {
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
	auto steps = ParseJsonoPath(StringValue::Get(path_value), function_name);
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("%s: wildcard paths are not supported", function_name);
		}
	}
	return make_uniq<JsonoPathBindData>(std::move(steps));
}

unique_ptr<FunctionData> JsonoTypePathBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	(void)bound_function;
	return JsonoPathBind(context, arguments, "jsono_type");
}

unique_ptr<FunctionData> JsonoKeysPathBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	(void)bound_function;
	return JsonoPathBind(context, arguments, "jsono_keys");
}

const vector<PathStep> *PathStepsFromState(ExpressionState &state, idx_t column_count) {
	if (column_count < 2) {
		return nullptr;
	}
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	return &func_expr.bind_info->Cast<JsonoPathBindData>().steps;
}

const char *JsonoTypeName(const JsonoView &view, size_t pos, size_t string_cursor) {
	if (pos >= view.Slots()) {
		return nullptr;
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		return "OBJECT";
	}
	if (slot_tag == tag::ARR_START) {
		return "ARRAY";
	}
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	return JsonoScalarTypeName(scalar.kind);
}

void JsonoTypeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		if (steps && !NavigateToPath(view, *steps, pos, string_cursor)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto type_name = JsonoTypeName(view, pos, string_cursor);
		if (!type_name) {
			FlatVector::SetNull(result, row, true);
		} else {
			result_data[row] = StringVector::AddString(result, type_name);
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoKeysExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetListRowNull(result, row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetListRowNull(result, row);
			continue;
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		if (steps && !NavigateToPath(view, *steps, pos, string_cursor)) {
			SetListRowNull(result, row);
			continue;
		}
		if (SlotTag(view.SlotAt(pos)) != tag::OBJ_START) {
			SetListRowNull(result, row);
			continue;
		}
		auto layout = ReadObjectLayout(view, pos);
		auto start = ListVector::GetListSize(result);
		idx_t length = 0;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			auto key = view.KeyAt(SlotPayload(key_slot));
			AppendListString(result, length, key.data(), key.size());
		}
		FinishListRow(result, row, start, length);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoPathOps(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{
		ScalarFunctionSet set("jsono_type");
		set.AddFunction(ScalarFunction({jsono_type}, LogicalType::VARCHAR, JsonoTypeExecute));
		set.AddFunction(ScalarFunction({jsono_type, LogicalType::VARCHAR}, LogicalType::VARCHAR, JsonoTypeExecute,
		                               JsonoTypePathBind));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_keys");
		set.AddFunction(ScalarFunction({jsono_type}, LogicalType::LIST(LogicalType::VARCHAR), JsonoKeysExecute));
		set.AddFunction(ScalarFunction({jsono_type, LogicalType::VARCHAR}, LogicalType::LIST(LogicalType::VARCHAR),
		                               JsonoKeysExecute, JsonoKeysPathBind));
		loader.RegisterFunction(set);
	}
}

} // namespace duckdb
