#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"

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

// Per-expression rank cache so repeated reads of the same path over many same-shape rows trust a
// cached key rank by a shape_hash/int compare instead of re-running the binary search each row.
struct PathOpsLocalState : public FunctionLocalState {
	RankCache rank_cache;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<PathOpsLocalState>();
	}
};

struct JsonoPathBindData : public FunctionData {
	explicit JsonoPathBindData(vector<PathStep> steps_p) : steps(std::move(steps_p)) {
	}
	vector<PathStep> steps;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoPathBindData>(steps);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoPathBindData>();
		return PathStepsEqual(steps, other.steps);
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

// Bind for the 1-argument jsono_type/jsono_keys overloads (ANY arg0): accept a shredded value by
// redeclaring arg0 to plain JSONO (the binder inserts the lossless reconstruct cast) and reject a
// non-JSONO argument loudly. Carries no path data.
unique_ptr<FunctionData> JsonoArgOnlyBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, true);
	return nullptr;
}

unique_ptr<FunctionData> JsonoPathOnlyBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, true);
	return JsonoPathBind(context, arguments, bound_function.name.c_str());
}

const vector<PathStep> *PathStepsFromState(ExpressionState &state, idx_t column_count) {
	if (column_count < 2) {
		return nullptr;
	}
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	return &func_expr.bind_info->Cast<JsonoPathBindData>().steps;
}

const char *JsonoTypeName(const JsonoView &view, JsonoCursor cursor) {
	if (cursor.pos >= view.Slots()) {
		return nullptr;
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		return "OBJECT";
	}
	if (slot_tag == tag::ARR_START) {
		return "ARRAY";
	}
	auto scalar = DecodeScalarAt(view, cursor);
	return JsonoScalarTypeName(scalar.kind);
}

// jsono_type applies no found-value check: the type tag is invariant under stripping (shred paths
// are object-key chains, so containers stay containers and a found scalar was never stripped), and
// the optimizer's whole-document rewrite legitimately reads a manifested residual's root. A path
// miss covered by the manifest still fails loud (the silent NULL would be the data loss).
void JsonoTypeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<PathOpsLocalState>();
	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		JsonoCursor cursor;
		if (steps && !LocatePathSteps(&lstate.rank_cache, 0, *steps, view, cursor)) {
			reader.CheckPathMiss(view, *steps);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto type_name = JsonoTypeName(view, cursor);
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

// jsono_keys lists a whole found object, so a manifest leaf strictly inside it (or a covered
// path miss) fails loud — the key list would silently omit the stripped key. The optimizer's
// introspect rewrite only feeds the residual a path that holds no shred beneath it, so a hit
// here is always a narrowed row.
void JsonoKeysExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<PathOpsLocalState>();
	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);

	const vector<PathStep> root_steps;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			SetListRowNull(result, row);
			continue;
		}
		JsonoCursor cursor;
		if (steps && !LocatePathSteps(&lstate.rank_cache, 0, *steps, view, cursor)) {
			reader.CheckPathMiss(view, *steps);
			SetListRowNull(result, row);
			continue;
		}
		if (SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			SetListRowNull(result, row);
			continue;
		}
		reader.CheckContainerRead(view, steps ? *steps : root_steps);
		auto layout = ReadObjectLayout(view, cursor.pos);
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

// jsono_type/jsono_keys carry a per-expression RankCache in their local state, so the constructed
// ScalarFunction must register PathOpsLocalState::Init in its init_local_state slot.
ScalarFunction MakePathOpFunction(vector<LogicalType> arguments, LogicalType return_type, scalar_function_t function,
                                  bind_scalar_function_t bind) {
	return ScalarFunction(std::move(arguments), std::move(return_type), std::move(function), bind, nullptr, nullptr,
	                      PathOpsLocalState::Init);
}

void RegisterJsonoPathOps(ExtensionLoader &loader) {
	{
		ScalarFunctionSet set("jsono_type");
		set.AddFunction(
		    MakePathOpFunction({LogicalType::ANY}, LogicalType::VARCHAR, JsonoTypeExecute, JsonoArgOnlyBind));
		set.AddFunction(MakePathOpFunction({LogicalType::ANY, LogicalType::VARCHAR}, LogicalType::VARCHAR,
		                                   JsonoTypeExecute, JsonoPathOnlyBind));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_keys");
		set.AddFunction(MakePathOpFunction({LogicalType::ANY}, LogicalType::LIST(LogicalType::VARCHAR),
		                                   JsonoKeysExecute, JsonoArgOnlyBind));
		set.AddFunction(MakePathOpFunction({LogicalType::ANY, LogicalType::VARCHAR},
		                                   LogicalType::LIST(LogicalType::VARCHAR), JsonoKeysExecute,
		                                   JsonoPathOnlyBind));
		loader.RegisterFunction(set);
	}
}

} // namespace duckdb
