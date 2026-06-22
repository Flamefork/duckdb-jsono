#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_path_function.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

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

void AppendListString(Vector &result, idx_t &length, const char *data, size_t size) {
	auto child_row = ListVector::GetListSize(result) + length;
	EnsureListCapacity(result, child_row + 1);
	auto &child = ListVector::GetEntry(result);
	FlatVector::GetData<string_t>(child)[child_row] = StringVector::AddString(child, data, size);
	length++;
}

unique_ptr<FunctionData> JsonoPathBind(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                       const char *function_name) {
	return BindJsonoSinglePath(context, *arguments[1], function_name, JsonoPathDialect::StrictJsonPath);
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
	return &func_expr.bind_info->Cast<JsonoSinglePathBindData>().path.steps;
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

struct JsonoTypePolicy {
	Vector &result;
	string_t *result_data;

	void Null(idx_t row) {
		FlatVector::SetNull(result, row, true);
	}

	void Found(JsonoRowReader &reader, const JsonoView &view, const vector<PathStep> *steps, const JsonoCursor &cursor,
	           idx_t row) {
		(void)reader;
		(void)steps;
		auto type_name = JsonoTypeName(view, cursor);
		if (!type_name) {
			FlatVector::SetNull(result, row, true);
		} else {
			result_data[row] = StringVector::AddString(result, type_name);
		}
	}
};

struct JsonoKeysPolicy {
	Vector &result;
	const vector<PathStep> &root_steps;

	void Null(idx_t row) {
		SetListRowNull(result, row);
	}

	void Found(JsonoRowReader &reader, const JsonoView &view, const vector<PathStep> *steps, const JsonoCursor &cursor,
	           idx_t row) {
		if (SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			SetListRowNull(result, row);
			return;
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
};

// jsono_type applies no found-value check: the type tag is invariant under stripping (shred paths
// are object-key chains, so containers stay containers and a found scalar was never stripped), and
// the optimizer's whole-document rewrite legitimately reads a manifested residual's root. A path
// miss covered by the manifest still fails loud (the silent NULL would be the data loss).
void JsonoTypeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoSinglePathLocalState>();
	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	JsonoTypePolicy policy {result, result_data};
	JsonoPointReadRows(reader, count, steps, lstate.locate_state, policy);
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
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoSinglePathLocalState>();
	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);

	const vector<PathStep> root_steps;
	JsonoKeysPolicy policy {result, root_steps};
	JsonoPointReadRows(reader, count, steps, lstate.locate_state, policy);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

// jsono_type/jsono_keys carry a per-expression RankCache in their local state, so the constructed
// ScalarFunction must register JsonoSinglePathLocalState::Init in its init_local_state slot.
ScalarFunction MakePathOpFunction(vector<LogicalType> arguments, LogicalType return_type, scalar_function_t function,
                                  bind_scalar_function_t bind) {
	return ScalarFunction(std::move(arguments), std::move(return_type), std::move(function), bind, nullptr, nullptr,
	                      JsonoSinglePathLocalState::Init);
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
