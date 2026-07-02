#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_path_function.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

namespace {

using namespace jsono;

// Bind for the 1-argument overload: accept a shredded value by redeclaring arg0 to plain JSONO (the
// binder inserts the lossless reconstruct cast), so the executor reads whole logical values via the
// validated read-path and never touches shred lanes. Reject a non-JSONO argument loudly.
unique_ptr<FunctionData> JsonoArrayElementsArgOnlyBind(ClientContext &context, ScalarFunction &bound_function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, true);
	return nullptr;
}

// Bind for the 2-argument overloads (VARCHAR JSONPath / literal key, or BIGINT index): resolve arg0
// to plain JSONO and parse the constant path with the Extract dialect (same path grammar as
// jsono_extract, so a bare key is a literal top-level key and a non-negative integer is an index).
unique_ptr<FunctionData> JsonoArrayElementsPathBind(ClientContext &context, ScalarFunction &bound_function,
                                                    vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, true);
	return BindJsonoSinglePath(context, *arguments[1], bound_function.name.c_str(), JsonoPathDialect::Extract);
}

const vector<PathStep> *PathStepsFromState(ExpressionState &state, idx_t column_count) {
	if (column_count < 2) {
		return nullptr;
	}
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	return &func_expr.bind_info->Cast<JsonoSinglePathBindData>().path.steps;
}

// Output sink for jsono_array_elements: appends one plain JSONO document per array element into the
// LIST child (a JsonoType() struct vector). `next_child` is the running write cursor; the six-blob
// writer's cached data pointers are refreshed only when a capacity grow moves the child buffers.
struct ElementsSink {
	Vector &list;
	JsonoBodyWriter writer;
	idx_t capacity;
	idx_t next_child;

	explicit ElementsSink(Vector &list_p) : list(list_p), next_child(ListVector::GetListSize(list_p)) {
		capacity = ListVector::GetListCapacity(list_p);
		writer.Init(ListVector::GetEntry(list_p));
	}

	void Emit(const JsonoBuilder &builder) {
		if (next_child >= capacity) {
			EnsureListCapacity(list, next_child + 1);
			capacity = ListVector::GetListCapacity(list);
			writer.Init(ListVector::GetEntry(list));
		}
		writer.WriteRow(next_child, builder);
		next_child++;
	}
};

void JsonoArrayElementsExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoSinglePathLocalState>();
	// Whole-document reader: after the bind reconstruct cast the input is plain JSONO, so a manifest
	// entry would be a row narrowed by a raw struct cast and fails loud on read.
	JsonoRowReader reader;
	reader.Init(args.data[0], count);
	auto steps = PathStepsFromState(state, args.ColumnCount());

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	ElementsSink sink(result);
	JsonoBuilder builder;

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		lstate.locate_state.NextRow();
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			SetListRowNull(result, row);
			continue;
		}
		JsonoCursor cursor;
		if (steps && !LocatePathSteps(lstate.locate_state, 0, *steps, view, cursor)) {
			SetListRowNull(result, row); // missing path -> SQL NULL
			continue;
		}
		if (SlotTag(view.SlotAt(cursor.pos)) != tag::ARR_START) {
			SetListRowNull(result, row); // non-array value (object / scalar / JSON null) -> SQL NULL
			continue;
		}
		auto start = sink.next_child;
		auto end_pos = ReadArrayEndPos(view, cursor.pos);
		cursor.pos++;
		while (cursor.pos < end_pos) {
			// Each element is re-emitted verbatim into a fresh builder, so its container ids and heap
			// offsets normalize from zero — a self-contained plain JSONO document.
			builder.Reset();
			EmitValueVerbatim(view, cursor, builder, 0);
			sink.Emit(builder);
		}
		FinishListRow(result, row, start, sink.next_child - start);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoArrayElements(ExtensionLoader &loader) {
	auto element_list = LogicalType::LIST(JsonoType());
	// arg0 is ANY so a shredded jsono struct (whose type varies by shred set) also binds; the bind
	// validates it is JSONO or shredded and reconstructs to plain. The optional second argument is the
	// path: VARCHAR (JSONPath / literal key) or BIGINT (array index), like jsono_extract.
	ScalarFunctionSet set("jsono_array_elements");
	set.AddFunction(ScalarFunction({LogicalType::ANY}, element_list, JsonoArrayElementsExecute,
	                               JsonoArrayElementsArgOnlyBind, nullptr, nullptr, JsonoSinglePathLocalState::Init));
	set.AddFunction(ScalarFunction({LogicalType::ANY, LogicalType::VARCHAR}, element_list, JsonoArrayElementsExecute,
	                               JsonoArrayElementsPathBind, nullptr, nullptr, JsonoSinglePathLocalState::Init));
	set.AddFunction(ScalarFunction({LogicalType::ANY, LogicalType::BIGINT}, element_list, JsonoArrayElementsExecute,
	                               JsonoArrayElementsPathBind, nullptr, nullptr, JsonoSinglePathLocalState::Init));
	loader.RegisterFunction(set);
}

} // namespace duckdb
