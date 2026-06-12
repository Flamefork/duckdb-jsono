#include "jsono.hpp"
#include "jsono_dom.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

namespace {

using namespace jsono;
using namespace jsono_dom;

// Per-thread state for jsono. Lifting the parser allocator, direct-writer
// state, and shape cache out of the per-chunk stack means heap capacities
// stabilise after one chunk and the shape cache survives across chunks for
// the duration of the query — the whole point of the cache.
struct JsonoParseLocalState : public FunctionLocalState {
	YyjsonAllocator parser;
	DomDirectState dom;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoParseLocalState>();
	}

	static unique_ptr<FunctionLocalState> InitCast(CastLocalStateParameters &parameters) {
		(void)parameters;
		return make_uniq<JsonoParseLocalState>();
	}
};

bool HandleJsonoInputError(const string &message, JsonoBodyWriter &writer, idx_t row, bool null_on_error,
                           CastParameters *parameters) {
	if (!null_on_error) {
		if (parameters) {
			HandleCastError::AssignError(message, *parameters);
		} else {
			throw InvalidInputException(message);
		}
	}
	writer.SetRowNull(row);
	return false;
}

bool ParseJsonoVector(Vector &input, idx_t count, Vector &result, JsonoParseLocalState &lstate, bool null_on_error,
                      CastParameters *parameters = nullptr) {
	UnifiedVectorFormat input_fmt;
	input.ToUnifiedFormat(count, input_fmt);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_fmt);

	JsonoBodyWriter writer;
	writer.Init(result);

	auto &dom = lstate.dom;
	auto &parser = lstate.parser;
	bool all_converted = true;
	// try_jsono sets null_on_error; TRY_CAST signals tolerance via a non-null error_message slot
	// instead. Writer-limit throws from the emit path (nesting depth, key/string size) must honour
	// both, while the strict forms keep throwing.
	bool tolerate_limit_errors = null_on_error || (parameters && parameters->error_message);

	for (idx_t i = 0; i < count; i++) {
		auto idx = input_fmt.sel->get_index(i);
		if (!input_fmt.validity.RowIsValid(idx)) {
			writer.SetRowNull(i);
			continue;
		}
		auto input = inputs[idx];
		if (input.GetSize() == 0) {
			all_converted = false;
			HandleJsonoInputError("jsono: input is empty", writer, i, null_on_error, parameters);
			continue;
		}

		yyjson_read_err err;
		auto doc = ReadJsonoDoc(input, parser.alc, err);
		if (!doc) {
			all_converted = false;
			auto message = StringUtil::Format("jsono: invalid JSON - %s", err.msg);
			HandleJsonoInputError(message, writer, i, null_on_error, parameters);
			continue;
		}

		try {
			EmitDomRowDirect(yyjson_doc_get_root(doc), dom, writer, i);
		} catch (const InvalidInputException &ex) {
			yyjson_doc_free(doc);
			if (!tolerate_limit_errors) {
				throw;
			}
			all_converted = false;
			HandleJsonoInputError(ex.what(), writer, i, null_on_error, parameters);
			continue;
		}
		yyjson_doc_free(doc);
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return all_converted;
}

void JsonoParseExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoParseLocalState>();
	ParseJsonoVector(args.data[0], args.size(), result, lstate, false);
}

void JsonoTryParseExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoParseLocalState>();
	ParseJsonoVector(args.data[0], args.size(), result, lstate, true);
}

bool JsonoCastToJsono(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	if (!parameters.local_state) {
		throw InternalException("JSONO cast missing local state");
	}
	auto &lstate = parameters.local_state->Cast<JsonoParseLocalState>();
	return ParseJsonoVector(source, count, result, lstate, false, &parameters);
}

} // namespace

void JsonoParseTextVector(Vector &source, idx_t count, Vector &result) {
	JsonoParseLocalState lstate;
	ParseJsonoVector(source, count, result, lstate, false);
}

void RegisterJsonoParse(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{
		ScalarFunctionSet set("jsono");
		ScalarFunction text_parse({LogicalType::VARCHAR}, jsono_type, JsonoParseExecute, nullptr, nullptr, nullptr,
		                          JsonoParseLocalState::Init);
		text_parse.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		set.AddFunction(text_parse);
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("try_jsono");
		ScalarFunction text_parse({LogicalType::VARCHAR}, jsono_type, JsonoTryParseExecute, nullptr, nullptr, nullptr,
		                          JsonoParseLocalState::Init);
		set.AddFunction(text_parse);
		loader.RegisterFunction(set);
	}
	{
		BoundCastInfo to_jsono(JsonoCastToJsono, nullptr, JsonoParseLocalState::InitCast);
		loader.RegisterCastFunction(LogicalType::VARCHAR, jsono_type, to_jsono.Copy(), -1);
		loader.RegisterCastFunction(LogicalType::JSON(), jsono_type, std::move(to_jsono), -1);
	}
}

} // namespace duckdb
