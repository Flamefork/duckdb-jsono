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

// Per-thread state for jsono. Lifting the parser allocator, builder, and
// shape cache out of the per-chunk stack means heap capacities stabilise
// after one chunk and the LRU shape cache survives across chunks for the
// duration of the query — the whole point of the cache.
struct JsonoParseLocalState : public FunctionLocalState {
	YyjsonAllocator parser;
	DomJsonoBuilder builder;

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

bool HandleJsonoInputError(const string &message, Vector &result, Vector &key_heap_vec, Vector &string_heap_vec,
                           Vector &slots_vec, Vector &skips_vec, idx_t row, bool null_on_error,
                           CastParameters *parameters) {
	if (!null_on_error) {
		if (parameters) {
			HandleCastError::AssignError(message, *parameters);
		} else {
			throw InvalidInputException(message);
		}
	}
	SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
	return false;
}

bool ParseJsonoVector(Vector &input, idx_t count, Vector &result, JsonoParseLocalState &lstate, bool null_on_error,
                      CastParameters *parameters = nullptr) {
	UnifiedVectorFormat input_fmt;
	input.ToUnifiedFormat(count, input_fmt);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_fmt);

	Vector *slots_p, *key_heap_p, *string_heap_p, *skips_p;
	InitJsonoBodyWrite(result, slots_p, key_heap_p, string_heap_p, skips_p);
	auto &slots_vec = *slots_p;
	auto &key_heap_vec = *key_heap_p;
	auto &string_heap_vec = *string_heap_p;
	auto &skips_vec = *skips_p;
	auto slots_data = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_data = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_data = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_data = FlatVector::GetData<string_t>(skips_vec);

	// One pass to sum input bytes for the reserve heuristic. Cheap relative to
	// the actual parse/emit work and pays for itself in avoided reallocs on
	// the first chunk; subsequent chunks no-op out of the reserve calls.
	size_t total_input_bytes = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = input_fmt.sel->get_index(i);
		if (input_fmt.validity.RowIsValid(idx)) {
			total_input_bytes += inputs[idx].GetSize();
		}
	}
	lstate.builder.ReserveForChunk(total_input_bytes);

	auto &builder = lstate.builder;
	auto &parser = lstate.parser;
	bool all_converted = true;

	for (idx_t i = 0; i < count; i++) {
		auto idx = input_fmt.sel->get_index(i);
		if (!input_fmt.validity.RowIsValid(idx)) {
			SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, i);
			continue;
		}
		auto input = inputs[idx];
		if (input.GetSize() == 0) {
			all_converted = false;
			HandleJsonoInputError("jsono: input is empty", result, key_heap_vec, string_heap_vec, slots_vec, skips_vec,
			                      i, null_on_error, parameters);
			continue;
		}

		yyjson_read_err err;
		auto doc = ReadJsonoDoc(input, parser.alc, err);
		if (!doc) {
			all_converted = false;
			auto message = StringUtil::Format("jsono: invalid JSON - %s", err.msg);
			HandleJsonoInputError(message, result, key_heap_vec, string_heap_vec, slots_vec, skips_vec, i,
			                      null_on_error, parameters);
			continue;
		}

		builder.Reset();
		bool emitted = EmitDomElement(yyjson_doc_get_root(doc), builder);
		yyjson_doc_free(doc);
		if (!emitted) {
			all_converted = false;
			HandleJsonoInputError("jsono: unsupported JSON value", result, key_heap_vec, string_heap_vec, slots_vec,
			                      skips_vec, i, null_on_error, parameters);
			continue;
		}

		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_data, key_heap_data,
		                 string_heap_data, skips_data, i, builder);
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
