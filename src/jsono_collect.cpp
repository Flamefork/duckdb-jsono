#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"

#include "string_view.hpp"

#include <map>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// A collected element of a jsono_group_array / jsono_group_object group is a standalone JSONO document
// stored in an OwnedJsonoBlob (six owned byte streams: header + metadata framing), so Finalize can view
// it and copy it verbatim into the group's array/object.
using CollectBlob = OwnedJsonoBlob;

// Serialize the value the reader is positioned on into a standalone element blob. A present value is
// copied verbatim; a SQL-NULL (or empty) row becomes a JSON null element — the jsono_group_array /
// _object contract (matching core json_group_array / json_group_object, which keep nulls as null).
void SerializeRowValue(JsonoRowReader &reader, idx_t row, JsonoBuilder &scratch, CollectBlob &out) {
	JsonoBlobRow blob;
	JsonoView view;
	auto state = reader.Read(row, blob, view);
	scratch.Reset();
	if (state == JsonoRowState::Value) {
		JsonoCursor cursor;
		EmitValueVerbatim(view, cursor, scratch, 0);
	} else {
		scratch.EmitNull();
	}
	SerializeBuilderToBlob(scratch, out);
}

// ===== jsono_group_array(value [ORDER BY ...]) -> JSONO array of the group's values, fold order =====

struct CollectArrayState {
	std::vector<CollectBlob> *elements;
};

struct CollectArrayFunction {
	static void Initialize(CollectArrayState &state) {
		state.elements = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.elements;
		state.elements = nullptr;
	}
};

unique_ptr<FunctionData> JsonoGroupArrayBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw BinderException("jsono_group_array() requires a single JSONO argument");
	}
	// reconstruct=true: a shredded input is cast to plain JSONO by the binder, so Update reads whole
	// plain documents. The return type is always plain, so there is no sticky-shredded re-bind hazard.
	function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], "jsono_group_array", true);
	return nullptr;
}

void AppendArrayElement(CollectArrayState &state, JsonoRowReader &reader, idx_t row, JsonoBuilder &scratch) {
	if (!state.elements) {
		state.elements = new std::vector<CollectBlob>();
	}
	state.elements->emplace_back();
	SerializeRowValue(reader, row, scratch, state.elements->back());
}

void JsonoGroupArraySimpleUpdate(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_ptr, idx_t count) {
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	auto &state = *reinterpret_cast<CollectArrayState *>(state_ptr);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		AppendArrayElement(state, reader, row, scratch);
	}
}

void JsonoGroupArrayUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<CollectArrayState *>(state_fmt);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		AppendArrayElement(*state_data[RowIndex(state_fmt, row)], reader, row, scratch);
	}
}

void JsonoGroupArrayCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<CollectArrayState *>(source_fmt);
	auto target_data = FlatVector::GetData<CollectArrayState *>(target);
	auto destructive = aggr_input_data.combine_type == AggregateCombineType::ALLOW_DESTRUCTIVE;
	for (idx_t row = 0; row < count; row++) {
		auto &src = *source_data[RowIndex(source_fmt, row)];
		if (!src.elements || src.elements->empty()) {
			continue;
		}
		auto &tgt = *target_data[row];
		if (!tgt.elements) {
			tgt.elements = new std::vector<CollectBlob>();
		}
		// Fold order: the source partial's elements follow the target's, matching how the ordered
		// aggregate replays sorted partials into one accumulator. A PRESERVE_INPUT combine (window
		// segment tree / distinct aggregator) reuses the source across frames, so only a destructive
		// combine may move its blobs out; the moved-from container is then cleared to stay valid.
		if (destructive) {
			for (auto &elem : *src.elements) {
				tgt.elements->push_back(std::move(elem));
			}
			src.elements->clear();
		} else {
			for (const auto &elem : *src.elements) {
				tgt.elements->push_back(elem);
			}
		}
	}
}

void JsonoGroupArrayFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<CollectArrayState *>(state_fmt);
	JsonoBodyWriter writer;
	writer.Init(result);
	JsonoBuilder builder;
	for (idx_t i = 0; i < count; i++) {
		auto rid = i + offset;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.elements) {
			// Empty group -> SQL NULL, matching core json_group_array over an empty group.
			writer.SetRowNull(rid);
			continue;
		}
		builder.Reset();
		builder.EmitArrayStart();
		for (auto &elem : *state.elements) {
			JsonoView view = ViewOfBlob(elem);
			if (!view.ParseHeader()) {
				throw InternalException("jsono_group_array: malformed collected element blob");
			}
			JsonoCursor cursor;
			EmitValueVerbatim(view, cursor, builder, 1);
		}
		builder.EmitArrayEnd();
		writer.WriteRow(rid, builder);
	}
}

// ===== jsono_group_object(key, value) -> JSONO object mapping key -> value =====

struct CollectObjectEntry {
	std::string key;
	CollectBlob value;
};

struct CollectObjectState {
	std::vector<CollectObjectEntry> *entries;
};

struct CollectObjectFunction {
	static void Initialize(CollectObjectState &state) {
		state.entries = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.entries;
		state.entries = nullptr;
	}
};

unique_ptr<FunctionData> JsonoGroupObjectBind(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 2) {
		throw BinderException("jsono_group_object() requires a (key, value) argument pair");
	}
	function.arguments[0] = LogicalType::VARCHAR;
	function.arguments[1] = JsonoResolveJsonoArgument(context, *arguments[1], "jsono_group_object", true);
	return nullptr;
}

void AppendObjectEntry(CollectObjectState &state, const string_t &key, JsonoRowReader &reader, idx_t row,
                       JsonoBuilder &scratch) {
	if (!state.entries) {
		state.entries = new std::vector<CollectObjectEntry>();
	}
	state.entries->emplace_back();
	auto &entry = state.entries->back();
	entry.key.assign(key.GetData(), key.GetSize());
	SerializeRowValue(reader, row, scratch, entry.value);
}

void JsonoGroupObjectSimpleUpdate(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_ptr, idx_t count) {
	UnifiedVectorFormat key_fmt;
	inputs[0].ToUnifiedFormat(count, key_fmt);
	auto key_data = UnifiedVectorFormat::GetData<string_t>(key_fmt);
	JsonoRowReader reader;
	reader.Init(inputs[1], count);
	auto &state = *reinterpret_cast<CollectObjectState *>(state_ptr);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		auto kidx = RowIndex(key_fmt, row);
		if (!key_fmt.validity.RowIsValid(kidx)) {
			throw InvalidInputException("jsono_group_object: key cannot be NULL");
		}
		AppendObjectEntry(state, key_data[kidx], reader, row, scratch);
	}
}

void JsonoGroupObjectUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat key_fmt;
	inputs[0].ToUnifiedFormat(count, key_fmt);
	auto key_data = UnifiedVectorFormat::GetData<string_t>(key_fmt);
	JsonoRowReader reader;
	reader.Init(inputs[1], count);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<CollectObjectState *>(state_fmt);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		auto kidx = RowIndex(key_fmt, row);
		if (!key_fmt.validity.RowIsValid(kidx)) {
			throw InvalidInputException("jsono_group_object: key cannot be NULL");
		}
		AppendObjectEntry(*state_data[RowIndex(state_fmt, row)], key_data[kidx], reader, row, scratch);
	}
}

void JsonoGroupObjectCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<CollectObjectState *>(source_fmt);
	auto target_data = FlatVector::GetData<CollectObjectState *>(target);
	auto destructive = aggr_input_data.combine_type == AggregateCombineType::ALLOW_DESTRUCTIVE;
	for (idx_t row = 0; row < count; row++) {
		auto &src = *source_data[RowIndex(source_fmt, row)];
		if (!src.entries || src.entries->empty()) {
			continue;
		}
		auto &tgt = *target_data[row];
		if (!tgt.entries) {
			tgt.entries = new std::vector<CollectObjectEntry>();
		}
		// A PRESERVE_INPUT combine (window segment tree / distinct aggregator) reuses the source across
		// frames, so only a destructive combine may move its entries out; the moved-from container is
		// then cleared to stay valid.
		if (destructive) {
			for (auto &entry : *src.entries) {
				tgt.entries->push_back(std::move(entry));
			}
			src.entries->clear();
		} else {
			for (const auto &entry : *src.entries) {
				tgt.entries->push_back(entry);
			}
		}
	}
}

void JsonoGroupObjectFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<CollectObjectState *>(state_fmt);
	JsonoBodyWriter writer;
	writer.Init(result);
	JsonoBuilder builder;
	for (idx_t i = 0; i < count; i++) {
		auto rid = i + offset;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.entries) {
			writer.SetRowNull(rid);
			continue;
		}
		// The format stores unique sorted keys. std::map sorts keys byte-lexicographically (the sort
		// the readers' binary search assumes) and, by overwriting on each fold occurrence, keeps the
		// last value per key — last-wins in fold order, then sorted.
		std::map<std::string, idx_t> last_by_key;
		for (idx_t e = 0; e < state.entries->size(); e++) {
			last_by_key[(*state.entries)[e].key] = e;
		}
		builder.Reset();
		builder.EmitObjectStart(last_by_key.size());
		for (auto &kv : last_by_key) {
			builder.EmitKeySlot(nonstd::string_view(kv.first.data(), kv.first.size()));
		}
		for (auto &kv : last_by_key) {
			builder.EmitObjectChildStart();
			JsonoView view = ViewOfBlob((*state.entries)[kv.second].value);
			if (!view.ParseHeader()) {
				throw InternalException("jsono_group_object: malformed collected value blob");
			}
			JsonoCursor cursor;
			EmitValueVerbatim(view, cursor, builder, 1);
		}
		builder.EmitObjectEnd();
		writer.WriteRow(rid, builder);
	}
}

} // namespace

void RegisterJsonoCollect(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{
		AggregateFunction fun("jsono_group_array", {LogicalType::ANY}, jsono_type,
		                      AggregateFunction::StateSize<CollectArrayState>,
		                      AggregateFunction::StateInitialize<CollectArrayState, CollectArrayFunction>,
		                      JsonoGroupArrayUpdate, JsonoGroupArrayCombine, JsonoGroupArrayFinalize,
		                      FunctionNullHandling::SPECIAL_HANDLING, JsonoGroupArraySimpleUpdate, JsonoGroupArrayBind,
		                      AggregateFunction::StateDestroy<CollectArrayState, CollectArrayFunction>);
		fun.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		loader.RegisterFunction(std::move(fun));
	}
	{
		AggregateFunction fun("jsono_group_object", {LogicalType::ANY, LogicalType::ANY}, jsono_type,
		                      AggregateFunction::StateSize<CollectObjectState>,
		                      AggregateFunction::StateInitialize<CollectObjectState, CollectObjectFunction>,
		                      JsonoGroupObjectUpdate, JsonoGroupObjectCombine, JsonoGroupObjectFinalize,
		                      FunctionNullHandling::SPECIAL_HANDLING, JsonoGroupObjectSimpleUpdate,
		                      JsonoGroupObjectBind,
		                      AggregateFunction::StateDestroy<CollectObjectState, CollectObjectFunction>);
		fun.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		loader.RegisterFunction(std::move(fun));
	}
}

} // namespace duckdb
