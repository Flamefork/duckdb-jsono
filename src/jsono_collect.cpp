#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_memory.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/helper.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include "string_view.hpp"

#include <map>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// Carries the buffer manager into the collect aggregates' Update/Combine/Destroy so their unbounded
// element storage is accounted (see JsonoMemoryReservation). Captured at bind time; on plan round-trips the
// bind re-runs (no serialize callback), re-capturing the buffer manager from the deserialize context.
struct JsonoCollectBindData : public FunctionData {
	explicit JsonoCollectBindData(BufferManager &buffer_manager) : buffer_manager(buffer_manager) {
	}
	BufferManager &buffer_manager;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoCollectBindData>(buffer_manager);
	}
	bool Equals(const FunctionData &) const override {
		// The buffer manager is environmental, not a semantic parameter: two collect aggregates over the
		// same database share it, so equality holds by construction.
		return true;
	}
};

BufferManager &CollectBufferManager(AggregateInputData &aggr_input_data) {
	return aggr_input_data.bind_data->Cast<JsonoCollectBindData>().buffer_manager;
}

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

// An element is written once by Update and read-only afterwards, so states reference it instead of owning a
// copy of its document. A PRESERVE_INPUT combine (window segment tree / distinct aggregator) duplicates the
// source's whole element list into every frame's accumulator, and that combine graph is quadratic in the
// partition: sharing keeps the quadratic term in reference slots rather than in document bytes -- hundreds of
// megabytes instead of gigabytes on a five-thousand-row window. The element carries the reservation of its own
// bytes and frees it when the last state referencing it drops it, so the accounted total tracks live bytes.
struct CollectArrayElement {
	CollectBlob blob;
	JsonoOwnedReservation mem;
};

using CollectArrayElementRef = shared_ptr<const CollectArrayElement>;

struct CollectArrayState {
	std::vector<CollectArrayElementRef> *elements;
	// Covers only the reference slots this state duplicated in a preserving combine: an element pays for its
	// document and for the one slot its creator stores.
	JsonoMemoryReservation mem;
};

struct CollectArrayFunction {
	static void Initialize(CollectArrayState &state) {
		state.elements = nullptr;
		state.mem.reserved = 0;
		state.mem.buffer_manager = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		state.mem.Release();
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
	return make_uniq<JsonoCollectBindData>(BufferManager::GetBufferManager(context));
}

void AppendArrayElement(CollectArrayState &state, BufferManager &buffer_manager, JsonoRowReader &reader, idx_t row,
                        JsonoBuilder &scratch) {
	if (!state.elements) {
		state.elements = new std::vector<CollectArrayElementRef>();
	}
	auto element = make_shared_ptr<CollectArrayElement>();
	SerializeRowValue(reader, row, scratch, element->blob);
	// Reserve before storing so the reservation (and any OOM throw) precedes the growth. The element pays for
	// its own footprint -- the six owned byte streams, the element object and the first reference slot (the
	// one its creator stores) -- and releases all of it when the last state referencing it drops it. Further
	// slots are paid by the state that duplicates them in a preserving combine.
	element->mem.Reserve(buffer_manager,
	                     BlobBytes(element->blob) + sizeof(CollectArrayElement) + sizeof(CollectArrayElementRef));
	state.elements->push_back(std::move(element));
}

void JsonoGroupArraySimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, data_ptr_t state_ptr,
                                 idx_t count) {
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	auto &state = *reinterpret_cast<CollectArrayState *>(state_ptr);
	auto &buffer_manager = CollectBufferManager(aggr_input_data);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		AppendArrayElement(state, buffer_manager, reader, row, scratch);
	}
}

void JsonoGroupArrayUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &states, idx_t count) {
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<CollectArrayState *>(state_fmt);
	auto &buffer_manager = CollectBufferManager(aggr_input_data);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		AppendArrayElement(*state_data[RowIndex(state_fmt, row)], buffer_manager, reader, row, scratch);
	}
}

void JsonoGroupArrayCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<CollectArrayState *>(source_fmt);
	auto target_data = FlatVector::GetData<CollectArrayState *>(target);
	auto destructive = aggr_input_data.combine_type == AggregateCombineType::ALLOW_DESTRUCTIVE;
	auto &buffer_manager = CollectBufferManager(aggr_input_data);
	for (idx_t row = 0; row < count; row++) {
		auto &src = *source_data[RowIndex(source_fmt, row)];
		if (!src.elements || src.elements->empty()) {
			continue;
		}
		auto &tgt = *target_data[row];
		if (!tgt.elements) {
			tgt.elements = new std::vector<CollectArrayElementRef>();
		}
		// Fold order: the source partial's elements follow the target's, matching how the ordered
		// aggregate replays sorted partials into one accumulator. A PRESERVE_INPUT combine (window
		// segment tree / distinct aggregator) reuses the source across frames, so only a destructive
		// combine may move its references out; the moved-from container is then cleared to stay valid.
		if (destructive) {
			for (auto &elem : *src.elements) {
				tgt.elements->push_back(std::move(elem));
			}
			src.elements->clear();
			// The reference slots moved from source to target; transfer the reservation without a global change.
			tgt.mem.AbsorbFrom(src.mem);
		} else {
			// A preserving combine shares the source's elements: only the reference slots are duplicated,
			// so only they are reserved. The documents stay accounted once, by the elements themselves.
			tgt.mem.Reserve(buffer_manager, src.elements->size() * sizeof(CollectArrayElementRef));
			tgt.elements->insert(tgt.elements->end(), src.elements->begin(), src.elements->end());
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
			JsonoView view = ViewOfBlob(elem->blob);
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

// Shared and self-accounting for the same reason as CollectArrayElement.
struct CollectObjectEntry {
	std::string key;
	CollectBlob value;
	JsonoOwnedReservation mem;
};

using CollectObjectEntryRef = shared_ptr<const CollectObjectEntry>;

struct CollectObjectState {
	std::vector<CollectObjectEntryRef> *entries;
	// Covers only the reference slots this state duplicated in a preserving combine: an entry pays for its key,
	// its document and the one slot its creator stores.
	JsonoMemoryReservation mem;
};

struct CollectObjectFunction {
	static void Initialize(CollectObjectState &state) {
		state.entries = nullptr;
		state.mem.reserved = 0;
		state.mem.buffer_manager = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		state.mem.Release();
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
	return make_uniq<JsonoCollectBindData>(BufferManager::GetBufferManager(context));
}

void AppendObjectEntry(CollectObjectState &state, BufferManager &buffer_manager, const string_t &key,
                       JsonoRowReader &reader, idx_t row, JsonoBuilder &scratch) {
	if (!state.entries) {
		state.entries = new std::vector<CollectObjectEntryRef>();
	}
	auto entry = make_shared_ptr<CollectObjectEntry>();
	entry->key.assign(key.GetData(), key.GetSize());
	SerializeRowValue(reader, row, scratch, entry->value);
	// Reserve before storing: the entry pays for its key bytes, the value's six byte streams, the entry object
	// and the first reference slot (the one its creator stores). Further slots are paid by the state that
	// duplicates them in a preserving combine.
	entry->mem.Reserve(buffer_manager, entry->key.size() + BlobBytes(entry->value) + sizeof(CollectObjectEntry) +
	                                       sizeof(CollectObjectEntryRef));
	state.entries->push_back(std::move(entry));
}

void JsonoGroupObjectSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, data_ptr_t state_ptr,
                                  idx_t count) {
	UnifiedVectorFormat key_fmt;
	inputs[0].ToUnifiedFormat(count, key_fmt);
	auto key_data = UnifiedVectorFormat::GetData<string_t>(key_fmt);
	JsonoRowReader reader;
	reader.Init(inputs[1], count);
	auto &state = *reinterpret_cast<CollectObjectState *>(state_ptr);
	auto &buffer_manager = CollectBufferManager(aggr_input_data);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		auto kidx = RowIndex(key_fmt, row);
		if (!key_fmt.validity.RowIsValid(kidx)) {
			throw InvalidInputException("jsono_group_object: key cannot be NULL");
		}
		AppendObjectEntry(state, buffer_manager, key_data[kidx], reader, row, scratch);
	}
}

void JsonoGroupObjectUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat key_fmt;
	inputs[0].ToUnifiedFormat(count, key_fmt);
	auto key_data = UnifiedVectorFormat::GetData<string_t>(key_fmt);
	JsonoRowReader reader;
	reader.Init(inputs[1], count);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<CollectObjectState *>(state_fmt);
	auto &buffer_manager = CollectBufferManager(aggr_input_data);
	JsonoBuilder scratch;
	for (idx_t row = 0; row < count; row++) {
		auto kidx = RowIndex(key_fmt, row);
		if (!key_fmt.validity.RowIsValid(kidx)) {
			throw InvalidInputException("jsono_group_object: key cannot be NULL");
		}
		AppendObjectEntry(*state_data[RowIndex(state_fmt, row)], buffer_manager, key_data[kidx], reader, row, scratch);
	}
}

void JsonoGroupObjectCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<CollectObjectState *>(source_fmt);
	auto target_data = FlatVector::GetData<CollectObjectState *>(target);
	auto destructive = aggr_input_data.combine_type == AggregateCombineType::ALLOW_DESTRUCTIVE;
	auto &buffer_manager = CollectBufferManager(aggr_input_data);
	for (idx_t row = 0; row < count; row++) {
		auto &src = *source_data[RowIndex(source_fmt, row)];
		if (!src.entries || src.entries->empty()) {
			continue;
		}
		auto &tgt = *target_data[row];
		if (!tgt.entries) {
			tgt.entries = new std::vector<CollectObjectEntryRef>();
		}
		// A PRESERVE_INPUT combine (window segment tree / distinct aggregator) reuses the source across
		// frames, so only a destructive combine may move its references out; the moved-from container is
		// then cleared to stay valid.
		if (destructive) {
			for (auto &entry : *src.entries) {
				tgt.entries->push_back(std::move(entry));
			}
			src.entries->clear();
			// The reference slots moved from source to target; transfer the reservation without a global change.
			tgt.mem.AbsorbFrom(src.mem);
		} else {
			// A preserving combine shares the source's entries: only the reference slots are duplicated,
			// so only they are reserved. The keys and documents stay accounted once, by the entries themselves.
			tgt.mem.Reserve(buffer_manager, src.entries->size() * sizeof(CollectObjectEntryRef));
			tgt.entries->insert(tgt.entries->end(), src.entries->begin(), src.entries->end());
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
			last_by_key[(*state.entries)[e]->key] = e;
		}
		builder.Reset();
		builder.EmitObjectStart(last_by_key.size());
		for (auto &kv : last_by_key) {
			builder.EmitKeySlot(nonstd::string_view(kv.first.data(), kv.first.size()));
		}
		for (auto &kv : last_by_key) {
			builder.EmitObjectChildStart();
			JsonoView view = ViewOfBlob((*state.entries)[kv.second]->value);
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
