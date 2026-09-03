#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_extension.hpp"
#include "jsono_memory.hpp"
#include "jsono_merge_core.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_reconstruct.hpp"
#include "jsono_row_read.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/string_vector.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include "string_view.hpp"

#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// Per-shred fold metadata for the direct shredded accumulator, derived from the bind shred set
// once: the shred kind, the scalar lane's primitive, and the compiled object-key path.
struct GroupMergeShredPlan {
	ShredKind kind = ShredKind::Scalar;
	jsono::JsonoScalarPrimitive prim = jsono::JsonoScalarPrimitive::Varchar;
	vector<PathStep> steps;
};

struct GroupMergeBindData : public FunctionData {
	MergeMode merge_mode;
	// Empty for a plain input (Finalize writes the plain accumulator verbatim). For a shredded
	// input the captured shreds (clean names, canonical order) are the sticky shredded return
	// type: they drive the direct fold plan and Finalize's native lane/residual write (plus its
	// manifest entries), so the shreds "stick" across the aggregation with no reshred.
	vector<std::pair<string, LogicalType>> shreds;
	// Derived (not part of Equals): fold plan per shred, the array-shred index filter for the
	// per-chunk array composition pass, and the write-side tables Finalize emits each group's
	// manifest and spill bits from.
	vector<GroupMergeShredPlan> shred_plan;
	vector<idx_t> array_shred_filter;
	JsonoShredWriteModel write_model;
	// Carried into Update/Combine so the accumulator's residual/lane growth is accounted; re-captured on
	// plan round-trips because this bind_data has no serialize callback (deserialize re-runs the bind).
	BufferManager &buffer_manager;

	GroupMergeBindData(MergeMode merge_mode, BufferManager &buffer_manager)
	    : merge_mode(merge_mode), buffer_manager(buffer_manager) {
	}

	void BuildShredPlan() {
		shred_plan.clear();
		array_shred_filter.clear();
		shred_plan.resize(shreds.size());
		for (idx_t f = 0; f < shreds.size(); f++) {
			auto &plan = shred_plan[f];
			plan.kind = ClassifyShredKind(shreds[f].second);
			plan.steps = ShredNamePath(shreds[f].first, "jsono_group_merge shred");
			if (plan.kind == ShredKind::Scalar) {
				plan.prim = jsono::JsonoScalarPrimitiveFromType(shreds[f].second, "jsono_group_merge shred");
			} else {
				array_shred_filter.push_back(f);
			}
		}
		write_model = JsonoBuildShredWriteModel(shreds);
	}

	unique_ptr<FunctionData> Copy() const override {
		// Copy every member, by copy-construction rather than by listing them and rebuilding the derived
		// tables: a hand-written list makes a new member's absence silent, and rebuilding hides it twice
		// over by producing a plausible copy that is missing whatever the list forgot.
		return make_uniq<GroupMergeBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<GroupMergeBindData>();
		return merge_mode == other.merge_mode && shreds == other.shreds;
	}
};

// One scalar shred lane of the direct accumulator: the latest lane value for the path, or
// inactive when the current value (if any) lives in the residual accumulator instead.
struct DirectLaneSlot {
	bool active = false;
	int64_t i = 0;
	uint64_t u = 0;
	double d = 0;
	bool b = false;
	std::string s;
};

// Direct accumulator for a shredded input: the group's value is the residual accumulator overlaid
// (residual-authoritative) with the active scalar lanes. Invariants: `residual` holds arrays fully
// composed to plain (array shreds are folded whole-array, never kept as skeleton+lane state), and
// lane ACTIVATION shadow-deletes the path from the residual so a stale diverted value cannot
// out-rank the newer lane at finalize; a later residual write to the same path simply lands in the
// residual and wins the overlay, which matches the reconstruct-then-fold order.
struct DirectShreddedAcc {
	enum class Kind : uint8_t { Empty, NonObject, Object };
	Kind kind = Kind::Empty;
	OwnedJsonoBlob residual;
	// Sized once to the bind shred count at allocation so the fold paths index by shred `f` directly;
	// array/nested-object shred slots simply stay inactive (their values live in the residual).
	vector<DirectLaneSlot> lanes;
};

struct GroupMergeState {
	OwnedJsonoBlob *acc;
	DirectShreddedAcc *direct;
	bool has_input;
	JsonoMemoryReservation mem;
};

// Current owned-heap footprint of a non-keyed group_merge accumulator (plain acc blob and/or the direct
// shredded accumulator's residual + scalar lanes). BlobBytes is O(1), so this is O(shred count) -- cheap
// enough to re-measure per fold on the hot path.
idx_t GroupMergeFootprint(const GroupMergeState &state) {
	idx_t bytes = 0;
	if (state.acc) {
		bytes += sizeof(OwnedJsonoBlob) + BlobBytes(*state.acc);
	}
	if (state.direct) {
		bytes += sizeof(DirectShreddedAcc) + BlobBytes(state.direct->residual);
		bytes += state.direct->lanes.capacity() * sizeof(DirectLaneSlot);
		for (auto &lane : state.direct->lanes) {
			bytes += lane.s.capacity();
		}
	}
	return bytes;
}

struct GroupMergeFunction {
	static void Initialize(GroupMergeState &state) {
		state.acc = nullptr;
		state.direct = nullptr;
		state.has_input = false;
		state.mem.reserved = 0;
		state.mem.buffer_manager = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		state.mem.Release();
		delete state.acc;
		state.acc = nullptr;
		delete state.direct;
		state.direct = nullptr;
		state.has_input = false;
	}
};

unique_ptr<FunctionData> JsonoGroupMergeBind(BindAggregateFunctionInput &input) {
	auto &context = input.GetClientContext();
	auto &function = input.GetBoundFunction();
	auto &arguments = input.GetArguments();
	if (arguments.size() != 1) {
		throw BinderException("jsono_group_merge() requires a single JSONO argument");
	}
	auto &arg = *arguments[0];
	if (arg.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto &type = arg.GetReturnType();
	JsonoRequireExtensionOptimizerForShredded(context, type, "jsono_group_merge");
	auto bind_data = make_uniq<GroupMergeBindData>(MergeMode::IgnoreNulls, BufferManager::GetBufferManager(context));
	if (IsShreddedJsonoType(type)) {
		// Capture the shreds (clean names, the type's canonical order) as the sticky shredded return
		// type — like jsono_merge_patch, the result stays shredded across the aggregation. They also
		// drive the direct fold plan; Finalize writes the native lanes/residual straight out (no
		// reshred). Update folds the shredded rows directly (residual merge + scalar lane state), so
		// no per-row reconstruct cast is involved.
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(type, layout);
		for (auto &shred : layout.shreds) {
			bind_data->shreds.emplace_back(shred.first, shred.second);
		}
		bind_data->BuildShredPlan();
		// Keep the shredded argument type (Update folds it directly; only array shreds are composed to
		// plain per chunk) rather than
		// forcing a plain argument with a binder cast. With a plain argument the bound aggregate's
		// argument type and its sticky shredded return type disagree, so a plan serialize→deserialize
		// re-bind (debug verification) would re-derive a plain return and fail to reconcile it back to
		// the shredded type. Arg type == return type makes the re-bind reproduce the same shredded type.
		function.GetArguments()[0] = type;
		function.SetReturnType(type);
	} else if (type.id() == LogicalTypeId::SQLNULL || IsJsonoType(type)) {
		function.GetArguments()[0] = JsonoType();
	} else {
		JsonoRejectForeignLayout(type, "jsono_group_merge()");
		throw BinderException("jsono_group_merge() input must be JSONO");
	}
	return std::move(bind_data);
}

// Fold one incoming value (a row blob, or a partial's accumulated blob) into the
// group state, matching jsono_group_merge IGNORE NULLS semantics: object+object →
// cursor merge (incoming wins per key, its nulls ignored, A kept on B-null, keys whose
// object value strips to empty omitted); a fresh object (no prior object accumulator)
// is seeded with nulls/empty-objects stripped; a non-object incoming replaces the
// accumulator wholesale (verbatim, including a top-level null) — the same rule
// jsono_merge_patch applies to a non-object patch. The merged builder is
// re-serialized so the next fold can view the accumulator.
void FoldIntoGroupState(GroupMergeState &state, const JsonoView &incoming) {
	// Deliberately never destroyed (leaked once per thread). A plain `static thread_local` scratch
	// registers a TLS destructor, and mingw's winpthreads runs those in a multi-pass loop at worker
	// thread exit where the frees corrupted the heap (STATUS_HEAP_CORRUPTION on windows_amd64_mingw,
	// backtrace: _pthread_cleanup_dest → ~JsonoBuilder → free). DuckDB gives aggregates no per-thread
	// state to move this into (scalar functions use FunctionLocalState instead; AggregateInputData is
	// only bind_data + arena), so aggregate scratch stays thread_local but binds a reference to a heap
	// object: a reference has a trivial destructor, nothing is registered, thread exit frees nothing.
	// Every never-destroyed scratch in this extension points at this comment.
	static thread_local JsonoBuilder &scratch = *(new JsonoBuilder());
	scratch.Reset();
	JsonoCursor cursor;
	bool incoming_is_object = SlotTag(incoming.SlotAt(0)) == tag::OBJ_START;
	bool merged_objects = false;
	if (incoming_is_object && state.acc) {
		JsonoView acc_view = ViewOfBlob(*state.acc);
		if (acc_view.ParseHeader() && acc_view.Slots() > 0 && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START) {
			MergeTwoObjects(acc_view, JsonoCursor(), incoming, JsonoCursor(), scratch, MergeMode::IgnoreNulls, 0);
			merged_objects = true;
		}
	}
	if (!merged_objects) {
		if (incoming_is_object) {
			EmitValueStrip(incoming, cursor, scratch, MergeMode::IgnoreNulls, 0);
		} else {
			EmitValueVerbatim(incoming, cursor, scratch, 0);
		}
	}
	if (!state.acc) {
		state.acc = new OwnedJsonoBlob();
	}
	SerializeBuilderToBlob(scratch, *state.acc);
	state.has_input = true;
}

// ---- Direct shredded fold ----
// Consumes shredded rows without reconstructing each one to plain (the accumulator contract is
// documented on DirectShreddedAcc above).

// Lazily allocate the group's direct accumulator with its lanes sized to the bind shred count, so
// every fold path indexes lanes by shred `f` directly (array/nested slots stay inactive). Sizing at
// allocation — not in Initialize, which runs before bind data exists — keeps "lanes are bind-sized"
// true by construction at all three fold entry points (SimpleUpdate, Update, Combine).
DirectShreddedAcc &EnsureDirectAcc(GroupMergeState &state, const GroupMergeBindData &bind_data) {
	if (!state.direct) {
		state.direct = new DirectShreddedAcc();
		state.direct->lanes.resize(bind_data.shreds.size());
	}
	return *state.direct;
}

// Chunk-level read handles for the direct fold. A type with array shreds first composes the
// array paths to plain for the whole chunk (skeleton + lockstep lane -> plain array, via the
// filtered reconstruct, which also verifies each row's manifest); scalar lanes are read from
// their typed vectors and folded as lane state.
void DirectFoldPrepare(Vector &input, idx_t count, const GroupMergeBindData &bind_data, Vector &arrays_plain,
                       JsonoRowReader &reader, vector<UnifiedVectorFormat> &lane_fmt) {
	D_ASSERT(bind_data.shred_plan.size() == bind_data.shreds.size());
	if (!bind_data.array_shred_filter.empty()) {
		JsonoOverlayShredsToPlain(input, count, bind_data.array_shred_filter, arrays_plain);
		reader.Init(arrays_plain, count);
	} else {
		reader.Init(input, count);
	}
	lane_fmt.resize(bind_data.shreds.size());
	for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
		if (bind_data.shred_plan[f].kind == ShredKind::Scalar) {
			jsono::JsonoShredVector(input, f).ToUnifiedFormat(count, lane_fmt[f]);
		}
	}
}

// Shadow-delete a newly activated lane's path from the residual accumulator: the exact path (a
// stale diverted value) or its first non-object prefix (a parent an earlier row replaced with a
// scalar). Without this the residual-authoritative finalize overlay would prefer the stale
// residual value over the newer lane. A later residual write to the path needs no bookkeeping:
// it lands in the residual and legitimately out-ranks the older lane at finalize.
void DirectShadowDeletePath(DirectShreddedAcc &acc, const vector<PathStep> &steps) {
	JsonoView acc_view = ViewOfBlob(acc.residual);
	if (!acc_view.ParseHeader() || acc_view.Slots() == 0) {
		return;
	}
	idx_t delete_depth;
	if (ClassifyResidualPath(acc_view, steps, delete_depth) == ResidualPathOutcome::Absent) {
		return; // path absent from the residual: nothing shadows the lane
	}
	// delete_depth is the whole path on Present and the prefix up to and including the non-object node
	// on NonObjectPrefix — deleting that prefix lets the finalize overlay rebuild the object chain.
	// Never destroyed on purpose — TLS destructors corrupt the mingw heap; see FoldIntoGroupState.
	static thread_local JsonoBuilder &patch_builder = *(new JsonoBuilder());
	static thread_local OwnedJsonoBlob &patch_blob = *(new OwnedJsonoBlob());
	static thread_local JsonoBuilder &merged = *(new JsonoBuilder());
	patch_builder.Reset();
	for (idx_t depth = 0; depth < delete_depth; depth++) {
		patch_builder.EmitObjectStart(1);
		patch_builder.EmitKeySlot(steps[depth].key);
		patch_builder.EmitObjectChildStart();
	}
	patch_builder.EmitNull();
	for (idx_t depth = 0; depth < delete_depth; depth++) {
		patch_builder.EmitObjectEnd();
	}
	SerializeBuilderToBlob(patch_builder, patch_blob);
	JsonoView patch_view = ViewOfBlob(patch_blob);
	patch_view.ParseHeader();
	merged.Reset();
	MergeTwoObjects(acc_view, JsonoCursor(), patch_view, JsonoCursor(), merged, MergeMode::Patch, 0);
	SerializeBuilderToBlob(merged, acc.residual);
}

// Fold one shredded row (residual view, arrays already plain, plus the scalar lane vectors) into
// the direct accumulator, preserving FoldIntoGroupState's IgnoreNulls semantics case by case.
void DirectFoldRow(DirectShreddedAcc &acc, const JsonoView &incoming, const GroupMergeBindData &bind_data,
                   const vector<UnifiedVectorFormat> &lane_fmt, idx_t row) {
	// Never destroyed on purpose — TLS destructors corrupt the mingw heap; see FoldIntoGroupState.
	static thread_local JsonoBuilder &scratch = *(new JsonoBuilder());
	bool incoming_is_object = SlotTag(incoming.SlotAt(0)) == tag::OBJ_START;
	if (!incoming_is_object) {
		// A non-object replaces the accumulator wholesale (jsono_merge_patch's non-object patch
		// rule). Every lane is NULL on such a row: shred paths are object-key chains.
		scratch.Reset();
		JsonoCursor cursor;
		EmitValueVerbatim(incoming, cursor, scratch, 0);
		SerializeBuilderToBlob(scratch, acc.residual);
		acc.kind = DirectShreddedAcc::Kind::NonObject;
		for (auto &lane : acc.lanes) {
			lane.active = false;
		}
		return;
	}
	if (acc.kind != DirectShreddedAcc::Kind::Object) {
		// Fresh object state (first row, or an object after a non-object): seed with the incoming
		// residual stripped (IgnoreNulls seed drops null members and empty objects), lanes reset
		// and adopted below.
		scratch.Reset();
		JsonoCursor cursor;
		EmitValueStrip(incoming, cursor, scratch, MergeMode::IgnoreNulls, 0);
		SerializeBuilderToBlob(scratch, acc.residual);
		acc.kind = DirectShreddedAcc::Kind::Object;
		for (auto &lane : acc.lanes) {
			lane.active = false;
		}
	} else if (ContainerChildCount(SlotPayload(incoming.SlotAt(0))) > 0) {
		// The incoming residual carries members: IgnoreNulls-merge it into the accumulator. A
		// fully-shredded row's empty residual skips the merge (and its re-serialize) entirely.
		JsonoView acc_view = ViewOfBlob(acc.residual);
		// acc.residual is a self-built fold output (kind == Object): SerializeBuilderToBlob always emits
		// a well-formed header, so the parse cannot fail. Assert the link; keep the call unconditional
		// so it still runs in release, where D_ASSERT vanishes.
		bool acc_parsed = acc_view.ParseHeader();
		D_ASSERT(acc_parsed);
		(void)acc_parsed;
		scratch.Reset();
		MergeTwoObjects(acc_view, JsonoCursor(), incoming, JsonoCursor(), scratch, MergeMode::IgnoreNulls, 0);
		SerializeBuilderToBlob(scratch, acc.residual);
	}
	// Adopt the row's present scalar lanes. Shadow-deletes only matter while the residual
	// accumulator has members at all — the fully-shredded hot case skips the locates. The view is
	// scoped out before the loop: DirectShadowDeletePath re-serializes the residual, dangling any
	// view of it.
	bool residual_has_members;
	{
		JsonoView acc_view = ViewOfBlob(acc.residual);
		residual_has_members =
		    acc_view.ParseHeader() && acc_view.Slots() > 0 && ContainerChildCount(SlotPayload(acc_view.SlotAt(0))) > 0;
	}
	for (idx_t f = 0; f < bind_data.shred_plan.size(); f++) {
		auto &plan = bind_data.shred_plan[f];
		if (plan.kind != ShredKind::Scalar) {
			continue;
		}
		auto &fmt = lane_fmt[f];
		auto idx = fmt.sel->get_index(row);
		if (!fmt.validity.RowIsValid(idx)) {
			continue;
		}
		auto &lane = acc.lanes[f];
		switch (plan.prim) {
		case jsono::JsonoScalarPrimitive::Varchar: {
			auto value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
			lane.s.assign(value.GetData(), value.GetSize());
			break;
		}
		case jsono::JsonoScalarPrimitive::Bigint:
			lane.i = UnifiedVectorFormat::GetData<int64_t>(fmt)[idx];
			break;
		case jsono::JsonoScalarPrimitive::Ubigint:
			lane.u = UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx];
			break;
		case jsono::JsonoScalarPrimitive::Double:
			lane.d = UnifiedVectorFormat::GetData<double>(fmt)[idx];
			break;
		case jsono::JsonoScalarPrimitive::Boolean:
			lane.b = UnifiedVectorFormat::GetData<bool>(fmt)[idx];
			break;
		}
		lane.active = true;
		if (residual_has_members) {
			DirectShadowDeletePath(acc, plan.steps);
		}
	}
}

// Combine: fold `source` into `target` as one later value (the current partial-state contract).
void DirectFoldState(DirectShreddedAcc &target, const DirectShreddedAcc &source, const GroupMergeBindData &bind_data) {
	if (source.kind == DirectShreddedAcc::Kind::Empty) {
		return;
	}
	if (source.kind == DirectShreddedAcc::Kind::NonObject || target.kind != DirectShreddedAcc::Kind::Object) {
		// A non-object source replaces the target wholesale; an object source over an empty or
		// non-object target starts the object state fresh — both are a state copy (the source
		// residual is already stripped/merged by its own folds).
		target.kind = source.kind;
		target.residual = source.residual;
		target.lanes = source.lanes;
		return;
	}
	JsonoView source_view = ViewOfBlob(source.residual);
	// source.residual is a self-built fold output; ResidualConflictsWithShredPath below reads its
	// slots and relies on this parse. Parse into a bool so the assert survives a release build.
	bool source_parsed = source_view.ParseHeader();
	D_ASSERT(source_parsed);
	if (source_parsed && source_view.Slots() > 0 && ContainerChildCount(SlotPayload(source_view.SlotAt(0))) > 0) {
		JsonoView target_view = ViewOfBlob(target.residual);
		// target.residual is a self-built fold output (kind == Object): the parse cannot fail. Assert the
		// link; keep the call unconditional so it still runs in release, where D_ASSERT vanishes.
		bool target_parsed = target_view.ParseHeader();
		D_ASSERT(target_parsed);
		(void)target_parsed;
		// Never destroyed on purpose — TLS destructors corrupt the mingw heap; see FoldIntoGroupState.
		static thread_local JsonoBuilder &scratch = *(new JsonoBuilder());
		scratch.Reset();
		MergeTwoObjects(target_view, JsonoCursor(), source_view, JsonoCursor(), scratch, MergeMode::IgnoreNulls, 0);
		SerializeBuilderToBlob(scratch, target.residual);
	}
	// Scoped out before the loop: DirectShadowDeletePath re-serializes target.residual, dangling any
	// view of it. source_view stays valid across the loop — shadow-deletes mutate only the target.
	bool residual_has_members;
	{
		JsonoView target_view = ViewOfBlob(target.residual);
		residual_has_members = target_view.ParseHeader() && target_view.Slots() > 0 &&
		                       ContainerChildCount(SlotPayload(target_view.SlotAt(0))) > 0;
	}
	D_ASSERT(source.lanes.size() == bind_data.shred_plan.size());
	for (idx_t f = 0; f < bind_data.shred_plan.size(); f++) {
		if (bind_data.shred_plan[f].kind != ShredKind::Scalar || !source.lanes[f].active) {
			continue;
		}
		// Adopt the source lane only when the SOURCE residual leaves the path free — i.e. source's own
		// finalize would honor the lane. When the source residual pins the path (a later divert) or
		// breaks its object skeleton (a parent replaced by a scalar), that residual value was just
		// merged into the target and legitimately out-ranks the now-stale source lane; adopting the
		// lane and shadow-deleting the merged value would silently resurrect the older lane. Skipping
		// needs no target-lane bookkeeping: the residual-authoritative finalize ignores lanes whose
		// path the residual carries, mirroring DirectFoldRow's per-row divert resolution.
		if (ResidualConflictsWithShredPath(source_view, bind_data.shred_plan[f].steps)) {
			continue;
		}
		target.lanes[f] = source.lanes[f];
		if (residual_has_members) {
			DirectShadowDeletePath(target, bind_data.shred_plan[f].steps);
		}
	}
}

void JsonoGroupMergeSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 data_ptr_t state_ptr, idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeBindData>();
	auto &bm = bind_data.buffer_manager;
	auto &state = *reinterpret_cast<GroupMergeState *>(state_ptr);
	JsonoView view;
	if (!bind_data.shreds.empty()) {
		// Shredded input: direct fold — residual merge plus scalar lane state, no per-row
		// reconstruct. The reader (or the filtered array composition) verifies each manifest.
		Vector arrays_plain(JsonoType(), count);
		JsonoRowReader reader;
		vector<UnifiedVectorFormat> lane_fmt;
		DirectFoldPrepare(inputs[0], count, bind_data, arrays_plain, reader, lane_fmt);
		auto &direct = EnsureDirectAcc(state, bind_data);
		for (idx_t row = 0; row < count; row++) {
			JsonoBlobRow blob;
			if (reader.Read(row, blob, view) != JsonoRowState::Value) {
				continue;
			}
			DirectFoldRow(direct, view, bind_data, lane_fmt, row);
			state.has_input = true;
			state.mem.Resize(bm, GroupMergeFootprint(state));
		}
		return;
	}
	// Plain input: any manifest entry is a narrowed row and the reader fails loud.
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		FoldIntoGroupState(state, view);
		state.mem.Resize(bm, GroupMergeFootprint(state));
	}
}

void JsonoGroupMergeUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                           idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeBindData>();
	auto &bm = bind_data.buffer_manager;
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	JsonoView view;
	if (!bind_data.shreds.empty()) {
		// Same direct fold as the simple update, per target group state.
		Vector arrays_plain(JsonoType(), count);
		JsonoRowReader reader;
		vector<UnifiedVectorFormat> lane_fmt;
		DirectFoldPrepare(inputs[0], count, bind_data, arrays_plain, reader, lane_fmt);
		for (idx_t row = 0; row < count; row++) {
			JsonoBlobRow blob;
			if (reader.Read(row, blob, view) != JsonoRowState::Value) {
				continue;
			}
			auto &state = *state_data[RowIndex(state_fmt, row)];
			DirectFoldRow(EnsureDirectAcc(state, bind_data), view, bind_data, lane_fmt, row);
			state.has_input = true;
			state.mem.Resize(bm, GroupMergeFootprint(state));
		}
		return;
	}
	// Plain input: any manifest entry is a narrowed row and the reader fails loud.
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto &state = *state_data[RowIndex(state_fmt, row)];
		FoldIntoGroupState(state, view);
		state.mem.Resize(bm, GroupMergeFootprint(state));
	}
}

void JsonoGroupMergeCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeBindData>();
	auto &bm = bind_data.buffer_manager;
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<GroupMergeState *>(source_fmt);
	auto target_data = FlatVector::GetData<GroupMergeState *>(target);

	// Only targets grow here (a source is folded in, never moved out), so accounting the target per fold
	// is sufficient.
	if (!bind_data.shreds.empty()) {
		for (idx_t row = 0; row < count; row++) {
			auto &source_state = *source_data[RowIndex(source_fmt, row)];
			if (!source_state.has_input || !source_state.direct) {
				continue;
			}
			auto &target_state = *target_data[row];
			DirectFoldState(EnsureDirectAcc(target_state, bind_data), *source_state.direct, bind_data);
			target_state.has_input = true;
			target_state.mem.Resize(bm, GroupMergeFootprint(target_state));
		}
		return;
	}
	for (idx_t row = 0; row < count; row++) {
		auto &source_state = *source_data[RowIndex(source_fmt, row)];
		if (!source_state.has_input || !source_state.acc) {
			continue;
		}
		JsonoView source_view = ViewOfBlob(*source_state.acc);
		if (!source_view.ParseHeader() || source_view.Slots() == 0) {
			continue;
		}
		auto &target_state = *target_data[row];
		FoldIntoGroupState(target_state, source_view);
		target_state.mem.Resize(bm, GroupMergeFootprint(target_state));
	}
}

// Copy a serialized accumulator blob's six streams into the body writer at row `rid`.
void WriteOwnedBlobRow(JsonoBodyWriter &writer, idx_t rid, const OwnedJsonoBlob &blob) {
	writer.data[BODY_SLOTS][rid] = WriteBlobInto(writer.Slots(), blob.slots.data(), blob.slots.size());
	writer.data[BODY_KEY_HEAP][rid] = WriteBlobInto(writer.KeyHeap(), blob.key_heap.data(), blob.key_heap.size());
	writer.data[BODY_STRING_HEAP][rid] =
	    WriteBlobInto(writer.StringHeap(), blob.string_heap.data(), blob.string_heap.size());
	writer.data[BODY_SKIPS][rid] = WriteBlobInto(writer.Skips(), blob.skips.data(), blob.skips.size());
	writer.data[BODY_LENGTHS][rid] = WriteBlobInto(writer.Lengths(), blob.lengths.data(), blob.lengths.size());
	writer.data[BODY_NUMS][rid] = WriteBlobInto(writer.Nums(), blob.nums.data(), blob.nums.size());
}

// Write each group's accumulated plain blob (or an empty object for an empty/NULL group) into a
// plain JSONO struct vector `out` at row base+i. Shared by the plain finalize (writes straight
// into the result at the chunk offset) and the shredded finalize (writes a temp at base 0).
void FinalizePlainGroups(Vector &out, UnifiedVectorFormat &state_fmt, GroupMergeState *const *state_data, idx_t count,
                         idx_t base) {
	// Writes at rows [base, base+count). A group with zero non-NULL inputs (has_input false)
	// finalizes to SQL NULL, honoring the aggregate's DEFAULT_NULL_HANDLING; a group that had
	// input but whose merge collapsed to an empty object writes {}. `out` is always plain here
	// (this runs only when shreds are empty), so SetRowNull alone nulls the whole row.
	JsonoBodyWriter writer;
	writer.Init(out);
	JsonoBuilder empty_builder;

	for (idx_t i = 0; i < count; i++) {
		auto rid = i + base;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.has_input) {
			writer.SetRowNull(rid);
			continue;
		}
		if (!state.acc) {
			empty_builder.Reset();
			empty_builder.EmitObjectStart(0);
			empty_builder.EmitObjectEnd();
			writer.WriteRow(rid, empty_builder);
			continue;
		}
		// The accumulator is already a serialized blob; copy each component verbatim.
		WriteOwnedBlobRow(writer, rid, *state.acc);
	}
}

void JsonoGroupMergeFinalize(Vector &states, AggregateFinalizeInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	auto &bind_data = aggr_input_data.bind_data->Cast<GroupMergeBindData>();
	if (bind_data.shreds.empty()) {
		FinalizePlainGroups(result, state_fmt, state_data, count, offset);
		return;
	}
	// Native shredded finalize from the direct accumulator: write each group's state straight into
	// the result — the residual blobs verbatim (skips re-emitted with the manifest of the lanes
	// actually stripped), the scalar lanes from the lane slots — with no staged reconstruct and no
	// reshred. Per scalar shred the residual decides, matching what compose(residual-authoritative)
	// + reshred used to emit:
	//   - path present in the residual (a later diverted write outranks the older lane) -> NULL
	//     lane, spill bit set (the value lives in the residual);
	//   - a non-object prefix (the parent was replaced by a scalar) -> NULL lane, no bit
	//     (the lane value is dropped, replace semantics);
	//   - path absent and the lane active -> the lane value, no bit, manifested as stripped;
	//   - path absent and no lane -> NULL lane, no bit (absent field).
	// The fully-shredded hot case (empty residual) skips every locate.
	result.SetVectorType(VectorType::FLAT_VECTOR);
	JsonoBodyWriter writer;
	writer.Init(result);
	jsono::JsonoSpillStamp stamp;
	stamp.Init(result);
	auto &spill_ranks = bind_data.write_model.spill_ranks;
	vector<Vector *> lane_out(bind_data.shreds.size());
	for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
		lane_out[f] = &jsono::JsonoShredVector(result, f);
		lane_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}
	JsonoBuilder empty_builder;
	std::string skips_buf;
	JsonoStrippedLanes stripped_lanes(bind_data.write_model);
	for (idx_t i = 0; i < count; i++) {
		auto rid = offset + i;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		auto *direct = state.direct;
		if (!state.has_input) {
			// Zero non-NULL inputs -> SQL NULL (DEFAULT_NULL_HANDLING). Null the body and the whole
			// shreds subtree so DuckDB's struct Verify sees every child of the NULL row NULL too.
			writer.SetRowNull(rid);
			JsonoSetRowMarkerNull(result, rid);
			continue;
		}
		bool live = direct && direct->kind != DirectShreddedAcc::Kind::Empty;
		if (!live) {
			// The group had input but folded to an empty object -> {} with NULL shreds (absent, no spill).
			stamp.ResetRow();
			stamp.StampRow(rid);
			empty_builder.Reset();
			empty_builder.EmitObjectStart(0);
			empty_builder.EmitObjectEnd();
			writer.WriteRow(rid, empty_builder);
			for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
				FlatVector::SetNull(*lane_out[f], rid, true);
			}
			continue;
		}
		JsonoView acc_view = ViewOfBlob(direct->residual);
		// direct->residual is a self-built fold output (kind != Empty): the parse cannot fail. Fold the
		// asserted parse flag into residual_has_members so the flag is used in both build configs.
		bool acc_parsed = acc_view.ParseHeader();
		D_ASSERT(acc_parsed);
		bool residual_has_members = acc_parsed && acc_view.Slots() > 0 &&
		                            SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START &&
		                            ContainerChildCount(SlotPayload(acc_view.SlotAt(0))) > 0;
		stripped_lanes.Clear();
		stamp.ResetRow();
		D_ASSERT(direct->lanes.size() == bind_data.shreds.size());
		for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
			auto &plan = bind_data.shred_plan[f];
			if (plan.kind != ShredKind::Scalar) {
				// Array shreds have no lane state: their values sit fully composed in the residual.
				FlatVector::SetNull(*lane_out[f], rid, true);
				continue;
			}
			bool lane_active = direct->lanes[f].active;
			bool write_lane = false;
			if (residual_has_members) {
				idx_t depth_reached;
				switch (ClassifyResidualPath(acc_view, plan.steps, depth_reached)) {
				case ResidualPathOutcome::Present:
					// a later diverted write outranks the older lane; the value lives in the residual
					stamp.SetBit(spill_ranks[f]);
					break;
				case ResidualPathOutcome::Absent:
					if (lane_active) {
						write_lane = true;
					}
					break;
				case ResidualPathOutcome::NonObjectPrefix:
					break; // a parent was replaced by a scalar: drop the lane, no spill (replace semantics)
				}
			} else if (lane_active) {
				write_lane = true;
			}
			if (write_lane) {
				auto &lane = direct->lanes[f];
				auto &out = *lane_out[f];
				switch (plan.prim) {
				case jsono::JsonoScalarPrimitive::Varchar:
					FlatVector::GetDataMutable<string_t>(out)[rid] = StringVector::AddStringOrBlob(out, lane.s);
					break;
				case jsono::JsonoScalarPrimitive::Bigint:
					FlatVector::GetDataMutable<int64_t>(out)[rid] = lane.i;
					break;
				case jsono::JsonoScalarPrimitive::Ubigint:
					FlatVector::GetDataMutable<uint64_t>(out)[rid] = lane.u;
					break;
				case jsono::JsonoScalarPrimitive::Double:
					FlatVector::GetDataMutable<double>(out)[rid] = lane.d;
					break;
				case jsono::JsonoScalarPrimitive::Boolean:
					FlatVector::GetDataMutable<bool>(out)[rid] = lane.b;
					break;
				}
				stripped_lanes.Mark(f);
			} else {
				FlatVector::SetNull(*lane_out[f], rid, true);
			}
		}
		stamp.StampRow(rid);
		auto &blob = direct->residual;
		writer.data[BODY_SLOTS][rid] = WriteBlobInto(writer.Slots(), blob.slots.data(), blob.slots.size());
		writer.data[BODY_KEY_HEAP][rid] = WriteBlobInto(writer.KeyHeap(), blob.key_heap.data(), blob.key_heap.size());
		writer.data[BODY_STRING_HEAP][rid] =
		    WriteBlobInto(writer.StringHeap(), blob.string_heap.data(), blob.string_heap.size());
		writer.data[BODY_LENGTHS][rid] = WriteBlobInto(writer.Lengths(), blob.lengths.data(), blob.lengths.size());
		writer.data[BODY_NUMS][rid] = WriteBlobInto(writer.Nums(), blob.nums.data(), blob.nums.size());
		skips_buf.assign(blob.skips.data(), blob.skips.size());
		if (!stripped_lanes.Empty()) {
			JsonoAppendStrippedShredManifest(skips_buf, stripped_lanes);
		}
		writer.data[BODY_SKIPS][rid] = WriteBlobInto(writer.Skips(), skips_buf.data(), skips_buf.size());
	}
}

} // namespace

void RegisterJsonoGroupMerge(ExtensionLoader &loader) {
	AggregateFunction fun(
	    "jsono_group_merge", {LogicalType::ANY}, JsonoType(), AggregateFunction::StateSize<GroupMergeState>,
	    AggregateFunction::StateInitialize<GroupMergeState, GroupMergeFunction>, JsonoGroupMergeUpdate,
	    JsonoGroupMergeCombine, JsonoGroupMergeFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr,
	    JsonoGroupMergeBind, AggregateFunction::StateDestroy<GroupMergeState, GroupMergeFunction>);
	fun.SetOrderDependent(AggregateOrderDependent::ORDER_DEPENDENT);
	fun.SetFallible();
	loader.RegisterFunction(std::move(fun));
}

} // namespace duckdb
