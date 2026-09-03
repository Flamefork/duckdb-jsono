#include "jsono_merge.hpp"
#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_extension.hpp"
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
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// Where one merged shred's lane sits in one input: the argument, and the field index inside that
// input's `shreds` struct.
struct LaneSource {
	idx_t arg = DConstants::INVALID_INDEX;
	idx_t field = DConstants::INVALID_INDEX;
};

// The fast path's per-input facts: which arguments carry lanes, where each merged shred's lane sits
// in each of them, and whether the lane types agree. All of it is a function of the ARGUMENT TYPES,
// which are fixed for a bound expression — but recognizing one type costs a decode of every lane
// name it carries, so resolving it per shred per input per chunk is quadratic in the shred set.
// Resolved once per expression instead.
struct MergeInputPlan {
	vector<bool> shredded;                   // per argument
	vector<idx_t> plain_inputs;              // arguments carrying a residual but no lanes (SQLNULL excluded)
	vector<vector<LaneSource>> lane_sources; // per merged shred, in argument order
	bool lane_types_agree = true;            // a shred declared with one type by every input that has it
	// The argument types this plan was resolved against, compared by VALUE. An ExtraTypeInfo
	// address is no identity for the types without one — SQLNULL, VARCHAR, any scalar all carry
	// nullptr, so two different such types would compare equal, and a stale hit here reads lanes at
	// another type's field indices. For a STRUCT the equality's fast case is still one pointer
	// compare, and the resolve runs once per chunk.
	vector<LogicalType> resolved_for;
};

struct JsonoMergeLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	std::vector<MergeChild> merge_children_a;
	std::vector<MergeChild> merge_children_b;
	std::vector<MergePlanEntry> merge_plan;
	// One per input column: manifest signatures survive across chunks so the per-lane name decode
	// does not repeat on every call (see JsonoShredSignatures).
	std::vector<JsonoShredSignatures> input_signatures;
	MergeInputPlan input_plan;
	// Accumulator blob for the left-to-right residual fold; lives here (not `static thread_local` in
	// RunResidualFold) so no TLS destructor is registered — see FoldIntoGroupState in
	// jsono_group_merge.cpp for why TLS destructors are banned in this extension.
	OwnedJsonoBlob acc_storage;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoMergeLocalState>();
	}
};

// A shred carried through a shred-aware merge: its two names (the PHYSICAL field `name` the lanes
// are matched by across inputs, and the LOGICAL `steps` the residual is probed with), the result
// struct field it lands in, and its type. The merge folds the six-BLOB residuals (existing
// logic) and copies each union shred from the last input that declares it.
struct MergeShred {
	string name;
	// the field this shred occupies inside the RESULT's `shreds` struct, resolved by name at bind
	// (the result type is a bind fact) so the per-chunk write never re-scans the field names. Unset
	// until the result type exists, and unset is INVALID_INDEX rather than 0: field 0 of `shreds` is
	// the `$jsono$set` marker, so a zero default is not "not yet known" but a live wrong answer that
	// walks straight past JsonoShredFieldVector's bounds check into the marker.
	idx_t result_field_index = DConstants::INVALID_INDEX;
	LogicalType type;
	vector<PathStep> steps;
	ShredKind kind = ShredKind::Scalar;
};

struct JsonoMergeBindData : public FunctionData {
	vector<MergeShred> shreds;
	// Indices of the single-step shreds, so the fast path can binary-search a plain input's top-level
	// object keys against them (ObjectKeyInShredSet). They arrive in key order for free: the shreds
	// are sorted by physical name, and for a one-step path that name encodes the key alone, so the
	// encoding's order preservation makes this subsequence key-sorted already.
	vector<idx_t> top_level_shreds;
	// The merged shred set compiled for writing: the fast path re-emits every row's skips from its
	// model's manifest entries, and the reshred fallback shreds the folded plain value through it —
	// a bind fact, not a per-chunk one.
	ShredWriteSet write;
	// Whether the fast path's SHAPE preconditions hold: no array shred, every path step an object
	// key, and no shred path prefixing another. A function of the merged shred set alone — the one
	// remaining precondition (the inputs must declare each lane with one type) is per-input and
	// lives in MergeInputPlan.
	bool fast_shape_viable = true;
	bool has_nested_shred = false;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoMergeBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoMergeBindData>();
		if (shreds.size() != other.shreds.size()) {
			return false;
		}
		for (idx_t i = 0; i < shreds.size(); i++) {
			if (shreds[i].name != other.shreds[i].name || shreds[i].type != other.shreds[i].type) {
				return false;
			}
		}
		return true;
	}
};

// True if two object-key shred paths overlap structurally: they agree on the full length of the
// shorter one, so one is a prefix of (or equal to) the other (e.g. `a` and `$.a.a`). Such a pair
// cannot be two independent lanes — one names a scalar leaf, the other lives under it — so only the
// actual merge (the reshred fallback) can decide which structure wins. Sibling paths (`$.a.b` vs
// `$.a.c`) diverge before the shorter ends and do not conflict. The same shape is legal for a single
// value (see ShredPathsOverlap): rendering one value only has to show what is there, while merging
// two must pick a winner.
bool ShredPathsStructurallyConflict(const vector<PathStep> &a, const vector<PathStep> &b) {
	idx_t common = std::min(a.size(), b.size());
	for (idx_t i = 0; i < common; i++) {
		if (a[i].key != b[i].key) {
			return false;
		}
	}
	return true;
}

unique_ptr<FunctionData> JsonoMergePatchBind(BindScalarFunctionInput &input) {
	auto &context = input.GetClientContext();
	auto &bound_function = input.GetBoundFunction();
	auto &arguments = input.GetArguments();
	if (arguments.empty()) {
		throw BinderException(bound_function.GetName() + "() requires at least one argument");
	}
	// Union the shreds of any shredded inputs (later inputs win a name conflict, matching
	// the patch-wins fold). A shredded input keeps its type so the executor can read its
	// shred columns; a plain input is bound as JSONO and contributes only its residual.
	auto bind_data = make_uniq<JsonoMergeBindData>();
	auto &shreds = bind_data->shreds;
	for (idx_t arg_index = 0; arg_index < arguments.size(); arg_index++) {
		auto &argument = arguments[arg_index];
		if (argument->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		auto &type = argument->GetReturnType();
		JsonoRequireExtensionOptimizerForShredded(context, type, bound_function.GetName().GetIdentifierName());
		if (type.id() == LogicalTypeId::SQLNULL) {
			bound_function.GetArguments()[arg_index] = JsonoType();
			continue;
		}
		// One layout parse decides both branches: IsShreddedJsonoType/IsJsonoType would each re-parse
		// the type, decoding every lane name again at bind.
		JsonoLayoutType layout;
		if (TryParseJsonoLayoutType(type, layout)) {
			if (layout.kind != JsonoLayoutKind::Shredded) {
				bound_function.GetArguments()[arg_index] = JsonoType();
				continue;
			}
			for (auto &layout_shred : layout.shreds) {
				// Union by the shred path (the inputs may carry different shred sets).
				auto &shred_name = layout_shred.first.GetIdentifierName();
				bool found = false;
				for (auto &merge_shred : shreds) {
					if (merge_shred.name == shred_name) {
						merge_shred.type = layout_shred.second;
						found = true;
						break;
					}
				}
				if (!found) {
					MergeShred merge_shred;
					merge_shred.name = shred_name;
					merge_shred.type = layout_shred.second;
					merge_shred.steps = ShredNamePath(shred_name, bound_function.GetName().GetIdentifierName().c_str());
					shreds.push_back(std::move(merge_shred));
				}
			}
			bound_function.GetArguments()[arg_index] = type;
			continue;
		}
		// The bind is shared with jsono_overlay, so both diagnostics name the function actually called.
		JsonoRejectForeignLayout(type, bound_function.GetName() + "()");
		throw BinderException(bound_function.GetName() + "() arguments must be JSONO");
	}
	if (shreds.empty()) {
		bound_function.SetReturnType(JsonoType());
		return std::move(bind_data);
	}
	// Canonical shred order (sorted by physical name) so the merged type is a pure function of the
	// shred set, not of argument order: the type lists its lanes in that same order.
	std::sort(shreds.begin(), shreds.end(), [](const MergeShred &a, const MergeShred &b) { return a.name < b.name; });
	vector<JsonoLaneSpec> lanes;
	for (auto &shred : shreds) {
		shred.kind = ClassifyShredKind(shred.type);
		lanes.push_back(JsonoLaneSpec {shred.steps, shred.type});
	}
	for (idx_t k = 0; k < shreds.size(); k++) {
		if (shreds[k].steps.size() == 1) {
			bind_data->top_level_shreds.push_back(k);
		}
	}
	// ObjectKeyInShredSet binary-searches this subsequence by the LOGICAL key while the shreds it
	// indexes are ordered by the PHYSICAL name; the two agree only because a one-step path encodes its
	// key alone and both codec stages preserve order. Assert it where the subsequence is built, so a
	// codec change surfaces in the relassert build rather than as a missed RFC 7396 null-delete on the
	// fast path. No runtime gate: every honest bind would pay for it.
	D_ASSERT(std::is_sorted(bind_data->top_level_shreds.begin(), bind_data->top_level_shreds.end(),
	                        [&](idx_t a, idx_t b) { return shreds[a].steps[0].key < shreds[b].steps[0].key; }));
	vector<std::pair<string, LogicalType>> manifest_shreds;
	manifest_shreds.reserve(shreds.size());
	for (auto &shred : shreds) {
		manifest_shreds.emplace_back(shred.name, shred.type);
	}
	bind_data->write = JsonoBuildShredWriteSet(manifest_shreds);
	bound_function.SetReturnType(JsonoShreddedStructType(lanes));
	auto &result_shreds_type = JsonoShredsStructType(bound_function.GetReturnType());
	for (auto &shred : shreds) {
		shred.result_field_index = JsonoFindShredsFieldIndex(result_shreds_type, shred.name);
		if (shred.result_field_index == DConstants::INVALID_INDEX) {
			throw InternalException("%s: shred lane '%s' is missing from the result type built from these lanes",
			                        bound_function.GetName(), shred.name);
		}
	}
	// The fast path's shape preconditions. An array shred merges multiple subfield lanes per element
	// over a per-element tail, and a non-Key step (only reachable through an array path) has no lane
	// the read overlay re-inserts — neither can ride the lane copy-through. A nested shred whose path
	// prefixes (or equals) another's is structurally contradictory: only the reshred fallback can
	// resolve which structure wins. Two distinct top-level keys never prefix each other, so that pass
	// only matters once a nested shred is present.
	for (auto &shred : shreds) {
		if (shred.kind == ShredKind::Array) {
			bind_data->fast_shape_viable = false;
		}
		for (auto &step : shred.steps) {
			if (step.kind != PathStepKind::Key) {
				bind_data->fast_shape_viable = false;
			}
		}
		if (shred.steps.size() > 1) {
			bind_data->has_nested_shred = true;
		}
	}
	if (bind_data->has_nested_shred) {
		for (idx_t a = 0; a < shreds.size() && bind_data->fast_shape_viable; a++) {
			for (idx_t b = a + 1; b < shreds.size(); b++) {
				if (ShredPathsStructurallyConflict(shreds[a].steps, shreds[b].steps)) {
					bind_data->fast_shape_viable = false;
					break;
				}
			}
		}
	}
	return std::move(bind_data);
}

// Fold the residual views in `inputs` left-to-right under `mode` and write the merged plain
// JSONO into `out` (the six-BLOB residual). Shared by the shred-aware fast path (raw residuals)
// and the reshred fallback (reconstructed values). Each input reader carries its own manifest
// policy: the fold rebuilds the residual without the inputs' manifests, so an unverified
// narrowed input would launder the loss into a permanently undetectable one — the readers throw
// before rebuilding (except jsono_overlay's trusted residual, see JsonoFoldExecute).
void RunResidualFold(MergeMode mode, vector<JsonoRowReader> &inputs, idx_t ncols, idx_t count, Vector &out,
                     JsonoMergeLocalState &lstate) {
	JsonoBodyWriter writer;
	writer.Init(out);
	// jsono_merge_patch folds left to right (RFC 7396): the first argument is the
	// target document (kept verbatim — its nulls are values), the rest are patches.
	auto &acc_storage = lstate.acc_storage;
	JsonoView acc_view;
	JsonoView patch_view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow target_blob {};
		bool acc_is_sqlnull = inputs[0].Read(row, target_blob, acc_view) != JsonoRowState::Value;
		if (ncols == 1) {
			// A lone target is returned verbatim (its nulls survive).
			if (acc_is_sqlnull) {
				writer.SetRowNull(row);
			} else {
				lstate.builder.Reset();
				JsonoCursor cursor;
				EmitValueVerbatim(acc_view, cursor, lstate.builder, 0);
				writer.WriteRow(row, lstate.builder);
			}
			continue;
		}
		for (idx_t i = 1; i < ncols; i++) {
			JsonoBlobRow patch_blob {};
			bool patch_present = inputs[i].Read(row, patch_blob, patch_view) == JsonoRowState::Value;
			bool result_sqlnull =
			    MergeFoldStep(mode, acc_is_sqlnull, acc_view, patch_present, patch_view, lstate.builder,
			                  lstate.merge_children_a, lstate.merge_children_b, lstate.merge_plan);
			if (i + 1 == ncols) {
				// Last step writes straight to the output builder — no serialize round-trip.
				if (result_sqlnull) {
					writer.SetRowNull(row);
				} else {
					writer.WriteRow(row, lstate.builder);
				}
			} else if (result_sqlnull) {
				acc_is_sqlnull = true;
			} else {
				SerializeBuilderToBlob(lstate.builder, acc_storage);
				acc_view = ViewOfBlob(acc_storage);
				acc_view.ParseHeader();
				acc_is_sqlnull = false;
			}
		}
	}
}

// True if the residual object has `key` at top level (binary search over the sorted keys).
bool ResidualHasTopLevelKey(const JsonoView &view, const string &key) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return false;
	}
	auto layout = ReadObjectLayout(view, 0);
	nonstd::string_view target(key);
	size_t lo = 0;
	size_t hi = layout.key_count;
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto ks = view.SlotAt(layout.key_start + mid);
		if (SlotTag(ks) != tag::KEY) {
			return false;
		}
		if (view.KeyAt(SlotPayload(ks)) < target) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	if (lo >= layout.key_count) {
		return false;
	}
	auto ks = view.SlotAt(layout.key_start + lo);
	return SlotTag(ks) == tag::KEY && view.KeyAt(SlotPayload(ks)) == target;
}

// True if the object has any top-level key that a single-step shred lifts. The comparison is
// against the shred's LOGICAL key — a document key and a lane name are different namespaces, and
// only the path the lane means can collide with a key in the document. `top_level_shreds` indexes
// the single-step shreds in key order (bind canonicalizes it), so each of the object's (few) keys
// is binary-searched. Used to gate the fast path against a PLAIN input whose top-level key names a
// shred: a value there collides into the residual (also caught by the per-row residual probe), but
// an RFC 7396 null at that key DELETES the lane and leaves no trace in the folded residual — the
// lane copy-through would wrongly keep it, so any such key forces the reshred fallback.
bool ObjectKeyInShredSet(const JsonoView &view, const vector<MergeShred> &shreds,
                         const vector<idx_t> &top_level_shreds) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return false;
	}
	auto layout = ReadObjectLayout(view, 0);
	for (size_t k = 0; k < layout.key_count; k++) {
		auto ks = view.SlotAt(layout.key_start + k);
		if (SlotTag(ks) != tag::KEY) {
			break;
		}
		auto key = view.KeyAt(SlotPayload(ks));
		auto shred_key = [&](size_t i) {
			return nonstd::string_view(shreds[top_level_shreds[i]].steps[0].key);
		};
		size_t lo = 0;
		size_t hi = top_level_shreds.size();
		while (lo < hi) {
			auto mid = lo + (hi - lo) / 2;
			if (shred_key(mid) < key) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		if (lo < top_level_shreds.size() && shred_key(lo) == key) {
			return true;
		}
	}
	return false;
}

// A scalar-array shred deliberately leaves an array skeleton at the shred path in the residual.
// That terminal array is not a conflict: the copied LIST lane overlays it element-for-element on
// reconstruct. Anything else present at that path is a real replacement/collision and must fall
// back to the full reshred path.
bool ResidualConflictsWithScalarArrayShredPath(const JsonoView &view, const vector<PathStep> &steps) {
	JsonoCursor cursor;
	for (idx_t s = 0; s < steps.size(); s++) {
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			return true;
		}
		if (!LocatePathStep(nullptr, s, view, steps[s], cursor)) {
			return false;
		}
	}
	return cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::ARR_START;
}

// Resolve `plan` against the argument types unless it already is. A bound expression's argument
// types never change between chunks, so this rebuilds exactly once; keying it on the types
// themselves (the same key JsonoShredSignatures uses) means a plan is never reused across a type
// it was not resolved against, rather than trusting the caller to notice.
const MergeInputPlan &ResolveInputPlan(DataChunk &args, const vector<MergeShred> &shreds, MergeInputPlan &plan) {
	idx_t ncols = args.ColumnCount();
	bool resolved = plan.resolved_for.size() == ncols;
	for (idx_t i = 0; i < ncols && resolved; i++) {
		resolved = plan.resolved_for[i] == args.data[i].GetType();
	}
	if (resolved) {
		return plan;
	}
	plan.shredded.assign(ncols, false);
	plan.plain_inputs.clear();
	plan.lane_sources.assign(shreds.size(), vector<LaneSource>());
	plan.lane_types_agree = true;
	plan.resolved_for.resize(ncols);
	for (idx_t i = 0; i < ncols; i++) {
		auto &type = args.data[i].GetType();
		plan.resolved_for[i] = type;
		JsonoLayoutType layout;
		if (!TryParseJsonoLayoutType(type, layout) || layout.kind != JsonoLayoutKind::Shredded) {
			// A SQLNULL argument carries neither a residual nor lanes; every other input here is plain
			// JSONO and folds its whole value into the residual.
			if (type.id() != LogicalTypeId::SQLNULL) {
				plan.plain_inputs.push_back(i);
			}
			continue;
		}
		plan.shredded[i] = true;
		auto &shreds_type = JsonoShredsStructType(type);
		for (auto &lane : layout.shreds) {
			for (idx_t k = 0; k < shreds.size(); k++) {
				if (shreds[k].name != lane.first) {
					continue;
				}
				LaneSource source;
				source.arg = i;
				source.field = JsonoFindShredsFieldIndex(shreds_type, lane.first.GetIdentifierName());
				plan.lane_sources[k].push_back(source);
				// A shred name declared with different types across inputs has incompatible lane
				// layouts, so the per-row lane copy cannot stage its candidates together. The reshred
				// fallback coerces every input to the merged (last-declared) type instead.
				if (shreds[k].type != lane.second) {
					plan.lane_types_agree = false;
				}
				break;
			}
		}
	}
	return plan;
}

// Fill the result shred lane by selecting, PER ROW, the winning input's lane value. Every input
// that declares the shred is a candidate; on a row the value lives in the input where the key is
// present (its lane slot is valid). Patch/IgnoreNulls take the LAST such input (RFC 7396 last-wins),
// Overlay the FIRST (base-authoritative). A per-INPUT winner would drop the value on a row where the
// chosen input happens to lack the key. Present-null / diverted rows never reach here — the conflict
// scan diverts them — so a NULL lane slot on a kept row always means the key is absent from that
// input. All candidate lanes share the shred's type (the fast path bails a name declared with mixed
// types), so they stage side by side for one gather.
void FastCopyShred(DataChunk &args, idx_t count, MergeMode mode, const vector<LaneSource> &sources, Vector &dst,
                   const ValidityMask &result_validity) {
	vector<Vector *> lanes;
	for (auto &source : sources) {
		args.data[source.arg].Flatten(count);
		lanes.push_back(&JsonoShredFieldVector(args.data[source.arg], source.field));
	}
	dst.SetVectorType(VectorType::FLAT_VECTOR);
	if (lanes.empty()) {
		auto &dv = FlatVector::ValidityMutable(dst);
		for (idx_t row = 0; row < count; row++) {
			dv.SetInvalid(row);
		}
		return;
	}
	if (lanes.size() == 1) {
		VectorOperations::Copy(*lanes[0], dst, count, 0, 0);
	} else {
		// Stage every candidate lane side by side ([candidate * count + row]) so one gather resolves
		// the per-row winner. A row with no valid candidate points at candidate 0's slot, which is
		// NULL there (no candidate carries the key), so the gather yields the correct NULL.
		idx_t num = lanes.size();
		Vector staging(dst.GetType(), num * count);
		for (idx_t ci = 0; ci < num; ci++) {
			VectorOperations::Copy(*lanes[ci], staging, count, 0, ci * count);
		}
		SelectionVector sel(count);
		for (idx_t row = 0; row < count; row++) {
			idx_t winner = 0;
			if (mode == MergeMode::Overlay) {
				for (idx_t ci = 0; ci < num; ci++) {
					if (FlatVector::Validity(*lanes[ci]).RowIsValid(row)) {
						winner = ci;
						break;
					}
				}
			} else {
				for (idx_t ci = num; ci-- > 0;) {
					if (FlatVector::Validity(*lanes[ci]).RowIsValid(row)) {
						winner = ci;
						break;
					}
				}
			}
			sel.set_index(row, winner * count + row);
		}
		VectorOperations::Copy(staging, dst, sel, count, 0, 0);
	}
	if (!result_validity.AllValid()) {
		dst.Flatten(count);
		auto &dv = FlatVector::ValidityMutable(dst);
		for (idx_t row = 0; row < count; row++) {
			if (!result_validity.RowIsValid(row)) {
				dv.SetInvalid(row);
			}
		}
	}
}

void JsonoFoldExecute(DataChunk &args, ExpressionState &state, Vector &result, MergeMode mode) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoMergeLocalState>();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<JsonoMergeBindData>();
	auto count = args.size();
	idx_t ncols = args.ColumnCount();
	bool has_shreds = !bind_data.shreds.empty();

	// Each input reader verifies its rows' manifests against that input's own shreds (a plain
	// input carries none, so any manifest entry on it fails loud). jsono_overlay is exempt: it
	// is the optimizer's reconstruction primitive and its residual argument legitimately
	// carries a manifest already verified by __jsono_internal_checked_residual.
	if (lstate.input_signatures.size() < ncols) {
		lstate.input_signatures.resize(ncols);
	}
	// `cache` is the per-input signature cache when the reader reads that input's own declared type
	// (the fast path, where it is hit on every chunk after the first); the reshred fallback passes
	// none, because there the same slot would alternate between an input's type and its reconstructed
	// plain type and never hit.
	auto init_input = [&](JsonoRowReader &reader, Vector &input, idx_t row_count,
	                      optional_ptr<JsonoShredSignatures> cache) {
		if (mode == MergeMode::Overlay) {
			reader.InitTrusted(input, row_count);
		} else if (cache) {
			reader.Init(input, row_count, *cache);
		} else {
			reader.Init(input, row_count);
		}
	};

	if (!has_shreds) {
		// Plain merge: fold straight into the result.
		vector<JsonoRowReader> inputs(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			init_input(inputs[i], args.data[i], count, &lstate.input_signatures[i]);
		}
		RunResidualFold(mode, inputs, ncols, count, result, lstate);
		if (args.AllConstant()) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Fast path: fold the raw residuals and copy shred lanes through verbatim, skipping the
	// per-row reconstruct+reshred. Both shredded AND plain inputs ride it: a plain input folds
	// its whole value into the residual and declares no lanes (FastCopyShred copies from none). Every
	// shred is an object-key lane (top-level `K` or nested `$.p.q`) the read overlay re-inserts
	// at its path; a scalar-array shred copies through with its residual skeleton. LIST<STRUCT>
	// array shreds still fall back because each element merges multiple subfield lanes over a
	// per-element tail. The per-row gate diverts to the fallback whenever an input could make the
	// lane copy-through wrong (a non-object replace, a key/path naming a lane).
	//
	// Both halves of the viability question are answers about TYPES: the shape of the merged shred
	// set (bind) and the inputs' lane layouts (the plan, resolved once per expression).
	auto &plan = ResolveInputPlan(args, bind_data.shreds, lstate.input_plan);
	bool fast_viable = bind_data.fast_shape_viable && plan.lane_types_agree;
	auto &plain_inputs = plan.plain_inputs;

	auto run_reshred_fallback = [&](const vector<Vector *> &fallback_args, idx_t fallback_count,
	                                Vector &fallback_result) {
		vector<unique_ptr<Vector>> reconstructed;
		vector<JsonoRowReader> inputs(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			auto &input = *fallback_args[i];
			if (plan.shredded[i]) {
				// The reconstruction verifies the shredded input's manifest itself; its plain
				// output never carries one.
				auto plain = make_uniq<Vector>(JsonoType(), fallback_count);
				JsonoReconstructToPlain(input, fallback_count, *plain);
				init_input(inputs[i], *plain, fallback_count, nullptr);
				reconstructed.push_back(std::move(plain));
			} else {
				init_input(inputs[i], input, fallback_count, nullptr);
			}
		}
		Vector fold_out(JsonoType(), fallback_count);
		RunResidualFold(mode, inputs, ncols, fallback_count, fold_out, lstate);
		JsonoShredFromLayout(fold_out, fallback_count, bind_data.write, fallback_result);
	};

	if (fast_viable) {
		vector<JsonoRowReader> raw(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			init_input(raw[i], args.data[i], count, &lstate.input_signatures[i]);
		}
		Vector fast_residual(JsonoType(), count);
		RunResidualFold(mode, raw, ncols, count, fast_residual, lstate);
		JsonoVectorData fr;
		InitJsonoVectorData(fast_residual, count, fr);
		vector<idx_t> conflict_rows;
		JsonoView pview;
		JsonoBlobRow pblob {};
		// The residual-conflict test for one shred against a residual view (the folded residual or a
		// raw input's residual): a top-level key present, a nested path resolving to a present value or
		// breaking its object skeleton, or a scalar-array path holding anything but its array skeleton.
		auto residual_hits_shred = [&](const JsonoView &view, idx_t k) -> bool {
			if (bind_data.shreds[k].kind == ShredKind::ScalarArray) {
				return ResidualConflictsWithScalarArrayShredPath(view, bind_data.shreds[k].steps);
			}
			if (bind_data.shreds[k].steps.size() == 1) {
				// The residual holds DOCUMENT keys, so a top-level shred is probed by the key it lifts,
				// not by the field name its lane occupies.
				return ResidualHasTopLevelKey(view, bind_data.shreds[k].steps[0].key);
			}
			return ResidualConflictsWithShredPath(view, bind_data.shreds[k].steps);
		};
		// A shredded input keeps a shred key in its OWN residual only for a present-null (explicit
		// JSON null) or a diverted value — both cases the lane slot is NULL, so a valid lane means the
		// key is stripped. Probing such an input's residual per row lets the fold divert those rows: a
		// present-null delete that the residual fold erases (against a base that stripped the key) is
		// invisible to the folded-residual scan, but the lane copy-through would wrongly keep the base
		// value. A lane with no NULL slot in the whole chunk can never carry a residual key, so only
		// lanes that actually have a NULL slot are probe candidates — the schema-stable case (every
		// lane present) collects none and the per-row probe is skipped entirely.
		struct ShreddedProbeInput {
			idx_t arg;
			vector<idx_t> shred_ks;
			vector<UnifiedVectorFormat> lane_fmt;
		};
		vector<ShreddedProbeInput> probe_inputs;
		vector<idx_t> probe_of_arg(ncols, DConstants::INVALID_INDEX);
		for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
			for (auto &source : plan.lane_sources[k]) {
				UnifiedVectorFormat fmt;
				JsonoShredFieldVector(args.data[source.arg], source.field).ToUnifiedFormat(count, fmt);
				if (fmt.validity.AllValid()) {
					continue;
				}
				if (probe_of_arg[source.arg] == DConstants::INVALID_INDEX) {
					probe_of_arg[source.arg] = probe_inputs.size();
					probe_inputs.push_back(ShreddedProbeInput {source.arg, {}, {}});
				}
				auto &pin = probe_inputs[probe_of_arg[source.arg]];
				pin.shred_ks.push_back(k);
				pin.lane_fmt.push_back(std::move(fmt));
			}
		}
		for (idx_t row = 0; row < count; row++) {
			bool row_conflict = false;
			JsonoBlobRow blob {};
			if (!ReadJsonoRowStrict(fr, row, blob)) {
				continue; // SQL NULL folded residual: the row is NULL and its lanes are nulled below
			}
			JsonoView v = MakeJsonoView(blob);
			if (!v.ParseHeader()) {
				throw InternalException("jsono merge fast path: malformed folded residual blob");
			}
			// A non-object merged value (a non-object plain patch replaced the document) carries no
			// lanes; copying them would attach stale lanes to a scalar/array. The all-shredded case
			// never folds to a non-object with non-null lanes, so gate this on plain inputs to keep
			// that case on the fast path unchanged.
			if (!plain_inputs.empty() && SlotTag(v.SlotAt(0)) != tag::OBJ_START) {
				row_conflict = true;
			}
			for (idx_t k = 0; k < bind_data.shreds.size() && !row_conflict; k++) {
				// A top-level shred conflicts only if its key sits in the folded residual (a
				// non-object residual is handled by the plain-input gate above, preserving the
				// all-shredded behaviour). A nested shred additionally conflicts if a node along
				// its path is a present non-object (its lane cannot overlay into a scalar).
				if (residual_hits_shred(v, k)) {
					row_conflict = true;
					break;
				}
			}
			for (idx_t pi : plain_inputs) {
				if (row_conflict) {
					break;
				}
				if (raw[pi].Read(row, pblob, pview) != JsonoRowState::Value) {
					continue;
				}
				// A non-object plain value replaces the whole document mid-fold (RFC 7396),
				// discarding the base's lanes — but the lane copy-through still copies them, and a
				// later object patch can rebuild an object residual so the non-object check on the
				// folded residual alone misses it. Divert to the fallback on any non-object plain input.
				if (pview.Slots() == 0 || SlotTag(pview.SlotAt(0)) != tag::OBJ_START) {
					row_conflict = true;
					break;
				}
				// A plain input naming a shred also forces the fallback: a value there is caught by
				// the residual probe above, but an RFC 7396 null-delete of a lane leaves no trace in
				// the folded residual and the lane copy-through would wrongly keep it. ObjectKeyInShredSet
				// covers the top-level shred keys in one pass; a plain input touching a NESTED shred path
				// (its leaf or a non-object along it) needs the per-path descent.
				if (ObjectKeyInShredSet(pview, bind_data.shreds, bind_data.top_level_shreds)) {
					row_conflict = true;
					break;
				}
				if (bind_data.has_nested_shred) {
					for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
						if (bind_data.shreds[k].steps.size() > 1 &&
						    ResidualConflictsWithShredPath(pview, bind_data.shreds[k].steps)) {
							row_conflict = true;
							break;
						}
					}
					if (row_conflict) {
						break;
					}
				}
			}
			for (auto &pin : probe_inputs) {
				if (row_conflict) {
					break;
				}
				// Only a NULL lane can hide a present key (present-null or divert) in the input's
				// residual — a valid lane strips its key. Read the residual only when some declared
				// shred is NULL on this row, and probe only those shreds.
				bool any_null = false;
				for (idx_t j = 0; j < pin.shred_ks.size(); j++) {
					if (!RowIsValid(pin.lane_fmt[j], row)) {
						any_null = true;
						break;
					}
				}
				if (!any_null) {
					continue;
				}
				JsonoBlobRow rblob {};
				JsonoView rview;
				if (raw[pin.arg].Read(row, rblob, rview) != JsonoRowState::Value) {
					continue;
				}
				for (idx_t j = 0; j < pin.shred_ks.size(); j++) {
					if (!RowIsValid(pin.lane_fmt[j], row) && residual_hits_shred(rview, pin.shred_ks[j])) {
						row_conflict = true;
						break;
					}
				}
			}
			if (row_conflict) {
				conflict_rows.push_back(row);
			}
		}
		auto write_fast_result = [&](bool preserve_constant) {
			// Capture constness before FastCopyShred flattens any input (which would otherwise
			// make AllConstant() spuriously false and trip the constant-fold assertion).
			bool all_constant = preserve_constant && args.AllConstant();
			fast_residual.Flatten(count);
			// Copy the folded plain residual's body blobs into the shredded result's body —
			// except `skips`, which is re-emitted per row below with the output's shred manifest.
			JsonoBodyWriter writer;
			writer.Init(result);
			// Reasoning below holds for the rows kept from this fast result: every row when there are
			// no conflicts, the non-conflict rows when the caller splits (conflict rows are written
			// speculatively here and overwritten by the reshred fallback below, so their possibly
			// diverted lanes never reach the output). In a kept row scalar shred keys do not appear in
			// the folded residual. Scalar-array keys may appear as their normal skeleton arrays, but they
			// never set spill bits. A diverted scalar (case B) would sit in that residual at the scalar
			// shred's key and trip the gate to the fallback, so every NULL scalar shred in a kept row is
			// an absent path (case A), never a diversion — the zero spill bitmap is exact.
			JsonoFillShredMarker(result, count);
			auto &fr_blobs = StructVector::GetEntries(JsonoBodyVector(fast_residual));
			FlatVector::ValidityMutable(result) = FlatVector::Validity(fast_residual);
			for (idx_t b = 0; b < BODY_BLOB_COUNT; b++) {
				if (b == BODY_SKIPS) {
					continue;
				}
				VectorOperations::Copy(fr_blobs[b], *writer.vec[b], count, 0, 0);
			}
			auto &result_validity = FlatVector::Validity(result);
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row)) {
					// The layout/body levels must go NULL with the row (struct Verify requires every
					// child of a NULL struct row to be NULL); the blob copies above only carry the
					// residual's own child validity, not the intermediate struct levels.
					writer.SetRowNull(row);
					JsonoSetRowMarkerNull(result, row);
				}
			}
			for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
				FastCopyShred(args, count, mode, plan.lane_sources[k],
				              JsonoShredFieldVector(result, bind_data.shreds[k].result_field_index), result_validity);
			}
			// The folded residual was rebuilt without the inputs' manifests, but a copied shred
			// that carries a value has no copy in the residual (the no-conflict gate above) —
			// exactly what the manifest must record, or a later raw narrowing cast would lose it
			// silently. Re-emit each row's skips with the manifest of its non-NULL shreds.
			vector<UnifiedVectorFormat> shred_fmt(bind_data.shreds.size());
			for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
				JsonoShredFieldVector(result, bind_data.shreds[k].result_field_index)
				    .ToUnifiedFormat(count, shred_fmt[k]);
			}
			auto fr_skips = FlatVector::GetData<string_t>(fr_blobs[BODY_SKIPS]);
			auto &fr_skips_validity = FlatVector::Validity(fr_blobs[BODY_SKIPS]);
			auto &r_skips = writer.Skips();
			auto skips_out = writer.data[BODY_SKIPS];
			std::string skips_buf;
			JsonoStrippedLanes stripped_lanes(bind_data.write.Model());
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row) || !fr_skips_validity.RowIsValid(row)) {
					FlatVector::SetNull(r_skips, row, true);
					continue;
				}
				skips_buf.clear();
				skips_buf.append(fr_skips[row].GetData(), fr_skips[row].GetSize());
				stripped_lanes.Clear();
				for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
					if (RowIsValid(shred_fmt[k], row)) {
						stripped_lanes.Mark(k);
					}
				}
				if (!stripped_lanes.Empty()) {
					JsonoAppendStrippedShredManifest(skips_buf, stripped_lanes);
				}
				skips_out[row] = WriteBlobInto(r_skips, skips_buf.data(), skips_buf.size());
			}
			if (all_constant) {
				result.SetVectorType(VectorType::CONSTANT_VECTOR);
			}
		};
		if (conflict_rows.empty()) {
			write_fast_result(true);
			return;
		}
		if (conflict_rows.size() < count) {
			write_fast_result(false);
			auto conflict_count = conflict_rows.size();
			SelectionVector conflict_sel(conflict_count);
			for (idx_t i = 0; i < conflict_count; i++) {
				conflict_sel.set_index(i, conflict_rows[i]);
			}
			vector<unique_ptr<Vector>> compact_vectors;
			vector<Vector *> compact_args;
			compact_vectors.reserve(ncols);
			compact_args.reserve(ncols);
			for (idx_t i = 0; i < ncols; i++) {
				auto compact = make_uniq<Vector>(args.data[i].GetType(), conflict_count);
				if (args.data[i].GetType().id() == LogicalTypeId::SQLNULL) {
					compact->SetVectorType(VectorType::CONSTANT_VECTOR);
					ConstantVector::SetNull(*compact, true);
				} else {
					// source_count counts the selection entries read (it sizes dict_sel.Slice for a
					// dictionary source); conflict_sel holds exactly conflict_count of them, so passing
					// the full chunk count over-reads conflict_sel into wild dictionary indices.
					VectorOperations::Copy(args.data[i], *compact, conflict_sel, conflict_count, 0, 0, conflict_count);
				}
				compact_args.push_back(compact.get());
				compact_vectors.push_back(std::move(compact));
			}
			Vector fallback_result(result.GetType(), conflict_count);
			run_reshred_fallback(compact_args, conflict_count, fallback_result);
			auto &identity = *FlatVector::IncrementalSelectionVector();
			for (idx_t i = 0; i < conflict_count; i++) {
				VectorOperations::Copy(fallback_result, result, identity, conflict_count, i, conflict_rows[i], 1);
			}
			return;
		}
		// A shred key landed in the residual (conflict) — fall through to the correct reshred path.
	}

	// Reshred fallback: reconstruct shredded inputs to plain, fold, reshred. Correct for any
	// shred/residual key overlap and for plain inputs that can shadow a shred.
	vector<Vector *> fallback_args;
	fallback_args.reserve(ncols);
	for (idx_t i = 0; i < ncols; i++) {
		fallback_args.push_back(&args.data[i]);
	}
	run_reshred_fallback(fallback_args, count, result);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoMergePatchExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	JsonoFoldExecute(args, state, result, MergeMode::Patch);
}

void JsonoOverlayExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	JsonoFoldExecute(args, state, result, MergeMode::Overlay);
}

// jsono_merge_patch(target, patch...): RFC 7396 merge_patch, the headline user operation of this
// file. Nothing outside it needs the factory — the optimizer folds shred patches with jsono_overlay.
ScalarFunction JsonoMergePatchFunction() {
	ScalarFunction fun("jsono_merge_patch", {}, JsonoType(), JsonoMergePatchExecute, JsonoMergePatchBind, nullptr,
	                   JsonoMergeLocalState::Init);
	fun.SetVarArgs(LogicalType::ANY);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.SetFallible();
	return fun;
}

} // namespace

// jsono_overlay(base, patch...): the base is authoritative, each patch fills only the
// keys the base lacks (B-null is a no-op). This is the shredded-reconstruction primitive
// (residual wins, shreds refill stripped paths), not a headline user operation — the
// optimizer binds it directly via this factory. It is registered in the catalog (see
// RegisterJsonoMerge) only so the optimizer-injected reconstruction expression survives
// plan (de)serialization via a name lookup; it is intentionally left out of the docs.
ScalarFunction JsonoOverlayFunction() {
	ScalarFunction fun("jsono_overlay", {}, JsonoType(), JsonoOverlayExecute, JsonoMergePatchBind, nullptr,
	                   JsonoMergeLocalState::Init);
	fun.SetVarArgs(LogicalType::ANY);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.SetFallible();
	return fun;
}

void RegisterJsonoMerge(ExtensionLoader &loader) {
	loader.RegisterFunction(JsonoMergePatchFunction());
	// Registered for serialization only (see JsonoOverlayFunction).
	loader.RegisterFunction(JsonoOverlayFunction());
}

} // namespace duckdb
