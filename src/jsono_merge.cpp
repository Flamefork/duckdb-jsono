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

struct JsonoMergeLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	std::vector<MergeChild> merge_children_a;
	std::vector<MergeChild> merge_children_b;
	std::vector<MergePlanEntry> merge_plan;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoMergeLocalState>();
	}
};

// A shred carried through a shred-aware merge: its name, the result struct child index it
// lands in, and its type. The merge folds the six-BLOB residuals (existing logic) and
// copies each union shred from the last input that declares it. Shred keys are assumed
// disjoint from residual keys (true when shreds are computed fields lifted into columns).
struct MergeShred {
	string name;
	idx_t result_child_index;
	LogicalType type;
};

struct JsonoMergeBindData : public FunctionData {
	vector<MergeShred> shreds;

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

unique_ptr<FunctionData> JsonoMergePatchBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	if (arguments.empty()) {
		throw BinderException("jsono_merge_patch() requires at least one argument");
	}
	// Union the shreds of any shredded inputs (later inputs win a name conflict, matching
	// the patch-wins fold). A shredded input keeps its type so the executor can read its
	// shred columns; a plain input is bound as JSONO and contributes only its residual.
	auto bind_data = make_uniq<JsonoMergeBindData>();
	auto &shreds = bind_data->shreds;
	for (auto &argument : arguments) {
		if (argument->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		auto &type = argument->return_type;
		JsonoRequireExtensionOptimizerForShredded(context, type, "jsono_merge_patch");
		if (type.id() == LogicalTypeId::SQLNULL) {
			bound_function.arguments.push_back(JsonoType());
			continue;
		}
		if (IsShreddedJsonoType(type)) {
			JsonoLayoutType layout;
			TryParseJsonoLayoutType(type, layout);
			for (auto &layout_shred : layout.shreds) {
				// Union by the shred path (the inputs may carry different shred sets).
				auto &shred_name = layout_shred.first;
				bool found = false;
				for (auto &merge_shred : shreds) {
					if (merge_shred.name == shred_name) {
						merge_shred.type = layout_shred.second;
						found = true;
						break;
					}
				}
				if (!found) {
					shreds.push_back(MergeShred {shred_name, 0, layout_shred.second});
				}
			}
			bound_function.arguments.push_back(type);
			continue;
		}
		if (IsJsonoType(type)) {
			bound_function.arguments.push_back(JsonoType());
			continue;
		}
		throw BinderException("jsono_merge_patch() arguments must be JSONO");
	}
	if (shreds.empty()) {
		bound_function.return_type = JsonoType();
		return std::move(bind_data);
	}
	// Canonical shred order (sorted by name) so the merged type is a pure function of the shred set,
	// not of argument order: the executor writes shreds by result_child_index, so the index order and
	// the type's shred order must agree.
	std::sort(shreds.begin(), shreds.end(), [](const MergeShred &a, const MergeShred &b) { return a.name < b.name; });
	child_list_t<LogicalType> shred_types;
	idx_t next = 0; // shred-relative index inside the result's `.shreds` struct
	for (auto &shred : shreds) {
		shred.result_child_index = next++;
		shred_types.emplace_back(shred.name, shred.type);
	}
	bound_function.return_type = JsonoShreddedStructType(shred_types);
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
	static thread_local OwnedJsonoBlob acc_storage;
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

// True if the object has any top-level key that matches a merged shred name. `shreds`
// is sorted by name (bind canonicalizes it), so each of the object's (few) keys is
// binary-searched. Used to gate the fast path against a PLAIN input whose top-level
// key names a shred: a value there collides into the residual (also caught by the
// per-row residual probe), but an RFC 7396 null at that key DELETES the lane and
// leaves no trace in the folded residual — the lane copy-through would wrongly keep
// it, so any such key forces the reshred fallback.
bool ObjectKeyInShredSet(const JsonoView &view, const vector<MergeShred> &shreds) {
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
		size_t lo = 0;
		size_t hi = shreds.size();
		while (lo < hi) {
			auto mid = lo + (hi - lo) / 2;
			if (nonstd::string_view(shreds[mid].name) < key) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		if (lo < shreds.size() && nonstd::string_view(shreds[lo].name) == key) {
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

// True if two object-key shred paths overlap structurally: they agree on the full length of the
// shorter one, so one is a prefix of (or equal to) the other (e.g. `a` and `$.a.a`, or `a` and
// `$.a`). Such a pair cannot be two independent lanes — one names a scalar leaf, the other lives
// under it — so only the actual merge (the reshred fallback) can decide which structure wins.
// Sibling paths (`$.a.b` vs `$.a.c`) diverge before the shorter ends and do not conflict.
bool ShredPathsStructurallyConflict(const vector<PathStep> &a, const vector<PathStep> &b) {
	idx_t common = std::min(a.size(), b.size());
	for (idx_t i = 0; i < common; i++) {
		if (a[i].key != b[i].key) {
			return false;
		}
	}
	return true;
}

// Fill the result shred lane by selecting, PER ROW, the winning input's lane value. Every input
// that declares the shred is a candidate; on a row the value lives in the input where the key is
// present (its lane slot is valid). Patch/IgnoreNulls take the LAST such input (RFC 7396 last-wins),
// Overlay the FIRST (base-authoritative). A per-INPUT winner would drop the value on a row where the
// chosen input happens to lack the key. Present-null / diverted rows never reach here — the conflict
// scan diverts them — so a NULL lane slot on a kept row always means the key is absent from that
// input. All candidate lanes share the shred's type (the fast path bails a name declared with mixed
// types), so they stage side by side for one gather.
void FastCopyShred(DataChunk &args, idx_t ncols, idx_t count, MergeMode mode, const MergeShred &shred, Vector &dst,
                   const ValidityMask &result_validity) {
	vector<Vector *> lanes;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (!IsShreddedJsonoType(t)) {
			continue; // only shredded inputs declare shreds to copy from
		}
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(t, layout);
		for (idx_t c = 0; c < layout.shreds.size(); c++) {
			if (layout.shreds[c].first == shred.name) {
				args.data[i].Flatten(count);
				lanes.push_back(&JsonoShredVector(args.data[i], c));
			}
		}
	}
	dst.SetVectorType(VectorType::FLAT_VECTOR);
	if (lanes.empty()) {
		auto &dv = FlatVector::Validity(dst);
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
		auto &dv = FlatVector::Validity(dst);
		for (idx_t row = 0; row < count; row++) {
			if (!result_validity.RowIsValid(row)) {
				dv.SetInvalid(row);
			}
		}
	}
}

void JsonoFoldExecute(DataChunk &args, ExpressionState &state, Vector &result, MergeMode mode) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoMergeLocalState>();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<JsonoMergeBindData>();
	auto count = args.size();
	idx_t ncols = args.ColumnCount();
	bool has_shreds = !bind_data.shreds.empty();

	// Each input reader verifies its rows' manifests against that input's own shreds (a plain
	// input carries none, so any manifest entry on it fails loud). jsono_overlay is exempt: it
	// is the optimizer's reconstruction primitive and its residual argument legitimately
	// carries a manifest already verified by __jsono_internal_checked_residual.
	auto init_input = [&](JsonoRowReader &reader, Vector &input, idx_t row_count) {
		if (mode == MergeMode::Overlay) {
			reader.InitTrusted(input, row_count);
		} else {
			reader.Init(input, row_count);
		}
	};

	if (!has_shreds) {
		// Plain merge: fold straight into the result.
		vector<JsonoRowReader> inputs(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			init_input(inputs[i], args.data[i], count);
		}
		RunResidualFold(mode, inputs, ncols, count, result, lstate);
		if (args.AllConstant()) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Fast path: fold the raw residuals and copy shred lanes through verbatim, skipping the
	// per-row reconstruct+reshred. Both shredded AND plain inputs ride it: a plain input folds
	// its whole value into the residual and declares no lanes (FastCopyShred skips it). Every
	// shred is an object-key lane (top-level `K` or nested `$.p.q`) the read overlay re-inserts
	// at its path; a scalar-array shred copies through with its residual skeleton. LIST<STRUCT>
	// array shreds still fall back because each element merges multiple subfield lanes over a
	// per-element tail. The per-row gate diverts to the fallback whenever an input could make the
	// lane copy-through wrong (a non-object replace, a key/path naming a lane).
	bool fast_viable = true;
	vector<idx_t> plain_inputs;
	for (idx_t i = 0; i < ncols; i++) {
		auto &t = args.data[i].GetType();
		if (t.id() != LogicalTypeId::SQLNULL && !IsShreddedJsonoType(t)) {
			plain_inputs.push_back(i);
		}
	}
	// Parse each shred's object-key path once (top-level shreds are a single Key step). An array
	// shred with element objects, or any non-Key step (only reachable through an array path),
	// disqualifies the fast path.
	vector<vector<PathStep>> shred_paths(bind_data.shreds.size());
	vector<ShredKind> shred_kinds(bind_data.shreds.size());
	bool has_jsonpath_shred = false;
	bool has_nested_shred = false;
	for (idx_t k = 0; k < bind_data.shreds.size() && fast_viable; k++) {
		auto &shred = bind_data.shreds[k];
		shred_kinds[k] = ClassifyShredKind(shred.type);
		if (shred_kinds[k] == ShredKind::Array) {
			fast_viable = false;
			break;
		}
		shred_paths[k] = ShredNamePath(shred.name, "jsono merge fast path");
		// The JSONPath form gates the per-row nested-shred probe below; a bare-literal top-level key
		// is already covered there by ObjectKeyInShredSet.
		if (shred.name.size() >= 2 && shred.name[0] == '$' && shred.name[1] == '.') {
			has_jsonpath_shred = true;
		}
		for (auto &step : shred_paths[k]) {
			if (step.kind != PathStepKind::Key) {
				fast_viable = false;
				break;
			}
		}
		if (shred_paths[k].size() > 1) {
			has_nested_shred = true;
		}
	}
	// A nested shred whose path prefixes (or equals) another shred's path is structurally
	// contradictory — only the reshred fallback can resolve which wins. Two distinct top-level
	// keys never prefix each other, so this only matters once a nested shred is present.
	if (fast_viable && has_nested_shred) {
		for (idx_t a = 0; a < shred_paths.size() && fast_viable; a++) {
			for (idx_t b = a + 1; b < shred_paths.size(); b++) {
				if (ShredPathsStructurallyConflict(shred_paths[a], shred_paths[b])) {
					fast_viable = false;
					break;
				}
			}
		}
	}
	// A shred name declared with different types across inputs has incompatible lane layouts, so the
	// per-row lane copy cannot stage its candidates together. The reshred fallback coerces every input
	// to the merged (last-declared) type instead.
	for (idx_t i = 0; i < ncols && fast_viable; i++) {
		auto &t = args.data[i].GetType();
		if (!IsShreddedJsonoType(t)) {
			continue;
		}
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(t, layout);
		for (auto &layout_shred : layout.shreds) {
			for (auto &merge_shred : bind_data.shreds) {
				if (merge_shred.name == layout_shred.first && merge_shred.type != layout_shred.second) {
					fast_viable = false;
					break;
				}
			}
			if (!fast_viable) {
				break;
			}
		}
	}

	auto run_reshred_fallback = [&](const vector<Vector *> &fallback_args, idx_t fallback_count,
	                                Vector &fallback_result) {
		vector<unique_ptr<Vector>> reconstructed;
		vector<JsonoRowReader> inputs(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			auto &input = *fallback_args[i];
			if (IsShreddedJsonoType(input.GetType())) {
				// The reconstruction verifies the shredded input's manifest itself; its plain
				// output never carries one.
				auto plain = make_uniq<Vector>(JsonoType(), fallback_count);
				JsonoReconstructToPlain(input, fallback_count, *plain);
				init_input(inputs[i], *plain, fallback_count);
				reconstructed.push_back(std::move(plain));
			} else {
				init_input(inputs[i], input, fallback_count);
			}
		}
		Vector fold_out(JsonoType(), fallback_count);
		RunResidualFold(mode, inputs, ncols, fallback_count, fold_out, lstate);
		vector<std::pair<string, LogicalType>> shred_specs;
		shred_specs.reserve(bind_data.shreds.size());
		for (auto &shred : bind_data.shreds) {
			shred_specs.emplace_back(shred.name, shred.type);
		}
		JsonoShredFromSpec(fold_out, fallback_count, shred_specs, fallback_result);
	};

	if (fast_viable) {
		vector<JsonoRowReader> raw(ncols);
		for (idx_t i = 0; i < ncols; i++) {
			init_input(raw[i], args.data[i], count);
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
			if (shred_kinds[k] == ShredKind::ScalarArray) {
				return ResidualConflictsWithScalarArrayShredPath(view, shred_paths[k]);
			}
			if (shred_paths[k].size() == 1) {
				return ResidualHasTopLevelKey(view, bind_data.shreds[k].name);
			}
			return ResidualConflictsWithShredPath(view, shred_paths[k]);
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
		for (idx_t i = 0; i < ncols; i++) {
			auto &t = args.data[i].GetType();
			if (!IsShreddedJsonoType(t)) {
				continue;
			}
			JsonoLayoutType layout;
			TryParseJsonoLayoutType(t, layout);
			ShreddedProbeInput pin;
			pin.arg = i;
			for (idx_t c = 0; c < layout.shreds.size(); c++) {
				for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
					if (bind_data.shreds[k].name == layout.shreds[c].first) {
						UnifiedVectorFormat fmt;
						JsonoShredVector(args.data[i], c).ToUnifiedFormat(count, fmt);
						if (!fmt.validity.AllValid()) {
							pin.shred_ks.push_back(k);
							pin.lane_fmt.push_back(std::move(fmt));
						}
						break;
					}
				}
			}
			if (!pin.shred_ks.empty()) {
				probe_inputs.push_back(std::move(pin));
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
				// covers the top-level shred names in one pass; a plain input touching a NESTED shred path
				// (its leaf or a non-object along it) needs the per-path descent.
				if (ObjectKeyInShredSet(pview, bind_data.shreds)) {
					row_conflict = true;
					break;
				}
				if (has_jsonpath_shred) {
					for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
						auto &name = bind_data.shreds[k].name;
						if (name.size() >= 2 && name[0] == '$' && name[1] == '.' &&
						    ResidualConflictsWithShredPath(pview, shred_paths[k])) {
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
			FlatVector::Validity(result) = FlatVector::Validity(fast_residual);
			for (idx_t b = 0; b < BODY_BLOB_COUNT; b++) {
				if (b == BODY_SKIPS) {
					continue;
				}
				VectorOperations::Copy(*fr_blobs[b], *writer.vec[b], count, 0, 0);
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
			for (auto &shred : bind_data.shreds) {
				FastCopyShred(args, ncols, count, mode, shred, JsonoShredVector(result, shred.result_child_index),
				              result_validity);
			}
			// The folded residual was rebuilt without the inputs' manifests, but a copied shred
			// that carries a value has no copy in the residual (the no-conflict gate above) —
			// exactly what the manifest must record, or a later raw narrowing cast would lose it
			// silently. Re-emit each row's skips with the manifest of its non-NULL shreds.
			vector<JsonoShredManifestEntryBytes> manifest_entries(bind_data.shreds.size());
			vector<UnifiedVectorFormat> shred_fmt(bind_data.shreds.size());
			for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
				auto &shred = bind_data.shreds[k];
				manifest_entries[k] = JsonoShredManifestEntry(shred.name, shred.type);
				JsonoShredVector(result, shred.result_child_index).ToUnifiedFormat(count, shred_fmt[k]);
			}
			auto fr_skips = FlatVector::GetData<string_t>(*fr_blobs[BODY_SKIPS]);
			auto &fr_skips_validity = FlatVector::Validity(*fr_blobs[BODY_SKIPS]);
			auto &r_skips = writer.Skips();
			auto skips_out = writer.data[BODY_SKIPS];
			std::string skips_buf;
			vector<idx_t> stripped_fields;
			for (idx_t row = 0; row < count; row++) {
				if (!result_validity.RowIsValid(row) || !fr_skips_validity.RowIsValid(row)) {
					FlatVector::SetNull(r_skips, row, true);
					continue;
				}
				skips_buf.clear();
				skips_buf.append(fr_skips[row].GetData(), fr_skips[row].GetSize());
				stripped_fields.clear();
				for (idx_t k = 0; k < bind_data.shreds.size(); k++) {
					if (RowIsValid(shred_fmt[k], row)) {
						stripped_fields.push_back(k);
					}
				}
				if (!stripped_fields.empty()) {
					JsonoAppendShredManifest(skips_buf, manifest_entries, stripped_fields);
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
	                   nullptr, JsonoMergeLocalState::Init);
	fun.varargs = LogicalType::ANY;
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
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
	ScalarFunction fun("jsono_overlay", {}, JsonoType(), JsonoOverlayExecute, JsonoMergePatchBind, nullptr, nullptr,
	                   JsonoMergeLocalState::Init);
	fun.varargs = LogicalType::ANY;
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

void RegisterJsonoMerge(ExtensionLoader &loader) {
	loader.RegisterFunction(JsonoMergePatchFunction());
	// Registered for serialization only (see JsonoOverlayFunction).
	loader.RegisterFunction(JsonoOverlayFunction());
}

} // namespace duckdb
