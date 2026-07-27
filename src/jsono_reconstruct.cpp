#include "jsono_reconstruct.hpp"
#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_extension.hpp"
#include "jsono_merge_core.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_scalar_write.hpp"
#include "jsono_shred.hpp"
#include "jsono_shred_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// Emit one shred lane value into the patch under construction. Marked ALWAYS_INLINE because every
// overlay path calls it once per shred per row: the primitive dispatch must fold into the caller.
JSONO_ALWAYS_INLINE void EmitReconShredScalar(JsonoBuilder &builder, const LogicalType &type,
                                              const UnifiedVectorFormat &fmt, idx_t idx) {
	auto kind = JsonoScalarPrimitiveFromType(type, "jsono reconstruct");
	EmitJsonoPrimitiveVectorValue(builder, kind, fmt, idx);
}

// Build ONE JSONO object patch covering all shreds in present[lo,hi) — a sorted, contiguous run
// sharing the same first `depth` path steps — recursing on shared key prefixes so the whole shred set
// folds into a single overlay instead of one merge per shred. A top-level scalar is a terminal leaf at
// depth 0; a nested path recurses. Every shred in the run has steps.size() > depth; the shred
// invariant makes a key either a terminal leaf (a single shred) or an object container (a deeper run),
// never both. `present` is sorted by path, so equal keys at a depth are contiguous and the emitted
// object keys come out ascending (the sorted-key invariant MergeTwoObjects requires).
void EmitShredPatchObject(JsonoBuilder &builder, const vector<ReconShred> &shreds, const vector<idx_t> &present,
                          const vector<UnifiedVectorFormat> &shred_fmt, idx_t row, idx_t lo, idx_t hi, idx_t depth) {
	auto run_end = [&](idx_t i) {
		auto &key = shreds[present[i]].steps[depth].key;
		idx_t j = i + 1;
		while (j < hi && shreds[present[j]].steps[depth].key == key) {
			j++;
		}
		return j;
	};
	idx_t distinct = 0;
	for (idx_t i = lo; i < hi; i = run_end(i)) {
		distinct++;
	}
	builder.EmitObjectStart(distinct);
	for (idx_t i = lo; i < hi; i = run_end(i)) {
		builder.EmitKeySlot(shreds[present[i]].steps[depth].key);
	}
	for (idx_t i = lo; i < hi;) {
		idx_t j = run_end(i);
		builder.EmitObjectChildStart();
		if (shreds[present[i]].steps.size() == depth + 1) {
			// Terminal leaf — a single shred on this key by the invariant.
			idx_t k = present[i];
			EmitReconShredScalar(builder, shreds[k].type, shred_fmt[k], RowIndex(shred_fmt[k], row));
		} else {
			EmitShredPatchObject(builder, shreds, present, shred_fmt, row, i, j, depth + 1);
		}
		i = j;
	}
	builder.EmitObjectEnd();
}

// ---- Array shred overlay (read side) ----

// One subfield of an array shred's element struct, as the reconstruct overlay needs it: the JSON
// key it is emitted under and its scalar type.
struct ReconArraySubfield {
	string key;
	LogicalType type;
};

// A LIST<STRUCT> shred column holds, per row, one struct per element of the skeleton array, in
// lockstep. Reconstruct rebuilds the array: each object element merges its present shred subfields
// (residual-authoritative) back over its skeleton tail; a non-object/null element or a NULL shred
// struct is emitted verbatim. The skeleton array length is authoritative — a non-NULL shred list of
// a different length is a corrupt row (e.g. a struct cast that truncated the LIST) and fails loud.
struct ArrayReconShred {
	idx_t child; // shred index over the shred set
	ShredKind kind = ShredKind::Array;
	vector<PathStep> path;        // pure object-key chain to the array
	UnifiedVectorFormat list_fmt; // list_entry_t + per-row validity (both kinds)
	const list_entry_t *list_entries = nullptr;
	// kind == Array: the element struct's subfields lifted into a LIST<STRUCT> column. `key` is the
	// element's JSON key the overlay emits, decoded from the subfield's physical field name (which is
	// that key encoded, not the key itself); the physical access is the index into `sub_fmt`.
	vector<ReconArraySubfield> subfields; // struct order
	// Subfield indices in sorted-key order: the overlay patch object must emit keys ascending
	// (the JSONO object invariant and the sorted-key two-pointer MergeTwoObjects both require it).
	vector<idx_t> sorted_subfields;
	UnifiedVectorFormat struct_fmt;      // per-element struct validity
	vector<UnifiedVectorFormat> sub_fmt; // per subfield: value + validity
	// kind == ScalarArray: each whole element lifted into a LIST<element_type> column.
	LogicalType element_type;
	UnifiedVectorFormat element_fmt; // the list child scalar vector: value + per-element validity
};

struct ArrayOverlayScratch {
	JsonoBuilder patch_builder;
	OwnedJsonoBlob patch_storage;
};

// Rebuild one array (cursor at ARR_START), overlaying the row's shred subfields onto each skeleton
// element, advancing `cursor` past ARR_END.
void OverlayArray(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder, const ArrayReconShred &shred,
                  idx_t row, ArrayOverlayScratch &scratch, size_t depth) {
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	builder.EmitArrayStart();
	cursor.pos++; // first element; ARR_START consumes no stream entries
	auto list_idx = shred.list_fmt.sel->get_index(row);
	if (!shred.list_fmt.validity.RowIsValid(list_idx)) {
		// NULL shred list: the array stayed whole in the residual (nothing lifted) — emit verbatim.
		while (cursor.pos < end_pos) {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
		builder.EmitArrayEnd();
		cursor.pos = end_pos + 1;
		return;
	}
	auto entry = shred.list_entries[list_idx];
	idx_t i = 0;
	while (cursor.pos < end_pos) {
		if (i >= entry.length) {
			throw InvalidInputException("malformed JSONO: array shred list is shorter than the residual array "
			                            "(a struct cast truncated it)");
		}
		idx_t child = entry.offset + i;
		if (shred.kind == ShredKind::ScalarArray) {
			// A non-NULL element slot is a lifted scalar (emit it; the skeleton placeholder is skipped);
			// a NULL slot is a kept element (a non-conforming scalar / null / object / array), emitted
			// verbatim from the skeleton. Validity alone disambiguates the lifted from the kept.
			auto elem_idx = shred.element_fmt.sel->get_index(child);
			if (shred.element_fmt.validity.RowIsValid(elem_idx)) {
				EmitReconShredScalar(builder, shred.element_type, shred.element_fmt, elem_idx);
				SkipValueFast(view, cursor);
			} else {
				EmitValueVerbatim(view, cursor, builder, depth + 1);
			}
			i++;
			continue;
		}
		bool is_object = SlotTag(view.SlotAt(cursor.pos)) == tag::OBJ_START;
		bool struct_present = is_object && shred.struct_fmt.validity.RowIsValid(shred.struct_fmt.sel->get_index(child));
		size_t present = 0;
		if (struct_present) {
			for (size_t j = 0; j < shred.subfields.size(); j++) {
				if (shred.sub_fmt[j].validity.RowIsValid(shred.sub_fmt[j].sel->get_index(child))) {
					present++;
				}
			}
		}
		if (present > 0) {
			// The patch object must list its keys ascending (sorted_subfields), so the merge's
			// sorted-key two-pointer walk and the JSONO object invariant both hold.
			scratch.patch_builder.Reset();
			scratch.patch_builder.EmitObjectStart(present);
			for (auto j : shred.sorted_subfields) {
				if (shred.sub_fmt[j].validity.RowIsValid(shred.sub_fmt[j].sel->get_index(child))) {
					scratch.patch_builder.EmitKeySlot(
					    nonstd::string_view(shred.subfields[j].key.data(), shred.subfields[j].key.size()));
				}
			}
			for (auto j : shred.sorted_subfields) {
				auto sub_idx = shred.sub_fmt[j].sel->get_index(child);
				if (shred.sub_fmt[j].validity.RowIsValid(sub_idx)) {
					scratch.patch_builder.EmitObjectChildStart();
					EmitReconShredScalar(scratch.patch_builder, shred.subfields[j].type, shred.sub_fmt[j], sub_idx);
				}
			}
			scratch.patch_builder.EmitObjectEnd();
			SerializeBuilderToBlob(scratch.patch_builder, scratch.patch_storage);
			JsonoView patch_view = ViewOfBlob(scratch.patch_storage);
			patch_view.ParseHeader();
			// A authoritative (the skeleton element keeps its tail and any kept null/diverted subfield);
			// B fills the lifted subfields that were stripped from the skeleton.
			MergeTwoObjects(view, cursor, patch_view, JsonoCursor(), builder, MergeMode::Overlay, depth + 1);
			SkipValueFast(view, cursor);
		} else {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
		i++;
	}
	if (i != entry.length) {
		throw InvalidInputException("malformed JSONO: array shred list is longer than the residual array "
		                            "(a struct cast altered it)");
	}
	builder.EmitArrayEnd();
	cursor.pos = end_pos + 1;
}

// Walk the scalar-overlaid value, replacing each array-shred path's skeleton array with the
// overlaid array; advances `cursor` past the emitted value (mirrors EmitValueVerbatim).
void EmitArrayOverlay(const JsonoView &view, JsonoCursor &cursor, JsonoBuilder &builder,
                      const std::vector<const ArrayReconShred *> &array_shreds, idx_t row, ArrayOverlayScratch &scratch,
                      size_t depth) {
	if (SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
		EmitValueVerbatim(view, cursor, builder, depth);
		return;
	}
	auto layout = ReadObjectLayout(view, cursor.pos);
	auto key_at = [&](size_t i) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		return view.KeyAt(SlotPayload(key_slot));
	};
	builder.EmitObjectStart(layout.key_count);
	for (size_t i = 0; i < layout.key_count; i++) {
		builder.EmitKeySlot(key_at(i));
	}
	cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key = key_at(i);
		builder.EmitObjectChildStart();
		const ArrayReconShred *terminal = nullptr;
		std::vector<const ArrayReconShred *> deeper;
		for (auto *shred : array_shreds) {
			if (PathTerminatesOnKey(shred->path, depth, key)) {
				terminal = shred;
			} else if (PathContinuesPastKey(shred->path, depth, key)) {
				deeper.push_back(shred);
			}
		}
		auto value_tag = SlotTag(view.SlotAt(cursor.pos));
		if (terminal && value_tag == tag::ARR_START) {
			OverlayArray(view, cursor, builder, *terminal, row, scratch, depth);
		} else if (!deeper.empty() && value_tag == tag::OBJ_START) {
			EmitArrayOverlay(view, cursor, builder, deeper, row, scratch, depth + 1);
		} else {
			EmitValueVerbatim(view, cursor, builder, depth + 1);
		}
	}
	builder.EmitObjectEnd();
	if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	cursor.pos++;
}

// Backs both reconstruct entry points; jsono_reconstruct.hpp carries their contract. Scalar shreds
// are disjoint from residual keys (the shred invariant), so the overlay re-inserts them instead of
// merging values: an all-top-level shred set takes the direct flat patch (the common jsono({...})
// case), and a set carrying any nested path (`$.a.b`) folds every present shred into ONE patch tree
// (EmitShredPatchObject) applied in a single overlay. An array shred (LIST<STRUCT>) instead carries
// a skeleton array in the residual; after the scalar overlay, EmitArrayOverlay rebuilds each such
// array in lockstep. `shred_filter` (shred indices over the shred set) restricts the overlay to
// those shreds, while the manifest check below still covers all of them.
void ReconstructShreddedToPlainImpl(Vector &input, idx_t count, Vector &result,
                                    const vector<idx_t> *shred_filter = nullptr) {
	JsonoLayoutType layout;
	if (!TryParseJsonoLayoutType(input.GetType(), layout)) {
		throw InternalException("jsono reconstruct: input type '%s' is not a JSONO value", input.GetType().ToString());
	}
	vector<ReconShred> shreds;
	vector<ArrayReconShred> array_shreds;
	for (idx_t i = 0; i < layout.shreds.size(); i++) {
		if (shred_filter && std::find(shred_filter->begin(), shred_filter->end(), i) == shred_filter->end()) {
			continue;
		}
		auto &name = layout.shreds[i].first;
		vector<PathStep> steps = ShredNamePath(name, "jsono reconstruct");
		if (IsShredArrayType(layout.shreds[i].second)) {
			ArrayReconShred ars;
			ars.child = i;
			ars.kind = ShredKind::Array;
			ars.path = std::move(steps);
			auto &element = ListType::GetChildType(layout.shreds[i].second);
			for (auto &sub : StructType::GetChildTypes(element)) {
				// The element field name is encoded like a lane name; the overlay emits the JSON key.
				ars.subfields.push_back(
				    ReconArraySubfield {JsonoLaneSubfieldKey(sub.first, "jsono reconstruct"), sub.second});
			}
			array_shreds.push_back(std::move(ars));
			continue;
		}
		if (IsShredScalarArrayType(layout.shreds[i].second)) {
			ArrayReconShred ars;
			ars.child = i;
			ars.kind = ShredKind::ScalarArray;
			ars.path = std::move(steps);
			ars.element_type = ListType::GetChildType(layout.shreds[i].second);
			array_shreds.push_back(std::move(ars));
			continue;
		}
		// No duplicate check: the lane name is a bijection on paths, so two lanes of one path are two
		// STRUCT fields of one name — a type DuckDB cannot even build.
		shreds.push_back(ReconShred {i, layout.shreds[i].second, std::move(steps)});
	}
	std::sort(shreds.begin(), shreds.end(), [](const ReconShred &a, const ReconShred &b) {
		auto n = std::min(a.steps.size(), b.steps.size());
		for (size_t i = 0; i < n; i++) {
			if (a.steps[i].key != b.steps[i].key) {
				return a.steps[i].key < b.steps[i].key;
			}
		}
		return a.steps.size() < b.steps.size();
	});
	// Whether any shred is a nested path (steps.size() > 1) decides the per-row strategy ONCE, off the
	// hot path. The common case — all shreds top-level — keeps the direct two-pass flat patch (one merge,
	// no per-row index vector, no grouping). Only a type that actually carries a nested shred pays the
	// general patch-tree path, which folds top-level + nested into one tree applied in a single overlay
	// (replacing the old per-shred re-serialize loop that went super-linear in shred count).
	bool has_nested_shred = false;
	for (auto &shred : shreds) {
		if (shred.steps.size() > 1) {
			has_nested_shred = true;
			break;
		}
	}
	vector<idx_t> present_shreds; // reused per row by the patch-tree path only

	// The row reader verifies each row's shred manifest against the shreds this type carries
	// (the manifest is checked against ALL of them even under a shred_filter).
	JsonoRowReader residual_reader;
	residual_reader.Init(input, count);
	vector<UnifiedVectorFormat> shred_fmt(shreds.size());
	for (idx_t k = 0; k < shreds.size(); k++) {
		JsonoShredVector(input, shreds[k].child).ToUnifiedFormat(count, shred_fmt[k]);
	}
	// Array shred read handles: the per-row list entries plus, per kind, the element lanes —
	// a LIST<STRUCT>'s per-element struct validity and per-subfield value+validity, or a
	// LIST<scalar>'s element value+validity — set up once the array_shreds vector is stable.
	for (auto &ars : array_shreds) {
		auto &list_vec = JsonoShredVector(input, ars.child);
		list_vec.ToUnifiedFormat(count, ars.list_fmt);
		ars.list_entries = UnifiedVectorFormat::GetData<list_entry_t>(ars.list_fmt);
		auto child_size = ListVector::GetListSize(list_vec);
		if (ars.kind == ShredKind::ScalarArray) {
			ListVector::GetEntry(list_vec).ToUnifiedFormat(child_size, ars.element_fmt);
			continue;
		}
		auto &struct_vec = ListVector::GetEntry(list_vec);
		struct_vec.ToUnifiedFormat(child_size, ars.struct_fmt);
		auto &subs = StructVector::GetEntries(struct_vec);
		ars.sub_fmt.resize(subs.size());
		for (idx_t j = 0; j < subs.size(); j++) {
			subs[j]->ToUnifiedFormat(child_size, ars.sub_fmt[j]);
		}
		ars.sorted_subfields.resize(ars.subfields.size());
		for (idx_t j = 0; j < ars.subfields.size(); j++) {
			ars.sorted_subfields[j] = j;
		}
		std::sort(ars.sorted_subfields.begin(), ars.sorted_subfields.end(),
		          [&](idx_t a, idx_t b) { return ars.subfields[a].key < ars.subfields[b].key; });
	}
	std::vector<const ArrayReconShred *> array_shred_ptrs;
	for (auto &ars : array_shreds) {
		array_shred_ptrs.push_back(&ars);
	}

	JsonoBodyWriter writer;
	writer.Init(result);

	JsonoBuilder patch_builder;
	JsonoBuilder out_builder;
	JsonoBuilder final_builder;
	OwnedJsonoBlob patch_storage;
	OwnedJsonoBlob scalar_storage;
	ArrayOverlayScratch overlay_scratch;
	JsonoView residual_view;
	// Finish a row: with array shreds, rebuild each skeleton array over the scalar-overlaid value;
	// otherwise write the scalar overlay directly.
	auto finish_row = [&](idx_t row, JsonoBuilder &value_builder) {
		if (array_shred_ptrs.empty()) {
			writer.WriteRow(row, value_builder);
			return;
		}
		SerializeBuilderToBlob(value_builder, scalar_storage);
		JsonoView scalar_view = ViewOfBlob(scalar_storage);
		scalar_view.ParseHeader();
		final_builder.Reset();
		JsonoCursor cursor;
		EmitArrayOverlay(scalar_view, cursor, final_builder, array_shred_ptrs, row, overlay_scratch, 0);
		writer.WriteRow(row, final_builder);
	};
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob {};
		if (residual_reader.Read(row, blob, residual_view) != JsonoRowState::Value) {
			writer.SetRowNull(row);
			continue;
		}
		if (SlotTag(residual_view.SlotAt(0)) != tag::OBJ_START) {
			// A non-object residual (scalar/array) is the whole value: object-key shreds (scalar or
			// array) cannot match it, so emit it verbatim — no overlay applies.
			out_builder.Reset();
			JsonoCursor cursor;
			EmitValueVerbatim(residual_view, cursor, out_builder, 0);
			writer.WriteRow(row, out_builder);
			continue;
		}
		if (shreds.empty()) {
			// Array-only shred set: no scalar overlay, the residual is the skeleton to rebuild.
			out_builder.Reset();
			JsonoCursor cursor;
			EmitValueVerbatim(residual_view, cursor, out_builder, 0);
			finish_row(row, out_builder);
			continue;
		}
		if (!has_nested_shred) {
			// Hot path: every shred is top-level. Build the flat patch directly — count present, emit keys
			// then values — with no per-row index vector and no grouping scan.
			idx_t present = 0;
			for (idx_t k = 0; k < shreds.size(); k++) {
				present += RowIsValid(shred_fmt[k], row) ? 1 : 0;
			}
			if (present == 0) {
				out_builder.Reset();
				JsonoCursor cursor;
				EmitValueVerbatim(residual_view, cursor, out_builder, 0);
				finish_row(row, out_builder);
				continue;
			}
			patch_builder.Reset();
			patch_builder.EmitObjectStart(present);
			for (idx_t k = 0; k < shreds.size(); k++) {
				if (RowIsValid(shred_fmt[k], row)) {
					patch_builder.EmitKeySlot(shreds[k].steps[0].key);
				}
			}
			for (idx_t k = 0; k < shreds.size(); k++) {
				if (!RowIsValid(shred_fmt[k], row)) {
					continue;
				}
				patch_builder.EmitObjectChildStart();
				EmitReconShredScalar(patch_builder, shreds[k].type, shred_fmt[k], RowIndex(shred_fmt[k], row));
			}
			patch_builder.EmitObjectEnd();
			SerializeBuilderToBlob(patch_builder, patch_storage);
			JsonoView patch_view = ViewOfBlob(patch_storage);
			patch_view.ParseHeader();
			out_builder.Reset();
			MergeTwoObjects(residual_view, JsonoCursor(), patch_view, JsonoCursor(), out_builder, MergeMode::Overlay,
			                0);
			finish_row(row, out_builder);
			continue;
		}
		// A nested shred is present: fold ALL present object-key shreds into ONE patch tree applied in a
		// SINGLE overlay. The residual is authoritative; the overlay refills only the stripped leaves,
		// recursing into a shared key so a nested shred never drops the residual's other subkeys.
		present_shreds.clear();
		for (idx_t k = 0; k < shreds.size(); k++) {
			if (RowIsValid(shred_fmt[k], row)) {
				present_shreds.push_back(k);
			}
		}
		if (present_shreds.empty()) {
			out_builder.Reset();
			JsonoCursor cursor;
			EmitValueVerbatim(residual_view, cursor, out_builder, 0);
			finish_row(row, out_builder);
			continue;
		}
		patch_builder.Reset();
		EmitShredPatchObject(patch_builder, shreds, present_shreds, shred_fmt, row, 0, present_shreds.size(), 0);
		SerializeBuilderToBlob(patch_builder, patch_storage);
		JsonoView patch_view = ViewOfBlob(patch_storage);
		patch_view.ParseHeader();
		out_builder.Reset();
		MergeTwoObjects(residual_view, JsonoCursor(), patch_view, JsonoCursor(), out_builder, MergeMode::Overlay, 0);
		finish_row(row, out_builder);
	}
}

struct DirectListJsonShred {
	string key;
	// The element subfields' JSON keys, decoded from the lane's element field names: they are both
	// emitted into the output and sort-merged against the residual's (byte-sorted) document keys, so
	// they must be the logical names. The lane itself is addressed by index.
	vector<string> subfield_keys;
	vector<idx_t> sorted_subfields;
	ShredLane lane;
};

void AppendDirectListLaneValue(const UnifiedVectorFormat &fmt, idx_t idx, JsonoScalarPrimitive kind, std::string &out) {
	AppendScalarJsonText(JsonoScalarFromPrimitiveVector(kind, fmt, idx), out);
}

void AppendDirectObjectArrayElement(const JsonoView &view, JsonoCursor &cursor, const DirectListJsonShred &shred,
                                    idx_t child, std::string &out, size_t depth) {
	auto layout = ReadObjectLayout(view, cursor.pos);
	JsonoCursor residual = cursor;
	residual.pos = layout.value_start;
	auto struct_idx = shred.lane.struct_fmt.sel->get_index(child);
	bool struct_present = shred.lane.struct_fmt.validity.RowIsValid(struct_idx);
	idx_t residual_field = 0;
	idx_t lane_field = 0;
	idx_t emitted = 0;
	out.push_back('{');
	while (residual_field < layout.key_count || lane_field < shred.sorted_subfields.size()) {
		while (lane_field < shred.sorted_subfields.size()) {
			auto field = shred.sorted_subfields[lane_field];
			auto sub_idx = shred.lane.sub_fmt[field].sel->get_index(child);
			if (struct_present && shred.lane.sub_fmt[field].validity.RowIsValid(sub_idx)) {
				break;
			}
			lane_field++;
		}
		if (residual_field >= layout.key_count && lane_field >= shred.sorted_subfields.size()) {
			break;
		}

		nonstd::string_view residual_key;
		if (residual_field < layout.key_count) {
			auto key_slot = view.SlotAt(layout.key_start + residual_field);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			residual_key = view.KeyAt(SlotPayload(key_slot));
		}
		int cmp = 1;
		if (lane_field >= shred.sorted_subfields.size()) {
			cmp = -1;
		} else if (residual_field < layout.key_count) {
			auto field = shred.sorted_subfields[lane_field];
			auto &lane_key = shred.subfield_keys[field];
			cmp = CompareJsonoKeys(residual_key, nonstd::string_view(lane_key.data(), lane_key.size()));
		}

		if (emitted++) {
			out.push_back(',');
		}
		if (cmp <= 0) {
			AppendJsonString(residual_key, out);
			out.push_back(':');
			AppendJsonValueText(view, residual, out, depth + 1);
			residual_field++;
			if (cmp == 0) {
				lane_field++;
			}
			continue;
		}

		auto field = shred.sorted_subfields[lane_field++];
		auto &lane_key = shred.subfield_keys[field];
		AppendJsonString(nonstd::string_view(lane_key.data(), lane_key.size()), out);
		out.push_back(':');
		auto sub_idx = shred.lane.sub_fmt[field].sel->get_index(child);
		AppendDirectListLaneValue(shred.lane.sub_fmt[field], sub_idx, shred.lane.sub_kind[field], out);
	}
	if (residual.pos >= view.Slots() || SlotTag(view.SlotAt(residual.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	residual.pos++;
	cursor = residual;
	out.push_back('}');
}

void AppendDirectArrayOverlay(const JsonoView &view, JsonoCursor &cursor, const DirectListJsonShred &shred, idx_t row,
                              std::string &out, size_t depth) {
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	out.push_back('[');
	cursor.pos++;
	auto list_idx = shred.lane.fmt.sel->get_index(row);
	if (!shred.lane.fmt.validity.RowIsValid(list_idx)) {
		idx_t element = 0;
		while (cursor.pos < end_pos) {
			if (element++) {
				out.push_back(',');
			}
			AppendJsonValueText(view, cursor, out, depth + 1);
		}
		out.push_back(']');
		cursor.pos = end_pos + 1;
		return;
	}

	auto entry = UnifiedVectorFormat::GetData<list_entry_t>(shred.lane.fmt)[list_idx];
	idx_t element = 0;
	while (cursor.pos < end_pos) {
		if (element >= entry.length) {
			throw InvalidInputException("malformed JSONO: array shred list is shorter than the residual array "
			                            "(a struct cast truncated it)");
		}
		if (element) {
			out.push_back(',');
		}
		auto child = entry.offset + element;
		if (shred.lane.kind == ShredKind::ScalarArray) {
			auto &element_fmt = shred.lane.sub_fmt[0];
			auto element_idx = element_fmt.sel->get_index(child);
			if (element_fmt.validity.RowIsValid(element_idx)) {
				AppendDirectListLaneValue(element_fmt, element_idx, shred.lane.sub_kind[0], out);
				SkipValueFast(view, cursor);
			} else {
				AppendJsonValueText(view, cursor, out, depth + 1);
			}
		} else if (SlotTag(view.SlotAt(cursor.pos)) == tag::OBJ_START) {
			AppendDirectObjectArrayElement(view, cursor, shred, child, out, depth + 1);
		} else {
			AppendJsonValueText(view, cursor, out, depth + 1);
		}
		element++;
	}
	if (element != entry.length) {
		throw InvalidInputException("malformed JSONO: array shred list is longer than the residual array "
		                            "(a struct cast altered it)");
	}
	out.push_back(']');
	cursor.pos = end_pos + 1;
}

void AppendDirectTopLevelListJson(const JsonoView &view, const vector<DirectListJsonShred> &shreds, idx_t row,
                                  std::string &out) {
	JsonoCursor cursor;
	if (SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		AppendJsonValueText(view, cursor, out, 0);
		return;
	}

	auto layout = ReadObjectLayout(view, 0);
	cursor.pos = layout.value_start;
	idx_t shred = 0;
	out.push_back('{');
	for (idx_t field = 0; field < layout.key_count; field++) {
		auto key_slot = view.SlotAt(layout.key_start + field);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		while (shred < shreds.size() &&
		       CompareJsonoKeys(nonstd::string_view(shreds[shred].key.data(), shreds[shred].key.size()), key) < 0) {
			shred++;
		}
		if (field) {
			out.push_back(',');
		}
		AppendJsonString(key, out);
		out.push_back(':');
		bool overlays =
		    shred < shreds.size() &&
		    CompareJsonoKeys(nonstd::string_view(shreds[shred].key.data(), shreds[shred].key.size()), key) == 0 &&
		    SlotTag(view.SlotAt(cursor.pos)) == tag::ARR_START;
		if (overlays) {
			AppendDirectArrayOverlay(view, cursor, shreds[shred], row, out, 1);
			shred++;
		} else {
			AppendJsonValueText(view, cursor, out, 1);
		}
	}
	if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	out.push_back('}');
}

void RenderShreddedListsToJsonImpl(Vector &input, idx_t count, Vector &result) {
	JsonoLayoutType layout;
	if (!TryParseJsonoLayoutType(input.GetType(), layout) || layout.shreds.empty()) {
		throw InternalException("__jsono_shredded_lists_to_json requires a shredded JSONO input");
	}

	vector<idx_t> scalar_shreds;
	vector<DirectListJsonShred> list_shreds;
	for (idx_t f = 0; f < layout.shreds.size(); f++) {
		auto &shred_type = layout.shreds[f].second;
		if (!IsShredListType(shred_type)) {
			scalar_shreds.push_back(f);
			continue;
		}
		auto steps = ShredNamePath(layout.shreds[f].first, "__jsono_shredded_lists_to_json");
		if (steps.size() != 1) {
			throw InternalException("__jsono_shredded_lists_to_json requires top-level list shred paths");
		}
		DirectListJsonShred shred;
		shred.key = std::move(steps[0].key);
		if (IsShredArrayType(shred_type)) {
			for (auto &field : StructType::GetChildTypes(ListType::GetChildType(shred_type))) {
				// Decoded: these keys are both emitted as JSON and sort-merged against the residual's
				// byte-sorted document keys, so they must be the logical names.
				shred.subfield_keys.push_back(JsonoLaneSubfieldKey(field.first, "__jsono_shredded_lists_to_json"));
			}
			shred.sorted_subfields.resize(shred.subfield_keys.size());
			for (idx_t field = 0; field < shred.sorted_subfields.size(); field++) {
				shred.sorted_subfields[field] = field;
			}
			std::sort(shred.sorted_subfields.begin(), shred.sorted_subfields.end(),
			          [&](idx_t a, idx_t b) { return shred.subfield_keys[a] < shred.subfield_keys[b]; });
		}
		InitShredLane(JsonoShredVector(input, f), count, shred_type, shred.lane);
		list_shreds.push_back(std::move(shred));
	}
	std::sort(list_shreds.begin(), list_shreds.end(),
	          [](const DirectListJsonShred &a, const DirectListJsonShred &b) { return a.key < b.key; });
	for (idx_t f = 1; f < list_shreds.size(); f++) {
		if (list_shreds[f - 1].key == list_shreds[f].key) {
			throw InternalException("__jsono_shredded_lists_to_json requires unique list shred paths");
		}
	}

	Vector scalar_plain(JsonoType(), count);
	Vector *source = &input;
	if (!scalar_shreds.empty()) {
		JsonoOverlayShredsToPlain(input, count, scalar_shreds, scalar_plain);
		source = &scalar_plain;
	}
	JsonoRowReader reader;
	reader.Init(*source, count);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	std::string out;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		out.clear();
		AppendDirectTopLevelListJson(view, list_shreds, row, out);
		result_data[row] = StringVector::AddString(result, out.data(), out.size());
	}
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// __jsono_internal_checked_residual(residual, shreds): pass the plain residual through after
// verifying its shred manifest against `shreds`, the shred signatures of the
// shredded type the optimizer read the residual out of. The optimizer wraps every residual
// reinterpret with this check, so a row narrowed by a raw struct cast (its manifest lists a
// shred the type no longer carries) fails loud on every optimizer read path — extract fallback,
// overlay reconstruction, introspection — instead of silently reading an incomplete residual.
void JsonoCheckedResidualExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	auto &shreds_value = args.data[1];
	if (shreds_value.GetVectorType() != VectorType::CONSTANT_VECTOR) {
		throw InvalidInputException("__jsono_internal_checked_residual: shreds must be constant");
	}
	auto shred_signatures = JsonoShredSignaturesFromValue(shreds_value.GetValue(0));

	JsonoRowReader reader;
	reader.Init(args.data[0], count, std::move(shred_signatures));
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		reader.ReadPermissive(row, blob, view);
	}
	result.Reference(args.data[0]);
}

// __jsono_internal_strip_manifest(skips): drop the shred-manifest tail from a residual's skips blob,
// so a reader sees it as manifest-free. The soft residual reinterpret (the COALESCE fallback arm of a
// typed shred read) wraps its skips with this: a read of a shred-lane path absent from the residual
// then yields plain NULL — the lane holds the value — instead of the point-read manifest guard's
// narrowing throw. That throw is only correct when the read context lacks the lane (a genuinely
// narrowed row), which reads through the checked residual, never this fallback. Letting the manifest
// survive made any EAGER evaluation of the fallback (the projector fuse, any future hoist out of the
// COALESCE) throw on every row whose value lives in its lane.
void JsonoStripManifestExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t skips) {
		auto offset = JsonoSkipsManifestOffset(reinterpret_cast<const uint8_t *>(skips.GetData()), skips.GetSize());
		return StringVector::AddStringOrBlob(result, skips.GetData(), offset);
	});
}

} // namespace

void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result) {
	ReconstructShreddedToPlainImpl(input, count, result);
}

void JsonoOverlayShredsToPlain(Vector &input, idx_t count, const vector<idx_t> &shreds, Vector &result) {
	ReconstructShreddedToPlainImpl(input, count, result, &shreds);
}

void JsonoRenderShreddedListsToJson(Vector &input, idx_t count, Vector &result) {
	RenderShreddedListsToJsonImpl(input, count, result);
}

ScalarFunction JsonoCheckedResidualFunction() {
	ScalarFunction fun("__jsono_internal_checked_residual", {JsonoType(), LogicalType::LIST(JsonoShredSignatureType())},
	                   JsonoType(), JsonoCheckedResidualExecute);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

ScalarFunction JsonoStripManifestFunction() {
	ScalarFunction fun("__jsono_internal_strip_manifest", {LogicalType::BLOB}, LogicalType::BLOB,
	                   JsonoStripManifestExecute);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

// The manifest guards are registered in the catalog only so the optimizer-injected expressions
// survive plan (de)serialization via a name lookup; they are intentionally left out of the docs.
void RegisterJsonoReconstruct(ExtensionLoader &loader) {
	loader.RegisterFunction(JsonoCheckedResidualFunction());
	loader.RegisterFunction(JsonoStripManifestFunction());
}

} // namespace duckdb
