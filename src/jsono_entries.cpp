#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_shred.hpp"
#include "jsono_shred_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/string_util.hpp"
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
#include <unordered_set>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// EnsureListCapacity / FinishListRow / SetListRowNull are shared LIST helpers in jsono_writer.hpp.

enum class JsonoEntriesKeyStyle : uint8_t { JsonPath, Dotted };

// Where the leaf-flattening walk stops at an array. IndexedElements (default) recurses into the
// array, indexing each element; WholeJson stops at the array boundary and emits the whole array as
// one JSON-text leaf (§3-§7 of plan 031). Orthogonal to key_style.
enum class JsonoEntriesArrayStyle : uint8_t { IndexedElements, WholeJson };

// A shred of a shredded input: its struct child index, type, and the entry key in both styles
// (precomputed at bind). A top-level literal shred name `n` keys as `n` (dotted) or `$.n`
// (jsonpath); a `$.`-prefixed path keys as itself (jsonpath) or without the prefix (dotted). A
// scalar shred flattens to one entry at its key; an array shred (LIST<STRUCT>) expands each
// element's lifted subfields into `<key>[i].<subfield>` leaves keyed by `subfield_names`, the key
// being the array's base path. The runtime value lanes and shred kinds come from InitShredLanes.
struct EntriesShred {
	idx_t child_index;
	LogicalType type;
	string key_jsonpath;
	string key_dotted;
	vector<string> subfield_names; // element-struct order; empty for a scalar shred
};

struct JsonoEntriesBindData : public FunctionData {
	JsonoEntriesBindData(JsonoEntriesKeyStyle style_p, JsonoEntriesArrayStyle array_style_p,
	                     vector<EntriesShred> shreds_p)
	    : style(style_p), array_style(array_style_p), shreds(std::move(shreds_p)) {
	}
	JsonoEntriesKeyStyle style;
	JsonoEntriesArrayStyle array_style;
	vector<EntriesShred> shreds;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoEntriesBindData>(style, array_style, shreds);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoEntriesBindData>();
		if (style != other.style || array_style != other.array_style || shreds.size() != other.shreds.size()) {
			return false;
		}
		for (idx_t i = 0; i < shreds.size(); i++) {
			if (shreds[i].child_index != other.shreds[i].child_index || shreds[i].type != other.shreds[i].type) {
				return false;
			}
		}
		return true;
	}
};

// Defined below with the walk-side path builders; the bind rebuilds each shred lane's entry keys
// through the same builders so a shredded value keys identically to a plain parse.
void AppendKeySegment(std::string &path, nonstd::string_view key, JsonoEntriesKeyStyle style);

unique_ptr<FunctionData> JsonoEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto style = JsonoEntriesKeyStyle::JsonPath;
	auto array_style = JsonoEntriesArrayStyle::IndexedElements;
	// key_style and array_style are named-only and order-independent: a call places them positionally in
	// argument order with the parameter name carried as the expression alias (TransformNamedArg), so we
	// dispatch by alias, not position. A bare positional value (empty alias) is rejected.
	for (idx_t i = 1; i < arguments.size(); i++) {
		auto &arg = arguments[i];
		auto alias = arg->GetAlias();
		if (alias != "key_style" && alias != "array_style") {
			throw BinderException("jsono_entries: unknown argument '%s' (pass key_style := 'jsonpath' | "
			                      "'dotted', array_style := 'indexed_elements' | 'whole_json')",
			                      alias);
		}
		if (arg->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arg->IsFoldable()) {
			throw BinderException("jsono_entries: %s must be constant", alias);
		}
		auto value = ExpressionExecutor::EvaluateScalar(context, *arg);
		if (value.IsNull()) {
			throw BinderException("jsono_entries: %s must not be NULL", alias);
		}
		auto name = StringValue::Get(value);
		StringUtil::Trim(name);
		name = StringUtil::Lower(name);
		if (alias == "key_style") {
			if (name == "jsonpath") {
				style = JsonoEntriesKeyStyle::JsonPath;
			} else if (name == "dotted") {
				style = JsonoEntriesKeyStyle::Dotted;
			} else {
				throw BinderException("jsono_entries: key_style must be 'jsonpath' or 'dotted'");
			}
		} else {
			if (name == "indexed_elements") {
				array_style = JsonoEntriesArrayStyle::IndexedElements;
			} else if (name == "whole_json") {
				array_style = JsonoEntriesArrayStyle::WholeJson;
			} else {
				throw BinderException("jsono_entries: array_style must be 'indexed_elements' or 'whole_json'");
			}
		}
	}
	auto &input_type = arguments[0]->return_type;
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], "jsono_entries", false);
	vector<EntriesShred> shreds;
	if (IsShreddedJsonoType(input_type)) {
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(input_type, layout);
		for (idx_t i = 0; i < layout.shreds.size(); i++) {
			auto &name = layout.shreds[i].first;
			EntriesShred shred {i, layout.shreds[i].second, "", "", {}};
			// Rebuild both key styles segment-by-segment through the walk's builders (via ShredNamePath),
			// so a quotable object key (containing . [ ] ") keys exactly as a plain parse does — jsonpath
			// quotes/escapes it, dotted joins it raw. A lane name is always a non-empty pure object-key
			// chain (enforced at spec parse and layout recognition), so every step is a Key.
			shred.key_jsonpath = "$";
			for (auto &step : ShredNamePath(name, "jsono_entries")) {
				auto key = nonstd::string_view(step.key.data(), step.key.size());
				AppendKeySegment(shred.key_jsonpath, key, JsonoEntriesKeyStyle::JsonPath);
				AppendKeySegment(shred.key_dotted, key, JsonoEntriesKeyStyle::Dotted);
			}
			if (ClassifyShredKind(shred.type) == ShredKind::Array) {
				for (auto &sub : StructType::GetChildTypes(ListType::GetChildType(shred.type))) {
					shred.subfield_names.push_back(sub.first);
				}
			}
			shreds.push_back(std::move(shred));
		}
	}
	return make_uniq<JsonoEntriesBindData>(style, array_style, std::move(shreds));
}

// A bare JSONPath key segment is parseable only while it avoids the grammar's
// delimiters; anything else (or an empty key) must be quoted.
bool JsonPathKeyNeedsQuote(nonstd::string_view key) {
	if (key.empty()) {
		return true;
	}
	for (char c : key) {
		if (c == '.' || c == '[' || c == ']' || c == '"') {
			return true;
		}
	}
	return false;
}

void AppendKeySegment(std::string &path, nonstd::string_view key, JsonoEntriesKeyStyle style) {
	if (style == JsonoEntriesKeyStyle::JsonPath) {
		path.push_back('.');
		if (JsonPathKeyNeedsQuote(key)) {
			path.push_back('"');
			for (char c : key) {
				if (c == '"' || c == '\\') {
					path.push_back('\\');
				}
				path.push_back(c);
			}
			path.push_back('"');
		} else {
			path.append(key.data(), key.size());
		}
		return;
	}
	if (!path.empty()) {
		path.push_back('.');
	}
	path.append(key.data(), key.size());
}

void AppendIndexSegment(std::string &path, idx_t index, JsonoEntriesKeyStyle style) {
	if (style == JsonoEntriesKeyStyle::JsonPath) {
		path.push_back('[');
		AppendUnsignedText(index, path);
		path.push_back(']');
		return;
	}
	if (!path.empty()) {
		path.push_back('.');
	}
	AppendUnsignedText(index, path);
}

// Output sink for jsono_entries. Resolves the LIST child (key/value) vectors once
// per execute instead of per leaf — the per-leaf GetEntry/GetEntries re-resolution
// dominated the profile. `next_child` is the running write cursor into the shared
// child vectors; data pointers are refreshed only when a capacity grow moves them.
// A single reused `scratch` renders non-text scalars, and text scalars are copied
// straight from the heap (no intermediate std::string per leaf).
struct EntriesSink {
	Vector &list;
	Vector *key_vec;
	Vector *value_vec;
	string_t *key_data;
	string_t *value_data;
	idx_t capacity;
	idx_t next_child;
	std::string scratch;

	explicit EntriesSink(Vector &list_p) : list(list_p), next_child(ListVector::GetListSize(list_p)) {
		auto &struct_entries = StructVector::GetEntries(ListVector::GetEntry(list_p));
		key_vec = struct_entries[0].get();
		value_vec = struct_entries[1].get();
		capacity = ListVector::GetListCapacity(list_p);
		key_data = FlatVector::GetData<string_t>(*key_vec);
		value_data = FlatVector::GetData<string_t>(*value_vec);
	}

	void Append(nonstd::string_view key, const JsonoScalar &scalar) {
		if (next_child >= capacity) {
			EnsureListCapacity(list, next_child + 1);
			capacity = ListVector::GetListCapacity(list);
			key_data = FlatVector::GetData<string_t>(*key_vec);
			value_data = FlatVector::GetData<string_t>(*value_vec);
		}
		auto child_row = next_child;
		key_data[child_row] = StringVector::AddString(*key_vec, key.data(), key.size());
		switch (scalar.kind) {
		case JsonoScalarKind::Null:
			FlatVector::SetNull(*value_vec, child_row, true);
			break;
		case JsonoScalarKind::String:
		case JsonoScalarKind::NumberText:
			// Already text in the input string_heap — point straight at it (zero-copy); the
			// caller referenced that heap into value_vec so the bytes outlive the result.
			value_data[child_row] = ZeroCopyHeapText(scalar.text);
			break;
		default:
			scratch.clear();
			RenderExtractText(scalar, scratch);
			value_data[child_row] = StringVector::AddString(*value_vec, scratch.data(), scratch.size());
			break;
		}
		next_child++;
	}

	// Append a shred entry whose value is already rendered text (or SQL NULL).
	void AppendText(nonstd::string_view key, bool value_null, nonstd::string_view value) {
		if (next_child >= capacity) {
			EnsureListCapacity(list, next_child + 1);
			capacity = ListVector::GetListCapacity(list);
			key_data = FlatVector::GetData<string_t>(*key_vec);
			value_data = FlatVector::GetData<string_t>(*value_vec);
		}
		auto child_row = next_child;
		key_data[child_row] = StringVector::AddString(*key_vec, key.data(), key.size());
		if (value_null) {
			FlatVector::SetNull(*value_vec, child_row, true);
		} else {
			value_data[child_row] = StringVector::AddString(*value_vec, value.data(), value.size());
		}
		next_child++;
	}
};

// Walk sink that materializes one (path, value) struct per leaf, growing and
// shrinking the dotted/JSONPath segments of `path` via the walk's member/element
// tokens (the saved path length lives in the recursion frame).
struct EntriesWalkSink {
	EntriesWalkSink(std::string &path_p, JsonoEntriesKeyStyle style_p, EntriesSink &sink_p)
	    : path(path_p), style(style_p), sink(sink_p) {
	}

	std::string &path;
	JsonoEntriesKeyStyle style;
	EntriesSink &sink;
	// When set, an array-element path in it is a lifted scalar-array position — the residual carries
	// only a VAL_NULL placeholder there, the value lives in the shred — so the walk skips it (the
	// scalar-array shred overlay emits the real value). NULL when no scalar-array shred is present.
	const std::unordered_set<std::string> *suppress = nullptr;

	void BeginObject() {
	}
	void EndObject() {
	}
	void BeginArray() {
	}
	void EndArray() {
	}
	size_t BeginMember(nonstd::string_view key, size_t) {
		auto saved = path.size();
		AppendKeySegment(path, key, style);
		return saved;
	}
	void EndMember(size_t saved) {
		path.resize(saved);
	}
	size_t BeginElement(idx_t index) {
		auto saved = path.size();
		AppendIndexSegment(path, index, style);
		return saved;
	}
	void EndElement(size_t saved) {
		path.resize(saved);
	}
	void Scalar(const JsonoScalar &scalar) {
		if (suppress && suppress->count(path)) {
			return; // a lifted scalar-array placeholder; the shred overlay emits the real value
		}
		sink.Append(path, scalar);
	}
};

nonstd::string_view FormatShredValue(const UnifiedVectorFormat &fmt, idx_t idx, JsonoScalarPrimitive kind,
                                     std::string &scratch) {
	return JsonoPrimitiveVectorText(kind, fmt, idx, scratch);
}

// whole_json walk (plan 031 §4): recurse objects key-by-key, emit each scalar leaf as bare text, and
// emit each array node as ONE leaf whose value is the array's whole JSON text (the shared to_json
// render). It stops at the array boundary — unlike WalkJsonoValue / indexed_elements it never enters
// array elements. The input is always a plain JSONO value (a shredded value is reconstructed first), so
// there is no shred overlay. A top-level array (depth 0) has no object-key path and is rejected (§5).
void WalkJsonoWholeJson(const JsonoView &view, JsonoCursor &cursor, std::string &path, JsonoEntriesKeyStyle style,
                        EntriesSink &sink, std::string &scratch, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, cursor.pos);
		cursor.pos = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: expected KEY slot");
			}
			auto saved = path.size();
			AppendKeySegment(path, view.KeyAt(SlotPayload(key_slot)), style);
			WalkJsonoWholeJson(view, cursor, path, style, sink, scratch, depth + 1);
			path.resize(saved);
		}
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		cursor.pos++;
		return;
	}
	case tag::ARR_START:
		if (depth == 0) {
			throw InvalidInputException(
			    "jsono_entries: whole_json requires an object document; the value is a top-level array");
		}
		scratch.clear();
		AppendJsonValueText(view, cursor, scratch, depth); // advances cursor past the whole array
		sink.AppendText(nonstd::string_view(path.data(), path.size()), false, scratch);
		return;
	default:
		sink.Append(path, DecodeScalarAt(view, cursor));
		return;
	}
}

void JsonoEntriesExecuteWholeJson(DataChunk &args, const JsonoEntriesBindData &bind_data, Vector &result) {
	auto count = args.size();
	auto style = bind_data.style;

	// A shredded value is reconstructed to a plain document via the validated read path (the same path
	// to_json / ::varchar use), so the whole_json walk never reimplements shred-array reconstruction —
	// this satisfies the plan-029 array-shred correctness requirement (plan 031 §8). A plain value is
	// read directly (the capacity-1 holder is unused in that case).
	bool shredded = IsShreddedJsonoType(args.data[0].GetType());
	Vector reconstructed(JsonoType(), shredded ? count : 1);
	if (shredded) {
		JsonoReconstructToPlain(args.data[0], count, reconstructed);
	}
	Vector &source = shredded ? reconstructed : args.data[0];

	JsonoRowReader reader;
	reader.Init(source, count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	EntriesSink sink(result);
	// Zero-copy scalar text points into the source string_heap; keep its buffer alive on the value child.
	StringVector::AddHeapReference(*sink.value_vec, reader.StringHeapVector());

	std::string path;
	std::string scratch;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			SetListRowNull(result, row);
			continue;
		}
		auto start = sink.next_child;
		path.clear();
		if (style == JsonoEntriesKeyStyle::JsonPath) {
			path.push_back('$');
		}
		JsonoCursor cursor;
		WalkJsonoWholeJson(view, cursor, path, style, sink, scratch, 0);
		FinishListRow(result, row, start, sink.next_child - start);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoEntriesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = func_expr.bind_info->Cast<JsonoEntriesBindData>();
	if (bind_data.array_style == JsonoEntriesArrayStyle::WholeJson) {
		JsonoEntriesExecuteWholeJson(args, bind_data, result);
		return;
	}
	auto count = args.size();
	// The row reader verifies each row's shred manifest against the shreds this input type
	// carries, so a row narrowed by a raw struct cast fails loud instead of emitting
	// incomplete entries.
	JsonoRowReader reader;
	reader.Init(args.data[0], count);
	auto style = bind_data.style;
	auto &shreds = bind_data.shreds;

	// Read each shred column's value lanes once (a scalar value lane, or a list_entry lane plus the
	// element-struct subfield lanes for an array shred). InitShredLanes resolves the LIST<STRUCT> walk
	// and the shred kinds in one place; lane f corresponds to shreds[f].
	vector<ShredLane> lanes;
	if (!shreds.empty()) {
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(args.data[0].GetType(), layout);
		InitShredLanes(args.data[0], count, layout.shreds, lanes);
	}

	// A scalar array leaves a VAL_NULL placeholder per lifted element in the residual skeleton; the
	// generic walk would emit it as a `<base>[i]: null` leaf, duplicating the shred overlay's real
	// value. When any scalar-array shred is present, the lifted positions are collected per row and
	// the walk suppresses those placeholder leaves (the overlay below emits the real value).
	bool has_scalar_array = false;
	for (auto &lane : lanes) {
		has_scalar_array = has_scalar_array || lane.kind == ShredKind::ScalarArray;
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	std::string path;
	std::string shred_scratch;
	std::unordered_set<std::string> suppress;
	std::string suppress_scratch;
	EntriesSink sink(result);
	// Zero-copy text leaves point into the input string_heap; keep its buffer alive on the
	// value child (the reference survives the list child's Reserve/Resize grows).
	StringVector::AddHeapReference(*sink.value_vec, reader.StringHeapVector());

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		auto row_state = reader.Read(row, blob, view);
		if (row_state == JsonoRowState::Null) {
			SetListRowNull(result, row);
			continue;
		}
		auto start = sink.next_child;
		if (has_scalar_array) {
			// Collect this row's lifted scalar-array positions (`<base>[i]` for each valid element slot),
			// built with the same key/index path builders the walk uses, so the strings match exactly.
			suppress.clear();
			for (idx_t f = 0; f < shreds.size(); f++) {
				auto &lane = lanes[f];
				if (lane.kind != ShredKind::ScalarArray) {
					continue;
				}
				auto list_idx = lane.fmt.sel->get_index(row);
				if (!lane.fmt.validity.RowIsValid(list_idx)) {
					continue;
				}
				auto entry = UnifiedVectorFormat::GetData<list_entry_t>(lane.fmt)[list_idx];
				auto &element_fmt = lane.sub_fmt[0];
				auto &base_key =
				    style == JsonoEntriesKeyStyle::JsonPath ? shreds[f].key_jsonpath : shreds[f].key_dotted;
				for (idx_t e = 0; e < entry.length; e++) {
					if (!element_fmt.validity.RowIsValid(element_fmt.sel->get_index(entry.offset + e))) {
						continue;
					}
					suppress_scratch.assign(base_key.begin(), base_key.end());
					AppendIndexSegment(suppress_scratch, e, style);
					suppress.insert(suppress_scratch);
				}
			}
		}
		if (row_state == JsonoRowState::Value) {
			path.clear();
			if (style == JsonoEntriesKeyStyle::JsonPath) {
				path.push_back('$');
			}
			JsonoCursor cursor;
			EntriesWalkSink walk(path, style, sink);
			if (has_scalar_array) {
				walk.suppress = &suppress;
			}
			WalkJsonoValue(view, cursor, walk, 0);
		} else if (shreds.empty()) {
			SetListRowNull(result, row);
			continue;
		}
		for (idx_t f = 0; f < shreds.size(); f++) {
			auto &lane = lanes[f];
			auto idx = lane.fmt.sel->get_index(row);
			if (!lane.fmt.validity.RowIsValid(idx)) {
				continue; // a NULL scalar shred or a NULL array list emits nothing
			}
			auto &base_key = style == JsonoEntriesKeyStyle::JsonPath ? shreds[f].key_jsonpath : shreds[f].key_dotted;
			switch (lane.kind) {
			case ShredKind::Scalar: {
				auto value = FormatShredValue(lane.fmt, idx, lane.scalar_kind, shred_scratch);
				sink.AppendText(nonstd::string_view(base_key.data(), base_key.size()), false, value);
				break;
			}
			case ShredKind::Array: {
				// Expand each element's lifted subfields into `<base>[i].<subfield>` leaves. A NULL
				// element struct or absent subfield (a non-object element, a missing/explicit-null/
				// type-mismatched value) is skipped here — it stays in the residual skeleton, which the
				// walk above already emitted, so the union matches a plain parse.
				auto entry = UnifiedVectorFormat::GetData<list_entry_t>(lane.fmt)[idx];
				path.assign(base_key.begin(), base_key.end());
				for (idx_t e = 0; e < entry.length; e++) {
					auto saved_index = path.size();
					AppendIndexSegment(path, e, style);
					for (idx_t j = 0; j < lane.sub_fmt.size(); j++) {
						auto &sub_fmt = lane.sub_fmt[j];
						auto sub_idx = sub_fmt.sel->get_index(entry.offset + e);
						if (!sub_fmt.validity.RowIsValid(sub_idx)) {
							continue;
						}
						auto saved_key = path.size();
						auto &sub_name = shreds[f].subfield_names[j];
						AppendKeySegment(path, nonstd::string_view(sub_name.data(), sub_name.size()), style);
						auto value = FormatShredValue(sub_fmt, sub_idx, lane.sub_kind[j], shred_scratch);
						sink.AppendText(nonstd::string_view(path.data(), path.size()), false, value);
						path.resize(saved_key);
					}
					path.resize(saved_index);
				}
				break;
			}
			case ShredKind::ScalarArray: {
				// Expand each lifted element into a `<base>[i]` leaf (value = the element's text). A NULL
				// slot is a kept element (non-conforming / null / object / array) that stays in the residual
				// skeleton the walk already emitted, so the union matches a plain parse. The single element
				// value lane lives at sub_fmt[0]/sub_kind[0] (see InitShredLanes).
				auto entry = UnifiedVectorFormat::GetData<list_entry_t>(lane.fmt)[idx];
				auto &element_fmt = lane.sub_fmt[0];
				path.assign(base_key.begin(), base_key.end());
				for (idx_t e = 0; e < entry.length; e++) {
					auto element_idx = element_fmt.sel->get_index(entry.offset + e);
					if (!element_fmt.validity.RowIsValid(element_idx)) {
						continue;
					}
					auto saved_index = path.size();
					AppendIndexSegment(path, e, style);
					auto value = FormatShredValue(element_fmt, element_idx, lane.sub_kind[0], shred_scratch);
					sink.AppendText(nonstd::string_view(path.data(), path.size()), false, value);
					path.resize(saved_index);
				}
				break;
			}
			}
		}
		FinishListRow(result, row, start, sink.next_child - start);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoEntries(ExtensionLoader &loader) {
	auto entry_type =
	    LogicalType::LIST(LogicalType::STRUCT({{"key", LogicalType::VARCHAR}, {"value", LogicalType::VARCHAR}}));
	// The input is ANY so a shredded jsono struct (whose type varies by shred set) also
	// binds; JsonoEntriesBind validates it is JSONO or shredded and collects the shreds.
	ScalarFunctionSet set("jsono_entries");
	// SPECIAL_HANDLING so a NULL named-arg constant (key_style/array_style) reaches the bind validator
	// instead of folding the whole call to NULL (DEFAULT_NULL_HANDLING short-circuits any NULL constant
	// argument before bind). Execute already handles NULL input rows per-row, so this changes only the
	// bind-time fold, not the output of any valid call.
	for (auto &signature :
	     {vector<LogicalType> {LogicalType::ANY}, vector<LogicalType> {LogicalType::ANY, LogicalType::VARCHAR},
	      vector<LogicalType> {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR}}) {
		ScalarFunction f(signature, entry_type, JsonoEntriesExecute, JsonoEntriesBind);
		f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		set.AddFunction(f);
	}
	loader.RegisterFunction(set);
}

} // namespace duckdb
