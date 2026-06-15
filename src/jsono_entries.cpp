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
	JsonoEntriesBindData(JsonoEntriesKeyStyle style_p, vector<EntriesShred> shreds_p)
	    : style(style_p), shreds(std::move(shreds_p)) {
	}
	JsonoEntriesKeyStyle style;
	vector<EntriesShred> shreds;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoEntriesBindData>(style, shreds);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoEntriesBindData>();
		if (style != other.style || shreds.size() != other.shreds.size()) {
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

unique_ptr<FunctionData> JsonoEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto style = JsonoEntriesKeyStyle::JsonPath;
	if (arguments.size() >= 2) {
		auto &style_arg = arguments[1];
		if (style_arg->GetAlias() != "key_style") {
			throw BinderException("jsono_entries: unknown argument '%s' (pass key_style := 'jsonpath' | 'dotted')",
			                      style_arg->GetAlias());
		}
		if (style_arg->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!style_arg->IsFoldable()) {
			throw BinderException("jsono_entries: key_style must be constant");
		}
		auto style_value = ExpressionExecutor::EvaluateScalar(context, *style_arg);
		if (style_value.IsNull()) {
			throw BinderException("jsono_entries: key_style must not be NULL");
		}
		auto name = StringValue::Get(style_value);
		StringUtil::Trim(name);
		name = StringUtil::Lower(name);
		if (name == "jsonpath") {
			style = JsonoEntriesKeyStyle::JsonPath;
		} else if (name == "dotted") {
			style = JsonoEntriesKeyStyle::Dotted;
		} else {
			throw BinderException("jsono_entries: key_style must be 'jsonpath' or 'dotted'");
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
			if (name.size() >= 2 && name[0] == '$' && name[1] == '.') {
				shred.key_jsonpath = name;
				shred.key_dotted = name.substr(2);
			} else {
				shred.key_jsonpath = "$." + name;
				shred.key_dotted = name;
			}
			if (ClassifyShredKind(shred.type) == ShredKind::Array) {
				for (auto &sub : StructType::GetChildTypes(ListType::GetChildType(shred.type))) {
					shred.subfield_names.push_back(sub.first);
				}
			}
			shreds.push_back(std::move(shred));
		}
	}
	return make_uniq<JsonoEntriesBindData>(style, std::move(shreds));
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
	// When set, every emitted leaf path is recorded so the VARCHAR-shred overlay can tell a
	// stripped value (residual lacks the path) from a read-copy (residual keeps it). NULL when no
	// VARCHAR shred is present, since only VARCHAR shreds store a read-copy of a non-string scalar.
	std::unordered_set<std::string> *seen = nullptr;

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
		if (seen) {
			seen->insert(path);
		}
		sink.Append(path, scalar);
	}
};

// Render a shredded shred's typed value as the text an entry carries. The switch is exhaustive over
// the closed scalar shred set (so a new scalar shred type fails to compile here); VARCHAR returns the
// heap string in place, the rest format into `scratch`.
nonstd::string_view FormatShredValue(const UnifiedVectorFormat &fmt, idx_t idx, ShredPrimitive kind,
                                     std::string &scratch) {
	switch (kind) {
	case ShredPrimitive::Varchar: {
		auto &s = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		return nonstd::string_view(s.GetData(), s.GetSize());
	}
	case ShredPrimitive::Bigint:
		scratch.clear();
		AppendSignedText(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx], scratch);
		return scratch;
	case ShredPrimitive::Ubigint:
		scratch.clear();
		AppendUnsignedText(UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx], scratch);
		return scratch;
	case ShredPrimitive::Double:
		scratch.clear();
		EmitDouble(UnifiedVectorFormat::GetData<double>(fmt)[idx], scratch);
		return scratch;
	case ShredPrimitive::Boolean:
		scratch = UnifiedVectorFormat::GetData<bool>(fmt)[idx] ? "true" : "false";
		return scratch;
	}
	return nonstd::string_view();
}

void JsonoEntriesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	// The row reader verifies each row's shred manifest against the shreds this input type
	// carries, so a row narrowed by a raw struct cast fails loud instead of emitting
	// incomplete entries.
	JsonoRowReader reader;
	reader.Init(args.data[0], count);
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = func_expr.bind_info->Cast<JsonoEntriesBindData>();
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

	// Only a VARCHAR shred keeps a read-copy of a non-string scalar (the value stays in the residual,
	// the shred holds the `->>` text for a fast typed read). Such a value would be emitted twice — once
	// by the residual walk, once by the shred overlay — so when any VARCHAR shred is present, the
	// residual leaf paths are recorded and the overlay skips a path the residual already carries
	// (residual is authoritative, matching the reconstruct overlay). ShredPrimitiveStoresReadCopy is
	// the single owner of which kinds read-copy.
	bool has_varchar_shred = false;
	for (auto &lane : lanes) {
		switch (lane.kind) {
		case ShredKind::Scalar:
			has_varchar_shred = has_varchar_shred || ShredPrimitiveStoresReadCopy(lane.scalar_kind);
			break;
		case ShredKind::Array:
			for (auto sub_kind : lane.sub_kind) {
				has_varchar_shred = has_varchar_shred || ShredPrimitiveStoresReadCopy(sub_kind);
			}
			break;
		}
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	std::string path;
	std::string shred_scratch;
	std::unordered_set<std::string> residual_paths;
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
		if (has_varchar_shred) {
			residual_paths.clear();
		}
		if (row_state == JsonoRowState::Value) {
			path.clear();
			if (style == JsonoEntriesKeyStyle::JsonPath) {
				path.push_back('$');
			}
			JsonoCursor cursor;
			EntriesWalkSink walk(path, style, sink);
			if (has_varchar_shred) {
				walk.seen = &residual_paths;
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
				// A VARCHAR read-copy the residual still carries is the residual's to emit, not ours.
				if (ShredPrimitiveStoresReadCopy(lane.scalar_kind) && residual_paths.count(base_key)) {
					break;
				}
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
						// A VARCHAR subfield read-copy the residual element still carries is the
						// residual's to emit (the walk above already did), so skip it here.
						if (ShredPrimitiveStoresReadCopy(lane.sub_kind[j]) && residual_paths.count(path)) {
							path.resize(saved_key);
							continue;
						}
						auto value = FormatShredValue(sub_fmt, sub_idx, lane.sub_kind[j], shred_scratch);
						sink.AppendText(nonstd::string_view(path.data(), path.size()), false, value);
						path.resize(saved_key);
					}
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
	auto jsono_type = JsonoType();
	auto entry_type =
	    LogicalType::LIST(LogicalType::STRUCT({{"key", LogicalType::VARCHAR}, {"value", LogicalType::VARCHAR}}));
	(void)jsono_type;
	// The input is ANY so a shredded jsono struct (whose type varies by shred set) also
	// binds; JsonoEntriesBind validates it is JSONO or shredded and collects the shreds.
	ScalarFunctionSet set("jsono_entries");
	set.AddFunction(ScalarFunction({LogicalType::ANY}, entry_type, JsonoEntriesExecute, JsonoEntriesBind));
	set.AddFunction(
	    ScalarFunction({LogicalType::ANY, LogicalType::VARCHAR}, entry_type, JsonoEntriesExecute, JsonoEntriesBind));
	loader.RegisterFunction(set);
}

} // namespace duckdb
