#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"

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
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

void EnsureListCapacity(Vector &result, idx_t needed) {
	if (needed <= ListVector::GetListCapacity(result)) {
		return;
	}
	auto next = std::max<idx_t>(needed, std::max<idx_t>(ListVector::GetListCapacity(result) * 2, 1));
	ListVector::Reserve(result, next);
}

void FinishListRow(Vector &result, idx_t row, idx_t start_offset, idx_t length) {
	ListVector::GetData(result)[row] = {start_offset, length};
	ListVector::SetListSize(result, start_offset + length);
}

void SetListRowNull(Vector &result, idx_t row) {
	FlatVector::SetNull(result, row, true);
	ListVector::GetData(result)[row] = {0, 0};
}

enum class JsonoEntriesKeyStyle : uint8_t { JsonPath, Dotted };

struct JsonoEntriesBindData : public FunctionData {
	explicit JsonoEntriesBindData(JsonoEntriesKeyStyle style_p) : style(style_p) {
	}
	JsonoEntriesKeyStyle style;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoEntriesBindData>(style);
	}
	bool Equals(const FunctionData &other_p) const override {
		return style == other_p.Cast<JsonoEntriesBindData>().style;
	}
};

unique_ptr<FunctionData> JsonoEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	(void)bound_function;
	auto style = JsonoEntriesKeyStyle::JsonPath;
	if (arguments.size() >= 2) {
		auto &style_arg = arguments[1];
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
	return make_uniq<JsonoEntriesBindData>(style);
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
		path.append(std::to_string(index));
		path.push_back(']');
		return;
	}
	if (!path.empty()) {
		path.push_back('.');
	}
	path.append(std::to_string(index));
}

// Render a leaf scalar as bare VARCHAR (no JSON quoting). Returns true for JSON
// null, where the caller emits a SQL NULL value rather than the text "null".
bool JsonoScalarToText(const JsonoScalar &scalar, std::string &out) {
	switch (scalar.kind) {
	case JsonoScalarKind::Null:
		return true;
	case JsonoScalarKind::String:
	case JsonoScalarKind::NumberText:
		out.assign(scalar.text.data(), scalar.text.size());
		return false;
	case JsonoScalarKind::Int64:
		out = std::to_string(scalar.int_value);
		return false;
	case JsonoScalarKind::UInt64:
		out = std::to_string(scalar.uint_value);
		return false;
	case JsonoScalarKind::Double:
		EmitDouble(scalar.double_value, out);
		return false;
	case JsonoScalarKind::Dec60:
		AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
		return false;
	case JsonoScalarKind::Bool:
		out = scalar.bool_value ? "true" : "false";
		return false;
	}
	return true;
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
			// Already text in the heap — copy straight in, no intermediate std::string.
			value_data[child_row] = StringVector::AddString(*value_vec, scalar.text.data(), scalar.text.size());
			break;
		default:
			scratch.clear();
			JsonoScalarToText(scalar, scratch);
			value_data[child_row] = StringVector::AddString(*value_vec, scratch.data(), scratch.size());
			break;
		}
		next_child++;
	}
};

// Depth-first walk that materializes one (path, value) struct per leaf. Object
// values advance a value cursor while the shared string cursor consumes heap bytes
// in walk order.
void WalkEntries(const JsonoView &view, size_t &pos, size_t &string_cursor, std::string &path,
                 JsonoEntriesKeyStyle style, EntriesSink &sink, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, pos);
		auto val_cursor = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			auto key = view.KeyAt(SlotPayload(key_slot));
			auto saved = path.size();
			AppendKeySegment(path, key, style);
			WalkEntries(view, val_cursor, string_cursor, path, style, sink, depth + 1);
			path.resize(saved);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, pos);
		pos++;
		idx_t index = 0;
		while (pos < end_pos) {
			auto saved = path.size();
			AppendIndexSegment(path, index, style);
			WalkEntries(view, pos, string_cursor, path, style, sink, depth + 1);
			path.resize(saved);
			index++;
		}
		pos = end_pos + 1;
		return;
	}
	default: {
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		sink.Append(path, scalar);
		return;
	}
	}
}

void JsonoEntriesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto style = func_expr.bind_info->Cast<JsonoEntriesBindData>().style;

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	std::string path;
	EntriesSink sink(result);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetListRowNull(result, row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetListRowNull(result, row);
			continue;
		}
		auto start = sink.next_child;
		path.clear();
		if (style == JsonoEntriesKeyStyle::JsonPath) {
			path.push_back('$');
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		WalkEntries(view, pos, string_cursor, path, style, sink, 0);
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
	ScalarFunctionSet set("jsono_entries");
	set.AddFunction(ScalarFunction({jsono_type}, entry_type, JsonoEntriesExecute, JsonoEntriesBind));
	set.AddFunction(
	    ScalarFunction({jsono_type, LogicalType::VARCHAR}, entry_type, JsonoEntriesExecute, JsonoEntriesBind));
	loader.RegisterFunction(set);
}

} // namespace duckdb
