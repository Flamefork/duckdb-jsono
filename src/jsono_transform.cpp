#include "jsono_transform.hpp"
#include "jsono.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
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

#include <algorithm>
#include <cctype>
#include <cstring>

namespace duckdb {
namespace {

using namespace jsono;

enum class TransformPrimitive : uint8_t { Bigint, Ubigint, Double, Varchar, Boolean };
enum class TransformMode : uint8_t { Scalar, List, Join };

struct TransformField {
	string name;
	TransformPrimitive primitive;
	TransformMode mode = TransformMode::Scalar;
	vector<PathStep> path;
	string join_separator;
	LogicalType logical_type;
	// When the input is a shredded JSONO whose lane path equals this scalar field's path,
	// the field reads residual-first and falls back to this lane child (lane_primitive gives
	// the lane's lossless kind for synthesizing the stripped value). INVALID = not lane-backed.
	idx_t lane_child_index = DConstants::INVALID_INDEX;
	TransformPrimitive lane_primitive = TransformPrimitive::Varchar;
};

struct TrieEdge {
	TrieEdge() = default;
	TrieEdge(string key_p, idx_t child_p) : key(std::move(key_p)), child(child_p) {
	}

	string key;
	idx_t child = DConstants::INVALID_INDEX;
};

struct TrieIndexEdge {
	TrieIndexEdge() = default;
	TrieIndexEdge(idx_t index_p, idx_t child_p) : index(index_p), child(child_p) {
	}

	idx_t index = 0;
	idx_t child = DConstants::INVALID_INDEX;
};

struct TransformTrieNode {
	vector<TrieEdge> key_edges;
	vector<TrieIndexEdge> index_edges;
	idx_t wildcard_child = DConstants::INVALID_INDEX;
	vector<idx_t> scalar_leaves;
	vector<idx_t> list_leaves;
	vector<idx_t> join_leaves;
	idx_t simple_scalar_leaf = DConstants::INVALID_INDEX;
};

struct TransformBindData : public FunctionData {
	string spec;
	vector<TransformField> fields;
	vector<TransformTrieNode> trie;
	LogicalType return_type;
	// Scalar fields backed by a shredded lane (field.lane_child_index set); handled directly
	// from the lane/residual rather than the trie. Empty for a plain JSONO input.
	vector<idx_t> lane_fields;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TransformBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<TransformBindData>();
		return spec == other.spec;
	}
};

struct Location {
	Location() = default;
	Location(size_t pos_p, size_t string_cursor_p, bool found_p)
	    : pos(pos_p), string_cursor(string_cursor_p), found(found_p) {
	}

	size_t pos = 0;
	size_t string_cursor = 0;
	bool found = false;
};

struct TransformRankCacheEntry {
	bool valid = false;
	idx_t node_index = DConstants::INVALID_INDEX;
	size_t key_count = 0;
	idx_t key_edge_count = 0;
	vector<size_t> ranks;
	vector<uint8_t> found;
};

constexpr idx_t RANK_CACHE_SIZE = 4096;

struct TransformLocalState : public FunctionLocalState {
	TransformLocalState() : rank_cache(RANK_CACHE_SIZE) {
	}

	vector<TransformRankCacheEntry> rank_cache;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<TransformLocalState>();
	}
};

TransformPrimitive ParsePrimitiveType(const string &type) {
	auto upper = StringUtil::Upper(type);
	if (upper == "BIGINT") {
		return TransformPrimitive::Bigint;
	}
	if (upper == "UBIGINT") {
		return TransformPrimitive::Ubigint;
	}
	if (upper == "DOUBLE") {
		return TransformPrimitive::Double;
	}
	if (upper == "VARCHAR") {
		return TransformPrimitive::Varchar;
	}
	if (upper == "BOOLEAN") {
		return TransformPrimitive::Boolean;
	}
	throw BinderException("jsono_transform: unsupported scalar type '%s'", type);
}

// Map a shredded lane's column type to the transform primitive that names its lossless
// kind. A JSON-aliased VARCHAR is not a lane (it holds a document, not a scalar), so it
// is rejected — matching the shred writer, which never lifts JSON fields into lanes.
bool LaneTypeToPrimitive(const LogicalType &type, TransformPrimitive &out) {
	switch (type.id()) {
	case LogicalTypeId::VARCHAR:
		if (type.HasAlias() && type.GetAlias() == LogicalType::JSON_TYPE_NAME) {
			return false;
		}
		out = TransformPrimitive::Varchar;
		return true;
	case LogicalTypeId::BIGINT:
		out = TransformPrimitive::Bigint;
		return true;
	case LogicalTypeId::UBIGINT:
		out = TransformPrimitive::Ubigint;
		return true;
	case LogicalTypeId::DOUBLE:
		out = TransformPrimitive::Double;
		return true;
	case LogicalTypeId::BOOLEAN:
		out = TransformPrimitive::Boolean;
		return true;
	default:
		return false;
	}
}

LogicalType PrimitiveLogicalType(TransformPrimitive primitive) {
	switch (primitive) {
	case TransformPrimitive::Bigint:
		return LogicalType::BIGINT;
	case TransformPrimitive::Ubigint:
		return LogicalType::UBIGINT;
	case TransformPrimitive::Double:
		return LogicalType::DOUBLE;
	case TransformPrimitive::Varchar:
		return LogicalType::VARCHAR;
	case TransformPrimitive::Boolean:
		return LogicalType::BOOLEAN;
	}
	throw InternalException("jsono_transform: unhandled primitive type");
}

idx_t FindWildcardIndex(const vector<PathStep> &path) {
	for (idx_t i = 0; i < path.size(); i++) {
		if (path[i].kind == PathStepKind::Wildcard) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

TransformField MakeField(nonstd::string_view name, TransformPrimitive primitive, TransformMode mode,
                         vector<PathStep> path, string join_separator) {
	TransformField field;
	field.name = string(name);
	field.primitive = primitive;
	field.mode = mode;
	field.path = std::move(path);
	field.join_separator = std::move(join_separator);
	field.logical_type =
	    mode == TransformMode::List ? LogicalType::LIST(LogicalType::VARCHAR) : PrimitiveLogicalType(primitive);
	auto wildcard_index = FindWildcardIndex(field.path);
	if (wildcard_index != DConstants::INVALID_INDEX && mode == TransformMode::Scalar) {
		throw BinderException("jsono_transform: wildcard path requires list type or join_separator");
	}
	if (wildcard_index == DConstants::INVALID_INDEX && mode != TransformMode::Scalar) {
		throw BinderException("jsono_transform: list and join modes require a wildcard path");
	}
	return field;
}

// A bare shorthand/wrapper key names a literal top-level object key, not a JSONPath
// expression: dots in the key (e.g. analytics "utm.source") must not be read as nesting.
vector<PathStep> LiteralKeyPath(nonstd::string_view name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, string(name), 0});
	return path;
}

TransformField ParseScalarShorthand(nonstd::string_view name, nonstd::string_view type_name) {
	return MakeField(name, ParsePrimitiveType(string(type_name)), TransformMode::Scalar, LiteralKeyPath(name),
	                 string());
}

// A wrapper field is a nested STRUCT {type: ..., path: '...', join_separator: '...'}: `type`
// is a type string (scalar), or a single-element list `['VARCHAR']` (list mode); `path` and
// `join_separator` are strings. Mirrors the scalar-vs-struct dispatch the JSON form had.
TransformField ParseWrapperField(nonstd::string_view name, const Value &wrapper) {
	bool has_type = false;
	bool has_join_separator = false;
	TransformPrimitive primitive = TransformPrimitive::Varchar;
	TransformMode mode = TransformMode::Scalar;
	bool has_path = false;
	string path;
	string join_separator;

	auto &wrapper_types = StructType::GetChildTypes(wrapper.type());
	auto &wrapper_values = StructValue::GetChildren(wrapper);
	for (idx_t i = 0; i < wrapper_types.size(); i++) {
		auto &key = wrapper_types[i].first;
		auto value = wrapper_values[i];
		if (key == "type") {
			has_type = true;
			if (value.type().id() == LogicalTypeId::LIST) {
				auto &items = ListValue::GetChildren(value);
				auto item = items.size() == 1 ? items[0] : Value();
				if (items.size() != 1 || item.IsNull() || !item.DefaultTryCastAs(LogicalType::VARCHAR) ||
				    StringUtil::Upper(StringValue::Get(item)) != "VARCHAR") {
					throw BinderException("jsono_transform: unsupported list item type");
				}
				primitive = TransformPrimitive::Varchar;
				mode = TransformMode::List;
				continue;
			}
			if (value.IsNull() || !value.DefaultTryCastAs(LogicalType::VARCHAR)) {
				throw BinderException("jsono_transform: type must be a string or single-item array");
			}
			primitive = ParsePrimitiveType(StringValue::Get(value));
			continue;
		}
		if (key == "path") {
			if (value.IsNull() || !value.DefaultTryCastAs(LogicalType::VARCHAR)) {
				throw BinderException("jsono_transform: path must be a string");
			}
			has_path = true;
			path = StringValue::Get(value);
			continue;
		}
		if (key == "join_separator") {
			if (value.IsNull() || !value.DefaultTryCastAs(LogicalType::VARCHAR)) {
				throw BinderException("jsono_transform: join_separator must be a string");
			}
			has_join_separator = true;
			join_separator = StringValue::Get(value);
			continue;
		}
		throw BinderException("jsono_transform: unknown wrapper key '%s'", key);
	}
	if (!has_type) {
		throw BinderException("jsono_transform: wrapper field must include type");
	}
	if (has_join_separator) {
		if (mode == TransformMode::List || primitive != TransformPrimitive::Varchar) {
			throw BinderException("jsono_transform: join_separator requires VARCHAR scalar type");
		}
		mode = TransformMode::Join;
	}
	auto steps = has_path ? ParseJsonoPath(path, "jsono_transform") : LiteralKeyPath(name);
	return MakeField(name, primitive, mode, std::move(steps), std::move(join_separator));
}

idx_t GetOrAddKeyChild(TransformBindData &bind_data, idx_t node_index, const string &key) {
	auto &edges = bind_data.trie[node_index].key_edges;
	for (auto &edge : edges) {
		if (edge.key == key) {
			return edge.child;
		}
	}
	auto child = bind_data.trie.size();
	bind_data.trie.emplace_back();
	bind_data.trie[node_index].key_edges.push_back(TrieEdge {key, child});
	return child;
}

idx_t GetOrAddIndexChild(TransformBindData &bind_data, idx_t node_index, idx_t index) {
	auto &edges = bind_data.trie[node_index].index_edges;
	for (auto &edge : edges) {
		if (edge.index == index) {
			return edge.child;
		}
	}
	auto child = bind_data.trie.size();
	bind_data.trie.emplace_back();
	bind_data.trie[node_index].index_edges.push_back(TrieIndexEdge {index, child});
	return child;
}

idx_t GetOrAddWildcardChild(TransformBindData &bind_data, idx_t node_index) {
	auto &node = bind_data.trie[node_index];
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		return node.wildcard_child;
	}
	auto child = bind_data.trie.size();
	bind_data.trie.emplace_back();
	bind_data.trie[node_index].wildcard_child = child;
	return child;
}

void InsertFieldIntoTrie(TransformBindData &bind_data, idx_t field_index) {
	idx_t node_index = 0;
	auto &field = bind_data.fields[field_index];
	if (field.lane_child_index != DConstants::INVALID_INDEX) {
		// Lane-backed scalar: read directly from the lane/residual, never via the trie.
		return;
	}
	for (auto &step : field.path) {
		switch (step.kind) {
		case PathStepKind::Key:
			node_index = GetOrAddKeyChild(bind_data, node_index, step.key);
			break;
		case PathStepKind::Index:
			node_index = GetOrAddIndexChild(bind_data, node_index, step.index);
			break;
		case PathStepKind::Wildcard:
			node_index = GetOrAddWildcardChild(bind_data, node_index);
			break;
		}
	}
	auto &node = bind_data.trie[node_index];
	switch (field.mode) {
	case TransformMode::Scalar:
		node.scalar_leaves.push_back(field_index);
		break;
	case TransformMode::List:
		node.list_leaves.push_back(field_index);
		break;
	case TransformMode::Join:
		node.join_leaves.push_back(field_index);
		break;
	}
}

void BuildTransformTrie(TransformBindData &bind_data) {
	bind_data.trie.clear();
	bind_data.trie.emplace_back();
	for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
		InsertFieldIntoTrie(bind_data, field_index);
	}
	for (auto &node : bind_data.trie) {
		std::sort(node.key_edges.begin(), node.key_edges.end(),
		          [](const TrieEdge &left, const TrieEdge &right) { return left.key < right.key; });
		std::sort(node.index_edges.begin(), node.index_edges.end(),
		          [](const TrieIndexEdge &left, const TrieIndexEdge &right) { return left.index < right.index; });
		if (node.scalar_leaves.size() == 1 && node.list_leaves.empty() && node.join_leaves.empty() &&
		    node.key_edges.empty() && node.index_edges.empty() && node.wildcard_child == DConstants::INVALID_INDEX) {
			node.simple_scalar_leaf = node.scalar_leaves[0];
		}
	}
}

// The spec is a constant STRUCT mapping each output field to its type: a type string
// (scalar shorthand) or a nested wrapper STRUCT {type, path, join_separator}. Field names are
// unique by struct construction. Same shape as core json_transform's structure, but expressed
// as a typed STRUCT literal rather than a JSON string.
unique_ptr<TransformBindData> ParseTransformSpec(const Value &spec) {
	if (spec.IsNull() || spec.type().id() != LogicalTypeId::STRUCT) {
		throw BinderException(
		    "jsono_transform: spec must be a non-NULL STRUCT mapping output field to type, e.g. {x: 'VARCHAR'}");
	}
	auto &child_types = StructType::GetChildTypes(spec.type());
	auto &child_values = StructValue::GetChildren(spec);

	auto bind_data = make_uniq<TransformBindData>();
	child_list_t<LogicalType> children;
	for (idx_t i = 0; i < child_types.size(); i++) {
		auto &name = child_types[i].first;
		auto value = child_values[i];
		TransformField transform_field;
		if (value.type().id() == LogicalTypeId::STRUCT) {
			transform_field = ParseWrapperField(name, value);
		} else if (!value.IsNull() && value.DefaultTryCastAs(LogicalType::VARCHAR)) {
			transform_field = ParseScalarShorthand(name, StringValue::Get(value));
		} else {
			throw BinderException("jsono_transform: field spec for '%s' must be a type string or wrapper struct", name);
		}
		children.emplace_back(transform_field.name, transform_field.logical_type);
		bind_data->fields.push_back(std::move(transform_field));
	}
	if (bind_data->fields.empty()) {
		throw BinderException("jsono_transform: empty spec");
	}
	bind_data->return_type = LogicalType::STRUCT(std::move(children));
	BuildTransformTrie(*bind_data);
	return bind_data;
}

bool PathStepsEqual(const vector<PathStep> &left, const vector<PathStep> &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (idx_t i = 0; i < left.size(); i++) {
		if (left[i].kind != right[i].kind) {
			return false;
		}
		if (left[i].kind == PathStepKind::Key && left[i].key != right[i].key) {
			return false;
		}
		if (left[i].kind == PathStepKind::Index && left[i].index != right[i].index) {
			return false;
		}
	}
	return true;
}

// Bind a shredded input's lane columns to the scalar fields they back: a scalar field whose
// path equals a lane's path reads that lane (residual-first, lane-fallback) instead of
// navigating the residual tape. The trie is rebuilt to exclude the bound fields.
void AssignShreddedLanes(TransformBindData &bind_data, const LogicalType &input_type) {
	auto &children = StructType::GetChildTypes(input_type);
	auto prefix = StructType::GetChildTypes(JsonoRawStructType()).size();
	for (idx_t i = prefix; i < children.size(); i++) {
		TransformPrimitive lane_primitive;
		if (!LaneTypeToPrimitive(children[i].second, lane_primitive)) {
			// A non-lane-typed trailing child is not a lane; any field on its path reads the
			// residual, where the value remains.
			continue;
		}
		auto &name = children[i].first;
		auto steps = !name.empty() && name[0] == '$' ? ParseJsonoPath(name, "jsono_transform") : LiteralKeyPath(name);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			if (field.mode != TransformMode::Scalar || field.lane_child_index != DConstants::INVALID_INDEX) {
				continue;
			}
			if (PathStepsEqual(field.path, steps)) {
				field.lane_child_index = i;
				field.lane_primitive = lane_primitive;
				bind_data.lane_fields.push_back(field_index);
			}
		}
	}
	if (!bind_data.lane_fields.empty()) {
		BuildTransformTrie(bind_data);
	}
}

unique_ptr<FunctionData> JsonoTransformBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("jsono_transform: spec must be constant");
	}
	auto spec_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	auto bind_data = ParseTransformSpec(spec_value);
	// The struct's canonical string identifies the full spec (names, types, paths, joins) for
	// expression caching — return_type alone is insufficient since a field's path may differ
	// from its name.
	bind_data->spec = spec_value.ToString();
	auto &input_type = arguments[0]->return_type;
	if (IsShreddedJsonoType(input_type)) {
		AssignShreddedLanes(*bind_data, input_type);
	} else if (input_type.id() != LogicalTypeId::SQLNULL && !IsJsonoType(input_type)) {
		throw BinderException("jsono_transform: argument must be JSONO");
	}
	bound_function.return_type = bind_data->return_type;
	return std::move(bind_data);
}

void WriteScalarValue(const TransformField &field, const JsonoScalar &scalar, Vector &result, idx_t row) {
	switch (field.primitive) {
	case TransformPrimitive::Bigint:
		// UInt64 values exceed INT64_MAX by construction, so only Int64 fits a BIGINT target.
		if (scalar.kind == JsonoScalarKind::Int64) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<int64_t>(result)[row] = scalar.int_value;
			return;
		}
		break;
	case TransformPrimitive::Ubigint:
		if (scalar.kind == JsonoScalarKind::Int64 && scalar.int_value >= 0) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<uint64_t>(result)[row] = uint64_t(scalar.int_value);
			return;
		}
		if (scalar.kind == JsonoScalarKind::UInt64) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<uint64_t>(result)[row] = scalar.uint_value;
			return;
		}
		break;
	case TransformPrimitive::Double: {
		double value = 0;
		bool ok = true;
		switch (scalar.kind) {
		case JsonoScalarKind::Double:
			value = scalar.double_value;
			break;
		case JsonoScalarKind::Dec60:
			value = Dec60ToDouble(scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			break;
		case JsonoScalarKind::Int64:
			value = double(scalar.int_value);
			break;
		case JsonoScalarKind::UInt64:
			value = double(scalar.uint_value);
			break;
		case JsonoScalarKind::NumberText:
			// Locale-independent parse via DuckDB's fast_float-backed cast (no allocation).
			ok =
			    TryCast::Operation<string_t, double>(string_t(scalar.text.data(), uint32_t(scalar.text.size())), value);
			break;
		default:
			ok = false;
			break;
		}
		if (ok) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<double>(result)[row] = value;
			return;
		}
		break;
	}
	case TransformPrimitive::Varchar:
		if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<string_t>(result)[row] =
			    StringVector::AddString(result, scalar.text.data(), scalar.text.size());
			return;
		}
		break;
	case TransformPrimitive::Boolean:
		if (scalar.kind == JsonoScalarKind::Bool) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<bool>(result)[row] = scalar.bool_value;
			return;
		}
		break;
	}
	FlatVector::SetNull(result, row, true);
}

void WriteScalarField(const TransformField &field, const JsonoView &view, const Location &location, Vector &result,
                      idx_t row) {
	if (!location.found) {
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto value_tag = SlotTag(view.SlotAt(location.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		// A container where a scalar was requested is a type mismatch, not malformed input.
		FlatVector::SetNull(result, row, true);
		return;
	}
	size_t pos = location.pos;
	size_t string_cursor = location.string_cursor;
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	WriteScalarValue(field, scalar, result, row);
}

void EnsureListCapacity(Vector &result, idx_t needed) {
	if (needed <= ListVector::GetListCapacity(result)) {
		return;
	}
	ListVector::Reserve(result, std::max<idx_t>(needed, std::max<idx_t>(ListVector::GetListCapacity(result) * 2, 1)));
}

void AppendListStringValue(Vector &result, idx_t child_row, nonstd::string_view value) {
	auto &child = ListVector::GetEntry(result);
	FlatVector::Validity(child).SetValid(child_row);
	FlatVector::GetData<string_t>(child)[child_row] = StringVector::AddString(child, value.data(), value.size());
}

void AppendListNullValue(Vector &result, idx_t child_row) {
	auto &child = ListVector::GetEntry(result);
	FlatVector::SetNull(child, child_row, true);
}

void ApplyTrieNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                   const JsonoView &view, size_t pos, size_t string_cursor, vector<unique_ptr<Vector>> &children,
                   idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts);

void ApplyWildcardElement(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                          const JsonoView &view, size_t pos, size_t string_cursor, vector<unique_ptr<Vector>> &children,
                          idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts);

int CompareTrieKeyToJsonKey(const string &trie_key, nonstd::string_view json_key) {
	if (!trie_key.empty() && !json_key.empty()) {
		auto trie_first = static_cast<unsigned char>(trie_key[0]);
		auto json_first = static_cast<unsigned char>(json_key[0]);
		if (trie_first != json_first) {
			return trie_first < json_first ? -1 : 1;
		}
	}
	return nonstd::string_view(trie_key).compare(json_key);
}

bool FindObjectKeyRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t &rank) {
	size_t lo = 0;
	size_t hi = layout.key_count;
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto key_slot = view.SlotAt(layout.key_start + mid);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		if (view.KeyAt(SlotPayload(key_slot)) < key) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	rank = lo;
	if (lo >= layout.key_count) {
		return false;
	}
	auto key_slot = view.SlotAt(layout.key_start + lo);
	if (SlotTag(key_slot) != tag::KEY) {
		throw InvalidInputException("malformed JSONO: expected KEY slot");
	}
	if (view.KeyAt(SlotPayload(key_slot)) != key) {
		return false;
	}
	return true;
}

// Walk Key/Index steps to a leaf in the residual tape. Used for lane-backed scalar fields:
// the residual is authoritative, so a found leaf wins; only when the residual lacks the path
// does the caller fall back to the lane. Mirrors the shred writer's path locator.
Location LocatePath(const JsonoView &view, const vector<PathStep> &steps) {
	size_t pos = 0;
	size_t string_cursor = 0;
	for (auto &step : steps) {
		if (pos >= view.Slots()) {
			return {};
		}
		auto slot_tag = SlotTag(view.SlotAt(pos));
		if (step.kind == PathStepKind::Key) {
			if (slot_tag != tag::OBJ_START) {
				return {};
			}
			auto layout = ReadObjectLayout(view, pos);
			size_t rank = 0;
			if (!FindObjectKeyRank(view, layout, step.key, rank)) {
				return {};
			}
			size_t value_rank = 0;
			size_t value_pos = layout.value_start;
			size_t value_string_cursor = string_cursor;
			MoveObjectValueCursorToRank(view, layout, string_cursor, rank, value_rank, value_pos, value_string_cursor);
			pos = value_pos;
			string_cursor = value_string_cursor;
		} else if (step.kind == PathStepKind::Index) {
			if (slot_tag != tag::ARR_START) {
				return {};
			}
			auto end_pos = ReadArrayEndPos(view, pos);
			size_t element_pos = pos + 1;
			idx_t current = 0;
			while (element_pos < end_pos && current < step.index) {
				SkipValueFast(view, element_pos, string_cursor);
				current++;
			}
			if (element_pos >= end_pos || current < step.index) {
				return {};
			}
			pos = element_pos;
		} else {
			return {};
		}
	}
	return Location {pos, string_cursor, true};
}

// Build the lossless scalar a lane stands for. The lane is read only when the residual lacks
// the path, which (by the shred contract) means the value was stripped, so its original kind
// is the lane's lossless kind: a VARCHAR lane held a JSON string, a typed lane its exact kind.
JsonoScalar SynthLaneScalar(TransformPrimitive lane_primitive, const UnifiedVectorFormat &fmt, idx_t idx) {
	JsonoScalar scalar;
	switch (lane_primitive) {
	case TransformPrimitive::Varchar: {
		// Reference the string_t in vector storage, not a stack copy: an inline (short) string_t
		// stores its bytes inside itself, so a copy would leave scalar.text dangling after return.
		auto &value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		scalar.kind = JsonoScalarKind::String;
		scalar.text = nonstd::string_view(value.GetData(), value.GetSize());
		return scalar;
	}
	case TransformPrimitive::Bigint:
		scalar.kind = JsonoScalarKind::Int64;
		scalar.int_value = UnifiedVectorFormat::GetData<int64_t>(fmt)[idx];
		return scalar;
	case TransformPrimitive::Ubigint:
		scalar.kind = JsonoScalarKind::UInt64;
		scalar.uint_value = UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx];
		return scalar;
	case TransformPrimitive::Double:
		scalar.kind = JsonoScalarKind::Double;
		scalar.double_value = UnifiedVectorFormat::GetData<double>(fmt)[idx];
		return scalar;
	case TransformPrimitive::Boolean:
		scalar.kind = JsonoScalarKind::Bool;
		scalar.bool_value = UnifiedVectorFormat::GetData<bool>(fmt)[idx];
		return scalar;
	}
	return scalar;
}

TransformRankCacheEntry &GetRankCacheEntry(vector<TransformRankCacheEntry> &rank_cache, idx_t node_index,
                                           size_t key_count) {
	auto cache_index = idx_t(((uint64_t(node_index) * 0x9E3779B97F4A7C15ULL) ^ key_count) & (RANK_CACHE_SIZE - 1));
	return rank_cache[cache_index];
}

nonstd::string_view ObjectKeyAtRank(const JsonoView &view, const ObjectLayout &layout, size_t rank) {
	if (rank >= layout.key_count) {
		throw InternalException("JSONO object key rank out of bounds");
	}
	auto key_slot = view.SlotAt(layout.key_start + rank);
	if (SlotTag(key_slot) != tag::KEY) {
		throw InvalidInputException("malformed JSONO: expected KEY slot");
	}
	return view.KeyAt(SlotPayload(key_slot));
}

bool ValidateCachedObjectRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t rank,
                              bool found) {
	if (rank > layout.key_count) {
		return false;
	}
	if (found) {
		return rank < layout.key_count && ObjectKeyAtRank(view, layout, rank) == key;
	}
	if (rank > 0 && ObjectKeyAtRank(view, layout, rank - 1) >= key) {
		return false;
	}
	if (rank < layout.key_count && ObjectKeyAtRank(view, layout, rank) <= key) {
		return false;
	}
	return true;
}

TransformRankCacheEntry &GetObjectRankCacheEntry(TransformLocalState &lstate, const TransformBindData &bind_data,
                                                 idx_t node_index, const JsonoView &view, const ObjectLayout &layout) {
	auto &node = bind_data.trie[node_index];
	auto &entry = GetRankCacheEntry(lstate.rank_cache, node_index, layout.key_count);
	auto cache_valid = entry.valid && entry.node_index == node_index && entry.key_count == layout.key_count &&
	                   entry.key_edge_count == node.key_edges.size();
	if (cache_valid) {
		for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
			if (!ValidateCachedObjectRank(view, layout, node.key_edges[edge_index].key, entry.ranks[edge_index],
			                              entry.found[edge_index])) {
				cache_valid = false;
				break;
			}
		}
	}
	if (cache_valid) {
		return entry;
	}

	entry.valid = true;
	entry.node_index = node_index;
	entry.key_count = layout.key_count;
	entry.key_edge_count = node.key_edges.size();
	entry.ranks.resize(node.key_edges.size());
	entry.found.resize(node.key_edges.size());
	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		size_t rank = 0;
		entry.found[edge_index] = FindObjectKeyRank(view, layout, node.key_edges[edge_index].key, rank);
		entry.ranks[edge_index] = rank;
	}
	return entry;
}

void SetTrieNodeOutputsNull(const TransformBindData &bind_data, idx_t node_index, vector<unique_ptr<Vector>> &children,
                            idx_t row) {
	auto &node = bind_data.trie[node_index];
	for (auto field_index : node.scalar_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
	for (auto field_index : node.list_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
		ListVector::GetData(*children[field_index])[row] = {0, 0};
	}
	for (auto field_index : node.join_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
	for (auto &edge : node.key_edges) {
		SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
	}
	for (auto &edge : node.index_edges) {
		SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
	}
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		SetTrieNodeOutputsNull(bind_data, node.wildcard_child, children, row);
	}
}

void AppendListElementNull(Vector &result, idx_t row) {
	auto &entry = ListVector::GetData(result)[row];
	auto child_row = entry.offset + entry.length;
	EnsureListCapacity(result, child_row + 1);
	AppendListNullValue(result, child_row);
	entry.length++;
	ListVector::SetListSize(result, entry.offset + entry.length);
}

void AppendListElementString(Vector &result, idx_t row, nonstd::string_view value) {
	auto &entry = ListVector::GetData(result)[row];
	auto child_row = entry.offset + entry.length;
	EnsureListCapacity(result, child_row + 1);
	AppendListStringValue(result, child_row, value);
	entry.length++;
	ListVector::SetListSize(result, entry.offset + entry.length);
}

void AppendWildcardMissing(const TransformBindData &bind_data, idx_t node_index, vector<unique_ptr<Vector>> &children,
                           idx_t row) {
	auto &node = bind_data.trie[node_index];
	for (auto field_index : node.list_leaves) {
		AppendListElementNull(*children[field_index], row);
	}
	for (auto &edge : node.key_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		AppendWildcardMissing(bind_data, node.wildcard_child, children, row);
	}
}

void ApplyObjectNodeWithCheckpoints(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                                    const JsonoView &view, const ObjectLayout &layout, size_t string_cursor,
                                    vector<unique_ptr<Vector>> &children, idx_t row, vector<string> &join_buffers,
                                    vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	size_t value_rank = 0;
	size_t value_pos = layout.value_start;
	size_t value_string_cursor = string_cursor;

	// Copy the cached ranks out of the cache slot: the recursive ApplyTrieNode below can
	// re-enter GetObjectRankCacheEntry for a descendant object that hashes to the same slot
	// and resize its vectors, which would dangle a held reference into entry.found/entry.ranks.
	auto &entry = GetObjectRankCacheEntry(lstate, bind_data, node_index, view, layout);
	vector<uint8_t> found = entry.found;
	vector<size_t> ranks = entry.ranks;

	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		auto &edge = node.key_edges[edge_index];
		auto simple_scalar_leaf = bind_data.trie[edge.child].simple_scalar_leaf;
		if (!found[edge_index]) {
			if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
				SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
			} else {
				FlatVector::SetNull(*children[simple_scalar_leaf], row, true);
			}
			continue;
		}
		auto rank = ranks[edge_index];
		MoveObjectValueCursorToRank(view, layout, string_cursor, rank, value_rank, value_pos, value_string_cursor);
		if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
			ApplyTrieNode(lstate, bind_data, edge.child, view, value_pos, value_string_cursor, children, row,
			              join_buffers, join_counts);
		} else {
			WriteScalarField(bind_data.fields[simple_scalar_leaf], view,
			                 Location {value_pos, value_string_cursor, true}, *children[simple_scalar_leaf], row);
		}
	}
}

void ApplyWildcardObjectNodeWithCheckpoints(TransformLocalState &lstate, const TransformBindData &bind_data,
                                            idx_t node_index, const JsonoView &view, const ObjectLayout &layout,
                                            size_t string_cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                                            vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	size_t value_rank = 0;
	size_t value_pos = layout.value_start;
	size_t value_string_cursor = string_cursor;

	// Copy the cached ranks out of the cache slot before recursing (see ApplyObjectNodeWithCheckpoints).
	auto &entry = GetObjectRankCacheEntry(lstate, bind_data, node_index, view, layout);
	vector<uint8_t> found = entry.found;
	vector<size_t> ranks = entry.ranks;

	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		auto &edge = node.key_edges[edge_index];
		if (!found[edge_index]) {
			AppendWildcardMissing(bind_data, edge.child, children, row);
			continue;
		}
		auto rank = ranks[edge_index];
		MoveObjectValueCursorToRank(view, layout, string_cursor, rank, value_rank, value_pos, value_string_cursor);
		ApplyWildcardElement(lstate, bind_data, edge.child, view, value_pos, value_string_cursor, children, row,
		                     join_buffers, join_counts);
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
}

void MaterializeWildcardLeaves(const TransformBindData &bind_data, idx_t node_index, const JsonoView &view, size_t pos,
                               size_t string_cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                               vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	if (slot_tag == tag::VAL_STR_HEAP) {
		auto len = size_t(StringLen(SlotPayload(slot)));
		auto value = view.StringAt(string_cursor, len);
		for (auto field_index : node.list_leaves) {
			AppendListElementString(*children[field_index], row, value);
		}
		for (auto field_index : node.join_leaves) {
			auto &buffer = join_buffers[field_index];
			if (join_counts[field_index] > 0) {
				buffer.append(bind_data.fields[field_index].join_separator);
			}
			buffer.append(value.data(), value.size());
			join_counts[field_index]++;
		}
		return;
	}
	for (auto field_index : node.list_leaves) {
		AppendListElementNull(*children[field_index], row);
	}
}

void ApplyObjectNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                     const JsonoView &view, size_t pos, size_t string_cursor, vector<unique_ptr<Vector>> &children,
                     idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto layout = ReadObjectLayout(view, pos);
	size_t value_pos = layout.value_start;
	size_t value_string_cursor = string_cursor;
	idx_t edge_index = 0;
	bool use_checkpoints =
	    layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && layout.key_count > OBJECT_CHECKPOINT_STRIDE;
	if (use_checkpoints) {
		ApplyObjectNodeWithCheckpoints(lstate, bind_data, node_index, view, layout, string_cursor, children, row,
		                               join_buffers, join_counts);
		return;
	}

	for (size_t key_index = 0; key_index < layout.key_count; key_index++) {
		auto key_slot = view.SlotAt(layout.key_start + key_index);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		int key_compare = 1;
		while (edge_index < node.key_edges.size()) {
			key_compare = CompareTrieKeyToJsonKey(node.key_edges[edge_index].key, key);
			if (key_compare >= 0) {
				break;
			}
			auto simple_scalar_leaf = bind_data.trie[node.key_edges[edge_index].child].simple_scalar_leaf;
			if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
				SetTrieNodeOutputsNull(bind_data, node.key_edges[edge_index].child, children, row);
			} else {
				FlatVector::SetNull(*children[simple_scalar_leaf], row, true);
			}
			edge_index++;
		}
		if (edge_index < node.key_edges.size() && key_compare == 0) {
			auto simple_scalar_leaf = bind_data.trie[node.key_edges[edge_index].child].simple_scalar_leaf;
			if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
				ApplyTrieNode(lstate, bind_data, node.key_edges[edge_index].child, view, value_pos, value_string_cursor,
				              children, row, join_buffers, join_counts);
			} else {
				WriteScalarField(bind_data.fields[simple_scalar_leaf], view,
				                 Location {value_pos, value_string_cursor, true}, *children[simple_scalar_leaf], row);
			}
			edge_index++;
		}
		if (edge_index >= node.key_edges.size()) {
			return;
		}
		SkipValueFast(view, value_pos, value_string_cursor);
	}
	while (edge_index < node.key_edges.size()) {
		auto simple_scalar_leaf = bind_data.trie[node.key_edges[edge_index].child].simple_scalar_leaf;
		if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
			SetTrieNodeOutputsNull(bind_data, node.key_edges[edge_index].child, children, row);
		} else {
			FlatVector::SetNull(*children[simple_scalar_leaf], row, true);
		}
		edge_index++;
	}
}

void ApplyArrayNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                    const JsonoView &view, size_t pos, size_t string_cursor, vector<unique_ptr<Vector>> &children,
                    idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto end_pos = ReadArrayEndPos(view, pos);
	size_t value_pos = pos + 1;
	size_t value_string_cursor = string_cursor;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;

	while (value_pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			SetTrieNodeOutputsNull(bind_data, node.index_edges[fixed_edge_index].child, children, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			ApplyTrieNode(lstate, bind_data, node.index_edges[fixed_edge_index].child, view, value_pos,
			              value_string_cursor, children, row, join_buffers, join_counts);
			fixed_edge_index++;
		}
		if (fixed_edge_index >= node.index_edges.size() && node.wildcard_child == DConstants::INVALID_INDEX) {
			return;
		}
		if (node.wildcard_child != DConstants::INVALID_INDEX) {
			ApplyWildcardElement(lstate, bind_data, node.wildcard_child, view, value_pos, value_string_cursor, children,
			                     row, join_buffers, join_counts);
		}
		SkipValueFast(view, value_pos, value_string_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		SetTrieNodeOutputsNull(bind_data, node.index_edges[fixed_edge_index].child, children, row);
		fixed_edge_index++;
	}
}

void ApplyWildcardObjectNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                             const JsonoView &view, size_t pos, size_t string_cursor,
                             vector<unique_ptr<Vector>> &children, idx_t row, vector<string> &join_buffers,
                             vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto layout = ReadObjectLayout(view, pos);
	size_t value_pos = layout.value_start;
	size_t value_string_cursor = string_cursor;
	idx_t edge_index = 0;
	bool use_checkpoints =
	    layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && layout.key_count > OBJECT_CHECKPOINT_STRIDE;
	if (use_checkpoints) {
		ApplyWildcardObjectNodeWithCheckpoints(lstate, bind_data, node_index, view, layout, string_cursor, children,
		                                       row, join_buffers, join_counts);
		return;
	}

	for (size_t key_index = 0; key_index < layout.key_count; key_index++) {
		auto key_slot = view.SlotAt(layout.key_start + key_index);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		int key_compare = 1;
		while (edge_index < node.key_edges.size()) {
			key_compare = CompareTrieKeyToJsonKey(node.key_edges[edge_index].key, key);
			if (key_compare >= 0) {
				break;
			}
			AppendWildcardMissing(bind_data, node.key_edges[edge_index].child, children, row);
			edge_index++;
		}
		if (edge_index < node.key_edges.size() && key_compare == 0) {
			ApplyWildcardElement(lstate, bind_data, node.key_edges[edge_index].child, view, value_pos,
			                     value_string_cursor, children, row, join_buffers, join_counts);
			edge_index++;
		}
		if (edge_index >= node.key_edges.size() && node.index_edges.empty()) {
			return;
		}
		SkipValueFast(view, value_pos, value_string_cursor);
	}
	while (edge_index < node.key_edges.size()) {
		AppendWildcardMissing(bind_data, node.key_edges[edge_index].child, children, row);
		edge_index++;
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
}

void ApplyWildcardArrayNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                            const JsonoView &view, size_t pos, size_t string_cursor,
                            vector<unique_ptr<Vector>> &children, idx_t row, vector<string> &join_buffers,
                            vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	for (auto &edge : node.key_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}

	auto end_pos = ReadArrayEndPos(view, pos);
	size_t value_pos = pos + 1;
	size_t value_string_cursor = string_cursor;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;
	while (value_pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			AppendWildcardMissing(bind_data, node.index_edges[fixed_edge_index].child, children, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			ApplyWildcardElement(lstate, bind_data, node.index_edges[fixed_edge_index].child, view, value_pos,
			                     value_string_cursor, children, row, join_buffers, join_counts);
			fixed_edge_index++;
		}
		if (fixed_edge_index >= node.index_edges.size() && node.wildcard_child == DConstants::INVALID_INDEX) {
			return;
		}
		SkipValueFast(view, value_pos, value_string_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		AppendWildcardMissing(bind_data, node.index_edges[fixed_edge_index].child, children, row);
		fixed_edge_index++;
	}
}

void ApplyWildcardElement(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                          const JsonoView &view, size_t pos, size_t string_cursor, vector<unique_ptr<Vector>> &children,
                          idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	MaterializeWildcardLeaves(bind_data, node_index, view, pos, string_cursor, children, row, join_buffers,
	                          join_counts);
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		ApplyWildcardObjectNode(lstate, bind_data, node_index, view, pos, string_cursor, children, row, join_buffers,
		                        join_counts);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		ApplyWildcardArrayNode(lstate, bind_data, node_index, view, pos, string_cursor, children, row, join_buffers,
		                       join_counts);
		return;
	}
	auto &node = bind_data.trie[node_index];
	for (auto &edge : node.key_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
}

void ApplyTrieNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                   const JsonoView &view, size_t pos, size_t string_cursor, vector<unique_ptr<Vector>> &children,
                   idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	for (auto field_index : node.scalar_leaves) {
		WriteScalarField(bind_data.fields[field_index], view, Location {pos, string_cursor, true},
		                 *children[field_index], row);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (!node.key_edges.empty()) {
		if (slot_tag == tag::OBJ_START) {
			ApplyObjectNode(lstate, bind_data, node_index, view, pos, string_cursor, children, row, join_buffers,
			                join_counts);
		} else {
			for (auto &edge : node.key_edges) {
				SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
			}
		}
	}
	if (!node.index_edges.empty() || node.wildcard_child != DConstants::INVALID_INDEX) {
		if (slot_tag == tag::ARR_START) {
			ApplyArrayNode(lstate, bind_data, node_index, view, pos, string_cursor, children, row, join_buffers,
			               join_counts);
		} else {
			for (auto &edge : node.index_edges) {
				SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
			}
			if (node.wildcard_child != DConstants::INVALID_INDEX) {
				SetTrieNodeOutputsNull(bind_data, node.wildcard_child, children, row);
			}
		}
	}
}

void SetTransformRowNull(const vector<TransformField> &fields, Vector &result, vector<unique_ptr<Vector>> &children,
                         idx_t row) {
	FlatVector::SetNull(result, row, true);
	for (idx_t col = 0; col < fields.size(); col++) {
		FlatVector::SetNull(*children[col], row, true);
		if (fields[col].mode == TransformMode::List) {
			ListVector::GetData(*children[col])[row] = {0, 0};
		}
	}
}

void JsonoTransformExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<TransformBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<TransformLocalState>();
	auto count = args.size();

	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (idx_t col = 0; col < bind_data.fields.size(); col++) {
		children[col]->SetVectorType(VectorType::FLAT_VECTOR);
		if (bind_data.fields[col].mode == TransformMode::List) {
			ListVector::SetListSize(*children[col], 0);
		}
	}

	// Read each lane child once; lane-backed scalar fields resolve residual-first and fall back
	// to these columns when the residual dropped the value.
	vector<UnifiedVectorFormat> lane_fmt(bind_data.lane_fields.size());
	if (!bind_data.lane_fields.empty()) {
		auto &struct_children = StructVector::GetEntries(args.data[0]);
		for (idx_t k = 0; k < bind_data.lane_fields.size(); k++) {
			auto lane_child = bind_data.fields[bind_data.lane_fields[k]].lane_child_index;
			struct_children[lane_child]->ToUnifiedFormat(count, lane_fmt[k]);
		}
	}

	vector<string> join_buffers(bind_data.fields.size());
	vector<idx_t> join_counts(bind_data.fields.size(), 0);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetTransformRowNull(bind_data.fields, result, children, row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetTransformRowNull(bind_data.fields, result, children, row);
			continue;
		}
		FlatVector::Validity(result).SetValid(row);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			if (field.mode == TransformMode::List) {
				auto start = ListVector::GetListSize(*children[field_index]);
				ListVector::GetData(*children[field_index])[row] = {start, 0};
				FlatVector::Validity(*children[field_index]).SetValid(row);
			} else if (field.mode == TransformMode::Join) {
				join_buffers[field_index].clear();
				join_counts[field_index] = 0;
			}
		}
		ApplyTrieNode(lstate, bind_data, 0, view, 0, 0, children, row, join_buffers, join_counts);
		for (idx_t k = 0; k < bind_data.lane_fields.size(); k++) {
			auto field_index = bind_data.lane_fields[k];
			auto &field = bind_data.fields[field_index];
			auto location = LocatePath(view, field.path);
			if (location.found) {
				// The residual is authoritative: a value the shred writer kept wins over the lane.
				WriteScalarField(field, view, location, *children[field_index], row);
				continue;
			}
			auto &fmt = lane_fmt[k];
			auto idx = RowIndex(fmt, row);
			if (fmt.validity.RowIsValid(idx)) {
				auto scalar = SynthLaneScalar(field.lane_primitive, fmt, idx);
				WriteScalarValue(field, scalar, *children[field_index], row);
			} else {
				FlatVector::SetNull(*children[field_index], row, true);
			}
		}
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			if (bind_data.fields[field_index].mode != TransformMode::Join) {
				continue;
			}
			auto &buffer = join_buffers[field_index];
			if (join_counts[field_index] == 0) {
				FlatVector::SetNull(*children[field_index], row, true);
				continue;
			}
			FlatVector::Validity(*children[field_index]).SetValid(row);
			FlatVector::GetData<string_t>(*children[field_index])[row] =
			    StringVector::AddString(*children[field_index], buffer.data(), buffer.size());
			buffer.clear();
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoTransform(ExtensionLoader &loader) {
	// ANY input so a shredded JSONO struct (four BLOB prefix + lane columns) also binds; the
	// bind validates it is a plain or shredded JSONO and maps lane columns to scalar fields.
	ScalarFunction fun("jsono_transform", {LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, JsonoTransformExecute,
	                   JsonoTransformBind, nullptr, nullptr, TransformLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	loader.RegisterFunction(fun);
}

} // namespace duckdb
