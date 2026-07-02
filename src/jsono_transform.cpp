#include "jsono_transform.hpp"
#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_scalar_write.hpp"
#include "jsono_shred.hpp"
#include "jsono_trie.hpp"
#include "jsono_trie_shape_plan.hpp"
#include "jsono_trie_walk.hpp"
#include "jsono_writer.hpp"

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
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace duckdb {
namespace {

using namespace jsono;

enum class TransformMode : uint8_t { Scalar, List, Join };
enum class TransformMismatchMode : uint8_t { Convert, Null, Fail };

struct TransformField {
	string name;
	JsonoScalarPrimitive primitive;
	TransformMode mode = TransformMode::Scalar;
	JsonoPathSpec path;
	string join_separator;
	// When the input is a shredded JSONO whose shred path equals this scalar field's path,
	// the field reads residual-first and falls back to this shred child (shred_primitive gives
	// the shred's lossless kind for synthesizing the stripped value). INVALID = not shred-backed.
	idx_t shred_child_index = DConstants::INVALID_INDEX;
	JsonoScalarPrimitive shred_primitive = JsonoScalarPrimitive::Varchar;
};

// Transform's trie reuses the shared skeleton (src/include/jsono_trie.hpp). The node carries
// transform-only leaf kinds (list/join) and wildcard edges; the shared walk consumes them through
// transform's leaf policy.
struct TransformBindData : public FunctionData {
	string spec;
	vector<TransformField> fields;
	JsonoTrie trie;
	LogicalType return_type;
	// Scalar fields backed by a shredded shred (field.shred_child_index set); handled directly
	// from the shred/residual rather than the trie. Empty for a plain JSONO input.
	vector<idx_t> shred_fields;
	TransformMismatchMode mismatch_mode = TransformMismatchMode::Convert;
	// Whether the shape-plan cache may run for this spec: the plan's per-row win is the
	// scalar fields it resolves to direct stream reads, so a spec with fewer than three of
	// them cannot recover the per-row key cost (hash + byte compare of slots and key_heap).
	bool shape_plan_eligible = false;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TransformBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<TransformBindData>();
		return spec == other.spec && mismatch_mode == other.mismatch_mode;
	}
};

// Shape-plan cache: V4 slots are shape-constant, so rows of one shape have byte-identical
// slots blobs. The plan resolves every spec field once per shape; matched rows skip the
// trie walk entirely — per row only a prefix-sum over the lengths stream (string leaf
// offsets) plus direct stream reads remain. Wildcard subtrees keep the trie walk (their
// output is per-element data), entered at a plan-recorded cursor.
//
// The cache key is the slots bytes PLUS the key_heap bytes: a KEY slot stores only its
// heap offset and length, so two rows whose keys differ in spelling but not in length
// have byte-identical slots — the key bytes that decided every locate verdict live in
// key_heap, which is itself byte-identical across rows of one shape.

struct ShapePlan : JsonoTrieShapePlan {
	// Indices into bind_data.shred_fields whose path is absent from the residual: the value
	// lives in the shred column.
	vector<idx_t> shred_reads;
};

// The per-row key cost (hash + byte compare of slots and key_heap) is O(row bytes), while
// the trie walk it replaces is O(spec) — sorted keys, rank caches and checkpoints keep the
// walk from ever scanning the whole row. Wide production rows (field_sample averages ~5.8KB
// of key bytes) therefore lose to the key cost even at a 93% hit rate; the plan only pays
// on narrow rows, where the key is a fraction of the walk. Rows past this bound take the
// walk path untouched.
constexpr idx_t SHAPE_PLAN_MIN_SCALAR_FIELDS = 3;

// The per-node rank cache is the shared JsonoTrieRankCache (src/include/jsono_trie.hpp), sized to
// the bind-time-constant trie. Transform's walk reads it through rank_cache.Get; the projector's
// walk reads its own copy through the same type.
struct TransformLocalState : public FunctionLocalState {
	explicit TransformLocalState(const TransformBindData &bind_data) {
		rank_cache.Init(bind_data.trie.nodes);
	}

	JsonoTrieRankCache rank_cache;
	JsonoTrieShapePlanCache<ShapePlan> shape_plans;
	// Per-row scratch for the plan apply: byte offsets for needed_length_indices.
	vector<uint64_t> shape_offsets;
	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		return make_uniq<TransformLocalState>(bind_data->Cast<TransformBindData>());
	}
};

JsonoScalarPrimitive ParsePrimitiveType(const string &type) {
	auto upper = StringUtil::Upper(type);
	if (upper == "BIGINT") {
		return JsonoScalarPrimitive::Bigint;
	}
	if (upper == "UBIGINT") {
		return JsonoScalarPrimitive::Ubigint;
	}
	if (upper == "DOUBLE") {
		return JsonoScalarPrimitive::Double;
	}
	if (upper == "VARCHAR") {
		return JsonoScalarPrimitive::Varchar;
	}
	if (upper == "BOOLEAN") {
		return JsonoScalarPrimitive::Boolean;
	}
	throw BinderException("jsono_transform: unsupported scalar type '%s'", type);
}

TransformMismatchMode ParseMismatchMode(Value mode) {
	if (mode.IsNull() || !mode.DefaultTryCastAs(LogicalType::VARCHAR)) {
		throw BinderException("jsono_transform: on_type_mismatch must be one of 'convert', 'null', or 'fail'");
	}
	auto value = StringUtil::Lower(StringValue::Get(mode));
	if (value == "convert") {
		return TransformMismatchMode::Convert;
	}
	if (value == "null") {
		return TransformMismatchMode::Null;
	}
	if (value == "fail") {
		return TransformMismatchMode::Fail;
	}
	throw BinderException("jsono_transform: on_type_mismatch must be one of 'convert', 'null', or 'fail'");
}

idx_t FindWildcardIndex(const vector<PathStep> &path) {
	for (idx_t i = 0; i < path.size(); i++) {
		if (path[i].kind == PathStepKind::Wildcard) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

string LiteralKeyPathText(nonstd::string_view key) {
	bool simple = !key.empty();
	for (auto c : key) {
		if (c == '.' || c == '[' || c == ']' || c == '"' || c == '\\') {
			simple = false;
			break;
		}
	}
	if (simple) {
		return "$." + string(key);
	}
	string result = "$.\"";
	for (auto c : key) {
		if (c == '"' || c == '\\') {
			result.push_back('\\');
		}
		result.push_back(c);
	}
	result.push_back('"');
	return result;
}

TransformField MakeField(nonstd::string_view name, JsonoScalarPrimitive primitive, TransformMode mode,
                         JsonoPathSpec path, string join_separator) {
	TransformField field;
	field.name = string(name);
	field.primitive = primitive;
	field.mode = mode;
	field.path = std::move(path);
	field.join_separator = std::move(join_separator);
	auto wildcard_index = FindWildcardIndex(field.path.steps);
	if (wildcard_index != DConstants::INVALID_INDEX && mode == TransformMode::Scalar) {
		throw BinderException("jsono_transform: wildcard path requires list type or join_separator");
	}
	if (wildcard_index == DConstants::INVALID_INDEX && mode != TransformMode::Scalar) {
		throw BinderException("jsono_transform: list and join modes require a wildcard path");
	}
	return field;
}

TransformField ParseScalarShorthand(nonstd::string_view name, nonstd::string_view type_name) {
	return MakeField(name, ParsePrimitiveType(string(type_name)), TransformMode::Scalar,
	                 JsonoPathSpec(LiteralKeyPathText(name), LiteralKeyPath(string(name))), string());
}

// A wrapper field is a nested STRUCT {type: ..., path: '...', join_separator: '...'}: `type`
// is a type string (scalar), or a single-element list `['TYPE']` (list mode); `path` and
// `join_separator` are strings. Mirrors the scalar-vs-struct dispatch the JSON form had.
TransformField ParseWrapperField(nonstd::string_view name, const Value &wrapper) {
	bool has_type = false;
	bool has_join_separator = false;
	JsonoScalarPrimitive primitive = JsonoScalarPrimitive::Varchar;
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
				if (items.size() != 1 || item.IsNull() || !item.DefaultTryCastAs(LogicalType::VARCHAR)) {
					throw BinderException("jsono_transform: unsupported list item type");
				}
				primitive = ParsePrimitiveType(StringValue::Get(item));
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
		if (mode == TransformMode::List || primitive != JsonoScalarPrimitive::Varchar) {
			throw BinderException("jsono_transform: join_separator requires VARCHAR scalar type");
		}
		mode = TransformMode::Join;
	}
	auto path_spec = has_path ? JsonoPathSpec(path, ParseJsonoPath(path, "jsono_transform"))
	                          : JsonoPathSpec(LiteralKeyPathText(name), LiteralKeyPath(string(name)));
	return MakeField(name, primitive, mode, std::move(path_spec), std::move(join_separator));
}

void InsertFieldIntoTrie(JsonoTrie &trie, const TransformField &field, idx_t field_index) {
	if (field.shred_child_index != DConstants::INVALID_INDEX) {
		// Shred-backed scalar: read directly from the shred/residual, never via the trie.
		return;
	}
	auto &node = trie.nodes[trie.WalkFieldToLeaf(field_index, field.path.steps)];
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
	JsonoTrie trie;
	trie.Reset();
	trie.PrepareFieldMetadata(bind_data.fields.size());
	for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
		InsertFieldIntoTrie(trie, bind_data.fields[field_index], field_index);
	}
	trie.SortEdges();
	for (auto &node : trie.nodes) {
		if (node.scalar_leaves.size() == 1 && node.list_leaves.empty() && node.join_leaves.empty() &&
		    node.key_edges.empty() && node.index_edges.empty() && node.wildcard_child == DConstants::INVALID_INDEX) {
			node.simple_scalar_leaf = node.scalar_leaves[0];
		}
	}
	bind_data.trie = std::move(trie);
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
		auto field_type = JsonoScalarPrimitiveLogicalType(transform_field.primitive);
		if (transform_field.mode == TransformMode::List) {
			field_type = LogicalType::LIST(field_type);
		}
		children.emplace_back(transform_field.name, std::move(field_type));
		bind_data->fields.push_back(std::move(transform_field));
	}
	if (bind_data->fields.empty()) {
		throw BinderException("jsono_transform: empty spec");
	}
	bind_data->return_type = LogicalType::STRUCT(std::move(children));
	BuildTransformTrie(*bind_data);
	return bind_data;
}

// Bind a shredded input's shred columns to the scalar fields they back: a scalar field whose
// path equals a shred's path reads that shred (residual-first, shred-fallback) instead of
// navigating the residual tape. The trie is rebuilt to exclude the bound fields.
void AssignShreddedShreds(TransformBindData &bind_data, const LogicalType &input_type) {
	JsonoLayoutType layout;
	TryParseJsonoLayoutType(input_type, layout);
	for (idx_t i = 0; i < layout.shreds.size(); i++) {
		JsonoScalarPrimitive shred_primitive;
		if (!TypeToJsonoScalarPrimitive(layout.shreds[i].second, shred_primitive)) {
			// A non-shred-typed shred is not a shred; any field on its path reads the
			// residual, where the value remains.
			continue;
		}
		auto &name = layout.shreds[i].first;
		auto steps = !name.empty() && name[0] == '$' ? ParseJsonoPath(name, "jsono_transform") : LiteralKeyPath(name);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			if (field.mode != TransformMode::Scalar || field.shred_child_index != DConstants::INVALID_INDEX) {
				continue;
			}
			if (PathStepsEqual(field.path.steps, steps)) {
				field.shred_child_index = i;
				field.shred_primitive = shred_primitive;
				bind_data.shred_fields.push_back(field_index);
			}
		}
	}
	if (!bind_data.shred_fields.empty()) {
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
	if (arguments.size() == 3) {
		if (arguments[2]->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arguments[2]->IsFoldable()) {
			throw BinderException("jsono_transform: on_type_mismatch must be constant");
		}
		if (arguments[2]->HasAlias() && !StringUtil::CIEquals(arguments[2]->GetAlias(), "on_type_mismatch")) {
			throw BinderException("jsono_transform: unknown named parameter '%s'; expected on_type_mismatch",
			                      arguments[2]->GetAlias());
		}
		bind_data->mismatch_mode = ParseMismatchMode(ExpressionExecutor::EvaluateScalar(context, *arguments[2]));
	}
	// The struct's canonical string identifies the full spec (names, types, paths, joins) for
	// expression caching — return_type alone is insufficient since a field's path may differ
	// from its name.
	bind_data->spec = spec_value.ToString() + ":" + std::to_string(uint8_t(bind_data->mismatch_mode));
	auto &input_type = arguments[0]->return_type;
	// An array shred (object or scalar) lifts element values out of the residual into a LIST column the
	// transform's residual navigation cannot read. Only a field that could touch that array — read it,
	// descend into it, or sit on a container subtree that holds it — needs the whole value reconstructed;
	// a sibling scalar like $.f0 stays on the native residual-first / shred-fallback fast path. So the
	// gate is per read field, not the mere presence of an array shred. A scalar shred in the value folds
	// back through that same reconstruct when one is forced.
	bool reconstruct_shredded = false;
	if (IsShreddedJsonoType(input_type)) {
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(input_type, layout);
		for (auto &shred : layout.shreds) {
			if (!IsShredListType(shred.second)) {
				continue;
			}
			auto &name = shred.first;
			auto shred_steps =
			    !name.empty() && name[0] == '$' ? ParseJsonoPath(name, "jsono_transform") : LiteralKeyPath(name);
			for (auto &field : bind_data->fields) {
				if (PathStepsMayShareBranch(field.path.steps, shred_steps)) {
					reconstruct_shredded = true;
					break;
				}
			}
			if (reconstruct_shredded) {
				break;
			}
		}
	}
	bound_function.arguments[0] =
	    JsonoResolveJsonoArgument(context, *arguments[0], "jsono_transform", reconstruct_shredded);
	if (IsShreddedJsonoType(input_type) && !reconstruct_shredded) {
		AssignShreddedShreds(*bind_data, input_type);
	}
	idx_t scalar_fields = 0;
	for (auto &field : bind_data->fields) {
		if (field.mode == TransformMode::Scalar) {
			scalar_fields++;
		}
	}
	bind_data->shape_plan_eligible =
	    bind_data->mismatch_mode != TransformMismatchMode::Fail && scalar_fields >= SHAPE_PLAN_MIN_SCALAR_FIELDS;
	bound_function.return_type = bind_data->return_type;
	// A field descending into an array shred forces the whole value to be reconstructed to plain
	// JSONO. Wrap the argument in __jsono_reconstruct so that reconstruct is a visible operator in
	// EXPLAIN rather than an anonymous cast the binder would otherwise insert (arguments[0] is now
	// plain, matching the plain type JsonoResolveJsonoArgument returned, so no cast is added).
	if (reconstruct_shredded) {
		arguments[0] = MakeJsonoReconstructExpression(std::move(arguments[0]));
	}
	return std::move(bind_data);
}

template <class T>
bool TryCastScalarText(nonstd::string_view text, T &value) {
	return TryCast::Operation<string_t, T>(string_t(text.data(), uint32_t(text.size())), value, false);
}

template <class T>
bool TryCastDec60Text(const JsonoScalar &scalar, T &value) {
	string text;
	AppendDec60Text(text, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
	return TryCastScalarText(nonstd::string_view(text.data(), text.size()), value);
}

// Shared scalar->numeric conversion ladder for the integral/boolean Convert targets. The DOUBLE
// target handles Dec60 via Dec60ToDouble instead of text and stays inline at its call site.
template <class T>
bool TryCastScalarTo(const JsonoScalar &scalar, T &value) {
	switch (scalar.kind) {
	case JsonoScalarKind::Bool:
		return TryCast::Operation(scalar.bool_value, value, false);
	case JsonoScalarKind::Int64:
		return TryCast::Operation(scalar.int_value, value, false);
	case JsonoScalarKind::UInt64:
		return TryCast::Operation(scalar.uint_value, value, false);
	case JsonoScalarKind::Double:
		return TryCast::Operation(scalar.double_value, value, false);
	case JsonoScalarKind::Dec60:
		return TryCastDec60Text(scalar, value);
	case JsonoScalarKind::NumberText:
	case JsonoScalarKind::String:
		return TryCastScalarText(scalar.text, value);
	default:
		return false;
	}
}

bool TryWriteScalarValue(const TransformField &field, TransformMismatchMode mode, const JsonoScalar &scalar,
                         Vector &result, idx_t row) {
	bool convert = mode != TransformMismatchMode::Null;
	switch (field.primitive) {
	case JsonoScalarPrimitive::Bigint: {
		// UInt64 values exceed INT64_MAX by construction, so only Int64 fits a BIGINT target.
		int64_t value = 0;
		if (convert && TryCastScalarTo(scalar, value)) {
			WriteJsonoBigintLane(result, row, value);
			return true;
		}
		if (!convert && scalar.kind == JsonoScalarKind::Int64) {
			WriteJsonoBigintLane(result, row, scalar.int_value);
			return true;
		}
		break;
	}
	case JsonoScalarPrimitive::Ubigint: {
		uint64_t value = 0;
		if (convert && TryCastScalarTo(scalar, value)) {
			WriteJsonoUbigintLane(result, row, value);
			return true;
		}
		if (!convert && scalar.kind == JsonoScalarKind::Int64 && scalar.int_value >= 0) {
			WriteJsonoUbigintLane(result, row, uint64_t(scalar.int_value));
			return true;
		}
		if (!convert && scalar.kind == JsonoScalarKind::UInt64) {
			WriteJsonoUbigintLane(result, row, scalar.uint_value);
			return true;
		}
		break;
	}
	case JsonoScalarPrimitive::Double: {
		double value = 0;
		bool ok = true;
		switch (scalar.kind) {
		case JsonoScalarKind::Bool:
			ok = convert && TryCast::Operation(scalar.bool_value, value, false);
			break;
		case JsonoScalarKind::Double:
			ok = convert ? TryCast::Operation(scalar.double_value, value, false) : true;
			if (!convert) {
				value = scalar.double_value;
			}
			break;
		case JsonoScalarKind::Dec60:
			value = Dec60ToDouble(scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			break;
		case JsonoScalarKind::Int64:
			ok = convert ? TryCast::Operation(scalar.int_value, value, false) : true;
			if (!convert) {
				value = double(scalar.int_value);
			}
			break;
		case JsonoScalarKind::UInt64:
			ok = convert ? TryCast::Operation(scalar.uint_value, value, false) : true;
			if (!convert) {
				value = double(scalar.uint_value);
			}
			break;
		case JsonoScalarKind::NumberText:
			// Locale-independent parse via DuckDB's fast_float-backed cast (no allocation).
			ok = convert ? TryCastScalarText(scalar.text, value)
			             : TryCast::Operation<string_t, double>(
			                   string_t(scalar.text.data(), uint32_t(scalar.text.size())), value);
			break;
		case JsonoScalarKind::String:
			ok = convert && TryCastScalarText(scalar.text, value);
			break;
		default:
			ok = false;
			break;
		}
		if (ok) {
			WriteJsonoDoubleLane(result, row, value);
			return true;
		}
		break;
	}
	case JsonoScalarPrimitive::Varchar: {
		if (scalar.kind == JsonoScalarKind::String) {
			WriteJsonoStringLane(result, row, scalar.text);
			return true;
		}
		if (!convert) {
			break;
		}
		string text;
		if (RenderExtractText(scalar, text)) {
			return false;
		}
		WriteJsonoStringLane(result, row, nonstd::string_view(text.data(), text.size()));
		return true;
	}
	case JsonoScalarPrimitive::Boolean: {
		bool value = false;
		if (convert && TryCastScalarTo(scalar, value)) {
			WriteJsonoBooleanLane(result, row, value);
			return true;
		}
		if (!convert && scalar.kind == JsonoScalarKind::Bool) {
			WriteJsonoBooleanLane(result, row, scalar.bool_value);
			return true;
		}
		break;
	}
	}
	return false;
}

[[noreturn]] void ThrowTransformMismatch(const TransformField &field, const char *actual_type) {
	throw InvalidInputException("jsono_transform: cannot convert value at %s from %s to %s", field.path.text.c_str(),
	                            actual_type, JsonoScalarPrimitiveTypeName(field.primitive));
}

void WriteScalarValue(const TransformField &field, TransformMismatchMode mode, const JsonoScalar &scalar,
                      Vector &result, idx_t row) {
	if (scalar.kind == JsonoScalarKind::Null) {
		FlatVector::SetNull(result, row, true);
		return;
	}
	bool ok = TryWriteScalarValue(field, mode, scalar, result, row);
	if (ok) {
		return;
	}
	if (mode == TransformMismatchMode::Fail) {
		ThrowTransformMismatch(field, JsonoScalarTypeName(scalar.kind));
	}
	FlatVector::SetNull(result, row, true);
}

void WriteScalarField(const TransformField &field, const JsonoView &view, const JsonoPathLocation &location,
                      Vector &result, idx_t row, TransformMismatchMode mode) {
	if (!location.found) {
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto value_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		// A container where a scalar was requested is a type mismatch, not malformed input.
		if (mode == TransformMismatchMode::Fail) {
			ThrowTransformMismatch(field, value_tag == tag::OBJ_START ? "OBJECT" : "ARRAY");
		}
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	WriteScalarValue(field, mode, scalar, result, row);
}

// What the trie walk does at a matched (or missing) scalar leaf, factored out of the walk so a
// second leaf output (the `->>` text projector) can ride the same trie/walk/shape-plan engine.
// The walk and edge structure are shared; only the per-leaf value write differs. Hooks are
// JSONO_ALWAYS_INLINE so the single transform instantiation monomorphizes to the prior code.
struct TypedValuePolicy {
	static JSONO_ALWAYS_INLINE void WriteScalarLeaf(const TransformBindData &bind_data, idx_t field_index,
	                                                const JsonoView &view, const JsonoCursor &cursor,
	                                                vector<unique_ptr<Vector>> &children, idx_t row) {
		WriteScalarField(bind_data.fields[field_index], view, JsonoPathLocation {cursor, true}, *children[field_index],
		                 row, bind_data.mismatch_mode);
	}

	static JSONO_ALWAYS_INLINE void NullScalarLeaf(vector<unique_ptr<Vector>> &children, idx_t field_index, idx_t row) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
};

template <class LEAF>
JSONO_ALWAYS_INLINE void NullTransformNodeLeaves(vector<unique_ptr<Vector>> &children, const JsonoTrieNode &node,
                                                 idx_t row) {
	for (auto field_index : node.scalar_leaves) {
		LEAF::NullScalarLeaf(children, field_index, row);
	}
	for (auto field_index : node.list_leaves) {
		SetListRowNull(*children[field_index], row);
	}
	for (auto field_index : node.join_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
}

void AppendListElementNull(Vector &result, idx_t row) {
	auto &entry = ListVector::GetData(result)[row];
	auto child_row = entry.offset + entry.length;
	EnsureListCapacity(result, child_row + 1);
	FlatVector::SetNull(ListVector::GetEntry(result), child_row, true);
	entry.length++;
	ListVector::SetListSize(result, entry.offset + entry.length);
}

void AppendListElementScalar(const TransformField &field, TransformMismatchMode mode, Vector &result, idx_t row,
                             const JsonoScalar &scalar) {
	auto &entry = ListVector::GetData(result)[row];
	auto child_row = entry.offset + entry.length;
	EnsureListCapacity(result, child_row + 1);
	WriteScalarValue(field, mode, scalar, ListVector::GetEntry(result), child_row);
	entry.length++;
	ListVector::SetListSize(result, entry.offset + entry.length);
}

bool TryGetJoinText(const TransformField &field, TransformMismatchMode mode, const JsonoScalar &scalar, string &scratch,
                    nonstd::string_view &value) {
	if (scalar.kind == JsonoScalarKind::Null) {
		return false;
	}
	if (mode == TransformMismatchMode::Null) {
		if (scalar.kind == JsonoScalarKind::String) {
			value = scalar.text;
			return true;
		}
		return false;
	}
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		value = scalar.text;
		return true;
	}
	JsonoScalarText text;
	if (!TryGetScalarExtractText(scalar, scratch, text)) {
		if (mode == TransformMismatchMode::Fail) {
			ThrowTransformMismatch(field, JsonoScalarTypeName(scalar.kind));
		}
		return false;
	}
	value = text.value;
	return true;
}

// Materialize one wildcard element for this node's list/join leaves. Missing and JSON null
// stay a positional NULL in list mode and are skipped by join mode; present containers are
// scalar mismatches for leaves at this exact wildcard node.
void MaterializeTransformWildcardLeaves(const TransformBindData &bind_data, idx_t node_index, const JsonoView &view,
                                        const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                                        vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie.nodes[node_index];
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag != tag::OBJ_START && slot_tag != tag::ARR_START) {
		auto value_cursor = cursor;
		auto scalar = DecodeScalarAt(view, value_cursor);
		for (auto field_index : node.list_leaves) {
			AppendListElementScalar(bind_data.fields[field_index], bind_data.mismatch_mode, *children[field_index], row,
			                        scalar);
		}
		for (auto field_index : node.join_leaves) {
			nonstd::string_view value;
			string scratch;
			if (TryGetJoinText(bind_data.fields[field_index], bind_data.mismatch_mode, scalar, scratch, value)) {
				auto &buffer = join_buffers[field_index];
				if (join_counts[field_index] > 0) {
					buffer.append(bind_data.fields[field_index].join_separator);
				}
				buffer.append(value.data(), value.size());
				join_counts[field_index]++;
			}
		}
		return;
	}
	for (auto field_index : node.list_leaves) {
		if (bind_data.mismatch_mode == TransformMismatchMode::Fail) {
			ThrowTransformMismatch(bind_data.fields[field_index], slot_tag == tag::OBJ_START ? "OBJECT" : "ARRAY");
		}
		AppendListElementNull(*children[field_index], row);
	}
	for (auto field_index : node.join_leaves) {
		if (bind_data.mismatch_mode == TransformMismatchMode::Fail) {
			ThrowTransformMismatch(bind_data.fields[field_index], slot_tag == tag::OBJ_START ? "OBJECT" : "ARRAY");
		}
	}
}

template <class LEAF>
struct TransformTrieWalkState {
	const TransformBindData &bind_data;
	vector<unique_ptr<Vector>> &children;
	vector<string> &join_buffers;
	vector<idx_t> &join_counts;
};

template <class LEAF>
struct TransformTrieWalkPolicy {
	using State = TransformTrieWalkState<LEAF>;
	static constexpr bool supports_wildcard = true;

	static JSONO_ALWAYS_INLINE void WriteScalarLeaf(State &state, idx_t field_index, const JsonoView &view,
	                                                const JsonoCursor &cursor, idx_t row) {
		LEAF::WriteScalarLeaf(state.bind_data, field_index, view, cursor, state.children, row);
	}

	static JSONO_ALWAYS_INLINE void NullScalarLeaf(State &state, idx_t field_index, const JsonoView &view, idx_t row) {
		(void)view;
		LEAF::NullScalarLeaf(state.children, field_index, row);
	}

	static JSONO_ALWAYS_INLINE void NullNodeLeaves(State &state, const JsonoTrieNode &node, const JsonoView &view,
	                                               idx_t row) {
		(void)view;
		NullTransformNodeLeaves<LEAF>(state.children, node, row);
	}

	static JSONO_ALWAYS_INLINE void AppendWildcardMissingLeaves(State &state, const JsonoTrieNode &node,
	                                                            const JsonoView &view, idx_t row) {
		(void)view;
		for (auto field_index : node.list_leaves) {
			AppendListElementNull(*state.children[field_index], row);
		}
	}

	static JSONO_ALWAYS_INLINE void MaterializeWildcardLeaves(State &state, idx_t node_index, const JsonoView &view,
	                                                          const JsonoCursor &cursor, idx_t row) {
		MaterializeTransformWildcardLeaves(state.bind_data, node_index, view, cursor, state.children, row,
		                                   state.join_buffers, state.join_counts);
	}
};

template <class LEAF>
void ApplyTrieNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                   const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                   vector<string> &join_buffers, vector<idx_t> &join_counts) {
	using WalkPolicy = TransformTrieWalkPolicy<LEAF>;
	typename WalkPolicy::State state {bind_data, children, join_buffers, join_counts};
	JsonoTrieWalkContext<WalkPolicy> ctx {bind_data.trie.nodes, lstate.rank_cache, state};
	JsonoTrieApplyNode<WalkPolicy>(ctx, node_index, view, cursor, row);
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

struct TransformShapeFieldPolicy {
	const TransformBindData &bind_data;

	bool IncludeField(idx_t field_index) {
		auto &field = bind_data.fields[field_index];
		return field.mode == TransformMode::Scalar && field.shred_child_index == DConstants::INVALID_INDEX;
	}

	void Missing(ShapePlan &plan, idx_t field_index) {
		plan.null_fields.push_back(field_index);
	}

	void Found(ShapePlan &plan, const JsonoView &view, const JsonoCursor &cursor, idx_t field_index,
	           vector<idx_t> &walked_nodes) {
		(void)walked_nodes;
		JsonoTrieShapePlanScalarAt(plan, view, cursor, field_index);
	}
};

// Resolve every spec field against one row of this shape. Wildcard fields resolve first:
// each distinct wildcard-prefix trie node becomes one walk (prefix found) or one
// null-subtree (prefix missing) action, deduplicated so an ancestor's walk
// covers descendants. Scalar fields whose trie chain passes through such a node are
// covered by it; the rest get a direct action or a NULL.
unique_ptr<ShapePlan> BuildShapePlan(TransformLocalState &lstate, const TransformBindData &bind_data,
                                     const JsonoView &view) {
	auto plan = make_uniq<ShapePlan>();

	// Wildcard-prefix candidates with their root chains (chain.back() == node).
	vector<JsonoTrieShapeCoverCandidate> candidate_nodes;
	for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
		auto &field = bind_data.fields[field_index];
		if (field.mode == TransformMode::Scalar || field.shred_child_index != DConstants::INVALID_INDEX) {
			continue;
		}
		auto prefix_len = FindWildcardIndex(field.path.steps);
		auto &chain = bind_data.trie.field_node_chains[field_index];
		if (prefix_len >= chain.size()) {
			throw InternalException("jsono_transform: wildcard prefix missing from trie metadata");
		}
		auto node = chain[prefix_len];
		candidate_nodes.push_back(JsonoTrieShapeCoverCandidate(
		    node, field_index, prefix_len, vector<idx_t>(chain.begin(), chain.begin() + prefix_len + 1)));
	}
	vector<JsonoTrieShapeCoverCandidate> covering_nodes;
	JsonoTrieShapePlanSelectCoverNodes(candidate_nodes, covering_nodes);
	TransformShapeFieldPolicy field_policy {bind_data};
	JsonoTrieShapePlanAddReads(*plan, bind_data.trie, bind_data.fields, covering_nodes, view, field_policy);

	// Shred-backed scalars: residual-first, exactly like the per-row fallback. A residual
	// hit is shape-constant; a miss reads the shred column per row.
	for (idx_t k = 0; k < bind_data.shred_fields.size(); k++) {
		auto &field = bind_data.fields[bind_data.shred_fields[k]];
		JsonoCursor cursor;
		if (LocatePathSteps(nullptr, 0, field.path.steps, view, cursor)) {
			JsonoTrieShapePlanScalarAt(*plan, view, cursor, bind_data.shred_fields[k]);
		} else {
			plan->shred_reads.push_back(k);
		}
	}

	JsonoTrieShapePlanFinalize(*plan);
	return plan;
}

void ApplyShapePlan(TransformLocalState &lstate, const TransformBindData &bind_data, const ShapePlan &plan,
                    const JsonoView &view, vector<unique_ptr<Vector>> &children, idx_t row,
                    vector<string> &join_buffers, vector<idx_t> &join_counts,
                    const vector<UnifiedVectorFormat> &shred_fmt) {
	// One pass over the lengths stream produces the byte offsets of every needed string
	// position (offset before needed index i = sum of lengths[0, i)).
	JsonoTrieShapePlanScanLengthOffsets(plan, view, lstate.shape_offsets);
	const uint8_t *lengths_raw = plan.lengths_bound > 0 ? view.LengthsBytes(0, plan.lengths_bound) : nullptr;
	const uint8_t *nums_raw = plan.nums_bound > 0 ? view.NumsBytes(0, plan.nums_bound) : nullptr;
	auto &offsets = lstate.shape_offsets;
	using WalkPolicy = TransformTrieWalkPolicy<TypedValuePolicy>;
	typename WalkPolicy::State walk_state {bind_data, children, join_buffers, join_counts};

	for (auto field_index : plan.null_fields) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
	JsonoTrieShapePlanApplyNullSubtrees<WalkPolicy>(bind_data.trie.nodes, walk_state, plan, view, row);
	for (auto &action : plan.scalars) {
		JsonoScalar scalar;
		scalar.kind = action.kind;
		switch (action.kind) {
		case JsonoScalarKind::String:
		case JsonoScalarKind::NumberText: {
			uint32_t len;
			std::memcpy(&len, lengths_raw + action.length_index * sizeof(uint32_t), sizeof(len));
			scalar.text = view.StringAt(offsets[action.offset_slot], len);
			break;
		}
		case JsonoScalarKind::Int64: {
			uint64_t word;
			std::memcpy(&word, nums_raw + action.num_index * sizeof(uint64_t), sizeof(word));
			scalar.int_value = int64_t(word);
			break;
		}
		case JsonoScalarKind::UInt64:
			std::memcpy(&scalar.uint_value, nums_raw + action.num_index * sizeof(uint64_t), sizeof(scalar.uint_value));
			break;
		case JsonoScalarKind::Double:
			std::memcpy(&scalar.double_value, nums_raw + action.num_index * sizeof(uint64_t),
			            sizeof(scalar.double_value));
			break;
		case JsonoScalarKind::Dec60: {
			uint64_t packed;
			std::memcpy(&packed, nums_raw + action.num_index * sizeof(uint64_t), sizeof(packed));
			scalar.dec_negative = Dec60Negative(packed);
			scalar.dec_mantissa = Dec60Mantissa(packed);
			scalar.dec_scale = Dec60Scale(packed);
			break;
		}
		case JsonoScalarKind::Bool:
			scalar.bool_value = action.bool_value;
			break;
		default:
			break;
		}
		WriteScalarValue(bind_data.fields[action.field_index], bind_data.mismatch_mode, scalar,
		                 *children[action.field_index], row);
	}
	JsonoTrieShapePlanApplyWalks<WalkPolicy>(bind_data.trie.nodes, lstate.rank_cache, walk_state, plan, view, offsets,
	                                         row);
	for (auto k : plan.shred_reads) {
		auto field_index = bind_data.shred_fields[k];
		auto &field = bind_data.fields[field_index];
		auto &fmt = shred_fmt[k];
		auto idx = RowIndex(fmt, row);
		if (fmt.validity.RowIsValid(idx)) {
			auto scalar = JsonoScalarFromPrimitiveVector(field.shred_primitive, fmt, idx);
			WriteScalarValue(field, bind_data.mismatch_mode, scalar, *children[field_index], row);
		} else {
			FlatVector::SetNull(*children[field_index], row, true);
		}
	}
}

void JsonoTransformExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<TransformBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<TransformLocalState>();
	auto count = args.size();

	// The row reader verifies each row's shred manifest against the shreds this input type
	// carries, so a row narrowed by a raw struct cast fails loud instead of transforming an
	// incomplete residual.
	JsonoRowReader reader;
	reader.Init(args.data[0], count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (idx_t col = 0; col < bind_data.fields.size(); col++) {
		children[col]->SetVectorType(VectorType::FLAT_VECTOR);
		if (bind_data.fields[col].mode == TransformMode::List) {
			ListVector::SetListSize(*children[col], 0);
		}
	}

	// Read each shred child once; shred-backed scalar fields resolve residual-first and fall back
	// to these columns when the residual dropped the value.
	vector<UnifiedVectorFormat> shred_fmt(bind_data.shred_fields.size());
	if (!bind_data.shred_fields.empty()) {
		for (idx_t k = 0; k < bind_data.shred_fields.size(); k++) {
			auto shred_child = bind_data.fields[bind_data.shred_fields[k]].shred_child_index;
			JsonoShredVector(args.data[0], shred_child).ToUnifiedFormat(count, shred_fmt[k]);
		}
	}

	vector<string> join_buffers(bind_data.fields.size());
	vector<idx_t> join_counts(bind_data.fields.size(), 0);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
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
		// The window check runs before the lookup: a disable clears the entries, which would
		// dangle a plan pointer handed out for this row.
		lstate.shape_plans.EndOfWindowCheck();
		const ShapePlan *plan = nullptr;
		if (bind_data.shape_plan_eligible &&
		    blob.slots.GetSize() + blob.key_heap.GetSize() <= JSONO_TRIE_SHAPE_PLAN_MAX_SHAPE_BYTES) {
			uint64_t hash;
			if (lstate.shape_plans.CanLookup(blob, hash)) {
				plan = lstate.shape_plans.Find(hash, blob);
				if (!plan) {
					plan = lstate.shape_plans.Insert(hash, blob, BuildShapePlan(lstate, bind_data, view));
				}
			}
		}
		if (plan) {
			ApplyShapePlan(lstate, bind_data, *plan, view, children, row, join_buffers, join_counts, shred_fmt);
		} else {
			ApplyTrieNode<TypedValuePolicy>(lstate, bind_data, 0, view, JsonoCursor(), children, row, join_buffers,
			                                join_counts);
			for (idx_t k = 0; k < bind_data.shred_fields.size(); k++) {
				auto field_index = bind_data.shred_fields[k];
				auto &field = bind_data.fields[field_index];
				auto location = LocatePath(view, field.path.steps);
				if (location.found) {
					// The residual is authoritative: a value the shred writer kept wins over the shred.
					WriteScalarField(field, view, location, *children[field_index], row, bind_data.mismatch_mode);
					continue;
				}
				auto &fmt = shred_fmt[k];
				auto idx = RowIndex(fmt, row);
				if (fmt.validity.RowIsValid(idx)) {
					auto scalar = JsonoScalarFromPrimitiveVector(field.shred_primitive, fmt, idx);
					WriteScalarValue(field, bind_data.mismatch_mode, scalar, *children[field_index], row);
				} else {
					FlatVector::SetNull(*children[field_index], row, true);
				}
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
			WriteJsonoStringLane(*children[field_index], row, nonstd::string_view(buffer.data(), buffer.size()));
			buffer.clear();
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoTransform(ExtensionLoader &loader) {
	// ANY input so a shredded JSONO struct (six BLOB prefix + shred columns) also binds; the
	// bind validates it is a plain or shredded JSONO and maps shred columns to scalar fields.
	ScalarFunctionSet set("jsono_transform");
	ScalarFunction binary({LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, JsonoTransformExecute,
	                      JsonoTransformBind, nullptr, nullptr, TransformLocalState::Init);
	binary.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	binary.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	set.AddFunction(std::move(binary));

	ScalarFunction ternary({LogicalType::ANY, LogicalType::ANY, LogicalType::VARCHAR}, LogicalType::ANY,
	                       JsonoTransformExecute, JsonoTransformBind, nullptr, nullptr, TransformLocalState::Init);
	ternary.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	ternary.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	set.AddFunction(std::move(ternary));
	loader.RegisterFunction(set);
}

} // namespace duckdb
