#include "jsono_optimizer.hpp"
#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/struct_functions.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace duckdb {

// Exposed jsono function factories used to build the shredded reconstruction patch
// and to read non-lane paths natively off the residual.
ScalarFunction JsonoStructConstructorFunction();
ScalarFunction JsonoOverlayFunction();
ScalarFunction JsonoExtractStringFunction(const LogicalType &path_type);
ScalarFunction JsonoExtractFunction(const LogicalType &path_type);

namespace {

using namespace jsono;

enum class JsonoCompiledPathKind { Generic, RootKey, RootKeyKey };

struct JsonoCompiledPath {
	JsonoCompiledPathKind kind = JsonoCompiledPathKind::Generic;
	string first_key;
	string second_key;
};

struct JsonoMatchPredicate {
	string path;
	vector<PathStep> steps;
	JsonoCompiledPath compiled_path;
	vector<string> values;
	idx_t step_base = 0;
};

struct JsonoMatchBindData : public FunctionData {
	vector<JsonoMatchPredicate> predicates;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<JsonoMatchBindData>();
		result->predicates = predicates;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoMatchBindData>();
		if (predicates.size() != other.predicates.size()) {
			return false;
		}
		for (idx_t predicate_index = 0; predicate_index < predicates.size(); predicate_index++) {
			auto &left = predicates[predicate_index];
			auto &right = other.predicates[predicate_index];
			if (left.path != right.path || left.step_base != right.step_base || left.values != right.values ||
			    left.steps.size() != right.steps.size()) {
				return false;
			}
			for (idx_t step_index = 0; step_index < left.steps.size(); step_index++) {
				if (left.steps[step_index].kind != right.steps[step_index].kind ||
				    left.steps[step_index].key != right.steps[step_index].key ||
				    left.steps[step_index].index != right.steps[step_index].index) {
					return false;
				}
			}
		}
		return true;
	}
};

struct JsonoProjectField {
	string name;
	string path;
	vector<PathStep> steps;
	JsonoCompiledPath compiled_path;
	idx_t step_base = 0;
};

struct JsonoProjectBindData : public FunctionData {
	vector<JsonoProjectField> fields;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<JsonoProjectBindData>();
		result->fields = fields;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoProjectBindData>();
		if (fields.size() != other.fields.size()) {
			return false;
		}
		for (idx_t field_index = 0; field_index < fields.size(); field_index++) {
			auto &left = fields[field_index];
			auto &right = other.fields[field_index];
			if (left.name != right.name || left.path != right.path || left.step_base != right.step_base ||
			    left.steps.size() != right.steps.size()) {
				return false;
			}
			for (idx_t step_index = 0; step_index < left.steps.size(); step_index++) {
				if (left.steps[step_index].kind != right.steps[step_index].kind ||
				    left.steps[step_index].key != right.steps[step_index].key ||
				    left.steps[step_index].index != right.steps[step_index].index) {
					return false;
				}
			}
		}
		return true;
	}
};

struct JsonoPathRankCacheEntry {
	bool valid = false;
	idx_t step_index = DConstants::INVALID_INDEX;
	size_t key_count = 0;
	uint64_t shape_hash = 0;
	string key;
	size_t rank = 0;
	bool found = false;
};

constexpr idx_t RANK_CACHE_SIZE = 4096;

// Per-row ObjectLayout cache. A row's matcher/projector resolves several paths that
// share a container (e.g. every root-key predicate re-reads the root object's layout;
// every commit.* predicate re-reads commit's). ReadObjectLayout was the top JSONO
// hotspot after H1 (~10% self-time on Q5), so cache it keyed by (generation, pos): a
// generation bump per row invalidates the whole cache in O(1) without clearing.
struct JsonoLayoutCacheEntry {
	uint64_t generation = 0;
	size_t pos = 0;
	ObjectLayout layout {};
};

constexpr idx_t LAYOUT_CACHE_SIZE = 16;

struct JsonoPathLocalState : public FunctionLocalState {
	JsonoPathLocalState() : rank_cache(RANK_CACHE_SIZE), layout_cache(LAYOUT_CACHE_SIZE) {
	}

	vector<JsonoPathRankCacheEntry> rank_cache;
	vector<JsonoLayoutCacheEntry> layout_cache;
	uint64_t layout_generation = 0;
	std::string scratch;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoPathLocalState>();
	}
};

struct JsonoMatchLocation {
	size_t pos = 0;
	size_t string_cursor = 0;
};

struct JsonoRewritePredicate {
	ColumnBinding binding;
	LogicalType column_type;
	string column_alias;
	JsonoMatchPredicate predicate;
	idx_t expression_index = 0;
};

struct JsonoRewriteGroup {
	ColumnBinding binding;
	LogicalType column_type;
	string column_alias;
	vector<JsonoMatchPredicate> predicates;
	vector<idx_t> expression_indexes;
};

vector<PathStep> LiteralKeyPath(nonstd::string_view name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, string(name), 0});
	return path;
}

vector<PathStep> ArrayIndexPath(idx_t index) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Index, string(), index});
	return path;
}

bool PathStepEquals(const PathStep &left, const PathStep &right) {
	return left.kind == right.kind && left.key == right.key && left.index == right.index;
}

bool PathStepsEqual(const vector<PathStep> &left, const vector<PathStep> &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (idx_t step_index = 0; step_index < left.size(); step_index++) {
		if (!PathStepEquals(left[step_index], right[step_index])) {
			return false;
		}
	}
	return true;
}

// True if `prefix` is a proper prefix of the object-key path `full`. A json-valued read at
// `prefix` would return a subtree that holds `full`'s leaf — which the shredded writer may
// have stripped from the residual — so such a read must reconstruct rather than read the
// residual directly. Only object-key lanes are stripped, so a non-key `full` is ignored.
bool PathStepsObjectKeyPrefix(const vector<PathStep> &prefix, const vector<PathStep> &full) {
	if (prefix.size() >= full.size()) {
		return false;
	}
	for (auto &step : full) {
		if (step.kind != PathStepKind::Key) {
			return false;
		}
	}
	for (idx_t i = 0; i < prefix.size(); i++) {
		if (prefix[i].kind != PathStepKind::Key || prefix[i].key != full[i].key) {
			return false;
		}
	}
	return true;
}

JsonoCompiledPath CompileJsonoPath(const vector<PathStep> &steps) {
	JsonoCompiledPath compiled_path;
	if (steps.size() == 1 && steps[0].kind == PathStepKind::Key) {
		compiled_path.kind = JsonoCompiledPathKind::RootKey;
		compiled_path.first_key = steps[0].key;
		return compiled_path;
	}
	if (steps.size() == 2 && steps[0].kind == PathStepKind::Key && steps[1].kind == PathStepKind::Key) {
		compiled_path.kind = JsonoCompiledPathKind::RootKeyKey;
		compiled_path.first_key = steps[0].key;
		compiled_path.second_key = steps[1].key;
		return compiled_path;
	}
	return compiled_path;
}

bool TryReadFoldableValue(ClientContext &context, Expression &expr, Value &value) {
	if (expr.HasParameter() || !expr.IsFoldable()) {
		return false;
	}
	value = ExpressionExecutor::EvaluateScalar(context, expr);
	return !value.IsNull();
}

bool TryReadStringValue(ClientContext &context, Expression &expr, string &result) {
	Value value;
	if (!TryReadFoldableValue(context, expr, value)) {
		return false;
	}
	if (!value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		return false;
	}
	result = StringValue::Get(value);
	return true;
}

bool TryReadStringListValue(ClientContext &context, Expression &expr, vector<string> &result) {
	Value value;
	if (!TryReadFoldableValue(context, expr, value) || value.type().id() != LogicalTypeId::LIST) {
		return false;
	}
	for (auto &child : ListValue::GetChildren(value)) {
		if (child.IsNull()) {
			return false;
		}
		auto string_child = child;
		if (!string_child.DefaultTryCastAs(LogicalType::VARCHAR)) {
			return false;
		}
		result.push_back(StringValue::Get(string_child));
	}
	return !result.empty();
}

bool TryReadExtractPath(ClientContext &context, Expression &expr, string &path, vector<PathStep> &steps) {
	Value value;
	if (!TryReadFoldableValue(context, expr, value)) {
		return false;
	}
	if (expr.return_type.IsIntegral()) {
		if (!value.DefaultTryCastAs(LogicalType::BIGINT)) {
			return false;
		}
		auto index = value.GetValue<int64_t>();
		if (index < 0) {
			return false;
		}
		path = std::to_string(index);
		steps = ArrayIndexPath(idx_t(index));
		return true;
	}
	if (!value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		return false;
	}
	path = StringValue::Get(value);
	if (path.empty()) {
		return false;
	}
	if (path[0] == '$') {
		steps = ParseJsonoPath(path, "__jsono_internal_match");
	} else {
		steps = LiteralKeyPath(path);
	}
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			return false;
		}
	}
	return true;
}

bool TryReadJsonoExtract(ClientContext &context, Expression &expr, BoundColumnRefExpression *&column_ref,
                         JsonoMatchPredicate &predicate) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	if (function.function.name != "->>" && function.function.name != "jsono_extract_string") {
		return false;
	}
	if (function.children.size() != 2 ||
	    function.children[0]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return false;
	}
	auto &candidate_ref = function.children[0]->Cast<BoundColumnRefExpression>();
	if (!IsJsonoType(candidate_ref.return_type)) {
		return false;
	}
	string path;
	vector<PathStep> steps;
	if (!TryReadExtractPath(context, *function.children[1], path, steps)) {
		return false;
	}
	column_ref = &candidate_ref;
	predicate.path = std::move(path);
	predicate.steps = std::move(steps);
	predicate.compiled_path = CompileJsonoPath(predicate.steps);
	return true;
}

bool TryReadEqualsPredicate(ClientContext &context, BoundComparisonExpression &comparison,
                            JsonoRewritePredicate &predicate) {
	if (comparison.GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
		return false;
	}
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate match_predicate;
	string value;
	if (!TryReadJsonoExtract(context, *comparison.left, column_ref, match_predicate) ||
	    !TryReadStringValue(context, *comparison.right, value)) {
		column_ref = nullptr;
		match_predicate = JsonoMatchPredicate();
		if (!TryReadJsonoExtract(context, *comparison.right, column_ref, match_predicate) ||
		    !TryReadStringValue(context, *comparison.left, value)) {
			return false;
		}
	}
	match_predicate.values.push_back(std::move(value));
	predicate.binding = column_ref->binding;
	predicate.column_type = column_ref->return_type;
	predicate.column_alias = column_ref->GetAlias();
	predicate.predicate = std::move(match_predicate);
	return true;
}

bool TryReadInPredicate(ClientContext &context, BoundOperatorExpression &comparison, JsonoRewritePredicate &predicate) {
	if (comparison.GetExpressionType() != ExpressionType::COMPARE_IN || comparison.children.size() < 2) {
		return false;
	}
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate match_predicate;
	if (!TryReadJsonoExtract(context, *comparison.children[0], column_ref, match_predicate)) {
		return false;
	}
	for (idx_t child_index = 1; child_index < comparison.children.size(); child_index++) {
		string value;
		if (!TryReadStringValue(context, *comparison.children[child_index], value)) {
			return false;
		}
		match_predicate.values.push_back(std::move(value));
	}
	predicate.binding = column_ref->binding;
	predicate.column_type = column_ref->return_type;
	predicate.column_alias = column_ref->GetAlias();
	predicate.predicate = std::move(match_predicate);
	return true;
}

bool TryReadContainsPredicate(ClientContext &context, BoundFunctionExpression &function,
                              JsonoRewritePredicate &predicate) {
	if (function.function.name != "contains" || function.children.size() != 2) {
		return false;
	}
	vector<string> values;
	if (!TryReadStringListValue(context, *function.children[0], values)) {
		return false;
	}
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate match_predicate;
	if (!TryReadJsonoExtract(context, *function.children[1], column_ref, match_predicate)) {
		return false;
	}
	match_predicate.values = std::move(values);
	predicate.binding = column_ref->binding;
	predicate.column_type = column_ref->return_type;
	predicate.column_alias = column_ref->GetAlias();
	predicate.predicate = std::move(match_predicate);
	return true;
}

bool TryReadPredicate(ClientContext &context, Expression &expr, JsonoRewritePredicate &predicate) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COMPARISON:
		return TryReadEqualsPredicate(context, expr.Cast<BoundComparisonExpression>(), predicate);
	case ExpressionClass::BOUND_FUNCTION:
		return TryReadContainsPredicate(context, expr.Cast<BoundFunctionExpression>(), predicate);
	case ExpressionClass::BOUND_OPERATOR:
		return TryReadInPredicate(context, expr.Cast<BoundOperatorExpression>(), predicate);
	default:
		return false;
	}
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
	return view.KeyAt(SlotPayload(key_slot)) == key;
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

JsonoPathRankCacheEntry &GetRankCacheEntry(vector<JsonoPathRankCacheEntry> &rank_cache, idx_t step_index,
                                           size_t key_count) {
	auto cache_index = idx_t(((uint64_t(step_index) * 0x9E3779B97F4A7C15ULL) ^ key_count) & (RANK_CACHE_SIZE - 1));
	return rank_cache[cache_index];
}

// Default fast path trusts the stored shape_hash; JSONO_RANK_VALIDATE=1 forces the
// old key-heap validation, isolating the format's storage/read overhead from the
// fast-path win in a single build (read once, not per row).
inline bool TrustShapeHash() {
	static const bool trust = []() {
		const char *env = std::getenv("JSONO_RANK_VALIDATE");
		return !(env && env[0] == '1');
	}();
	return trust;
}

bool FindPathObjectRank(JsonoPathLocalState &lstate, idx_t step_index, const JsonoView &view,
                        const ObjectLayout &layout, const string &key, size_t &rank) {
	auto &entry = GetRankCacheEntry(lstate.rank_cache, step_index, layout.key_count);
	// The stored shape_hash uniquely identifies the object's sorted key sequence, so a
	// matching (step_index, shape_hash, key) lets us trust the cached rank with a single
	// int compare instead of re-reading the key heap (ValidateCachedObjectRank). This is
	// the safe form of the H7 rank-cache fast path: the format carries shape identity, so
	// it stays correct on non-shape-stable data where (step_index, key_count) collides.
	bool hit;
	if (TrustShapeHash()) {
		hit =
		    entry.valid && entry.step_index == step_index && entry.shape_hash == layout.shape_hash && entry.key == key;
	} else {
		hit = entry.valid && entry.step_index == step_index && entry.key_count == layout.key_count &&
		      entry.key == key && ValidateCachedObjectRank(view, layout, key, entry.rank, entry.found);
	}
	if (hit) {
		rank = entry.rank;
		return entry.found;
	}
	bool found = FindObjectKeyRank(view, layout, key, rank);
	entry.valid = true;
	entry.step_index = step_index;
	entry.key_count = layout.key_count;
	entry.shape_hash = layout.shape_hash;
	entry.key = key;
	entry.rank = rank;
	entry.found = found;
	return found;
}

bool LocateObjectKey(JsonoPathLocalState &lstate, idx_t step_index, const JsonoView &view, const string &key,
                     size_t &pos, size_t &string_cursor) {
	if (pos >= view.Slots()) {
		return false;
	}
	// On a per-row cache hit the entry was stored only after a successful OBJ_START
	// read, so both the tag check and ReadObjectLayout are skipped.
	auto &entry = lstate.layout_cache[pos & (LAYOUT_CACHE_SIZE - 1)];
	if (entry.generation != lstate.layout_generation || entry.pos != pos) {
		if (SlotTag(view.SlotAt(pos)) != tag::OBJ_START) {
			return false;
		}
		entry.generation = lstate.layout_generation;
		entry.pos = pos;
		entry.layout = ReadObjectLayout(view, pos);
	}
	const auto &layout = entry.layout;
	size_t rank = 0;
	if (!FindPathObjectRank(lstate, step_index, view, layout, key, rank)) {
		return false;
	}
	size_t value_rank = 0;
	size_t value_pos = layout.value_start;
	size_t value_string_cursor = string_cursor;
	MoveObjectValueCursorToRank(view, layout, string_cursor, rank, value_rank, value_pos, value_string_cursor);
	pos = value_pos;
	string_cursor = value_string_cursor;
	return true;
}

bool LocatePathStep(JsonoPathLocalState &lstate, idx_t step_index, const JsonoView &view, const PathStep &step,
                    size_t &pos, size_t &string_cursor) {
	if (pos >= view.Slots()) {
		return false;
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (step.kind) {
	case PathStepKind::Key: {
		return LocateObjectKey(lstate, step_index, view, step.key, pos, string_cursor);
	}
	case PathStepKind::Index: {
		if (slot_tag != tag::ARR_START) {
			return false;
		}
		auto end_pos = ReadArrayEndPos(view, pos);
		size_t elem_pos = pos + 1;
		idx_t current = 0;
		while (elem_pos < end_pos && current < step.index) {
			SkipValueFast(view, elem_pos, string_cursor);
			current++;
		}
		if (elem_pos >= end_pos || current < step.index) {
			return false;
		}
		pos = elem_pos;
		return true;
	}
	case PathStepKind::Wildcard:
		return false;
	}
	return false;
}

bool LocateCompiledPath(JsonoPathLocalState &lstate, const JsonoCompiledPath &compiled_path, idx_t step_base,
                        const JsonoView &view, JsonoMatchLocation &location) {
	size_t pos = 0;
	size_t string_cursor = 0;
	switch (compiled_path.kind) {
	case JsonoCompiledPathKind::RootKey:
		if (!LocateObjectKey(lstate, step_base, view, compiled_path.first_key, pos, string_cursor)) {
			return false;
		}
		break;
	case JsonoCompiledPathKind::RootKeyKey:
		if (!LocateObjectKey(lstate, step_base, view, compiled_path.first_key, pos, string_cursor) ||
		    !LocateObjectKey(lstate, step_base + 1, view, compiled_path.second_key, pos, string_cursor)) {
			return false;
		}
		break;
	case JsonoCompiledPathKind::Generic:
		return false;
	}
	location.pos = pos;
	location.string_cursor = string_cursor;
	return true;
}

bool LocateSteps(JsonoPathLocalState &lstate, idx_t step_base, const vector<PathStep> &steps, const JsonoView &view,
                 JsonoMatchLocation &location) {
	size_t pos = 0;
	size_t string_cursor = 0;
	for (idx_t step_index = 0; step_index < steps.size(); step_index++) {
		if (!LocatePathStep(lstate, step_base + step_index, view, steps[step_index], pos, string_cursor)) {
			return false;
		}
	}
	location.pos = pos;
	location.string_cursor = string_cursor;
	return true;
}

bool LocatePath(JsonoPathLocalState &lstate, const JsonoMatchPredicate &predicate, const JsonoView &view,
                JsonoMatchLocation &location) {
	if (predicate.compiled_path.kind != JsonoCompiledPathKind::Generic) {
		return LocateCompiledPath(lstate, predicate.compiled_path, predicate.step_base, view, location);
	}
	return LocateSteps(lstate, predicate.step_base, predicate.steps, view, location);
}

bool LocateProjectField(JsonoPathLocalState &lstate, const JsonoProjectField &field, const JsonoView &view,
                        JsonoMatchLocation &location) {
	if (field.compiled_path.kind != JsonoCompiledPathKind::Generic) {
		return LocateCompiledPath(lstate, field.compiled_path, field.step_base, view, location);
	}
	return LocateSteps(lstate, field.step_base, field.steps, view, location);
}

bool ContainsExpectedValue(const vector<string> &values, nonstd::string_view candidate) {
	for (auto &value : values) {
		if (candidate == value) {
			return true;
		}
	}
	return false;
}

void AppendJsonString(nonstd::string_view value, std::string &out) {
	out.push_back('"');
	for (char c : value) {
		switch (c) {
		case '"':
			out.append("\\\"");
			break;
		case '\\':
			out.append("\\\\");
			break;
		case '\b':
			out.append("\\b");
			break;
		case '\f':
			out.append("\\f");
			break;
		case '\n':
			out.append("\\n");
			break;
		case '\r':
			out.append("\\r");
			break;
		case '\t':
			out.append("\\t");
			break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
				out.append(buf);
			} else {
				out.push_back(c);
			}
		}
	}
	out.push_back('"');
}

bool AppendExtractStringScalar(const JsonoScalar &scalar, std::string &out) {
	switch (scalar.kind) {
	case JsonoScalarKind::Null:
		return true;
	case JsonoScalarKind::String:
	case JsonoScalarKind::NumberText:
		out.append(scalar.text.data(), scalar.text.size());
		return false;
	case JsonoScalarKind::Int64:
		out.append(std::to_string(scalar.int_value));
		return false;
	case JsonoScalarKind::UInt64:
		out.append(std::to_string(scalar.uint_value));
		return false;
	case JsonoScalarKind::Double:
		EmitDouble(scalar.double_value, out);
		return false;
	case JsonoScalarKind::Dec60:
		AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
		return false;
	case JsonoScalarKind::Bool:
		out.append(scalar.bool_value ? "true" : "false");
		return false;
	}
	return true;
}

void AppendJsonValueText(const JsonoView &view, size_t &pos, size_t &string_cursor, std::string &out, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, pos);
		auto val_cursor = layout.value_start;
		out.push_back('{');
		for (size_t i = 0; i < layout.key_count; i++) {
			if (i) {
				out.push_back(',');
			}
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: expected KEY slot");
			}
			AppendJsonString(view.KeyAt(SlotPayload(key_slot)), out);
			out.push_back(':');
			AppendJsonValueText(view, val_cursor, string_cursor, out, depth + 1);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		out.push_back('}');
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, pos);
		pos++;
		out.push_back('[');
		bool first = true;
		while (pos < end_pos) {
			if (!first) {
				out.push_back(',');
			}
			first = false;
			AppendJsonValueText(view, pos, string_cursor, out, depth + 1);
		}
		pos = end_pos + 1;
		out.push_back(']');
		return;
	}
	default: {
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		if (scalar.kind == JsonoScalarKind::String) {
			AppendJsonString(scalar.text, out);
			return;
		}
		if (AppendExtractStringScalar(scalar, out)) {
			out.append("null");
		}
		return;
	}
	}
}

bool LocatedValueMatches(JsonoPathLocalState &lstate, const vector<string> &values, const JsonoView &view,
                         const JsonoMatchLocation &location) {
	auto slot_tag = SlotTag(view.SlotAt(location.pos));
	if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
		lstate.scratch.clear();
		auto pos = location.pos;
		auto string_cursor = location.string_cursor;
		AppendJsonValueText(view, pos, string_cursor, lstate.scratch, 0);
		return ContainsExpectedValue(values, nonstd::string_view(lstate.scratch.data(), lstate.scratch.size()));
	}
	auto pos = location.pos;
	auto string_cursor = location.string_cursor;
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	if (scalar.kind == JsonoScalarKind::Null) {
		return false;
	}
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		return ContainsExpectedValue(values, scalar.text);
	}
	lstate.scratch.clear();
	if (AppendExtractStringScalar(scalar, lstate.scratch)) {
		return false;
	}
	return ContainsExpectedValue(values, nonstd::string_view(lstate.scratch.data(), lstate.scratch.size()));
}

bool PredicateMatches(JsonoPathLocalState &lstate, const JsonoMatchPredicate &predicate, const JsonoView &view) {
	JsonoMatchLocation location;
	if (!LocatePath(lstate, predicate, view, location)) {
		return false;
	}
	return LocatedValueMatches(lstate, predicate.values, view, location);
}

void WriteLocatedExtractString(JsonoPathLocalState &lstate, Vector &result, idx_t row, const JsonoView &view,
                               const JsonoMatchLocation &location) {
	auto result_data = FlatVector::GetData<string_t>(result);
	auto slot_tag = SlotTag(view.SlotAt(location.pos));
	if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
		lstate.scratch.clear();
		auto pos = location.pos;
		auto string_cursor = location.string_cursor;
		AppendJsonValueText(view, pos, string_cursor, lstate.scratch, 0);
		FlatVector::Validity(result).SetValid(row);
		result_data[row] = StringVector::AddString(result, lstate.scratch.data(), lstate.scratch.size());
		return;
	}

	auto pos = location.pos;
	auto string_cursor = location.string_cursor;
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		FlatVector::Validity(result).SetValid(row);
		result_data[row] = StringVector::AddString(result, scalar.text.data(), scalar.text.size());
		return;
	}
	lstate.scratch.clear();
	if (AppendExtractStringScalar(scalar, lstate.scratch)) {
		FlatVector::SetNull(result, row, true);
		return;
	}
	FlatVector::Validity(result).SetValid(row);
	result_data[row] = StringVector::AddString(result, lstate.scratch.data(), lstate.scratch.size());
}

enum class JsonoExtremaKind : uint8_t { Min, Max };

struct JsonoExtremaAggregateBindData : public FunctionData {
	JsonoExtremaKind kind;

	explicit JsonoExtremaAggregateBindData(JsonoExtremaKind kind_p) : kind(kind_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoExtremaAggregateBindData>(kind);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoExtremaAggregateBindData>();
		return kind == other.kind;
	}
};

struct JsonoExtractExtremaState {
	bool has_value = false;
	bool numeric_mode = true;
	uint64_t numeric_value = 0;
	uint32_t numeric_width = 0;
	string_t text_value;
};

struct JsonoExtractExtremaFunction {
	static void Initialize(JsonoExtractExtremaState &state) {
		state.has_value = false;
		state.numeric_mode = true;
		state.numeric_value = 0;
		state.numeric_width = 0;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		DestroyText(state);
		state.has_value = false;
	}

	static void DestroyText(JsonoExtractExtremaState &state) {
		if (!state.numeric_mode && state.has_value && !state.text_value.IsInlined()) {
			delete[] state.text_value.GetData();
		}
	}
};

bool ExtremaStringCandidateWins(JsonoExtremaKind kind, nonstd::string_view current, nonstd::string_view candidate) {
	if (kind == JsonoExtremaKind::Min) {
		return candidate < current;
	}
	return candidate > current;
}

bool ExtremaNumericCandidateWins(JsonoExtremaKind kind, uint64_t current, uint64_t candidate) {
	if (kind == JsonoExtremaKind::Min) {
		return candidate < current;
	}
	return candidate > current;
}

uint32_t DecimalDigitCount(uint64_t value) {
	uint32_t width = 1;
	while (value >= 10) {
		value /= 10;
		width++;
	}
	return width;
}

bool TryParseUnsignedIntegerText(nonstd::string_view text, uint64_t &value) {
	if (text.empty()) {
		return false;
	}
	uint64_t result = 0;
	for (auto c : text) {
		if (c < '0' || c > '9') {
			return false;
		}
		auto digit = uint64_t(c - '0');
		if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
			return false;
		}
		result = result * 10 + digit;
	}
	value = result;
	return true;
}

bool TryParseCanonicalUnsignedIntegerText(nonstd::string_view text, uint64_t &value) {
	if (text.size() > 1 && text[0] == '0') {
		return false;
	}
	return TryParseUnsignedIntegerText(text, value);
}

void EnsureExtremaStringMode(JsonoExtractExtremaState &state) {
	if (!state.numeric_mode) {
		return;
	}
	auto rendered = std::to_string(state.numeric_value);
	state.text_value = string_t(rendered.data(), UnsafeNumericCast<uint32_t>(rendered.size()));
	state.numeric_mode = false;
}

void AssignExtremaText(JsonoExtractExtremaState &state, nonstd::string_view candidate) {
	JsonoExtractExtremaFunction::DestroyText(state);
	if (candidate.size() <= string_t::INLINE_LENGTH) {
		state.text_value = string_t(candidate.data(), UnsafeNumericCast<uint32_t>(candidate.size()));
		return;
	}
	auto owned = new char[candidate.size()];
	memcpy(owned, candidate.data(), candidate.size());
	state.text_value = string_t(owned, UnsafeNumericCast<uint32_t>(candidate.size()));
}

void FoldStringExtrema(JsonoExtractExtremaState &state, JsonoExtremaKind kind, nonstd::string_view candidate) {
	if (!state.has_value) {
		state.has_value = true;
		state.numeric_mode = false;
		AssignExtremaText(state, candidate);
		return;
	}
	EnsureExtremaStringMode(state);
	nonstd::string_view current(state.text_value.GetData(), state.text_value.GetSize());
	if (ExtremaStringCandidateWins(kind, current, candidate)) {
		AssignExtremaText(state, candidate);
	}
}

void FoldNumericExtrema(JsonoExtractExtremaState &state, JsonoExtremaKind kind, uint64_t candidate,
                        uint32_t candidate_width) {
	if (!state.has_value) {
		state.has_value = true;
		state.numeric_mode = true;
		state.numeric_value = candidate;
		state.numeric_width = candidate_width;
		return;
	}
	if (state.numeric_mode && state.numeric_width == candidate_width) {
		if (ExtremaNumericCandidateWins(kind, state.numeric_value, candidate)) {
			state.numeric_value = candidate;
		}
		return;
	}
	auto candidate_text = std::to_string(candidate);
	FoldStringExtrema(state, kind, nonstd::string_view(candidate_text.data(), candidate_text.size()));
}

void FoldVarcharExtrema(JsonoExtractExtremaState &state, JsonoExtremaKind kind, nonstd::string_view candidate) {
	uint64_t numeric_value = 0;
	if (state.numeric_mode && TryParseCanonicalUnsignedIntegerText(candidate, numeric_value)) {
		FoldNumericExtrema(state, kind, numeric_value, uint32_t(candidate.size()));
		return;
	}
	FoldStringExtrema(state, kind, candidate);
}

void FoldStateExtrema(JsonoExtractExtremaState &target, const JsonoExtractExtremaState &source, JsonoExtremaKind kind) {
	if (!source.has_value) {
		return;
	}
	if (source.numeric_mode) {
		FoldNumericExtrema(target, kind, source.numeric_value, source.numeric_width);
		return;
	}
	FoldStringExtrema(target, kind, nonstd::string_view(source.text_value.GetData(), source.text_value.GetSize()));
}

void JsonoExtractExtremaCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<JsonoExtremaAggregateBindData>();
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<JsonoExtractExtremaState *>(source_fmt);
	auto target_data = FlatVector::GetData<JsonoExtractExtremaState *>(target);

	for (idx_t row = 0; row < count; row++) {
		auto &source_state = *source_data[RowIndex(source_fmt, row)];
		FoldStateExtrema(*target_data[row], source_state, bind_data.kind);
	}
}

void JsonoExtractExtremaFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                 idx_t offset) {
	(void)aggr_input_data;
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<JsonoExtractExtremaState *>(state_fmt);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto rid = i + offset;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.has_value) {
			FlatVector::SetNull(result, rid, true);
			continue;
		}
		FlatVector::Validity(result).SetValid(rid);
		if (state.numeric_mode) {
			auto rendered = std::to_string(state.numeric_value);
			result_data[rid] = StringVector::AddString(result, rendered.data(), rendered.size());
		} else {
			result_data[rid] = StringVector::AddString(result, state.text_value.GetData(), state.text_value.GetSize());
		}
	}
}

void JsonoVarcharExtremaSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                     data_ptr_t state_ptr, idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<JsonoExtremaAggregateBindData>();
	UnifiedVectorFormat input_fmt;
	inputs[0].ToUnifiedFormat(count, input_fmt);
	auto input_data = UnifiedVectorFormat::GetData<string_t>(input_fmt);
	auto &state = *reinterpret_cast<JsonoExtractExtremaState *>(state_ptr);
	for (idx_t row = 0; row < count; row++) {
		auto input_idx = RowIndex(input_fmt, row);
		if (!input_fmt.validity.RowIsValid(input_idx)) {
			continue;
		}
		auto value = input_data[input_idx];
		FoldVarcharExtrema(state, bind_data.kind, nonstd::string_view(value.GetData(), value.GetSize()));
	}
}

void JsonoVarcharExtremaUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                               idx_t count) {
	(void)input_count;
	auto &bind_data = aggr_input_data.bind_data->Cast<JsonoExtremaAggregateBindData>();
	UnifiedVectorFormat input_fmt;
	inputs[0].ToUnifiedFormat(count, input_fmt);
	auto input_data = UnifiedVectorFormat::GetData<string_t>(input_fmt);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<JsonoExtractExtremaState *>(state_fmt);

	for (idx_t row = 0; row < count; row++) {
		auto input_idx = RowIndex(input_fmt, row);
		if (!input_fmt.validity.RowIsValid(input_idx)) {
			continue;
		}
		auto state_idx = RowIndex(state_fmt, row);
		auto value = input_data[input_idx];
		FoldVarcharExtrema(*state_data[state_idx], bind_data.kind,
		                   nonstd::string_view(value.GetData(), value.GetSize()));
	}
}

// The optimizer injects these helpers fully bound: they are query-local artifacts
// that carry their bind data inline and are never reconstructed from the catalog.
// Declaring them non-serializable makes DuckDB's debug plan verification
// (LogicalOperator::Verify, and the verify_serializer round-trip) skip them via the
// NotImplementedException path instead of failing a catalog lookup for an
// unregistered name. They genuinely cannot round-trip, so refusing is honest.
template <class FUNC>
void JsonoInternalSerializeUnsupported(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                                       const FUNC &function) {
	throw NotImplementedException("JSONO optimizer-internal function '%s' cannot be serialized", function.name);
}

template <class FUNC>
unique_ptr<FunctionData> JsonoInternalDeserializeUnsupported(Deserializer &deserializer, FUNC &function) {
	throw NotImplementedException("JSONO optimizer-internal function '%s' cannot be deserialized", function.name);
}

AggregateFunction MakeJsonoVarcharExtremaAggregate(JsonoExtremaKind kind) {
	auto name =
	    kind == JsonoExtremaKind::Min ? "__jsono_internal_min_varchar_fast" : "__jsono_internal_max_varchar_fast";
	AggregateFunction fun(name, {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                      AggregateFunction::StateSize<JsonoExtractExtremaState>,
	                      AggregateFunction::StateInitialize<JsonoExtractExtremaState, JsonoExtractExtremaFunction>,
	                      JsonoVarcharExtremaUpdate, JsonoExtractExtremaCombine, JsonoExtractExtremaFinalize,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, JsonoVarcharExtremaSimpleUpdate, nullptr,
	                      AggregateFunction::StateDestroy<JsonoExtractExtremaState, JsonoExtractExtremaFunction>);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	fun.distinct_dependent = AggregateDistinctDependent::NOT_DISTINCT_DEPENDENT;
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	fun.SetSerializeCallback(JsonoInternalSerializeUnsupported<AggregateFunction>);
	fun.SetDeserializeCallback(JsonoInternalDeserializeUnsupported<AggregateFunction>);
	return fun;
}

void SetProjectRowNull(Vector &result, vector<unique_ptr<Vector>> &children, idx_t row) {
	FlatVector::SetNull(result, row, true);
	for (auto &child : children) {
		FlatVector::SetNull(*child, row, true);
	}
}

void JsonoInternalProjectExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<JsonoProjectBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoPathLocalState>();
	auto count = args.size();

	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}

	for (idx_t row = 0; row < count; row++) {
		lstate.layout_generation++;
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetProjectRowNull(result, children, row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetProjectRowNull(result, children, row);
			continue;
		}
		FlatVector::Validity(result).SetValid(row);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			JsonoMatchLocation location;
			if (!LocateProjectField(lstate, field, view, location)) {
				FlatVector::SetNull(*children[field_index], row, true);
				continue;
			}
			WriteLocatedExtractString(lstate, *children[field_index], row, view, location);
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

ScalarFunction MakeJsonoInternalProjectFunction(const LogicalType &return_type) {
	ScalarFunction fun("__jsono_internal_project", {JsonoType()}, return_type, JsonoInternalProjectExecute, nullptr,
	                   nullptr, nullptr, JsonoPathLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	fun.SetSerializeCallback(JsonoInternalSerializeUnsupported<ScalarFunction>);
	fun.SetDeserializeCallback(JsonoInternalDeserializeUnsupported<ScalarFunction>);
	return fun;
}

idx_t ConservativePredicateRank(const JsonoMatchPredicate &predicate) {
	auto root_path = predicate.steps.size() <= 1;
	auto multi_value = predicate.values.size() > 1;
	if (!root_path && !multi_value) {
		return 0;
	}
	if (!root_path) {
		return 1;
	}
	if (!multi_value) {
		return 2;
	}
	return 3;
}

void ApplyConservativePredicateOrder(vector<JsonoMatchPredicate> &predicates) {
	if (predicates.size() < 3) {
		return;
	}
	std::stable_sort(predicates.begin(), predicates.end(),
	                 [](const JsonoMatchPredicate &left, const JsonoMatchPredicate &right) {
		                 auto left_rank = ConservativePredicateRank(left);
		                 auto right_rank = ConservativePredicateRank(right);
		                 if (left_rank != right_rank) {
			                 return left_rank < right_rank;
		                 }
		                 if (left.steps.size() != right.steps.size()) {
			                 return left.steps.size() > right.steps.size();
		                 }
		                 return left.path < right.path;
	                 });
}

void JsonoInternalMatchExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<JsonoMatchBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoPathLocalState>();
	auto count = args.size();

	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);
	for (idx_t row = 0; row < count; row++) {
		lstate.layout_generation++;
		result_data[row] = false;
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			continue;
		}
		auto matches = true;
		for (auto &predicate : bind_data.predicates) {
			if (!PredicateMatches(lstate, predicate, view)) {
				matches = false;
				break;
			}
		}
		result_data[row] = matches;
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

ScalarFunction MakeJsonoInternalMatchFunction() {
	ScalarFunction fun("__jsono_internal_match", {JsonoType()}, LogicalType::BOOLEAN, JsonoInternalMatchExecute,
	                   nullptr, nullptr, nullptr, JsonoPathLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	fun.SetSerializeCallback(JsonoInternalSerializeUnsupported<ScalarFunction>);
	fun.SetDeserializeCallback(JsonoInternalDeserializeUnsupported<ScalarFunction>);
	return fun;
}

unique_ptr<Expression> MakeJsonoInternalMatchExpression(const JsonoRewriteGroup &group) {
	auto bind_data = make_uniq<JsonoMatchBindData>();
	auto predicates = group.predicates;
	ApplyConservativePredicateOrder(predicates);
	idx_t step_base = 0;
	for (auto predicate : predicates) {
		predicate.step_base = step_base;
		step_base += predicate.steps.size();
		bind_data->predicates.push_back(std::move(predicate));
	}
	vector<unique_ptr<Expression>> children;
	children.push_back(make_uniq<BoundColumnRefExpression>(group.column_alias, group.column_type, group.binding));
	return make_uniq<BoundFunctionExpression>(LogicalType::BOOLEAN, MakeJsonoInternalMatchFunction(),
	                                          std::move(children), std::move(bind_data));
}

idx_t FindRewriteGroup(vector<JsonoRewriteGroup> &groups, const ColumnBinding &binding) {
	for (idx_t group_index = 0; group_index < groups.size(); group_index++) {
		if (groups[group_index].binding == binding) {
			return group_index;
		}
	}
	return DConstants::INVALID_INDEX;
}

void RewriteFilter(ClientContext &context, LogicalFilter &filter) {
	vector<JsonoRewriteGroup> groups;
	for (idx_t expression_index = 0; expression_index < filter.expressions.size(); expression_index++) {
		JsonoRewritePredicate predicate;
		if (!TryReadPredicate(context, *filter.expressions[expression_index], predicate)) {
			continue;
		}
		predicate.expression_index = expression_index;
		auto group_index = FindRewriteGroup(groups, predicate.binding);
		if (group_index == DConstants::INVALID_INDEX) {
			JsonoRewriteGroup group;
			group.binding = predicate.binding;
			group.column_type = predicate.column_type;
			group.column_alias = predicate.column_alias;
			groups.push_back(std::move(group));
			group_index = groups.size() - 1;
		}
		groups[group_index].predicates.push_back(std::move(predicate.predicate));
		groups[group_index].expression_indexes.push_back(expression_index);
	}

	vector<reference<JsonoRewriteGroup>> rewrite_groups;
	vector<bool> replaced(filter.expressions.size(), false);
	for (auto &group : groups) {
		if (group.predicates.size() < 2) {
			continue;
		}
		rewrite_groups.push_back(group);
		for (auto expression_index : group.expression_indexes) {
			replaced[expression_index] = true;
		}
	}
	if (rewrite_groups.empty()) {
		return;
	}

	vector<unique_ptr<Expression>> expressions;
	for (idx_t expression_index = 0; expression_index < filter.expressions.size(); expression_index++) {
		if (!replaced[expression_index]) {
			expressions.push_back(std::move(filter.expressions[expression_index]));
		}
	}
	for (auto &group : rewrite_groups) {
		expressions.push_back(MakeJsonoInternalMatchExpression(group.get()));
	}
	filter.expressions = std::move(expressions);
}

bool TryReadInternalMatchSource(Expression &expr, ColumnBinding &binding, LogicalType &type, string &alias,
                                idx_t &predicate_count) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	if (function.function.name != "__jsono_internal_match" || function.children.size() != 1 ||
	    function.children[0]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return false;
	}
	auto &column_ref = function.children[0]->Cast<BoundColumnRefExpression>();
	auto &bind_data = function.bind_info->Cast<JsonoMatchBindData>();
	binding = column_ref.binding;
	type = column_ref.return_type;
	alias = column_ref.GetAlias();
	predicate_count = bind_data.predicates.size();
	return true;
}

bool TryReadProjectField(ClientContext &context, Expression &expr, const ColumnBinding &source_binding,
                         JsonoProjectField &field) {
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate predicate;
	if (!TryReadJsonoExtract(context, expr, column_ref, predicate)) {
		return false;
	}
	if (column_ref->binding != source_binding) {
		return false;
	}
	field.path = std::move(predicate.path);
	field.steps = std::move(predicate.steps);
	field.compiled_path = std::move(predicate.compiled_path);
	return true;
}

bool ProjectPathEquals(const vector<PathStep> &left, const vector<PathStep> &right) {
	return PathStepsEqual(left, right);
}

idx_t FindProjectField(const vector<JsonoProjectField> &fields, const vector<PathStep> &steps) {
	for (idx_t field_index = 0; field_index < fields.size(); field_index++) {
		if (ProjectPathEquals(fields[field_index].steps, steps)) {
			return field_index;
		}
	}
	return DConstants::INVALID_INDEX;
}

void CollectProjectFields(ClientContext &context, Expression &expr, const ColumnBinding &source_binding,
                          vector<JsonoProjectField> &fields) {
	JsonoProjectField field;
	if (TryReadProjectField(context, expr, source_binding, field) &&
	    FindProjectField(fields, field.steps) == DConstants::INVALID_INDEX) {
		field.name = "p" + std::to_string(fields.size());
		fields.push_back(std::move(field));
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { CollectProjectFields(context, child, source_binding, fields); });
}

idx_t FindBindingIndex(const vector<ColumnBinding> &bindings, const ColumnBinding &binding) {
	for (idx_t binding_index = 0; binding_index < bindings.size(); binding_index++) {
		if (bindings[binding_index] == binding) {
			return binding_index;
		}
	}
	return DConstants::INVALID_INDEX;
}

bool TryRewriteExtractExtremaAggregate(ClientContext &context, unique_ptr<Expression> &expr) {
	if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
		return false;
	}
	auto &aggregate = expr->Cast<BoundAggregateExpression>();
	if (aggregate.children.size() != 1 || aggregate.IsDistinct() || aggregate.filter || aggregate.order_bys) {
		return false;
	}
	JsonoExtremaKind kind;
	if (aggregate.function.name == "min") {
		kind = JsonoExtremaKind::Min;
	} else if (aggregate.function.name == "max") {
		kind = JsonoExtremaKind::Max;
	} else {
		return false;
	}
	if (aggregate.return_type.id() != LogicalTypeId::VARCHAR) {
		return false;
	}
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate predicate;
	if (!TryReadJsonoExtract(context, *aggregate.children[0], column_ref, predicate)) {
		return false;
	}

	auto alias = expr->GetAlias();
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(aggregate.children[0]));
	auto replacement = make_uniq<BoundAggregateExpression>(MakeJsonoVarcharExtremaAggregate(kind), std::move(children),
	                                                       nullptr, make_uniq<JsonoExtremaAggregateBindData>(kind),
	                                                       AggregateType::NON_DISTINCT);
	replacement->SetAlias(std::move(alias));
	expr = std::move(replacement);
	return true;
}

bool RewriteExtractExtremaAggregates(ClientContext &context, LogicalAggregate &aggregate) {
	auto rewritten = false;
	for (auto &expr : aggregate.expressions) {
		if (TryRewriteExtractExtremaAggregate(context, expr)) {
			rewritten = true;
		}
	}
	if (rewritten) {
		aggregate.ResolveOperatorTypes();
	}
	return rewritten;
}

struct JsonoProjectRewriteState {
	ClientContext &context;
	ColumnBinding source_binding;
	LogicalType source_type;
	string source_alias;
	vector<ColumnBinding> child_bindings;
	vector<LogicalType> child_types;
	vector<JsonoProjectField> fields;
	LogicalType struct_type;
	ColumnBinding struct_binding;
	idx_t projection_index;
};

unique_ptr<Expression> MakeStructExtractAtExpression(JsonoProjectRewriteState &state, idx_t field_index,
                                                     const string &alias) {
	vector<unique_ptr<Expression>> children;
	children.push_back(
	    make_uniq<BoundColumnRefExpression>("__jsono_internal_project", state.struct_type, state.struct_binding));
	children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(int64_t(field_index + 1))));
	FunctionBinder function_binder(state.context);
	auto result = function_binder.BindScalarFunction(StructExtractAtFun::GetFunction(), std::move(children));
	result->alias = alias;
	return result;
}

void RewriteProjectExpression(unique_ptr<Expression> &expr, JsonoProjectRewriteState &state) {
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate predicate;
	if (TryReadJsonoExtract(state.context, *expr, column_ref, predicate) &&
	    column_ref->binding == state.source_binding) {
		auto field_index = FindProjectField(state.fields, predicate.steps);
		if (field_index == DConstants::INVALID_INDEX) {
			throw InternalException("JSONO projector rewrite missing collected field for path '%s'", predicate.path);
		}
		auto alias = expr->GetAlias();
		expr = MakeStructExtractAtExpression(state, field_index, alias);
		return;
	}
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &bound_column_ref = expr->Cast<BoundColumnRefExpression>();
		auto binding_index = FindBindingIndex(state.child_bindings, bound_column_ref.binding);
		if (binding_index != DConstants::INVALID_INDEX) {
			bound_column_ref.binding = ColumnBinding(state.projection_index, binding_index);
			bound_column_ref.return_type = state.child_types[binding_index];
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { RewriteProjectExpression(child, state); });
}

unique_ptr<Expression> MakeJsonoInternalProjectExpression(JsonoProjectRewriteState &state) {
	auto bind_data = make_uniq<JsonoProjectBindData>();
	idx_t step_base = 0;
	for (auto field : state.fields) {
		field.step_base = step_base;
		step_base += field.steps.size();
		bind_data->fields.push_back(std::move(field));
	}
	child_list_t<LogicalType> child_types;
	for (auto &field : bind_data->fields) {
		child_types.push_back(make_pair(field.name, LogicalType::VARCHAR));
	}
	state.struct_type = LogicalType::STRUCT(child_types);

	vector<unique_ptr<Expression>> children;
	children.push_back(
	    make_uniq<BoundColumnRefExpression>(state.source_alias, state.source_type, state.source_binding));
	return make_uniq<BoundFunctionExpression>(state.struct_type, MakeJsonoInternalProjectFunction(state.struct_type),
	                                          std::move(children), std::move(bind_data));
}

// Insert one __jsono_internal_project STRUCT projection between `parent` and its child
// LOGICAL_FILTER (which must carry a >=2-predicate matcher), then rewrite every JSONO `->>`
// in the parent's expression lists to a cheap struct_extract over that single shared pass.
// The struct sits above the filter, so it materializes only surviving rows, and the shared
// header-parse/root-layout read is paid once per row instead of once per extracted path.
// Works for any parent that reads several constant `->>` paths off the same filtered column:
// a grouped aggregate (groups + aggregate args) or a plain projection (select/order/window).
bool RewriteJsonoProjector(OptimizerExtensionInput &input, LogicalOperator &parent,
                           const vector<vector<unique_ptr<Expression>> *> &expression_lists) {
	if (parent.children.size() != 1 || parent.children[0]->type != LogicalOperatorType::LOGICAL_FILTER) {
		return false;
	}
	auto &filter = parent.children[0]->Cast<LogicalFilter>();
	ColumnBinding source_binding;
	LogicalType source_type;
	string source_alias;
	idx_t predicate_count = 0;
	auto found_matcher = false;
	for (auto &expression : filter.expressions) {
		if (TryReadInternalMatchSource(*expression, source_binding, source_type, source_alias, predicate_count) &&
		    predicate_count >= 2) {
			found_matcher = true;
			break;
		}
	}
	if (!found_matcher) {
		return false;
	}

	vector<JsonoProjectField> fields;
	for (auto *expression_list : expression_lists) {
		for (auto &expression : *expression_list) {
			CollectProjectFields(input.context, *expression, source_binding, fields);
		}
	}
	if (fields.size() < 2) {
		return false;
	}

	filter.ResolveOperatorTypes();
	JsonoProjectRewriteState state {input.context,
	                                source_binding,
	                                source_type,
	                                source_alias,
	                                filter.GetColumnBindings(),
	                                filter.types,
	                                std::move(fields),
	                                LogicalType::INVALID,
	                                ColumnBinding(),
	                                input.optimizer.binder.GenerateTableIndex()};
	if (state.child_bindings.size() != state.child_types.size()) {
		throw InternalException("JSONO projector rewrite saw mismatched child bindings and types");
	}

	vector<unique_ptr<Expression>> projection_expressions;
	projection_expressions.reserve(state.child_bindings.size() + 1);
	for (idx_t column_index = 0; column_index < state.child_bindings.size(); column_index++) {
		projection_expressions.push_back(
		    make_uniq<BoundColumnRefExpression>(state.child_types[column_index], state.child_bindings[column_index]));
	}
	state.struct_binding = ColumnBinding(state.projection_index, projection_expressions.size());
	projection_expressions.push_back(MakeJsonoInternalProjectExpression(state));

	for (auto *expression_list : expression_lists) {
		for (auto &expression : *expression_list) {
			RewriteProjectExpression(expression, state);
		}
	}

	auto projection = make_uniq<LogicalProjection>(state.projection_index, std::move(projection_expressions));
	if (filter.has_estimated_cardinality) {
		projection->SetEstimatedCardinality(filter.estimated_cardinality);
	}
	projection->children.push_back(std::move(parent.children[0]));
	projection->ResolveOperatorTypes();
	parent.children[0] = std::move(projection);
	parent.ResolveOperatorTypes();
	return true;
}

bool RewriteAggregateProjector(OptimizerExtensionInput &input, LogicalAggregate &aggregate) {
	return RewriteJsonoProjector(input, aggregate, {&aggregate.groups, &aggregate.expressions});
}

bool RewriteProjectionProjector(OptimizerExtensionInput &input, LogicalProjection &projection) {
	return RewriteJsonoProjector(input, projection, {&projection.expressions});
}

//===--------------------------------------------------------------------===//
// Shredded JSONO transparency
//
// A shredded JSONO column reaches the binder as a plain STRUCT (four BLOB prefix
// followed by VARCHAR lane columns named by canonical path). No implicit cast turns
// that struct into JSONO, so a bare `j->>'path'` / `to_json(j)` binds to core json's
// STRUCT->JSON path, which serializes the raw struct (wrong: leaks the blobs, never
// reads the value). This pre-optimize pass rewrites those bound expressions off the
// lane manifest carried in the column type itself:
//   - `->>`/json_extract_string on a lane path -> struct_extract of the lane column.
//     The lane already holds the extracted text, so this skips the JSONO parse and
//     lets projection/row-group pruning act on the lane column directly.
//   - any CAST(shredded->JSON) and to_json(shredded) -> reinterpret(col->JSONO)::JSON,
//     which reconstructs the authoritative value from the four-blob prefix. Because a
//     non-lane extract's inner cast is rewritten too, correctness never depends on a
//     path being shredded; lanes are purely a fast path.
//===--------------------------------------------------------------------===//

bool IsJsonTextType(const LogicalType &type) {
	return type == LogicalType::JSON();
}

struct JsonoLane {
	idx_t child_index;
	LogicalType type;
	vector<PathStep> steps;
};

// Trailing children past the four-blob prefix are the lanes, each named by its
// canonical path. A lane may be typed (a hot number stored as BIGINT/DOUBLE), so
// the type is carried: the `->>` text contract reads the lane and casts it to
// VARCHAR, while reconstruction injects it back at its JSON type.
vector<JsonoLane> CollectShreddedLanes(const LogicalType &shredded) {
	vector<JsonoLane> lanes;
	auto &children = StructType::GetChildTypes(shredded);
	for (idx_t i = StructType::GetChildTypes(JsonoRawStructType()).size(); i < children.size(); i++) {
		auto &name = children[i].first;
		vector<PathStep> steps;
		if (!name.empty() && name[0] == '$') {
			steps = ParseJsonoPath(name, "__jsono_shredded_lane");
		} else {
			steps = LiteralKeyPath(name);
		}
		lanes.push_back(JsonoLane {i, children[i].second, std::move(steps)});
	}
	return lanes;
}

// A node in the lane reconstruction patch. Lanes are inserted by their path so the patch
// rebuilds the original nesting as a struct_pack tree; a leaf (no children) carries the
// lane's struct child index, a group carries nested keys.
struct LanePatchNode {
	idx_t child_index = 0;
	vector<pair<string, LanePatchNode>> children;

	LanePatchNode &Child(const string &key) {
		for (auto &entry : children) {
			if (entry.first == key) {
				return entry.second;
			}
		}
		children.emplace_back(key, LanePatchNode {});
		return children.back().second;
	}
};

class JsonoShreddedRewriter : public LogicalOperatorVisitor {
public:
	explicit JsonoShreddedRewriter(ClientContext &context) : context(context) {
	}

protected:
	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		(void)expr_ptr;
		// Whole-value conversion: reconstruct the full value from residual + lanes.
		if ((expr.function.name == "to_json" || expr.function.name == "json_quote") && expr.children.size() == 1 &&
		    IsShreddedJsonoType(expr.children[0]->return_type)) {
			auto alias = expr.GetAlias();
			auto rewritten = ReconstructShreddedToJson(std::move(expr.children[0]));
			rewritten->SetAlias(alias);
			return rewritten;
		}
		// Path extraction off a shredded value.
		if (IsExtractFunction(expr.function.name) && expr.children.size() == 2) {
			auto shredded_cast = ShreddedJsonCast(*expr.children[0]);
			if (shredded_cast) {
				return RewriteShreddedExtract(expr, *shredded_cast);
			}
		}
		return nullptr;
	}

	unique_ptr<Expression> VisitReplace(BoundCastExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		(void)expr_ptr;
		// Fold CAST(lane string extract AS T) where the lane is itself typed T: read the
		// typed lane directly, skipping the BIGINT->VARCHAR->BIGINT round-trip and giving
		// the planner the lane's typed column statistics.
		auto folded = TryFoldTypedLaneCast(expr);
		if (folded) {
			return folded;
		}
		// Whole-value -> JSON: reconstruct from residual + lanes.
		if (IsJsonTextType(expr.return_type) && IsShreddedJsonoType(expr.child->return_type)) {
			auto alias = expr.GetAlias();
			auto rewritten = ReconstructShreddedToJson(std::move(expr.child));
			rewritten->SetAlias(alias);
			return rewritten;
		}
		return nullptr;
	}

private:
	static bool IsExtractFunction(const string &name) {
		return name == "->>" || name == "json_extract_string" || name == "->" || name == "json_extract";
	}

	// The extract argument is `CAST(shredded->JSON)` from core json's STRUCT->JSON
	// path; return that cast when its child is a shredded JSONO struct.
	optional_ptr<BoundCastExpression> ShreddedJsonCast(Expression &arg) {
		if (arg.GetExpressionClass() != ExpressionClass::BOUND_CAST) {
			return nullptr;
		}
		auto &cast = arg.Cast<BoundCastExpression>();
		if (IsJsonTextType(cast.return_type) && IsShreddedJsonoType(cast.child->return_type)) {
			return &cast;
		}
		return nullptr;
	}

	// A constant-path read off a shredded value. A string extract of a lane reads the lane
	// column directly. A non-lane path reads the residual natively (cheap): only lane leaves
	// are stripped, so a non-lane path keeps its own value there. Two cases reconstruct
	// instead: a json-valued extract of a lane (its value may be stripped from the residual),
	// and a json-valued extract of a subtree that contains a stripped lane leaf.
	unique_ptr<Expression> RewriteShreddedExtract(BoundFunctionExpression &expr, BoundCastExpression &shredded_cast) {
		bool string_fn = expr.function.name == "->>" || expr.function.name == "json_extract_string";
		string path;
		vector<PathStep> steps;
		if (TryReadExtractPath(context, *expr.children[1], path, steps)) {
			for (auto &lane : CollectShreddedLanes(shredded_cast.child->return_type)) {
				if (PathStepsEqual(lane.steps, steps)) {
					if (string_fn) {
						auto alias = expr.GetAlias();
						return MakeLaneRead(std::move(shredded_cast.child), lane, std::move(expr.children[1]), alias);
					}
					return nullptr; // json-valued lane extract: reconstruct via the inner cast
				}
				// A json-valued read of a subtree holding a stripped lane leaf would miss it
				// in the residual; reconstruct. (`->>` of an object is NULL regardless, and a
				// stripped scalar leaf is itself a lane, handled by the exact match above.)
				if (!string_fn && PathStepsObjectKeyPrefix(steps, lane.steps)) {
					return nullptr;
				}
			}
		}
		return MakeResidualExtract(expr, shredded_cast, string_fn);
	}

	// Reinterpret only the four-BLOB residual prefix of a shredded column to JSONO, bypassing
	// the public shredded->JSONO cast (which now reconstructs the full value losslessly).
	// struct_pack of the four blobs is an exactly-four-blob struct, so casting it to JSONO
	// stays a zero-copy reinterpret rather than a reconstruction.
	unique_ptr<Expression> ResidualReinterpret(const Expression &column) {
		FunctionBinder function_binder(context);
		auto &child_types = StructType::GetChildTypes(column.return_type);
		vector<unique_ptr<Expression>> pack_children;
		for (idx_t i = 0; i < 4; i++) {
			vector<unique_ptr<Expression>> se_children;
			se_children.push_back(column.Copy());
			se_children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(int64_t(i + 1))));
			auto extracted =
			    function_binder.BindScalarFunction(StructExtractAtFun::GetFunction(), std::move(se_children));
			extracted->SetAlias(child_types[i].first);
			pack_children.push_back(std::move(extracted));
		}
		auto packed = function_binder.BindScalarFunction(StructPackFun::GetFunction(), std::move(pack_children));
		return BoundCastExpression::AddCastToType(context, std::move(packed), JsonoType());
	}

	// Extract a non-lane path natively from the residual JSONO, matching a
	// non-shredded column's extract cost — no serialize/parse round-trip through core
	// json. Correct because only lane leaves are stripped: a non-lane scalar keeps its
	// value, and a json-valued read overlapping a stripped lane already reconstructed.
	unique_ptr<Expression> MakeResidualExtract(BoundFunctionExpression &expr, BoundCastExpression &shredded_cast,
	                                           bool string_fn) {
		auto path_type = expr.children[1]->return_type;
		auto residual = ResidualReinterpret(*shredded_cast.child);
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(residual));
		children.push_back(std::move(expr.children[1]));
		auto alias = expr.GetAlias();
		FunctionBinder function_binder(context);
		auto lane_path_type = path_type.IsIntegral() ? LogicalType::BIGINT : LogicalType::VARCHAR;
		if (string_fn) {
			auto result =
			    function_binder.BindScalarFunction(JsonoExtractStringFunction(lane_path_type), std::move(children));
			result->SetAlias(alias);
			return result;
		}
		// json-valued extract returns JSONO natively; cast back to the JSON the
		// original `->`/json_extract produced.
		auto extracted = function_binder.BindScalarFunction(JsonoExtractFunction(lane_path_type), std::move(children));
		auto result = BoundCastExpression::AddCastToType(context, std::move(extracted), LogicalType::JSON());
		result->SetAlias(alias);
		return result;
	}

	// Detect CAST(string-lane-extract AS T) where the lane's own type is exactly T, and
	// fold it to a direct typed read so the cast no longer round-trips through VARCHAR.
	unique_ptr<Expression> TryFoldTypedLaneCast(BoundCastExpression &expr) {
		auto &target = expr.return_type;
		if (target.id() == LogicalTypeId::VARCHAR || IsJsonTextType(target) ||
		    expr.child->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
			return nullptr;
		}
		auto &fn = expr.child->Cast<BoundFunctionExpression>();
		if (!(fn.function.name == "->>" || fn.function.name == "json_extract_string") || fn.children.size() != 2) {
			return nullptr;
		}
		auto shredded_cast = ShreddedJsonCast(*fn.children[0]);
		if (!shredded_cast) {
			return nullptr;
		}
		string path;
		vector<PathStep> steps;
		if (!TryReadExtractPath(context, *fn.children[1], path, steps)) {
			return nullptr;
		}
		for (auto &lane : CollectShreddedLanes(shredded_cast->child->return_type)) {
			if (!PathStepsEqual(lane.steps, steps) || lane.type != target) {
				continue;
			}
			return MakeTypedLaneRead(std::move(shredded_cast->child), lane, std::move(fn.children[1]), target,
			                         expr.try_cast, expr.GetAlias());
		}
		return nullptr;
	}

	// Read a typed lane directly as its own type T. The fallback re-extracts from the
	// residual as text and casts to T, matching the original CAST's try/strict mode, so
	// a value kept in the residual still resolves; COALESCE short-circuits so a
	// homogeneous lane stays a plain typed struct_extract (with column statistics).
	unique_ptr<Expression> MakeTypedLaneRead(unique_ptr<Expression> column, const JsonoLane &lane,
	                                         unique_ptr<Expression> path, const LogicalType &target, bool try_cast,
	                                         const string &alias) {
		vector<unique_ptr<Expression>> se_children;
		se_children.push_back(column->Copy());
		se_children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(int64_t(lane.child_index + 1))));
		FunctionBinder function_binder(context);
		auto lane_value = function_binder.BindScalarFunction(StructExtractAtFun::GetFunction(), std::move(se_children));
		auto path_type = path->return_type.IsIntegral() ? LogicalType::BIGINT : LogicalType::VARCHAR;
		auto residual = ResidualReinterpret(*column);
		vector<unique_ptr<Expression>> ext_children;
		ext_children.push_back(std::move(residual));
		ext_children.push_back(std::move(path));
		auto fallback_text =
		    function_binder.BindScalarFunction(JsonoExtractStringFunction(path_type), std::move(ext_children));
		auto fallback = BoundCastExpression::AddCastToType(context, std::move(fallback_text), target, try_cast);
		auto coalesce = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, target);
		coalesce->children.push_back(std::move(lane_value));
		coalesce->children.push_back(std::move(fallback));
		coalesce->SetAlias(alias);
		return coalesce;
	}

	// Read a string-extracted lane. A VARCHAR lane holds the `->>` text for every row,
	// so it reads directly. A typed lane holds only values of its type; rows whose value
	// did not fit were kept in the residual with a NULL lane, so the typed read casts the
	// lane to text and falls back to a native residual extract. COALESCE short-circuits
	// per row, so a homogeneous lane (the common case) never evaluates the fallback.
	unique_ptr<Expression> MakeLaneRead(unique_ptr<Expression> column, const JsonoLane &lane,
	                                    unique_ptr<Expression> path, const string &alias) {
		vector<unique_ptr<Expression>> se_children;
		se_children.push_back(column->Copy());
		se_children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(int64_t(lane.child_index + 1))));
		FunctionBinder function_binder(context);
		auto lane_value = function_binder.BindScalarFunction(StructExtractAtFun::GetFunction(), std::move(se_children));
		if (lane.type.id() == LogicalTypeId::VARCHAR) {
			lane_value->SetAlias(alias);
			return lane_value;
		}
		auto lane_text = BoundCastExpression::AddCastToType(context, std::move(lane_value), LogicalType::VARCHAR);
		auto path_type = path->return_type.IsIntegral() ? LogicalType::BIGINT : LogicalType::VARCHAR;
		auto residual = ResidualReinterpret(*column);
		vector<unique_ptr<Expression>> ext_children;
		ext_children.push_back(std::move(residual));
		ext_children.push_back(std::move(path));
		auto fallback =
		    function_binder.BindScalarFunction(JsonoExtractStringFunction(path_type), std::move(ext_children));
		auto coalesce = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, LogicalType::VARCHAR);
		coalesce->children.push_back(std::move(lane_text));
		coalesce->children.push_back(std::move(fallback));
		coalesce->SetAlias(alias);
		return coalesce;
	}

	// Build the lane patch as a nested struct_pack tree mirroring each lane's path: a leaf
	// reads its lane column, a group packs its nested keys. jsono_struct_constructor turns
	// it into a JSONO patch for the overlay.
	unique_ptr<Expression> BuildPatchStruct(Expression &column, const vector<pair<string, LanePatchNode>> &children) {
		FunctionBinder function_binder(context);
		vector<unique_ptr<Expression>> pack_children;
		for (auto &entry : children) {
			unique_ptr<Expression> value;
			if (entry.second.children.empty()) {
				vector<unique_ptr<Expression>> extract_children;
				extract_children.push_back(column.Copy());
				extract_children.push_back(
				    make_uniq<BoundConstantExpression>(Value::BIGINT(int64_t(entry.second.child_index + 1))));
				value =
				    function_binder.BindScalarFunction(StructExtractAtFun::GetFunction(), std::move(extract_children));
			} else {
				value = BuildPatchStruct(column, entry.second.children);
			}
			value->SetAlias(entry.first);
			pack_children.push_back(std::move(value));
		}
		return function_binder.BindScalarFunction(StructPackFun::GetFunction(), std::move(pack_children));
	}

	// Reconstruct the full JSONO value as JSON by overlaying the lanes onto the residual:
	// the residual is authoritative and each lane only fills a path the residual dropped
	// (jsono_overlay deep-merges, so nested lanes refill nested leaves). Correct for both
	// storage modes — a PRESERVE residual keeps every path so the lanes are no-ops; a
	// stripped residual is refilled from its lanes. Index/array lanes are never stripped,
	// so they stay in the residual and sit out the patch; with no object-key lanes the
	// overlay is skipped.
	unique_ptr<Expression> ReconstructShreddedToJson(unique_ptr<Expression> column) {
		auto lanes = CollectShreddedLanes(column->return_type);
		LanePatchNode root;
		for (auto &lane : lanes) {
			bool object_key = !lane.steps.empty();
			for (auto &step : lane.steps) {
				if (step.kind != PathStepKind::Key) {
					object_key = false;
					break;
				}
			}
			if (!object_key) {
				continue;
			}
			auto *node = &root;
			for (auto &step : lane.steps) {
				node = &node->Child(step.key);
			}
			node->child_index = lane.child_index;
		}
		auto residual = ResidualReinterpret(*column);
		if (root.children.empty()) {
			// No object-key lanes to merge; the residual reinterpret already carries
			// row validity, so a NULL value casts straight to a NULL JSON.
			return BoundCastExpression::AddCastToType(context, std::move(residual), LogicalType::JSON());
		}
		FunctionBinder function_binder(context);
		auto patch_struct = BuildPatchStruct(*column, root.children);
		vector<unique_ptr<Expression>> ctor_children;
		ctor_children.push_back(std::move(patch_struct));
		auto patch = function_binder.BindScalarFunction(JsonoStructConstructorFunction(), std::move(ctor_children));
		vector<unique_ptr<Expression>> merge_children;
		merge_children.push_back(std::move(residual));
		merge_children.push_back(std::move(patch));
		auto merged = function_binder.BindScalarFunction(JsonoOverlayFunction(), std::move(merge_children));
		auto merged_json = BoundCastExpression::AddCastToType(context, std::move(merged), LogicalType::JSON());
		// struct_pack builds a non-null struct even when every field is NULL, so the
		// merge would emit `{}` for a NULL value. Guard it: a NULL shredded value is NULL.
		auto is_null = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
		is_null->children.push_back(column->Copy());
		auto null_json = make_uniq<BoundConstantExpression>(Value(LogicalType::JSON()));
		return make_uniq<BoundCaseExpression>(std::move(is_null), std::move(null_json), std::move(merged_json));
	}

	ClientContext &context;
};

void RewriteShreddedJsono(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return;
	}
	JsonoShreddedRewriter rewriter(context);
	rewriter.VisitOperator(*plan);
}

void RewritePlan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return;
	}
	for (auto &child : plan->children) {
		RewritePlan(input, child);
	}
	if (plan->type == LogicalOperatorType::LOGICAL_FILTER) {
		RewriteFilter(input.context, plan->Cast<LogicalFilter>());
		return;
	}
	if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		RewriteProjectionProjector(input, plan->Cast<LogicalProjection>());
		return;
	}
	if (plan->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggregate = plan->Cast<LogicalAggregate>();
		RewriteExtractExtremaAggregates(input.context, aggregate);
		RewriteAggregateProjector(input, aggregate);
	}
}

void JsonoOptimizerPreOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	// Normalize shredded JSONO ops before the built-in pipeline so the lane
	// struct_extract is visible to projection and row-group pruning.
	RewriteShreddedJsono(input.context, plan);
	RewritePlan(input, plan);
}

void JsonoOptimizerOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	RewritePlan(input, plan);
}

} // namespace

void RegisterJsonoOptimizer(ExtensionLoader &loader) {
	OptimizerExtension extension;
	extension.pre_optimize_function = JsonoOptimizerPreOptimize;
	extension.optimize_function = JsonoOptimizerOptimize;
	OptimizerExtension::Register(loader.GetDatabaseInstance().config, std::move(extension));
}

} // namespace duckdb
