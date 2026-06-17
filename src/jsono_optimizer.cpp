#include "jsono_optimizer.hpp"
#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_shred.hpp"

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
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/struct_stats.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace duckdb {

// Exposed jsono function factories used to build the shredded reconstruction patch
// and to read non-shred paths natively off the residual.
ScalarFunction JsonoStructConstructorFunction();
ScalarFunction JsonoOverlayFunction();
ScalarFunction JsonoCheckedResidualFunction();
ScalarFunction JsonoStripManifestFunction();
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
	JsonoPathLocalState() : layout_cache(LAYOUT_CACHE_SIZE) {
	}

	RankCache rank_cache;
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
	JsonoCursor cursor;
};

struct JsonoRewritePredicate {
	// The JSONO-typed value the extract reads. A bare column for a plain JSONO column, or the
	// residual reinterpret (a CASE over __jsono_internal_checked_residual) the shredded rewrite
	// produces for a non-shred path. Both are read once per row by the fused matcher.
	optional_ptr<Expression> source;
	JsonoMatchPredicate predicate;
	idx_t expression_index = 0;
};

struct JsonoRewriteGroup {
	optional_ptr<Expression> source;
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
// residual directly. Only object-key shreds are stripped, so a non-key `full` is ignored.
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

// True if `read` descends INTO an array shred (object array or scalar array) — the shred's pure-key
// path is a strict prefix of `read` (e.g. read `$.products[0].category` for the array shred
// `$.products`, or `$.item_ids[0]` for the scalar array `$.item_ids`), whatever follows. The residual
// carries only the skeleton array — the lifted element subfields / scalars are stripped — so such a
// read cannot be served from the residual and must reconstruct.
bool ReadDescendsIntoArrayShred(const LogicalType &shred_type, const vector<PathStep> &shred_steps,
                                const vector<PathStep> &read) {
	if (!IsShredListType(shred_type) || shred_steps.size() >= read.size()) {
		return false;
	}
	for (idx_t i = 0; i < shred_steps.size(); i++) {
		if (shred_steps[i].kind != PathStepKind::Key || read[i].kind != PathStepKind::Key ||
		    shred_steps[i].key != read[i].key) {
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

// Like TryReadJsonoExtract, but captures the JSONO-typed value the extract reads as an
// arbitrary expression rather than requiring a bare column ref. The matcher groups predicates by
// this source (column-equal for plain columns, Expression::Equals for the residual reinterpret
// the shredded rewrite emits), so K predicates over one shredded column fuse into a single
// __jsono_internal_match over one residual read with one manifest check, mirroring the projector.
bool TryReadJsonoExtractSource(ClientContext &context, Expression &expr, Expression *&source,
                               JsonoMatchPredicate &predicate) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	if (function.function.name != "->>" && function.function.name != "jsono_extract_string") {
		return false;
	}
	if (function.children.size() != 2 || !IsJsonoType(function.children[0]->return_type)) {
		return false;
	}
	string path;
	vector<PathStep> steps;
	if (!TryReadExtractPath(context, *function.children[1], path, steps)) {
		return false;
	}
	source = function.children[0].get();
	predicate.path = std::move(path);
	predicate.steps = std::move(steps);
	predicate.compiled_path = CompileJsonoPath(predicate.steps);
	return true;
}

void SetPredicateSource(JsonoRewritePredicate &predicate, Expression &source, JsonoMatchPredicate match_predicate) {
	predicate.source = &source;
	predicate.predicate = std::move(match_predicate);
}

bool TryReadEqualsPredicate(ClientContext &context, BoundComparisonExpression &comparison,
                            JsonoRewritePredicate &predicate) {
	if (comparison.GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
		return false;
	}
	Expression *source = nullptr;
	JsonoMatchPredicate match_predicate;
	string value;
	if (!TryReadJsonoExtractSource(context, *comparison.left, source, match_predicate) ||
	    !TryReadStringValue(context, *comparison.right, value)) {
		source = nullptr;
		match_predicate = JsonoMatchPredicate();
		if (!TryReadJsonoExtractSource(context, *comparison.right, source, match_predicate) ||
		    !TryReadStringValue(context, *comparison.left, value)) {
			return false;
		}
	}
	match_predicate.values.push_back(std::move(value));
	SetPredicateSource(predicate, *source, std::move(match_predicate));
	return true;
}

bool TryReadInPredicate(ClientContext &context, BoundOperatorExpression &comparison, JsonoRewritePredicate &predicate) {
	if (comparison.GetExpressionType() != ExpressionType::COMPARE_IN || comparison.children.size() < 2) {
		return false;
	}
	Expression *source = nullptr;
	JsonoMatchPredicate match_predicate;
	if (!TryReadJsonoExtractSource(context, *comparison.children[0], source, match_predicate)) {
		return false;
	}
	for (idx_t child_index = 1; child_index < comparison.children.size(); child_index++) {
		string value;
		if (!TryReadStringValue(context, *comparison.children[child_index], value)) {
			return false;
		}
		match_predicate.values.push_back(std::move(value));
	}
	SetPredicateSource(predicate, *source, std::move(match_predicate));
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
	Expression *source = nullptr;
	JsonoMatchPredicate match_predicate;
	if (!TryReadJsonoExtractSource(context, *function.children[1], source, match_predicate)) {
		return false;
	}
	match_predicate.values = std::move(values);
	SetPredicateSource(predicate, *source, std::move(match_predicate));
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

bool LocateObjectKey(JsonoPathLocalState &lstate, idx_t step_index, const JsonoView &view, const string &key,
                     JsonoCursor &cursor) {
	if (cursor.pos >= view.Slots()) {
		return false;
	}
	// On a per-row cache hit the entry was stored only after a successful OBJ_START
	// read, so both the tag check and ReadObjectLayout are skipped.
	auto &entry = lstate.layout_cache[cursor.pos & (LAYOUT_CACHE_SIZE - 1)];
	if (entry.generation != lstate.layout_generation || entry.pos != cursor.pos) {
		if (SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			return false;
		}
		entry.generation = lstate.layout_generation;
		entry.pos = cursor.pos;
		entry.layout = ReadObjectLayout(view, cursor.pos, true);
	}
	const auto &layout = entry.layout;
	size_t rank = 0;
	if (!lstate.rank_cache.Find(step_index, view, layout, key, rank)) {
		return false;
	}
	MoveCursorToObjectValueRank(view, layout, rank, cursor);
	return true;
}

bool LocateOneStep(JsonoPathLocalState &lstate, idx_t step_index, const JsonoView &view, const PathStep &step,
                   JsonoCursor &cursor) {
	if (step.kind == PathStepKind::Key) {
		// Not the shared LocatePathStep: the matcher/projector resolves several paths per
		// row, so the Key step goes through the per-row ObjectLayout cache.
		return LocateObjectKey(lstate, step_index, view, step.key, cursor);
	}
	return LocatePathStep(&lstate.rank_cache, step_index, view, step, cursor);
}

bool LocateCompiledPath(JsonoPathLocalState &lstate, const JsonoCompiledPath &compiled_path, idx_t step_base,
                        const JsonoView &view, JsonoMatchLocation &location) {
	JsonoCursor cursor;
	switch (compiled_path.kind) {
	case JsonoCompiledPathKind::RootKey:
		if (!LocateObjectKey(lstate, step_base, view, compiled_path.first_key, cursor)) {
			return false;
		}
		break;
	case JsonoCompiledPathKind::RootKeyKey:
		if (!LocateObjectKey(lstate, step_base, view, compiled_path.first_key, cursor) ||
		    !LocateObjectKey(lstate, step_base + 1, view, compiled_path.second_key, cursor)) {
			return false;
		}
		break;
	case JsonoCompiledPathKind::Generic:
		return false;
	}
	location.cursor = cursor;
	return true;
}

bool LocateSteps(JsonoPathLocalState &lstate, idx_t step_base, const vector<PathStep> &steps, const JsonoView &view,
                 JsonoMatchLocation &location) {
	JsonoCursor cursor;
	for (idx_t step_index = 0; step_index < steps.size(); step_index++) {
		if (!LocateOneStep(lstate, step_base + step_index, view, steps[step_index], cursor)) {
			return false;
		}
	}
	location.cursor = cursor;
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

bool LocatedValueMatches(JsonoPathLocalState &lstate, JsonoRowReader &reader, const JsonoMatchPredicate &predicate,
                         const JsonoView &view, const JsonoMatchLocation &location) {
	auto &values = predicate.values;
	auto slot_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
		// The compare serializes the whole found container; a manifest leaf inside it would
		// silently change the text.
		reader.CheckContainerRead(view, predicate.steps);
		lstate.scratch.clear();
		auto cursor = location.cursor;
		AppendJsonValueText(view, cursor, lstate.scratch, 0);
		return ContainsExpectedValue(values, nonstd::string_view(lstate.scratch.data(), lstate.scratch.size()));
	}
	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	if (scalar.kind == JsonoScalarKind::Null) {
		return false;
	}
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		return ContainsExpectedValue(values, scalar.text);
	}
	lstate.scratch.clear();
	if (RenderExtractText(scalar, lstate.scratch)) {
		return false;
	}
	return ContainsExpectedValue(values, nonstd::string_view(lstate.scratch.data(), lstate.scratch.size()));
}

bool PredicateMatches(JsonoPathLocalState &lstate, JsonoRowReader &reader, const JsonoMatchPredicate &predicate,
                      const JsonoView &view) {
	JsonoMatchLocation location;
	if (!LocatePath(lstate, predicate, view, location)) {
		// A miss covered by the row's shred manifest must not read as a silent non-match.
		reader.CheckPathMiss(view, predicate.steps);
		return false;
	}
	return LocatedValueMatches(lstate, reader, predicate, view, location);
}

void WriteLocatedExtractString(JsonoPathLocalState &lstate, JsonoRowReader &reader, const JsonoProjectField &field,
                               Vector &result, idx_t row, const JsonoView &view, const JsonoMatchLocation &location) {
	auto result_data = FlatVector::GetData<string_t>(result);
	auto slot_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
		// `->>` serializes a found container; a manifest leaf inside it would silently drop
		// from the text.
		reader.CheckContainerRead(view, field.steps);
		lstate.scratch.clear();
		auto cursor = location.cursor;
		AppendJsonValueText(view, cursor, lstate.scratch, 0);
		FlatVector::Validity(result).SetValid(row);
		result_data[row] = StringVector::AddString(result, lstate.scratch.data(), lstate.scratch.size());
		return;
	}

	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		FlatVector::Validity(result).SetValid(row);
		result_data[row] = ZeroCopyHeapText(scalar.text);
		return;
	}
	lstate.scratch.clear();
	if (RenderExtractText(scalar, lstate.scratch)) {
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
		// State memory is raw (StateInitialize never runs a constructor), so text_value must be put
		// into a valid, unowned form here. This holds the invariant that DestroyText only ever frees a
		// buffer AssignExtremaText allocated, never an uninitialized pointer.
		state.text_value = string_t(uint32_t(0));
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

void EnsureExtremaStringMode(JsonoExtractExtremaState &state) {
	if (!state.numeric_mode) {
		return;
	}
	auto rendered = std::to_string(state.numeric_value);
	// Assign while still in numeric mode so DestroyText inside stays a no-op
	// (text_value is empty-inlined until a string is assigned); the copy must own its buffer,
	// a string_t over the local rendering would dangle for >inline-length values.
	AssignExtremaText(state, nonstd::string_view(rendered.data(), rendered.size()));
	state.numeric_mode = false;
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

	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
		// Zero-copy text values point into the input string_heap; keep its buffer alive per child.
		StringVector::AddHeapReference(*child, reader.StringHeapVector());
	}

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		lstate.layout_generation++;
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			SetProjectRowNull(result, children, row);
			continue;
		}
		FlatVector::Validity(result).SetValid(row);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			JsonoMatchLocation location;
			if (!LocateProjectField(lstate, field, view, location)) {
				// A miss covered by the row's shred manifest must not project a silent NULL.
				reader.CheckPathMiss(view, field.steps);
				FlatVector::SetNull(*children[field_index], row, true);
				continue;
			}
			WriteLocatedExtractString(lstate, reader, field, *children[field_index], row, view, location);
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

// Render a DOUBLE through jsono's own formatter — the shortest round-trippable decimal yyjson emits,
// byte-identical to to_json and a plain `->>`. A typed DOUBLE shred read by `->>` must use this, not
// DuckDB's native DOUBLE->VARCHAR cast: the cast switches to scientific notation at a different
// threshold (1e+16, 1e-07, 1e+308), which would make a double shred's `->>` text disagree with
// to_json for the same value. NULL propagates so the shred-read COALESCE falls back to a residual
// read for a row whose value stayed in the residual.
void JsonoInternalDoubleTextExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	UnifiedVectorFormat in;
	args.data[0].ToUnifiedFormat(count, in);
	auto in_data = UnifiedVectorFormat::GetData<double>(in);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &result_validity = FlatVector::Validity(result);
	std::string buf;
	for (idx_t row = 0; row < count; row++) {
		auto idx = in.sel->get_index(row);
		if (!in.validity.RowIsValid(idx)) {
			result_validity.SetInvalid(row);
			continue;
		}
		buf.clear();
		EmitDouble(in_data[idx], buf);
		result_data[row] = StringVector::AddString(result, buf.data(), buf.size());
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

ScalarFunction MakeJsonoInternalDoubleTextFunction() {
	ScalarFunction fun("__jsono_internal_double_text", {LogicalType::DOUBLE}, LogicalType::VARCHAR,
	                   JsonoInternalDoubleTextExecute);
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

	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		lstate.layout_generation++;
		result_data[row] = false;
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto matches = true;
		for (auto &predicate : bind_data.predicates) {
			if (!PredicateMatches(lstate, reader, predicate, view)) {
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
	children.push_back(group.source->Copy());
	return make_uniq<BoundFunctionExpression>(LogicalType::BOOLEAN, MakeJsonoInternalMatchFunction(),
	                                          std::move(children), std::move(bind_data));
}

idx_t FindRewriteGroup(vector<JsonoRewriteGroup> &groups, const Expression &source) {
	for (idx_t group_index = 0; group_index < groups.size(); group_index++) {
		if (groups[group_index].source->Equals(source)) {
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
		auto group_index = FindRewriteGroup(groups, *predicate.source);
		if (group_index == DConstants::INVALID_INDEX) {
			JsonoRewriteGroup group;
			group.source = predicate.source;
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

// A JSONO column read by constant-path extracts in the parent's expression lists, with the
// distinct paths collected in first-appearance order. The projector fuse fires when one
// column accumulates >=2 distinct paths.
struct JsonoProjectSource {
	ColumnBinding binding;
	LogicalType type;
	string alias;
	vector<JsonoProjectField> fields;
};

void CollectProjectSources(ClientContext &context, Expression &expr, vector<JsonoProjectSource> &sources) {
	BoundColumnRefExpression *column_ref = nullptr;
	JsonoMatchPredicate predicate;
	if (TryReadJsonoExtract(context, expr, column_ref, predicate)) {
		JsonoProjectSource *source = nullptr;
		for (auto &candidate : sources) {
			if (candidate.binding == column_ref->binding) {
				source = &candidate;
				break;
			}
		}
		if (!source) {
			sources.push_back(
			    JsonoProjectSource {column_ref->binding, column_ref->return_type, column_ref->GetAlias(), {}});
			source = &sources.back();
		}
		if (FindProjectField(source->fields, predicate.steps) == DConstants::INVALID_INDEX) {
			JsonoProjectField field;
			field.name = "p" + std::to_string(source->fields.size());
			field.path = std::move(predicate.path);
			field.steps = std::move(predicate.steps);
			field.compiled_path = std::move(predicate.compiled_path);
			source->fields.push_back(std::move(field));
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(expr,
	                                      [&](Expression &child) { CollectProjectSources(context, child, sources); });
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

// Insert one __jsono_internal_project STRUCT projection between `parent` and its child, then
// rewrite every JSONO `->>` in the parent's expression lists to a cheap struct_extract over
// that single shared pass: the shared header-parse/root-layout read is paid once per row
// instead of once per extracted path. Fires for any parent that reads >=2 distinct constant
// `->>` paths off the same JSONO column: a grouped aggregate (groups + aggregate args) or a
// plain projection (select/order-after-filter or bare filterless multi-extract). When the
// child is a filter the struct sits above it, so it materializes only surviving rows.
bool RewriteJsonoProjector(OptimizerExtensionInput &input, LogicalOperator &parent,
                           const vector<vector<unique_ptr<Expression>> *> &expression_lists) {
	if (parent.children.size() != 1) {
		return false;
	}
	vector<JsonoProjectSource> sources;
	for (auto *expression_list : expression_lists) {
		for (auto &expression : *expression_list) {
			CollectProjectSources(input.context, *expression, sources);
		}
	}
	JsonoProjectSource *source = nullptr;
	for (auto &candidate : sources) {
		if (candidate.fields.size() >= 2) {
			source = &candidate;
			break;
		}
	}
	if (!source) {
		return false;
	}

	auto &child = *parent.children[0];
	child.ResolveOperatorTypes();
	auto child_bindings = child.GetColumnBindings();
	if (FindBindingIndex(child_bindings, source->binding) == DConstants::INVALID_INDEX) {
		return false;
	}
	JsonoProjectRewriteState state {input.context,
	                                source->binding,
	                                source->type,
	                                source->alias,
	                                std::move(child_bindings),
	                                child.types,
	                                std::move(source->fields),
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
	if (child.has_estimated_cardinality) {
		projection->SetEstimatedCardinality(child.estimated_cardinality);
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
// followed by VARCHAR shred columns named by canonical path). No implicit cast turns
// that struct into JSONO, so a bare `j->>'path'` / `to_json(j)` binds to core json's
// STRUCT->JSON path, which serializes the raw struct (wrong: leaks the blobs, never
// reads the value). This pre-optimize pass rewrites those bound expressions off the
// shred manifest carried in the column type itself:
//   - `->>`/json_extract_string on a shred path -> struct_extract of the shred column.
//     The shred already holds the extracted text, so this skips the JSONO parse and
//     lets projection/row-group pruning act on the shred column directly.
//   - any CAST(shredded->JSON) and to_json(shredded) -> reinterpret(col->JSONO)::JSON,
//     which reconstructs the authoritative value from the four-blob prefix. Because a
//     non-shred extract's inner cast is rewritten too, correctness never depends on a
//     path being shredded; shreds are purely a fast path.
//===--------------------------------------------------------------------===//

bool IsJsonTextType(const LogicalType &type) {
	return type == LogicalType::JSON();
}

struct JsonoShred {
	idx_t child_index; // index inside the layout's `.shreds` struct
	LogicalType type;
	vector<PathStep> steps;
};

// The shreds of a shredded layout, each named by its canonical path. A shred may be typed (a hot
// number stored as BIGINT/DOUBLE), so the type is carried: the `->>` text contract reads the shred
// and casts it to VARCHAR, while reconstruction injects it back at its JSON type. child_index is the
// shred's 0-based position over the shred set (layout child 1 + child_index, after `body`).
vector<JsonoShred> CollectShreddedShreds(const LogicalType &shredded) {
	vector<JsonoShred> shreds;
	JsonoLayoutType layout;
	if (!TryParseJsonoLayoutType(shredded, layout)) {
		return shreds;
	}
	for (idx_t i = 0; i < layout.shreds.size(); i++) {
		auto &name = layout.shreds[i].first;
		vector<PathStep> steps;
		if (!name.empty() && name[0] == '$') {
			steps = ParseJsonoPath(name, "__jsono_shredded_shred");
		} else {
			steps = LiteralKeyPath(name);
		}
		shreds.push_back(JsonoShred {i, layout.shreds[i].second, std::move(steps)});
	}
	return shreds;
}

// A node in the shred reconstruction patch. Shreds are inserted by their path so the patch
// rebuilds the original nesting as a struct_pack tree; a leaf (no children) carries the
// shred's struct child index, a group carries nested keys.
struct ShredPatchNode {
	idx_t child_index = 0;
	vector<pair<string, ShredPatchNode>> children;

	ShredPatchNode &Child(const string &key) {
		for (auto &entry : children) {
			if (entry.first == key) {
				return entry.second;
			}
		}
		children.emplace_back(key, ShredPatchNode {});
		return children.back().second;
	}
};

// Per shredded jsono column (keyed by its base table ColumnBinding), the totality of each shred:
// entry[child_index] == false means the shred has no SQL NULL across the whole table, i.e. no row
// fell into the residual for that path (NULL-shred <=> value-in-residual, see jsono_shred WriteShred).
// A total shred lets the rewriter drop the residual COALESCE arm so a plain typed struct_extract is
// emitted, which the planner pushes MIN/MAX/filter into the scan's zone-map statistics. Absent /
// unknown / has-null all leave entry true so the safe COALESCE form stays.
using ShredTotalityMap = column_binding_map_t<vector<bool>>;

// Walk the plan's LogicalGet leaves and, for each shredded jsono column the scan exposes, pull
// per-shred null statistics through the table function's statistics_extended pointer (the same one
// DuckDB's own StatisticsPropagator uses; native tables and Parquet both register it). Descend the
// returned StructStats twice: top struct child 0 is the layout field, layout child `1 + child_index`
// is the shred. Fail-safe: no statistics function, nullptr stats, or CanHaveNull() leaves the shred
// non-total (true).
void CollectShredTotality(ClientContext &context, const LogicalOperator &op, ShredTotalityMap &totality) {
	for (auto &child : op.children) {
		CollectShredTotality(context, *child, totality);
	}
	if (op.type != LogicalOperatorType::LOGICAL_GET) {
		return;
	}
	auto &get = op.Cast<LogicalGet>();
	if (!get.function.statistics_extended && !get.function.statistics) {
		return;
	}
	auto &column_ids = get.GetColumnIds();
	auto &returned_types = get.returned_types;
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto table_index = column_ids[i].GetPrimaryIndex();
		if (table_index >= returned_types.size()) {
			continue; // virtual column (rowid etc.); no shredded jsono payload
		}
		auto shreds = CollectShreddedShreds(returned_types[table_index]);
		if (shreds.empty()) {
			continue;
		}
		unique_ptr<BaseStatistics> stats;
		if (get.function.statistics_extended) {
			TableFunctionGetStatisticsInput input(get.bind_data.get(), column_ids[i]);
			stats = get.function.statistics_extended(context, input);
		} else {
			stats = get.function.statistics(context, get.bind_data.get(), table_index);
		}
		JsonoLayoutType layout_info;
		bool has_value_complete =
		    TryParseJsonoLayoutType(returned_types[table_index], layout_info) && layout_info.has_value_complete;
		vector<bool> per_shred(shreds.size(), true);
		if (stats && stats->GetType().id() == LogicalTypeId::STRUCT &&
		    StructType::GetChildCount(stats->GetType()) >= 1) {
			auto &layout_stats = StructStats::GetChildStats(*stats, 0);
			if (layout_stats.GetType().id() == LogicalTypeId::STRUCT) {
				auto layout_children = StructType::GetChildCount(layout_stats.GetType());
				for (auto &shred : shreds) {
					// Canonical base-table layout: body (0), value-complete marker (1), shreds (2+).
					idx_t layout_index = 2 + shred.child_index;
					if (layout_index < layout_children &&
					    !StructStats::GetChildStats(layout_stats, layout_index).CanHaveNull()) {
						per_shred[shred.child_index] = false;
					}
				}
				// Value-complete: the marker field carries no NULL, so no row diverted a present
				// scalar into the residual. Every typed shred's NULLs are then absent paths, where a
				// bare struct_extract reads the correct NULL — so all shreds are safe to read
				// COALESCE-free, even those that still carry (absent) NULLs. (A base-table column is
				// canonical, so the marker is the last layout child; value_complete_index carries its
				// exact position regardless.)
				idx_t vc_index = layout_info.value_complete_index;
				if (has_value_complete && vc_index < layout_children &&
				    !StructStats::GetChildStats(layout_stats, vc_index).CanHaveNull()) {
					for (auto &shred : shreds) {
						per_shred[shred.child_index] = false;
					}
				}
			}
		}
		// The same binding can recur across plan leaves (self-join); keep the conservative view —
		// a shred is total only when every occurrence proves it total.
		auto binding = ColumnBinding(get.table_index, i);
		auto existing = totality.find(binding);
		if (existing == totality.end()) {
			totality.emplace(binding, std::move(per_shred));
		} else {
			for (idx_t k = 0; k < existing->second.size() && k < per_shred.size(); k++) {
				existing->second[k] = existing->second[k] || per_shred[k];
			}
		}
	}
}

class JsonoShreddedRewriter : public LogicalOperatorVisitor {
public:
	explicit JsonoShreddedRewriter(ClientContext &context, const ShredTotalityMap &totality)
	    : context(context), totality(totality) {
	}

protected:
	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		(void)expr_ptr;
		// Whole-value conversion: reconstruct the full value from residual + shreds.
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
		// Introspection (jsono_type / jsono_keys) off a shredded value: feed the residual natively
		// instead of reconstructing when that is value-equivalent (mutates the argument in place).
		if (expr.function.name == "jsono_type" || expr.function.name == "jsono_keys") {
			ResidualizeIntrospect(expr);
		}
		return nullptr;
	}

	unique_ptr<Expression> VisitReplace(BoundCastExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		(void)expr_ptr;
		// Fold CAST(shred string extract AS T) where the shred is itself typed T: read the
		// typed shred directly, skipping the BIGINT->VARCHAR->BIGINT round-trip and giving
		// the planner the shred's typed column statistics.
		auto folded = TryFoldTypedShredCast(expr);
		if (folded) {
			return folded;
		}
		// Whole-value -> JSON: reconstruct from residual + shreds.
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
		return name == "->>" || name == "json_extract_string" || name == "->" || name == "json_extract" ||
		       name == "jsono_extract" || name == "jsono_extract_string";
	}

	// The extract argument is a CAST whose child is a shredded JSONO struct: either
	// `CAST(shredded->JSON)` from core json's STRUCT->JSON path (the core-named extracts), or
	// `CAST(shredded->JSONO)` from the jsono-named extracts' bind-time reconstruct cast. Return that
	// cast so the extract reads shreds/residual directly instead of reconstructing per row.
	optional_ptr<BoundCastExpression> ShreddedJsonCast(Expression &arg) {
		if (arg.GetExpressionClass() != ExpressionClass::BOUND_CAST) {
			return nullptr;
		}
		auto &cast = arg.Cast<BoundCastExpression>();
		if ((IsJsonTextType(cast.return_type) || IsJsonoType(cast.return_type)) &&
		    IsShreddedJsonoType(cast.child->return_type)) {
			return &cast;
		}
		return nullptr;
	}

	// A constant-path read off a shredded value. A string extract of a shred reads the shred
	// column directly. A non-shred path reads the residual natively (cheap): only shred leaves
	// are stripped, so a non-shred path keeps its own value there. Two cases reconstruct
	// instead: a json-valued extract of a shred (its value may be stripped from the residual),
	// and any extract of a subtree that contains a stripped shred leaf — string extracts
	// serialize containers, so they need the leaf as much as json-valued ones.
	unique_ptr<Expression> RewriteShreddedExtract(BoundFunctionExpression &expr, BoundCastExpression &shredded_cast) {
		bool string_fn = expr.function.name == "->>" || expr.function.name == "json_extract_string" ||
		                 expr.function.name == "jsono_extract_string";
		string path;
		vector<PathStep> steps;
		if (!TryReadExtractPath(context, *expr.children[1], path, steps)) {
			// Non-constant (or otherwise unreadable) path: leave the plan untouched. The
			// core-named extracts support per-row paths over the reconstruct cast, and the
			// jsono-named extracts already refused such a path at their own bind.
			return nullptr;
		}
		for (auto &shred : CollectShreddedShreds(shredded_cast.child->return_type)) {
			if (PathStepsEqual(shred.steps, steps)) {
				if (string_fn && !IsShredListType(shred.type)) {
					auto alias = expr.GetAlias();
					return MakeShredRead(std::move(shredded_cast.child), shred, std::move(expr.children[1]), alias);
				}
				// A json-valued extract, or an array shred (whose LIST<STRUCT>/LIST<scalar> column is
				// not the JSON array text a `->>` expects): reconstruct and extract natively (jsono's
				// renderer), not core json's `->>`, so numbers/escapes match to_json.
				return MakeReconstructExtract(expr, shredded_cast, string_fn);
			}
			// A read of a subtree holding a stripped shred leaf would miss it in the
			// residual; reconstruct. (A stripped scalar leaf is itself a shred, handled
			// by the exact match above.)
			if (PathStepsObjectKeyPrefix(steps, shred.steps)) {
				return MakeReconstructExtract(expr, shredded_cast, string_fn);
			}
			// A read descending into a shredded array ($.products[i].subfield) cannot be served from
			// the skeleton residual (the lifted element subfields are stripped); reconstruct.
			if (ReadDescendsIntoArrayShred(shred.type, shred.steps, steps)) {
				return MakeReconstructExtract(expr, shredded_cast, string_fn);
			}
		}
		return MakeResidualExtract(expr, shredded_cast, string_fn);
	}

	// jsono_type / jsono_keys over a shredded value: replace the reconstruct cast on the argument
	// with a native residual read when that is value-equivalent, so the scan reads the residual
	// blobs instead of materializing the whole document per row.
	//   - jsono_type(shredded): the root container type is what shredding never changes, so the
	//     residual alone answers it (a 1-arg jsono_keys is NOT residual-equivalent — it would miss
	//     top-level shred keys — so it is left to reconstruct).
	//   - jsono_type/jsono_keys(shredded, path): residual-equivalent when the path resolves to no
	//     shred and holds no shred beneath it, since only shred leaves are stripped from the residual.
	void ResidualizeIntrospect(BoundFunctionExpression &expr) {
		auto shredded_cast = ShreddedJsonCast(*expr.children[0]);
		if (!shredded_cast) {
			return;
		}
		if (expr.children.size() == 1) {
			if (expr.function.name == "jsono_type") {
				expr.children[0] = ResidualReinterpret(*shredded_cast->child);
			}
			return;
		}
		string path;
		vector<PathStep> steps;
		if (!TryReadExtractPath(context, *expr.children[1], path, steps)) {
			return;
		}
		for (auto &shred : CollectShreddedShreds(shredded_cast->child->return_type)) {
			if (PathStepsEqual(shred.steps, steps) || PathStepsObjectKeyPrefix(steps, shred.steps) ||
			    ReadDescendsIntoArrayShred(shred.type, shred.steps, steps)) {
				return;
			}
		}
		expr.children[0] = ResidualReinterpret(*shredded_cast->child);
	}

	// struct_extract_at(child, one_based): 1-based positional struct field read.
	unique_ptr<Expression> StructExtractAt(unique_ptr<Expression> child, int64_t one_based) {
		FunctionBinder function_binder(context);
		vector<unique_ptr<Expression>> se_children;
		se_children.push_back(std::move(child));
		se_children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(one_based)));
		return function_binder.BindScalarFunction(StructExtractAtFun::GetFunction(), std::move(se_children));
	}

	// Shred `child_index` (0-based over the shred set) of a shredded column: a canonical layout is
	// `body` (1-based 1), the value-complete marker (2), then the shreds, so shred k is layout child
	// 3 + k (1-based). Base-table columns the optimizer reads are always canonical.
	unique_ptr<Expression> ShredExtract(const Expression &column, idx_t child_index) {
		return StructExtractAt(StructExtractAt(column.Copy(), 1), int64_t(child_index) + 3);
	}

	// True when statistics proved this shred carries no SQL NULL anywhere in the table: NULL-shred
	// <=> value-in-residual, so a null-free shred means no row was narrowed for this path and the
	// residual COALESCE fallback is provably dead. The shred read can then be a plain struct_extract,
	// which the planner pushes MIN/MAX/filter into the scan zone-map. Fail-safe: a column that is not
	// a bare base-table reference (a projection-wrapped value, a CTE, an expression result) is absent
	// from the map and stays non-total, so the COALESCE form is kept.
	bool ShredIsTotal(const Expression &column, idx_t child_index) {
		if (column.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		auto &column_ref = column.Cast<BoundColumnRefExpression>();
		auto entry = totality.find(column_ref.binding);
		if (entry == totality.end() || child_index >= entry->second.size()) {
			return false;
		}
		return !entry->second[child_index];
	}

	// Rebuild the body struct (STRUCT(slots,key_heap,string_heap,skips,lengths,nums)) with the
	// per-row shred manifest stripped out of its skips blob, so downstream reads see a manifest-free
	// residual. Used only by the soft residual reinterpret (the shred-lane read fallback).
	unique_ptr<Expression> StripManifestBody(const Expression &body_struct) {
		static const char *const body_fields[] = {"slots", "key_heap", "string_heap", "skips", "lengths", "nums"};
		FunctionBinder function_binder(context);
		vector<unique_ptr<Expression>> body_children;
		for (idx_t i = 0; i < 6; i++) {
			auto field = StructExtractAt(body_struct.Copy(), int64_t(i) + 1);
			if (i == 3) {
				vector<unique_ptr<Expression>> strip_children;
				strip_children.push_back(std::move(field));
				field = function_binder.BindScalarFunction(JsonoStripManifestFunction(), std::move(strip_children));
			}
			field->SetAlias(body_fields[i]);
			body_children.push_back(std::move(field));
		}
		return function_binder.BindScalarFunction(StructPackFun::GetFunction(), std::move(body_children));
	}

	// Read only the residual body of a shredded column as plain JSONO: extract its `.body` struct
	// (layout field [1] -> body [1], already STRUCT(slots,key_heap,string_heap,skips,...)) and wrap it
	// back into the plain layout STRUCT("jsono" STRUCT(body ...)). The result type IS plain JSONO, so
	// the cast is a no-op reinterpret rather than a reconstruction.
	//
	// `soft` selects which manifest discipline the residual carries. A hard residual (the default)
	// wraps the reinterpret in __jsono_internal_checked_residual against the column type's shreds, so
	// a row narrowed by a raw struct cast fails loud on every residual read instead of silently
	// reading an incomplete residual — the right policy for whole-value reads and non-shred-path
	// extracts (a narrowed lane reads as a missing residual path there). A soft residual instead
	// strips the per-row manifest: it backs the COALESCE fallback of a shred-lane read, where the
	// path IS a lane in the type (so narrowing is impossible) and an absent residual path must read as
	// plain NULL — the lane holds the value. Stripping makes the fallback safe under EAGER evaluation
	// (the projector fuse, any hoist out of the COALESCE), which the point-read guard would otherwise
	// turn into a narrowing throw on every row whose value lives in its lane.
	unique_ptr<Expression> ResidualReinterpret(const Expression &column, bool soft = false) {
		FunctionBinder function_binder(context);
		auto body_struct = StructExtractAt(StructExtractAt(column.Copy(), 1), 1);
		auto body = soft ? StripManifestBody(*body_struct) : std::move(body_struct);
		body->SetAlias("body");
		vector<unique_ptr<Expression>> inner_children;
		inner_children.push_back(std::move(body));
		auto inner = function_binder.BindScalarFunction(StructPackFun::GetFunction(), std::move(inner_children));
		inner->SetAlias(JsonoLayoutName());
		vector<unique_ptr<Expression>> outer_children;
		outer_children.push_back(std::move(inner));
		auto outer = function_binder.BindScalarFunction(StructPackFun::GetFunction(), std::move(outer_children));
		auto residual = BoundCastExpression::AddCastToType(context, std::move(outer), JsonoType());
		JsonoLayoutType layout;
		if (!soft && TryParseJsonoLayoutType(column.return_type, layout) && !layout.shreds.empty()) {
			vector<Value> shred_signatures;
			shred_signatures.reserve(layout.shreds.size());
			for (auto &shred : layout.shreds) {
				shred_signatures.push_back(Value(shred.first + '\x01' + shred.second.ToString()));
			}
			auto shreds =
			    make_uniq<BoundConstantExpression>(Value::LIST(LogicalType::VARCHAR, std::move(shred_signatures)));
			vector<unique_ptr<Expression>> check_children;
			check_children.push_back(std::move(residual));
			check_children.push_back(std::move(shreds));
			residual = function_binder.BindScalarFunction(JsonoCheckedResidualFunction(), std::move(check_children));
		}
		auto is_null = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
		is_null->children.push_back(column.Copy());
		auto null_jsono = make_uniq<BoundConstantExpression>(Value(JsonoType()));
		return make_uniq<BoundCaseExpression>(std::move(is_null), std::move(null_jsono), std::move(residual));
	}

	// Bind a native jsono extract of the original path over `source` (a plain-JSONO expression: a
	// residual reinterpret or a full reconstruct). Keeps `->>`/json_extract on jsono's own renderer
	// — text-preserving numbers, lowercase \u escapes — instead of core json's STRUCT->JSON->extract
	// path, which re-renders numbers and uppercases escapes (a visible to_json-vs-extract divergence).
	unique_ptr<Expression> MakeNativeExtractOver(BoundFunctionExpression &expr, unique_ptr<Expression> source,
	                                             bool string_fn) {
		auto path_type = expr.children[1]->return_type;
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(source));
		children.push_back(std::move(expr.children[1]));
		auto alias = expr.GetAlias();
		FunctionBinder function_binder(context);
		auto shred_path_type = path_type.IsIntegral() ? LogicalType::BIGINT : LogicalType::VARCHAR;
		if (string_fn) {
			auto result =
			    function_binder.BindScalarFunction(JsonoExtractStringFunction(shred_path_type), std::move(children));
			result->SetAlias(alias);
			return result;
		}
		// json-valued extract returns JSONO natively; cast back to the type the original extract
		// produced — JSON for core `->`/json_extract, JSONO (a no-op) for jsono_extract.
		auto target_type = expr.return_type;
		auto extracted = function_binder.BindScalarFunction(JsonoExtractFunction(shred_path_type), std::move(children));
		auto result = BoundCastExpression::AddCastToType(context, std::move(extracted), target_type);
		result->SetAlias(alias);
		return result;
	}

	// Extract a non-shred path natively from the residual JSONO, matching a
	// non-shredded column's extract cost — no serialize/parse round-trip through core
	// json. Correct because only shred leaves are stripped: a non-shred scalar keeps its
	// value, and a json-valued read overlapping a stripped shred already reconstructed.
	unique_ptr<Expression> MakeResidualExtract(BoundFunctionExpression &expr, BoundCastExpression &shredded_cast,
	                                           bool string_fn) {
		return MakeNativeExtractOver(expr, ResidualReinterpret(*shredded_cast.child), string_fn);
	}

	// A shredded read whose value the residual alone cannot serve — its leaf is stripped into a
	// shred, the read descends into a shredded array, or it is a json-valued extract of a shred:
	// reconstruct the full plain JSONO value (the array-aware reconstruct cast) and extract natively.
	// This keeps the result on jsono's own renderer, so it is byte-identical to to_json and a plain
	// read instead of diverging through core json's normalizing/uppercasing `->>`.
	unique_ptr<Expression> MakeReconstructExtract(BoundFunctionExpression &expr, BoundCastExpression &shredded_cast,
	                                              bool string_fn) {
		auto reconstruct = BoundCastExpression::AddCastToType(context, shredded_cast.child->Copy(), JsonoType());
		return MakeNativeExtractOver(expr, std::move(reconstruct), string_fn);
	}

	// Detect CAST(string-shred-extract AS T) where the shred's own type is exactly T, and
	// fold it to a direct typed read so the cast no longer round-trips through VARCHAR.
	unique_ptr<Expression> TryFoldTypedShredCast(BoundCastExpression &expr) {
		auto &target = expr.return_type;
		if (target.id() == LogicalTypeId::VARCHAR || IsJsonTextType(target) ||
		    expr.child->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
			return nullptr;
		}
		auto &fn = expr.child->Cast<BoundFunctionExpression>();
		if (!(fn.function.name == "->>" || fn.function.name == "json_extract_string" ||
		      fn.function.name == "jsono_extract_string") ||
		    fn.children.size() != 2) {
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
		for (auto &shred : CollectShreddedShreds(shredded_cast->child->return_type)) {
			if (!PathStepsEqual(shred.steps, steps) || shred.type != target) {
				continue;
			}
			return MakeTypedShredRead(std::move(shredded_cast->child), shred, std::move(fn.children[1]), target,
			                          expr.try_cast, expr.GetAlias());
		}
		return nullptr;
	}

	// Read a typed shred directly as its own type T. The fallback re-extracts from the
	// residual as text and casts to T, matching the original CAST's try/strict mode, so
	// a value kept in the residual still resolves; COALESCE short-circuits so a
	// homogeneous shred stays a plain typed struct_extract (with column statistics).
	// When statistics prove the shred total (no residual row), the fallback arm is provably
	// dead and dropped entirely, leaving a bare struct_extract the planner pushes into the scan.
	unique_ptr<Expression> MakeTypedShredRead(unique_ptr<Expression> column, const JsonoShred &shred,
	                                          unique_ptr<Expression> path, const LogicalType &target, bool try_cast,
	                                          const string &alias) {
		auto shred_value = ShredExtract(*column, shred.child_index);
		if (ShredIsTotal(*column, shred.child_index)) {
			shred_value->SetAlias(alias);
			return shred_value;
		}
		FunctionBinder function_binder(context);
		auto path_type = path->return_type.IsIntegral() ? LogicalType::BIGINT : LogicalType::VARCHAR;
		// Soft residual: this fallback only runs for rows whose value diverted into the residual (the
		// lane is NULL). A non-diverted row's lane already won the COALESCE, so a residual miss here
		// must read as NULL, not a narrowing throw — see ResidualReinterpret.
		auto residual = ResidualReinterpret(*column, /*soft=*/true);
		vector<unique_ptr<Expression>> ext_children;
		ext_children.push_back(std::move(residual));
		ext_children.push_back(std::move(path));
		auto fallback_text =
		    function_binder.BindScalarFunction(JsonoExtractStringFunction(path_type), std::move(ext_children));
		auto fallback = BoundCastExpression::AddCastToType(context, std::move(fallback_text), target, try_cast);
		auto coalesce = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, target);
		coalesce->children.push_back(std::move(shred_value));
		coalesce->children.push_back(std::move(fallback));
		coalesce->SetAlias(alias);
		return std::move(coalesce);
	}

	// Read a string-extracted shred. A VARCHAR shred holds the `->>` text for every row,
	// so it reads directly. A typed shred holds only values of its type; rows whose value
	// did not fit were kept in the residual with a NULL shred, so the typed read casts the
	// shred to text and falls back to a native residual extract. COALESCE short-circuits
	// per row, so a homogeneous shred (the common case) never evaluates the fallback.
	unique_ptr<Expression> MakeShredRead(unique_ptr<Expression> column, const JsonoShred &shred,
	                                     unique_ptr<Expression> path, const string &alias) {
		auto shred_value = ShredExtract(*column, shred.child_index);
		FunctionBinder function_binder(context);
		if (shred.type.id() == LogicalTypeId::VARCHAR) {
			shred_value->SetAlias(alias);
			return shred_value;
		}
		unique_ptr<Expression> shred_text;
		if (shred.type.id() == LogicalTypeId::DOUBLE) {
			// DuckDB's native DOUBLE->VARCHAR cast (1e+16) diverges from jsono's canonical JSON
			// number text (10000000000000000.0, == to_json and a plain `->>`). A typed DOUBLE shred
			// only ever holds a binary double (a text number fails the kind gate and stays in the
			// residual), so its sole canonical text is jsono's own EmitDouble — render through that.
			vector<unique_ptr<Expression>> double_text_children;
			double_text_children.push_back(std::move(shred_value));
			shred_text = make_uniq<BoundFunctionExpression>(LogicalType::VARCHAR, MakeJsonoInternalDoubleTextFunction(),
			                                                std::move(double_text_children), nullptr);
		} else {
			shred_text = BoundCastExpression::AddCastToType(context, std::move(shred_value), LogicalType::VARCHAR);
		}
		// A total shred has no residual rows, so the text read needs no fallback.
		if (ShredIsTotal(*column, shred.child_index)) {
			shred_text->SetAlias(alias);
			return shred_text;
		}
		auto path_type = path->return_type.IsIntegral() ? LogicalType::BIGINT : LogicalType::VARCHAR;
		// Soft residual (see MakeTypedShredRead): a non-diverted row's lane wins the COALESCE, so a
		// residual miss here reads as NULL rather than the point-read guard's narrowing throw.
		auto residual = ResidualReinterpret(*column, /*soft=*/true);
		vector<unique_ptr<Expression>> ext_children;
		ext_children.push_back(std::move(residual));
		ext_children.push_back(std::move(path));
		auto fallback =
		    function_binder.BindScalarFunction(JsonoExtractStringFunction(path_type), std::move(ext_children));
		auto coalesce = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, LogicalType::VARCHAR);
		coalesce->children.push_back(std::move(shred_text));
		coalesce->children.push_back(std::move(fallback));
		coalesce->SetAlias(alias);
		return std::move(coalesce);
	}

	// Build the shred patch as a nested struct_pack tree mirroring each shred's path: a leaf
	// reads its shred column, a group packs its nested keys. jsono_struct_constructor turns
	// it into a JSONO patch for the overlay.
	unique_ptr<Expression> BuildPatchStruct(Expression &column, const vector<pair<string, ShredPatchNode>> &children) {
		FunctionBinder function_binder(context);
		vector<unique_ptr<Expression>> pack_children;
		for (auto &entry : children) {
			unique_ptr<Expression> value;
			if (entry.second.children.empty()) {
				value = ShredExtract(column, entry.second.child_index);
			} else {
				value = BuildPatchStruct(column, entry.second.children);
			}
			value->SetAlias(entry.first);
			pack_children.push_back(std::move(value));
		}
		return function_binder.BindScalarFunction(StructPackFun::GetFunction(), std::move(pack_children));
	}

	// Reconstruct the full JSONO value as JSON by overlaying the shreds onto the residual:
	// the residual is authoritative and each shred only fills a path the residual dropped
	// (jsono_overlay deep-merges, so nested shreds refill nested leaves). Correct for both
	// storage modes — a PRESERVE residual keeps every path so the shreds are no-ops; a
	// stripped residual is refilled from its shreds. Index/array shreds are never stripped,
	// so they stay in the residual and sit out the patch; with no object-key shreds the
	// overlay is skipped.
	unique_ptr<Expression> ReconstructShreddedToJson(unique_ptr<Expression> column) {
		auto shreds = CollectShreddedShreds(column->return_type);
		// The struct_pack patch overlay below cannot represent two shred shapes, so fall back to the
		// array-aware reconstruct cast (which applies each shred as an independent residual-authoritative
		// overlay and so handles both): (1) an array shred (LIST<STRUCT>/LIST<scalar>) leaves a skeleton
		// array in the residual the object-key patch would keep verbatim; (2) two object-key paths where
		// one STRICTLY prefixes another (a scalar `a` and a nested `$.a.a`) — a patch node would be both a
		// leaf and a group, so building the tree would silently drop the shorter shred. Equal-length
		// overlapping paths (the duplicate-path spec `a` and `$.a`) are NOT a problem: they collapse to
		// one patch node, which is the intended dedup. The cast propagates a NULL value through to a NULL
		// JSON, so no struct_pack guard is needed.
		bool needs_full_reconstruct = false;
		for (idx_t i = 0; i < shreds.size() && !needs_full_reconstruct; i++) {
			if (IsShredListType(shreds[i].type)) {
				needs_full_reconstruct = true;
				break;
			}
			for (idx_t j = i + 1; j < shreds.size(); j++) {
				if (shreds[i].steps.size() != shreds[j].steps.size() &&
				    ShredPathsOverlap(shreds[i].steps, shreds[j].steps)) {
					needs_full_reconstruct = true;
					break;
				}
			}
		}
		if (needs_full_reconstruct) {
			auto alias = column->GetAlias();
			auto plain = BoundCastExpression::AddCastToType(context, std::move(column), JsonoType());
			auto json = BoundCastExpression::AddCastToType(context, std::move(plain), LogicalType::JSON());
			json->SetAlias(alias);
			return json;
		}
		ShredPatchNode root;
		for (auto &shred : shreds) {
			bool object_key = !shred.steps.empty();
			for (auto &step : shred.steps) {
				if (step.kind != PathStepKind::Key) {
					object_key = false;
					break;
				}
			}
			if (!object_key) {
				continue;
			}
			auto *node = &root;
			for (auto &step : shred.steps) {
				node = &node->Child(step.key);
			}
			node->child_index = shred.child_index;
		}
		auto residual = ResidualReinterpret(*column);
		if (root.children.empty()) {
			// No object-key shreds to merge; the residual reinterpret already carries
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
	const ShredTotalityMap &totality;
};

//===--------------------------------------------------------------------===//
// Lossless shredded-cast normalization
//
// DuckDB's by-name struct cast between two JSONO types with different shred sets (an INSERT into a
// column shredded differently, a set-operation branch widening to the merged shred union, a
// CASE/COALESCE arm) maps the shreds it can and NULL-fills or drops the rest. NULL-filling a shred is
// harmless — the value stayed in the residual and every reader falls back there — but dropping a
// shred loses the value (it was stripped out of the residual when the source was written), and a
// shred the cast converts to a different type silently changes the value's JSON type. Rewrite every
// such cast into reconstruct-to-plain + reshred-to-target, which is lossless by construction (a
// path that cannot keep its shred falls back into the residual) and gives set operations
// byte-identical encodings (honest UNION DISTINCT / GROUP BY comparison). Type-preserving: the
// replacement returns exactly the cast's target type, so the plan needs no retyping. Without the
// extension optimizer the raw struct cast still binds; a shred it drops or converts is then caught
// at read time by the shred manifest stored in the residual (fail loud, not silent).
//===--------------------------------------------------------------------===//

// Reshred `column` (a plain or shredded jsono value) to the shred set of `target` through the
// regular jsono(value, shredding := <const STRUCT>) bind, so the expression re-binds identically
// on plan deserialization. The bind plans a single-pass reshred when every source shred survives
// into the target; otherwise it routes through the lossless reconstruct cast itself. With no
// shreds the plain reconstruct IS the target.
unique_ptr<Expression> MakeReshredExpression(ClientContext &context, unique_ptr<Expression> column,
                                             const child_list_t<LogicalType> &shreds, const LogicalType &target) {
	auto alias = column->GetAlias();
	if (column->return_type == target) {
		return column;
	}
	unique_ptr<Expression> result;
	if (shreds.empty()) {
		result = BoundCastExpression::AddCastToType(context, std::move(column), JsonoType());
	} else {
		child_list_t<Value> spec_fields;
		for (auto &shred : shreds) {
			spec_fields.emplace_back(shred.first, Value(shred.second.ToString()));
		}
		auto spec = make_uniq<BoundConstantExpression>(Value::STRUCT(std::move(spec_fields)));
		spec->SetAlias("shredding");
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(column));
		children.push_back(std::move(spec));
		FunctionBinder function_binder(context);
		result = function_binder.BindScalarFunction(JsonoShredFromJsonoFunction(), std::move(children));
		if (result->return_type != target) {
			throw InternalException("jsono cast normalization: reshred type mismatch");
		}
	}
	result->SetAlias(alias);
	return result;
}

child_list_t<LogicalType> SortedShreds(const child_list_t<LogicalType> &shreds) {
	auto sorted = shreds;
	std::sort(sorted.begin(), sorted.end(),
	          [](const std::pair<string, LogicalType> &a, const std::pair<string, LogicalType> &b) {
		          return a.first < b.first;
	          });
	return sorted;
}

bool SameShredSet(const child_list_t<LogicalType> &a, const child_list_t<LogicalType> &b) {
	return a.size() == b.size() && SortedShreds(a) == SortedShreds(b);
}

void NormalizeShreddedCastsInExpression(ClientContext &context, unique_ptr<Expression> &expr) {
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { NormalizeShreddedCastsInExpression(context, child); });
	if (expr->GetExpressionClass() != ExpressionClass::BOUND_CAST) {
		return;
	}
	auto &cast = expr->Cast<BoundCastExpression>();
	if (!IsShreddedJsonoType(cast.return_type)) {
		return;
	}
	auto &source_type = cast.child->return_type;
	if (!IsJsonoType(source_type) && !IsShreddedJsonoType(source_type)) {
		return;
	}
	JsonoLayoutType target_layout;
	TryParseJsonoLayoutType(cast.return_type, target_layout);
	if (IsShreddedJsonoType(source_type)) {
		JsonoLayoutType source_layout;
		TryParseJsonoLayoutType(source_type, source_layout);
		// Same shred set, different field order (a set-operation merged type lists the left
		// branch's shreds first): the by-name cast only reorders shreds and is lossless as is.
		if (SameShredSet(source_layout.shreds, target_layout.shreds)) {
			return;
		}
	}
	auto alias = expr->GetAlias();
	auto target = cast.return_type;
	// The reshred constructor canonicalizes shred order (sorted by path), so reshred to the
	// canonical type first and let a reorder-only by-name cast produce the exact target type.
	auto canonical = SortedShreds(target_layout.shreds);
	auto canonical_type = JsonoShreddedStructType(canonical);
	auto replacement = MakeReshredExpression(context, std::move(cast.child), canonical, canonical_type);
	if (canonical_type != target) {
		replacement = BoundCastExpression::AddCastToType(context, std::move(replacement), target);
	}
	replacement->SetAlias(alias);
	expr = std::move(replacement);
}

void NormalizeShreddedCasts(ClientContext &context, LogicalOperator &op) {
	for (auto &child : op.children) {
		NormalizeShreddedCasts(context, *child);
	}
	// EnumerateExpressions reaches every top-level expression of the operator, including the
	// operator-specific ones (a DISTINCT's targets, an ORDER BY's keys) plain `op.expressions`
	// iteration would miss.
	LogicalOperatorVisitor::EnumerateExpressions(
	    op, [&](unique_ptr<Expression> *child) { NormalizeShreddedCastsInExpression(context, *child); });
}

void RewriteShreddedJsono(ClientContext &context, unique_ptr<LogicalOperator> &plan, const ShredTotalityMap &totality) {
	if (!plan) {
		return;
	}
	JsonoShreddedRewriter rewriter(context, totality);
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
	// Collect per-shred totality from base-table statistics first, while column bindings still match
	// the LogicalGet leaves (our own rewrites below interpose projections that would obscure them).
	ShredTotalityMap totality;
	if (plan) {
		CollectShredTotality(input.context, *plan, totality);
	}
	// Replace lossy by-name shredded casts (INSERT into a differently-shredded column,
	// set-operation branch widening, CASE/COALESCE arms) with lossless reconstruct+reshred first,
	// so the shredded rewrites below see canonical shredded values.
	if (plan) {
		NormalizeShreddedCasts(input.context, *plan);
	}
	// Normalize shredded JSONO ops before the built-in pipeline so the shred
	// struct_extract is visible to projection and row-group pruning.
	RewriteShreddedJsono(input.context, plan, totality);
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
