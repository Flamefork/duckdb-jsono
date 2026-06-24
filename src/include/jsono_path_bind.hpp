#pragma once

#include "jsono_path.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {
namespace jsono {

enum class JsonoPathDialect : uint8_t { Extract, StrictJsonPath };

inline bool TryReadJsonoPathSpec(ClientContext &context, Expression &expr, const char *function_name,
                                 JsonoPathDialect dialect, JsonoPathSpec &spec) {
	if (expr.HasParameter() || !expr.IsFoldable()) {
		return false;
	}
	// A LIST-typed argument is the batch/list extract overload's path vector, not a scalar path.
	// DefaultTryCastAs(VARCHAR) below would stringify it (list_value('$.k') -> "[$.k]") and accept it as
	// a literal key, so the optimizer would rewrite a LIST extract into a scalar one — a VARCHAR result
	// vector under a declared LIST(VARCHAR) type that crashes materialization. Decline so the list
	// extract is left to core json's list overload (its STRUCT->JSON arg cast is reconstructed
	// independently, exactly like a non-constant scalar path).
	if (expr.return_type.id() == LogicalTypeId::LIST) {
		return false;
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		return false;
	}
	if (dialect == JsonoPathDialect::Extract && expr.return_type.IsIntegral()) {
		if (!value.DefaultTryCastAs(LogicalType::BIGINT)) {
			return false;
		}
		auto index = value.GetValue<int64_t>();
		if (index < 0) {
			return false;
		}
		spec.text = std::to_string(index);
		spec.steps = ArrayIndexPath(idx_t(index));
		return true;
	}
	if (!value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		return false;
	}
	spec.text = StringValue::Get(value);
	if (spec.text.empty()) {
		return false;
	}
	if (spec.text[0] == '$') {
		spec.steps = ParseJsonoPath(spec.text, function_name);
	} else if (dialect == JsonoPathDialect::Extract) {
		spec.steps = LiteralKeyPath(spec.text);
	} else {
		return false;
	}
	for (auto &step : spec.steps) {
		if (step.kind == PathStepKind::Wildcard) {
			return false;
		}
	}
	return true;
}

inline JsonoPathSpec BindJsonoPathSpec(ClientContext &context, Expression &expr, const char *function_name,
                                       JsonoPathDialect dialect) {
	if (expr.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!expr.IsFoldable()) {
		throw BinderException("%s: path must be constant", function_name);
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		throw BinderException("%s: path must not be NULL", function_name);
	}
	if (dialect == JsonoPathDialect::Extract && expr.return_type.IsIntegral()) {
		if (!value.DefaultTryCastAs(LogicalType::BIGINT)) {
			throw BinderException("%s: path must be BIGINT", function_name);
		}
		auto index = value.GetValue<int64_t>();
		if (index < 0) {
			throw BinderException("%s: array index must be non-negative", function_name);
		}
		return JsonoPathSpec(std::to_string(index), ArrayIndexPath(idx_t(index)));
	}
	if (!value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		throw BinderException("%s: path must be VARCHAR", function_name);
	}
	auto path = StringValue::Get(value);
	if (path.empty()) {
		ThrowInvalidPath(function_name, path);
	}
	vector<PathStep> steps;
	if (path[0] == '$') {
		steps = ParseJsonoPath(path, function_name);
	} else if (dialect == JsonoPathDialect::Extract) {
		steps = LiteralKeyPath(path);
	} else {
		ThrowInvalidPath(function_name, path);
	}
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("%s: wildcard paths are not supported", function_name);
		}
	}
	return JsonoPathSpec(std::move(path), std::move(steps));
}

} // namespace jsono
} // namespace duckdb
