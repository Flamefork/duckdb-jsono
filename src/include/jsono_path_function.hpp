#pragma once

#include "jsono_locate.hpp"
#include "jsono_path_bind.hpp"

#include "duckdb/function/function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {
namespace jsono {

struct JsonoSinglePathBindData : public FunctionData {
	explicit JsonoSinglePathBindData(JsonoPathSpec path_p) : path(std::move(path_p)) {
	}

	JsonoPathSpec path;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoSinglePathBindData>(path);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoSinglePathBindData>();
		return JsonoPathSpecEqual(path, other.path);
	}
};

inline unique_ptr<FunctionData> BindJsonoSinglePath(ClientContext &context, Expression &expr, const char *function_name,
                                                    JsonoPathDialect dialect) {
	return make_uniq<JsonoSinglePathBindData>(BindJsonoPathSpec(context, expr, function_name, dialect));
}

struct JsonoSinglePathLocalState : public FunctionLocalState {
	JsonoPathLocateState locate_state;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoSinglePathLocalState>();
	}
};

} // namespace jsono
} // namespace duckdb
