#pragma once

#include "duckdb/common/types.hpp"

namespace duckdb {

class ScalarFunction;

// Native jsono extract functions, exposed so the optimizer can read a non-shred path
// straight off the shredded residual (a JSONO value) instead of routing it through
// core json's serialize-then-parse path.
ScalarFunction JsonoExtractStringFunction(const LogicalType &path_type);
ScalarFunction JsonoExtractFunction(const LogicalType &path_type);

} // namespace duckdb
