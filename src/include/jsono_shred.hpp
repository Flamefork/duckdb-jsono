#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <string>
#include <utility>

namespace duckdb {

class ExtensionLoader;

void RegisterJsonoShred(ExtensionLoader &loader);

// True if `type` is a scalar a shred lane can hold losslessly (VARCHAR/BIGINT/UBIGINT/
// DOUBLE/BOOLEAN). The struct constructor lifts only such top-level fields into lanes;
// every other field (other scalars, nested objects/arrays) stays in the residual tape.
bool IsShredLaneType(const LogicalType &type);

// Shred a plain JSONO `input` vector into the shredded `result` STRUCT (the four-BLOB
// residual prefix followed by one lane column per `lanes` entry, in order). Each lane is
// a top-level key named by `lanes[i].first` with type `lanes[i].second`; its value is
// lifted from the document and (when losslessly captured) stripped from the residual.
// Reuses the jsono_shred executor so the constructor and the shred function share strip
// and lane-write semantics.
void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &lanes,
                        Vector &result);

} // namespace duckdb
