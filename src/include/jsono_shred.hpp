#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <string>
#include <utility>

namespace duckdb {

class ExtensionLoader;
class ScalarFunction;

void RegisterJsonoShred(ExtensionLoader &loader);

// The jsono(jsono, shredding := spec) overload, exposed so the optimizer's set-operation
// normalization can reshred a branch to the merged shred set without a catalog lookup.
ScalarFunction JsonoShredFromJsonoFunction();

// True if `type` is a scalar a shred value can hold losslessly (VARCHAR/BIGINT/UBIGINT/
// DOUBLE/BOOLEAN). The struct constructor lifts only such top-level fields into shreds;
// every other field (other scalars, nested objects/arrays) stays in the residual tape.
bool IsShredValueType(const LogicalType &type);

// Shred a plain JSONO `input` vector into the shredded `result` STRUCT (the four-BLOB
// residual prefix followed by one shred column per `shreds` entry, in order). Each shred is
// a top-level key named by `shreds[i].first` with type `shreds[i].second`; its value is
// lifted from the document and (when losslessly captured) stripped from the residual.
// Reuses the jsono_shred executor so the constructor and the shred function share strip
// and shred-write semantics.
void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &shreds,
                        Vector &result);

} // namespace duckdb
