#pragma once

#include "duckdb/common/types.hpp"

namespace duckdb {

class ScalarFunction;

// __jsono_shred_patch(shredded): build the JSONO patch holding just the shred values, keyed by the
// paths the type declares. It is a patch, not a value — the caller must overlay it onto the row's
// residual (jsono_overlay) to get the plain document back. Exposed so the optimizer's shredded-cast
// normalization can bind it without a catalog lookup; unlike jsono_overlay it is NOT registered in
// the catalog, so the caller owns the throwing plan-(de)serialization callbacks that keep an
// injected expression from silently failing to round-trip.
ScalarFunction JsonoShreddedPatchFunction(const LogicalType &input_type);

} // namespace duckdb
