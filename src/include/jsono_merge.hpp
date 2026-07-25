#pragma once

namespace duckdb {

class ScalarFunction;

// jsono_overlay(base, patch...): base-authoritative fill — the shredded-reconstruction
// primitive the optimizer binds directly (registered in the catalog only for plan
// serialization).
ScalarFunction JsonoOverlayFunction();

} // namespace duckdb
