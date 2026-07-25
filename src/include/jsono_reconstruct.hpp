#pragma once

#include "jsono.hpp"
#include "jsono_path.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class Expression;
class ScalarFunction;

// One shred of a shredded type resolved for reading: the shred column index, its lane type, and the
// object-key path its value re-enters the document at. Shared by the reconstruct overlay (which
// resolves only scalar lanes here and keeps LIST lanes in its own array-shred struct) and the keyed
// group_merge direct fold (which stages LIST lanes in this struct too).
struct ReconShred {
	idx_t child;
	LogicalType type;
	vector<PathStep> steps; // object-key path; size 1 is a top-level key
	// Manifest name of the path, filled only by the keyed fold (which rewrites the residual's shred
	// manifest as lanes move); the reconstruct overlay leaves it empty.
	string manifest_path;
};

// Reconstruct a shredded JSONO `input` (six-BLOB residual + shred columns) into the
// lossless plain JSONO `result` by overlaying each row's shred values onto its residual.
// This is how a shredded value becomes usable where a plain JSONO is required (an implicit
// cast, an INSERT into a plain JSONO column) without dropping the shred data. An all-top-level
// shred set overlays as one flat patch; a set carrying any nested path folds top-level and nested
// together into ONE patch tree applied in a single overlay. Throws on a valid row whose residual
// blob is NULL: the writer never emits that, so the row is the trace of a struct cast NULL-filling
// the body.
void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result);

// Overlay only `shreds` (shred indices over the shred set) of the shredded `input` onto its
// residual, producing plain JSONO. The single-pass narrowing reshred folds the shreds the target
// type drops back into the residual with this, leaving the kept shreds untouched. Each row's shred
// manifest is still verified against ALL of the type's shreds, not just the overlaid ones — callers
// that fold a subset (the non-keyed group_merge array path) rely on that for their loud narrowing
// failure.
void JsonoOverlayShredsToPlain(Vector &input, idx_t count, const vector<idx_t> &shreds, Vector &result);

// Render a shredded JSONO carrying top-level LIST shreds directly to JSON text. Scalar shreds are
// first overlaid into the residual; list lanes then merge into their residual arrays by index,
// without materializing the reconstructed arrays as a plain JSONO blob. The optimizer binds this
// only where it already proved the shape, so a non-shredded input, an empty shred set, a list shred
// on a nested path, or two list shreds on one path are all InternalException.
void JsonoRenderShreddedListsToJson(Vector &input, idx_t count, Vector &result);

// Wrap a shredded JSONO argument expression in the internal __jsono_reconstruct scalar operator,
// whose executor is JsonoReconstructToPlain. Only jsono_transform's array-shred reconstruct path uses
// this wrapper: it would otherwise redeclare the argument as plain JSONO and let the binder insert an
// anonymous reconstruct cast, so the wrapper makes that 3-10x shredded->plain cost an explicit, named
// operator in EXPLAIN / EXPLAIN ANALYZE. The other reconstruct binds (collect, elements, diff fallback)
// still inject anonymous casts. Returns the wrapping expression (plain JSONO return type);
// `shredded_arg` must have a shredded JSONO type. Defined in jsono_optimizer.cpp, next to the other
// optimizer-injected internal operators that share its throwing plan-serialization callbacks.
unique_ptr<Expression> MakeJsonoReconstructExpression(unique_ptr<Expression> shredded_arg);

// The optimizer-injected manifest guards: __jsono_internal_checked_residual verifies a residual's
// shred manifest against the shredded type it was read out of (narrowed rows fail loud);
// __jsono_internal_strip_manifest drops the manifest tail for the soft residual reinterpret.
ScalarFunction JsonoCheckedResidualFunction();
ScalarFunction JsonoStripManifestFunction();

} // namespace duckdb
