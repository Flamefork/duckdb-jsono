#pragma once

#include "jsono_path.hpp"
#include "jsono_scalar_write.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <string>
#include <utility>
#include <vector>

namespace duckdb {

class ExtensionLoader;
class ScalarFunction;

// How deep auto-shred (the struct constructor) and jsono_suggest_shredding descend when lifting
// nested scalar leaves into shreds: leaves at key depth 1..N are lifted (`$.a`, `$.a.b`, … up to N),
// a leaf strictly deeper stays in the residual. The cap is set DELIBERATELY HIGH — far above any
// realistic analytical nesting (web-event hot leaves sit at depth 3: `$.URL.query_params.utm_source`,
// `$.params.*`; deeper analytical leaves are rare) — while still cutting a real pathology: shredding a
// deeply-nested value makes a whole-value reconstruct (`to_json`/`::JSON`/`::VARCHAR`) super-linear in
// depth (each nested lane merges back into the residual skeleton), 6× slower than the plain value at
// depth 8 and 15×+ by depth 48, catastrophic (seconds) past ~100. Point-path shred reads stay cheap
// at any depth; only reconstruct pays. So the bound exists to stop unbounded depth from silently
// turning a deep value's reconstruct into a super-linear cost, not to limit realistic use. It is a
// fixed cap, not configurable: the constructor has no per-row frequency signal at bind, so depth is
// the only honest lever; the advisor is additionally presence/fit-gated but shares the cap so a
// pasted suggestion and the auto path agree. Deeper lanes go through explicit `shredding := {...}`
// (no cap). The read/write/manifest machinery itself is depth-agnostic.
constexpr idx_t JSONO_AUTO_SHRED_MAX_DEPTH = 16;

// The category of a shred column: one scalar value, a LIST<STRUCT> array shred (every element an
// object whose subfields lift), or a LIST<scalar> array shred (every element a scalar that lifts as
// a whole). Readers switch over it with no `default`, so adding a future shred category (e.g. a map
// shred) is a compile error in each native reader rather than a silently-swallowed runtime `default`.
enum class ShredKind : uint8_t { Scalar, Array, ScalarArray };

// Classify a shred column type into its category. Throws InternalException if `type` is neither a
// scalar nor an array shred — the single fail-loud point, so readers carry no swallowing `default`.
// Bind validates shred types up front, so a miss here is a broken invariant, not user input.
ShredKind ClassifyShredKind(const LogicalType &type);

void RegisterJsonoShred(ExtensionLoader &loader);

// The jsono(jsono, shredding := spec) overload, exposed so the optimizer's set-operation
// normalization can reshred a branch to the merged shred set without a catalog lookup.
ScalarFunction JsonoShredFromJsonoFunction();

// True if `type` is a scalar a shred value can hold losslessly (VARCHAR/BIGINT/UBIGINT/
// DOUBLE/BOOLEAN). The struct constructor lifts only such top-level fields into shreds;
// every other field (other scalars, nested objects/arrays) stays in the residual tape.
bool IsShredValueType(const LogicalType &type);

// True if `type` is an object-array shred column type: LIST<STRUCT<...>> whose every struct child is
// itself an IsShredValueType scalar. Such an array shred lifts the chosen leaf subfields of each
// element of a regular array (`$.products`) into a parallel typed LIST<STRUCT>, leaving the
// element tail in the residual skeleton — see docs/jsono_format.md "Array shreds".
bool IsShredArrayType(const LogicalType &type);

// True if `type` is a scalar-array shred column type: LIST<TYPE> whose element TYPE is an
// IsShredValueType scalar (LIST<UBIGINT>, LIST<VARCHAR>, …). Such an array shred lifts each whole
// scalar element of a regular array (`$.item_ids`) into a parallel typed LIST<TYPE>, leaving a
// VAL_NULL placeholder per lifted element in the residual skeleton (non-conforming elements stay
// verbatim) — see docs/jsono_format.md "Scalar array shreds". Mutually exclusive with
// IsShredArrayType (a LIST element is either a struct or a scalar, never both).
bool IsShredScalarArrayType(const LogicalType &type);

// True if `type` is any LIST array shred (object array OR scalar array) — the cases a read must
// reconstruct rather than serve from the residual skeleton, and the cases the constructor routes
// through the two-pass materialize-then-shred writer. The single owner of "is this a list shred".
bool IsShredListType(const LogicalType &type);

void JsonoValidateShredField(const string &path, const LogicalType &type);

// One array shred for the residual-skeleton emit: the object-key chain to the array, plus the
// lifted-element primitive description. An object array (kind == Array) lifts element subfields into
// a LIST<STRUCT> column; a scalar array (kind == ScalarArray) lifts each whole element into a
// LIST<element_type> column. The two carry disjoint extra fields; `kind` selects which is valid.
struct JsonoArrayShredSpec {
	vector<PathStep> path;
	ShredKind kind = ShredKind::Array;
	vector<std::pair<string, jsono::JsonoScalarPrimitive>> subfields; // kind == Array
	jsono::JsonoScalarPrimitive element_primitive;                    // kind == ScalarArray
};

// True if two shred paths overlap structurally: every step they share is the same object key, so
// one is a prefix of (or equal to) the other (e.g. `a` and `$.a.a`, or `a` and `$.a`). Such a pair
// is a sparse multi-shape layout (one row populates at most one lane), not an error — but the
// whole-value to_json overlay cannot pack both into one patch tree (a node would be both a leaf and
// a group), so that reader falls back to the independent-overlay reconstruct. Index/wildcard steps
// never match (they cannot form the scalar-vs-nested overlap).
bool ShredPathsOverlap(const vector<PathStep> &a, const vector<PathStep> &b);

struct JsonoShredManifestEntryBytes {
	std::string full;
	std::string compact;
};

JsonoShredManifestEntryBytes JsonoShredManifestEntry(const string &path, const LogicalType &type);

void JsonoAppendShredManifest(std::string &manifest, const vector<JsonoShredManifestEntryBytes> &entries);

void JsonoAppendShredManifest(std::string &manifest, const vector<JsonoShredManifestEntryBytes> &entries,
                              const vector<idx_t> &entry_indices);

// Shred a plain JSONO `input` vector into the shredded `result` STRUCT (the six-BLOB
// residual prefix followed by one shred column per `shreds` entry, in order). Each shred is
// a top-level key named by `shreds[i].first` with type `shreds[i].second`; its value is
// lifted from the document and (when losslessly captured) stripped from the residual.
// Reuses the jsono_shred executor so the constructor and the shred function share strip
// and shred-write semantics.
void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &shreds,
                        Vector &result);

} // namespace duckdb
