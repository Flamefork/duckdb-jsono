#pragma once

#include "jsono.hpp"
#include "jsono_path.hpp"
#include "jsono_scalar_write.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {

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

// Parse and validate one entry of the public shredding-spec DSL — a `$.`-rooted JSONPath or a bare
// literal key, plus its shred type — into the lane it declares: the LOGICAL path, and the PHYSICAL
// column type (an object-array lane's element subfield names encoded, since a subfield is a one-step
// path). The single spec-side validator: the DDL helper (jsono_storage_type) and the constructor both
// go through it, so a declared storage column always matches a value the constructor can produce.
JsonoLaneSpec JsonoParseShredSpecField(const string &path, const LogicalType &type);

// One array shred for the residual-skeleton emit: the object-key chain to the array, plus the
// lifted-element primitive description. An object array (kind == Array) lifts element subfields into
// a LIST<STRUCT> column; a scalar array (kind == ScalarArray) lifts each whole element into a
// LIST<element_type> column. The two carry disjoint extra fields; `kind` selects which is valid.
// Both `path` and the subfield names are LOGICAL: the emit matches them against document keys.
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

// One lane's manifest entry, in both framings.
struct JsonoShredManifestEntryBytes {
	std::string full;
	std::string compact;
};

// Everything a per-row shred write reads that is a function of the shred SET alone, built once by
// the bind that decided the set. None of it is cheap: building it decodes every lane name back to
// its logical path and renders every lane type, so a per-chunk rebuild charges the whole shred set
// to every batch of rows.
//
// `entries[f]`/`paths[f]` describe lane f. `shreds[f]` is the lane's PHYSICAL (encoded) field name
// and type, while the entry records the lane's LOGICAL path, decoded from that name, because the
// manifest is a per-row statement about the DOCUMENT while the encoding is a transport artifact of
// the TYPE. The reader's verification (VerifyShredManifestEntries) compares those bytes against
// signatures it decodes from its own type the same way, so both sides name the lane identically by
// construction.
//
// `manifest_order` lists the lanes in the manifest's emission order, which is the logical-path order
// and NOT the lane order (that follows the encoded name, and the two genuinely differ: `$.a-c` sorts
// before `$.a.b` as text, while the nested path sorts first structurally). It is the whole reason
// this is a table and not a bare vector: the write loops reach lanes in their own order (field
// order, document order, tree order), so ordering the manifest is a bind-time permutation walked per
// row, never a per-row sort.
//
// `spill_ranks[f]` is lane f's bit in the `$jsono$spill` bitmap (see JsonoSpillRanksOfNames). It is
// the PHYSICAL name that ranks there, because every reader recomputes the ranks from the stored
// type's field names — writer and readers must rank the same string or the bits mean different lanes
// on each side.
struct JsonoShredWriteModel {
	vector<JsonoShredManifestEntryBytes> entries;
	vector<string> paths;
	vector<idx_t> manifest_order;
	vector<idx_t> spill_ranks;
};

JsonoShredWriteModel JsonoBuildShredWriteModel(const vector<std::pair<string, LogicalType>> &shreds);

// The lanes one row strips, as membership only: a write loop marks lanes by LANE index as it reaches
// them and the manifest walk supplies the order. Sized once per shred set, cleared per row.
struct JsonoStrippedLanes {
	void Init(idx_t lane_count) {
		marks.assign(lane_count, 0);
		any = false;
	}
	void Clear() {
		std::fill(marks.begin(), marks.end(), uint8_t(0));
		any = false;
	}
	JSONO_ALWAYS_INLINE void Mark(idx_t lane) {
		D_ASSERT(lane < marks.size());
		marks[lane] = 1;
		any = true;
	}
	bool Empty() const {
		return !any;
	}

	vector<uint8_t> marks;
	bool any = false;
	// Scratch for the manifest walk, held here so a per-row manifest write allocates nothing.
	vector<const JsonoShredManifestEntryBytes *> selected;
};

void JsonoAppendShredManifest(std::string &manifest, const JsonoShredWriteModel &model);

// Appends the entries of the marked lanes, in manifest order.
void JsonoAppendShredManifest(std::string &manifest, const JsonoShredWriteModel &model, JsonoStrippedLanes &lanes);

// Shred a plain JSONO `input` vector into the shredded `result` STRUCT (the six-BLOB residual
// prefix followed by one shred column per `shreds` entry, in order). `shreds[i]` names a lane by
// its PHYSICAL field name and type — the caller already has the result layout (a merged shred set,
// a group_merge accumulator's, the constructor's auto-shred set) and this must reproduce it
// exactly, so the names are decoded to paths, not re-parsed as the public spec DSL. Reuses the
// jsono_shred executor so the constructor and the shred function share strip and shred-write
// semantics. A name that is not a canonical lane name is a broken invariant, not user input.
void JsonoShredFromLayout(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &shreds,
                          Vector &result);

} // namespace duckdb
