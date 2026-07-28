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

class ClientContext;
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
// pasted suggestion and the auto path agree. Deeper lanes go through explicit `shredding := '{...}'`
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

// __jsono_internal_reshred(value, target): the same reshred, but declaring its lanes by TYPE — the
// second argument is a NULL constant of the target shredded type, and only its type is read. The
// optimizer uses this instead of the public spec form because a spec is text: rendering a lane type
// back to a string and re-parsing it collapses an object-array element's `id` and `ID` into one
// subfield, undoing exactly what the lane-name codec guarantees.
ScalarFunction JsonoReshredFunction();

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

// The public shredding spec: a constant VARCHAR holding a JSON object that maps each path to its
// shred type string, e.g. '{"$.kind": "VARCHAR", "$.commit.seq": "BIGINT"}' — the key is the path
// (a bare top-level key or a `$.`-rooted JSONPath), the value the stringified shred type (the spec
// language core json_transform takes; read_json's `columns` and Parquet's SHREDDING option are the
// same idea). An array shred names a LIST<STRUCT<...>> type, e.g.
// '{"$.products": "STRUCT(id UBIGINT, name VARCHAR)[]"}'. JSON keys are byte-exact — a
// STRUCT-literal carrier would fold two case-spellings of one key into a duplicate-field error,
// un-declaring exactly the lanes the lane-name codec keeps distinct.
//
// Parses one spec into its (path, type-string) entries, spec-text errors prefixed with `fn_name`;
// every consumer of the spec language goes through here, so the surfaces cannot drift apart.
vector<std::pair<string, string>> JsonoShredSpecEntries(const Value &spec, const char *fn_name);

// Parse one spec entry's type string as SQL type text, rewrapping a parse failure with the caller's
// context WITHOUT erasing the parser's own reason: "Duplicate STRUCT type argument name" points at
// the actual defect (SQL type text folds STRUCT field names case-insensitively — the paste-back
// limit README documents under jsono_layout_lanes), where a bare "unsupported" pointed away from
// it. A type that parses but is no shred type is the caller's refusal, not this one.
LogicalType JsonoParseShredSpecType(const string &type_name, ClientContext &context, const char *fn_name);

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
// one is a prefix of (or equal to) the other (e.g. `a` and `$.a.a`). Such a pair is a sparse
// multi-shape layout (one row populates at most one lane), not an error — but the whole-value
// to_json overlay cannot pack both into one patch tree (a node would be both a leaf and a group),
// so that reader falls back to the independent-overlay reconstruct. Index/wildcard steps never
// match (they cannot form the scalar-vs-nested overlap).
//
// jsono_merge.cpp's ShredPathsStructurallyConflict tests the same shape and reaches the opposite
// verdict — there it disqualifies the lane copy-through fast path rather than declaring the set
// invalid, because merging two inputs must decide which structure survives, while reading one value
// only has to render what is already there.
bool ShredPathsOverlap(const vector<PathStep> &a, const vector<PathStep> &b);

// One lane's manifest entry, ready to append: the length-prefixed logical path, the type code, and
// for an object-array lane its element subfield list.
struct JsonoShredManifestEntryBytes {
	std::string bytes;
};

// Everything a per-row shred write reads that is a function of the shred SET alone, built once by
// the bind that decided the set. None of it is cheap: building it decodes every lane name back to
// its logical path and renders every lane type, so a per-chunk rebuild charges the whole shred set
// to every batch of rows.
//
// `entries`, `paths` and `spill_ranks` are indexed by lane, in the order of the `shreds` the model
// was built from — the (PHYSICAL name, type) pairs the stored type carries. `manifest_order` is the
// one that is NOT: it is indexed by position in the manifest and HOLDS lane indices (see below).
// `entries[f]` is lane f's manifest record and
// `paths[f]` its logical path text, both decoded from that physical name, because the manifest is a
// per-row statement about the DOCUMENT while the encoding is a transport artifact of the TYPE.
// `paths[f]` is also what a carry-over probe matches an input row's manifest against. The reader's
// verification (VerifyShredManifestEntries) compares those bytes against signatures it decodes from
// its own type the same way, so both sides name the lane identically by construction.
//
// `manifest_order` lists the lanes in the manifest's emission order, which is the logical-path order
// and NOT the lane order (the two are different permutations of the same lanes — see
// JsonoRanksInByteOrder). It is the whole reason this is a table and not a bare vector: the write
// loops reach lanes in their own order (field order, document order, tree order), so ordering the
// manifest is a bind-time permutation walked per row, never a per-row sort.
//
// `spill_ranks[f]` is lane f's bit in the `$jsono$spill` bitmap (see JsonoRanksInByteOrder). It is
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
//
// The trade this makes: a per-row sort of the stripped lanes (O(k log k) in what the row actually
// stripped) becomes a per-row walk of the whole set (O(lanes), plus the clear). It pays whenever a
// row strips a decent share of the lanes — the common case, since a shred set is chosen for the
// document — and loses only on a wide set over a sparse document, where it is still linear.
struct JsonoStrippedLanes {
	// Sized FROM the model it will be emitted against, and holding it: the marks are indexed by lane,
	// so a set sized from anything else would mark the wrong lane. Passing the model here makes that
	// agreement a property of construction instead of a check at the far end of the row write, where
	// it could only fire after the mismatched Mark() had already run.
	void Init(const JsonoShredWriteModel &model_p) {
		model = &model_p;
		marks.assign(model_p.entries.size(), 0);
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

	// The model Init sized these against; the emitter reads it from here rather than being handed one
	// separately, so there is no second model to disagree with.
	const JsonoShredWriteModel *model = nullptr;
	vector<uint8_t> marks;
	bool any = false;
	// Scratch for the manifest walk, held here so a per-row manifest write allocates nothing.
	vector<const JsonoShredManifestEntryBytes *> selected;
};

void JsonoAppendShredManifest(std::string &manifest, const JsonoShredWriteModel &model);

// Appends the entries of the marked lanes, in manifest order.
void JsonoAppendShredManifest(std::string &manifest, JsonoStrippedLanes &lanes);

// One subfield lifted out of each array element into the LIST<STRUCT> shred column. `key` is the
// element's JSON key — what the residual skeleton strips and what reconstruct and jsono_entries
// emit; `name` is the STRUCT field it occupies inside the lane's element type, which is that key
// run through the lane-name codec (a subfield is a one-step path).
struct ShredArraySubfield {
	string key;
	string name;
	jsono::JsonoScalarPrimitive primitive;
};

struct ShredField {
	// The lane's two names (see the lane-name boundary in jsono_path.hpp): `steps` is the object-key
	// path lifted out of the document, `lane_name` the STRUCT field the lane occupies inside
	// `shreds` — the encoding of that path. The canonical spill ranks and the type's field order are
	// keyed by lane_name; the manifest records the path, decoded back out of it.
	vector<PathStep> steps;
	string lane_name;
	// kind == Scalar: one leaf scalar lifted at `steps` (use `primitive`). kind == Array: `steps`
	// addresses a regular array; `subfields` are the element leaves lifted into a parallel LIST<STRUCT>
	// column. kind == ScalarArray: `steps` addresses a regular array; each whole scalar element lifts
	// into a parallel LIST<element_type> column (use `element_primitive`).
	ShredKind kind = ShredKind::Scalar;
	jsono::JsonoScalarPrimitive primitive;         // valid when kind == Scalar
	vector<ShredArraySubfield> subfields;          // valid when kind == Array, element-struct order
	jsono::JsonoScalarPrimitive element_primitive; // valid when kind == ScalarArray
};

// A shred set compiled for writing: the lanes, plus the per-lane tables every per-row write reads.
// Built once where the set is decided — a bind — because each table decodes every lane name and
// allocates; rebuilding them per chunk charged the whole shred set to every batch of rows.
struct ShredWriteSet {
	vector<ShredField> fields;
	JsonoShredWriteModel model;
	// The non-scalar lanes' residual-skeleton specs (path + lifted-element description). The skeleton
	// emit strips, per array element, exactly what the WriteArrayShred / WriteScalarArrayShred pass
	// reports lifted (both gate on JsonoScalarFitsPrimitive), keeping the array as the position
	// carrier. Empty — the common case — means the plain leaf-strip emit.
	vector<JsonoArrayShredSpec> array_specs;

	void Build();
};

// Compile a result layout's shred set for writing. `shreds[i]` names a lane by its PHYSICAL field
// name and type — the caller already has the result layout (a merged shred set, a group_merge
// accumulator's, the constructor's auto-shred set) and the write must reproduce it exactly, so the
// names are decoded to paths, never re-parsed as the public spec DSL (a name fed back through the
// DSL would reinterpret it as a fresh path spelling and mint a lane the layout does not have). A
// name that is not a canonical lane name is a broken invariant, not user input.
ShredWriteSet JsonoBuildShredWriteSet(const vector<std::pair<string, LogicalType>> &shreds);

// Shred a plain JSONO `input` vector into the shredded `result` STRUCT (the six-BLOB residual
// prefix followed by one shred column per lane of `write`, in order). Reuses the jsono_shred
// executor so the constructor and the shred function share strip and shred-write semantics.
void JsonoShredFromLayout(Vector &input, idx_t count, const ShredWriteSet &write, Vector &result);

} // namespace duckdb
