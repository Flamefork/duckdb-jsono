#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"

#include "string_view.hpp"

// Force inlining where a small decode is called once per scalar from the hot
// readers: left as a plain call the compiler keeps it out-of-line and pays a
// per-scalar 72-byte struct return; inlined, the struct is scalarized away.
// JSONO_COLD marks the throw-only slow paths out-of-line so the inlined hot
// path stays a compare-and-load.
#if defined(__GNUC__) || defined(__clang__)
#define JSONO_ALWAYS_INLINE inline __attribute__((always_inline))
#define JSONO_COLD          __attribute__((noinline, cold))
#else
#define JSONO_ALWAYS_INLINE inline
#define JSONO_COLD
#endif

namespace duckdb {

class ExtensionLoader;
class ClientContext;

// A jsono value is physically a nested STRUCT with exactly one layout field, named "jsono" for
// both plain and shredded values. Inside the layout field live a `body` STRUCT of six BLOBs (the
// binary JSONO body, see JsonoBodyStructType) and, for shredded values, a `shreds` STRUCT holding a
// reserved `$jsono$set` marker (the shred-set hash), a reserved `$jsono$spill` bitmap (per-row
// spilled-shred bits) plus one field per shred — a scalar shred is a bare typed lane, an array
// shred is a LIST. The single layout name is
// deliberate: DuckDB reconciles set-operation branch types by field name (CombineStructTypes), so
// differently-shredded branches merge into one ordinary shredded type whose `shreds` is the union of
// the branches' shred sets — never a field-superset "union-merged" double struct. The struct cast
// NULL-fills a branch's missing shreds, which is exactly the writer's own "value stayed in the
// residual" encoding, so the merged value reads losslessly by construction. The flip side — a
// by-name cast silently dropping shreds when the target carries fewer shreds — is caught at read time
// by the shred manifest stored in the residual (see docs/jsono_format.md), not by the type system.
// The `$jsono$set` marker is the mandatory shared `shreds` member that lets such a by-name cast bind
// between ANY two shred sets, including fully disjoint ones. There is no user-defined logical-type
// alias: such an alias cannot survive Parquet (stripped) or DuckLake (rejected), so every stored
// value — DuckDB-native, Parquet or DuckLake — is just this nested STRUCT, recognised structurally
// (see TryParseJsonoLayoutType / IsJsonoType). Dispatch is structural rather than name-based; no JSON
// key or shred path is reserved, because user paths live only inside `.shreds`.
//
// Split heap design: keys (which are byte-identical across rows that share
// a schema) live in key_heap; per-row string values live in string_heap.
// Parquet column-dictionary encoding then dedups key_heap to a single
// entry per column chunk for shape-stable data, and KEY slot byte patterns
// become identical across rows (the offset into key_heap for "browser" is
// always the same byte position when the keys section is the same).

// Materialise the plain jsono STRUCT type. Helper so writers/readers and
// pluck-on-jsono can agree on the exact STRUCT shape; identical to
// JsonoRawStructType (kept as the name the function registrations read against).
LogicalType JsonoType();

// The physical plain JSONO STRUCT: STRUCT("jsono" STRUCT(body STRUCT(6 BLOB))). There is no
// logical-type alias on top (DuckLake/Parquet reject user-defined aliases, so jsono carries none);
// this is the single source of truth for the plain layout. `jsono_storage_type()` exposes its DDL
// string so writers can declare storage columns without hardcoding the fields.
LogicalType JsonoRawStructType();

// The six-BLOB body STRUCT shared by every layout: STRUCT(slots BLOB, key_heap BLOB,
// string_heap BLOB, skips BLOB). The binary JSONO body lives here, identical for plain and shredded.
LogicalType JsonoBodyStructType();

// The physical STRUCT of a shredded JSONO value: STRUCT("jsono" STRUCT(body STRUCT(6 BLOB),
// shreds STRUCT("$jsono$set" BIGINT, "$jsono$spill" BIGINT [, "$jsono$spill1" …], <shred fields>))).
// One shred field per `shreds` entry — a scalar shred is its bare value type, an array shred is the
// LIST as-is. This is the single source of truth for the shredded shape: the constructor and
// `jsono_storage_type` both build it here, so a column declared from the same shreds is
// byte-identical to the value's type. Shred field order follows `shreds`; callers pass them in
// canonical (name-sorted) order so the type is a pure function of the shred set, and they write
// shreds in that same order (shreds are written by index, so field and value order must agree).
LogicalType JsonoShreddedStructType(const child_list_t<LogicalType> &shreds);

// The top-level field name of every layout (plain and shredded): "jsono".
string JsonoLayoutName();

// The name of the nested STRUCT (sibling of `body`) that holds the shred set. Shredded-ness is
// exactly the presence of this field.
string JsonoShredsName();

// The reserved name of the shred-set marker, the first field inside `shreds`. A BIGINT carrying the
// canonical layout hash (JsonoLayoutHashOf) of the shred set the row was written under — its uint64
// bits reinterpreted as signed — on a clean row, and that hash with its lowest bit flipped
// (JSONO_DIRTY_HASH_FLIP) on a dirty row (one whose `$jsono$spill` bitmap is non-empty); NULL only
// on a SQL-NULL row. The two-valued stamp keeps every totality proof a plain min/max question,
// which survives Parquet zone-maps, native storage and DuckLake's global min/max strings alike:
// min==max==H proves the scan all-clean under exactly the read shred set (every scalar shred reads
// as a bare struct_extract, zone-map pushable), while markers within the {H, H^1} pair prove the
// set identity with some dirty rows — whose spill statistics then decide per shred. A multi-file
// read unioning narrower shred sets carries an unrelated hash per file, which fails both patterns
// and keeps the residual COALESCE fallback. BIGINT not UBIGINT so the Parquet writer emits its
// min/max zone-map (unsigned ints get none). Reserved so no shred path can collide; the layout
// parser identifies it by name with any integer width (IsIntegral), tolerating a value
// round-tripped through a generic value->SQL->value path (e.g. DuckLake inlined-data INSERT) where
// the width may shift to HUGEINT.
string JsonoShredSetName();

// The reserved name of a per-row spill bitmap column — the ⌈N/63⌉ fields right after the marker
// inside `shreds`: "$jsono$spill", then "$jsono$spill1", "$jsono$spill2", … (column 0 carries no
// ordinal: nearly every real set fits one column). Each is a BIGINT whose bit b (0-based, low 63
// bits — the sign bit stays clear so masks order as non-negative numbers in every zone-map) is set
// when the scalar shred at canonical rank 63*column + b diverted on this row: the value stayed in
// the residual (a type-gate miss or a container at the path) and a bare lane read would miss it.
// Canonical rank = the shred's position in the byte-wise sorted name list of the shred set, NOT
// the physical field position — a reorder-only cast (SameShredSet) keeps field order
// branch-dependent while the order-independent `$jsono$set` hash still matches, so the numbering
// must be permutation-invariant too. Array shreds get ranks but never set bits (they are never
// bare-read). All spill columns are NULL on a clean row (an all-zero mask is never stored) and on
// a SQL-NULL row, and all non-NULL on a dirty one (zeroes allowed per column), so min/max
// statistics describe the DIRTY rows alone: a systematic divert gives min==max==M per column,
// which decodes per shred exactly — one source's dirty lane never demotes its row-group's other
// shreds. The all-clean case deliberately leaves the columns with no values at all (the clean
// proof lives in the marker, see JsonoShredSetName): DuckLake's global stats carry no num_values,
// so neither an all-NULL bitmap nor an all-NULL LIST (the encoding rejected in plan 054 phase 0)
// can prove anything there by itself. Interpretation is identity-gated: a raw by-name cast copies
// the bitmaps across shred sets, where the foreign `$jsono$set` hash fails the identity check and
// the foreign numbering is conservatively ignored.
string JsonoShredSpillName(idx_t column);

// The marker stamp of a dirty row: the layout hash with its lowest bit flipped. Two adjacent
// values keep the marker's min/max decodable ({H} all-clean, {H^1} all-dirty, the pair mixed);
// stamping the mask itself would scatter min/max across 2^63 values and prove nothing.
constexpr uint64_t JSONO_DIRTY_HASH_FLIP = 1;

// Bits per spill column (the sign bit stays clear).
constexpr idx_t JSONO_SPILL_BITS = 63;

// The spill columns a shred set of `shred_count` needs: ⌈N/63⌉. Recognition additionally accepts
// FEWER columns than this (never more): a set-op merged type unites both branches' shreds but only
// the by-name union of their spill columns, which can under-provision a crossing union's set. Such
// a type stays fully readable — bits for ranks beyond its columns simply do not exist, so those
// shreds are never proven total and every read keeps the residual COALESCE.
inline idx_t JsonoSpillColumnCount(idx_t shred_count) {
	return (shred_count + JSONO_SPILL_BITS - 1) / JSONO_SPILL_BITS;
}

// Throw a BinderException if `name` collides with a reserved layout field: `body` (the residual,
// rejected for consistency — a value-path shred cannot be named it either) or the reserved
// `$jsono$` name prefix (the marker and the spill bitmap live there; reserving the whole prefix
// keeps future layout fields collision-free). Shared by the shredding-spec constructor
// (ParseShredSpec) and the DDL path (jsono_storage_type) so both reject the same names —
// otherwise jsono_storage_type emits a type the constructor can never produce or insert into.
void JsonoValidateShredFieldName(const string &name);

// Canonical layout hash of a shredded jsono type: HashShredManifestSignatures over its shreds
// (path + logical value type) in type order. Returns 0 for a plain / non-shredded type.
// Writers stamp this into the shred-set marker; the optimizer recomputes it from the read type
// to verify per-scan shred-set coverage before trusting the spill bitmap.
uint64_t JsonoLayoutHashOf(const LogicalType &type);

// The canonical spill-bit rank of each name in `names` (parallel vector): its position in the
// byte-wise sorted name list. Field lists every constructor emits are already name-sorted, so
// rank == index there; a set-op reorder-only cast permutes fields while the order-independent
// set hash still matches, which is why the bit numbering must be permutation-invariant.
vector<idx_t> JsonoSpillRanksOfNames(const vector<string> &names);

// The classification of one JSONO layout field.
enum class JsonoLayoutKind : uint8_t { Plain, Shredded };

// A parsed JSONO layout field: its top-level name, the inner STRUCT(body[, shreds]) type, its
// shred (path, LOGICAL-value-type) columns (empty for plain; a scalar shred's bare lane and an
// array shred's list are recorded by their value type), and the number of spill bitmap columns.
// The single grammar all predicates read against. The shred-set marker always sits at `shreds`
// field 0 and the spill columns at fields 1..spill_columns; shred k is field 1 + spill_columns + k.
struct JsonoLayoutType {
	JsonoLayoutKind kind;
	string layout_name;
	LogicalType layout_type;
	child_list_t<LogicalType> shreds;
	idx_t spill_columns = 0;
};

// Parse `type` as an ordinary JSONO value: a top-level STRUCT with exactly one valid `jsono`
// layout field (`body` only for plain, `body` + non-empty `shreds` for shredded).
// The single classifier the thin predicates (IsJsonoType / IsShreddedJsonoType) delegate to.
bool TryParseJsonoLayoutType(const LogicalType &type, JsonoLayoutType &out);

// True when `type` is the plain JSONO STRUCT (a `jsono` layout field carrying only `body`). Strict:
// a shredded value is not a plain value (extending this to shredded would give silently wrong
// results where call sites read only the residual body).
bool IsJsonoType(const LogicalType &type);

// True when `type` is a shredded JSONO struct: a `jsono` layout field carrying `body` plus a
// non-empty `shreds`. Set operations over differently-shredded values reconcile into this same
// shape (the shred union), so there is no separate merged classification.
bool IsShreddedJsonoType(const LogicalType &type);

class Expression;

// Resolve a function's first (JSONO) argument and return the type to assign to its
// arguments[0]. SQLNULL/plain JSONO -> JsonoType(). Shredded -> JsonoType() when
// `reconstruct_shredded` (the binder then inserts the lossless reconstruct cast so the executor
// sees plain JSONO), else the shredded type itself (the executor reads its residual natively).
// Throws BinderException naming the function for any non-JSONO type, so an arbitrary struct is
// rejected loudly rather than silently routed through the struct constructor cast;
// ParameterNotResolvedException for an unresolved prepared parameter.
void JsonoRequireExtensionOptimizerForShredded(ClientContext &context, const LogicalType &type,
                                               const string &function_name);
LogicalType JsonoResolveJsonoArgument(ClientContext &context, const Expression &arg, const string &function_name,
                                      bool reconstruct_shredded);

class Vector;

// Reconstruct a shredded JSONO `input` (six-BLOB residual + shred columns) into the
// lossless plain JSONO `result` by overlaying each row's shred values onto its residual.
// This is how a shredded value becomes usable where a plain JSONO is required (an implicit
// cast, an INSERT into a plain JSONO column) without dropping the shred data. Top-level shreds
// overlay in one flat pass; a nested-path shred overlays its single key chain onto the residual.
void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result);

// Wrap a shredded JSONO argument expression in the internal __jsono_reconstruct scalar operator,
// whose executor is JsonoReconstructToPlain. Only jsono_transform's array-shred reconstruct path uses
// this wrapper: it would otherwise redeclare the argument as plain JSONO and let the binder insert an
// anonymous reconstruct cast, so the wrapper makes that 3-10x shredded->plain cost an explicit, named
// operator in EXPLAIN / EXPLAIN ANALYZE. The other reconstruct binds (collect, elements, diff fallback)
// still inject anonymous casts. Returns the wrapping expression (plain JSONO return type);
// `shredded_arg` must have a shredded JSONO type.
unique_ptr<Expression> MakeJsonoReconstructExpression(unique_ptr<Expression> shredded_arg);

// Overlay only `shreds` (shred indices over the shred set) of the shredded `input` onto its
// residual, producing plain JSONO. The single-pass narrowing reshred folds the shreds the target
// type drops back into the residual with this, leaving the kept shreds untouched.
void JsonoOverlayShredsToPlain(Vector &input, idx_t count, const vector<idx_t> &shreds, Vector &result);

// Render a shredded JSONO carrying top-level LIST shreds directly to JSON text. Scalar shreds are
// first overlaid into the residual; list lanes then merge into their residual arrays by index,
// without materializing the reconstructed arrays as a plain JSONO blob.
void JsonoRenderShreddedListsToJson(Vector &input, idx_t count, Vector &result);

// Parse a VARCHAR/JSON `source` vector into a plain JSONO `result` vector, throwing on
// invalid JSON (the jsono() text contract). Exposed so the shred constructor overload can
// parse-then-shred in one pass without duplicating the parser.
void JsonoParseTextVector(Vector &source, idx_t count, Vector &result);

//===--------------------------------------------------------------------===//
// JSONO binary format
//
// A JSONO value is physically split into six BLOB children:
//   slots        — [Header (8 bytes)] [Slots (n × 8 bytes)]
//   key_heap     — key bytes referenced by KEY slots
//   string_heap  — string bytes consumed by STRING/NUMBER slots in walk order
//   skips        — [ContainerMetadataHeader] [sorted u32 container ids]
//                  [ContainerSpan...] [ObjectCheckpointIndex...]
//                  [ObjectCursorCheckpoint...] [shred manifest]
//   lengths      — u32 LE byte lengths of VAL_STR_HEAP / VAL_EXT-NUMBER values,
//                  walk order
//   nums         — u64 LE numeric payloads of VAL_INT60 / VAL_DEC60 /
//                  VAL_EXT-INT64/UINT64/DOUBLE values, walk order
//
// Header (little-endian throughout):
//   u32 magic        = 'JSNO' (0x4F4E534A)
//   u8  version
//   u8  flags        (bit0 = SORTED_KEYS)
//   u16 reserved
//
// Slot encoding: 64-bit little-endian word.
//   Top 4 bits  = tag
//   Low 60 bits = payload (semantics depend on tag)
//
// Key invariants enforced by writer:
//   - Every object's keys appear in ascending byte-wise order.
//   - Heaps may contain duplicate strings; deduplication is delegated to
//     Parquet column-dictionary encoding.
//===--------------------------------------------------------------------===//

namespace jsono {

constexpr uint32_t MAGIC = 0x4F4E534A; // 'JSNO' little-endian
// version 2: ContainerSpan carries a per-object shape_hash (sorted-key
// fingerprint) so readers can trust a cached object key-rank by an int compare
// instead of a key-heap read+string-compare. See docs/jsono_format.md.
// version 3: the skips blob may carry a trailing shred manifest — the (path, type)
// entries this row's shred writer stripped out of the residual. Readers verify the
// manifest against the shreds actually present, turning a shred silently dropped or
// retyped by a raw struct cast into a loud read error. See docs/jsono_format.md.
// version 4: variable payloads leave the slots. String/number-text lengths move to
// the `lengths` stream (u32 LE, walk order) and numeric payloads (INT60/DEC60 and
// the former VAL_EXT trailing data slots) to the `nums` stream (u64 LE, walk
// order), making slot words shape-constant across rows. ContainerSpan gains
// length_count/num_count, ObjectCursorCheckpoint gains length_delta/num_delta, and
// spans are stored only where a skip needs them — non-empty arrays and objects
// with child_count > OBJECT_CHECKPOINT_STRIDE — addressed through a sorted sparse
// container-id index.
constexpr uint8_t VERSION = 0x04;

namespace flags {
constexpr uint8_t SORTED_KEYS = 0x01;
}

// Tag namespace follows docs/jsono_format.md. Structure tags (0x0–0x4) carry
// containers and keys; literals (0x5–0x7) are payload-free single slots; hot
// value tags (0x8, 0xA, 0xB) cover the mass text-parse path; VAL_EXT (0xF) is
// the escape that absorbs the rare numeric tail via a subtype byte.
namespace tag {
constexpr uint64_t OBJ_START = 0x0;
constexpr uint64_t OBJ_END = 0x1;
constexpr uint64_t ARR_START = 0x2;
constexpr uint64_t ARR_END = 0x3;
constexpr uint64_t KEY = 0x4;
constexpr uint64_t VAL_NULL = 0x5;
constexpr uint64_t VAL_TRUE = 0x6;
constexpr uint64_t VAL_FALSE = 0x7;
constexpr uint64_t VAL_STR_HEAP = 0x8;
constexpr uint64_t VAL_INT60 = 0xA;
constexpr uint64_t VAL_DEC60 = 0xB; // fractional mantissa/scale, value = mantissa / 10^scale
constexpr uint64_t VAL_EXT = 0xF;
} // namespace tag

// VAL_EXT subtype byte (low 8 bits of the VAL_EXT slot payload). Selects the
// concrete extended encoding. Every VAL_EXT value is a single slot: INT64/
// UINT64/DOUBLE store their 64 raw bits as one `nums` entry, NUMBER stores its
// byte-exact text in string_heap with its length as one `lengths` entry.
namespace ext_subtype {
constexpr uint64_t INT64 = 0;  // signed 2^59..2^63, one nums entry
constexpr uint64_t UINT64 = 1; // unsigned 2^63..2^64-1, one nums entry
constexpr uint64_t DOUBLE = 2; // IEEE double bits (jsono(STRUCT) path), one nums entry
constexpr uint64_t NUMBER = 3; // raw bignum/high-precision text in string_heap, one lengths entry
constexpr uint64_t COUNT = 4;
} // namespace ext_subtype

constexpr uint64_t TAG_SHIFT = 60;
constexpr uint64_t PAYLOAD_MASK = (uint64_t(1) << 60) - 1;

inline uint64_t MakeSlot(uint64_t t, uint64_t payload) {
	return (t << TAG_SHIFT) | (payload & PAYLOAD_MASK);
}

inline uint64_t SlotTag(uint64_t slot) {
	return slot >> TAG_SHIFT;
}

inline uint64_t SlotPayload(uint64_t slot) {
	return slot & PAYLOAD_MASK;
}

inline uint64_t MakeExtSlot(uint64_t subtype) {
	return MakeSlot(tag::VAL_EXT, subtype);
}

inline uint64_t ExtSubtype(uint64_t slot) {
	return SlotPayload(slot) & 0xFF;
}

// KEY payload layout: heap_offset (36 bits high) << 24 | heap_len (24 bits low).
// KEYs still carry their heap offset because key_heap is dict-encoded by
// Parquet (every row has the same key set in the same sorted order, so the
// dictionary collapses to a single entry per column chunk and the offsets
// are byte-identical across rows already).
//
// VAL_STR_HEAP carries no payload: the byte length lives in the `lengths`
// stream and the byte offset into string_heap is implicit — the reader
// maintains a cursor pair (heap bytes + lengths index) in walk order.
constexpr uint64_t HEAP_LEN_BITS = 24;
constexpr uint64_t HEAP_LEN_MASK = (uint64_t(1) << HEAP_LEN_BITS) - 1;

inline uint64_t MakeKeyPayload(uint64_t heap_offset, uint64_t heap_len) {
	return (heap_offset << HEAP_LEN_BITS) | (heap_len & HEAP_LEN_MASK);
}

inline uint64_t KeyOffset(uint64_t payload) {
	return payload >> HEAP_LEN_BITS;
}

inline uint64_t KeyLen(uint64_t payload) {
	return payload & HEAP_LEN_MASK;
}

// OBJ_START/ARR_START payload: container id into skips plus inline
// child count. Only OBJ_START uses child_count; arrays store 0.
constexpr uint64_t CONTAINER_CHILD_COUNT_BITS = 24;
constexpr uint64_t CONTAINER_CHILD_COUNT_MASK = (uint64_t(1) << CONTAINER_CHILD_COUNT_BITS) - 1;
constexpr uint64_t CONTAINER_ID_MAX = PAYLOAD_MASK >> CONTAINER_CHILD_COUNT_BITS;

constexpr uint32_t NO_OBJECT_CHECKPOINTS = std::numeric_limits<uint32_t>::max();
// Container ids below this resolve through the downward hint scan in TryContainerSpan
// (<= SPAN_HINT_LIMIT probes, usually 1-2) instead of the binary search; the bound also
// caps the scan length.
constexpr uint64_t SPAN_HINT_LIMIT = 4;
constexpr uint32_t OBJECT_CHECKPOINT_STRIDE = 16;

// Whether this translation unit is built with AddressSanitizer instrumentation.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define JSONO_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define JSONO_ADDRESS_SANITIZER 1
#endif
// Whether this translation unit is built with UBSan instrumentation (without ASan). The detection
// is clang-only (`__has_feature`); GCC defines no equivalent, so a GCC UBSan build keeps the
// uninstrumented bounds.
#if defined(__has_feature) && !defined(JSONO_ADDRESS_SANITIZER)
#if __has_feature(undefined_behavior_sanitizer)
#define JSONO_UB_SANITIZER 1
#endif
#endif

// Maximum container nesting accepted by the writer and the recursive readers.
// The tape walkers recurse per nesting level, so an unbounded depth overflows
// the C++ stack; this bound makes the guard trip before that happens. The release
// bounds (1000, or 512 on macOS) stay far above any realistic document (real JSON
// nests a few dozen levels deep); a sanitizer build drops far lower, close to that
// ceiling, because its instrumented frames overflow the stack much sooner.
// AddressSanitizer's fat frames sit atop a deep execution pipeline on a ~512 KiB
// macOS worker-thread stack, and the
// over-limit exception is itself serialized (yyjson, via DuckDB's ToJSON exception
// constructor) at the deepest recursion frame, so the guard must fire much sooner
// there to stay ahead of a stack overflow. 50 was still too high — the serialized
// throw overflowed ~2/3 of runs (ASLR-dependent). 32 clears the deepest exercised
// document (depth 30 in jsono_depth_limit.test) yet leaves ~18 recursion frames of
// headroom below the observed overflow point for the throw's own serialization.
// A UBSan-only build keeps the full depth EXCEPT on macOS: UBSan frames are thinner than
// ASan's but still overflow macOS's ~512 KiB worker-thread stacks ASLR-dependently anywhere
// past a few dozen levels (observed: strip recursion died between 128 and 150; the over-limit
// guard throw at the deepest parse frame died 4/10 runs even at 64). Linux worker threads get
// the platform-default 8 MiB stack, which fits the full 1000-level recursion — that is where
// the UBSan full-depth CI job runs depths the ASan bound never reaches.
// The depth-999 stress showed even the RELEASE recursion plus the guard throw overflowing the macOS
// stack ASLR-dependently (2/10 unittest runs, bus error) — the guard must trip before the stack does,
// so the release bound is halved on macOS.
#if defined(JSONO_ADDRESS_SANITIZER)
constexpr size_t JSONO_MAX_NESTING_DEPTH = 32;
#elif defined(JSONO_UB_SANITIZER) && defined(__APPLE__)
constexpr size_t JSONO_MAX_NESTING_DEPTH = 32;
#elif defined(__APPLE__)
constexpr size_t JSONO_MAX_NESTING_DEPTH = 512;
#else
constexpr size_t JSONO_MAX_NESTING_DEPTH = 1000;
#endif

// Shape-hash helpers — short-input wyhash-inspired mixer. Both the writer (per-object
// shape fingerprint stored in ContainerSpan) and the DOM builder (DOM-order key
// fingerprint for its shape cache) hash key bytes; one definition shared here.
//
// The two consumers trust a hash match differently. The persisted ContainerSpan.shape_hash
// is always re-validated: read caches (RankCache/JsonoTrieRankCache) use it only as a
// prefilter and re-confirm the actual key on every hit, and jsono_validate recomputes it —
// a collision costs a recompute, never a wrong answer. The DOM shape-cache LOOKUP
// fingerprint is the one site that trusts a hash match authoritatively (a hit reuses the
// cached key permutation with no re-check); its collision failure mode and the per-builder
// salt that guards against constructed collisions are documented in jsono_dom.hpp. The
// mixer is NOT keyed (fixed seed/prime below) — which is exactly why that lookup is salted
// while the re-validated persisted hash is not.
inline uint64_t HashMix64(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
	__uint128_t r = static_cast<__uint128_t>(a) * b;
	return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
#else
	uint64_t ah = a >> 32, al = a & 0xFFFFFFFFULL, bh = b >> 32, bl = b & 0xFFFFFFFFULL;
	uint64_t low = al * bl;
	uint64_t mid = ah * bl + al * bh + (low >> 32);
	uint64_t high = ah * bh + (mid >> 32);
	return (low & 0xFFFFFFFFULL) ^ (mid << 32) ^ high;
#endif
}

constexpr uint64_t HASH_PRIME = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t HASH_SEED = 0xCBF29CE484222325ULL;

inline uint64_t HashBytes(uint64_t state, const char *data, size_t len) {
	while (len >= 8) {
		uint64_t v;
		std::memcpy(&v, data, 8);
		state = HashMix64(state ^ v, HASH_PRIME);
		data += 8;
		len -= 8;
	}
	if (len > 0) {
		uint64_t v = 0;
		std::memcpy(&v, data, len);
		state = HashMix64(state ^ v, HASH_PRIME);
	}
	return state;
}

inline uint64_t HashKey(uint64_t state, nonstd::string_view key) {
	state = HashBytes(state, key.data(), key.size());
	// Length-prefix so {"ab","c"} and {"a","bc"} don't collide on byte concat.
	return HashMix64(state ^ uint64_t(key.size()), HASH_PRIME);
}

struct ContainerMetadataHeader {
	uint32_t span_count;
	uint32_t checkpoint_index_count;
	uint32_t checkpoint_count;
};

// ContainerSpan carries the subtree's slot span, string_heap byte span, lengths/
// nums entry counts, and (for objects) a shape_hash: the fingerprint of the
// object's sorted key sequence. Spans are stored only for non-empty arrays and
// objects with child_count > OBJECT_CHECKPOINT_STRIDE; a sorted sparse
// container-id index precedes the span records (see docs/jsono_format.md). Arrays
// store shape_hash = 0 (never read — the rank cache only runs on objects).
struct ContainerSpan {
	uint32_t slot_span;
	uint32_t string_byte_span;
	uint64_t shape_hash;
	uint32_t length_count;
	uint32_t num_count;
};

struct ObjectCheckpointIndex {
	uint32_t container_id;
	uint32_t checkpoint_offset;
	uint16_t checkpoint_stride;
	uint16_t reserved;
};

struct ObjectCursorCheckpoint {
	uint32_t slot_delta;
	uint32_t string_delta;
	uint32_t length_delta;
	uint32_t num_delta;
};

static_assert(sizeof(ContainerMetadataHeader) == 12, "ContainerMetadataHeader must be exactly 12 bytes");
static_assert(sizeof(ContainerSpan) == 24, "ContainerSpan must be exactly 24 bytes");
static_assert(sizeof(ObjectCheckpointIndex) == 12, "ObjectCheckpointIndex must be exactly 12 bytes");
static_assert(sizeof(ObjectCursorCheckpoint) == 16, "ObjectCursorCheckpoint must be exactly 16 bytes");

// The reader's walk state over one JSONO value: the slot position plus the three
// implicit-offset stream cursors. string_cursor counts string_heap BYTES;
// length_cursor / num_cursor count `lengths` (u32) / `nums` (u64) ENTRIES. All
// four advance together in tree-walk order; ContainerSpan / ObjectCursorCheckpoint
// carry per-stream deltas so a subtree skip or checkpoint jump moves the whole
// cursor in O(1).
struct JsonoCursor {
	size_t pos = 0;
	size_t string_cursor = 0;
	size_t length_cursor = 0;
	size_t num_cursor = 0;
};

// One shred-manifest entry: a path the shred writer stripped out of this row's residual (its value
// lives only in the shred) and the shred type it was written as. The views point into the skips blob.
struct ShredManifestEntry {
	nonstd::string_view path;
	nonstd::string_view type;
};

constexpr uint32_t SHRED_MANIFEST_COMPACT_TYPE_MARKER = 0xFFFFFFFFU;
constexpr uint8_t SHRED_MANIFEST_TYPE_EXTENDED = 0;
constexpr uint8_t SHRED_MANIFEST_TYPE_VARCHAR = 1;
constexpr uint8_t SHRED_MANIFEST_TYPE_BIGINT = 2;
constexpr uint8_t SHRED_MANIFEST_TYPE_UBIGINT = 3;
constexpr uint8_t SHRED_MANIFEST_TYPE_DOUBLE = 4;
constexpr uint8_t SHRED_MANIFEST_TYPE_BOOLEAN = 5;
constexpr uint8_t SHRED_MANIFEST_TYPE_VARCHAR_LIST = 6;
constexpr uint8_t SHRED_MANIFEST_TYPE_BIGINT_LIST = 7;
constexpr uint8_t SHRED_MANIFEST_TYPE_UBIGINT_LIST = 8;
constexpr uint8_t SHRED_MANIFEST_TYPE_DOUBLE_LIST = 9;
constexpr uint8_t SHRED_MANIFEST_TYPE_BOOLEAN_LIST = 10;

struct ShredManifestCompactType {
	uint8_t code;
	nonstd::string_view name;
};

// Codes are wire format (format-locked): they are written into the residual skips blob and must not
// be renumbered. The EXTENDED sentinel (0) is not in this table — it is the out-of-band escape.
static constexpr ShredManifestCompactType SHRED_MANIFEST_COMPACT_TYPES[] = {
    {SHRED_MANIFEST_TYPE_VARCHAR, nonstd::string_view("VARCHAR", 7)},
    {SHRED_MANIFEST_TYPE_BIGINT, nonstd::string_view("BIGINT", 6)},
    {SHRED_MANIFEST_TYPE_UBIGINT, nonstd::string_view("UBIGINT", 7)},
    {SHRED_MANIFEST_TYPE_DOUBLE, nonstd::string_view("DOUBLE", 6)},
    {SHRED_MANIFEST_TYPE_BOOLEAN, nonstd::string_view("BOOLEAN", 7)},
    {SHRED_MANIFEST_TYPE_VARCHAR_LIST, nonstd::string_view("VARCHAR[]", 9)},
    {SHRED_MANIFEST_TYPE_BIGINT_LIST, nonstd::string_view("BIGINT[]", 8)},
    {SHRED_MANIFEST_TYPE_UBIGINT_LIST, nonstd::string_view("UBIGINT[]", 9)},
    {SHRED_MANIFEST_TYPE_DOUBLE_LIST, nonstd::string_view("DOUBLE[]", 8)},
    {SHRED_MANIFEST_TYPE_BOOLEAN_LIST, nonstd::string_view("BOOLEAN[]", 9)},
};
static_assert(sizeof(SHRED_MANIFEST_COMPACT_TYPES) / sizeof(SHRED_MANIFEST_COMPACT_TYPES[0]) == 10,
              "compact shred manifest type table must cover all 10 non-EXTENDED codes");

inline nonstd::string_view ShredManifestCompactTypeName(uint8_t code) {
	for (auto &entry : SHRED_MANIFEST_COMPACT_TYPES) {
		if (entry.code == code) {
			return entry.name;
		}
	}
	throw InvalidInputException("malformed JSONO: unknown compact shred manifest type code");
}

inline uint64_t HashShredManifestSignatures(const std::vector<std::pair<std::string, std::string>> &signatures) {
	uint64_t h = HashMix64(HASH_SEED ^ uint64_t(signatures.size()), HASH_PRIME);
	for (auto &signature : signatures) {
		h = HashKey(h, nonstd::string_view(signature.first.data(), signature.first.size()));
		h = HashKey(h, nonstd::string_view(signature.second.data(), signature.second.size()));
	}
	return h;
}

// The shred manifest is the tail of the skips blob, after the checkpoint sections. It is either a
// full path/type entry list or a compact type-code entry list (see docs/jsono_format.md). Entries
// follow canonical shred order. A plain value writes no manifest (the skips blob ends at the
// checkpoints), which reads as zero entries.
// The manifest is the write-time record of which paths are NOT in the residual: a reader holding
// fewer (or differently typed) shreds than the manifest lists cannot reproduce the value — that row
// was narrowed by a raw struct cast — and must fail loud instead of silently dropping the value.

// The byte offset within a residual's skips blob where the shred-manifest tail begins (== skips_size
// for a value with no manifest). The fixed metadata header at the blob's front declares the span and
// checkpoint section sizes; whatever follows them is the manifest. Shared by JsonoView::ParseHeader
// and the soft-residual manifest strip so both agree on the boundary.
inline size_t JsonoSkipsManifestOffset(const uint8_t *skips, size_t skips_size) {
	if (skips_size == 0) {
		return 0;
	}
	if (skips_size < sizeof(ContainerMetadataHeader)) {
		// The writer always emits the fixed metadata header, so a shorter non-empty blob is corruption.
		throw InvalidInputException("malformed JSONO: metadata blob is shorter than its header");
	}
	ContainerMetadataHeader header;
	std::memcpy(&header, skips, sizeof(header));
	auto required = uint64_t(sizeof(ContainerMetadataHeader)) + uint64_t(header.span_count) * sizeof(uint32_t) +
	                uint64_t(header.span_count) * sizeof(ContainerSpan) +
	                uint64_t(header.checkpoint_index_count) * sizeof(ObjectCheckpointIndex) +
	                uint64_t(header.checkpoint_count) * sizeof(ObjectCursorCheckpoint);
	if (required > skips_size) {
		throw InvalidInputException("malformed JSONO: metadata blob is shorter than its declared spans");
	}
	return size_t(required);
}

// The shred-manifest framing walker, shared by the parse (ParseShredManifestBytes) and validate
// (ValidateShredManifestBytes) paths via a sink — the same pattern as DecodeScalarSlot's decode/skip
// twins. The walker owns how the tail's bytes frame into entries; the sink decides what to do with
// each one: collect it (ShredManifestCollectSink) or discard it (ShredManifestDiscardSink). It
// streams entries one at a time through the cursor's bounds-checked reads and NEVER sizes an
// allocation from the untrusted entry_count, so a corrupt over-long count throws
// InvalidInputException on the first over-read before any large allocation — keeping the validate
// path allocation-free (jsono_validate catches only InvalidInputException; a reserve(entry_count) on
// a bogus huge count would std::bad_alloc and crash instead of reporting corruption as `false`).
template <class SINK>
inline void WalkShredManifestBytes(const char *data, size_t size, SINK &sink) {
	if (size == 0) {
		return;
	}
	size_t cursor = 0;
	auto read_bytes = [&](void *dst, size_t bytes) {
		if (bytes > size - cursor) {
			throw InvalidInputException("malformed JSONO: shred manifest is shorter than its declared entries");
		}
		std::memcpy(dst, data + cursor, bytes);
		cursor += bytes;
	};
	uint32_t entry_count;
	read_bytes(&entry_count, sizeof(entry_count));
	bool compact_types = false;
	if (entry_count == SHRED_MANIFEST_COMPACT_TYPE_MARKER) {
		compact_types = true;
		read_bytes(&entry_count, sizeof(entry_count));
	}
	for (uint32_t i = 0; i < entry_count; i++) {
		uint16_t path_len;
		read_bytes(&path_len, sizeof(path_len));
		if (path_len > size - cursor) {
			throw InvalidInputException("malformed JSONO: shred manifest is shorter than its declared entries");
		}
		auto path = nonstd::string_view(data + cursor, path_len);
		cursor += path_len;
		if (compact_types) {
			uint8_t type_code;
			read_bytes(&type_code, sizeof(type_code));
			if (type_code != SHRED_MANIFEST_TYPE_EXTENDED) {
				sink.OnEntry(ShredManifestEntry {path, ShredManifestCompactTypeName(type_code)});
				continue;
			}
		}
		uint16_t type_len;
		read_bytes(&type_len, sizeof(type_len));
		if (type_len > size - cursor) {
			throw InvalidInputException("malformed JSONO: shred manifest is shorter than its declared entries");
		}
		auto type = nonstd::string_view(data + cursor, type_len);
		cursor += type_len;
		sink.OnEntry(ShredManifestEntry {path, type});
	}
	if (cursor != size) {
		throw InvalidInputException("malformed JSONO: shred manifest has trailing bytes");
	}
}

// Collecting sink: grows `entries` incrementally with push_back (never reserve(entry_count)), so the
// parse path never allocates beyond bytes the cursor has already bounds-checked.
struct ShredManifestCollectSink {
	std::vector<ShredManifestEntry> &entries;
	void OnEntry(ShredManifestEntry entry) {
		entries.push_back(entry);
	}
};

// Discarding sink: keeps the walker's bounds checks but throws each framed entry away, so the
// validate path stays allocation-free.
struct ShredManifestDiscardSink {
	void OnEntry(const ShredManifestEntry &) {
	}
};

// Parse a shred manifest from its raw tail bytes into `entries` (string_views into `data`).
// Returns the entry count; an empty tail is a value without a manifest. Throws on framing
// corruption (declared entries longer than the tail, or trailing bytes).
inline size_t ParseShredManifestBytes(const char *data, size_t size, std::vector<ShredManifestEntry> &entries) {
	entries.clear();
	ShredManifestCollectSink sink {entries};
	WalkShredManifestBytes(data, size, sink);
	return entries.size();
}

// Validate the framing of a shred-manifest tail without materializing its entries. Accepts exactly
// the same byte sequences as ParseShredManifestBytes; the discarding sink keeps the walk
// allocation-free (see WalkShredManifestBytes).
inline void ValidateShredManifestBytes(const char *data, size_t size) {
	ShredManifestDiscardSink sink;
	WalkShredManifestBytes(data, size, sink);
}

// Fingerprint an object's stored KEY slots (sorted order, length-prefixed) into a
// shape_hash. Path-agnostic: every object-emitting writer routes through this on
// the final KEY slots, so the stored hash is a pure function of the object's key
// set regardless of how the keys were produced.
inline uint64_t HashObjectKeySlots(const uint64_t *slots, size_t key_start, size_t child_count, const char *key_heap) {
	uint64_t h = HASH_SEED;
	for (size_t i = 0; i < child_count; i++) {
		auto payload = SlotPayload(slots[key_start + i]);
		h = HashKey(h, nonstd::string_view(key_heap + KeyOffset(payload), KeyLen(payload)));
	}
	return h;
}

inline uint64_t MakeContainerPayload(uint64_t container_id, uint64_t child_count) {
	return (container_id << CONTAINER_CHILD_COUNT_BITS) | (child_count & CONTAINER_CHILD_COUNT_MASK);
}

inline uint64_t ContainerId(uint64_t payload) {
	return payload >> CONTAINER_CHILD_COUNT_BITS;
}

inline uint64_t ContainerChildCount(uint64_t payload) {
	return payload & CONTAINER_CHILD_COUNT_MASK;
}

// i60 signed: the VAL_INT60 eligibility range. The value itself is stored as a
// full 64-bit two's-complement word in the `nums` stream (the slot is a pure
// tag), so the bound only decides the tag, not the stored width.
constexpr int64_t INT60_MIN = -(int64_t(1) << 59);
constexpr int64_t INT60_MAX = (int64_t(1) << 59) - 1;

inline bool FitsInt60(int64_t value) {
	return value >= INT60_MIN && value <= INT60_MAX;
}

// VAL_DEC60: value = mantissa / 10^scale, packed sign + scale + mantissa. The
// packed word is stored as this value's `nums` entry (top four bits zero); the
// slot is a pure tag.
//   bit 59      : sign (1 = negative mantissa)
//   bits 54..58 : scale  (0..22, 5 bits)
//   bits 0..53  : |mantissa| (< 2^53, 54-bit field)
// Lossless bounds (docs/jsono_format.md §VAL_DEC60): |mantissa| < 2^53 and
// scale <= 22, so both 10^scale and mantissa are exact as double and the
// division `mantissa / pow10[scale]` is correctly-rounded.
constexpr uint64_t DEC60_SCALE_MAX = 22;
constexpr uint64_t DEC60_MANTISSA_LIMIT = uint64_t(1) << 53;
constexpr uint64_t DEC60_MANTISSA_BITS = 54;
constexpr uint64_t DEC60_MANTISSA_MASK = (uint64_t(1) << DEC60_MANTISSA_BITS) - 1;
constexpr uint64_t DEC60_SCALE_SHIFT = 54;
constexpr uint64_t DEC60_SIGN_SHIFT = 59;

// Exact powers of ten as double, 10^0 .. 10^22 (all integers < 2^53 except the
// last two, which remain exactly representable — 10^22 is the largest exact one).
constexpr double DEC60_POW10[DEC60_SCALE_MAX + 1] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                                                     1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                                     1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

inline bool FitsDec60(uint64_t abs_mantissa, uint64_t scale) {
	return abs_mantissa < DEC60_MANTISSA_LIMIT && scale <= DEC60_SCALE_MAX;
}

inline uint64_t MakeDec60Payload(bool negative, uint64_t abs_mantissa, uint64_t scale) {
	return (uint64_t(negative) << DEC60_SIGN_SHIFT) | (scale << DEC60_SCALE_SHIFT) |
	       (abs_mantissa & DEC60_MANTISSA_MASK);
}

inline bool Dec60Negative(uint64_t payload) {
	return (payload >> DEC60_SIGN_SHIFT) & 1;
}

inline uint64_t Dec60Mantissa(uint64_t payload) {
	return payload & DEC60_MANTISSA_MASK;
}

inline uint64_t Dec60Scale(uint64_t payload) {
	return (payload >> DEC60_SCALE_SHIFT) & 0x1F;
}

// Reconstruct the double value strictly by division (never multiply by
// 10^-scale): mantissa and 10^scale are exact, IEEE division is
// correctly-rounded, so the result is bit-equal to parse("<text>").
inline double Dec60ToDouble(bool negative, uint64_t mantissa, uint64_t scale) {
	if (scale > DEC60_SCALE_MAX) {
		throw InvalidInputException("malformed JSONO: invalid DEC60 payload");
	}
	double value = double(mantissa);
	if (negative) {
		value = -value;
	}
	return value / DEC60_POW10[scale];
}

// Header (8 bytes) prepended to the slots BLOB.
constexpr size_t JSONO_HEADER_SIZE = 8;

struct JsonoHeader {
	uint32_t magic;
	uint8_t version;
	uint8_t flags;
	uint16_t reserved;
};

static_assert(sizeof(JsonoHeader) == JSONO_HEADER_SIZE, "JsonoHeader must be exactly 8 bytes");

// View over a single JSONO row. The six borrowed blobs (slots, key_heap,
// string_heap, skips, lengths, nums) must stay alive for the view's lifetime. No
// allocations, no copies — slot access reads via aligned memcpy from the
// slots buffer; heap lookups return string_views into the appropriate heap.
class JsonoView {
public:
	// An empty view (no header, zero slots): the state a row reader leaves its out-param in
	// for an absent value. Every accessor on it fails the same bounds checks as a zero-length
	// blob, so it can never read.
	JsonoView() = default;

	JsonoView(const char *slots, size_t slots_size, const char *key_heap, size_t key_heap_size, const char *string_heap,
	          size_t string_heap_size, const char *skips, size_t skips_size, const char *lengths, size_t lengths_size,
	          const char *nums, size_t nums_size)
	    : slots_(reinterpret_cast<const uint8_t *>(slots)), slots_size_(slots_size),
	      key_heap_(reinterpret_cast<const uint8_t *>(key_heap)), key_heap_size_(key_heap_size),
	      string_heap_(reinterpret_cast<const uint8_t *>(string_heap)), string_heap_size_(string_heap_size),
	      skips_(reinterpret_cast<const uint8_t *>(skips)), skips_size_(skips_size),
	      lengths_(reinterpret_cast<const uint8_t *>(lengths)), lengths_size_(lengths_size),
	      nums_(reinterpret_cast<const uint8_t *>(nums)), nums_size_(nums_size) {
	}

	// Returns false only for an *absent* value: the slots blob is too short to even hold a
	// header (a NULL/empty jsono_slots field), which callers map to SQL NULL. A real jsono
	// value always has a header plus at least one slot, so any blob that has a header but is
	// unreadable — wrong magic, a different format version, misaligned slots, or a metadata
	// blob shorter than its declared spans — is corruption or non-JSONO bytes and must fail
	// loud, never silently read as SQL NULL. The validate function recovers a boolean by
	// catching these (see ValidateJsonoBlob); read/extract paths let them propagate.
	bool ParseHeader() {
		if (slots_size_ < JSONO_HEADER_SIZE) {
			return false;
		}
		std::memcpy(&header_, slots_, JSONO_HEADER_SIZE);
		if (header_.magic != MAGIC) {
			throw InvalidInputException("not a JSONO value: header magic mismatch (the column holds non-JSONO bytes)");
		}
		if (header_.version != VERSION) {
			throw InvalidInputException("JSONO format version mismatch: stored value is v%d, this extension reads v%d. "
			                            "Re-materialize the jsono column with the current extension.",
			                            int(header_.version), int(VERSION));
		}
		auto slots_bytes = slots_size_ - JSONO_HEADER_SIZE;
		if (slots_bytes % sizeof(uint64_t) != 0) {
			throw InvalidInputException("malformed JSONO: slots blob is not a whole number of 8-byte slots");
		}
		if (lengths_size_ % sizeof(uint32_t) != 0) {
			throw InvalidInputException("malformed JSONO: lengths blob is not a whole number of 4-byte entries");
		}
		if (nums_size_ % sizeof(uint64_t) != 0) {
			throw InvalidInputException("malformed JSONO: nums blob is not a whole number of 8-byte entries");
		}
		manifest_offset_ = JsonoSkipsManifestOffset(skips_, skips_size_);
		if (skips_size_ >= sizeof(ContainerMetadataHeader)) {
			std::memcpy(&metadata_header_, skips_, sizeof(metadata_header_));
		} else {
			metadata_header_ = {};
		}
		slot_count_ = slots_bytes / sizeof(uint64_t);
		return true;
	}

	// Parse the shred manifest from the skips tail into `entries`. Returns the entry count
	// (zero for a plain value, whose skips blob ends at the checkpoints).
	size_t ReadShredManifest(std::vector<ShredManifestEntry> &entries) const {
		auto tail = ManifestTail();
		return ParseShredManifestBytes(tail.data(), tail.size(), entries);
	}

	// Raw bytes of the shred-manifest tail (empty for a value without one). The verification
	// layer (jsono_row_read.hpp) memoizes verdicts by these bytes: equal tails are the same
	// manifest, so a vector of same-spec rows verifies with one byte-compare per row.
	nonstd::string_view ManifestTail() const {
		return nonstd::string_view(reinterpret_cast<const char *>(skips_) + manifest_offset_,
		                           skips_size_ - manifest_offset_);
	}

	// True when this row's writer stripped at least one path out of the residual (cheap: a
	// single offset compare, no manifest parse).
	bool HasShredManifest() const {
		return manifest_offset_ < skips_size_;
	}

	uint8_t HeaderFlags() const {
		return header_.flags;
	}

	uint16_t HeaderReserved() const {
		return header_.reserved;
	}

	const ContainerMetadataHeader &MetadataHeader() const {
		return metadata_header_;
	}

	JSONO_ALWAYS_INLINE uint64_t SlotAt(size_t i) const {
		if (i >= slot_count_) {
			ThrowMalformed("malformed JSONO: slot index out of bounds");
		}
		uint64_t v;
		std::memcpy(&v, slots_ + JSONO_HEADER_SIZE + i * sizeof(uint64_t), sizeof(v));
		return v;
	}

	// Lookup for a KEY slot's payload — reads into key_heap.
	JSONO_ALWAYS_INLINE nonstd::string_view KeyAt(uint64_t payload) const {
		return KeyHeapSlice(KeyOffset(payload), KeyLen(payload));
	}

	// Slice key_heap directly, for bulk copies of contiguous key-byte runs.
	JSONO_ALWAYS_INLINE nonstd::string_view KeyHeapSlice(uint64_t offset, uint64_t len) const {
		if (offset > key_heap_size_ || len > key_heap_size_ - offset) {
			ThrowMalformed("malformed JSONO: key heap reference out of bounds");
		}
		return nonstd::string_view(reinterpret_cast<const char *>(key_heap_) + offset, len);
	}

	// Slice the next `len` bytes of string_heap starting at `string_cursor`.
	// Caller is responsible for advancing the cursor; the view itself is
	// stateless w.r.t. position.
	JSONO_ALWAYS_INLINE nonstd::string_view StringAt(size_t string_cursor, size_t len) const {
		if (string_cursor > string_heap_size_ || len > string_heap_size_ - string_cursor) {
			ThrowMalformed("malformed JSONO: string heap reference out of bounds");
		}
		return nonstd::string_view(reinterpret_cast<const char *>(string_heap_) + string_cursor, len);
	}

	// Sparse span lookup over the sorted container-id index. False when the container
	// stores no span (a small object or an empty array — skipped by a linear walk).
	// The ids are strictly ascending, so ids[i] >= i: the entry for `container_id`, if
	// stored, sits at index <= container_id. Near-root containers (the ids key lookups
	// hit hottest — the row root is always id 0) are therefore found by a short downward
	// scan from min(container_id, count-1), bounded by SPAN_HINT_LIMIT steps; larger ids
	// fall back to the binary search.
	JSONO_ALWAYS_INLINE bool TryContainerSpan(uint64_t container_id, ContainerSpan &span) const {
		size_t count = metadata_header_.span_count;
		if (count == 0) {
			return false;
		}
		auto ids_start = skips_ + sizeof(ContainerMetadataHeader);
		size_t lo;
		if (container_id < SPAN_HINT_LIMIT) {
			lo = container_id < count ? size_t(container_id) : count - 1;
			uint32_t id;
			std::memcpy(&id, ids_start + lo * sizeof(uint32_t), sizeof(id));
			while (id > container_id) {
				if (lo == 0) {
					return false;
				}
				lo--;
				std::memcpy(&id, ids_start + lo * sizeof(uint32_t), sizeof(id));
			}
			if (id != container_id) {
				return false;
			}
		} else {
			lo = 0;
			size_t hi = count;
			while (lo < hi) {
				auto mid = lo + (hi - lo) / 2;
				uint32_t id;
				std::memcpy(&id, ids_start + mid * sizeof(uint32_t), sizeof(id));
				if (id < container_id) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			if (lo >= count) {
				return false;
			}
			uint32_t id;
			std::memcpy(&id, ids_start + lo * sizeof(uint32_t), sizeof(id));
			if (id != container_id) {
				return false;
			}
		}
		auto offset = SpansOffset() + lo * sizeof(ContainerSpan);
		std::memcpy(&span, skips_ + offset, sizeof(span));
		return true;
	}

	// Span lookup for containers that must be spanned (non-empty arrays, objects with
	// child_count > OBJECT_CHECKPOINT_STRIDE): a missing span is corruption.
	JSONO_ALWAYS_INLINE ContainerSpan ContainerSpanAt(uint64_t container_id) const {
		ContainerSpan span;
		if (!TryContainerSpan(container_id, span)) {
			ThrowMalformed("malformed JSONO: container span missing");
		}
		return span;
	}

	uint32_t SpanIdAt(size_t index) const {
		if (index >= metadata_header_.span_count) {
			throw InvalidInputException("malformed JSONO: span id index out of bounds");
		}
		uint32_t id;
		std::memcpy(&id, skips_ + sizeof(ContainerMetadataHeader) + index * sizeof(uint32_t), sizeof(id));
		return id;
	}

	JSONO_ALWAYS_INLINE uint32_t LengthAt(size_t length_index) const {
		if (length_index >= LengthCount()) {
			ThrowMalformed("malformed JSONO: lengths stream index out of bounds");
		}
		uint32_t length;
		std::memcpy(&length, lengths_ + length_index * sizeof(uint32_t), sizeof(length));
		return length;
	}

	JSONO_ALWAYS_INLINE uint64_t NumAt(size_t num_index) const {
		if (num_index >= NumCount()) {
			ThrowMalformed("malformed JSONO: nums stream index out of bounds");
		}
		uint64_t value;
		std::memcpy(&value, nums_ + num_index * sizeof(uint64_t), sizeof(value));
		return value;
	}

	size_t LengthCount() const {
		return lengths_size_ / sizeof(uint32_t);
	}

	size_t NumCount() const {
		return nums_size_ / sizeof(uint64_t);
	}

	// Raw byte slices over the streams, for bulk copies of contiguous walk-order runs.
	const uint8_t *LengthsBytes(size_t length_index, size_t count) const {
		if (length_index > LengthCount() || count > LengthCount() - length_index) {
			throw InvalidInputException("malformed JSONO: lengths stream slice out of bounds");
		}
		return lengths_ + length_index * sizeof(uint32_t);
	}

	const uint8_t *NumsBytes(size_t num_index, size_t count) const {
		if (num_index > NumCount() || count > NumCount() - num_index) {
			throw InvalidInputException("malformed JSONO: nums stream slice out of bounds");
		}
		return nums_ + num_index * sizeof(uint64_t);
	}

	// Raw byte slices over the span / checkpoint record sections, for bulk copies of a
	// subtree's contiguous metadata ranges (a subtree's containers occupy a contiguous id
	// range, so its spans and checkpoints are contiguous record runs).
	const uint8_t *SpanRecordsBytes(size_t index, size_t count) const {
		if (index > size_t(metadata_header_.span_count) || count > size_t(metadata_header_.span_count) - index) {
			throw InvalidInputException("malformed JSONO: span record slice out of bounds");
		}
		return skips_ + SpansOffset() + index * sizeof(ContainerSpan);
	}

	const uint8_t *CheckpointRecordsBytes(size_t index, size_t count) const {
		if (index > size_t(metadata_header_.checkpoint_count) ||
		    count > size_t(metadata_header_.checkpoint_count) - index) {
			throw InvalidInputException("malformed JSONO: checkpoint record slice out of bounds");
		}
		return skips_ + CheckpointsOffset() + index * sizeof(ObjectCursorCheckpoint);
	}

	JSONO_ALWAYS_INLINE ObjectCursorCheckpoint ObjectCheckpointAt(uint32_t checkpoint_index) const {
		if (checkpoint_index >= metadata_header_.checkpoint_count) {
			ThrowMalformed("malformed JSONO: object cursor checkpoint index out of bounds");
		}
		auto offset = CheckpointsOffset() + size_t(checkpoint_index) * sizeof(ObjectCursorCheckpoint);
		ObjectCursorCheckpoint checkpoint;
		std::memcpy(&checkpoint, skips_ + offset, sizeof(checkpoint));
		return checkpoint;
	}

	ObjectCheckpointIndex ObjectCheckpointIndexAt(uint32_t checkpoint_index) const {
		if (checkpoint_index >= metadata_header_.checkpoint_index_count) {
			throw InvalidInputException("malformed JSONO: object checkpoint index out of bounds");
		}
		auto offset = CheckpointIndexOffset() + size_t(checkpoint_index) * sizeof(ObjectCheckpointIndex);
		ObjectCheckpointIndex index;
		std::memcpy(&index, skips_ + offset, sizeof(index));
		return index;
	}

	ObjectCheckpointIndex ObjectCheckpointIndexForContainer(uint32_t container_id) const {
		size_t lo = 0;
		size_t hi = metadata_header_.checkpoint_index_count;
		auto index_start = skips_ + CheckpointIndexOffset();
		while (lo < hi) {
			auto mid = lo + (hi - lo) / 2;
			ObjectCheckpointIndex index;
			std::memcpy(&index, index_start + mid * sizeof(ObjectCheckpointIndex), sizeof(index));
			if (index.container_id < container_id) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		if (lo >= metadata_header_.checkpoint_index_count) {
			return ObjectCheckpointIndex {container_id, NO_OBJECT_CHECKPOINTS, 0, 0};
		}
		ObjectCheckpointIndex index;
		std::memcpy(&index, index_start + lo * sizeof(ObjectCheckpointIndex), sizeof(index));
		if (index.container_id != container_id) {
			return ObjectCheckpointIndex {container_id, NO_OBJECT_CHECKPOINTS, 0, 0};
		}
		return index;
	}

	size_t StringHeapSize() const {
		return string_heap_size_;
	}

	size_t SkipsSize() const {
		return skips_size_;
	}

	size_t Slots() const {
		return slot_count_;
	}

private:
	// Out-of-line throw for the always-inlined accessors: the hot path stays a
	// compare-and-load, the exception machinery lives here.
	[[noreturn]] JSONO_COLD static void ThrowMalformed(const char *message) {
		throw InvalidInputException("%s", message);
	}

	size_t SpansOffset() const {
		return sizeof(ContainerMetadataHeader) + size_t(metadata_header_.span_count) * sizeof(uint32_t);
	}

	size_t CheckpointIndexOffset() const {
		return SpansOffset() + size_t(metadata_header_.span_count) * sizeof(ContainerSpan);
	}

	size_t CheckpointsOffset() const {
		return CheckpointIndexOffset() +
		       size_t(metadata_header_.checkpoint_index_count) * sizeof(ObjectCheckpointIndex);
	}

	const uint8_t *slots_ = nullptr;
	size_t slots_size_ = 0;
	const uint8_t *key_heap_ = nullptr;
	size_t key_heap_size_ = 0;
	const uint8_t *string_heap_ = nullptr;
	size_t string_heap_size_ = 0;
	const uint8_t *skips_ = nullptr;
	size_t skips_size_ = 0;
	const uint8_t *lengths_ = nullptr;
	size_t lengths_size_ = 0;
	const uint8_t *nums_ = nullptr;
	size_t nums_size_ = 0;
	JsonoHeader header_ {};
	ContainerMetadataHeader metadata_header_ {};
	size_t slot_count_ = 0;
	size_t manifest_offset_ = 0;
};

// Verify a row's parsed shred-manifest entries against the shreds the reading type actually
// carries, given as (path, type-string) pairs. Every manifest entry must have a shred of the same
// path and type: the manifest lists the paths stripped out of this residual at write time, so a
// missing shred means the value was narrowed by a raw struct cast (the shred was dropped) and a
// retyped shred means its value was silently converted — either way the original value cannot be
// reproduced and the read must fail loud. Extra shreds are fine (a widening cast NULL-fills them;
// readers fall back to the residual). Callers go through the row-read layer
// (jsono_row_read.hpp), which parses and memoizes the entries.
inline void VerifyShredManifestEntries(const std::vector<ShredManifestEntry> &manifest,
                                       const std::vector<std::pair<std::string, std::string>> &shred_signatures) {
	for (auto &entry : manifest) {
		bool found = false;
		for (auto &shred : shred_signatures) {
			if (entry.path == nonstd::string_view(shred.first.data(), shred.first.size())) {
				if (entry.type != nonstd::string_view(shred.second.data(), shred.second.size())) {
					throw InvalidInputException(
					    "JSONO: row was shredded with shred '%s %s' but the column carries it as a different type; "
					    "the shred value was converted by a raw struct cast and the original cannot be reproduced",
					    std::string(entry.path).c_str(), std::string(entry.type).c_str());
				}
				found = true;
				break;
			}
		}
		if (!found) {
			throw InvalidInputException(
			    "JSONO: row was shredded with shred '%s %s' but the column no longer carries that shred; the value "
			    "was narrowed by a raw struct cast and cannot be read losslessly. Reshred through "
			    "jsono(value, shredding := {...}) (the extension optimizer does this automatically)",
			    std::string(entry.path).c_str(), std::string(entry.type).c_str());
		}
	}
}

} // namespace jsono

void RegisterJsonoType(ExtensionLoader &loader);

} // namespace duckdb
