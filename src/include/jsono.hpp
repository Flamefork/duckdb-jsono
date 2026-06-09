#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"

#include "string_view.hpp"

// Force inlining where a small decode is called once per scalar from the hot
// readers: left as a plain call the compiler keeps it out-of-line and pays a
// per-scalar 72-byte struct return; inlined, the struct is scalarized away.
#if defined(__GNUC__) || defined(__clang__)
#define JSONO_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define JSONO_ALWAYS_INLINE inline
#endif

namespace duckdb {

class ExtensionLoader;
class ClientContext;

// A jsono value is physically a nested STRUCT with exactly one layout field, named "jsono" for
// both plain and shredded values. Inside the layout field live a `body` STRUCT of four BLOBs (the
// binary JSONO body, see JsonoBodyStructType) and, for shredded values, a `shreds` STRUCT of typed
// shred columns. The single layout name is deliberate: DuckDB reconciles set-operation branch
// types by field name (CombineStructTypes), so differently-shredded branches merge into one
// ordinary shredded type whose `shreds` is the union of the branches' shred sets — never a
// field-superset "union-merged" double struct. The struct cast NULL-fills a branch's missing
// shreds, which is exactly the writer's own "value stayed in the residual" encoding, so the merged
// value reads losslessly by construction. The flip side — a by-name cast silently dropping shreds
// when the target carries fewer shreds — is caught at read time by the shred manifest stored in the
// residual (see docs/jsono_format.md), not by the type system. There is no user-defined
// logical-type alias: such an alias cannot survive Parquet (stripped) or DuckLake (rejected), so
// every stored value — DuckDB-native, Parquet or DuckLake — is just this nested STRUCT, recognised
// structurally (see TryParseJsonoLayoutType / IsJsonoType). Dispatch is structural rather than
// name-based; no JSON key or shred path is reserved, because user paths live only inside `.shreds`.
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

// The physical plain JSONO STRUCT: STRUCT("jsono" STRUCT(body STRUCT(4 BLOB))). There is no
// logical-type alias on top (DuckLake/Parquet reject user-defined aliases, so jsono carries none);
// this is the single source of truth for the plain layout. `jsono_storage_type()` exposes its DDL
// string so writers can declare storage columns without hardcoding the fields.
LogicalType JsonoRawStructType();

// The four-BLOB body STRUCT shared by every layout: STRUCT(slots BLOB, key_heap BLOB,
// string_heap BLOB, skips BLOB). The binary JSONO body lives here, identical for plain and shredded.
LogicalType JsonoBodyStructType();

// The physical STRUCT of a shredded JSONO value: STRUCT("jsono" STRUCT(body STRUCT(4 BLOB),
// shreds STRUCT(<shred columns>))). One shred column per `shreds` entry (each a path name and its
// shred type). This is the single source of truth for the shredded shape: the constructor and
// `jsono_storage_type` both build it here, so a column declared from the same shreds is
// byte-identical to the value's type. Shred field order follows `shreds`; callers pass them in
// canonical (name-sorted) order so the type is a pure function of the shred set, and they write
// shreds in that same order (shreds are written by index, so field and value order must agree).
LogicalType JsonoShreddedStructType(const child_list_t<LogicalType> &shreds);

// The top-level field name of every layout (plain and shredded): "jsono".
string JsonoLayoutName();

// The classification of one JSONO layout field.
enum class JsonoLayoutKind : uint8_t { Plain, Shredded };

// A parsed JSONO layout field: its top-level name, the inner STRUCT(body[, shreds]) type, and its
// shred (path, type) columns (empty for plain). The single grammar all predicates read against.
struct JsonoLayoutType {
	JsonoLayoutKind kind;
	string layout_name;
	LogicalType layout_type;
	child_list_t<LogicalType> shreds;
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

// Reconstruct a shredded JSONO `input` (four-BLOB residual + shred columns) into the
// lossless plain JSONO `result` by overlaying each row's shred values onto its residual.
// This is how a shredded value becomes usable where a plain JSONO is required (an implicit
// cast, an INSERT into a plain JSONO column) without dropping the shred data. Top-level shreds
// overlay in one flat pass; a nested-path shred overlays its single key chain onto the residual.
void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result);

// Overlay only `shreds` (shred indices over the shred set) of the shredded `input` onto its
// residual, producing plain JSONO. The single-pass narrowing reshred folds the shreds the target
// type drops back into the residual with this, leaving the kept shreds untouched.
void JsonoOverlayShredsToPlain(Vector &input, idx_t count, const vector<idx_t> &shreds, Vector &result);

// Parse a VARCHAR/JSON `source` vector into a plain JSONO `result` vector, throwing on
// invalid JSON (the jsono() text contract). Exposed so the shred constructor overload can
// parse-then-shred in one pass without duplicating the parser.
void JsonoParseTextVector(Vector &source, idx_t count, Vector &result);

//===--------------------------------------------------------------------===//
// JSONO binary format
//
// A JSONO value is physically split into four BLOB children:
//   slots        — [Header (8 bytes)] [Slots (n × 8 bytes)]
//   key_heap     — key bytes referenced by KEY slots
//   string_heap  — string bytes consumed by STRING slots in walk order
//   skips        — [ContainerMetadataHeader] [ContainerSpan...]
//                  [ObjectCheckpointIndex...] [ObjectCursorCheckpoint...]
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
constexpr uint8_t VERSION = 0x03;

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
// concrete extended encoding; the value bytes live in the following slot(s).
namespace ext_subtype {
constexpr uint64_t INT64 = 0;  // signed 2^59..2^63, one trailing slot
constexpr uint64_t UINT64 = 1; // unsigned 2^63..2^64-1, one trailing slot
constexpr uint64_t DOUBLE = 2; // IEEE double bits (jsono(STRUCT) path), one trailing slot
constexpr uint64_t NUMBER = 3; // raw bignum/high-precision text in string_heap
constexpr uint64_t COUNT = 4;
} // namespace ext_subtype

// Number of trailing value slots per subtype, kept as data so readers count
// slots through one table instead of branching on each subtype. NUMBER stores
// its value in the heap, so it carries no trailing slot.
constexpr uint8_t EXT_SUBTYPE_TRAILING_SLOTS[ext_subtype::COUNT] = {
    1, // INT64
    1, // UINT64
    1, // DOUBLE
    0, // NUMBER (text in string_heap, no trailing slot)
};

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

// VAL_EXT/NUMBER payload: subtype byte (low 8 bits) plus the raw-text byte
// length in the remaining payload bits. The text itself lives in string_heap
// and is consumed in walk order exactly like VAL_STR_HEAP, so NUMBER carries
// no trailing slot.
constexpr uint64_t EXT_NUMBER_LEN_SHIFT = 8;

inline uint64_t MakeNumberExtSlot(uint64_t text_len) {
	return MakeSlot(tag::VAL_EXT, (text_len << EXT_NUMBER_LEN_SHIFT) | ext_subtype::NUMBER);
}

inline uint64_t NumberExtLen(uint64_t slot) {
	return SlotPayload(slot) >> EXT_NUMBER_LEN_SHIFT;
}

// Total slots a VAL_EXT value occupies: the VAL_EXT slot plus its trailing
// value slots.
inline size_t ExtSlotCount(uint64_t subtype) {
	if (subtype >= ext_subtype::COUNT) {
		throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
	}
	return 1 + size_t(EXT_SUBTYPE_TRAILING_SLOTS[subtype]);
}

// KEY payload layout: heap_offset (36 bits high) << 24 | heap_len (24 bits low).
// KEYs still carry their heap offset because key_heap is dict-encoded by
// Parquet (every row has the same key set in the same sorted order, so the
// dictionary collapses to a single entry per column chunk and the offsets
// are byte-identical across rows already).
//
// STRING payload layout: the full 60-bit payload holds heap_len. The byte
// offset into string_heap is implicit — the reader maintains a cursor as it
// walks the JSONO and advances by heap_len on each STRING slot.
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

inline uint64_t MakeStringPayload(uint64_t heap_len) {
	return heap_len & PAYLOAD_MASK;
}

inline uint64_t StringLen(uint64_t payload) {
	return payload;
}

// OBJ_START/ARR_START payload: container id into skips plus inline
// child count. Only OBJ_START uses child_count; arrays store 0.
constexpr uint64_t CONTAINER_CHILD_COUNT_BITS = 24;
constexpr uint64_t CONTAINER_CHILD_COUNT_MASK = (uint64_t(1) << CONTAINER_CHILD_COUNT_BITS) - 1;
constexpr uint64_t CONTAINER_ID_MAX = PAYLOAD_MASK >> CONTAINER_CHILD_COUNT_BITS;

constexpr uint32_t NO_OBJECT_CHECKPOINTS = std::numeric_limits<uint32_t>::max();
constexpr uint32_t OBJECT_CHECKPOINT_STRIDE = 16;
constexpr uint16_t LARGE_OBJECT_CHECKPOINT_STRIDE = 16;
constexpr uint32_t LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT = 64;

// Whether this translation unit is built with AddressSanitizer instrumentation.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define JSONO_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define JSONO_ADDRESS_SANITIZER 1
#endif

// Maximum container nesting accepted by the writer and the recursive readers.
// The tape walkers recurse per nesting level, so an unbounded depth overflows
// the C++ stack; this bound makes the guard trip before that happens. Both values
// stay above any realistic document (real JSON nests a few dozen levels deep). A
// sanitizer build gets a far lower bound: AddressSanitizer's fat frames sit atop a
// deep execution pipeline on a ~512 KiB macOS worker-thread stack, and the
// over-limit exception is itself serialized (yyjson) at the throw site, so the
// guard must fire much sooner there to stay ahead of a stack overflow.
#ifdef JSONO_ADDRESS_SANITIZER
constexpr size_t JSONO_MAX_NESTING_DEPTH = 50;
#else
constexpr size_t JSONO_MAX_NESTING_DEPTH = 1000;
#endif

// Shape-hash helpers — short-input wyhash-inspired mixer. Both the writer (per
// object shape fingerprint stored in ContainerSpan) and the DOM builder (DOM-order
// key fingerprint for its shape cache) hash key bytes; one definition shared here.
// 64-bit collision risk is negligible (~5e-20 per object) at any realistic scale,
// so a matching shape_hash is treated as authoritative without re-validation.
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
	uint32_t container_count;
	uint32_t checkpoint_index_count;
	uint32_t checkpoint_count;
};

// ContainerSpan carries the subtree slot span, string_heap byte span, and (for
// objects) a shape_hash: the fingerprint of the object's sorted key sequence.
// Arrays store shape_hash = 0 (never read — the rank cache only runs on objects).
struct ContainerSpan {
	uint32_t slot_span;
	uint32_t string_byte_span;
	uint64_t shape_hash;
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
};

static_assert(sizeof(ContainerMetadataHeader) == 12, "ContainerMetadataHeader must be exactly 12 bytes");
static_assert(sizeof(ContainerSpan) == 16, "ContainerSpan must be exactly 16 bytes");
static_assert(sizeof(ObjectCheckpointIndex) == 12, "ObjectCheckpointIndex must be exactly 12 bytes");
static_assert(sizeof(ObjectCursorCheckpoint) == 8, "ObjectCursorCheckpoint must be exactly 8 bytes");

// One shred-manifest entry: a path the shred writer stripped out of this row's residual (its value
// lives only in the shred) and the shred type it was written as. The views point into the skips blob.
struct ShredManifestEntry {
	nonstd::string_view path;
	nonstd::string_view type;
};

// The shred manifest is the tail of the skips blob, after the checkpoint sections:
//   u32 entry_count, then per entry: u16 path_len, path bytes, u16 type_len, type bytes.
// Entries are sorted by path (the shred writer emits fields in canonical order). A plain value
// writes no manifest (the skips blob ends at the checkpoints), which reads as zero entries.
// The manifest is the write-time record of which paths are NOT in the residual: a reader holding
// fewer (or differently typed) shreds than the manifest lists cannot reproduce the value — that row
// was narrowed by a raw struct cast — and must fail loud instead of silently dropping the value.

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

// i60 signed. Sign-extends from bit 59.
constexpr int64_t INT60_MIN = -(int64_t(1) << 59);
constexpr int64_t INT60_MAX = (int64_t(1) << 59) - 1;

inline uint64_t EncodeInt60(int64_t value) {
	return uint64_t(value) & PAYLOAD_MASK;
}

inline int64_t DecodeInt60(uint64_t payload) {
	if (payload & (uint64_t(1) << 59)) {
		return int64_t(payload | (uint64_t(0xF) << 60));
	}
	return int64_t(payload);
}

inline bool FitsInt60(int64_t value) {
	return value >= INT60_MIN && value <= INT60_MAX;
}

// VAL_DEC60: value = mantissa / 10^scale, packed sign + scale + mantissa.
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

// View over a single JSONO row. The four borrowed blobs (key_heap,
// string_heap, slots, skips) must stay alive for the view's lifetime. No
// allocations, no copies — slot access reads via aligned memcpy from the
// slots buffer; heap lookups return string_views into the appropriate heap.
class JsonoView {
public:
	JsonoView(const char *slots, size_t slots_size, const char *key_heap, size_t key_heap_size, const char *string_heap,
	          size_t string_heap_size, const char *skips, size_t skips_size)
	    : slots_(reinterpret_cast<const uint8_t *>(slots)), slots_size_(slots_size),
	      key_heap_(reinterpret_cast<const uint8_t *>(key_heap)), key_heap_size_(key_heap_size),
	      string_heap_(reinterpret_cast<const uint8_t *>(string_heap)), string_heap_size_(string_heap_size),
	      skips_(reinterpret_cast<const uint8_t *>(skips)), skips_size_(skips_size) {
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
		if (skips_size_ >= sizeof(ContainerMetadataHeader)) {
			std::memcpy(&metadata_header_, skips_, sizeof(metadata_header_));
			auto spans_bytes = uint64_t(metadata_header_.container_count) * sizeof(ContainerSpan);
			auto index_bytes = uint64_t(metadata_header_.checkpoint_index_count) * sizeof(ObjectCheckpointIndex);
			auto checkpoints_bytes = uint64_t(metadata_header_.checkpoint_count) * sizeof(ObjectCursorCheckpoint);
			auto required_bytes =
			    uint64_t(sizeof(ContainerMetadataHeader)) + spans_bytes + index_bytes + checkpoints_bytes;
			if (required_bytes > skips_size_) {
				throw InvalidInputException("malformed JSONO: metadata blob is shorter than its declared spans");
			}
			manifest_offset_ = size_t(required_bytes);
		} else {
			metadata_header_ = {};
			manifest_offset_ = skips_size_;
		}
		slot_count_ = slots_bytes / sizeof(uint64_t);
		return true;
	}

	// Parse the shred manifest from the skips tail into `entries`. Returns the entry count
	// (zero for a plain value, whose skips blob ends at the checkpoints).
	size_t ReadShredManifest(std::vector<ShredManifestEntry> &entries) const {
		entries.clear();
		if (manifest_offset_ >= skips_size_) {
			return 0;
		}
		auto cursor = manifest_offset_;
		auto read_bytes = [&](void *dst, size_t bytes) {
			if (bytes > skips_size_ - cursor) {
				throw InvalidInputException("malformed JSONO: shred manifest is shorter than its declared entries");
			}
			std::memcpy(dst, skips_ + cursor, bytes);
			cursor += bytes;
		};
		uint32_t entry_count;
		read_bytes(&entry_count, sizeof(entry_count));
		entries.reserve(entry_count);
		for (uint32_t i = 0; i < entry_count; i++) {
			uint16_t path_len;
			read_bytes(&path_len, sizeof(path_len));
			if (path_len > skips_size_ - cursor) {
				throw InvalidInputException("malformed JSONO: shred manifest is shorter than its declared entries");
			}
			auto path = nonstd::string_view(reinterpret_cast<const char *>(skips_) + cursor, path_len);
			cursor += path_len;
			uint16_t type_len;
			read_bytes(&type_len, sizeof(type_len));
			if (type_len > skips_size_ - cursor) {
				throw InvalidInputException("malformed JSONO: shred manifest is shorter than its declared entries");
			}
			auto type = nonstd::string_view(reinterpret_cast<const char *>(skips_) + cursor, type_len);
			cursor += type_len;
			entries.push_back(ShredManifestEntry {path, type});
		}
		if (cursor != skips_size_) {
			throw InvalidInputException("malformed JSONO: shred manifest has trailing bytes");
		}
		return entries.size();
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

	uint64_t SlotAt(size_t i) const {
		if (i >= slot_count_) {
			throw InvalidInputException("malformed JSONO: slot index out of bounds");
		}
		uint64_t v;
		std::memcpy(&v, slots_ + JSONO_HEADER_SIZE + i * sizeof(uint64_t), sizeof(v));
		return v;
	}

	// Lookup for a KEY slot's payload — reads into key_heap.
	nonstd::string_view KeyAt(uint64_t payload) const {
		auto off = KeyOffset(payload);
		auto len = KeyLen(payload);
		if (off > key_heap_size_ || len > key_heap_size_ - off) {
			throw InvalidInputException("malformed JSONO: key heap reference out of bounds");
		}
		return nonstd::string_view(reinterpret_cast<const char *>(key_heap_) + off, len);
	}

	// Slice the next `len` bytes of string_heap starting at `string_cursor`.
	// Caller is responsible for advancing the cursor; the view itself is
	// stateless w.r.t. position.
	nonstd::string_view StringAt(size_t string_cursor, size_t len) const {
		if (string_cursor > string_heap_size_ || len > string_heap_size_ - string_cursor) {
			throw InvalidInputException("malformed JSONO: string heap reference out of bounds");
		}
		return nonstd::string_view(reinterpret_cast<const char *>(string_heap_) + string_cursor, len);
	}

	ContainerSpan ContainerSpanAt(uint64_t container_id) const {
		if (container_id >= metadata_header_.container_count) {
			throw InvalidInputException("malformed JSONO: container span index out of bounds");
		}
		auto offset = sizeof(ContainerMetadataHeader) + size_t(container_id) * sizeof(ContainerSpan);
		ContainerSpan span;
		std::memcpy(&span, skips_ + offset, sizeof(span));
		return span;
	}

	ObjectCursorCheckpoint ObjectCheckpointAt(uint32_t checkpoint_index) const {
		if (checkpoint_index >= metadata_header_.checkpoint_count) {
			throw InvalidInputException("malformed JSONO: object cursor checkpoint index out of bounds");
		}
		auto spans_bytes = size_t(metadata_header_.container_count) * sizeof(ContainerSpan);
		auto index_bytes = size_t(metadata_header_.checkpoint_index_count) * sizeof(ObjectCheckpointIndex);
		auto offset = sizeof(ContainerMetadataHeader) + spans_bytes + index_bytes +
		              size_t(checkpoint_index) * sizeof(ObjectCursorCheckpoint);
		ObjectCursorCheckpoint checkpoint;
		std::memcpy(&checkpoint, skips_ + offset, sizeof(checkpoint));
		return checkpoint;
	}

	ObjectCheckpointIndex ObjectCheckpointIndexAt(uint32_t checkpoint_index) const {
		if (checkpoint_index >= metadata_header_.checkpoint_index_count) {
			throw InvalidInputException("malformed JSONO: object checkpoint index out of bounds");
		}
		auto offset = sizeof(ContainerMetadataHeader) +
		              size_t(metadata_header_.container_count) * sizeof(ContainerSpan) +
		              size_t(checkpoint_index) * sizeof(ObjectCheckpointIndex);
		ObjectCheckpointIndex index;
		std::memcpy(&index, skips_ + offset, sizeof(index));
		return index;
	}

	ObjectCheckpointIndex ObjectCheckpointIndexForContainer(uint32_t container_id) const {
		size_t lo = 0;
		size_t hi = metadata_header_.checkpoint_index_count;
		auto index_start =
		    skips_ + sizeof(ContainerMetadataHeader) + size_t(metadata_header_.container_count) * sizeof(ContainerSpan);
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
	const uint8_t *slots_;
	size_t slots_size_;
	const uint8_t *key_heap_;
	size_t key_heap_size_;
	const uint8_t *string_heap_;
	size_t string_heap_size_;
	const uint8_t *skips_;
	size_t skips_size_;
	JsonoHeader header_ {};
	ContainerMetadataHeader metadata_header_ {};
	size_t slot_count_ = 0;
	size_t manifest_offset_ = 0;
};

// Verify a row's shred manifest against the shreds the reading type actually carries, given as
// (path, type-string) pairs. Every manifest entry must have a shred of the same path and type:
// the manifest lists the paths stripped out of this residual at write time, so a missing shred
// means the value was narrowed by a raw struct cast (the shred was dropped) and a retyped shred
// means its value was silently converted — either way the original value cannot be reproduced
// and the read must fail loud. Extra shreds are fine (a widening cast NULL-fills them; readers
// fall back to the residual). `manifest` is caller-provided scratch to keep the per-row parse
// allocation-free.
inline void VerifyShredManifest(const JsonoView &view,
                                const std::vector<std::pair<std::string, std::string>> &shred_signatures,
                                std::vector<ShredManifestEntry> &manifest) {
	// Always reset the caller's scratch: a row without a manifest must not inherit the previous
	// row's entries (callers also read `manifest` after this to learn the row's stripped paths).
	manifest.clear();
	if (!view.HasShredManifest()) {
		return;
	}
	view.ReadShredManifest(manifest);
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
