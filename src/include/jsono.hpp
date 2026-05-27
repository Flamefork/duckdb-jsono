#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

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

// Logical type name registered with DuckDB. The physical layout is a STRUCT of
// four BLOB children (exact shape: see JsonoType()); the alias survives
// round-trips through DuckDB-native storage (catalog metadata) but is lost
// through plain read_parquet, which delivers the structurally-compatible
// STRUCT.
//
// Split heap design: keys (which are byte-identical across rows that share
// a schema) live in key_heap; per-row string values live in string_heap.
// Parquet column-dictionary encoding then dedups key_heap to a single
// entry per column chunk for shape-stable data, and KEY slot byte patterns
// become identical across rows (the offset into key_heap for "browser" is
// always the same byte position when the keys section is the same).
constexpr const char *JSONO_TYPE_NAME = "JSONO";

// Materialise the JSONO logical type. Helper so writers/readers and
// pluck-on-JSONO can agree on the exact STRUCT shape.
LogicalType JsonoType();

// True when `type` is the JSONO STRUCT — by alias, or structurally (the four
// jsono_* BLOB children survive read_parquet, which drops the alias). Single
// source of truth for the field-name contract shared by jsono.cpp and
// jsono_ops.cpp.
bool IsJsonoType(const LogicalType &type);

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
constexpr uint8_t VERSION = 0x01;

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

struct ContainerMetadataHeader {
	uint32_t container_count;
	uint32_t checkpoint_index_count;
	uint32_t checkpoint_count;
};

// ContainerSpan carries the subtree slot span and string_heap byte span.
struct ContainerSpan {
	uint32_t slot_span;
	uint32_t string_byte_span;
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
static_assert(sizeof(ContainerSpan) == 8, "ContainerSpan must be exactly 8 bytes");
static_assert(sizeof(ObjectCheckpointIndex) == 12, "ObjectCheckpointIndex must be exactly 12 bytes");
static_assert(sizeof(ObjectCursorCheckpoint) == 8, "ObjectCursorCheckpoint must be exactly 8 bytes");

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

inline double DecodeDec60(uint64_t payload) {
	return Dec60ToDouble(Dec60Negative(payload), Dec60Mantissa(payload), Dec60Scale(payload));
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

	bool ParseHeader() {
		if (slots_size_ < JSONO_HEADER_SIZE) {
			return false;
		}
		std::memcpy(&header_, slots_, JSONO_HEADER_SIZE);
		if (header_.magic != MAGIC || header_.version != VERSION) {
			return false;
		}
		auto slots_bytes = slots_size_ - JSONO_HEADER_SIZE;
		if (slots_bytes % sizeof(uint64_t) != 0) {
			return false;
		}
		if (skips_size_ >= sizeof(ContainerMetadataHeader)) {
			std::memcpy(&metadata_header_, skips_, sizeof(metadata_header_));
			auto spans_bytes = uint64_t(metadata_header_.container_count) * sizeof(ContainerSpan);
			auto index_bytes = uint64_t(metadata_header_.checkpoint_index_count) * sizeof(ObjectCheckpointIndex);
			auto checkpoints_bytes = uint64_t(metadata_header_.checkpoint_count) * sizeof(ObjectCursorCheckpoint);
			auto required_bytes =
			    uint64_t(sizeof(ContainerMetadataHeader)) + spans_bytes + index_bytes + checkpoints_bytes;
			if (required_bytes > skips_size_) {
				return false;
			}
		} else {
			metadata_header_ = {};
		}
		slot_count_ = slots_bytes / sizeof(uint64_t);
		return true;
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
};

} // namespace jsono

void RegisterJsonoType(ExtensionLoader &loader);

} // namespace duckdb
