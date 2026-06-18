#pragma once

#include "jsono.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/types/vector.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace duckdb {
namespace jsono {

// The number of body BLOBs and their fixed child order inside the body STRUCT.
constexpr idx_t BODY_BLOB_COUNT = 6;
constexpr idx_t BODY_SLOTS = 0;
constexpr idx_t BODY_KEY_HEAP = 1;
constexpr idx_t BODY_STRING_HEAP = 2;
constexpr idx_t BODY_SKIPS = 3;
constexpr idx_t BODY_LENGTHS = 4;
constexpr idx_t BODY_NUMS = 5;

struct JsonoBuilder {
	std::vector<uint64_t> slots;
	std::vector<char> key_heap;
	std::vector<char> string_heap;
	std::vector<uint32_t> lengths;
	std::vector<uint64_t> nums;
	// Sparse span storage: spans exist only for non-empty arrays and objects with
	// child_count > OBJECT_CHECKPOINT_STRIDE. `span_ids` holds their container ids; it is sorted by
	// construction (ids are assigned in container-start order, and a span is pushed
	// either at the container's own start or at its first container child's start —
	// at which point its subtree has produced no spans yet). `skips` is the parallel
	// span array.
	std::vector<uint32_t> span_ids;
	std::vector<ContainerSpan> skips;
	std::vector<ObjectCheckpointIndex> object_checkpoint_index;
	std::vector<ObjectCursorCheckpoint> object_checkpoints;
	uint64_t container_count = 0;

	static constexpr size_t NO_SPAN = std::numeric_limits<size_t>::max();

	struct OpenContainer {
		uint64_t id;
		uint64_t tag;
		uint32_t child_count;
		uint32_t checkpoint_offset;
		uint16_t checkpoint_stride;
		uint32_t next_child_index;
		size_t span_index;
		size_t start_slot;
		size_t start_string;
		size_t start_length;
		size_t start_num;
	};

	std::vector<OpenContainer> open_containers;

	void Reset() {
		slots.clear();
		key_heap.clear();
		string_heap.clear();
		lengths.clear();
		nums.clear();
		span_ids.clear();
		skips.clear();
		object_checkpoint_index.clear();
		object_checkpoints.clear();
		open_containers.clear();
		container_count = 0;
	}

	// Push a KEY slot referencing key bytes already present at [offset, offset+len).
	// KEY payload packs a 36-bit heap offset and 24-bit length; reject rather than
	// silently truncate.
	void PushKeySlot(uint64_t offset, uint64_t len) {
		if (len > HEAP_LEN_MASK) {
			throw InvalidInputException("jsono: object key exceeds maximum length");
		}
		if (offset > (PAYLOAD_MASK >> HEAP_LEN_BITS)) {
			throw InvalidInputException("jsono: key heap exceeds storage limits");
		}
		slots.push_back(MakeSlot(tag::KEY, MakeKeyPayload(offset, len)));
	}

	void EmitKeySlot(nonstd::string_view sv) {
		auto offset = key_heap.size();
		PushKeySlot(uint64_t(offset), uint64_t(sv.size()));
		key_heap.insert(key_heap.end(), sv.begin(), sv.end());
	}

	// Append one `lengths` entry. The fixed u32 width is what keeps LengthAt(i) O(1)
	// for checkpoint jumps, so longer payloads are rejected rather than widened.
	void PushLength(size_t len) {
		if (len > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: string value exceeds storage limits");
		}
		lengths.push_back(uint32_t(len));
	}

	void EmitString(nonstd::string_view s) {
		PushLength(s.size());
		string_heap.insert(string_heap.end(), s.begin(), s.end());
		slots.push_back(MakeSlot(tag::VAL_STR_HEAP, 0));
	}

	void EmitInt(int64_t v) {
		slots.push_back(FitsInt60(v) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64));
		nums.push_back(uint64_t(v));
	}

	void EmitUInt(uint64_t v) {
		// Stay on the encoding ladder: an int64-representable value takes INT60/VAL_EXT-INT64 so
		// its slot tag never depends on which writer produced it; only 2^63..2^64-1 is UINT64.
		if (v <= uint64_t(std::numeric_limits<int64_t>::max())) {
			EmitInt(int64_t(v));
			return;
		}
		slots.push_back(MakeExtSlot(ext_subtype::UINT64));
		nums.push_back(v);
	}

	void EmitDouble(double v) {
		// JSON has no representation for NaN/Infinity; reject at the single point where
		// a double enters the tape rather than emit invalid JSON later.
		if (!std::isfinite(v)) {
			throw InvalidInputException("jsono: cannot store non-finite double value (NaN/Infinity)");
		}
		slots.push_back(MakeExtSlot(ext_subtype::DOUBLE));
		uint64_t bits;
		std::memcpy(&bits, &v, sizeof(bits));
		nums.push_back(bits);
	}

	void EmitDec60(bool negative, uint64_t abs_mantissa, uint64_t scale) {
		slots.push_back(MakeSlot(tag::VAL_DEC60, 0));
		nums.push_back(MakeDec60Payload(negative, abs_mantissa, scale));
	}

	// VAL_EXT/NUMBER: raw byte-exact number text stored in string_heap and
	// consumed in walk order, like a string value.
	void EmitNumberText(nonstd::string_view text) {
		PushLength(text.size());
		string_heap.insert(string_heap.end(), text.begin(), text.end());
		slots.push_back(MakeExtSlot(ext_subtype::NUMBER));
	}

	void EmitBool(bool b) {
		slots.push_back(MakeSlot(b ? tag::VAL_TRUE : tag::VAL_FALSE, 0));
	}

	void EmitNull() {
		slots.push_back(MakeSlot(tag::VAL_NULL, 0));
	}

	// Whether a container starts with a ContainerSpan placeholder: arrays first reserve
	// one because their final size is only known at close; empty arrays drop it again in
	// FinishContainer. Large objects keep a span for subtree skip and cursor checkpoints.
	// Small objects store no span, even with container children: their key count is bounded,
	// so skipping them by walking their immediate children is cheap.
	static bool ContainerStoresSpan(uint64_t container_tag, uint32_t child_count) {
		return container_tag == tag::ARR_START ||
		       (container_tag == tag::OBJ_START && child_count > OBJECT_CHECKPOINT_STRIDE);
	}

	void PushContainerSpanPlaceholder(uint64_t id, size_t &span_index) {
		span_index = skips.size();
		span_ids.push_back(uint32_t(id));
		skips.push_back(ContainerSpan {0, 0, 0, 0, 0});
	}

	void EmitContainerStart(uint64_t container_tag, uint32_t child_count) {
		// open_containers.size() is the current nesting depth; reject before opening
		// another level so deep input cannot overflow the recursive tape walkers.
		if (open_containers.size() >= JSONO_MAX_NESTING_DEPTH) {
			throw InvalidInputException("jsono: nesting depth exceeds maximum of %llu",
			                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
		}
		// The sparse span index stores u32 ids (and CONTAINER_ID_MAX bounds the payload).
		if (container_count > std::numeric_limits<uint32_t>::max() || container_count > CONTAINER_ID_MAX) {
			throw InvalidInputException("jsono: too many containers for storage");
		}
		auto id = container_count++;
		auto span_index = NO_SPAN;
		if (ContainerStoresSpan(container_tag, child_count)) {
			PushContainerSpanPlaceholder(id, span_index);
		}
		auto checkpoint_offset = NO_OBJECT_CHECKPOINTS;
		auto checkpoint_stride = uint16_t(0);
		if (container_tag == tag::OBJ_START && child_count > OBJECT_CHECKPOINT_STRIDE) {
			checkpoint_stride = child_count >= LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT ? LARGE_OBJECT_CHECKPOINT_STRIDE
			                                                                           : OBJECT_CHECKPOINT_STRIDE;
			auto checkpoint_count = (child_count - 1) / checkpoint_stride;
			if (object_checkpoints.size() > std::numeric_limits<uint32_t>::max() - checkpoint_count) {
				throw InvalidInputException("jsono: too many object cursor checkpoints for storage");
			}
			checkpoint_offset = uint32_t(object_checkpoints.size());
			object_checkpoint_index.push_back(
			    ObjectCheckpointIndex {uint32_t(id), checkpoint_offset, checkpoint_stride, 0});
			object_checkpoints.resize(object_checkpoints.size() + checkpoint_count);
		}
		open_containers.push_back(OpenContainer {id, container_tag, child_count, checkpoint_offset, checkpoint_stride,
		                                         0, span_index, slots.size(), string_heap.size(), lengths.size(),
		                                         nums.size()});
		slots.push_back(MakeSlot(container_tag, MakeContainerPayload(id, child_count)));
	}

	void EmitObjectChildStart() {
		if (open_containers.empty() || open_containers.back().tag != tag::OBJ_START) {
			throw InternalException("jsono writer: object child start without object container");
		}
		auto &open = open_containers.back();
		if (open.next_child_index >= open.child_count) {
			throw InternalException("jsono writer: too many object children");
		}
		if (open.checkpoint_offset != NO_OBJECT_CHECKPOINTS && open.next_child_index > 0 &&
		    open.next_child_index % open.checkpoint_stride == 0) {
			auto value_start = open.start_slot + 1 + size_t(open.child_count);
			auto slot_delta = slots.size() - value_start;
			auto string_delta = string_heap.size() - open.start_string;
			auto length_delta = lengths.size() - open.start_length;
			auto num_delta = nums.size() - open.start_num;
			if (slot_delta > std::numeric_limits<uint32_t>::max() ||
			    string_delta > std::numeric_limits<uint32_t>::max() ||
			    length_delta > std::numeric_limits<uint32_t>::max() ||
			    num_delta > std::numeric_limits<uint32_t>::max()) {
				throw InvalidInputException("jsono: object cursor checkpoint exceeds storage limits");
			}
			object_checkpoints[open.checkpoint_offset + open.next_child_index / open.checkpoint_stride - 1] =
			    ObjectCursorCheckpoint {uint32_t(slot_delta), uint32_t(string_delta), uint32_t(length_delta),
			                            uint32_t(num_delta)};
		}
		open.next_child_index++;
	}

	void FinishContainer() {
		if (open_containers.empty()) {
			throw InternalException("jsono writer: container end without start");
		}
		auto open = open_containers.back();
		open_containers.pop_back();
		if (open.tag == tag::OBJ_START && open.next_child_index != open.child_count) {
			throw InternalException("jsono writer: object child count mismatch");
		}
		if (open.span_index == NO_SPAN) {
			return;
		}
		auto slot_span = slots.size() - open.start_slot;
		auto string_byte_span = string_heap.size() - open.start_string;
		auto length_count = lengths.size() - open.start_length;
		auto num_count = nums.size() - open.start_num;
		if (open.tag == tag::ARR_START && slot_span == 2) {
			if (open.span_index + 1 != skips.size() || span_ids[open.span_index] != open.id) {
				throw InternalException("jsono writer: empty array span is not the last sparse span");
			}
			span_ids.pop_back();
			skips.pop_back();
			return;
		}
		if (slot_span > std::numeric_limits<uint32_t>::max() ||
		    string_byte_span > std::numeric_limits<uint32_t>::max() ||
		    length_count > std::numeric_limits<uint32_t>::max() || num_count > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: container span exceeds storage limits");
		}
		// Spanned objects carry a shape_hash over their stored (sorted) KEY slots; arrays
		// leave it 0. The struct constructor's external_root_keys fast path writes KEY slots
		// whose offsets point into a shared external key_heap (not this builder's), so guard
		// the fingerprint on the last key being in-bounds; that owner patches skips afterward.
		uint64_t shape_hash = 0;
		if (open.tag == tag::OBJ_START && open.child_count > 0) {
			auto last_key_payload = SlotPayload(slots[open.start_slot + open.child_count]);
			if (KeyOffset(last_key_payload) + KeyLen(last_key_payload) <= key_heap.size()) {
				shape_hash = HashObjectKeySlots(slots.data(), open.start_slot + 1, open.child_count, key_heap.data());
			}
		} else if (open.tag == tag::OBJ_START) {
			shape_hash = HASH_SEED; // empty object: stable constant, matches HashObjectKeySlots(0 keys)
		}
		skips[open.span_index] = ContainerSpan {uint32_t(slot_span), uint32_t(string_byte_span), shape_hash,
		                                        uint32_t(length_count), uint32_t(num_count)};
	}

	void EmitObjectStart(uint64_t N) {
		if (N > CONTAINER_CHILD_COUNT_MASK) {
			throw InvalidInputException("jsono: object has too many keys for storage");
		}
		EmitContainerStart(tag::OBJ_START, uint32_t(N));
	}

	void EmitObjectEnd() {
		slots.push_back(MakeSlot(tag::OBJ_END, 0));
		FinishContainer();
	}

	void EmitArrayStart() {
		EmitContainerStart(tag::ARR_START, 0);
	}

	void EmitArrayEnd() {
		slots.push_back(MakeSlot(tag::ARR_END, 0));
		FinishContainer();
	}
};

inline string_t WriteBlobInto(Vector &vec, const char *data, size_t size) {
	auto s = StringVector::EmptyString(vec, size);
	if (size > 0) {
		std::memcpy(s.GetDataWriteable(), data, size);
	}
	s.Finalize();
	return s;
}

inline string_t WriteJsonoBlobInto(Vector &vec, const JsonoBuilder &builder) {
	auto slots_bytes = builder.slots.size() * sizeof(uint64_t);
	auto slots_total = JSONO_HEADER_SIZE + slots_bytes;
	auto slots_str = StringVector::EmptyString(vec, slots_total);
	auto slots_dst = slots_str.GetDataWriteable();

	JsonoHeader header;
	header.magic = MAGIC;
	header.version = VERSION;
	header.flags = flags::SORTED_KEYS;
	header.reserved = 0;
	std::memcpy(slots_dst, &header, JSONO_HEADER_SIZE);
	if (slots_bytes > 0) {
		std::memcpy(slots_dst + JSONO_HEADER_SIZE, builder.slots.data(), slots_bytes);
	}
	slots_str.Finalize();
	return slots_str;
}

// `manifest` is the pre-serialized shred-manifest tail (see ShredManifestEntry in jsono.hpp),
// appended after the checkpoint sections; only the shred writer passes one.
inline string_t WriteSkipsBlobInto(Vector &vec, const JsonoBuilder &builder, const std::string *manifest = nullptr) {
	auto span_ids_bytes = builder.span_ids.size() * sizeof(uint32_t);
	auto spans_bytes = builder.skips.size() * sizeof(ContainerSpan);
	auto index_bytes = builder.object_checkpoint_index.size() * sizeof(ObjectCheckpointIndex);
	auto checkpoints_bytes = builder.object_checkpoints.size() * sizeof(ObjectCursorCheckpoint);
	auto manifest_bytes = manifest ? manifest->size() : 0;
	auto total = sizeof(ContainerMetadataHeader) + span_ids_bytes + spans_bytes + index_bytes + checkpoints_bytes +
	             manifest_bytes;
	auto skips_str = StringVector::EmptyString(vec, total);
	auto skips_dst = skips_str.GetDataWriteable();

	if (builder.skips.size() > std::numeric_limits<uint32_t>::max() ||
	    builder.object_checkpoint_index.size() > std::numeric_limits<uint32_t>::max() ||
	    builder.object_checkpoints.size() > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: skips blob exceeds storage limits");
	}
	ContainerMetadataHeader header;
	header.span_count = uint32_t(builder.skips.size());
	header.checkpoint_index_count = uint32_t(builder.object_checkpoint_index.size());
	header.checkpoint_count = uint32_t(builder.object_checkpoints.size());
	std::memcpy(skips_dst, &header, sizeof(header));
	auto offset = sizeof(header);
	if (span_ids_bytes > 0) {
		std::memcpy(skips_dst + offset, builder.span_ids.data(), span_ids_bytes);
	}
	offset += span_ids_bytes;
	if (spans_bytes > 0) {
		std::memcpy(skips_dst + offset, builder.skips.data(), spans_bytes);
	}
	offset += spans_bytes;
	if (index_bytes > 0) {
		std::memcpy(skips_dst + offset, builder.object_checkpoint_index.data(), index_bytes);
	}
	offset += index_bytes;
	if (checkpoints_bytes > 0) {
		std::memcpy(skips_dst + offset, builder.object_checkpoints.data(), checkpoints_bytes);
	}
	offset += checkpoints_bytes;
	if (manifest_bytes > 0) {
		std::memcpy(skips_dst + offset, manifest->data(), manifest_bytes);
	}
	skips_str.Finalize();
	return skips_str;
}

inline string_t WriteLengthsBlobInto(Vector &vec, const JsonoBuilder &builder) {
	return WriteBlobInto(vec, reinterpret_cast<const char *>(builder.lengths.data()),
	                     builder.lengths.size() * sizeof(uint32_t));
}

inline string_t WriteNumsBlobInto(Vector &vec, const JsonoBuilder &builder) {
	return WriteBlobInto(vec, reinterpret_cast<const char *>(builder.nums.data()),
	                     builder.nums.size() * sizeof(uint64_t));
}

// The body struct vector of a jsono result vector (plain or shredded): top-level layout field [0]
// -> body [0]. The six blob vectors are its children.
inline Vector &JsonoBodyVector(Vector &result) {
	auto &layout = *StructVector::GetEntries(result)[0];
	return *StructVector::GetEntries(layout)[0];
}

// Shred `shred_index` (0-based over the shred set) of a shredded jsono vector: a canonically built
// layout is `body`, the value-complete marker, then the shreds, so shred k is layout child 2 + k.
inline Vector &JsonoShredVector(Vector &result, idx_t shred_index) {
	auto &layout = *StructVector::GetEntries(result)[0];
	return *StructVector::GetEntries(layout)[2 + shred_index];
}

// The value-complete marker vector of a canonically built shredded jsono result: it sits right after
// `body` at layout child 1 (before the shreds). The caller must know the layout has it
// (JsonoLayoutType::has_value_complete); writers always build the canonical layout.
inline Vector &JsonoVcVector(Vector &result) {
	auto &layout = *StructVector::GetEntries(result)[0];
	return *StructVector::GetEntries(layout)[1];
}

// Mark the whole result value-complete (marker = the layout hash of the result's shred set on every
// row): for a producer that by construction never diverts a present scalar into the residual — e.g.
// type-driven auto-shred, where each shred's type IS its source field's type, so a present value
// always fits and a NULL is only an absent field, never a residual diversion. The marker is part of
// the struct, so this also keeps such a value structurally equal (under `=`) to one the text/struct
// path built with the same data and the same shred set.
inline void JsonoFillValueComplete(Vector &result, idx_t count) {
	auto &layout_type = StructType::GetChildTypes(result.GetType())[0].second;
	auto &fields = StructType::GetChildTypes(layout_type);
	if (fields.size() < 2 || fields[1].first != JsonoValueCompleteName()) {
		return;
	}
	auto layout_hash = JsonoLayoutHashOf(result.GetType());
	auto &vc = JsonoVcVector(result);
	vc.SetVectorType(VectorType::FLAT_VECTOR);
	auto data = FlatVector::GetData<uint64_t>(vc);
	for (idx_t row = 0; row < count; row++) {
		data[row] = layout_hash;
	}
}

// Write-side handle over a jsono result vector's six body blob children, flattening the nested
// struct chain (result -> layout -> body -> blobs) once so per-row writes are flat data stores.
// Mirrors InitJsonoVectorData's navigation on the write side; the rest of the writer is
// layout-agnostic. The validity masks of all nine levels are cached at Init so a NULL row is
// nine direct mask stores instead of FlatVector::SetNull's per-row recursion through
// StructVector::GetEntries (which dominated extract scenarios with many absent-path rows).
struct JsonoBodyWriter {
	static constexpr idx_t NULL_MASK_COUNT = BODY_BLOB_COUNT + 3;

	Vector *result = nullptr;
	Vector *vec[BODY_BLOB_COUNT] = {nullptr};
	string_t *data[BODY_BLOB_COUNT] = {nullptr};
	ValidityMask *null_masks[NULL_MASK_COUNT] = {nullptr};

	void Init(Vector &result_p) {
		result = &result_p;
		result_p.SetVectorType(VectorType::FLAT_VECTOR);
		auto &layout = *StructVector::GetEntries(result_p)[0];
		auto &body = *StructVector::GetEntries(layout)[0];
		null_masks[0] = &FlatVector::Validity(result_p);
		null_masks[1] = &FlatVector::Validity(layout);
		null_masks[2] = &FlatVector::Validity(body);
		auto &blobs = StructVector::GetEntries(body);
		for (idx_t i = 0; i < BODY_BLOB_COUNT; i++) {
			blobs[i]->SetVectorType(VectorType::FLAT_VECTOR);
			vec[i] = blobs[i].get();
			data[i] = FlatVector::GetData<string_t>(*blobs[i]);
			null_masks[3 + i] = &FlatVector::Validity(*blobs[i]);
		}
	}

	Vector &Slots() {
		return *vec[BODY_SLOTS];
	}
	Vector &KeyHeap() {
		return *vec[BODY_KEY_HEAP];
	}
	Vector &StringHeap() {
		return *vec[BODY_STRING_HEAP];
	}
	Vector &Skips() {
		return *vec[BODY_SKIPS];
	}
	Vector &Lengths() {
		return *vec[BODY_LENGTHS];
	}
	Vector &Nums() {
		return *vec[BODY_NUMS];
	}

	void WriteRow(idx_t row, const JsonoBuilder &builder, const std::string *manifest = nullptr) {
		data[BODY_SLOTS][row] = WriteJsonoBlobInto(Slots(), builder);
		data[BODY_KEY_HEAP][row] = WriteBlobInto(KeyHeap(), builder.key_heap.data(), builder.key_heap.size());
		data[BODY_STRING_HEAP][row] =
		    WriteBlobInto(StringHeap(), builder.string_heap.data(), builder.string_heap.size());
		data[BODY_SKIPS][row] = WriteSkipsBlobInto(Skips(), builder, manifest);
		data[BODY_LENGTHS][row] = WriteLengthsBlobInto(Lengths(), builder);
		data[BODY_NUMS][row] = WriteNumsBlobInto(Nums(), builder);
	}

	// Null a jsono row: all nine levels (result, layout, body, six blobs), matching what
	// FlatVector::SetNull's struct recursion would set — the top-level NULL is what makes the
	// row read back as SQL NULL, and the children must be null too so downstream consumers
	// never read undefined string_t data. A shredded result's shred columns are nulled by the
	// caller (they are written per-row anyway).
	void SetRowNull(idx_t row) {
		for (idx_t i = 0; i < NULL_MASK_COUNT; i++) {
			null_masks[i]->SetInvalid(row);
		}
	}
};

// LIST result-vector helpers shared by the producers that emit LIST columns (jsono_entries and the
// array shred writer): grow the child capacity, finalize a row's list entry, and null a list row.
inline void EnsureListCapacity(Vector &result, idx_t needed) {
	if (needed <= ListVector::GetListCapacity(result)) {
		return;
	}
	auto next = std::max<idx_t>(needed, std::max<idx_t>(ListVector::GetListCapacity(result) * 2, 1));
	ListVector::Reserve(result, next);
}

inline void FinishListRow(Vector &result, idx_t row, idx_t start_offset, idx_t length) {
	ListVector::GetData(result)[row] = {start_offset, length};
	ListVector::SetListSize(result, start_offset + length);
}

inline void SetListRowNull(Vector &result, idx_t row) {
	FlatVector::SetNull(result, row, true);
	ListVector::GetData(result)[row] = {0, 0};
}

} // namespace jsono
} // namespace duckdb
