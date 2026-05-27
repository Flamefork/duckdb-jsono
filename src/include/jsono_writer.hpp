#pragma once

#include "jsono.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/types/vector.hpp"

#include "string_view.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace duckdb {
namespace jsono {

struct JsonoBuilder {
	std::vector<uint64_t> slots;
	std::vector<char> key_heap;
	std::vector<char> string_heap;
	std::vector<ContainerSpan> skips;
	std::vector<ObjectCheckpointIndex> object_checkpoint_index;
	std::vector<ObjectCursorCheckpoint> object_checkpoints;

	struct OpenContainer {
		uint64_t id;
		uint64_t tag;
		uint32_t child_count;
		uint32_t checkpoint_offset;
		size_t checkpoint_index_pos;
		uint16_t checkpoint_stride;
		uint32_t next_child_index;
		size_t start_slot;
		size_t start_string;
	};

	std::vector<OpenContainer> open_containers;

	void Reset() {
		slots.clear();
		key_heap.clear();
		string_heap.clear();
		skips.clear();
		object_checkpoint_index.clear();
		object_checkpoints.clear();
		open_containers.clear();
	}

	void Reserve(size_t slot_count, size_t key_bytes, size_t value_bytes) {
		if (slots.capacity() < slot_count) {
			slots.reserve(slot_count);
		}
		if (key_heap.capacity() < key_bytes) {
			key_heap.reserve(key_bytes);
		}
		if (string_heap.capacity() < value_bytes) {
			string_heap.reserve(value_bytes);
		}
	}

	// Push a KEY slot referencing key bytes already present at [offset, offset+len).
	// KEY payload packs a 36-bit heap offset and 24-bit length; reject rather than
	// silently truncate (string values get the full 60-bit length, but keys do not).
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

	void EmitString(nonstd::string_view s) {
		string_heap.insert(string_heap.end(), s.begin(), s.end());
		slots.push_back(MakeSlot(tag::VAL_STR_HEAP, MakeStringPayload(uint64_t(s.size()))));
	}

	void EmitInt(int64_t v) {
		if (FitsInt60(v)) {
			slots.push_back(MakeSlot(tag::VAL_INT60, EncodeInt60(v)));
		} else {
			slots.push_back(MakeExtSlot(ext_subtype::INT64));
			slots.push_back(uint64_t(v));
		}
	}

	void EmitUInt(uint64_t v) {
		if (v <= uint64_t(INT60_MAX)) {
			slots.push_back(MakeSlot(tag::VAL_INT60, EncodeInt60(int64_t(v))));
		} else {
			slots.push_back(MakeExtSlot(ext_subtype::UINT64));
			slots.push_back(v);
		}
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
		slots.push_back(bits);
	}

	void EmitDec60(bool negative, uint64_t abs_mantissa, uint64_t scale) {
		slots.push_back(MakeSlot(tag::VAL_DEC60, MakeDec60Payload(negative, abs_mantissa, scale)));
	}

	// VAL_EXT/NUMBER: raw byte-exact number text stored in string_heap and
	// consumed in walk order, like a string value.
	void EmitNumberText(nonstd::string_view text) {
		string_heap.insert(string_heap.end(), text.begin(), text.end());
		slots.push_back(MakeNumberExtSlot(uint64_t(text.size())));
	}

	void EmitBool(bool b) {
		slots.push_back(MakeSlot(b ? tag::VAL_TRUE : tag::VAL_FALSE, 0));
	}

	void EmitNull() {
		slots.push_back(MakeSlot(tag::VAL_NULL, 0));
	}

	void EmitContainerStart(uint64_t container_tag, uint32_t child_count) {
		// open_containers.size() is the current nesting depth; reject before opening
		// another level so deep input cannot overflow the recursive tape walkers.
		if (open_containers.size() >= JSONO_MAX_NESTING_DEPTH) {
			throw InvalidInputException("jsono: nesting depth exceeds maximum of %llu",
			                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
		}
		if (skips.size() > CONTAINER_ID_MAX) {
			throw InvalidInputException("jsono: too many containers for storage");
		}
		auto id = uint64_t(skips.size());
		auto checkpoint_offset = NO_OBJECT_CHECKPOINTS;
		auto checkpoint_index_pos = std::numeric_limits<size_t>::max();
		auto checkpoint_stride = uint16_t(0);
		if (container_tag == tag::OBJ_START && child_count > OBJECT_CHECKPOINT_STRIDE) {
			if (id > std::numeric_limits<uint32_t>::max()) {
				throw InvalidInputException("jsono: too many containers for object cursor index");
			}
			checkpoint_stride = child_count >= LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT ? LARGE_OBJECT_CHECKPOINT_STRIDE
			                                                                           : OBJECT_CHECKPOINT_STRIDE;
			auto checkpoint_count = (child_count - 1) / checkpoint_stride;
			if (object_checkpoints.size() > std::numeric_limits<uint32_t>::max() - checkpoint_count) {
				throw InvalidInputException("jsono: too many object cursor checkpoints for storage");
			}
			checkpoint_offset = uint32_t(object_checkpoints.size());
			checkpoint_index_pos = object_checkpoint_index.size();
			object_checkpoint_index.push_back(
			    ObjectCheckpointIndex {uint32_t(id), checkpoint_offset, checkpoint_stride, 0});
			object_checkpoints.resize(object_checkpoints.size() + checkpoint_count);
		}
		skips.push_back(ContainerSpan {0, 0});
		open_containers.push_back(OpenContainer {id, container_tag, child_count, checkpoint_offset,
		                                         checkpoint_index_pos, checkpoint_stride, 0, slots.size(),
		                                         string_heap.size()});
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
			if (slot_delta > std::numeric_limits<uint32_t>::max() ||
			    string_delta > std::numeric_limits<uint32_t>::max()) {
				throw InvalidInputException("jsono: object cursor checkpoint exceeds storage limits");
			}
			object_checkpoints[open.checkpoint_offset + open.next_child_index / open.checkpoint_stride - 1] =
			    ObjectCursorCheckpoint {uint32_t(slot_delta), uint32_t(string_delta)};
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
		auto slot_span = slots.size() - open.start_slot;
		auto string_byte_span = string_heap.size() - open.start_string;
		if (slot_span > std::numeric_limits<uint32_t>::max() ||
		    string_byte_span > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: container span exceeds storage limits");
		}
		skips[open.id] = ContainerSpan {uint32_t(slot_span), uint32_t(string_byte_span)};
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

inline string_t WriteSkipsBlobInto(Vector &vec, const JsonoBuilder &builder) {
	auto spans_bytes = builder.skips.size() * sizeof(ContainerSpan);
	auto index_bytes = builder.object_checkpoint_index.size() * sizeof(ObjectCheckpointIndex);
	auto checkpoints_bytes = builder.object_checkpoints.size() * sizeof(ObjectCursorCheckpoint);
	auto total = sizeof(ContainerMetadataHeader) + spans_bytes + index_bytes + checkpoints_bytes;
	auto skips_str = StringVector::EmptyString(vec, total);
	auto skips_dst = skips_str.GetDataWriteable();

	if (builder.skips.size() > std::numeric_limits<uint32_t>::max() ||
	    builder.object_checkpoint_index.size() > std::numeric_limits<uint32_t>::max() ||
	    builder.object_checkpoints.size() > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: skips blob exceeds storage limits");
	}
	ContainerMetadataHeader header;
	header.container_count = uint32_t(builder.skips.size());
	header.checkpoint_index_count = uint32_t(builder.object_checkpoint_index.size());
	header.checkpoint_count = uint32_t(builder.object_checkpoints.size());
	std::memcpy(skips_dst, &header, sizeof(header));
	if (spans_bytes > 0) {
		std::memcpy(skips_dst + sizeof(header), builder.skips.data(), spans_bytes);
	}
	if (index_bytes > 0) {
		std::memcpy(skips_dst + sizeof(header) + spans_bytes, builder.object_checkpoint_index.data(), index_bytes);
	}
	if (checkpoints_bytes > 0) {
		std::memcpy(skips_dst + sizeof(header) + spans_bytes + index_bytes, builder.object_checkpoints.data(),
		            checkpoints_bytes);
	}
	skips_str.Finalize();
	return skips_str;
}

inline void WriteJsonoStruct(Vector &slots_vec, Vector &key_heap_vec, Vector &string_heap_vec, Vector &skips_vec,
                             string_t *slots_data, string_t *key_heap_data, string_t *string_heap_data,
                             string_t *skips_data, idx_t row, const JsonoBuilder &builder) {
	slots_data[row] = WriteJsonoBlobInto(slots_vec, builder);
	key_heap_data[row] = WriteBlobInto(key_heap_vec, builder.key_heap.data(), builder.key_heap.size());
	string_heap_data[row] = WriteBlobInto(string_heap_vec, builder.string_heap.data(), builder.string_heap.size());
	skips_data[row] = WriteSkipsBlobInto(skips_vec, builder);
}

inline void SetJsonoStructNull(Vector &result, Vector &slots_vec, Vector &key_heap_vec, Vector &string_heap_vec,
                               Vector &skips_vec, idx_t row) {
	FlatVector::SetNull(result, row, true);
	FlatVector::SetNull(slots_vec, row, true);
	FlatVector::SetNull(key_heap_vec, row, true);
	FlatVector::SetNull(string_heap_vec, row, true);
	FlatVector::SetNull(skips_vec, row, true);
}

} // namespace jsono
} // namespace duckdb
