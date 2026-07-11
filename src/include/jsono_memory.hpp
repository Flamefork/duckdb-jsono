#pragma once

#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

// Accounted-memory tracker for one jsono aggregate state. jsono aggregates accumulate on the raw C++ heap;
// without this the engine's memory manager never sees that growth, so SET max_memory is not enforced and a
// big-group query OS-OOMs the whole process instead of raising a clean error. Each growing state embeds one
// of these and pairs every growth with a ReserveMemory delta against the buffer manager (which throws the
// engine's Out-Of-Memory error, tagged MemoryTag::EXTENSION, when the reservation would exceed max_memory)
// and frees exactly the tracked amount on Destroy.
//
// Invariant: `reserved` always equals the bytes currently reserved for this state. Reserve bumps the
// counter only after the (throwing) ReserveMemory returns, so a growth path that OOMs leaves the counter
// consistent and Destroy frees precisely what was reserved -- no leak on early throw, no double free.
//
// The buffer manager pointer is cached on the state at reserve time rather than fetched from bind_data in
// Destroy: DuckDB does not guarantee a live bind_data in state destructors (the ungrouped aggregate captures
// raw bind_info pointers that dangle by teardown), so Destroy must be self-sufficient. The buffer manager
// itself is database-lifetime, so the cached pointer always outlives the state.
struct JsonoMemoryReservation {
	idx_t reserved = 0;
	BufferManager *buffer_manager = nullptr;

	// Reserve `delta` more bytes before the state stores them. ReserveMemory first, then bump the counter:
	// if the reservation throws, `reserved` is unchanged.
	void Reserve(BufferManager &bm, idx_t delta) {
		if (delta == 0) {
			return;
		}
		bm.ReserveMemory(delta);
		buffer_manager = &bm;
		reserved += delta;
	}

	// Free everything this state reserved. Called once from Destroy; idempotent (a state that never grew has
	// reserved == 0 and no cached buffer manager, so this is a no-op).
	void Release() {
		if (reserved == 0) {
			return;
		}
		buffer_manager->FreeReservedMemory(reserved);
		reserved = 0;
	}

	// Move a source state's reservation into this one without touching the global counter: a destructive
	// Combine transfers ownership of the source's bytes to the target, so total reserved is unchanged.
	void AbsorbFrom(JsonoMemoryReservation &source) {
		if (source.reserved == 0) {
			return;
		}
		buffer_manager = source.buffer_manager;
		reserved += source.reserved;
		source.reserved = 0;
	}
};

// Heap bytes an owned jsono document occupies (the six owned byte streams). This is the dominant term of a
// collected/accumulated element's footprint; callers add per-element container overhead on top.
inline idx_t BlobBytes(const OwnedJsonoBlob &blob) {
	return blob.slots.size() + blob.key_heap.size() + blob.string_heap.size() + blob.skips.size() +
	       blob.lengths.size() + blob.nums.size();
}

} // namespace duckdb
