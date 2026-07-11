#pragma once

#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <unordered_set>

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

	// Sync the reservation to the state's current footprint (rewrite-style accumulators re-measure their
	// footprint rather than tracking append deltas). Growth reserves the delta and may throw the engine's
	// OOM; shrink frees the delta. Absolute-footprint syncing makes mis-pairing structurally impossible:
	// `reserved` always tracks the last measured footprint, and Destroy frees exactly that.
	void Resize(BufferManager &bm, idx_t new_total) {
		buffer_manager = &bm;
		if (new_total > reserved) {
			bm.ReserveMemory(new_total - reserved);
			reserved = new_total;
		} else if (new_total < reserved) {
			bm.FreeReservedMemory(reserved - new_total);
			reserved = new_total;
		}
	}
};

// Resize the reservation of every DISTINCT state touched in a grouped chunk to its current footprint,
// measuring each state once even when many rows in the chunk share it. Used by the rewrite-style
// accumulators whose footprint walk (an LWW tree, a candidate-path map) is too costly to repeat per row.
template <class STATE, class StateFor, class Footprint>
void AccountDistinctStates(idx_t count, BufferManager &bm, StateFor state_for, Footprint footprint) {
	std::unordered_set<STATE *> seen;
	for (idx_t row = 0; row < count; row++) {
		STATE *state = &state_for(row);
		if (seen.insert(state).second) {
			state->mem.Resize(bm, footprint(*state));
		}
	}
}

// Heap bytes an owned jsono document occupies (the six owned byte streams). This is the dominant term of a
// collected/accumulated element's footprint; callers add per-element container overhead on top.
inline idx_t BlobBytes(const jsono::OwnedJsonoBlob &blob) {
	return blob.slots.size() + blob.key_heap.size() + blob.string_heap.size() + blob.skips.size() +
	       blob.lengths.size() + blob.nums.size();
}

} // namespace duckdb
