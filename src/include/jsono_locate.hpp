#pragma once

#include "jsono.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

#include "string_view.hpp"

#include <cstdlib>

namespace duckdb {
namespace jsono {

// Shared path-locate engine: key->rank lookup over an object's sorted key block, the
// per-expression rank cache that amortizes it across rows, and the Key/Index step walk
// every path-addressed reader uses (extract, optimizer match/project, transform residual
// fallback, shred writer, jsono_type/jsono_keys). One implementation so the lookup policy
// cannot drift between call sites.

inline nonstd::string_view ObjectKeyAtRank(const JsonoView &view, const ObjectLayout &layout, size_t rank) {
	if (rank >= layout.key_count) {
		throw InternalException("JSONO object key rank out of bounds");
	}
	auto key_slot = view.SlotAt(layout.key_start + rank);
	if (SlotTag(key_slot) != tag::KEY) {
		throw InvalidInputException("malformed JSONO: expected KEY slot");
	}
	return view.KeyAt(SlotPayload(key_slot));
}

// Binary search for `key` in the object's sorted key block. `rank` is always set to the
// key's insertion point, so a miss still positions cursors for the caller.
inline bool FindObjectKeyRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t &rank) {
	size_t lo = 0;
	size_t hi = layout.key_count;
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto key_slot = view.SlotAt(layout.key_start + mid);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		if (view.KeyAt(SlotPayload(key_slot)) < key) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	rank = lo;
	if (lo >= layout.key_count) {
		return false;
	}
	return ObjectKeyAtRank(view, layout, lo) == key;
}

// Re-check a cached (rank, found) against this row's key block: the cached rank is valid
// when the keys around it still bracket `key` the same way the original lookup did.
inline bool ValidateCachedObjectRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t rank,
                                     bool found) {
	if (rank > layout.key_count) {
		return false;
	}
	if (found) {
		return rank < layout.key_count && ObjectKeyAtRank(view, layout, rank) == key;
	}
	if (rank > 0 && ObjectKeyAtRank(view, layout, rank - 1) >= key) {
		return false;
	}
	if (rank < layout.key_count && ObjectKeyAtRank(view, layout, rank) <= key) {
		return false;
	}
	return true;
}

constexpr idx_t RANK_CACHE_SIZE = 4096;

inline idx_t RankCacheIndex(idx_t step_index, size_t key_count) {
	return idx_t(((uint64_t(step_index) * 0x9E3779B97F4A7C15ULL) ^ key_count) & (RANK_CACHE_SIZE - 1));
}

// Default fast path trusts the stored shape_hash; JSONO_RANK_VALIDATE=1 forces the
// key-heap validation instead, isolating the format's storage/read overhead from the
// fast-path win in a single build (read once, not per row).
inline bool TrustShapeHash() {
	static const bool trust = []() {
		const char *env = std::getenv("JSONO_RANK_VALIDATE");
		return !(env && env[0] == '1');
	}();
	return trust;
}

struct RankCacheEntry {
	bool valid = false;
	idx_t step_index = DConstants::INVALID_INDEX;
	size_t key_count = 0;
	uint64_t shape_hash = 0;
	bool has_shape_hash = false;
	string key;
	size_t rank = 0;
	bool found = false;
};

// Direct-mapped cache of key->rank lookups keyed by (step id, object key_count). A stored
// shape_hash uniquely identifies the object's sorted key sequence, so a matching hash
// proves the cached rank by an int compare; layouts without one (small objects read
// without a span) fall back to re-validating the cached rank against the key slots.
struct RankCache {
	RankCache() : entries(RANK_CACHE_SIZE) {
	}

	vector<RankCacheEntry> entries;

	bool Find(idx_t step_index, const JsonoView &view, const ObjectLayout &layout, const string &key, size_t &rank) {
		auto &entry = entries[RankCacheIndex(step_index, layout.key_count)];
		auto hit = entry.valid && entry.step_index == step_index && entry.key_count == layout.key_count &&
		           entry.key == key &&
		           (TrustShapeHash() && layout.has_span && entry.has_shape_hash
		                ? entry.shape_hash == layout.shape_hash
		                : ValidateCachedObjectRank(view, layout, key, entry.rank, entry.found));
		if (hit) {
			rank = entry.rank;
			return entry.found;
		}
		bool found = FindObjectKeyRank(view, layout, key, rank);
		entry.valid = true;
		entry.step_index = step_index;
		entry.key_count = layout.key_count;
		entry.shape_hash = layout.shape_hash;
		entry.has_shape_hash = layout.has_span;
		entry.key = key;
		entry.rank = rank;
		entry.found = found;
		return found;
	}
};

// Move `cursor` (at an OBJ_START whose `layout` was read off it) into the value at `rank`,
// jumping through the object's value-block checkpoints when it has them.
inline void MoveCursorToObjectValueRank(const JsonoView &view, const ObjectLayout &layout, size_t rank,
                                        JsonoCursor &cursor) {
	size_t value_rank = 0;
	JsonoCursor value_block_base = cursor;
	value_block_base.pos = layout.value_start;
	JsonoCursor value_cursor = value_block_base;
	MoveObjectValueCursorToRank(view, layout, value_block_base, rank, value_rank, value_cursor);
	cursor = value_cursor;
}

// Position `cursor` (at an ARR_START) on element `index`; false when out of range.
inline bool LocateArrayIndex(const JsonoView &view, idx_t index, JsonoCursor &cursor) {
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	cursor.pos++;
	idx_t current = 0;
	while (cursor.pos < end_pos && current < index) {
		SkipValueFast(view, cursor);
		current++;
	}
	return cursor.pos < end_pos && current == index;
}

// One Key/Index step from the value at `cursor`. `cache` may be null: point readers that
// resolve the same path per row pass their rank cache (and get the span probe on small
// objects, so spanned ones validate by shape_hash); one-shot walkers pass null and pay the
// plain binary search. A miss (absent key, out-of-range index, scalar, wildcard) is false.
inline bool LocatePathStep(RankCache *cache, idx_t step_index, const JsonoView &view, const PathStep &step,
                           JsonoCursor &cursor) {
	if (cursor.pos >= view.Slots()) {
		return false;
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	switch (step.kind) {
	case PathStepKind::Key: {
		if (slot_tag != tag::OBJ_START) {
			return false;
		}
		auto layout = ReadObjectLayout(view, cursor.pos, cache != nullptr);
		size_t rank = 0;
		bool found = cache ? cache->Find(step_index, view, layout, step.key, rank)
		                   : FindObjectKeyRank(view, layout, step.key, rank);
		if (!found) {
			return false;
		}
		MoveCursorToObjectValueRank(view, layout, rank, cursor);
		return true;
	}
	case PathStepKind::Index: {
		if (slot_tag != tag::ARR_START) {
			return false;
		}
		return LocateArrayIndex(view, step.index, cursor);
	}
	case PathStepKind::Wildcard:
		return false;
	}
	return false;
}

inline bool LocatePathSteps(RankCache *cache, idx_t step_base, const vector<PathStep> &steps, const JsonoView &view,
                            JsonoCursor &cursor) {
	for (idx_t step_index = 0; step_index < steps.size(); step_index++) {
		if (!LocatePathStep(cache, step_base + step_index, view, steps[step_index], cursor)) {
			return false;
		}
	}
	return true;
}

} // namespace jsono
} // namespace duckdb
