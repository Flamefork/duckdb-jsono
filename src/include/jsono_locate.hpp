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

// Order a trie's stored object key against a document key the same way the sorted key block is
// ordered (first byte then full compare). The sorted-merge walk both the transform and projector
// tries use relies on this ordering to advance edge and key cursors in lockstep.
inline int CompareTrieKeyToJsonKey(const string &trie_key, nonstd::string_view json_key) {
	if (!trie_key.empty() && !json_key.empty()) {
		auto trie_first = static_cast<unsigned char>(trie_key[0]);
		auto json_first = static_cast<unsigned char>(json_key[0]);
		if (trie_first != json_first) {
			return trie_first < json_first ? -1 : 1;
		}
	}
	return nonstd::string_view(trie_key).compare(json_key);
}

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
// key's insertion point, so a miss still positions cursors for the caller. The block is trusted
// sorted (the writer emits keys in order); a cache hit confirms the key at `rank` per hit rather
// than re-scanning the whole block.
inline bool FindObjectKeyRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t &rank) {
	size_t lo = 0;
	size_t hi = layout.key_count;
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		auto mid_key = ObjectKeyAtRank(view, layout, mid);
		if (mid_key < key) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	rank = lo;
	if (lo >= layout.key_count) {
		return false;
	}
	auto found_key = ObjectKeyAtRank(view, layout, lo);
	return found_key == key;
}

// Re-check a cached (rank, found) against this row's key block without re-scanning it: a found rank
// is confirmed by reading the key at `rank` (so a colliding cache slot is rejected), a miss by its
// bracketing neighbors. Trusts the writer-guaranteed sort order instead of validating it per hit.
inline bool ValidateCachedObjectRank(const JsonoView &view, const ObjectLayout &layout, const string &key, size_t rank,
                                     bool found) {
	if (rank > layout.key_count) {
		return false;
	}
	if (found) {
		if (rank >= layout.key_count) {
			return false;
		}
		auto found_key = ObjectKeyAtRank(view, layout, rank);
		return found_key == key;
	}
	if (rank > 0) {
		auto previous_key = ObjectKeyAtRank(view, layout, rank - 1);
		if (previous_key >= key) {
			return false;
		}
	}
	if (rank < layout.key_count) {
		auto next_key = ObjectKeyAtRank(view, layout, rank);
		if (next_key <= key) {
			return false;
		}
	}
	return true;
}

constexpr idx_t RANK_CACHE_SIZE = 4096;

inline idx_t RankCacheIndex(idx_t step_index, size_t key_count) {
	return idx_t(((uint64_t(step_index) * 0x9E3779B97F4A7C15ULL) ^ key_count) & (RANK_CACHE_SIZE - 1));
}

// Default cache matching uses the stored shape_hash as a cheap prefilter before the per-hit key
// confirmation; JSONO_RANK_VALIDATE=1 skips the prefilter so every slot match confirms via the key
// read (a control for measuring the prefilter).
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
// shape_hash rejects obvious shape mismatches quickly; every hit then confirms the key at the
// cached rank (no full-block scan).
struct RankCache {
	RankCache() : entries(RANK_CACHE_SIZE) {
	}

	vector<RankCacheEntry> entries;

	bool Find(idx_t step_index, const JsonoView &view, const ObjectLayout &layout, const string &key, size_t &rank) {
		auto &entry = entries[RankCacheIndex(step_index, layout.key_count)];
		auto hit =
		    entry.valid && entry.step_index == step_index && entry.key_count == layout.key_count && entry.key == key;
		if (hit && TrustShapeHash() && layout.has_span && entry.has_shape_hash) {
			hit = entry.shape_hash == layout.shape_hash;
		}
		if (hit) {
			hit = ValidateCachedObjectRank(view, layout, key, entry.rank, entry.found);
		}
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

struct JsonoLayoutCacheEntry {
	uint64_t generation = 0;
	size_t pos = 0;
	ObjectLayout layout {};
};

constexpr idx_t JSONO_LAYOUT_CACHE_SIZE = 16;

struct JsonoLayoutCache {
	JsonoLayoutCache() : entries(JSONO_LAYOUT_CACHE_SIZE) {
	}

	vector<JsonoLayoutCacheEntry> entries;
	uint64_t generation = 0;

	void NextRow() {
		generation++;
	}
};

struct JsonoPathLocateState {
	RankCache rank_cache;
	JsonoLayoutCache layout_cache;

	void NextRow() {
		layout_cache.NextRow();
	}
};

struct JsonoPathLocation {
	JsonoPathLocation() = default;
	JsonoPathLocation(const JsonoCursor &cursor_p, bool found_p) : cursor(cursor_p), found(found_p) {
	}

	JsonoCursor cursor;
	bool found = false;
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

// Element count of the array at `cursor` (an ARR_START). Arrays store no child_count in the
// format (only OBJ_START does), so this skip-walks the value block once, one SkipValueFast per
// element — the one place that walk is written.
inline int64_t CountArrayElements(const JsonoView &view, JsonoCursor cursor) {
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	cursor.pos++;
	int64_t count = 0;
	while (cursor.pos < end_pos) {
		count++;
		SkipValueFast(view, cursor);
	}
	return count;
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

inline bool LocateObjectKeyCached(JsonoPathLocateState &state, idx_t step_index, const JsonoView &view,
                                  const string &key, JsonoCursor &cursor) {
	if (cursor.pos >= view.Slots()) {
		return false;
	}
	auto &entry = state.layout_cache.entries[cursor.pos & (JSONO_LAYOUT_CACHE_SIZE - 1)];
	if (entry.generation != state.layout_cache.generation || entry.pos != cursor.pos) {
		if (SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_START) {
			return false;
		}
		entry.generation = state.layout_cache.generation;
		entry.pos = cursor.pos;
		entry.layout = ReadObjectLayout(view, cursor.pos, true);
	}
	size_t rank = 0;
	if (!state.rank_cache.Find(step_index, view, entry.layout, key, rank)) {
		return false;
	}
	MoveCursorToObjectValueRank(view, entry.layout, rank, cursor);
	return true;
}

// One Key/Index step from the value at `cursor`. `cache` may be null: point readers that
// resolve the same path per row pass their rank cache (and get the span probe on small
// objects, so spanned ones expose shape_hash to key-block validation); one-shot walkers pass null
// and pay the plain binary search. A miss (absent key, out-of-range index, scalar, wildcard) is false.
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

inline bool LocatePathStep(JsonoPathLocateState &state, idx_t step_index, const JsonoView &view, const PathStep &step,
                           JsonoCursor &cursor) {
	if (cursor.pos >= view.Slots()) {
		return false;
	}
	if (step.kind == PathStepKind::Key) {
		return LocateObjectKeyCached(state, step_index, view, step.key, cursor);
	}
	return LocatePathStep(&state.rank_cache, step_index, view, step, cursor);
}

inline bool LocatePathPrefix(RankCache *cache, idx_t step_base, const vector<PathStep> &steps, idx_t step_count,
                             const JsonoView &view, JsonoCursor &cursor) {
	for (idx_t step_index = 0; step_index < step_count; step_index++) {
		if (!LocatePathStep(cache, step_base + step_index, view, steps[step_index], cursor)) {
			return false;
		}
	}
	return true;
}

inline bool LocatePathPrefix(JsonoPathLocateState &state, idx_t step_base, const vector<PathStep> &steps,
                             idx_t step_count, const JsonoView &view, JsonoCursor &cursor) {
	for (idx_t step_index = 0; step_index < step_count; step_index++) {
		if (!LocatePathStep(state, step_base + step_index, view, steps[step_index], cursor)) {
			return false;
		}
	}
	return true;
}

inline bool LocatePathSteps(RankCache *cache, idx_t step_base, const vector<PathStep> &steps, const JsonoView &view,
                            JsonoCursor &cursor) {
	return LocatePathPrefix(cache, step_base, steps, steps.size(), view, cursor);
}

inline bool LocatePathSteps(JsonoPathLocateState &state, idx_t step_base, const vector<PathStep> &steps,
                            const JsonoView &view, JsonoCursor &cursor) {
	return LocatePathPrefix(state, step_base, steps, steps.size(), view, cursor);
}

inline JsonoPathLocation LocatePath(const JsonoView &view, const vector<PathStep> &steps) {
	JsonoCursor cursor;
	if (!LocatePathSteps(nullptr, 0, steps, view, cursor)) {
		return {};
	}
	return JsonoPathLocation {cursor, true};
}

} // namespace jsono
} // namespace duckdb
