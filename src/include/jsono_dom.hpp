#pragma once

#include "jsono.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "yyjson.hpp"
#include "string_view.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace duckdb {
namespace jsono_dom {

using namespace jsono;
using namespace duckdb_yyjson;

// Shape-hash helpers (HashMix64/HashBytes/HashKey, HASH_SEED/HASH_PRIME) live in
// jsono.hpp and are visible via `using namespace jsono`. The DOM builder hashes each
// object's DOM-order key sequence into a 64-bit fingerprint and looks it up in the
// shape_cache; on hit the cached sort/dedup permutation is applied to THIS object's keys
// and std::sort is skipped.
//
// This is the ONE site in the extension that trusts a hash match authoritatively: the
// permutation is reused with no key re-check (every read-side cache instead re-confirms
// the key per hit, so a collision there only costs a recompute). A collision here writes a
// PERSISTED blob whose object keys are not sorted, and the readers' binary search
// (jsono_locate/jsono_trie) then returns wrong or NULL values SILENTLY — the rank-cache
// boundary check only self-corrects to the same unsorted binary search, it does not raise;
// only an explicit jsono_validate catches it (strict sort-order check + shape_hash
// recompute). perm[i] stays in [0,N) and the n_keys==N guard holds, so the corruption is
// purely logical, never a crash.
//
// Random 64-bit collisions are accepted (~5e-20 per object pair, full-hash compare). But
// the mixer is NOT keyed (fixed HASH_SEED/HASH_PRIME), so equal-key-count adversarial key
// sets can be CONSTRUCTED to collide; the lookup fingerprint is therefore salted with a
// per-builder-state random shape_salt (folded into the HASH_SEED start state for
// ShapeCacheFind/Insert only, see jsono_dom.cpp) so the target slot is unpredictable. The
// salt never reaches the persisted ContainerSpan.shape_hash (computed separately from the
// sorted KEY slots) — output bytes stay salt-independent, only cache keying changes.

struct ShapeCacheEntry {
	uint64_t hash = 0;
	uint32_t n_keys = 0;
	// perm[i] tells emit-pass-i which kvs[kv_offset + perm[i]] to use.
	// Reserved on JsonoBuilder construction so steady-state operation
	// doesn't churn the allocator across LRU evictions.
	std::vector<uint32_t> perm;
	// Fingerprint of the post-dedup sorted key sequence — what
	// HashObjectKeySlots computes from the emitted KEY slots. Cached by the
	// direct write path so spanned objects skip per-row rehashing on a hit;
	// the builder path leaves it 0 (its FinishContainer hashes the slots).
	uint64_t sorted_hash = 0;
};

// Direct-mapped shape cache. Slot = hash & (SIZE-1); collisions evict.
// O(1) lookup with one L1/L2 access — no linear scan, no LRU bookkeeping.
// On workloads with many distinct schemas, a direct-mapped table sized 2-8× the
// working schema set gives uniformly high hit rates without the scan-cost ceiling
// of a linear-probe LRU.
//
// SIZE must be a power of two. Default 8192 → 64KB per builder × per-thread.
// Override via JSONO_SHAPE_CACHE_SIZE env var (rounded up to power of 2).
inline size_t ShapeCacheSize() {
	static const size_t sz = []() -> size_t {
		size_t requested = 8192;
		const char *env = std::getenv("JSONO_SHAPE_CACHE_SIZE");
		if (env && *env) {
			long v = std::strtol(env, nullptr, 10);
			if (v > 0 && v <= (1 << 20)) {
				requested = static_cast<size_t>(v);
			}
		}
		// Round up to power of 2 (mask-friendly).
		size_t p = 1;
		while (p < requested) {
			p <<= 1;
		}
		return p;
	}();
	return sz;
}

// Per-local-state random salt for the shape-cache LOOKUP fingerprint ONLY (see the
// shape-hash trust comment at the top of this file). std::random_device gives one
// non-deterministic draw per state init; two draws fill the full 64 bits. This salt is
// never folded into the persisted ContainerSpan.shape_hash, so output bytes stay
// salt-independent — it only makes the target cache slot unpredictable to an adversary
// who controls incoming keys.
inline uint64_t MakeShapeSalt() {
	std::random_device rd;
	return (uint64_t(rd()) << 32) ^ uint64_t(rd());
}

// Slot/heap buffers carry across rows and across chunks (via FunctionLocalState)
// so the vectors' capacity stabilises after a warmup chunk and per-document
// emission becomes pure appends.
//
// Object emission is two-pass against the wire layout
// [OBJ_START][KEY1..KEYN][VAL1..VALN][OBJ_END]:
//   1. collect kv pairs into the shared `kvs` buffer (a stack-via-offset, so
//      recursive objects/arrays share one allocation), hashing key bytes
//      into a shape fingerprint as we go,
//   2. look the fingerprint up in `shape_cache` — on hit reuse the cached
//      sort permutation; on miss std::sort indices and cache the result,
//   3. emit OBJ_START + N KEY slots inline into `slots`,
//   4. recurse on each value, emitting directly into `slots`.
struct DomJsonoBuilder : public JsonoBuilder {
	// Shared per-object kv staging. Each EmitDomObject pushes its kvs to the
	// end, computes shape hash + sort permutation, then truncates back on
	// return — so the underlying allocation amortises across every object.
	using KVPair = std::pair<nonstd::string_view, yyjson_val *>;
	std::vector<KVPair> kvs;
	// Parallel indices buffer for sort permutation, same stack-via-offset.
	std::vector<uint32_t> indices;
	std::vector<size_t> static_key_offsets;
	std::vector<uint8_t> static_key_offset_valid;
	bool external_root_keys = false;

	// Direct-mapped shape cache — see ShapeCacheSize() comment above.
	// hash & shape_cache_mask → slot index.
	std::vector<ShapeCacheEntry> shape_cache;
	uint64_t shape_cache_mask = 0;
	// Salts the LOOKUP fingerprint only (never the persisted hash) — see MakeShapeSalt().
	uint64_t shape_salt = MakeShapeSalt();

	DomJsonoBuilder() {
		size_t sz = ShapeCacheSize();
		shape_cache.resize(sz);
		shape_cache_mask = uint64_t(sz - 1);
		for (auto &entry : shape_cache) {
			entry.perm.reserve(64); // typical N for analytics JSON
		}
	}

	void Reset() {
		JsonoBuilder::Reset();
		kvs.clear();
		indices.clear();
		std::fill(static_key_offset_valid.begin(), static_key_offset_valid.end(), 0);
		external_root_keys = false;
		// shape_cache deliberately NOT reset — that's the whole point.
	}

	// Direct-mapped lookup. One memory access; check hash + n_keys to guard
	// against collisions (different schemas mapping to same slot).
	const ShapeCacheEntry *ShapeCacheFind(uint64_t hash, uint32_t n_keys) const {
		const auto &slot = shape_cache[hash & shape_cache_mask];
		if (slot.hash == hash && slot.n_keys == n_keys) {
			return &slot;
		}
		return nullptr;
	}

	// Insert overwrites whatever entry was at that slot. perm storage is
	// reused (assign keeps reserved capacity), no allocator churn in
	// steady state.
	void ShapeCacheInsert(uint64_t hash, uint32_t n_keys, const uint32_t *perm_src, uint32_t perm_n) {
		auto &slot = shape_cache[hash & shape_cache_mask];
		slot.hash = hash;
		slot.n_keys = n_keys;
		slot.perm.assign(perm_src, perm_src + perm_n);
	}
};

// NUMBER_AS_RAW delivers every number as YYJSON_TYPE_RAW carrying the original
// source bytes. That raw text is the only way to make fractional numbers
// byte-exact (yyjson has no "fractional only" raw mode), so the cost of moving
// integer classification onto our own digit scanner is accepted for byte-exact
// (a hard requirement). Without ALLOW_INF_AND_NAN, inf/nan are still rejected
// at parse time, preserving RFC 8259 compliance.
constexpr yyjson_read_flag JSONO_READ_FLAGS = YYJSON_READ_NUMBER_AS_RAW;

// yyjson reads into a copy under non-insitu mode; the const_cast is safe.
inline yyjson_doc *ReadJsonoDoc(const string_t &input, yyjson_alc *alc, yyjson_read_err &err) {
	return yyjson_read_opts(const_cast<char *>(input.GetData()), input.GetSize(), JSONO_READ_FLAGS, alc, &err);
}

struct YyjsonAllocator {
	yyjson_alc *alc = yyjson_alc_dyn_new();

	~YyjsonAllocator() {
		if (alc) {
			yyjson_alc_dyn_free(alc);
		}
	}
};

bool EmitDomElement(yyjson_val *element, DomJsonoBuilder &b);

// Two-pass direct DOM writer (text parse path). Pass 1 walks the DOM once to
// compute the exact byte size of all six body streams (collecting per-object
// sort permutations and classifying numbers along the way); the row then
// allocates six exact-size result strings and pass 2 writes every stream
// in place — no intermediate builder vectors, no growth reallocation, and no
// final per-stream memcpy.
struct DomSizing {
	size_t slot_count = 0;
	size_t key_bytes = 0;
	size_t string_bytes = 0;
	size_t length_count = 0;
	size_t num_count = 0;
	size_t span_count = 0;
	size_t checkpoint_index_count = 0;
	size_t checkpoint_count = 0;
	uint64_t container_count = 0;
};

// Per-object emit plan recorded by pass 1 in DFS pre-order; pass 2 replays
// plans with a cursor. kv/idx offsets point into the row-retained kvs/indices
// buffers (append-only during a row, so absolute offsets stay valid).
struct DomObjectPlan {
	uint64_t sorted_hash;
	size_t kv_offset;
	size_t idx_offset;
	uint32_t emit_count;
	bool spanned;
};

struct DomDirectState {
	using KVPair = DomJsonoBuilder::KVPair;
	// kvs/indices are retained for the whole row (unlike the builder's
	// per-object truncation): pass 2 reads keys and value handles through the
	// recorded plans instead of re-iterating yyjson objects.
	std::vector<KVPair> kvs;
	std::vector<uint32_t> indices;
	std::vector<DomObjectPlan> plans;
	// Numbers classified by pass 1, replayed by pass 2 in walk order: the slot
	// word, followed by the nums-stream word when the slot consumes one.
	// VAL_EXT/NUMBER raw text is not staged — pass 2 re-reads it from the DOM.
	std::vector<uint64_t> staged_numbers;
	DomSizing sz;

	// Direct-mapped shape cache, same policy as DomJsonoBuilder's — separate
	// instance because entries here also carry the sorted-key hash.
	std::vector<ShapeCacheEntry> shape_cache;
	uint64_t shape_cache_mask = 0;
	// Salts the LOOKUP fingerprint only (never the persisted hash) — see MakeShapeSalt().
	uint64_t shape_salt = MakeShapeSalt();

	DomDirectState() {
		size_t cache_size = ShapeCacheSize();
		shape_cache.resize(cache_size);
		shape_cache_mask = uint64_t(cache_size - 1);
		for (auto &entry : shape_cache) {
			entry.perm.reserve(64); // typical N for analytics JSON
		}
	}

	void ResetRow() {
		kvs.clear();
		indices.clear();
		plans.clear();
		staged_numbers.clear();
		sz = DomSizing();
	}

	const ShapeCacheEntry *ShapeCacheFind(uint64_t hash, uint32_t n_keys) const {
		const auto &slot = shape_cache[hash & shape_cache_mask];
		if (slot.hash == hash && slot.n_keys == n_keys) {
			return &slot;
		}
		return nullptr;
	}

	void ShapeCacheInsert(uint64_t hash, uint32_t n_keys, const uint32_t *perm_src, uint32_t perm_n,
	                      uint64_t sorted_hash) {
		auto &slot = shape_cache[hash & shape_cache_mask];
		slot.hash = hash;
		slot.n_keys = n_keys;
		slot.perm.assign(perm_src, perm_src + perm_n);
		slot.sorted_hash = sorted_hash;
	}
};

// One-pass shredded text write (jsono(text, shredding := ...)): pass 1 matches the spec's
// object-key paths against each object's deduped sorted keys while sizing. A lossless leaf
// (value kind matches the shred type exactly) is stripped from the emit plan — pass 2 then
// never visits it — and captured below; a present-but-lossy leaf stays in the residual with
// a NULL, incomplete lane.
struct DomShredCapture {
	enum class State : uint8_t { Missing, String, Int, Uint, Bool };
	State state = State::Missing;
	bool stripped = false;
	// A present value at a shred path that the shred did not capture — a mismatched scalar or a
	// container (kept in the residual, shred NULL): a bare struct_extract would read NULL while the
	// residual `->>` would yield the value, so this shred is NOT complete on this row.
	// Absent paths and present nulls stay false (both read NULL either way).
	bool diverted_scalar = false;
	nonstd::string_view text;
	int64_t int_value = 0;
	uint64_t uint_value = 0;
	bool bool_value = false;
};

// One node per distinct object-key path prefix; node 0 is the root. `field` is the shred
// terminating at this node (-1 if none — a node can both terminate a path and carry deeper
// edges, e.g. '$.a' next to '$.a.b').
struct DomShredTrieNode {
	int64_t field = -1;
	std::vector<std::pair<std::string, uint32_t>> children;
};

struct DomShredContext {
	const std::vector<DomShredTrieNode> *nodes = nullptr;
	std::vector<JsonoScalarPrimitive> kinds; // per shred field
	const vector<JsonoShredManifestEntryBytes> *manifest_entries = nullptr;
	// Per-row outputs: captures parallel to the shred fields, manifest = the row's serialized
	// shred-manifest tail (empty when nothing was stripped).
	std::vector<DomShredCapture> captures;
	vector<idx_t> stripped_fields;
	std::string manifest;
};

// Size (pass 1) and write (pass 2) one parsed DOM row straight into the
// writer's six exact-size blobs. Throws InvalidInputException on writer limits
// (nesting depth, key/string/container bounds) — all from pass 1, before any
// result allocation. With `shred` set, pass 1 also captures/strips shred leaves
// and the row's skips blob carries the shred manifest.
void EmitDomRowDirect(yyjson_val *root, DomDirectState &state, JsonoBodyWriter &writer, idx_t row,
                      DomShredContext *shred = nullptr);

} // namespace jsono_dom
} // namespace duckdb
