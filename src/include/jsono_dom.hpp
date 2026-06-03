#pragma once

#include "jsono.hpp"
#include "jsono_writer.hpp"

#include "yyjson.hpp"
#include "string_view.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace duckdb {
namespace jsono_dom {

using namespace jsono;
using namespace duckdb_yyjson;

// Hash helpers — short-input wyhash-inspired mixer. The writer hashes each
// object's DOM-order key sequence into a 64-bit fingerprint and looks the
// result up in JsonoBuilder.shape_cache; on hit, the precomputed sort
// permutation is reused and std::sort is skipped entirely.
//
// 64-bit collision risk is negligible (~5e-20 per row) for any realistic
// data scale, so cache hits are treated as authoritative without a separate
// verification pass.
inline uint64_t HashMix64(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
	__uint128_t r = static_cast<__uint128_t>(a) * b;
	return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
#else
	// Portable fallback — slower but correct.
	uint64_t ah = a >> 32;
	uint64_t al = a & 0xFFFFFFFFULL;
	uint64_t bh = b >> 32;
	uint64_t bl = b & 0xFFFFFFFFULL;
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

struct ShapeCacheEntry {
	uint64_t hash = 0;
	uint32_t n_keys = 0;
	// perm[i] tells emit-pass-i which kvs[kv_offset + perm[i]] to use.
	// Reserved on JsonoBuilder construction so steady-state operation
	// doesn't churn the allocator across LRU evictions.
	std::vector<uint32_t> perm;
};

// Direct-mapped shape cache. Slot = hash & (SIZE-1); collisions evict.
// O(1) lookup with one L1/L2 access — no linear scan, no LRU bookkeeping.
// On workloads with many distinct schemas (Yandex has 46k top-level keysets
// across 200k rows) a direct-mapped table sized 2-8× the working schema set
// gives uniformly high hit rates without the scan-cost ceiling of a
// linear-probe LRU.
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

	// Sizing heuristic for event-tracking JSON (~5% keys, ~50% string values,
	// ~1 slot per ~24 input bytes). Called once per chunk with the chunk's
	// total input bytes; vector::reserve is a no-op if capacity already covers.
	void ReserveForChunk(size_t total_input_bytes) {
		auto need_slots = total_input_bytes / 24 + 64;
		auto need_keys = total_input_bytes / 20 + 64;
		auto need_values = total_input_bytes / 2 + 64;
		Reserve(need_slots, need_keys, need_values);
		if (kvs.capacity() < 256) {
			kvs.reserve(256);
		}
		if (indices.capacity() < 256) {
			indices.reserve(256);
		}
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

} // namespace jsono_dom
} // namespace duckdb
