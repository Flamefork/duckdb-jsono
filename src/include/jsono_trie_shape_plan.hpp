#pragma once

#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_trie.hpp"
#include "jsono_trie_walk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace duckdb {
namespace jsono {

struct JsonoTrieShapeScalar {
	idx_t field_index = DConstants::INVALID_INDEX;
	JsonoScalarKind kind = JsonoScalarKind::Null;
	bool bool_value = false;
	size_t length_index = 0;
	size_t num_index = 0;
	idx_t offset_slot = 0;
};

struct JsonoTrieShapeWalk {
	idx_t node_index;
	size_t pos;
	size_t length_cursor;
	size_t num_cursor;
	idx_t offset_slot;
};

struct JsonoTrieShapePlan {
	vector<idx_t> null_fields;
	vector<idx_t> null_subtree_nodes;
	vector<JsonoTrieShapeScalar> scalars;
	vector<JsonoTrieShapeWalk> walks;
	vector<size_t> needed_length_indices;
	size_t lengths_bound = 0;
	size_t nums_bound = 0;
};

struct JsonoTrieShapeCoverCandidate {
	JsonoTrieShapeCoverCandidate() = default;
	JsonoTrieShapeCoverCandidate(idx_t node_index_p, idx_t field_index_p, idx_t prefix_len_p,
	                             vector<idx_t> node_chain_p)
	    : node_index(node_index_p), field_index(field_index_p), prefix_len(prefix_len_p),
	      node_chain(std::move(node_chain_p)) {
	}

	idx_t node_index = DConstants::INVALID_INDEX;
	idx_t field_index = DConstants::INVALID_INDEX;
	idx_t prefix_len = 0;
	vector<idx_t> node_chain;
};

constexpr idx_t JSONO_TRIE_SHAPE_PLAN_BUCKETS = 256;
constexpr idx_t JSONO_TRIE_SHAPE_PLAN_WAYS = 4;
constexpr size_t JSONO_TRIE_SHAPE_PLAN_MAX_SHAPE_BYTES = 2048;
constexpr uint64_t JSONO_TRIE_SHAPE_PLAN_WINDOW = 1024;
constexpr uint64_t JSONO_TRIE_SHAPE_PLAN_WARMUP = JSONO_TRIE_SHAPE_PLAN_BUCKETS * JSONO_TRIE_SHAPE_PLAN_WAYS * 4;

JSONO_ALWAYS_INLINE uint64_t JsonoTrieShapeHash(uint64_t h, const char *data, size_t size) {
	uint64_t lane0 = h ^ HASH_SEED;
	uint64_t lane1 = h ^ 0x9E3779B97F4A7C15ULL;
	uint64_t lane2 = h ^ 0xC2B2AE3D27D4EB4FULL;
	uint64_t lane3 = h ^ 0x165667B19E3779F9ULL;
	size_t i = 0;
	for (; i + 4 * sizeof(uint64_t) <= size; i += 4 * sizeof(uint64_t)) {
		uint64_t w0, w1, w2, w3;
		std::memcpy(&w0, data + i, sizeof(w0));
		std::memcpy(&w1, data + i + 8, sizeof(w1));
		std::memcpy(&w2, data + i + 16, sizeof(w2));
		std::memcpy(&w3, data + i + 24, sizeof(w3));
		lane0 = HashMix64(lane0 ^ w0, HASH_PRIME);
		lane1 = HashMix64(lane1 ^ w1, HASH_PRIME);
		lane2 = HashMix64(lane2 ^ w2, HASH_PRIME);
		lane3 = HashMix64(lane3 ^ w3, HASH_PRIME);
	}
	uint64_t tail = h;
	for (; i + sizeof(uint64_t) <= size; i += sizeof(uint64_t)) {
		uint64_t w;
		std::memcpy(&w, data + i, sizeof(w));
		tail = HashMix64(tail ^ w, HASH_PRIME);
	}
	if (i < size) {
		uint64_t w = 0;
		std::memcpy(&w, data + i, size - i);
		tail = HashMix64(tail ^ w, HASH_PRIME);
	}
	return HashMix64(lane0 ^ lane1, HASH_PRIME) ^ HashMix64(lane2 ^ lane3, HASH_PRIME) ^
	       HashMix64(tail ^ uint64_t(size), HASH_PRIME);
}

JSONO_ALWAYS_INLINE uint64_t JsonoTrieShapeBlobHash(const JsonoBlobRow &blob) {
	auto hash = JsonoTrieShapeHash(HASH_SEED, blob.slots.GetData(), blob.slots.GetSize());
	return JsonoTrieShapeHash(hash, blob.key_heap.GetData(), blob.key_heap.GetSize());
}

template <class PLAN>
struct JsonoTrieShapePlanCache {
	struct Entry {
		uint64_t sample_hash = 0;
		uint32_t stamp = 0;
		string slots;
		string key_heap;
		unique_ptr<PLAN> plan;
	};

	explicit JsonoTrieShapePlanCache(const char *stats_label_p = "jsono shape-plan") : stats_label(stats_label_p) {
	}

	~JsonoTrieShapePlanCache() {
		const char *env = std::getenv("JSONO_SHAPE_PLAN_STATS");
		if (env && env[0] == '1') {
			std::fprintf(stderr, "%s: lookups=%llu hits=%llu builds=%llu disabled=%d\n", stats_label,
			             (unsigned long long)(total_lookups + window_lookups),
			             (unsigned long long)(total_hits + window_hits), (unsigned long long)builds, int(disabled));
		}
	}

	const char *stats_label;
	vector<Entry> entries;
	uint32_t clock = 0;
	bool disabled = false;
	uint64_t window_lookups = 0;
	uint64_t window_hits = 0;
	uint64_t total_lookups = 0;
	uint64_t total_hits = 0;
	uint64_t builds = 0;
	uint64_t recovery_cooldown_lookups = 0;
	uint64_t recovery_lookups = 0;
	uint64_t recovery_hits = 0;
	uint64_t recovery_hash = 0;
	string recovery_slots;
	string recovery_key_heap;

	static bool ShapeBytesEqual(const string &slots, const string &key_heap, const JsonoBlobRow &blob) {
		return slots.size() == blob.slots.GetSize() && key_heap.size() == blob.key_heap.GetSize() &&
		       std::memcmp(slots.data(), blob.slots.GetData(), slots.size()) == 0 &&
		       std::memcmp(key_heap.data(), blob.key_heap.GetData(), key_heap.size()) == 0;
	}

	void ResetRecoveryProbe() {
		recovery_lookups = 0;
		recovery_hits = 0;
		recovery_hash = 0;
		recovery_slots.clear();
		recovery_key_heap.clear();
	}

	// Gates whether a row consults the plan cache and yields its shape hash when it does. While
	// enabled every row looks up. Once EndOfWindowCheck disables the cache (a window thrashed below
	// the 75% hit-rate floor), recovery probes whether the stream has since stabilized: after a
	// WARMUP-row cooldown, sample one WINDOW-row probe and re-enable only if at least 75% of probe
	// rows repeat the first probe row's shape byte-for-byte (the same 3/4 floor disable uses, so a
	// stable tail recovers and a still-thrashing one stays disabled and re-enters cooldown). Probe
	// rows themselves take the per-row walk (return false), so recovery never serves an unbuilt plan.
	bool CanLookup(const JsonoBlobRow &blob, uint64_t &hash) {
		if (!disabled) {
			hash = JsonoTrieShapeBlobHash(blob);
			return true;
		}
		if (recovery_lookups == 0 && recovery_cooldown_lookups < JSONO_TRIE_SHAPE_PLAN_WARMUP) {
			recovery_cooldown_lookups++;
			return false;
		}
		hash = JsonoTrieShapeBlobHash(blob);
		if (recovery_lookups == 0) {
			recovery_hash = hash;
			recovery_slots.assign(blob.slots.GetData(), blob.slots.GetSize());
			recovery_key_heap.assign(blob.key_heap.GetData(), blob.key_heap.GetSize());
		} else if (recovery_hash == hash && ShapeBytesEqual(recovery_slots, recovery_key_heap, blob)) {
			recovery_hits++;
		}
		recovery_lookups++;
		if (recovery_lookups < JSONO_TRIE_SHAPE_PLAN_WINDOW) {
			return false;
		}
		if (recovery_hits * 4 >= recovery_lookups * 3) {
			disabled = false;
			recovery_cooldown_lookups = 0;
			ResetRecoveryProbe();
			return true;
		}
		recovery_cooldown_lookups = 0;
		ResetRecoveryProbe();
		return false;
	}

	PLAN *Find(uint64_t hash, const JsonoBlobRow &blob) {
		if (entries.empty()) {
			entries.resize(JSONO_TRIE_SHAPE_PLAN_BUCKETS * JSONO_TRIE_SHAPE_PLAN_WAYS);
		}
		window_lookups++;
		auto base = idx_t(hash & (JSONO_TRIE_SHAPE_PLAN_BUCKETS - 1)) * JSONO_TRIE_SHAPE_PLAN_WAYS;
		for (idx_t way = 0; way < JSONO_TRIE_SHAPE_PLAN_WAYS; way++) {
			auto &entry = entries[base + way];
			if (entry.plan && entry.sample_hash == hash && ShapeBytesEqual(entry.slots, entry.key_heap, blob)) {
				entry.stamp = ++clock;
				window_hits++;
				return entry.plan.get();
			}
		}
		return nullptr;
	}

	PLAN *Insert(uint64_t hash, const JsonoBlobRow &blob, unique_ptr<PLAN> plan) {
		builds++;
		auto base = idx_t(hash & (JSONO_TRIE_SHAPE_PLAN_BUCKETS - 1)) * JSONO_TRIE_SHAPE_PLAN_WAYS;
		auto victim = base;
		for (idx_t way = 0; way < JSONO_TRIE_SHAPE_PLAN_WAYS; way++) {
			auto &entry = entries[base + way];
			if (!entry.plan) {
				victim = base + way;
				break;
			}
			if (entry.stamp < entries[victim].stamp) {
				victim = base + way;
			}
		}
		auto &entry = entries[victim];
		entry.sample_hash = hash;
		entry.stamp = ++clock;
		entry.slots.assign(blob.slots.GetData(), blob.slots.GetSize());
		entry.key_heap.assign(blob.key_heap.GetData(), blob.key_heap.GetSize());
		entry.plan = std::move(plan);
		return entry.plan.get();
	}

	void EndOfWindowCheck() {
		if (window_lookups < JSONO_TRIE_SHAPE_PLAN_WINDOW) {
			return;
		}
		total_lookups += window_lookups;
		total_hits += window_hits;
		if (total_lookups > JSONO_TRIE_SHAPE_PLAN_WARMUP && window_hits * 4 < window_lookups * 3) {
			disabled = true;
			entries.clear();
			entries.shrink_to_fit();
			recovery_cooldown_lookups = 0;
			ResetRecoveryProbe();
		}
		window_lookups = 0;
		window_hits = 0;
	}
};

inline void JsonoTrieShapePlanScalarAt(JsonoTrieShapePlan &plan, const JsonoView &view, const JsonoCursor &cursor,
                                       idx_t field_index) {
	auto slot = view.SlotAt(cursor.pos);
	JsonoTrieShapeScalar action;
	action.field_index = field_index;
	switch (SlotTag(slot)) {
	case tag::OBJ_START:
	case tag::ARR_START:
	case tag::VAL_NULL:
		plan.null_fields.push_back(field_index);
		return;
	case tag::VAL_STR_HEAP:
		action.kind = JsonoScalarKind::String;
		action.length_index = cursor.length_cursor;
		break;
	case tag::VAL_INT60:
		action.kind = JsonoScalarKind::Int64;
		action.num_index = cursor.num_cursor;
		break;
	case tag::VAL_DEC60:
		action.kind = JsonoScalarKind::Dec60;
		action.num_index = cursor.num_cursor;
		break;
	case tag::VAL_EXT:
		switch (ExtSubtype(slot)) {
		case ext_subtype::NUMBER:
			action.kind = JsonoScalarKind::NumberText;
			action.length_index = cursor.length_cursor;
			break;
		case ext_subtype::INT64:
			action.kind = JsonoScalarKind::Int64;
			action.num_index = cursor.num_cursor;
			break;
		case ext_subtype::UINT64:
			action.kind = JsonoScalarKind::UInt64;
			action.num_index = cursor.num_cursor;
			break;
		case ext_subtype::DOUBLE:
			action.kind = JsonoScalarKind::Double;
			action.num_index = cursor.num_cursor;
			break;
		default:
			throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
		}
		break;
	case tag::VAL_TRUE:
		action.kind = JsonoScalarKind::Bool;
		action.bool_value = true;
		break;
	case tag::VAL_FALSE:
		action.kind = JsonoScalarKind::Bool;
		action.bool_value = false;
		break;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
	plan.scalars.push_back(action);
}

inline void JsonoTrieShapePlanFinalize(JsonoTrieShapePlan &plan) {
	auto &needed = plan.needed_length_indices;
	for (auto &action : plan.scalars) {
		if (action.kind == JsonoScalarKind::String || action.kind == JsonoScalarKind::NumberText) {
			needed.push_back(action.length_index);
		}
	}
	for (auto &walk : plan.walks) {
		needed.push_back(walk.length_cursor);
	}
	std::sort(needed.begin(), needed.end());
	needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
	for (auto &action : plan.scalars) {
		switch (action.kind) {
		case JsonoScalarKind::String:
		case JsonoScalarKind::NumberText:
			action.offset_slot =
			    idx_t(std::lower_bound(needed.begin(), needed.end(), action.length_index) - needed.begin());
			plan.lengths_bound = std::max(plan.lengths_bound, action.length_index + 1);
			break;
		case JsonoScalarKind::Int64:
		case JsonoScalarKind::UInt64:
		case JsonoScalarKind::Double:
		case JsonoScalarKind::Dec60:
			plan.nums_bound = std::max(plan.nums_bound, action.num_index + 1);
			break;
		default:
			break;
		}
	}
	for (auto &walk : plan.walks) {
		walk.offset_slot = idx_t(std::lower_bound(needed.begin(), needed.end(), walk.length_cursor) - needed.begin());
		plan.lengths_bound = std::max(plan.lengths_bound, walk.length_cursor);
	}
}

JSONO_ALWAYS_INLINE void JsonoTrieShapePlanScanLengthOffsets(const JsonoTrieShapePlan &plan, const JsonoView &view,
                                                             vector<uint64_t> &offsets) {
	auto &needed = plan.needed_length_indices;
	offsets.resize(needed.size());
	if (needed.empty()) {
		return;
	}
	auto lengths_raw = view.LengthsBytes(0, plan.lengths_bound);
	uint64_t sum = 0;
	idx_t next = 0;
	for (size_t i = 0; next < needed.size(); i++) {
		if (needed[next] == i) {
			offsets[next] = sum;
			next++;
			if (next >= needed.size()) {
				break;
			}
		}
		uint32_t len;
		std::memcpy(&len, lengths_raw + i * sizeof(uint32_t), sizeof(len));
		sum += len;
	}
}

inline void JsonoTrieShapePlanSelectCoverNodes(const vector<JsonoTrieShapeCoverCandidate> &candidates,
                                               vector<JsonoTrieShapeCoverCandidate> &cover_nodes) {
	cover_nodes.clear();
	for (idx_t i = 0; i < candidates.size(); i++) {
		auto &candidate = candidates[i];
		bool covered = false;
		for (idx_t j = 0; j < candidates.size() && !covered; j++) {
			if (i == j) {
				continue;
			}
			auto &other = candidates[j];
			covered = std::find(candidate.node_chain.begin(), candidate.node_chain.end() - 1, other.node_index) !=
			          candidate.node_chain.end() - 1;
		}
		if (covered) {
			continue;
		}
		auto duplicate =
		    std::find_if(cover_nodes.begin(), cover_nodes.end(), [&](const JsonoTrieShapeCoverCandidate &cover) {
			    return cover.node_index == candidate.node_index;
		    });
		if (duplicate == cover_nodes.end()) {
			cover_nodes.push_back(candidate);
		}
	}
}

inline idx_t JsonoTrieShapePlanCountScalarLeaves(const vector<JsonoTrieNode> &nodes, idx_t node_index,
                                                 vector<idx_t> &leaf_counts) {
	auto &node = nodes[node_index];
	idx_t count = node.scalar_leaves.size();
	for (auto &edge : node.key_edges) {
		count += JsonoTrieShapePlanCountScalarLeaves(nodes, edge.child, leaf_counts);
	}
	for (auto &edge : node.index_edges) {
		count += JsonoTrieShapePlanCountScalarLeaves(nodes, edge.child, leaf_counts);
	}
	leaf_counts[node_index] = count;
	return count;
}

inline void JsonoTrieShapePlanBuildScalarCoverNodes(const JsonoTrie &trie,
                                                    vector<JsonoTrieShapeCoverCandidate> &cover_nodes) {
	cover_nodes.clear();
	vector<idx_t> leaf_counts(trie.nodes.size(), 0);
	JsonoTrieShapePlanCountScalarLeaves(trie.nodes, 0, leaf_counts);
	vector<JsonoTrieShapeCoverCandidate> candidates;
	if (!trie.nodes[0].scalar_leaves.empty() && leaf_counts[0] >= 2) {
		candidates.push_back(JsonoTrieShapeCoverCandidate(0, trie.nodes[0].scalar_leaves[0], 0, vector<idx_t>(1, 0)));
		JsonoTrieShapePlanSelectCoverNodes(candidates, cover_nodes);
		return;
	}
	for (idx_t field_index = 0; field_index < trie.field_node_chains.size(); field_index++) {
		auto &chain = trie.field_node_chains[field_index];
		for (idx_t prefix_len = 1; prefix_len < chain.size(); prefix_len++) {
			auto node_index = chain[prefix_len];
			if (leaf_counts[node_index] < 2) {
				continue;
			}
			candidates.push_back(JsonoTrieShapeCoverCandidate(
			    node_index, field_index, prefix_len, vector<idx_t>(chain.begin(), chain.begin() + prefix_len + 1)));
			break;
		}
	}
	JsonoTrieShapePlanSelectCoverNodes(candidates, cover_nodes);
}

inline bool JsonoTrieShapePlanFieldCovered(const JsonoTrie &trie, idx_t field_index,
                                           const vector<JsonoTrieShapeCoverCandidate> &cover_nodes) {
	auto &chain = trie.field_node_chains[field_index];
	for (auto &cover : cover_nodes) {
		if (std::find(chain.begin(), chain.end(), cover.node_index) != chain.end()) {
			return true;
		}
	}
	return false;
}

template <class PLAN, class FIELD>
inline void JsonoTrieShapePlanAddCoverWalks(PLAN &plan, const vector<FIELD> &fields,
                                            const vector<JsonoTrieShapeCoverCandidate> &cover_nodes,
                                            const JsonoView &view) {
	for (auto &cover : cover_nodes) {
		auto &field = fields[cover.field_index];
		JsonoCursor cursor;
		if (LocatePathPrefix(nullptr, 0, field.path.steps, cover.prefix_len, view, cursor)) {
			plan.walks.push_back(
			    JsonoTrieShapeWalk {cover.node_index, cursor.pos, cursor.length_cursor, cursor.num_cursor, 0});
		} else {
			plan.null_subtree_nodes.push_back(cover.node_index);
		}
	}
}

template <class PLAN, class FIELD, class POLICY>
inline void JsonoTrieShapePlanAddFieldActions(PLAN &plan, const JsonoTrie &trie, const vector<FIELD> &fields,
                                              const vector<JsonoTrieShapeCoverCandidate> &cover_nodes,
                                              const JsonoView &view, POLICY &policy) {
	vector<idx_t> walked_nodes;
	for (idx_t field_index = 0; field_index < fields.size(); field_index++) {
		if (!policy.IncludeField(field_index)) {
			continue;
		}
		if (JsonoTrieShapePlanFieldCovered(trie, field_index, cover_nodes)) {
			continue;
		}
		JsonoCursor cursor;
		if (!LocatePathSteps(nullptr, 0, fields[field_index].path.steps, view, cursor)) {
			policy.Missing(plan, field_index);
			continue;
		}
		policy.Found(plan, view, cursor, field_index, walked_nodes);
	}
}

template <class PLAN, class FIELD, class POLICY>
inline void JsonoTrieShapePlanAddReads(PLAN &plan, const JsonoTrie &trie, const vector<FIELD> &fields,
                                       const vector<JsonoTrieShapeCoverCandidate> &cover_nodes, const JsonoView &view,
                                       POLICY &policy) {
	JsonoTrieShapePlanAddCoverWalks(plan, fields, cover_nodes, view);
	JsonoTrieShapePlanAddFieldActions(plan, trie, fields, cover_nodes, view, policy);
}

template <class POLICY>
inline void JsonoTrieShapePlanApplyNullSubtrees(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state,
                                                const JsonoTrieShapePlan &plan, const JsonoView &view, idx_t row) {
	for (auto node_index : plan.null_subtree_nodes) {
		JsonoTrieNullSubtree<POLICY>(nodes, state, node_index, view, row);
	}
}

template <class POLICY>
inline void JsonoTrieShapePlanApplyWalks(const vector<JsonoTrieNode> &nodes, JsonoTrieRankCache &rank_cache,
                                         typename POLICY::State &state, const JsonoTrieShapePlan &plan,
                                         const JsonoView &view, const vector<uint64_t> &offsets, idx_t row) {
	JsonoTrieWalkContext<POLICY> ctx {nodes, rank_cache, state};
	for (auto &walk : plan.walks) {
		JsonoCursor cursor;
		cursor.pos = walk.pos;
		cursor.string_cursor = offsets[walk.offset_slot];
		cursor.length_cursor = walk.length_cursor;
		cursor.num_cursor = walk.num_cursor;
		JsonoTrieApplyNode<POLICY>(ctx, walk.node_index, view, cursor, row);
	}
}

} // namespace jsono
} // namespace duckdb
