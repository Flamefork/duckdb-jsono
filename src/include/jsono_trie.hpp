#pragma once

#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

#include <algorithm>

namespace duckdb {
namespace jsono {

// Shared trie skeleton for the two path-set engines: jsono_transform's per-row STRUCT build and
// the optimizer's fused `->>` projector. Both merge their N paths into one trie, then walk the
// document once dispatching every leaf. This header owns only the trie's STRUCTURAL parts — the
// edge/node layout, the prefix-merge build (GetOrAdd* + the per-node edge sort), and the
// simple-scalar-leaf shortcut. It is bind-time machinery; nothing here runs per row.

struct JsonoTrieEdge {
	JsonoTrieEdge() = default;
	JsonoTrieEdge(string key_p, idx_t child_p) : key(std::move(key_p)), child(child_p) {
	}

	string key;
	idx_t child = DConstants::INVALID_INDEX;
};

struct JsonoTrieIndexEdge {
	JsonoTrieIndexEdge() = default;
	JsonoTrieIndexEdge(idx_t index_p, idx_t child_p) : index(index_p), child(child_p) {
	}

	idx_t index = 0;
	idx_t child = DConstants::INVALID_INDEX;
};

// One node type carries every leaf kind. The projector only ever populates `scalar_leaves` and
// `simple_scalar_leaf` and leaves the transform-only fields (`list_leaves`/`join_leaves`/
// `wildcard_child`) at their defaults; a single struct keeps each walk's node-field accesses
// byte-for-byte identical to before the share (no base/derived dispatch), and the default-empty
// vectors cost only bind-time, never a per-row branch.
struct JsonoTrieNode {
	vector<JsonoTrieEdge> key_edges;        // sorted object-key edges
	vector<JsonoTrieIndexEdge> index_edges; // sorted forward array-index edges
	idx_t wildcard_child = DConstants::INVALID_INDEX;
	vector<idx_t> scalar_leaves;
	vector<idx_t> list_leaves;
	vector<idx_t> join_leaves;
	// A leaf node holding exactly one scalar field and no child edges: the edge dispatch can
	// write/null it directly without re-loading the child node. Set by the caller's leaf-finalize
	// hook (the predicate differs between engines: the projector only has scalar leaves, transform
	// must also confirm no list/join/wildcard).
	idx_t simple_scalar_leaf = DConstants::INVALID_INDEX;
};

// The merge-build over a path set. The caller drives it: insert each field's steps to reach its
// leaf node, attach the leaf there (its own leaf-kind policy), then sort. GetOrAdd* dedups shared
// prefixes; the sort gives every node's edges the sorted order the lockstep walk relies on.
struct JsonoTrie {
	vector<JsonoTrieNode> nodes;
	vector<idx_t> field_leaf_nodes;
	vector<vector<idx_t>> field_node_chains;

	void Reset() {
		nodes.clear();
		nodes.emplace_back();
		field_leaf_nodes.clear();
		field_node_chains.clear();
	}

	void PrepareFieldMetadata(idx_t field_count) {
		field_leaf_nodes.assign(field_count, DConstants::INVALID_INDEX);
		field_node_chains.assign(field_count, vector<idx_t>());
	}

	idx_t GetOrAddKeyChild(idx_t node_index, const string &key) {
		for (auto &edge : nodes[node_index].key_edges) {
			if (edge.key == key) {
				return edge.child;
			}
		}
		auto child = nodes.size();
		nodes.emplace_back();
		nodes[node_index].key_edges.push_back(JsonoTrieEdge {key, child});
		return child;
	}

	idx_t GetOrAddIndexChild(idx_t node_index, idx_t index) {
		for (auto &edge : nodes[node_index].index_edges) {
			if (edge.index == index) {
				return edge.child;
			}
		}
		auto child = nodes.size();
		nodes.emplace_back();
		nodes[node_index].index_edges.push_back(JsonoTrieIndexEdge {index, child});
		return child;
	}

	idx_t GetOrAddWildcardChild(idx_t node_index) {
		auto &node = nodes[node_index];
		if (node.wildcard_child != DConstants::INVALID_INDEX) {
			return node.wildcard_child;
		}
		auto child = nodes.size();
		nodes.emplace_back();
		nodes[node_index].wildcard_child = child;
		return child;
	}

	// Descend the steps from the root, materializing edges, and return the leaf node index the
	// caller attaches its leaf to. The caller decides whether a wildcard step is admissible.
	idx_t WalkToLeaf(const vector<PathStep> &steps) {
		vector<idx_t> *chain = nullptr;
		return WalkToLeaf(steps, chain);
	}

	idx_t WalkToLeaf(const vector<PathStep> &steps, vector<idx_t> *chain) {
		idx_t node_index = 0;
		if (chain) {
			chain->clear();
			chain->push_back(node_index);
		}
		for (auto &step : steps) {
			switch (step.kind) {
			case PathStepKind::Key:
				node_index = GetOrAddKeyChild(node_index, step.key);
				break;
			case PathStepKind::Index:
				node_index = GetOrAddIndexChild(node_index, step.index);
				break;
			case PathStepKind::Wildcard:
				node_index = GetOrAddWildcardChild(node_index);
				break;
			}
			if (chain) {
				chain->push_back(node_index);
			}
		}
		return node_index;
	}

	idx_t WalkFieldToLeaf(idx_t field_index, const vector<PathStep> &steps) {
		if (field_index >= field_leaf_nodes.size()) {
			throw InternalException("jsono trie: field metadata not prepared");
		}
		auto leaf = WalkToLeaf(steps, &field_node_chains[field_index]);
		field_leaf_nodes[field_index] = leaf;
		return leaf;
	}

	// Sort every node's edges; the lockstep object/array walk advances edge and key cursors in
	// sorted order. The simple-scalar-leaf shortcut is caller-finalized (see JsonoTrieNode).
	void SortEdges() {
		for (auto &node : nodes) {
			std::sort(node.key_edges.begin(), node.key_edges.end(),
			          [](const JsonoTrieEdge &left, const JsonoTrieEdge &right) { return left.key < right.key; });
			std::sort(node.index_edges.begin(), node.index_edges.end(),
			          [](const JsonoTrieIndexEdge &left, const JsonoTrieIndexEdge &right) {
				          return left.index < right.index;
			          });
		}
	}
};

// Per-node set-associative rank cache shared by both walks. Rows interleave several object shapes
// per node (event archetypes), so each trie node keeps a small set of ways: one per recently seen
// shape class. A matching stored shape_hash proves the cached ranks of every edge with one int
// compare; otherwise the per-edge ValidateCachedObjectRank check (or a full re-resolve) runs.
//
// Slots are per trie node (the trie is bind-time constant), sized to the node's key edges at Init.
// No aliasing across nodes and no resizing means a held entry reference survives the recursive
// descent: recursion only touches descendant nodes. The cache is keyed by the node vector, so both
// engines pass their own node storage (transform's bind-data vector, the projector's JsonoTrie).
constexpr idx_t JSONO_TRIE_RANK_WAYS = 8;

struct JsonoTrieRankCacheEntry {
	bool valid = false;
	size_t key_count = 0;
	uint64_t shape_hash = 0;
	bool has_shape_hash = false;
	vector<size_t> ranks;
	vector<uint8_t> found;
};

inline idx_t JsonoTrieRankWay(const ObjectLayout &layout) {
	return idx_t((uint64_t(layout.key_count) * 0x9E3779B97F4A7C15ULL ^ layout.shape_hash) >> 61);
}

struct JsonoTrieRankCache {
	vector<JsonoTrieRankCacheEntry> entries;

	void Init(const vector<JsonoTrieNode> &nodes) {
		entries.assign(nodes.size() * JSONO_TRIE_RANK_WAYS, JsonoTrieRankCacheEntry {});
		for (idx_t node_index = 0; node_index < nodes.size(); node_index++) {
			auto edge_count = nodes[node_index].key_edges.size();
			for (idx_t way = 0; way < JSONO_TRIE_RANK_WAYS; way++) {
				auto &entry = entries[node_index * JSONO_TRIE_RANK_WAYS + way];
				entry.ranks.resize(edge_count);
				entry.found.resize(edge_count);
			}
		}
	}

	const JsonoTrieRankCacheEntry &Get(const vector<JsonoTrieNode> &nodes, idx_t node_index, const JsonoView &view,
	                                   const ObjectLayout &layout) {
		auto &node = nodes[node_index];
		auto &entry = entries[node_index * JSONO_TRIE_RANK_WAYS + JsonoTrieRankWay(layout)];
		auto cache_valid = entry.valid && entry.key_count == layout.key_count;
		if (cache_valid) {
			// A matching stored shape_hash proves the object's sorted key sequence, validating every
			// edge's cached rank with one int compare. The slot is per node, so the (bind-time
			// constant) key set needs no re-check.
			if (TrustShapeHash() && layout.has_span && entry.has_shape_hash) {
				cache_valid = entry.shape_hash == layout.shape_hash;
			} else {
				for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
					if (!ValidateCachedObjectRank(view, layout, node.key_edges[edge_index].key, entry.ranks[edge_index],
					                              entry.found[edge_index])) {
						cache_valid = false;
						break;
					}
				}
			}
		}
		if (cache_valid) {
			return entry;
		}
		entry.valid = true;
		entry.key_count = layout.key_count;
		entry.shape_hash = layout.shape_hash;
		entry.has_shape_hash = layout.has_span;
		for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
			size_t rank = 0;
			entry.found[edge_index] = FindObjectKeyRank(view, layout, node.key_edges[edge_index].key, rank);
			entry.ranks[edge_index] = rank;
		}
		return entry;
	}
};

} // namespace jsono
} // namespace duckdb
