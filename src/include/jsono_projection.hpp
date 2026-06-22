#pragma once

#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_trie.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

// One-pass multi-field projection over a jsono document, sharing the same trie engine as
// jsono_transform: N paths merge into a trie, the document is walked once, and each scalar leaf
// is dispatched to a POLICY. The optimizer's fused `->>` projector rides this instead of running
// N independent locate descents per row, inheriting the trie's shared-prefix merge and per-node
// rank cache. The trie here carries only object-key and forward array-index edges with scalar
// leaves — the exact path subset the projector fuse admits (TryReadExtractPath rejects wildcards
// and negative indexes), so there is no wildcard/list/join leaf to materialize.

namespace jsono {

// The trie built from a field set, on the shared JsonoTrie skeleton. `FIELD` must expose
// `const vector<PathStep> &steps`. Built once at bind time; the walk reads it as a constant. The
// projector admits only scalar leaves over key/index edges — wildcards are rejected upstream by
// TryReadExtractPath, so a wildcard step here is an internal invariant break.
struct ProjectTrie : JsonoTrie {
	template <class FIELD>
	void Build(const vector<FIELD> &fields) {
		Reset();
		for (idx_t field_index = 0; field_index < fields.size(); field_index++) {
			for (auto &step : fields[field_index].steps) {
				if (step.kind == PathStepKind::Wildcard) {
					throw InternalException("jsono projector trie: wildcard step is not projectable");
				}
			}
			nodes[WalkToLeaf(fields[field_index].steps)].scalar_leaves.push_back(field_index);
		}
		SortEdges();
		for (auto &node : nodes) {
			if (node.scalar_leaves.size() == 1 && node.key_edges.empty() && node.index_edges.empty()) {
				node.simple_scalar_leaf = node.scalar_leaves[0];
			}
		}
	}
};

// The projector's per-node rank cache is the shared JsonoTrieRankCache (src/include/jsono_trie.hpp),
// keyed by this trie's node vector. It carries its own copy (not transform's) only because the two
// engines run from separate local states; the cache machinery itself is one definition.

// The projection walk context: the trie (bind-time constant), the per-row rank cache, and the
// POLICY's own state (carried by reference so the leaf hooks reach the reader/result without the
// walk knowing about either).
template <class POLICY>
struct ProjectWalkContext {
	const ProjectTrie &trie;
	JsonoTrieRankCache &rank_cache;
	typename POLICY::State &state;
};

template <class POLICY>
void ApplyProjectNode(ProjectWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                      const JsonoCursor &cursor, idx_t row);

// A trie edge whose child key is absent: a simple scalar leaf nulls its own output, any other child
// nulls its whole subtree. Mirrors transform's NullScalarEdge.
template <class POLICY>
void NullProjectSubtree(ProjectWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view, idx_t row) {
	auto &node = ctx.trie.nodes[node_index];
	for (auto field_index : node.scalar_leaves) {
		POLICY::NullLeaf(ctx.state, field_index, view, row);
	}
	for (auto &edge : node.key_edges) {
		NullProjectSubtree(ctx, edge.child, view, row);
	}
	for (auto &edge : node.index_edges) {
		NullProjectSubtree(ctx, edge.child, view, row);
	}
}

template <class POLICY>
JSONO_ALWAYS_INLINE void NullProjectEdge(ProjectWalkContext<POLICY> &ctx, idx_t simple_scalar_leaf, idx_t child,
                                         const JsonoView &view, idx_t row) {
	if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
		NullProjectSubtree(ctx, child, view, row);
	} else {
		POLICY::NullLeaf(ctx.state, simple_scalar_leaf, view, row);
	}
}

template <class POLICY>
JSONO_ALWAYS_INLINE void ApplyProjectEdge(ProjectWalkContext<POLICY> &ctx, idx_t simple_scalar_leaf, idx_t child,
                                          const JsonoView &view, const JsonoCursor &value_cursor, idx_t row) {
	if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
		ApplyProjectNode(ctx, child, view, value_cursor, row);
	} else {
		POLICY::WriteLeaf(ctx.state, simple_scalar_leaf, view, value_cursor, row);
	}
}

template <class POLICY>
void ApplyProjectObjectCheckpoints(ProjectWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                                   const ObjectLayout &layout, const JsonoCursor &obj_cursor, idx_t row) {
	auto &node = ctx.trie.nodes[node_index];
	size_t value_rank = 0;
	JsonoCursor value_block_base = obj_cursor;
	value_block_base.pos = layout.value_start;
	JsonoCursor value_cursor = value_block_base;

	auto &entry = ctx.rank_cache.Get(ctx.trie.nodes, node_index, view, layout);
	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		auto &edge = node.key_edges[edge_index];
		if (!entry.found[edge_index]) {
			NullProjectEdge(ctx, ctx.trie.nodes[edge.child].simple_scalar_leaf, edge.child, view, row);
			continue;
		}
		auto rank = entry.ranks[edge_index];
		MoveObjectValueCursorToRank(view, layout, value_block_base, rank, value_rank, value_cursor);
		ApplyProjectEdge(ctx, ctx.trie.nodes[edge.child].simple_scalar_leaf, edge.child, view, value_cursor, row);
	}
}

template <class POLICY>
void ApplyProjectObjectNode(ProjectWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                            const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.trie.nodes[node_index];
	auto layout = ReadObjectLayout(view, cursor.pos);
	bool use_checkpoints =
	    layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && layout.key_count > OBJECT_CHECKPOINT_STRIDE;
	if (use_checkpoints) {
		ApplyProjectObjectCheckpoints(ctx, node_index, view, layout, cursor, row);
		return;
	}

	JsonoCursor value_cursor = cursor;
	value_cursor.pos = layout.value_start;
	idx_t edge_index = 0;
	for (size_t key_index = 0; key_index < layout.key_count; key_index++) {
		auto key_slot = view.SlotAt(layout.key_start + key_index);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		int key_compare = 1;
		while (edge_index < node.key_edges.size()) {
			key_compare = CompareTrieKeyToJsonKey(node.key_edges[edge_index].key, key);
			if (key_compare >= 0) {
				break;
			}
			auto child = node.key_edges[edge_index].child;
			NullProjectEdge(ctx, ctx.trie.nodes[child].simple_scalar_leaf, child, view, row);
			edge_index++;
		}
		if (edge_index < node.key_edges.size() && key_compare == 0) {
			auto child = node.key_edges[edge_index].child;
			ApplyProjectEdge(ctx, ctx.trie.nodes[child].simple_scalar_leaf, child, view, value_cursor, row);
			edge_index++;
		}
		if (edge_index >= node.key_edges.size()) {
			return;
		}
		SkipValueFast(view, value_cursor);
	}
	while (edge_index < node.key_edges.size()) {
		auto child = node.key_edges[edge_index].child;
		NullProjectEdge(ctx, ctx.trie.nodes[child].simple_scalar_leaf, child, view, row);
		edge_index++;
	}
}

template <class POLICY>
void ApplyProjectArrayNode(ProjectWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                           const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.trie.nodes[node_index];
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos++;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;
	while (value_cursor.pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			NullProjectSubtree(ctx, node.index_edges[fixed_edge_index].child, view, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			ApplyProjectNode(ctx, node.index_edges[fixed_edge_index].child, view, value_cursor, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index >= node.index_edges.size()) {
			return;
		}
		SkipValueFast(view, value_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		NullProjectSubtree(ctx, node.index_edges[fixed_edge_index].child, view, row);
		fixed_edge_index++;
	}
}

template <class POLICY>
void ApplyProjectNode(ProjectWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                      const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.trie.nodes[node_index];
	for (auto field_index : node.scalar_leaves) {
		POLICY::WriteLeaf(ctx.state, field_index, view, cursor, row);
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (!node.key_edges.empty()) {
		if (slot_tag == tag::OBJ_START) {
			ApplyProjectObjectNode(ctx, node_index, view, cursor, row);
		} else {
			for (auto &edge : node.key_edges) {
				NullProjectSubtree(ctx, edge.child, view, row);
			}
		}
	}
	if (!node.index_edges.empty()) {
		if (slot_tag == tag::ARR_START) {
			ApplyProjectArrayNode(ctx, node_index, view, cursor, row);
		} else {
			for (auto &edge : node.index_edges) {
				NullProjectSubtree(ctx, edge.child, view, row);
			}
		}
	}
}

// Walk the trie once over one document root, dispatching every field to POLICY. The root is always
// the whole value at the JsonoCursor() origin.
template <class POLICY>
JSONO_ALWAYS_INLINE void WalkProjectTrie(const ProjectTrie &trie, JsonoTrieRankCache &rank_cache,
                                         typename POLICY::State &state, const JsonoView &view, idx_t row) {
	ProjectWalkContext<POLICY> ctx {trie, rank_cache, state};
	ApplyProjectNode(ctx, 0, view, JsonoCursor(), row);
}

} // namespace jsono
} // namespace duckdb
