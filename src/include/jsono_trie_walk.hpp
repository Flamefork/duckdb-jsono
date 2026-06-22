#pragma once

#include "jsono_trie.hpp"

#include <type_traits>

namespace duckdb {
namespace jsono {

template <class POLICY>
struct JsonoTrieWalkContext {
	const vector<JsonoTrieNode> &nodes;
	JsonoTrieRankCache &rank_cache;
	typename POLICY::State &state;
};

template <class POLICY>
void JsonoTrieApplyNode(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                        const JsonoCursor &cursor, idx_t row);

template <class POLICY>
void JsonoTrieApplyWildcardElement(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                                   const JsonoCursor &cursor, idx_t row);

template <class POLICY>
void JsonoTrieAppendWildcardMissing(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state, idx_t node_index,
                                    const JsonoView &view, idx_t row);

template <class POLICY>
void JsonoTrieNullSubtree(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state, idx_t node_index,
                          const JsonoView &view, idx_t row);

template <class POLICY>
struct JsonoTrieWildcardTag {
	typedef typename std::conditional<POLICY::supports_wildcard, std::true_type, std::false_type>::type Type;
};

template <class POLICY>
JSONO_ALWAYS_INLINE void JsonoTrieNullWildcardSubtree(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state,
                                                      const JsonoTrieNode &node, const JsonoView &view, idx_t row,
                                                      std::false_type) {
}

template <class POLICY>
JSONO_ALWAYS_INLINE void JsonoTrieNullWildcardSubtree(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state,
                                                      const JsonoTrieNode &node, const JsonoView &view, idx_t row,
                                                      std::true_type) {
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		JsonoTrieNullSubtree<POLICY>(nodes, state, node.wildcard_child, view, row);
	}
}

template <class POLICY>
void JsonoTrieNullSubtree(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state, idx_t node_index,
                          const JsonoView &view, idx_t row) {
	auto &node = nodes[node_index];
	POLICY::NullNodeLeaves(state, node, view, row);
	for (auto &edge : node.key_edges) {
		JsonoTrieNullSubtree<POLICY>(nodes, state, edge.child, view, row);
	}
	for (auto &edge : node.index_edges) {
		JsonoTrieNullSubtree<POLICY>(nodes, state, edge.child, view, row);
	}
	JsonoTrieNullWildcardSubtree<POLICY>(nodes, state, node, view, row, typename JsonoTrieWildcardTag<POLICY>::Type());
}

template <class POLICY>
JSONO_ALWAYS_INLINE void JsonoTrieNullEdge(JsonoTrieWalkContext<POLICY> &ctx, idx_t simple_scalar_leaf, idx_t child,
                                           const JsonoView &view, idx_t row) {
	if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
		JsonoTrieNullSubtree<POLICY>(ctx.nodes, ctx.state, child, view, row);
	} else {
		POLICY::NullScalarLeaf(ctx.state, simple_scalar_leaf, view, row);
	}
}

template <class POLICY>
JSONO_ALWAYS_INLINE void JsonoTrieApplyEdge(JsonoTrieWalkContext<POLICY> &ctx, idx_t simple_scalar_leaf, idx_t child,
                                            const JsonoView &view, const JsonoCursor &value_cursor, idx_t row) {
	if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
		JsonoTrieApplyNode<POLICY>(ctx, child, view, value_cursor, row);
	} else {
		POLICY::WriteScalarLeaf(ctx.state, simple_scalar_leaf, view, value_cursor, row);
	}
}

template <class POLICY>
struct JsonoTrieNormalEdgePolicy {
	static constexpr bool emit_index_tail = false;

	static JSONO_ALWAYS_INLINE void OnMissing(JsonoTrieWalkContext<POLICY> &ctx, idx_t child, const JsonoView &view,
	                                          const JsonoCursor &value_cursor, idx_t row) {
		(void)value_cursor;
		JsonoTrieNullEdge<POLICY>(ctx, ctx.nodes[child].simple_scalar_leaf, child, view, row);
	}

	static JSONO_ALWAYS_INLINE void OnPresent(JsonoTrieWalkContext<POLICY> &ctx, idx_t child, const JsonoView &view,
	                                          const JsonoCursor &value_cursor, idx_t row) {
		JsonoTrieApplyEdge<POLICY>(ctx, ctx.nodes[child].simple_scalar_leaf, child, view, value_cursor, row);
	}
};

template <class POLICY>
struct JsonoTrieWildcardEdgePolicy {
	static constexpr bool emit_index_tail = true;

	static JSONO_ALWAYS_INLINE void OnMissing(JsonoTrieWalkContext<POLICY> &ctx, idx_t child, const JsonoView &view,
	                                          const JsonoCursor &value_cursor, idx_t row) {
		(void)value_cursor;
		JsonoTrieAppendWildcardMissing<POLICY>(ctx.nodes, ctx.state, child, view, row);
	}

	static JSONO_ALWAYS_INLINE void OnPresent(JsonoTrieWalkContext<POLICY> &ctx, idx_t child, const JsonoView &view,
	                                          const JsonoCursor &value_cursor, idx_t row) {
		JsonoTrieApplyWildcardElement<POLICY>(ctx, child, view, value_cursor, row);
	}
};

template <class POLICY>
void JsonoTrieAppendWildcardMissing(const vector<JsonoTrieNode> &nodes, typename POLICY::State &state, idx_t node_index,
                                    const JsonoView &view, idx_t row) {
	auto &node = nodes[node_index];
	POLICY::AppendWildcardMissingLeaves(state, node, view, row);
	for (auto &edge : node.key_edges) {
		JsonoTrieAppendWildcardMissing<POLICY>(nodes, state, edge.child, view, row);
	}
	for (auto &edge : node.index_edges) {
		JsonoTrieAppendWildcardMissing<POLICY>(nodes, state, edge.child, view, row);
	}
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		JsonoTrieAppendWildcardMissing<POLICY>(nodes, state, node.wildcard_child, view, row);
	}
}

template <class POLICY, class EDGE_POLICY>
void JsonoTrieWalkObjectCheckpoints(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                                    const ObjectLayout &layout, const JsonoCursor &obj_cursor, idx_t row) {
	auto &node = ctx.nodes[node_index];
	size_t value_rank = 0;
	JsonoCursor value_block_base = obj_cursor;
	value_block_base.pos = layout.value_start;
	JsonoCursor value_cursor = value_block_base;

	auto &entry = ctx.rank_cache.Get(ctx.nodes, node_index, view, layout);
	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		auto &edge = node.key_edges[edge_index];
		if (!entry.found[edge_index]) {
			EDGE_POLICY::OnMissing(ctx, edge.child, view, value_cursor, row);
			continue;
		}
		auto rank = entry.ranks[edge_index];
		MoveObjectValueCursorToRank(view, layout, value_block_base, rank, value_rank, value_cursor);
		EDGE_POLICY::OnPresent(ctx, edge.child, view, value_cursor, row);
	}
	if (EDGE_POLICY::emit_index_tail) {
		for (auto &edge : node.index_edges) {
			EDGE_POLICY::OnMissing(ctx, edge.child, view, value_cursor, row);
		}
	}
}

template <class POLICY, class EDGE_POLICY>
void JsonoTrieWalkObject(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                         const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.nodes[node_index];
	auto layout = ReadObjectLayout(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos = layout.value_start;
	idx_t edge_index = 0;
	bool use_checkpoints =
	    layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && layout.key_count > OBJECT_CHECKPOINT_STRIDE;
	if (use_checkpoints) {
		JsonoTrieWalkObjectCheckpoints<POLICY, EDGE_POLICY>(ctx, node_index, view, layout, cursor, row);
		return;
	}

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
			EDGE_POLICY::OnMissing(ctx, node.key_edges[edge_index].child, view, value_cursor, row);
			edge_index++;
		}
		if (edge_index < node.key_edges.size() && key_compare == 0) {
			EDGE_POLICY::OnPresent(ctx, node.key_edges[edge_index].child, view, value_cursor, row);
			edge_index++;
		}
		if (edge_index >= node.key_edges.size() && (!EDGE_POLICY::emit_index_tail || node.index_edges.empty())) {
			return;
		}
		SkipValueFast(view, value_cursor);
	}
	while (edge_index < node.key_edges.size()) {
		EDGE_POLICY::OnMissing(ctx, node.key_edges[edge_index].child, view, value_cursor, row);
		edge_index++;
	}
	if (EDGE_POLICY::emit_index_tail) {
		for (auto &edge : node.index_edges) {
			EDGE_POLICY::OnMissing(ctx, edge.child, view, value_cursor, row);
		}
	}
}

template <class POLICY>
JSONO_ALWAYS_INLINE bool JsonoTrieApplyArrayTail(JsonoTrieWalkContext<POLICY> &ctx, const JsonoTrieNode &node,
                                                 idx_t fixed_edge_index, const JsonoView &view,
                                                 const JsonoCursor &value_cursor, idx_t row, std::false_type);

template <class POLICY>
JSONO_ALWAYS_INLINE bool JsonoTrieApplyArrayTail(JsonoTrieWalkContext<POLICY> &ctx, const JsonoTrieNode &node,
                                                 idx_t fixed_edge_index, const JsonoView &view,
                                                 const JsonoCursor &value_cursor, idx_t row, std::true_type);

template <class POLICY>
void JsonoTrieApplyArrayNode(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                             const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.nodes[node_index];
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos++;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;

	while (value_cursor.pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			JsonoTrieNullSubtree<POLICY>(ctx.nodes, ctx.state, node.index_edges[fixed_edge_index].child, view, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			JsonoTrieApplyNode<POLICY>(ctx, node.index_edges[fixed_edge_index].child, view, value_cursor, row);
			fixed_edge_index++;
		}
		if (JsonoTrieApplyArrayTail<POLICY>(ctx, node, fixed_edge_index, view, value_cursor, row,
		                                    typename JsonoTrieWildcardTag<POLICY>::Type())) {
			return;
		}
		SkipValueFast(view, value_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		JsonoTrieNullSubtree<POLICY>(ctx.nodes, ctx.state, node.index_edges[fixed_edge_index].child, view, row);
		fixed_edge_index++;
	}
}

template <class POLICY>
JSONO_ALWAYS_INLINE bool JsonoTrieApplyArrayTail(JsonoTrieWalkContext<POLICY> &ctx, const JsonoTrieNode &node,
                                                 idx_t fixed_edge_index, const JsonoView &view,
                                                 const JsonoCursor &value_cursor, idx_t row, std::false_type) {
	(void)ctx;
	(void)view;
	(void)value_cursor;
	(void)row;
	return fixed_edge_index >= node.index_edges.size();
}

template <class POLICY>
JSONO_ALWAYS_INLINE bool JsonoTrieApplyArrayTail(JsonoTrieWalkContext<POLICY> &ctx, const JsonoTrieNode &node,
                                                 idx_t fixed_edge_index, const JsonoView &view,
                                                 const JsonoCursor &value_cursor, idx_t row, std::true_type) {
	if (fixed_edge_index >= node.index_edges.size() && node.wildcard_child == DConstants::INVALID_INDEX) {
		return true;
	}
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		JsonoTrieApplyWildcardElement<POLICY>(ctx, node.wildcard_child, view, value_cursor, row);
	}
	return false;
}

template <class POLICY>
void JsonoTrieApplyWildcardArrayNode(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                                     const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.nodes[node_index];
	for (auto &edge : node.key_edges) {
		JsonoTrieAppendWildcardMissing<POLICY>(ctx.nodes, ctx.state, edge.child, view, row);
	}

	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos++;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;
	while (value_cursor.pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			JsonoTrieAppendWildcardMissing<POLICY>(ctx.nodes, ctx.state, node.index_edges[fixed_edge_index].child, view,
			                                       row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			JsonoTrieApplyWildcardElement<POLICY>(ctx, node.index_edges[fixed_edge_index].child, view, value_cursor,
			                                      row);
			fixed_edge_index++;
		}
		if (fixed_edge_index >= node.index_edges.size() && node.wildcard_child == DConstants::INVALID_INDEX) {
			return;
		}
		SkipValueFast(view, value_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		JsonoTrieAppendWildcardMissing<POLICY>(ctx.nodes, ctx.state, node.index_edges[fixed_edge_index].child, view,
		                                       row);
		fixed_edge_index++;
	}
}

template <class POLICY>
void JsonoTrieApplyWildcardElement(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                                   const JsonoCursor &cursor, idx_t row) {
	POLICY::MaterializeWildcardLeaves(ctx.state, node_index, view, cursor, row);
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		JsonoTrieWalkObject<POLICY, JsonoTrieWildcardEdgePolicy<POLICY>>(ctx, node_index, view, cursor, row);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		JsonoTrieApplyWildcardArrayNode<POLICY>(ctx, node_index, view, cursor, row);
		return;
	}
	auto &node = ctx.nodes[node_index];
	for (auto &edge : node.key_edges) {
		JsonoTrieAppendWildcardMissing<POLICY>(ctx.nodes, ctx.state, edge.child, view, row);
	}
	for (auto &edge : node.index_edges) {
		JsonoTrieAppendWildcardMissing<POLICY>(ctx.nodes, ctx.state, edge.child, view, row);
	}
}

template <class POLICY>
JSONO_ALWAYS_INLINE bool JsonoTrieHasArrayWork(const JsonoTrieNode &node, std::false_type) {
	return !node.index_edges.empty();
}

template <class POLICY>
JSONO_ALWAYS_INLINE bool JsonoTrieHasArrayWork(const JsonoTrieNode &node, std::true_type) {
	return !node.index_edges.empty() || node.wildcard_child != DConstants::INVALID_INDEX;
}

template <class POLICY>
void JsonoTrieApplyNode(JsonoTrieWalkContext<POLICY> &ctx, idx_t node_index, const JsonoView &view,
                        const JsonoCursor &cursor, idx_t row) {
	auto &node = ctx.nodes[node_index];
	for (auto field_index : node.scalar_leaves) {
		POLICY::WriteScalarLeaf(ctx.state, field_index, view, cursor, row);
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (!node.key_edges.empty()) {
		if (slot_tag == tag::OBJ_START) {
			JsonoTrieWalkObject<POLICY, JsonoTrieNormalEdgePolicy<POLICY>>(ctx, node_index, view, cursor, row);
		} else {
			for (auto &edge : node.key_edges) {
				JsonoTrieNullSubtree<POLICY>(ctx.nodes, ctx.state, edge.child, view, row);
			}
		}
	}
	if (JsonoTrieHasArrayWork<POLICY>(node, typename JsonoTrieWildcardTag<POLICY>::Type())) {
		if (slot_tag == tag::ARR_START) {
			JsonoTrieApplyArrayNode<POLICY>(ctx, node_index, view, cursor, row);
		} else {
			for (auto &edge : node.index_edges) {
				JsonoTrieNullSubtree<POLICY>(ctx.nodes, ctx.state, edge.child, view, row);
			}
			JsonoTrieNullWildcardSubtree<POLICY>(ctx.nodes, ctx.state, node, view, row,
			                                     typename JsonoTrieWildcardTag<POLICY>::Type());
		}
	}
}

} // namespace jsono
} // namespace duckdb
