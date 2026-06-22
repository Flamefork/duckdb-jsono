#pragma once

#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_trie.hpp"
#include "jsono_trie_walk.hpp"

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

// Build the projector's trie over a field set. `FIELD` exposes `JsonoPathSpec path`. The projector
// admits only scalar leaves over key/index edges — wildcards are rejected upstream by
// TryReadExtractPath, so a wildcard step here is an internal invariant break.
template <class FIELD>
void BuildProjectTrie(JsonoTrie &trie, const vector<FIELD> &fields) {
	trie.Reset();
	trie.PrepareFieldMetadata(fields.size());
	for (idx_t field_index = 0; field_index < fields.size(); field_index++) {
		for (auto &step : fields[field_index].path.steps) {
			if (step.kind == PathStepKind::Wildcard) {
				throw InternalException("jsono projector trie: wildcard step is not projectable");
			}
		}
		trie.nodes[trie.WalkFieldToLeaf(field_index, fields[field_index].path.steps)].scalar_leaves.push_back(
		    field_index);
	}
	trie.SortEdges();
	for (auto &node : trie.nodes) {
		if (node.scalar_leaves.size() == 1 && node.key_edges.empty() && node.index_edges.empty() &&
		    node.wildcard_child == DConstants::INVALID_INDEX) {
			node.simple_scalar_leaf = node.scalar_leaves[0];
		}
	}
}

// The projector's per-node rank cache is the shared JsonoTrieRankCache (src/include/jsono_trie.hpp),
// keyed by this trie's node vector. It carries its own copy (not transform's) only because the two
// engines run from separate local states; the cache machinery itself is one definition.

template <class PROJECT_POLICY>
struct ProjectTrieWalkPolicy {
	using State = typename PROJECT_POLICY::State;
	static constexpr bool supports_wildcard = false;

	static JSONO_ALWAYS_INLINE void WriteScalarLeaf(State &state, idx_t field_index, const JsonoView &view,
	                                                const JsonoCursor &cursor, idx_t row) {
		PROJECT_POLICY::WriteLeaf(state, field_index, view, cursor, row);
	}

	static JSONO_ALWAYS_INLINE void NullScalarLeaf(State &state, idx_t field_index, const JsonoView &view, idx_t row) {
		PROJECT_POLICY::NullLeaf(state, field_index, view, row);
	}

	static JSONO_ALWAYS_INLINE void NullNodeLeaves(State &state, const JsonoTrieNode &node, const JsonoView &view,
	                                               idx_t row) {
		for (auto field_index : node.scalar_leaves) {
			PROJECT_POLICY::NullLeaf(state, field_index, view, row);
		}
	}
};

// Walk the trie once over one document root, dispatching every field to POLICY. The root is always
// the whole value at the JsonoCursor() origin.
template <class POLICY>
JSONO_ALWAYS_INLINE void WalkProjectTrie(const JsonoTrie &trie, JsonoTrieRankCache &rank_cache,
                                         typename POLICY::State &state, const JsonoView &view, idx_t row) {
	using WalkPolicy = ProjectTrieWalkPolicy<POLICY>;
	JsonoTrieWalkContext<WalkPolicy> ctx {trie.nodes, rank_cache, state};
	JsonoTrieApplyNode<WalkPolicy>(ctx, 0, view, JsonoCursor(), row);
}

} // namespace jsono
} // namespace duckdb
