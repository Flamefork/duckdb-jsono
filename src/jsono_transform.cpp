#include "jsono_transform.hpp"
#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_shred.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace duckdb {
namespace {

using namespace jsono;

enum class TransformPrimitive : uint8_t { Bigint, Ubigint, Double, Varchar, Boolean };
enum class TransformMode : uint8_t { Scalar, List, Join };

struct TransformField {
	string name;
	TransformPrimitive primitive;
	TransformMode mode = TransformMode::Scalar;
	vector<PathStep> path;
	string join_separator;
	LogicalType logical_type;
	// When the input is a shredded JSONO whose shred path equals this scalar field's path,
	// the field reads residual-first and falls back to this shred child (shred_primitive gives
	// the shred's lossless kind for synthesizing the stripped value). INVALID = not shred-backed.
	idx_t shred_child_index = DConstants::INVALID_INDEX;
	TransformPrimitive shred_primitive = TransformPrimitive::Varchar;
};

struct TrieEdge {
	TrieEdge() = default;
	TrieEdge(string key_p, idx_t child_p) : key(std::move(key_p)), child(child_p) {
	}

	string key;
	idx_t child = DConstants::INVALID_INDEX;
};

struct TrieIndexEdge {
	TrieIndexEdge() = default;
	TrieIndexEdge(idx_t index_p, idx_t child_p) : index(index_p), child(child_p) {
	}

	idx_t index = 0;
	idx_t child = DConstants::INVALID_INDEX;
};

struct TransformTrieNode {
	vector<TrieEdge> key_edges;
	vector<TrieIndexEdge> index_edges;
	idx_t wildcard_child = DConstants::INVALID_INDEX;
	vector<idx_t> scalar_leaves;
	vector<idx_t> list_leaves;
	vector<idx_t> join_leaves;
	idx_t simple_scalar_leaf = DConstants::INVALID_INDEX;
};

struct TransformBindData : public FunctionData {
	string spec;
	vector<TransformField> fields;
	vector<TransformTrieNode> trie;
	LogicalType return_type;
	// Scalar fields backed by a shredded shred (field.shred_child_index set); handled directly
	// from the shred/residual rather than the trie. Empty for a plain JSONO input.
	vector<idx_t> shred_fields;
	// Whether the shape-plan cache may run for this spec: the plan's per-row win is the
	// scalar fields it resolves to direct stream reads, so a spec with fewer than three of
	// them cannot recover the per-row key cost (hash + byte compare of slots and key_heap).
	bool shape_plan_eligible = false;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TransformBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<TransformBindData>();
		return spec == other.spec;
	}
};

struct Location {
	Location() = default;
	Location(const jsono::JsonoCursor &cursor_p, bool found_p) : cursor(cursor_p), found(found_p) {
	}

	jsono::JsonoCursor cursor;
	bool found = false;
};

struct TransformRankCacheEntry {
	bool valid = false;
	size_t key_count = 0;
	uint64_t shape_hash = 0;
	bool has_shape_hash = false;
	vector<size_t> ranks;
	vector<uint8_t> found;
};

// Shape-plan cache: V4 slots are shape-constant, so rows of one shape have byte-identical
// slots blobs. The plan resolves every spec field once per shape; matched rows skip the
// trie walk entirely — per row only a prefix-sum over the lengths stream (string leaf
// offsets) plus direct stream reads remain. Wildcard subtrees keep the trie walk (their
// output is per-element data), entered at a plan-recorded cursor.
//
// The cache key is the slots bytes PLUS the key_heap bytes: a KEY slot stores only its
// heap offset and length, so two rows whose keys differ in spelling but not in length
// have byte-identical slots — the key bytes that decided every locate verdict live in
// key_heap, which is itself byte-identical across rows of one shape.

// A scalar field resolved by the plan. `kind` (and a Bool's value) is fixed by the slot
// word at the located position, which the memcmp guarantees; only the stream payloads
// vary per row. String/NumberText read lengths[length_index] with the byte offset from
// the per-row prefix sum; numeric kinds read nums[num_index].
struct ShapePlanScalar {
	idx_t field_index = DConstants::INVALID_INDEX;
	JsonoScalarKind kind = JsonoScalarKind::Null;
	bool bool_value = false;
	size_t length_index = 0;
	size_t num_index = 0;
	idx_t offset_slot = 0;
};

// A wildcard-prefix subtree: per row the trie walk runs from `node_index` with the cursor
// rebuilt from these shape-constant positions (string_cursor comes from the prefix sum).
struct ShapePlanWalk {
	idx_t node_index;
	size_t pos;
	size_t length_cursor;
	size_t num_cursor;
	idx_t offset_slot;
};

struct ShapePlan {
	vector<idx_t> null_fields;
	vector<idx_t> null_subtree_nodes;
	vector<ShapePlanScalar> scalars;
	vector<ShapePlanWalk> walks;
	// Indices into bind_data.shred_fields whose path is absent from the residual: the value
	// lives in the shred column.
	vector<idx_t> shred_reads;
	// Ascending unique lengths-stream indices whose byte offsets the row scan must produce.
	vector<size_t> needed_length_indices;
	size_t lengths_bound = 0;
	size_t nums_bound = 0;
};

constexpr idx_t SHAPE_PLAN_BUCKETS = 256;
constexpr idx_t SHAPE_PLAN_WAYS = 4;
// The per-row key cost (hash + byte compare of slots and key_heap) is O(row bytes), while
// the trie walk it replaces is O(spec) — sorted keys, rank caches and checkpoints keep the
// walk from ever scanning the whole row. Wide production rows (field_sample averages ~5.8KB
// of key bytes) therefore lose to the key cost even at a 93% hit rate; the plan only pays
// on narrow rows, where the key is a fraction of the walk. Rows past this bound take the
// walk path untouched.
constexpr size_t SHAPE_PLAN_MAX_SHAPE_BYTES = 2048;
constexpr idx_t SHAPE_PLAN_MIN_SCALAR_FIELDS = 3;
// Hit-rate window: heterogeneous data where shapes outnumber the cache pays hash+memcmp+
// plan rebuilds for nothing, so the cache disables itself when a window's hit rate drops
// below 3/4. The first windows are cold-cache warmup and don't count.
constexpr uint64_t SHAPE_PLAN_WINDOW = 1024;
constexpr uint64_t SHAPE_PLAN_WARMUP = SHAPE_PLAN_BUCKETS * SHAPE_PLAN_WAYS * 4;

// Full-blob shape fingerprint. Sampling is not enough here: production shapes differ in a
// handful of words of an otherwise identical wide layout (field_sample: 5094 shapes
// collapse to ~550 sampled fingerprints, piling hot buckets past the way count). Four
// independent lanes keep the multiply mixing throughput-bound rather than latency-bound.
inline uint64_t BulkShapeHash(uint64_t h, const char *data, size_t size) {
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

struct ShapePlanCache {
	struct Entry {
		uint64_t sample_hash = 0;
		uint32_t stamp = 0;
		string slots;
		string key_heap;
		unique_ptr<ShapePlan> plan;
	};

	~ShapePlanCache() {
		static const bool stats = []() {
			const char *env = std::getenv("JSONO_SHAPE_PLAN_STATS");
			return env && env[0] == '1';
		}();
		if (stats) {
			std::fprintf(stderr, "jsono shape-plan: lookups=%llu hits=%llu builds=%llu disabled=%d\n",
			             (unsigned long long)(total_lookups + window_lookups),
			             (unsigned long long)(total_hits + window_hits), (unsigned long long)builds, int(disabled));
		}
	}

	vector<Entry> entries;
	uint32_t clock = 0;
	bool disabled = false;
	uint64_t window_lookups = 0;
	uint64_t window_hits = 0;
	uint64_t total_lookups = 0;
	uint64_t total_hits = 0;
	uint64_t builds = 0;

	// The entry for this (slots, key_heap) byte pair, or null on a miss (the caller builds
	// and Inserts). Entries allocate on first use: short inputs never pay for the table.
	ShapePlan *Find(uint64_t hash, const JsonoBlobRow &blob) {
		if (entries.empty()) {
			entries.resize(SHAPE_PLAN_BUCKETS * SHAPE_PLAN_WAYS);
		}
		window_lookups++;
		auto base = idx_t(hash & (SHAPE_PLAN_BUCKETS - 1)) * SHAPE_PLAN_WAYS;
		for (idx_t way = 0; way < SHAPE_PLAN_WAYS; way++) {
			auto &entry = entries[base + way];
			if (entry.plan && entry.sample_hash == hash && entry.slots.size() == blob.slots.GetSize() &&
			    entry.key_heap.size() == blob.key_heap.GetSize() &&
			    std::memcmp(entry.slots.data(), blob.slots.GetData(), entry.slots.size()) == 0 &&
			    std::memcmp(entry.key_heap.data(), blob.key_heap.GetData(), entry.key_heap.size()) == 0) {
				entry.stamp = ++clock;
				window_hits++;
				return entry.plan.get();
			}
		}
		return nullptr;
	}

	ShapePlan *Insert(uint64_t hash, const JsonoBlobRow &blob, unique_ptr<ShapePlan> plan) {
		builds++;
		auto base = idx_t(hash & (SHAPE_PLAN_BUCKETS - 1)) * SHAPE_PLAN_WAYS;
		auto victim = base;
		for (idx_t way = 0; way < SHAPE_PLAN_WAYS; way++) {
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
		if (window_lookups < SHAPE_PLAN_WINDOW) {
			return;
		}
		total_lookups += window_lookups;
		total_hits += window_hits;
		if (total_lookups > SHAPE_PLAN_WARMUP && window_hits * 4 < window_lookups * 3) {
			disabled = true;
			entries.clear();
			entries.shrink_to_fit();
		}
		window_lookups = 0;
		window_hits = 0;
	}
};

// Rows interleave several object shapes per node (event archetypes), so each node keeps a
// small set-associative cache: one way per recently seen shape class.
constexpr idx_t TRANSFORM_RANK_WAYS = 8;

inline idx_t TransformRankWay(const ObjectLayout &layout) {
	return idx_t((uint64_t(layout.key_count) * 0x9E3779B97F4A7C15ULL ^ layout.shape_hash) >> 61);
}

// Rank-cache slots are per trie node (the trie is bind-time constant), with ranks/found
// pre-sized to the node's key edges. No aliasing across nodes and no resizing means a held
// entry reference survives the recursive descent: recursion only touches descendant nodes.
struct TransformLocalState : public FunctionLocalState {
	explicit TransformLocalState(const TransformBindData &bind_data)
	    : rank_cache(bind_data.trie.size() * TRANSFORM_RANK_WAYS) {
		for (idx_t node_index = 0; node_index < bind_data.trie.size(); node_index++) {
			auto edge_count = bind_data.trie[node_index].key_edges.size();
			for (idx_t way = 0; way < TRANSFORM_RANK_WAYS; way++) {
				auto &entry = rank_cache[node_index * TRANSFORM_RANK_WAYS + way];
				entry.ranks.resize(edge_count);
				entry.found.resize(edge_count);
			}
		}
	}

	vector<TransformRankCacheEntry> rank_cache;
	ShapePlanCache shape_plans;
	// Per-row scratch for the plan apply: byte offsets for needed_length_indices.
	vector<uint64_t> shape_offsets;
	// Plan-build scratch: trie node chain of one field's path.
	vector<idx_t> chain_scratch;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		return make_uniq<TransformLocalState>(bind_data->Cast<TransformBindData>());
	}
};

TransformPrimitive ParsePrimitiveType(const string &type) {
	auto upper = StringUtil::Upper(type);
	if (upper == "BIGINT") {
		return TransformPrimitive::Bigint;
	}
	if (upper == "UBIGINT") {
		return TransformPrimitive::Ubigint;
	}
	if (upper == "DOUBLE") {
		return TransformPrimitive::Double;
	}
	if (upper == "VARCHAR") {
		return TransformPrimitive::Varchar;
	}
	if (upper == "BOOLEAN") {
		return TransformPrimitive::Boolean;
	}
	throw BinderException("jsono_transform: unsupported scalar type '%s'", type);
}

// Map a shredded shred's column type to the transform primitive that names its lossless
// kind. A JSON-aliased VARCHAR is not a shred (it holds a document, not a scalar), so it
// is rejected — matching the shred writer, which never lifts JSON fields into shreds.
bool ShredTypeToPrimitive(const LogicalType &type, TransformPrimitive &out) {
	switch (type.id()) {
	case LogicalTypeId::VARCHAR:
		if (type.HasAlias() && type.GetAlias() == LogicalType::JSON_TYPE_NAME) {
			return false;
		}
		out = TransformPrimitive::Varchar;
		return true;
	case LogicalTypeId::BIGINT:
		out = TransformPrimitive::Bigint;
		return true;
	case LogicalTypeId::UBIGINT:
		out = TransformPrimitive::Ubigint;
		return true;
	case LogicalTypeId::DOUBLE:
		out = TransformPrimitive::Double;
		return true;
	case LogicalTypeId::BOOLEAN:
		out = TransformPrimitive::Boolean;
		return true;
	default:
		return false;
	}
}

LogicalType PrimitiveLogicalType(TransformPrimitive primitive) {
	switch (primitive) {
	case TransformPrimitive::Bigint:
		return LogicalType::BIGINT;
	case TransformPrimitive::Ubigint:
		return LogicalType::UBIGINT;
	case TransformPrimitive::Double:
		return LogicalType::DOUBLE;
	case TransformPrimitive::Varchar:
		return LogicalType::VARCHAR;
	case TransformPrimitive::Boolean:
		return LogicalType::BOOLEAN;
	}
	throw InternalException("jsono_transform: unhandled primitive type");
}

idx_t FindWildcardIndex(const vector<PathStep> &path) {
	for (idx_t i = 0; i < path.size(); i++) {
		if (path[i].kind == PathStepKind::Wildcard) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

TransformField MakeField(nonstd::string_view name, TransformPrimitive primitive, TransformMode mode,
                         vector<PathStep> path, string join_separator) {
	TransformField field;
	field.name = string(name);
	field.primitive = primitive;
	field.mode = mode;
	field.path = std::move(path);
	field.join_separator = std::move(join_separator);
	field.logical_type =
	    mode == TransformMode::List ? LogicalType::LIST(LogicalType::VARCHAR) : PrimitiveLogicalType(primitive);
	auto wildcard_index = FindWildcardIndex(field.path);
	if (wildcard_index != DConstants::INVALID_INDEX && mode == TransformMode::Scalar) {
		throw BinderException("jsono_transform: wildcard path requires list type or join_separator");
	}
	if (wildcard_index == DConstants::INVALID_INDEX && mode != TransformMode::Scalar) {
		throw BinderException("jsono_transform: list and join modes require a wildcard path");
	}
	return field;
}

// A bare shorthand/wrapper key names a literal top-level object key, not a JSONPath
// expression: dots in the key (e.g. analytics "utm.source") must not be read as nesting.
vector<PathStep> LiteralKeyPath(nonstd::string_view name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, string(name), 0});
	return path;
}

TransformField ParseScalarShorthand(nonstd::string_view name, nonstd::string_view type_name) {
	return MakeField(name, ParsePrimitiveType(string(type_name)), TransformMode::Scalar, LiteralKeyPath(name),
	                 string());
}

// A wrapper field is a nested STRUCT {type: ..., path: '...', join_separator: '...'}: `type`
// is a type string (scalar), or a single-element list `['VARCHAR']` (list mode); `path` and
// `join_separator` are strings. Mirrors the scalar-vs-struct dispatch the JSON form had.
TransformField ParseWrapperField(nonstd::string_view name, const Value &wrapper) {
	bool has_type = false;
	bool has_join_separator = false;
	TransformPrimitive primitive = TransformPrimitive::Varchar;
	TransformMode mode = TransformMode::Scalar;
	bool has_path = false;
	string path;
	string join_separator;

	auto &wrapper_types = StructType::GetChildTypes(wrapper.type());
	auto &wrapper_values = StructValue::GetChildren(wrapper);
	for (idx_t i = 0; i < wrapper_types.size(); i++) {
		auto &key = wrapper_types[i].first;
		auto value = wrapper_values[i];
		if (key == "type") {
			has_type = true;
			if (value.type().id() == LogicalTypeId::LIST) {
				auto &items = ListValue::GetChildren(value);
				auto item = items.size() == 1 ? items[0] : Value();
				if (items.size() != 1 || item.IsNull() || !item.DefaultTryCastAs(LogicalType::VARCHAR) ||
				    StringUtil::Upper(StringValue::Get(item)) != "VARCHAR") {
					throw BinderException("jsono_transform: unsupported list item type");
				}
				primitive = TransformPrimitive::Varchar;
				mode = TransformMode::List;
				continue;
			}
			if (value.IsNull() || !value.DefaultTryCastAs(LogicalType::VARCHAR)) {
				throw BinderException("jsono_transform: type must be a string or single-item array");
			}
			primitive = ParsePrimitiveType(StringValue::Get(value));
			continue;
		}
		if (key == "path") {
			if (value.IsNull() || !value.DefaultTryCastAs(LogicalType::VARCHAR)) {
				throw BinderException("jsono_transform: path must be a string");
			}
			has_path = true;
			path = StringValue::Get(value);
			continue;
		}
		if (key == "join_separator") {
			if (value.IsNull() || !value.DefaultTryCastAs(LogicalType::VARCHAR)) {
				throw BinderException("jsono_transform: join_separator must be a string");
			}
			has_join_separator = true;
			join_separator = StringValue::Get(value);
			continue;
		}
		throw BinderException("jsono_transform: unknown wrapper key '%s'", key);
	}
	if (!has_type) {
		throw BinderException("jsono_transform: wrapper field must include type");
	}
	if (has_join_separator) {
		if (mode == TransformMode::List || primitive != TransformPrimitive::Varchar) {
			throw BinderException("jsono_transform: join_separator requires VARCHAR scalar type");
		}
		mode = TransformMode::Join;
	}
	auto steps = has_path ? ParseJsonoPath(path, "jsono_transform") : LiteralKeyPath(name);
	return MakeField(name, primitive, mode, std::move(steps), std::move(join_separator));
}

idx_t GetOrAddKeyChild(TransformBindData &bind_data, idx_t node_index, const string &key) {
	auto &edges = bind_data.trie[node_index].key_edges;
	for (auto &edge : edges) {
		if (edge.key == key) {
			return edge.child;
		}
	}
	auto child = bind_data.trie.size();
	bind_data.trie.emplace_back();
	bind_data.trie[node_index].key_edges.push_back(TrieEdge {key, child});
	return child;
}

idx_t GetOrAddIndexChild(TransformBindData &bind_data, idx_t node_index, idx_t index) {
	auto &edges = bind_data.trie[node_index].index_edges;
	for (auto &edge : edges) {
		if (edge.index == index) {
			return edge.child;
		}
	}
	auto child = bind_data.trie.size();
	bind_data.trie.emplace_back();
	bind_data.trie[node_index].index_edges.push_back(TrieIndexEdge {index, child});
	return child;
}

idx_t GetOrAddWildcardChild(TransformBindData &bind_data, idx_t node_index) {
	auto &node = bind_data.trie[node_index];
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		return node.wildcard_child;
	}
	auto child = bind_data.trie.size();
	bind_data.trie.emplace_back();
	bind_data.trie[node_index].wildcard_child = child;
	return child;
}

void InsertFieldIntoTrie(TransformBindData &bind_data, idx_t field_index) {
	idx_t node_index = 0;
	auto &field = bind_data.fields[field_index];
	if (field.shred_child_index != DConstants::INVALID_INDEX) {
		// Shred-backed scalar: read directly from the shred/residual, never via the trie.
		return;
	}
	for (auto &step : field.path) {
		switch (step.kind) {
		case PathStepKind::Key:
			node_index = GetOrAddKeyChild(bind_data, node_index, step.key);
			break;
		case PathStepKind::Index:
			node_index = GetOrAddIndexChild(bind_data, node_index, step.index);
			break;
		case PathStepKind::Wildcard:
			node_index = GetOrAddWildcardChild(bind_data, node_index);
			break;
		}
	}
	auto &node = bind_data.trie[node_index];
	switch (field.mode) {
	case TransformMode::Scalar:
		node.scalar_leaves.push_back(field_index);
		break;
	case TransformMode::List:
		node.list_leaves.push_back(field_index);
		break;
	case TransformMode::Join:
		node.join_leaves.push_back(field_index);
		break;
	}
}

void BuildTransformTrie(TransformBindData &bind_data) {
	bind_data.trie.clear();
	bind_data.trie.emplace_back();
	for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
		InsertFieldIntoTrie(bind_data, field_index);
	}
	for (auto &node : bind_data.trie) {
		std::sort(node.key_edges.begin(), node.key_edges.end(),
		          [](const TrieEdge &left, const TrieEdge &right) { return left.key < right.key; });
		std::sort(node.index_edges.begin(), node.index_edges.end(),
		          [](const TrieIndexEdge &left, const TrieIndexEdge &right) { return left.index < right.index; });
		if (node.scalar_leaves.size() == 1 && node.list_leaves.empty() && node.join_leaves.empty() &&
		    node.key_edges.empty() && node.index_edges.empty() && node.wildcard_child == DConstants::INVALID_INDEX) {
			node.simple_scalar_leaf = node.scalar_leaves[0];
		}
	}
}

// The spec is a constant STRUCT mapping each output field to its type: a type string
// (scalar shorthand) or a nested wrapper STRUCT {type, path, join_separator}. Field names are
// unique by struct construction. Same shape as core json_transform's structure, but expressed
// as a typed STRUCT literal rather than a JSON string.
unique_ptr<TransformBindData> ParseTransformSpec(const Value &spec) {
	if (spec.IsNull() || spec.type().id() != LogicalTypeId::STRUCT) {
		throw BinderException(
		    "jsono_transform: spec must be a non-NULL STRUCT mapping output field to type, e.g. {x: 'VARCHAR'}");
	}
	auto &child_types = StructType::GetChildTypes(spec.type());
	auto &child_values = StructValue::GetChildren(spec);

	auto bind_data = make_uniq<TransformBindData>();
	child_list_t<LogicalType> children;
	for (idx_t i = 0; i < child_types.size(); i++) {
		auto &name = child_types[i].first;
		auto value = child_values[i];
		TransformField transform_field;
		if (value.type().id() == LogicalTypeId::STRUCT) {
			transform_field = ParseWrapperField(name, value);
		} else if (!value.IsNull() && value.DefaultTryCastAs(LogicalType::VARCHAR)) {
			transform_field = ParseScalarShorthand(name, StringValue::Get(value));
		} else {
			throw BinderException("jsono_transform: field spec for '%s' must be a type string or wrapper struct", name);
		}
		children.emplace_back(transform_field.name, transform_field.logical_type);
		bind_data->fields.push_back(std::move(transform_field));
	}
	if (bind_data->fields.empty()) {
		throw BinderException("jsono_transform: empty spec");
	}
	bind_data->return_type = LogicalType::STRUCT(std::move(children));
	BuildTransformTrie(*bind_data);
	return bind_data;
}

bool PathStepsEqual(const vector<PathStep> &left, const vector<PathStep> &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (idx_t i = 0; i < left.size(); i++) {
		if (left[i].kind != right[i].kind) {
			return false;
		}
		if (left[i].kind == PathStepKind::Key && left[i].key != right[i].key) {
			return false;
		}
		if (left[i].kind == PathStepKind::Index && left[i].index != right[i].index) {
			return false;
		}
	}
	return true;
}

// Bind a shredded input's shred columns to the scalar fields they back: a scalar field whose
// path equals a shred's path reads that shred (residual-first, shred-fallback) instead of
// navigating the residual tape. The trie is rebuilt to exclude the bound fields.
void AssignShreddedShreds(TransformBindData &bind_data, const LogicalType &input_type) {
	JsonoLayoutType layout;
	TryParseJsonoLayoutType(input_type, layout);
	for (idx_t i = 0; i < layout.shreds.size(); i++) {
		TransformPrimitive shred_primitive;
		if (!ShredTypeToPrimitive(layout.shreds[i].second, shred_primitive)) {
			// A non-shred-typed shred is not a shred; any field on its path reads the
			// residual, where the value remains.
			continue;
		}
		auto &name = layout.shreds[i].first;
		auto steps = !name.empty() && name[0] == '$' ? ParseJsonoPath(name, "jsono_transform") : LiteralKeyPath(name);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			if (field.mode != TransformMode::Scalar || field.shred_child_index != DConstants::INVALID_INDEX) {
				continue;
			}
			if (PathStepsEqual(field.path, steps)) {
				field.shred_child_index = i;
				field.shred_primitive = shred_primitive;
				bind_data.shred_fields.push_back(field_index);
			}
		}
	}
	if (!bind_data.shred_fields.empty()) {
		BuildTransformTrie(bind_data);
	}
}

unique_ptr<FunctionData> JsonoTransformBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("jsono_transform: spec must be constant");
	}
	auto spec_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	auto bind_data = ParseTransformSpec(spec_value);
	// The struct's canonical string identifies the full spec (names, types, paths, joins) for
	// expression caching — return_type alone is insufficient since a field's path may differ
	// from its name.
	bind_data->spec = spec_value.ToString();
	auto &input_type = arguments[0]->return_type;
	// An array shred (object or scalar) strips values out of the residual into a list column the
	// transform's residual navigation cannot see, so a shredded value carrying one is reconstructed
	// whole (scalar shreds in it fold back through the same reconstruct). A scalar-only shred set keeps
	// the native residual-first / shred-fallback fast path via AssignShreddedShreds.
	bool reconstruct_shredded = false;
	if (IsShreddedJsonoType(input_type)) {
		JsonoLayoutType layout;
		TryParseJsonoLayoutType(input_type, layout);
		for (auto &shred : layout.shreds) {
			if (IsShredListType(shred.second)) {
				reconstruct_shredded = true;
				break;
			}
		}
	}
	bound_function.arguments[0] =
	    JsonoResolveJsonoArgument(context, *arguments[0], "jsono_transform", reconstruct_shredded);
	if (IsShreddedJsonoType(input_type) && !reconstruct_shredded) {
		AssignShreddedShreds(*bind_data, input_type);
	}
	idx_t scalar_fields = 0;
	for (auto &field : bind_data->fields) {
		if (field.mode == TransformMode::Scalar) {
			scalar_fields++;
		}
	}
	bind_data->shape_plan_eligible = scalar_fields >= SHAPE_PLAN_MIN_SCALAR_FIELDS;
	bound_function.return_type = bind_data->return_type;
	return std::move(bind_data);
}

void WriteScalarValue(const TransformField &field, const JsonoScalar &scalar, Vector &result, idx_t row) {
	switch (field.primitive) {
	case TransformPrimitive::Bigint:
		// UInt64 values exceed INT64_MAX by construction, so only Int64 fits a BIGINT target.
		if (scalar.kind == JsonoScalarKind::Int64) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<int64_t>(result)[row] = scalar.int_value;
			return;
		}
		break;
	case TransformPrimitive::Ubigint:
		if (scalar.kind == JsonoScalarKind::Int64 && scalar.int_value >= 0) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<uint64_t>(result)[row] = uint64_t(scalar.int_value);
			return;
		}
		if (scalar.kind == JsonoScalarKind::UInt64) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<uint64_t>(result)[row] = scalar.uint_value;
			return;
		}
		break;
	case TransformPrimitive::Double: {
		double value = 0;
		bool ok = true;
		switch (scalar.kind) {
		case JsonoScalarKind::Double:
			value = scalar.double_value;
			break;
		case JsonoScalarKind::Dec60:
			value = Dec60ToDouble(scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			break;
		case JsonoScalarKind::Int64:
			value = double(scalar.int_value);
			break;
		case JsonoScalarKind::UInt64:
			value = double(scalar.uint_value);
			break;
		case JsonoScalarKind::NumberText:
			// Locale-independent parse via DuckDB's fast_float-backed cast (no allocation).
			ok =
			    TryCast::Operation<string_t, double>(string_t(scalar.text.data(), uint32_t(scalar.text.size())), value);
			break;
		default:
			ok = false;
			break;
		}
		if (ok) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<double>(result)[row] = value;
			return;
		}
		break;
	}
	case TransformPrimitive::Varchar:
		if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<string_t>(result)[row] =
			    StringVector::AddString(result, scalar.text.data(), scalar.text.size());
			return;
		}
		break;
	case TransformPrimitive::Boolean:
		if (scalar.kind == JsonoScalarKind::Bool) {
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<bool>(result)[row] = scalar.bool_value;
			return;
		}
		break;
	}
	FlatVector::SetNull(result, row, true);
}

void WriteScalarField(const TransformField &field, const JsonoView &view, const Location &location, Vector &result,
                      idx_t row) {
	if (!location.found) {
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto value_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		// A container where a scalar was requested is a type mismatch, not malformed input.
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	WriteScalarValue(field, scalar, result, row);
}

void EnsureListCapacity(Vector &result, idx_t needed) {
	if (needed <= ListVector::GetListCapacity(result)) {
		return;
	}
	ListVector::Reserve(result, std::max<idx_t>(needed, std::max<idx_t>(ListVector::GetListCapacity(result) * 2, 1)));
}

void AppendListStringValue(Vector &result, idx_t child_row, nonstd::string_view value) {
	auto &child = ListVector::GetEntry(result);
	FlatVector::Validity(child).SetValid(child_row);
	FlatVector::GetData<string_t>(child)[child_row] = StringVector::AddString(child, value.data(), value.size());
}

void AppendListNullValue(Vector &result, idx_t child_row) {
	auto &child = ListVector::GetEntry(result);
	FlatVector::SetNull(child, child_row, true);
}

void ApplyTrieNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                   const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                   vector<string> &join_buffers, vector<idx_t> &join_counts);

void ApplyWildcardElement(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                          const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children,
                          idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts);

int CompareTrieKeyToJsonKey(const string &trie_key, nonstd::string_view json_key) {
	if (!trie_key.empty() && !json_key.empty()) {
		auto trie_first = static_cast<unsigned char>(trie_key[0]);
		auto json_first = static_cast<unsigned char>(json_key[0]);
		if (trie_first != json_first) {
			return trie_first < json_first ? -1 : 1;
		}
	}
	return nonstd::string_view(trie_key).compare(json_key);
}

// Walk Key/Index steps to a leaf in the residual tape. Used for shred-backed scalar fields:
// the residual is authoritative, so a found leaf wins; only when the residual lacks the path
// does the caller fall back to the shred. Mirrors the shred writer's path locator.
Location LocatePath(const JsonoView &view, const vector<PathStep> &steps) {
	JsonoCursor cursor;
	if (!LocatePathSteps(nullptr, 0, steps, view, cursor)) {
		return {};
	}
	return Location {cursor, true};
}

// Build the lossless scalar a shred stands for. The shred is read only when the residual lacks
// the path, which (by the shred contract) means the value was stripped, so its original kind
// is the shred's lossless kind: a VARCHAR shred held a JSON string, a typed shred its exact kind.
JsonoScalar SynthShredScalar(TransformPrimitive shred_primitive, const UnifiedVectorFormat &fmt, idx_t idx) {
	JsonoScalar scalar;
	switch (shred_primitive) {
	case TransformPrimitive::Varchar: {
		// Reference the string_t in vector storage, not a stack copy: an inline (short) string_t
		// stores its bytes inside itself, so a copy would leave scalar.text dangling after return.
		auto &value = UnifiedVectorFormat::GetData<string_t>(fmt)[idx];
		scalar.kind = JsonoScalarKind::String;
		scalar.text = nonstd::string_view(value.GetData(), value.GetSize());
		return scalar;
	}
	case TransformPrimitive::Bigint:
		scalar.kind = JsonoScalarKind::Int64;
		scalar.int_value = UnifiedVectorFormat::GetData<int64_t>(fmt)[idx];
		return scalar;
	case TransformPrimitive::Ubigint:
		scalar.kind = JsonoScalarKind::UInt64;
		scalar.uint_value = UnifiedVectorFormat::GetData<uint64_t>(fmt)[idx];
		return scalar;
	case TransformPrimitive::Double:
		scalar.kind = JsonoScalarKind::Double;
		scalar.double_value = UnifiedVectorFormat::GetData<double>(fmt)[idx];
		return scalar;
	case TransformPrimitive::Boolean:
		scalar.kind = JsonoScalarKind::Bool;
		scalar.bool_value = UnifiedVectorFormat::GetData<bool>(fmt)[idx];
		return scalar;
	}
	return scalar;
}

const TransformRankCacheEntry &GetObjectRankCacheEntry(TransformLocalState &lstate, const TransformBindData &bind_data,
                                                       idx_t node_index, const JsonoView &view,
                                                       const ObjectLayout &layout) {
	auto &node = bind_data.trie[node_index];
	auto &entry = lstate.rank_cache[node_index * TRANSFORM_RANK_WAYS + TransformRankWay(layout)];
	auto cache_valid = entry.valid && entry.key_count == layout.key_count;
	if (cache_valid) {
		// Same policy as RankCache::Find: a matching stored shape_hash proves the object's
		// sorted key sequence, validating every edge's cached rank with one int compare.
		// The slot is per node, so the (bind-time constant) key set needs no re-check.
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

void SetTrieNodeOutputsNull(const TransformBindData &bind_data, idx_t node_index, vector<unique_ptr<Vector>> &children,
                            idx_t row) {
	auto &node = bind_data.trie[node_index];
	for (auto field_index : node.scalar_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
	for (auto field_index : node.list_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
		ListVector::GetData(*children[field_index])[row] = {0, 0};
	}
	for (auto field_index : node.join_leaves) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
	for (auto &edge : node.key_edges) {
		SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
	}
	for (auto &edge : node.index_edges) {
		SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
	}
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		SetTrieNodeOutputsNull(bind_data, node.wildcard_child, children, row);
	}
}

void AppendListElementNull(Vector &result, idx_t row) {
	auto &entry = ListVector::GetData(result)[row];
	auto child_row = entry.offset + entry.length;
	EnsureListCapacity(result, child_row + 1);
	AppendListNullValue(result, child_row);
	entry.length++;
	ListVector::SetListSize(result, entry.offset + entry.length);
}

void AppendListElementString(Vector &result, idx_t row, nonstd::string_view value) {
	auto &entry = ListVector::GetData(result)[row];
	auto child_row = entry.offset + entry.length;
	EnsureListCapacity(result, child_row + 1);
	AppendListStringValue(result, child_row, value);
	entry.length++;
	ListVector::SetListSize(result, entry.offset + entry.length);
}

void AppendWildcardMissing(const TransformBindData &bind_data, idx_t node_index, vector<unique_ptr<Vector>> &children,
                           idx_t row) {
	auto &node = bind_data.trie[node_index];
	for (auto field_index : node.list_leaves) {
		AppendListElementNull(*children[field_index], row);
	}
	for (auto &edge : node.key_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
	if (node.wildcard_child != DConstants::INVALID_INDEX) {
		AppendWildcardMissing(bind_data, node.wildcard_child, children, row);
	}
}

void ApplyObjectNodeWithCheckpoints(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                                    const JsonoView &view, const ObjectLayout &layout, const JsonoCursor &obj_cursor,
                                    vector<unique_ptr<Vector>> &children, idx_t row, vector<string> &join_buffers,
                                    vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	size_t value_rank = 0;
	JsonoCursor value_block_base = obj_cursor;
	value_block_base.pos = layout.value_start;
	JsonoCursor value_cursor = value_block_base;

	// The reference stays valid across the recursive ApplyTrieNode below: descendants own
	// their own per-node slots and the entry's vectors are pre-sized, never resized.
	auto &entry = GetObjectRankCacheEntry(lstate, bind_data, node_index, view, layout);

	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		auto &edge = node.key_edges[edge_index];
		auto simple_scalar_leaf = bind_data.trie[edge.child].simple_scalar_leaf;
		if (!entry.found[edge_index]) {
			if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
				SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
			} else {
				FlatVector::SetNull(*children[simple_scalar_leaf], row, true);
			}
			continue;
		}
		auto rank = entry.ranks[edge_index];
		MoveObjectValueCursorToRank(view, layout, value_block_base, rank, value_rank, value_cursor);
		if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
			ApplyTrieNode(lstate, bind_data, edge.child, view, value_cursor, children, row, join_buffers, join_counts);
		} else {
			WriteScalarField(bind_data.fields[simple_scalar_leaf], view, Location {value_cursor, true},
			                 *children[simple_scalar_leaf], row);
		}
	}
}

void ApplyWildcardObjectNodeWithCheckpoints(TransformLocalState &lstate, const TransformBindData &bind_data,
                                            idx_t node_index, const JsonoView &view, const ObjectLayout &layout,
                                            const JsonoCursor &obj_cursor, vector<unique_ptr<Vector>> &children,
                                            idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	size_t value_rank = 0;
	JsonoCursor value_block_base = obj_cursor;
	value_block_base.pos = layout.value_start;
	JsonoCursor value_cursor = value_block_base;

	// Holding the reference across recursion is safe (see ApplyObjectNodeWithCheckpoints).
	auto &entry = GetObjectRankCacheEntry(lstate, bind_data, node_index, view, layout);

	for (idx_t edge_index = 0; edge_index < node.key_edges.size(); edge_index++) {
		auto &edge = node.key_edges[edge_index];
		if (!entry.found[edge_index]) {
			AppendWildcardMissing(bind_data, edge.child, children, row);
			continue;
		}
		auto rank = entry.ranks[edge_index];
		MoveObjectValueCursorToRank(view, layout, value_block_base, rank, value_rank, value_cursor);
		ApplyWildcardElement(lstate, bind_data, edge.child, view, value_cursor, children, row, join_buffers,
		                     join_counts);
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
}

// Materialize one wildcard element for this node's list/join leaves. Every scalar kind
// carries its `->>` text (the render shared with VARCHAR shreds); JSON null and containers
// stay a positional NULL in list mode and are skipped by join mode.
void MaterializeWildcardLeaves(const TransformBindData &bind_data, idx_t node_index, const JsonoView &view,
                               const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                               vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag != tag::OBJ_START && slot_tag != tag::ARR_START) {
		auto value_cursor = cursor;
		auto scalar = DecodeScalarAt(view, value_cursor);
		if (scalar.kind != JsonoScalarKind::Null) {
			// Text scalars render straight from the heap; scratch holds the others.
			nonstd::string_view value;
			std::string scratch;
			if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
				value = scalar.text;
			} else {
				RenderExtractText(scalar, scratch);
				value = nonstd::string_view(scratch.data(), scratch.size());
			}
			for (auto field_index : node.list_leaves) {
				AppendListElementString(*children[field_index], row, value);
			}
			for (auto field_index : node.join_leaves) {
				auto &buffer = join_buffers[field_index];
				if (join_counts[field_index] > 0) {
					buffer.append(bind_data.fields[field_index].join_separator);
				}
				buffer.append(value.data(), value.size());
				join_counts[field_index]++;
			}
			return;
		}
	}
	for (auto field_index : node.list_leaves) {
		AppendListElementNull(*children[field_index], row);
	}
}

void ApplyObjectNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                     const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                     vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto layout = ReadObjectLayout(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos = layout.value_start;
	idx_t edge_index = 0;
	bool use_checkpoints =
	    layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && layout.key_count > OBJECT_CHECKPOINT_STRIDE;
	if (use_checkpoints) {
		ApplyObjectNodeWithCheckpoints(lstate, bind_data, node_index, view, layout, cursor, children, row, join_buffers,
		                               join_counts);
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
			auto simple_scalar_leaf = bind_data.trie[node.key_edges[edge_index].child].simple_scalar_leaf;
			if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
				SetTrieNodeOutputsNull(bind_data, node.key_edges[edge_index].child, children, row);
			} else {
				FlatVector::SetNull(*children[simple_scalar_leaf], row, true);
			}
			edge_index++;
		}
		if (edge_index < node.key_edges.size() && key_compare == 0) {
			auto simple_scalar_leaf = bind_data.trie[node.key_edges[edge_index].child].simple_scalar_leaf;
			if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
				ApplyTrieNode(lstate, bind_data, node.key_edges[edge_index].child, view, value_cursor, children, row,
				              join_buffers, join_counts);
			} else {
				WriteScalarField(bind_data.fields[simple_scalar_leaf], view, Location {value_cursor, true},
				                 *children[simple_scalar_leaf], row);
			}
			edge_index++;
		}
		if (edge_index >= node.key_edges.size()) {
			return;
		}
		SkipValueFast(view, value_cursor);
	}
	while (edge_index < node.key_edges.size()) {
		auto simple_scalar_leaf = bind_data.trie[node.key_edges[edge_index].child].simple_scalar_leaf;
		if (simple_scalar_leaf == DConstants::INVALID_INDEX) {
			SetTrieNodeOutputsNull(bind_data, node.key_edges[edge_index].child, children, row);
		} else {
			FlatVector::SetNull(*children[simple_scalar_leaf], row, true);
		}
		edge_index++;
	}
}

void ApplyArrayNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                    const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                    vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos++;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;

	while (value_cursor.pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			SetTrieNodeOutputsNull(bind_data, node.index_edges[fixed_edge_index].child, children, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			ApplyTrieNode(lstate, bind_data, node.index_edges[fixed_edge_index].child, view, value_cursor, children,
			              row, join_buffers, join_counts);
			fixed_edge_index++;
		}
		if (fixed_edge_index >= node.index_edges.size() && node.wildcard_child == DConstants::INVALID_INDEX) {
			return;
		}
		if (node.wildcard_child != DConstants::INVALID_INDEX) {
			ApplyWildcardElement(lstate, bind_data, node.wildcard_child, view, value_cursor, children, row,
			                     join_buffers, join_counts);
		}
		SkipValueFast(view, value_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		SetTrieNodeOutputsNull(bind_data, node.index_edges[fixed_edge_index].child, children, row);
		fixed_edge_index++;
	}
}

void ApplyWildcardObjectNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                             const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children,
                             idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	auto layout = ReadObjectLayout(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos = layout.value_start;
	idx_t edge_index = 0;
	bool use_checkpoints =
	    layout.checkpoint_offset != NO_OBJECT_CHECKPOINTS && layout.key_count > OBJECT_CHECKPOINT_STRIDE;
	if (use_checkpoints) {
		ApplyWildcardObjectNodeWithCheckpoints(lstate, bind_data, node_index, view, layout, cursor, children, row,
		                                       join_buffers, join_counts);
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
			AppendWildcardMissing(bind_data, node.key_edges[edge_index].child, children, row);
			edge_index++;
		}
		if (edge_index < node.key_edges.size() && key_compare == 0) {
			ApplyWildcardElement(lstate, bind_data, node.key_edges[edge_index].child, view, value_cursor, children, row,
			                     join_buffers, join_counts);
			edge_index++;
		}
		if (edge_index >= node.key_edges.size() && node.index_edges.empty()) {
			return;
		}
		SkipValueFast(view, value_cursor);
	}
	while (edge_index < node.key_edges.size()) {
		AppendWildcardMissing(bind_data, node.key_edges[edge_index].child, children, row);
		edge_index++;
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
}

void ApplyWildcardArrayNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                            const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children,
                            idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	for (auto &edge : node.key_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}

	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	JsonoCursor value_cursor = cursor;
	value_cursor.pos++;
	idx_t element_index = 0;
	idx_t fixed_edge_index = 0;
	while (value_cursor.pos < end_pos) {
		while (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index < element_index) {
			AppendWildcardMissing(bind_data, node.index_edges[fixed_edge_index].child, children, row);
			fixed_edge_index++;
		}
		if (fixed_edge_index < node.index_edges.size() && node.index_edges[fixed_edge_index].index == element_index) {
			ApplyWildcardElement(lstate, bind_data, node.index_edges[fixed_edge_index].child, view, value_cursor,
			                     children, row, join_buffers, join_counts);
			fixed_edge_index++;
		}
		if (fixed_edge_index >= node.index_edges.size() && node.wildcard_child == DConstants::INVALID_INDEX) {
			return;
		}
		SkipValueFast(view, value_cursor);
		element_index++;
	}
	while (fixed_edge_index < node.index_edges.size()) {
		AppendWildcardMissing(bind_data, node.index_edges[fixed_edge_index].child, children, row);
		fixed_edge_index++;
	}
}

void ApplyWildcardElement(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                          const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children,
                          idx_t row, vector<string> &join_buffers, vector<idx_t> &join_counts) {
	MaterializeWildcardLeaves(bind_data, node_index, view, cursor, children, row, join_buffers, join_counts);
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (slot_tag == tag::OBJ_START) {
		ApplyWildcardObjectNode(lstate, bind_data, node_index, view, cursor, children, row, join_buffers, join_counts);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		ApplyWildcardArrayNode(lstate, bind_data, node_index, view, cursor, children, row, join_buffers, join_counts);
		return;
	}
	auto &node = bind_data.trie[node_index];
	for (auto &edge : node.key_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
	for (auto &edge : node.index_edges) {
		AppendWildcardMissing(bind_data, edge.child, children, row);
	}
}

void ApplyTrieNode(TransformLocalState &lstate, const TransformBindData &bind_data, idx_t node_index,
                   const JsonoView &view, const JsonoCursor &cursor, vector<unique_ptr<Vector>> &children, idx_t row,
                   vector<string> &join_buffers, vector<idx_t> &join_counts) {
	auto &node = bind_data.trie[node_index];
	for (auto field_index : node.scalar_leaves) {
		WriteScalarField(bind_data.fields[field_index], view, Location {cursor, true}, *children[field_index], row);
	}
	auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
	if (!node.key_edges.empty()) {
		if (slot_tag == tag::OBJ_START) {
			ApplyObjectNode(lstate, bind_data, node_index, view, cursor, children, row, join_buffers, join_counts);
		} else {
			for (auto &edge : node.key_edges) {
				SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
			}
		}
	}
	if (!node.index_edges.empty() || node.wildcard_child != DConstants::INVALID_INDEX) {
		if (slot_tag == tag::ARR_START) {
			ApplyArrayNode(lstate, bind_data, node_index, view, cursor, children, row, join_buffers, join_counts);
		} else {
			for (auto &edge : node.index_edges) {
				SetTrieNodeOutputsNull(bind_data, edge.child, children, row);
			}
			if (node.wildcard_child != DConstants::INVALID_INDEX) {
				SetTrieNodeOutputsNull(bind_data, node.wildcard_child, children, row);
			}
		}
	}
}

void SetTransformRowNull(const vector<TransformField> &fields, Vector &result, vector<unique_ptr<Vector>> &children,
                         idx_t row) {
	FlatVector::SetNull(result, row, true);
	for (idx_t col = 0; col < fields.size(); col++) {
		FlatVector::SetNull(*children[col], row, true);
		if (fields[col].mode == TransformMode::List) {
			ListVector::GetData(*children[col])[row] = {0, 0};
		}
	}
}

// Follow the trie from the root along steps[0, upto), recording the visited nodes
// (root first). The edges exist by construction: InsertFieldIntoTrie added them.
void TrieChainForSteps(const TransformBindData &bind_data, const vector<PathStep> &steps, idx_t upto,
                       vector<idx_t> &chain) {
	idx_t node_index = 0;
	chain.clear();
	chain.push_back(node_index);
	for (idx_t i = 0; i < upto; i++) {
		auto &step = steps[i];
		auto &node = bind_data.trie[node_index];
		idx_t child = DConstants::INVALID_INDEX;
		switch (step.kind) {
		case PathStepKind::Key:
			for (auto &edge : node.key_edges) {
				if (edge.key == step.key) {
					child = edge.child;
					break;
				}
			}
			break;
		case PathStepKind::Index:
			for (auto &edge : node.index_edges) {
				if (edge.index == step.index) {
					child = edge.child;
					break;
				}
			}
			break;
		case PathStepKind::Wildcard:
			child = node.wildcard_child;
			break;
		}
		if (child == DConstants::INVALID_INDEX) {
			throw InternalException("jsono_transform: trie edge missing for shape plan");
		}
		node_index = child;
		chain.push_back(node_index);
	}
}

// Record the scalar at `cursor` (its slot word, fixed for this shape) as a direct plan
// action for `field_index`. Containers and JSON null resolve to a NULL output for every
// row of the shape, so they go to null_fields.
void PlanScalarAt(ShapePlan &plan, const JsonoView &view, const JsonoCursor &cursor, idx_t field_index) {
	auto slot = view.SlotAt(cursor.pos);
	ShapePlanScalar action;
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

// Resolve every spec field against one row of this shape. Wildcard fields resolve first:
// each distinct wildcard-prefix trie node becomes one walk (prefix found) or one
// SetTrieNodeOutputsNull (prefix missing) action, deduplicated so an ancestor's walk
// covers descendants. Scalar fields whose trie chain passes through such a node are
// covered by it; the rest get a direct action or a NULL.
unique_ptr<ShapePlan> BuildShapePlan(TransformLocalState &lstate, const TransformBindData &bind_data,
                                     const JsonoView &view) {
	auto plan = make_uniq<ShapePlan>();
	auto &chain = lstate.chain_scratch;

	// Distinct wildcard-prefix nodes with their root chains (chain.back() == node).
	vector<idx_t> candidate_nodes;
	vector<vector<idx_t>> candidate_chains;
	vector<const vector<PathStep> *> candidate_paths;
	vector<idx_t> candidate_prefix_lens;
	for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
		auto &field = bind_data.fields[field_index];
		if (field.mode == TransformMode::Scalar || field.shred_child_index != DConstants::INVALID_INDEX) {
			continue;
		}
		auto prefix_len = FindWildcardIndex(field.path);
		TrieChainForSteps(bind_data, field.path, prefix_len, chain);
		auto node = chain.back();
		if (std::find(candidate_nodes.begin(), candidate_nodes.end(), node) != candidate_nodes.end()) {
			continue;
		}
		candidate_nodes.push_back(node);
		candidate_chains.push_back(chain);
		candidate_paths.push_back(&field.path);
		candidate_prefix_lens.push_back(prefix_len);
	}
	// Covering nodes: candidates no other candidate is an ancestor of.
	vector<idx_t> covering_nodes;
	for (idx_t i = 0; i < candidate_nodes.size(); i++) {
		bool covered = false;
		auto &node_chain = candidate_chains[i];
		for (idx_t j = 0; j < candidate_nodes.size() && !covered; j++) {
			if (j == i) {
				continue;
			}
			covered = std::find(node_chain.begin(), node_chain.end() - 1, candidate_nodes[j]) != node_chain.end() - 1;
		}
		if (covered) {
			continue;
		}
		covering_nodes.push_back(candidate_nodes[i]);
		JsonoCursor cursor;
		bool found = true;
		for (idx_t step = 0; step < candidate_prefix_lens[i]; step++) {
			if (!LocatePathStep(nullptr, step, view, (*candidate_paths[i])[step], cursor)) {
				found = false;
				break;
			}
		}
		if (found) {
			plan->walks.push_back(
			    ShapePlanWalk {candidate_nodes[i], cursor.pos, cursor.length_cursor, cursor.num_cursor, 0});
		} else {
			plan->null_subtree_nodes.push_back(candidate_nodes[i]);
		}
	}

	for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
		auto &field = bind_data.fields[field_index];
		if (field.mode != TransformMode::Scalar || field.shred_child_index != DConstants::INVALID_INDEX) {
			continue;
		}
		TrieChainForSteps(bind_data, field.path, field.path.size(), chain);
		bool covered = false;
		for (auto node : chain) {
			if (std::find(covering_nodes.begin(), covering_nodes.end(), node) != covering_nodes.end()) {
				covered = true;
				break;
			}
		}
		if (covered) {
			continue;
		}
		JsonoCursor cursor;
		if (LocatePathSteps(nullptr, 0, field.path, view, cursor)) {
			PlanScalarAt(*plan, view, cursor, field_index);
		} else {
			plan->null_fields.push_back(field_index);
		}
	}

	// Shred-backed scalars: residual-first, exactly like the per-row fallback. A residual
	// hit is shape-constant; a miss reads the shred column per row.
	for (idx_t k = 0; k < bind_data.shred_fields.size(); k++) {
		auto &field = bind_data.fields[bind_data.shred_fields[k]];
		JsonoCursor cursor;
		if (LocatePathSteps(nullptr, 0, field.path, view, cursor)) {
			PlanScalarAt(*plan, view, cursor, bind_data.shred_fields[k]);
		} else {
			plan->shred_reads.push_back(k);
		}
	}

	// Assign prefix-sum offset slots and the per-row stream bounds.
	auto &needed = plan->needed_length_indices;
	for (auto &action : plan->scalars) {
		if (action.kind == JsonoScalarKind::String || action.kind == JsonoScalarKind::NumberText) {
			needed.push_back(action.length_index);
		}
	}
	for (auto &walk : plan->walks) {
		needed.push_back(walk.length_cursor);
	}
	std::sort(needed.begin(), needed.end());
	needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
	for (auto &action : plan->scalars) {
		switch (action.kind) {
		case JsonoScalarKind::String:
		case JsonoScalarKind::NumberText:
			action.offset_slot =
			    idx_t(std::lower_bound(needed.begin(), needed.end(), action.length_index) - needed.begin());
			plan->lengths_bound = std::max(plan->lengths_bound, action.length_index + 1);
			break;
		case JsonoScalarKind::Int64:
		case JsonoScalarKind::UInt64:
		case JsonoScalarKind::Double:
		case JsonoScalarKind::Dec60:
			plan->nums_bound = std::max(plan->nums_bound, action.num_index + 1);
			break;
		default:
			break;
		}
	}
	for (auto &walk : plan->walks) {
		walk.offset_slot = idx_t(std::lower_bound(needed.begin(), needed.end(), walk.length_cursor) - needed.begin());
		plan->lengths_bound = std::max(plan->lengths_bound, walk.length_cursor);
	}
	return plan;
}

void ApplyShapePlan(TransformLocalState &lstate, const TransformBindData &bind_data, const ShapePlan &plan,
                    const JsonoView &view, vector<unique_ptr<Vector>> &children, idx_t row,
                    vector<string> &join_buffers, vector<idx_t> &join_counts,
                    const vector<UnifiedVectorFormat> &shred_fmt) {
	// One pass over the lengths stream produces the byte offsets of every needed string
	// position (offset before needed index i = sum of lengths[0, i)).
	const uint8_t *lengths_raw = plan.lengths_bound > 0 ? view.LengthsBytes(0, plan.lengths_bound) : nullptr;
	const uint8_t *nums_raw = plan.nums_bound > 0 ? view.NumsBytes(0, plan.nums_bound) : nullptr;
	auto &offsets = lstate.shape_offsets;
	auto &needed = plan.needed_length_indices;
	offsets.resize(needed.size());
	{
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

	for (auto field_index : plan.null_fields) {
		FlatVector::SetNull(*children[field_index], row, true);
	}
	for (auto node : plan.null_subtree_nodes) {
		SetTrieNodeOutputsNull(bind_data, node, children, row);
	}
	for (auto &action : plan.scalars) {
		JsonoScalar scalar;
		scalar.kind = action.kind;
		switch (action.kind) {
		case JsonoScalarKind::String:
		case JsonoScalarKind::NumberText: {
			uint32_t len;
			std::memcpy(&len, lengths_raw + action.length_index * sizeof(uint32_t), sizeof(len));
			scalar.text = view.StringAt(offsets[action.offset_slot], len);
			break;
		}
		case JsonoScalarKind::Int64: {
			uint64_t word;
			std::memcpy(&word, nums_raw + action.num_index * sizeof(uint64_t), sizeof(word));
			scalar.int_value = int64_t(word);
			break;
		}
		case JsonoScalarKind::UInt64:
			std::memcpy(&scalar.uint_value, nums_raw + action.num_index * sizeof(uint64_t), sizeof(scalar.uint_value));
			break;
		case JsonoScalarKind::Double:
			std::memcpy(&scalar.double_value, nums_raw + action.num_index * sizeof(uint64_t),
			            sizeof(scalar.double_value));
			break;
		case JsonoScalarKind::Dec60: {
			uint64_t packed;
			std::memcpy(&packed, nums_raw + action.num_index * sizeof(uint64_t), sizeof(packed));
			scalar.dec_negative = Dec60Negative(packed);
			scalar.dec_mantissa = Dec60Mantissa(packed);
			scalar.dec_scale = Dec60Scale(packed);
			break;
		}
		case JsonoScalarKind::Bool:
			scalar.bool_value = action.bool_value;
			break;
		default:
			break;
		}
		WriteScalarValue(bind_data.fields[action.field_index], scalar, *children[action.field_index], row);
	}
	for (auto &walk : plan.walks) {
		JsonoCursor cursor;
		cursor.pos = walk.pos;
		cursor.string_cursor = offsets[walk.offset_slot];
		cursor.length_cursor = walk.length_cursor;
		cursor.num_cursor = walk.num_cursor;
		ApplyTrieNode(lstate, bind_data, walk.node_index, view, cursor, children, row, join_buffers, join_counts);
	}
	for (auto k : plan.shred_reads) {
		auto field_index = bind_data.shred_fields[k];
		auto &field = bind_data.fields[field_index];
		auto &fmt = shred_fmt[k];
		auto idx = RowIndex(fmt, row);
		if (fmt.validity.RowIsValid(idx)) {
			auto scalar = SynthShredScalar(field.shred_primitive, fmt, idx);
			WriteScalarValue(field, scalar, *children[field_index], row);
		} else {
			FlatVector::SetNull(*children[field_index], row, true);
		}
	}
}

void JsonoTransformExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<TransformBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<TransformLocalState>();
	auto count = args.size();

	// The row reader verifies each row's shred manifest against the shreds this input type
	// carries, so a row narrowed by a raw struct cast fails loud instead of transforming an
	// incomplete residual.
	JsonoRowReader reader;
	reader.Init(args.data[0], count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (idx_t col = 0; col < bind_data.fields.size(); col++) {
		children[col]->SetVectorType(VectorType::FLAT_VECTOR);
		if (bind_data.fields[col].mode == TransformMode::List) {
			ListVector::SetListSize(*children[col], 0);
		}
	}

	// Read each shred child once; shred-backed scalar fields resolve residual-first and fall back
	// to these columns when the residual dropped the value.
	vector<UnifiedVectorFormat> shred_fmt(bind_data.shred_fields.size());
	if (!bind_data.shred_fields.empty()) {
		for (idx_t k = 0; k < bind_data.shred_fields.size(); k++) {
			auto shred_child = bind_data.fields[bind_data.shred_fields[k]].shred_child_index;
			JsonoShredVector(args.data[0], shred_child).ToUnifiedFormat(count, shred_fmt[k]);
		}
	}

	vector<string> join_buffers(bind_data.fields.size());
	vector<idx_t> join_counts(bind_data.fields.size(), 0);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			SetTransformRowNull(bind_data.fields, result, children, row);
			continue;
		}
		FlatVector::Validity(result).SetValid(row);
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			auto &field = bind_data.fields[field_index];
			if (field.mode == TransformMode::List) {
				auto start = ListVector::GetListSize(*children[field_index]);
				ListVector::GetData(*children[field_index])[row] = {start, 0};
				FlatVector::Validity(*children[field_index]).SetValid(row);
			} else if (field.mode == TransformMode::Join) {
				join_buffers[field_index].clear();
				join_counts[field_index] = 0;
			}
		}
		// The window check runs before the lookup: a disable clears the entries, which would
		// dangle a plan pointer handed out for this row.
		lstate.shape_plans.EndOfWindowCheck();
		const ShapePlan *plan = nullptr;
		if (bind_data.shape_plan_eligible && !lstate.shape_plans.disabled &&
		    blob.slots.GetSize() + blob.key_heap.GetSize() <= SHAPE_PLAN_MAX_SHAPE_BYTES) {
			auto hash = BulkShapeHash(HASH_SEED, blob.slots.GetData(), blob.slots.GetSize());
			hash = BulkShapeHash(hash, blob.key_heap.GetData(), blob.key_heap.GetSize());
			plan = lstate.shape_plans.Find(hash, blob);
			if (!plan) {
				plan = lstate.shape_plans.Insert(hash, blob, BuildShapePlan(lstate, bind_data, view));
			}
		}
		if (plan) {
			ApplyShapePlan(lstate, bind_data, *plan, view, children, row, join_buffers, join_counts, shred_fmt);
		} else {
			ApplyTrieNode(lstate, bind_data, 0, view, JsonoCursor(), children, row, join_buffers, join_counts);
			for (idx_t k = 0; k < bind_data.shred_fields.size(); k++) {
				auto field_index = bind_data.shred_fields[k];
				auto &field = bind_data.fields[field_index];
				auto location = LocatePath(view, field.path);
				if (location.found) {
					// The residual is authoritative: a value the shred writer kept wins over the shred.
					WriteScalarField(field, view, location, *children[field_index], row);
					continue;
				}
				auto &fmt = shred_fmt[k];
				auto idx = RowIndex(fmt, row);
				if (fmt.validity.RowIsValid(idx)) {
					auto scalar = SynthShredScalar(field.shred_primitive, fmt, idx);
					WriteScalarValue(field, scalar, *children[field_index], row);
				} else {
					FlatVector::SetNull(*children[field_index], row, true);
				}
			}
		}
		for (idx_t field_index = 0; field_index < bind_data.fields.size(); field_index++) {
			if (bind_data.fields[field_index].mode != TransformMode::Join) {
				continue;
			}
			auto &buffer = join_buffers[field_index];
			if (join_counts[field_index] == 0) {
				FlatVector::SetNull(*children[field_index], row, true);
				continue;
			}
			FlatVector::Validity(*children[field_index]).SetValid(row);
			FlatVector::GetData<string_t>(*children[field_index])[row] =
			    StringVector::AddString(*children[field_index], buffer.data(), buffer.size());
			buffer.clear();
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoTransform(ExtensionLoader &loader) {
	// ANY input so a shredded JSONO struct (four BLOB prefix + shred columns) also binds; the
	// bind validates it is a plain or shredded JSONO and maps shred columns to scalar fields.
	ScalarFunction fun("jsono_transform", {LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, JsonoTransformExecute,
	                   JsonoTransformBind, nullptr, nullptr, TransformLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	loader.RegisterFunction(fun);
}

} // namespace duckdb
