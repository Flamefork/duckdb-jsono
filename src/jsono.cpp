#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/blob.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/cast/default_casts.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "yyjson.hpp"
#include "string_view.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;
using namespace duckdb_yyjson;

//===--------------------------------------------------------------------===//
// Slots blob layout — only the slot region is stored as a BLOB; heaps live in
// sibling BLOB columns inside JSONO = STRUCT(key_heap, string_heap, slots, skips).
// Header/slot layout is defined in jsono.hpp so transform-on-JSONO and
// to_json can agree on it.
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
// Writer: text JSON → slots blob
//
// Uses yyjson's immutable DOM for parsing because it gives us random access to
// keys (we need to sort them before emitting). Stage 1 cost is paid once per
// document at write time; the SELECT side never pays it.
//===--------------------------------------------------------------------===//

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

bool EmitDomElement(yyjson_val *element, DomJsonoBuilder &b);

bool EmitDomObject(yyjson_val *obj, DomJsonoBuilder &b) {
	// Stack-via-offset on both kvs and indices; collect this object's data at
	// the end of each shared buffer, do the work, truncate back on return.
	size_t kv_offset = b.kvs.size();
	size_t idx_offset = b.indices.size();

	// Collect kvs and hash the DOM-order key sequence in one pass. Both touch
	// the same bytes (key bytes) so the hash is "free" — it rides along the
	// memory loads we'd be doing anyway for the emplace.
	uint64_t shape_hash = HASH_SEED;
	yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
	yyjson_val *key;
	while ((key = yyjson_obj_iter_next(&iter))) {
		nonstd::string_view key_view(yyjson_get_str(key), yyjson_get_len(key));
		shape_hash = HashKey(shape_hash, key_view);
		b.kvs.emplace_back(key_view, yyjson_obj_iter_get_val(key));
	}
	size_t N = b.kvs.size() - kv_offset;
	b.indices.resize(idx_offset + N);

	// Shape cache lookup. On hit we copy the cached (deduplicated) permutation into
	// this object's indices slice and skip std::sort. On miss we sort indices
	// (cheaper than sorting kv pairs: 4-byte swaps vs 24-byte), drop duplicate keys
	// keeping the last occurrence, and cache the result for next time. emit_count is
	// the post-dedup key count and equals N for the common unique-key case.
	size_t emit_count;
	auto cached = b.ShapeCacheFind(shape_hash, uint32_t(N));
	if (cached) {
		emit_count = cached->perm.size();
		std::memcpy(b.indices.data() + idx_offset, cached->perm.data(), emit_count * sizeof(uint32_t));
	} else {
		for (size_t i = 0; i < N; i++) {
			b.indices[idx_offset + i] = uint32_t(i);
		}
		// Tie-break equal keys by input index so the last duplicate sorts last in its run.
		std::sort(b.indices.begin() + idx_offset, b.indices.begin() + idx_offset + N,
		          [&b, kv_offset](uint32_t a, uint32_t c) {
			          int cmp = b.kvs[kv_offset + a].first.compare(b.kvs[kv_offset + c].first);
			          return cmp != 0 ? cmp < 0 : a < c;
		          });
		// Collapse runs of equal keys in place, keeping the last (last-wins), so the
		// stored object has the unique sorted keys the readers' binary search assumes.
		size_t write = idx_offset;
		for (size_t read = idx_offset; read < idx_offset + N;) {
			size_t run_end = read + 1;
			while (run_end < idx_offset + N &&
			       b.kvs[kv_offset + b.indices[run_end]].first == b.kvs[kv_offset + b.indices[read]].first) {
				run_end++;
			}
			b.indices[write++] = b.indices[run_end - 1];
			read = run_end;
		}
		emit_count = write - idx_offset;
		b.ShapeCacheInsert(shape_hash, uint32_t(N), b.indices.data() + idx_offset, uint32_t(emit_count));
	}

	b.EmitObjectStart(emit_count);

	// Pass 1 — KEY slots. No recursion here, so iterators into b.kvs stay
	// valid for the duration of this loop.
	for (size_t i = 0; i < emit_count; i++) {
		auto k = b.indices[idx_offset + i];
		b.EmitKeySlot(b.kvs[kv_offset + k].first);
	}

	// Pass 2 — VAL slots. Recursion may emplace_back into b.kvs and trigger
	// realloc; absolute-index access is safe because we read each element
	// (dom::element is a small handle, copied by value into the recursive
	// call) before potential growth.
	for (size_t i = 0; i < emit_count; i++) {
		auto k = b.indices[idx_offset + i];
		b.EmitObjectChildStart();
		if (!EmitDomElement(b.kvs[kv_offset + k].second, b)) {
			b.kvs.resize(kv_offset);
			b.indices.resize(idx_offset);
			return false;
		}
	}

	b.EmitObjectEnd();
	b.kvs.resize(kv_offset);
	b.indices.resize(idx_offset);
	return true;
}

bool EmitDomArray(yyjson_val *arr, DomJsonoBuilder &b) {
	b.EmitArrayStart();
	yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
	yyjson_val *elem;
	while ((elem = yyjson_arr_iter_next(&iter))) {
		if (!EmitDomElement(elem, b)) {
			return false;
		}
	}
	b.EmitArrayEnd();
	return true;
}

bool EmitDomElement(yyjson_val *element, DomJsonoBuilder &b) {
	switch (yyjson_get_type(element)) {
	case YYJSON_TYPE_OBJ:
		return EmitDomObject(element, b);
	case YYJSON_TYPE_ARR:
		return EmitDomArray(element, b);
	case YYJSON_TYPE_STR:
		b.EmitString(nonstd::string_view(yyjson_get_str(element), yyjson_get_len(element)));
		return true;
	case YYJSON_TYPE_RAW:
		// Under NUMBER_AS_RAW every number arrives here as its source text. The
		// classifier walks the spec ladder (INT60/INT64/UINT64/DEC60/NUMBER) and
		// keeps fractional/bignum forms byte-exact.
		EmitNumberFromText(nonstd::string_view(yyjson_get_raw(element), yyjson_get_len(element)), b);
		return true;
	case YYJSON_TYPE_BOOL:
		b.EmitBool(yyjson_get_bool(element));
		return true;
	default: // YYJSON_TYPE_NULL or unknown
		b.EmitNull();
		return true;
	}
}

// Reusable per-thread yyjson allocator. A dynamic allocator grows to the
// largest document seen and is reused across parses, so steady-state reads do
// no malloc churn — the yyjson analogue of the persistent builder buffers.
struct YyjsonAllocator {
	yyjson_alc *alc = yyjson_alc_dyn_new();

	~YyjsonAllocator() {
		if (alc) {
			yyjson_alc_dyn_free(alc);
		}
	}
};

// Per-thread state for jsono. Lifting the parser allocator, builder, and
// shape cache out of the per-chunk stack means heap capacities stabilise
// after one chunk and the LRU shape cache survives across chunks for the
// duration of the query — the whole point of the cache.
struct JsonoParseLocalState : public FunctionLocalState {
	YyjsonAllocator parser;
	DomJsonoBuilder builder;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoParseLocalState>();
	}

	static unique_ptr<FunctionLocalState> InitCast(CastLocalStateParameters &parameters) {
		(void)parameters;
		return make_uniq<JsonoParseLocalState>();
	}
};

struct JsonoConstructorOptions {
	bool omit_nulls = false;
};

enum class StructValueStrategy : uint8_t {
	Null,
	Int,
	Double,
	String,
	RawJson,
	Bool,
	Jsono,
	List,
	Struct,
	NumberText,
	Decimal
};

bool IsPrimitiveConstructorListChild(StructValueStrategy strategy) {
	switch (strategy) {
	case StructValueStrategy::Null:
	case StructValueStrategy::Int:
	case StructValueStrategy::Double:
	case StructValueStrategy::String:
	case StructValueStrategy::Bool:
		return true;
	case StructValueStrategy::RawJson:
	case StructValueStrategy::Jsono:
	case StructValueStrategy::List:
	case StructValueStrategy::Struct:
	case StructValueStrategy::NumberText:
	case StructValueStrategy::Decimal:
		return false;
	}
	return false;
}

bool IsPrimitiveConstructorScalar(StructValueStrategy strategy) {
	switch (strategy) {
	case StructValueStrategy::Null:
	case StructValueStrategy::Int:
	case StructValueStrategy::Double:
	case StructValueStrategy::String:
	case StructValueStrategy::Bool:
		return true;
	case StructValueStrategy::RawJson:
	case StructValueStrategy::Jsono:
	case StructValueStrategy::List:
	case StructValueStrategy::Struct:
	case StructValueStrategy::NumberText:
	case StructValueStrategy::Decimal:
		return false;
	}
	return false;
}

struct JsonoStructPlan {
	StructValueStrategy strategy = StructValueStrategy::Null;
	LogicalType bound_type = LogicalType::SQLNULL;
	vector<string> field_names;
	vector<uint32_t> field_perm;
	vector<char> key_heap;
	vector<uint64_t> key_slots;
	vector<uint64_t> string_object_slot_template;
	vector<JsonoStructPlan> children;
	idx_t key_cache_index = DConstants::INVALID_INDEX;
	bool omit_nulls = false;
	bool flat_scalar_object = false;
	bool string_object = false;
	bool primitive_list = false;
	bool one_list_flat_scalar_object = false;
};

struct JsonoStructBindData : public FunctionData {
	JsonoStructPlan plan;

	explicit JsonoStructBindData(JsonoStructPlan plan_p) : plan(std::move(plan_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoStructBindData>(plan);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoStructBindData>();
		return plan.bound_type == other.plan.bound_type && plan.omit_nulls == other.plan.omit_nulls;
	}
};

struct JsonoStructCastData : public BoundCastData {
	JsonoStructPlan plan;
	unique_ptr<BoundCastInfo> source_cast;

	JsonoStructCastData(JsonoStructPlan plan_p, unique_ptr<BoundCastInfo> source_cast_p)
	    : plan(std::move(plan_p)), source_cast(std::move(source_cast_p)) {
	}

	unique_ptr<BoundCastData> Copy() const override {
		return make_uniq<JsonoStructCastData>(plan,
		                                      source_cast ? make_uniq<BoundCastInfo>(source_cast->Copy()) : nullptr);
	}
};

struct JsonoStructLocalState : public FunctionLocalState {
	YyjsonAllocator parser;
	DomJsonoBuilder builder;
	unique_ptr<FunctionLocalState> source_cast_state;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoStructLocalState>();
	}

	static unique_ptr<FunctionLocalState> InitCast(CastLocalStateParameters &parameters) {
		auto state = make_uniq<JsonoStructLocalState>();
		if (parameters.cast_data) {
			auto &cast_data = parameters.cast_data->Cast<JsonoStructCastData>();
			if (cast_data.source_cast && cast_data.source_cast->init_local_state) {
				CastLocalStateParameters child_parameters(parameters, cast_data.source_cast->cast_data);
				state->source_cast_state = cast_data.source_cast->init_local_state(child_parameters);
			}
		}
		return std::move(state);
	}
};

struct JsonoStructVectorData {
	const JsonoStructPlan *plan = nullptr;
	UnifiedVectorFormat fmt;
	const int64_t *int_data = nullptr;
	const double *double_data = nullptr;
	const string_t *string_data = nullptr;
	const bool *bool_data = nullptr;
	const list_entry_t *list_entries = nullptr;
	JsonoVectorData jsono_data;
	vector<JsonoStructVectorData> children;
};

void SetJsonoRowNull(Vector &result, Vector &slots_vec, Vector &key_heap_vec, Vector &string_heap_vec,
                     Vector &skips_vec, idx_t row) {
	FlatVector::SetNull(result, row, true);
	FlatVector::SetNull(slots_vec, row, true);
	FlatVector::SetNull(key_heap_vec, row, true);
	FlatVector::SetNull(string_heap_vec, row, true);
	FlatVector::SetNull(skips_vec, row, true);
}

bool IsJsonType(const LogicalType &type) {
	return type.id() == LogicalTypeId::VARCHAR && type.HasAlias() && type.GetAlias() == LogicalType::JSON_TYPE_NAME;
}

JsonoStructPlan BuildStructConstructorPlan(const LogicalType &source_type, const JsonoConstructorOptions &options,
                                           const char *function_name) {
	JsonoStructPlan plan;
	plan.omit_nulls = options.omit_nulls;

	switch (source_type.id()) {
	case LogicalTypeId::UNKNOWN:
		throw ParameterNotResolvedException();
	case LogicalTypeId::SQLNULL:
		plan.strategy = StructValueStrategy::Null;
		plan.bound_type = LogicalType::SQLNULL;
		return plan;
	case LogicalTypeId::BOOLEAN:
		plan.strategy = StructValueStrategy::Bool;
		plan.bound_type = LogicalType::BOOLEAN;
		return plan;
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
		// These all fit INT64 losslessly; UBIGINT does not and takes the NumberText path.
		plan.strategy = StructValueStrategy::Int;
		plan.bound_type = LogicalType::BIGINT;
		return plan;
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		plan.strategy = StructValueStrategy::Double;
		plan.bound_type = LogicalType::DOUBLE;
		return plan;
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::UBIGINT:
		// Integers past INT64 don't round-trip through BIGINT. Cast to text and reuse the
		// parse-path classifier (INT60/INT64/UINT64/NUMBER) so construction matches jsono('...').
		plan.strategy = StructValueStrategy::NumberText;
		plan.bound_type = LogicalType::VARCHAR;
		return plan;
	case LogicalTypeId::DECIMAL:
		// Keep the native (mantissa, scale) and emit VAL_DEC60 straight from it (no VARCHAR
		// round-trip); the rare out-of-range value falls back to the same exact text.
		plan.strategy = StructValueStrategy::Decimal;
		plan.bound_type = source_type;
		return plan;
	case LogicalTypeId::VARCHAR:
		plan.strategy = IsJsonType(source_type) ? StructValueStrategy::RawJson : StructValueStrategy::String;
		plan.bound_type = IsJsonType(source_type) ? LogicalType::JSON() : LogicalType::VARCHAR;
		return plan;
	case LogicalTypeId::STRUCT: {
		if (IsJsonoType(source_type)) {
			plan.strategy = StructValueStrategy::Jsono;
			plan.bound_type = JsonoType();
			return plan;
		}
		plan.strategy = StructValueStrategy::Struct;
		auto &children = StructType::GetChildTypes(source_type);
		child_list_t<LogicalType> bound_children;
		plan.children.reserve(children.size());
		plan.field_names.reserve(children.size());
		plan.field_perm.reserve(children.size());
		plan.flat_scalar_object = !plan.omit_nulls;
		plan.string_object = !plan.omit_nulls;
		plan.one_list_flat_scalar_object = !plan.omit_nulls && children.size() <= OBJECT_CHECKPOINT_STRIDE;
		idx_t list_flat_scalar_object_count = 0;
		for (idx_t i = 0; i < children.size(); i++) {
			auto child_plan = BuildStructConstructorPlan(children[i].second, options, function_name);
			switch (child_plan.strategy) {
			case StructValueStrategy::Null:
			case StructValueStrategy::Int:
			case StructValueStrategy::Double:
			case StructValueStrategy::String:
			case StructValueStrategy::Bool:
				break;
			default:
				plan.flat_scalar_object = false;
				break;
			}
			if (child_plan.strategy != StructValueStrategy::String) {
				plan.string_object = false;
			}
			if (child_plan.strategy == StructValueStrategy::List &&
			    child_plan.children[0].strategy == StructValueStrategy::Struct &&
			    child_plan.children[0].flat_scalar_object &&
			    child_plan.children[0].field_perm.size() <= OBJECT_CHECKPOINT_STRIDE) {
				list_flat_scalar_object_count++;
			} else if (!IsPrimitiveConstructorScalar(child_plan.strategy)) {
				plan.one_list_flat_scalar_object = false;
			}
			bound_children.emplace_back(children[i].first, child_plan.bound_type);
			plan.field_names.push_back(children[i].first);
			plan.field_perm.push_back(uint32_t(i));
			plan.children.push_back(std::move(child_plan));
		}
		if (list_flat_scalar_object_count != 1) {
			plan.one_list_flat_scalar_object = false;
		}
		std::sort(plan.field_perm.begin(), plan.field_perm.end(),
		          [&](uint32_t left, uint32_t right) { return plan.field_names[left] < plan.field_names[right]; });
		if (!plan.omit_nulls) {
			for (auto field_idx : plan.field_perm) {
				auto &field_name = plan.field_names[field_idx];
				auto offset = plan.key_heap.size();
				plan.key_heap.insert(plan.key_heap.end(), field_name.begin(), field_name.end());
				plan.key_slots.push_back(
				    MakeSlot(tag::KEY, MakeKeyPayload(uint64_t(offset), uint64_t(field_name.size()))));
			}
		}
		if (plan.string_object) {
			plan.string_object_slot_template.reserve(2 + plan.field_perm.size() * 2);
			plan.string_object_slot_template.push_back(
			    MakeSlot(tag::OBJ_START, MakeContainerPayload(0, plan.field_perm.size())));
			for (auto slot : plan.key_slots) {
				plan.string_object_slot_template.push_back(slot);
			}
			for (idx_t i = 0; i < plan.field_perm.size(); i++) {
				plan.string_object_slot_template.push_back(MakeSlot(tag::VAL_NULL, 0));
			}
			plan.string_object_slot_template.push_back(MakeSlot(tag::OBJ_END, 0));
		}
		plan.bound_type = LogicalType::STRUCT(std::move(bound_children));
		return plan;
	}
	case LogicalTypeId::LIST: {
		plan.strategy = StructValueStrategy::List;
		plan.children.push_back(
		    BuildStructConstructorPlan(ListType::GetChildType(source_type), options, function_name));
		plan.primitive_list = IsPrimitiveConstructorListChild(plan.children[0].strategy);
		plan.bound_type = LogicalType::LIST(plan.children[0].bound_type);
		return plan;
	}
	case LogicalTypeId::ARRAY: {
		plan.strategy = StructValueStrategy::List;
		plan.children.push_back(
		    BuildStructConstructorPlan(ArrayType::GetChildType(source_type), options, function_name));
		plan.primitive_list = IsPrimitiveConstructorListChild(plan.children[0].strategy);
		plan.bound_type = LogicalType::LIST(plan.children[0].bound_type);
		return plan;
	}
	default:
		throw BinderException("%s: unsupported value type '%s'", function_name, source_type.ToString());
	}
}

void AssignStructConstructorKeyCacheIndexes(JsonoStructPlan &plan, idx_t &next_key_cache_index) {
	if (plan.strategy == StructValueStrategy::Struct && !plan.omit_nulls) {
		plan.key_cache_index = next_key_cache_index++;
	}
	for (auto &child : plan.children) {
		AssignStructConstructorKeyCacheIndexes(child, next_key_cache_index);
	}
}

void ApplyConstructorOption(const string &name, Value value, JsonoConstructorOptions &options) {
	auto lower_name = StringUtil::Lower(name);
	if (value.IsNull()) {
		throw BinderException("jsono() option '%s' cannot be NULL", name);
	}
	if (lower_name == "nulls") {
		if (!value.DefaultTryCastAs(LogicalType::VARCHAR)) {
			throw BinderException("jsono() option 'nulls' must be VARCHAR");
		}
		auto nulls = StringUtil::Lower(StringValue::Get(value));
		if (nulls == "include") {
			options.omit_nulls = false;
			return;
		}
		if (nulls == "omit") {
			options.omit_nulls = true;
			return;
		}
		throw BinderException("jsono() option 'nulls' must be 'include' or 'omit'");
	}
	if (lower_name == "omit_nulls") {
		if (!value.DefaultTryCastAs(LogicalType::BOOLEAN)) {
			throw BinderException("jsono() option 'omit_nulls' must be BOOLEAN");
		}
		options.omit_nulls = BooleanValue::Get(value);
		return;
	}
	throw BinderException("jsono() unknown option '%s'", name);
}

JsonoConstructorOptions ParseConstructorOptions(ClientContext &context, vector<unique_ptr<Expression>> &arguments) {
	JsonoConstructorOptions options;
	if (arguments.size() == 1) {
		return options;
	}
	if (arguments.size() != 2) {
		throw BinderException("jsono() STRUCT constructor requires one STRUCT argument and optional STRUCT options");
	}
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (arguments[1]->return_type.id() != LogicalTypeId::STRUCT) {
		throw BinderException("jsono() options must be a STRUCT");
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("jsono() options must be constant");
	}
	auto options_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (options_value.IsNull()) {
		return options;
	}
	auto &option_types = StructType::GetChildTypes(options_value.type());
	auto &option_values = StructValue::GetChildren(options_value);
	for (idx_t i = 0; i < option_types.size(); i++) {
		ApplyConstructorOption(option_types[i].first, option_values[i], options);
	}
	return options;
}

void InitStructConstructorVectorData(Vector &input, idx_t count, const JsonoStructPlan &plan,
                                     JsonoStructVectorData &data) {
	data.plan = &plan;
	if (plan.strategy == StructValueStrategy::Null) {
		return;
	}
	if (plan.strategy == StructValueStrategy::Jsono) {
		InitJsonoVectorData(input, count, data.jsono_data);
		return;
	}
	input.ToUnifiedFormat(count, data.fmt);
	switch (plan.strategy) {
	case StructValueStrategy::Int:
		data.int_data = UnifiedVectorFormat::GetData<int64_t>(data.fmt);
		break;
	case StructValueStrategy::Double:
		data.double_data = UnifiedVectorFormat::GetData<double>(data.fmt);
		break;
	case StructValueStrategy::String:
	case StructValueStrategy::RawJson:
	case StructValueStrategy::NumberText:
		data.string_data = UnifiedVectorFormat::GetData<string_t>(data.fmt);
		break;
	case StructValueStrategy::Bool:
		data.bool_data = UnifiedVectorFormat::GetData<bool>(data.fmt);
		break;
	case StructValueStrategy::List: {
		data.list_entries = UnifiedVectorFormat::GetData<list_entry_t>(data.fmt);
		data.children.emplace_back();
		auto &child = ListVector::GetEntry(input);
		InitStructConstructorVectorData(child, ListVector::GetListSize(input), plan.children[0], data.children.back());
		break;
	}
	case StructValueStrategy::Struct: {
		auto &struct_children = StructVector::GetEntries(input);
		data.children.reserve(plan.children.size());
		for (idx_t i = 0; i < plan.children.size(); i++) {
			data.children.emplace_back();
			InitStructConstructorVectorData(*struct_children[i], count, plan.children[i], data.children.back());
		}
		break;
	}
	case StructValueStrategy::Null:
	case StructValueStrategy::Jsono:
	case StructValueStrategy::Decimal:
		// fmt (set above) carries the native mantissa; EmitConstructorDecimal reads it per row.
		break;
	}
}

bool ConstructorValueRowIsNull(const JsonoStructVectorData &data, idx_t row) {
	switch (data.plan->strategy) {
	case StructValueStrategy::Null:
		return true;
	case StructValueStrategy::Jsono: {
		JsonoBlobRow ignored;
		return !ReadJsonoRow(data.jsono_data, row, ignored);
	}
	default:
		return !RowIsValid(data.fmt, row);
	}
}

void EmitJsonoSubtree(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder) {
	if (pos >= view.Slots()) {
		builder.EmitNull();
		return;
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, pos);
		auto key_start = layout.key_start;
		auto key_count = layout.key_count;
		auto val_cursor = layout.value_start;
		builder.EmitObjectStart(key_count);
		for (size_t i = 0; i < key_count; i++) {
			auto key_slot = view.SlotAt(key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
		for (size_t i = 0; i < key_count; i++) {
			builder.EmitObjectChildStart();
			EmitJsonoSubtree(view, val_cursor, string_cursor, builder);
		}
		builder.EmitObjectEnd();
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, pos);
		builder.EmitArrayStart();
		pos++;
		while (pos < end_pos) {
			EmitJsonoSubtree(view, pos, string_cursor, builder);
		}
		pos = end_pos + 1;
		builder.EmitArrayEnd();
		return;
	}
	default: {
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		switch (scalar.kind) {
		case JsonoScalarKind::String:
			builder.EmitString(scalar.text);
			return;
		case JsonoScalarKind::Int64:
			builder.EmitInt(scalar.int_value);
			return;
		case JsonoScalarKind::UInt64:
			builder.EmitUInt(scalar.uint_value);
			return;
		case JsonoScalarKind::Double:
			builder.EmitDouble(scalar.double_value);
			return;
		case JsonoScalarKind::Dec60:
			builder.EmitDec60(scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			return;
		case JsonoScalarKind::NumberText:
			builder.EmitNumberText(scalar.text);
			return;
		case JsonoScalarKind::Bool:
			builder.EmitBool(scalar.bool_value);
			return;
		case JsonoScalarKind::Null:
			builder.EmitNull();
			return;
		}
		return;
	}
	}
}

void EmitConstructorValue(const JsonoStructVectorData &data, idx_t row, JsonoStructLocalState &lstate,
                          DomJsonoBuilder &builder);

hugeint_t ReadDecimalMantissa(const UnifiedVectorFormat &fmt, idx_t idx, PhysicalType physical_type) {
	switch (physical_type) {
	case PhysicalType::INT16:
		return hugeint_t(int64_t(UnifiedVectorFormat::GetData<int16_t>(fmt)[idx]));
	case PhysicalType::INT32:
		return hugeint_t(int64_t(UnifiedVectorFormat::GetData<int32_t>(fmt)[idx]));
	case PhysicalType::INT64:
		return hugeint_t(UnifiedVectorFormat::GetData<int64_t>(fmt)[idx]);
	case PhysicalType::INT128:
		return UnifiedVectorFormat::GetData<hugeint_t>(fmt)[idx];
	default:
		throw InternalException("jsono constructor: unexpected DECIMAL physical type");
	}
}

// Emit a DECIMAL field straight from its native (mantissa, scale). A fractional value
// whose magnitude fits the 54-bit DEC60 field reduces to the exact VAL_DEC60 the text
// classifier would choose, with no per-row string. Integers and out-of-range mantissas
// fall back to the same exact decimal text, classified identically to jsono('<text>').
void EmitConstructorDecimal(const JsonoStructVectorData &data, idx_t idx, DomJsonoBuilder &builder) {
	auto &type = data.plan->bound_type;
	auto scale = DecimalType::GetScale(type);
	hugeint_t mantissa = ReadDecimalMantissa(data.fmt, idx, type.InternalType());
	bool negative = mantissa.upper < 0;
	hugeint_t magnitude = negative ? -mantissa : mantissa;
	if (scale >= 1 && scale <= DEC60_SCALE_MAX && magnitude < hugeint_t(int64_t(DEC60_MANTISSA_LIMIT))) {
		builder.EmitDec60(negative, magnitude.lower, scale);
		return;
	}
	EmitNumberFromText(Decimal::ToString(mantissa, DecimalType::GetWidth(type), scale), builder);
}

void EmitConstructorPrimitiveList(const JsonoStructVectorData &child, idx_t offset, idx_t length,
                                  DomJsonoBuilder &builder) {
	auto child_all_valid = child.plan->strategy == StructValueStrategy::Null || child.fmt.validity.AllValid();
	switch (child.plan->strategy) {
	case StructValueStrategy::Null:
		for (idx_t child_i = offset; child_i < offset + length; child_i++) {
			builder.EmitNull();
		}
		return;
	case StructValueStrategy::Int:
		for (idx_t child_i = offset; child_i < offset + length; child_i++) {
			auto child_idx = RowIndex(child.fmt, child_i);
			if (!child_all_valid && !child.fmt.validity.RowIsValid(child_idx)) {
				builder.EmitNull();
			} else {
				builder.EmitInt(child.int_data[child_idx]);
			}
		}
		return;
	case StructValueStrategy::Double:
		for (idx_t child_i = offset; child_i < offset + length; child_i++) {
			auto child_idx = RowIndex(child.fmt, child_i);
			if (!child_all_valid && !child.fmt.validity.RowIsValid(child_idx)) {
				builder.EmitNull();
			} else {
				builder.EmitDouble(child.double_data[child_idx]);
			}
		}
		return;
	case StructValueStrategy::String:
		for (idx_t child_i = offset; child_i < offset + length; child_i++) {
			auto child_idx = RowIndex(child.fmt, child_i);
			if (!child_all_valid && !child.fmt.validity.RowIsValid(child_idx)) {
				builder.EmitNull();
			} else {
				auto value = child.string_data[child_idx];
				builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
			}
		}
		return;
	case StructValueStrategy::Bool:
		for (idx_t child_i = offset; child_i < offset + length; child_i++) {
			auto child_idx = RowIndex(child.fmt, child_i);
			if (!child_all_valid && !child.fmt.validity.RowIsValid(child_idx)) {
				builder.EmitNull();
			} else {
				builder.EmitBool(child.bool_data[child_idx]);
			}
		}
		return;
	case StructValueStrategy::RawJson:
	case StructValueStrategy::Jsono:
	case StructValueStrategy::List:
	case StructValueStrategy::Struct:
	case StructValueStrategy::NumberText:
	case StructValueStrategy::Decimal:
		throw InternalException("jsono constructor: non-primitive child in primitive list emitter");
	}
}

bool EmitConstructorScalarValue(const JsonoStructVectorData &data, idx_t row, DomJsonoBuilder &builder) {
	if (data.plan->strategy == StructValueStrategy::Null) {
		builder.EmitNull();
		return true;
	}
	auto idx = RowIndex(data.fmt, row);
	if (!data.fmt.validity.RowIsValid(idx)) {
		builder.EmitNull();
		return true;
	}
	switch (data.plan->strategy) {
	case StructValueStrategy::Int:
		builder.EmitInt(data.int_data[idx]);
		return true;
	case StructValueStrategy::Double:
		builder.EmitDouble(data.double_data[idx]);
		return true;
	case StructValueStrategy::String: {
		auto value = data.string_data[idx];
		builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
		return true;
	}
	case StructValueStrategy::Bool:
		builder.EmitBool(data.bool_data[idx]);
		return true;
	case StructValueStrategy::Null:
		return true;
	case StructValueStrategy::RawJson:
	case StructValueStrategy::Jsono:
	case StructValueStrategy::List:
	case StructValueStrategy::Struct:
	case StructValueStrategy::NumberText:
	case StructValueStrategy::Decimal:
		return false;
	}
	return false;
}

uint64_t BeginConstructorDirectContainer(DomJsonoBuilder &builder, uint64_t container_tag, uint32_t child_count) {
	if (child_count > CONTAINER_CHILD_COUNT_MASK) {
		throw InvalidInputException("jsono: object has too many keys for storage");
	}
	if (builder.skips.size() > CONTAINER_ID_MAX) {
		throw InvalidInputException("jsono: too many containers for storage");
	}
	auto id = uint64_t(builder.skips.size());
	builder.skips.push_back(ContainerSpan {0, 0});
	builder.slots.push_back(MakeSlot(container_tag, MakeContainerPayload(id, child_count)));
	return id;
}

void FinishConstructorDirectContainer(DomJsonoBuilder &builder, uint64_t container_id, size_t start_slot,
                                      size_t start_string) {
	auto slot_span = builder.slots.size() - start_slot;
	auto string_byte_span = builder.string_heap.size() - start_string;
	if (slot_span > std::numeric_limits<uint32_t>::max() || string_byte_span > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: container span exceeds storage limits");
	}
	builder.skips[container_id] = ContainerSpan {uint32_t(slot_span), uint32_t(string_byte_span)};
}

void EmitConstructorObjectKeys(const JsonoStructPlan &plan, DomJsonoBuilder &builder) {
	if (!plan.omit_nulls) {
		if (builder.external_root_keys) {
			builder.external_root_keys = false;
			builder.slots.insert(builder.slots.end(), plan.key_slots.begin(), plan.key_slots.end());
			return;
		}
		size_t key_offset;
		if (plan.key_cache_index != DConstants::INVALID_INDEX) {
			auto cache_index = plan.key_cache_index;
			if (builder.static_key_offsets.size() <= cache_index) {
				builder.static_key_offsets.resize(cache_index + 1);
				builder.static_key_offset_valid.resize(cache_index + 1, 0);
			}
			if (builder.static_key_offset_valid[cache_index]) {
				key_offset = builder.static_key_offsets[cache_index];
			} else {
				key_offset = builder.key_heap.size();
				builder.key_heap.insert(builder.key_heap.end(), plan.key_heap.begin(), plan.key_heap.end());
				builder.static_key_offsets[cache_index] = key_offset;
				builder.static_key_offset_valid[cache_index] = 1;
			}
		} else {
			key_offset = builder.key_heap.size();
			builder.key_heap.insert(builder.key_heap.end(), plan.key_heap.begin(), plan.key_heap.end());
		}
		if (key_offset == 0) {
			builder.slots.insert(builder.slots.end(), plan.key_slots.begin(), plan.key_slots.end());
		} else {
			for (auto slot : plan.key_slots) {
				auto payload = SlotPayload(slot);
				auto rebased_slot =
				    MakeSlot(tag::KEY, MakeKeyPayload(KeyOffset(payload) + key_offset, KeyLen(payload)));
				builder.slots.push_back(rebased_slot);
			}
		}
		return;
	}
	for (auto field_idx : plan.field_perm) {
		builder.EmitKeySlot(plan.field_names[field_idx]);
	}
}

void EmitConstructorStringObject(const JsonoStructVectorData &data, idx_t row, DomJsonoBuilder &builder) {
	auto &plan = *data.plan;
	builder.EmitObjectStart(plan.field_perm.size());
	EmitConstructorObjectKeys(plan, builder);
	for (auto field_idx : plan.field_perm) {
		auto &child = data.children[field_idx];
		auto idx = RowIndex(child.fmt, row);
		builder.EmitObjectChildStart();
		if (!child.fmt.validity.RowIsValid(idx)) {
			builder.EmitNull();
		} else {
			auto value = child.string_data[idx];
			builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
		}
	}
	builder.EmitObjectEnd();
}

void EmitConstructorFlatScalarObject(const JsonoStructVectorData &data, idx_t row, DomJsonoBuilder &builder) {
	auto &plan = *data.plan;
	builder.EmitObjectStart(plan.field_perm.size());
	EmitConstructorObjectKeys(plan, builder);
	for (auto field_idx : plan.field_perm) {
		auto &child = data.children[field_idx];
		builder.EmitObjectChildStart();
		if (child.plan->strategy == StructValueStrategy::Null) {
			builder.EmitNull();
			continue;
		}
		auto idx = RowIndex(child.fmt, row);
		if (!child.fmt.validity.RowIsValid(idx)) {
			builder.EmitNull();
			continue;
		}
		switch (child.plan->strategy) {
		case StructValueStrategy::Int:
			builder.EmitInt(child.int_data[idx]);
			break;
		case StructValueStrategy::Double:
			builder.EmitDouble(child.double_data[idx]);
			break;
		case StructValueStrategy::String: {
			auto value = child.string_data[idx];
			builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
			break;
		}
		case StructValueStrategy::Bool:
			builder.EmitBool(child.bool_data[idx]);
			break;
		case StructValueStrategy::Null:
		case StructValueStrategy::RawJson:
		case StructValueStrategy::Jsono:
		case StructValueStrategy::List:
		case StructValueStrategy::Struct:
		case StructValueStrategy::NumberText:
		case StructValueStrategy::Decimal:
			break;
		}
	}
	builder.EmitObjectEnd();
}

void EmitConstructorDirectFlatScalarObject(const JsonoStructVectorData &data, idx_t row, DomJsonoBuilder &builder) {
	auto &plan = *data.plan;
	auto start_slot = builder.slots.size();
	auto start_string = builder.string_heap.size();
	auto container_id = BeginConstructorDirectContainer(builder, tag::OBJ_START, uint32_t(plan.field_perm.size()));
	EmitConstructorObjectKeys(plan, builder);
	for (auto field_idx : plan.field_perm) {
		if (!EmitConstructorScalarValue(data.children[field_idx], row, builder)) {
			throw InternalException("jsono constructor: non-scalar child in direct flat scalar object emitter");
		}
	}
	builder.slots.push_back(MakeSlot(tag::OBJ_END, 0));
	FinishConstructorDirectContainer(builder, container_id, start_slot, start_string);
}

void EmitConstructorDirectListOfFlatScalarObjects(const JsonoStructVectorData &data, idx_t row,
                                                  DomJsonoBuilder &builder) {
	auto idx = RowIndex(data.fmt, row);
	if (!data.fmt.validity.RowIsValid(idx)) {
		builder.EmitNull();
		return;
	}
	auto entry = data.list_entries[idx];
	auto &child = data.children[0];
	auto start_slot = builder.slots.size();
	auto start_string = builder.string_heap.size();
	auto container_id = BeginConstructorDirectContainer(builder, tag::ARR_START, 0);
	auto child_all_valid = child.fmt.validity.AllValid();
	for (idx_t child_i = entry.offset; child_i < entry.offset + entry.length; child_i++) {
		if (!child_all_valid && ConstructorValueRowIsNull(child, child_i)) {
			builder.EmitNull();
		} else {
			EmitConstructorDirectFlatScalarObject(child, child_i, builder);
		}
	}
	builder.slots.push_back(MakeSlot(tag::ARR_END, 0));
	FinishConstructorDirectContainer(builder, container_id, start_slot, start_string);
}

void EmitConstructorOneListFlatScalarObject(const JsonoStructVectorData &data, idx_t row, DomJsonoBuilder &builder) {
	auto &plan = *data.plan;
	auto start_slot = builder.slots.size();
	auto start_string = builder.string_heap.size();
	auto container_id = BeginConstructorDirectContainer(builder, tag::OBJ_START, uint32_t(plan.field_perm.size()));
	EmitConstructorObjectKeys(plan, builder);
	for (auto field_idx : plan.field_perm) {
		auto &child = data.children[field_idx];
		if (child.plan->strategy == StructValueStrategy::List && child.plan->children[0].flat_scalar_object) {
			EmitConstructorDirectListOfFlatScalarObjects(child, row, builder);
		} else if (!EmitConstructorScalarValue(child, row, builder)) {
			throw InternalException("jsono constructor: non-products-like child in direct object emitter");
		}
	}
	builder.slots.push_back(MakeSlot(tag::OBJ_END, 0));
	FinishConstructorDirectContainer(builder, container_id, start_slot, start_string);
}

void EmitConstructorObject(const JsonoStructVectorData &data, idx_t row, JsonoStructLocalState &lstate,
                           DomJsonoBuilder &builder) {
	auto &plan = *data.plan;
	if (plan.string_object) {
		EmitConstructorStringObject(data, row, builder);
		return;
	}
	if (plan.flat_scalar_object) {
		EmitConstructorFlatScalarObject(data, row, builder);
		return;
	}
	if (plan.one_list_flat_scalar_object) {
		EmitConstructorOneListFlatScalarObject(data, row, builder);
		return;
	}

	idx_t field_count = 0;
	if (plan.omit_nulls) {
		for (auto field_idx : plan.field_perm) {
			if (!ConstructorValueRowIsNull(data.children[field_idx], row)) {
				field_count++;
			}
		}
	} else {
		field_count = plan.field_perm.size();
	}

	builder.EmitObjectStart(field_count);
	if (plan.omit_nulls) {
		for (auto field_idx : plan.field_perm) {
			if (ConstructorValueRowIsNull(data.children[field_idx], row)) {
				continue;
			}
			builder.EmitKeySlot(plan.field_names[field_idx]);
		}
	} else {
		EmitConstructorObjectKeys(plan, builder);
	}
	for (auto field_idx : plan.field_perm) {
		if (plan.omit_nulls && ConstructorValueRowIsNull(data.children[field_idx], row)) {
			continue;
		}
		builder.EmitObjectChildStart();
		EmitConstructorValue(data.children[field_idx], row, lstate, builder);
	}
	builder.EmitObjectEnd();
}

void EmitConstructorValue(const JsonoStructVectorData &data, idx_t row, JsonoStructLocalState &lstate,
                          DomJsonoBuilder &builder) {
	if (data.plan->strategy == StructValueStrategy::Null) {
		builder.EmitNull();
		return;
	}
	if (data.plan->strategy == StructValueStrategy::Jsono) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(data.jsono_data, row, blob)) {
			builder.EmitNull();
			return;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			builder.EmitNull();
			return;
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		EmitJsonoSubtree(view, pos, string_cursor, builder);
		return;
	}

	auto idx = RowIndex(data.fmt, row);
	if (!data.fmt.validity.RowIsValid(idx)) {
		builder.EmitNull();
		return;
	}
	switch (data.plan->strategy) {
	case StructValueStrategy::Int:
		builder.EmitInt(data.int_data[idx]);
		return;
	case StructValueStrategy::Double:
		builder.EmitDouble(data.double_data[idx]);
		return;
	case StructValueStrategy::String: {
		auto value = data.string_data[idx];
		builder.EmitString(nonstd::string_view(value.GetData(), value.GetSize()));
		return;
	}
	case StructValueStrategy::NumberText: {
		// UBIGINT and 128-bit integers arrive as their exact text; the parse-path
		// classifier picks the lossless slot encoding (INT60/INT64/UINT64/NUMBER).
		auto value = data.string_data[idx];
		EmitNumberFromText(nonstd::string_view(value.GetData(), value.GetSize()), builder);
		return;
	}
	case StructValueStrategy::Decimal:
		EmitConstructorDecimal(data, idx, builder);
		return;
	case StructValueStrategy::RawJson: {
		auto value = data.string_data[idx];
		yyjson_read_err err;
		auto doc = ReadJsonoDoc(value, lstate.parser.alc, err);
		if (!doc) {
			throw InvalidInputException("jsono: invalid JSON - %s", err.msg);
		}
		bool ok = EmitDomElement(yyjson_doc_get_root(doc), builder);
		yyjson_doc_free(doc);
		if (!ok) {
			throw InvalidInputException("jsono: unsupported JSON value");
		}
		return;
	}
	case StructValueStrategy::Bool:
		builder.EmitBool(data.bool_data[idx]);
		return;
	case StructValueStrategy::List: {
		auto entry = data.list_entries[idx];
		auto &child = data.children[0];
		builder.EmitArrayStart();
		if (data.plan->primitive_list &&
		    (child.plan->strategy == StructValueStrategy::Null || child.fmt.validity.AllValid())) {
			EmitConstructorPrimitiveList(child, entry.offset, entry.length, builder);
		} else {
			for (idx_t child_i = entry.offset; child_i < entry.offset + entry.length; child_i++) {
				EmitConstructorValue(child, child_i, lstate, builder);
			}
		}
		builder.EmitArrayEnd();
		return;
	}
	case StructValueStrategy::Struct:
		EmitConstructorObject(data, row, lstate, builder);
		return;
	case StructValueStrategy::Null:
	case StructValueStrategy::Jsono:
		return;
	}
}

void WriteSlotInto(char *slots_dst, idx_t slot_idx, uint64_t slot) {
	std::memcpy(slots_dst + JSONO_HEADER_SIZE + slot_idx * sizeof(uint64_t), &slot, sizeof(slot));
}

string_t WriteSingleContainerSkipsBlobInto(Vector &vec, uint32_t slot_span, uint32_t string_byte_span) {
	auto total = sizeof(ContainerMetadataHeader) + sizeof(ContainerSpan);
	auto skips_str = StringVector::EmptyString(vec, total);
	auto skips_dst = skips_str.GetDataWriteable();

	ContainerMetadataHeader header;
	header.container_count = 1;
	header.checkpoint_index_count = 0;
	header.checkpoint_count = 0;
	std::memcpy(skips_dst, &header, sizeof(header));

	ContainerSpan span {slot_span, string_byte_span};
	std::memcpy(skips_dst + sizeof(header), &span, sizeof(span));

	skips_str.Finalize();
	return skips_str;
}

void WriteConstructorStringObjectDirect(const JsonoStructVectorData &data, idx_t row, Vector &string_heap_vec,
                                        Vector &slots_vec, Vector &skips_vec, string_t *string_heap_data,
                                        string_t *slots_data, string_t *skips_data) {
	auto &plan = *data.plan;
	auto field_count = plan.field_perm.size();
	auto slot_count = uint32_t(2 + field_count * 2);
	auto slots_total = JSONO_HEADER_SIZE + size_t(slot_count) * sizeof(uint64_t);
	auto slots_str = StringVector::EmptyString(slots_vec, slots_total);
	auto slots_dst = slots_str.GetDataWriteable();

	JsonoHeader header;
	header.magic = MAGIC;
	header.version = VERSION;
	header.flags = flags::SORTED_KEYS;
	header.reserved = 0;
	std::memcpy(slots_dst, &header, JSONO_HEADER_SIZE);
	std::memcpy(slots_dst + JSONO_HEADER_SIZE, plan.string_object_slot_template.data(),
	            plan.string_object_slot_template.size() * sizeof(uint64_t));

	std::array<string_t, 64> values;
	std::array<uint8_t, 64> value_valid;
	size_t values_size = 0;
	if (field_count <= values.size()) {
		for (idx_t value_idx = 0; value_idx < field_count; value_idx++) {
			auto &child = data.children[plan.field_perm[value_idx]];
			auto child_idx = RowIndex(child.fmt, row);
			if (child.fmt.validity.RowIsValid(child_idx)) {
				value_valid[value_idx] = 1;
				values[value_idx] = child.string_data[child_idx];
				values_size += values[value_idx].GetSize();
			} else {
				value_valid[value_idx] = 0;
			}
		}
	} else {
		for (idx_t value_idx = 0; value_idx < field_count; value_idx++) {
			auto &child = data.children[plan.field_perm[value_idx]];
			auto child_idx = RowIndex(child.fmt, row);
			if (child.fmt.validity.RowIsValid(child_idx)) {
				values_size += child.string_data[child_idx].GetSize();
			}
		}
	}
	if (values_size > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: container string span exceeds storage limits");
	}
	auto values_str = StringVector::EmptyString(string_heap_vec, values_size);
	auto values_dst = values_str.GetDataWriteable();
	size_t value_offset = 0;
	auto value_start = field_count + 1;
	for (idx_t value_idx = 0; value_idx < field_count; value_idx++) {
		string_t value;
		if (field_count <= values.size()) {
			if (!value_valid[value_idx]) {
				WriteSlotInto(slots_dst, value_start + value_idx, MakeSlot(tag::VAL_NULL, 0));
				continue;
			}
			value = values[value_idx];
		} else {
			auto &child = data.children[plan.field_perm[value_idx]];
			auto child_idx = RowIndex(child.fmt, row);
			if (!child.fmt.validity.RowIsValid(child_idx)) {
				WriteSlotInto(slots_dst, value_start + value_idx, MakeSlot(tag::VAL_NULL, 0));
				continue;
			}
			value = child.string_data[child_idx];
		}
		if (value.GetSize() > 0) {
			std::memcpy(values_dst + value_offset, value.GetData(), value.GetSize());
		}
		value_offset += value.GetSize();
		WriteSlotInto(slots_dst, value_start + value_idx,
		              MakeSlot(tag::VAL_STR_HEAP, MakeStringPayload(value.GetSize())));
	}
	values_str.Finalize();
	slots_str.Finalize();

	string_heap_data[row] = values_str;
	slots_data[row] = slots_str;
	skips_data[row] = WriteSingleContainerSkipsBlobInto(skips_vec, slot_count, uint32_t(values_size));
}

void ExecuteStructConstructor(Vector &input, idx_t count, Vector &result, const JsonoStructPlan &plan,
                              JsonoStructLocalState &lstate) {
	JsonoStructVectorData input_data;
	InitStructConstructorVectorData(input, count, plan, input_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_children = StructVector::GetEntries(result);
	auto &slots_vec = *result_children[0];
	auto &key_heap_vec = *result_children[1];
	auto &string_heap_vec = *result_children[2];
	auto &skips_vec = *result_children[3];
	auto slots_data = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_data = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_data = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_data = FlatVector::GetData<string_t>(skips_vec);

	if (plan.strategy == StructValueStrategy::Jsono) {
		for (idx_t row = 0; row < count; row++) {
			JsonoBlobRow blob;
			if (!ReadJsonoRow(input_data.jsono_data, row, blob)) {
				SetJsonoRowNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
				continue;
			}
			key_heap_data[row] = StringVector::AddStringOrBlob(key_heap_vec, blob.key_heap);
			string_heap_data[row] = StringVector::AddStringOrBlob(string_heap_vec, blob.string_heap);
			slots_data[row] = StringVector::AddStringOrBlob(slots_vec, blob.slots);
			skips_data[row] = StringVector::AddStringOrBlob(skips_vec, blob.skips);
		}
		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	string_t static_keys;
	auto use_static_keys = plan.strategy == StructValueStrategy::Struct && plan.flat_scalar_object && !plan.omit_nulls;
	auto use_direct_string_object = use_static_keys && plan.string_object;
	if (use_static_keys) {
		static_keys = WriteBlobInto(key_heap_vec, plan.key_heap.data(), plan.key_heap.size());
	}

	auto &builder = lstate.builder;
	for (idx_t row = 0; row < count; row++) {
		if (ConstructorValueRowIsNull(input_data, row)) {
			SetJsonoRowNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			continue;
		}
		builder.Reset();
		if (use_static_keys) {
			builder.external_root_keys = true;
		}
		if (use_direct_string_object) {
			key_heap_data[row] = static_keys;
			WriteConstructorStringObjectDirect(input_data, row, string_heap_vec, slots_vec, skips_vec, string_heap_data,
			                                   slots_data, skips_data);
			continue;
		}
		if (plan.strategy == StructValueStrategy::Struct) {
			EmitConstructorObject(input_data, row, lstate, builder);
		} else {
			EmitConstructorValue(input_data, row, lstate, builder);
		}
		if (use_static_keys) {
			key_heap_data[row] = static_keys;
			string_heap_data[row] =
			    WriteBlobInto(string_heap_vec, builder.string_heap.data(), builder.string_heap.size());
			slots_data[row] = WriteJsonoBlobInto(slots_vec, builder);
			skips_data[row] = WriteSkipsBlobInto(skips_vec, builder);
		} else {
			WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_data, key_heap_data,
			                 string_heap_data, skips_data, row, builder);
		}
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

unique_ptr<FunctionData> JsonoStructBind(ClientContext &context, ScalarFunction &bound_function,
                                         vector<unique_ptr<Expression>> &arguments) {
	if (arguments.empty() || arguments[0]->return_type.id() != LogicalTypeId::STRUCT) {
		throw BinderException("jsono() STRUCT constructor requires a STRUCT argument");
	}
	auto options = ParseConstructorOptions(context, arguments);
	auto plan = BuildStructConstructorPlan(arguments[0]->return_type, options, "jsono()");
	idx_t next_key_cache_index = 0;
	AssignStructConstructorKeyCacheIndexes(plan, next_key_cache_index);
	bound_function.arguments[0] = plan.bound_type;
	if (arguments.size() == 2) {
		Function::EraseArgument(bound_function, arguments, 1);
	}
	return make_uniq<JsonoStructBindData>(std::move(plan));
}

void JsonoStructExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoStructLocalState>();
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = func_expr.bind_info->Cast<JsonoStructBindData>();
	ExecuteStructConstructor(args.data[0], args.size(), result, bind_data.plan, lstate);
}

bool JsonoStructCastToJsono(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	if (!parameters.cast_data || !parameters.local_state) {
		throw InternalException("JSONO STRUCT cast missing state");
	}
	auto &cast_data = parameters.cast_data->Cast<JsonoStructCastData>();
	auto &lstate = parameters.local_state->Cast<JsonoStructLocalState>();
	if (cast_data.source_cast) {
		Vector casted(cast_data.plan.bound_type, count);
		CastParameters child_parameters(parameters, cast_data.source_cast->cast_data, lstate.source_cast_state);
		if (!cast_data.source_cast->function(source, casted, count, child_parameters)) {
			return false;
		}
		ExecuteStructConstructor(casted, count, result, cast_data.plan, lstate);
		return true;
	}
	ExecuteStructConstructor(source, count, result, cast_data.plan, lstate);
	return true;
}

BoundCastInfo JsonoStructCastBind(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	(void)input;
	(void)target;
	JsonoConstructorOptions options;
	auto plan = BuildStructConstructorPlan(source, options, "JSONO cast");
	idx_t next_key_cache_index = 0;
	AssignStructConstructorKeyCacheIndexes(plan, next_key_cache_index);
	unique_ptr<BoundCastInfo> source_cast;
	if (source != plan.bound_type) {
		source_cast = make_uniq<BoundCastInfo>(input.GetCastFunction(source, plan.bound_type));
	}
	return BoundCastInfo(JsonoStructCastToJsono,
	                     make_uniq<JsonoStructCastData>(std::move(plan), std::move(source_cast)),
	                     JsonoStructLocalState::InitCast);
}

bool HandleJsonoInputError(const string &message, Vector &result, Vector &key_heap_vec, Vector &string_heap_vec,
                           Vector &slots_vec, Vector &skips_vec, idx_t row, bool null_on_error,
                           CastParameters *parameters) {
	if (!null_on_error) {
		if (parameters) {
			HandleCastError::AssignError(message, *parameters);
		} else {
			throw InvalidInputException(message);
		}
	}
	SetJsonoRowNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
	return false;
}

bool ParseJsonoVector(Vector &input, idx_t count, Vector &result, JsonoParseLocalState &lstate, bool null_on_error,
                      CastParameters *parameters = nullptr) {
	UnifiedVectorFormat input_fmt;
	input.ToUnifiedFormat(count, input_fmt);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_fmt);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_children = StructVector::GetEntries(result);
	auto &slots_vec = *result_children[0];
	auto &key_heap_vec = *result_children[1];
	auto &string_heap_vec = *result_children[2];
	auto &skips_vec = *result_children[3];
	auto slots_data = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_data = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_data = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_data = FlatVector::GetData<string_t>(skips_vec);

	// One pass to sum input bytes for the reserve heuristic. Cheap relative to
	// the actual parse/emit work and pays for itself in avoided reallocs on
	// the first chunk; subsequent chunks no-op out of the reserve calls.
	size_t total_input_bytes = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = input_fmt.sel->get_index(i);
		if (input_fmt.validity.RowIsValid(idx)) {
			total_input_bytes += inputs[idx].GetSize();
		}
	}
	lstate.builder.ReserveForChunk(total_input_bytes);

	auto &builder = lstate.builder;
	auto &parser = lstate.parser;
	bool all_converted = true;

	for (idx_t i = 0; i < count; i++) {
		auto idx = input_fmt.sel->get_index(i);
		if (!input_fmt.validity.RowIsValid(idx)) {
			SetJsonoRowNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, i);
			continue;
		}
		auto input = inputs[idx];
		if (input.GetSize() == 0) {
			all_converted = false;
			HandleJsonoInputError("jsono: input is empty", result, key_heap_vec, string_heap_vec, slots_vec, skips_vec,
			                      i, null_on_error, parameters);
			continue;
		}

		yyjson_read_err err;
		auto doc = ReadJsonoDoc(input, parser.alc, err);
		if (!doc) {
			all_converted = false;
			auto message = StringUtil::Format("jsono: invalid JSON - %s", err.msg);
			HandleJsonoInputError(message, result, key_heap_vec, string_heap_vec, slots_vec, skips_vec, i,
			                      null_on_error, parameters);
			continue;
		}

		builder.Reset();
		bool emitted = EmitDomElement(yyjson_doc_get_root(doc), builder);
		yyjson_doc_free(doc);
		if (!emitted) {
			all_converted = false;
			HandleJsonoInputError("jsono: unsupported JSON value", result, key_heap_vec, string_heap_vec, slots_vec,
			                      skips_vec, i, null_on_error, parameters);
			continue;
		}

		WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_data, key_heap_data,
		                 string_heap_data, skips_data, i, builder);
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return all_converted;
}

void JsonoParseExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoParseLocalState>();
	ParseJsonoVector(args.data[0], args.size(), result, lstate, false);
}

void JsonoTryParseExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoParseLocalState>();
	ParseJsonoVector(args.data[0], args.size(), result, lstate, true);
}

bool JsonoCastToJsono(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	if (!parameters.local_state) {
		throw InternalException("JSONO cast missing local state");
	}
	auto &lstate = parameters.local_state->Cast<JsonoParseLocalState>();
	return ParseJsonoVector(source, count, result, lstate, false, &parameters);
}

void JsonoNullExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

void JsonoIdentityExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	result.Reference(args.data[0]);
}

//===--------------------------------------------------------------------===//
// Reader: slots blob → JSON text (round-trip). Walks JsonoView and emits JSON
// text into a std::string. Spec-driven extraction (jsono_transform) walks
// JsonoView directly without the intermediate string.
//===--------------------------------------------------------------------===//

void EmitJsonValue(const JsonoView &r, size_t &pos, size_t &string_cursor, std::string &out, size_t depth);

void EscapeJsonString(nonstd::string_view s, std::string &out) {
	out.push_back('"');
	for (char c : s) {
		switch (c) {
		case '"':
			out.append("\\\"");
			break;
		case '\\':
			out.append("\\\\");
			break;
		case '\b':
			out.append("\\b");
			break;
		case '\f':
			out.append("\\f");
			break;
		case '\n':
			out.append("\\n");
			break;
		case '\r':
			out.append("\\r");
			break;
		case '\t':
			out.append("\\t");
			break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
				out.append(buf);
			} else {
				out.push_back(c);
			}
		}
	}
	out.push_back('"');
}

void EmitJsonValue(const JsonoView &r, size_t &pos, size_t &string_cursor, std::string &out, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto t = SlotTag(r.SlotAt(pos));

	switch (t) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(r, pos);
		auto key_start = layout.key_start;
		auto key_count = layout.key_count;
		auto val_cursor = layout.value_start;
		out.push_back('{');
		for (size_t i = 0; i < key_count; i++) {
			if (i) {
				out.push_back(',');
			}
			auto key_slot = r.SlotAt(key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			EscapeJsonString(r.KeyAt(SlotPayload(key_slot)), out);
			out.push_back(':');
			EmitJsonValue(r, val_cursor, string_cursor, out, depth + 1);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		out.push_back('}');
		break;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(r, pos);
		pos++;
		out.push_back('[');
		bool first = true;
		while (pos < end_pos) {
			if (!first) {
				out.push_back(',');
			}
			first = false;
			EmitJsonValue(r, pos, string_cursor, out, depth + 1);
		}
		pos = end_pos + 1;
		out.push_back(']');
		break;
	}
	default: {
		auto scalar = DecodeScalarAt(r, pos, string_cursor);
		switch (scalar.kind) {
		case JsonoScalarKind::String:
			EscapeJsonString(scalar.text, out);
			break;
		case JsonoScalarKind::Int64:
			out.append(std::to_string(scalar.int_value));
			break;
		case JsonoScalarKind::UInt64:
			out.append(std::to_string(scalar.uint_value));
			break;
		case JsonoScalarKind::Double:
			EmitDouble(scalar.double_value, out);
			break;
		case JsonoScalarKind::Dec60:
			AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
			break;
		case JsonoScalarKind::NumberText:
			out.append(scalar.text.data(), scalar.text.size());
			break;
		case JsonoScalarKind::Bool:
			out.append(scalar.bool_value ? "true" : "false");
			break;
		case JsonoScalarKind::Null:
			out.append("null");
			break;
		}
		break;
	}
	}
}

void WriteJsonoAsJson(Vector &input, idx_t count, Vector &result) {
	JsonoVectorData input_data;
	InitJsonoVectorData(input, count, input_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);

	std::string buf;

	for (idx_t i = 0; i < count; i++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input_data, i, blob)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		JsonoView reader = MakeJsonoView(blob);
		if (!reader.ParseHeader() || reader.Slots() == 0) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		buf.clear();
		size_t pos = 0;
		size_t string_cursor = 0;
		EmitJsonValue(reader, pos, string_cursor, buf, 0);

		result_data[i] = StringVector::AddString(result, buf.data(), buf.size());
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoToJsonExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	WriteJsonoAsJson(args.data[0], args.size(), result);
}

bool JsonoCastToJson(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	WriteJsonoAsJson(source, count, result);
	return true;
}

// Retype between the raw STRUCT(4 BLOB) and JSONO. The two are physically
// identical and differ only by the JSONO alias, so this is a no-op conversion.
// It cannot use DefaultCasts::ReinterpretCast: Vector::Reinterpret is defined
// only for non-nested types (it shares the auxiliary child buffer by pointer
// without retyping it), and a debug build asserts on a nested reinterpret.
// Reinterpreting the scalar BLOB children individually is allowed; the parent's
// vector type and validity carry across unchanged so a constant input (e.g. a
// folded literal cast) stays constant.
bool JsonoStructRetypeCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, ConstantVector::IsNull(source));
	} else {
		source.Flatten(count);
		result.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::Validity(result) = FlatVector::Validity(source);
	}
	auto &source_children = StructVector::GetEntries(source);
	auto &result_children = StructVector::GetEntries(result);
	for (idx_t i = 0; i < source_children.size(); i++) {
		result_children[i]->Reinterpret(*source_children[i]);
	}
	return true;
}

} // namespace

bool IsJsonoType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	if (type.HasAlias() && type.GetAlias() == JSONO_TYPE_NAME) {
		return true;
	}
	auto &children = StructType::GetChildTypes(type);
	return children.size() == 4 && children[0].first == "jsono_slots" && children[0].second == LogicalType::BLOB &&
	       children[1].first == "jsono_key_heap" && children[1].second == LogicalType::BLOB &&
	       children[2].first == "jsono_string_heap" && children[2].second == LogicalType::BLOB &&
	       children[3].first == "jsono_skips" && children[3].second == LogicalType::BLOB;
}

LogicalType JsonoType() {
	child_list_t<LogicalType> children;
	children.emplace_back("jsono_slots", LogicalType::BLOB);
	children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	children.emplace_back("jsono_skips", LogicalType::BLOB);
	auto t = LogicalType::STRUCT(std::move(children));
	t.SetAlias(JSONO_TYPE_NAME);
	return t;
}

void RegisterJsonoType(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	loader.RegisterType(JSONO_TYPE_NAME, jsono_type);

	{
		ScalarFunctionSet set("jsono");
		set.AddFunction(ScalarFunction({LogicalType::SQLNULL}, jsono_type, JsonoNullExecute));
		set.AddFunction(ScalarFunction({jsono_type}, jsono_type, JsonoIdentityExecute));

		ScalarFunction struct_ctor({LogicalTypeId::STRUCT}, jsono_type, JsonoStructExecute, JsonoStructBind, nullptr,
		                           nullptr, JsonoStructLocalState::Init);
		struct_ctor.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		struct_ctor.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		set.AddFunction(struct_ctor);

		ScalarFunction struct_ctor_with_options({LogicalTypeId::STRUCT, LogicalTypeId::STRUCT}, jsono_type,
		                                        JsonoStructExecute, JsonoStructBind, nullptr, nullptr,
		                                        JsonoStructLocalState::Init);
		struct_ctor_with_options.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		struct_ctor_with_options.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		set.AddFunction(struct_ctor_with_options);

		ScalarFunction text_parse({LogicalType::VARCHAR}, jsono_type, JsonoParseExecute, nullptr, nullptr, nullptr,
		                          JsonoParseLocalState::Init);
		text_parse.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		set.AddFunction(text_parse);

		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("try_jsono");
		set.AddFunction(ScalarFunction({LogicalType::SQLNULL}, jsono_type, JsonoNullExecute));

		ScalarFunction text_parse({LogicalType::VARCHAR}, jsono_type, JsonoTryParseExecute, nullptr, nullptr, nullptr,
		                          JsonoParseLocalState::Init);
		set.AddFunction(text_parse);

		loader.RegisterFunction(set);
	}
	{
		ScalarFunction f("to_json", {jsono_type}, LogicalType::JSON(), JsonoToJsonExecute);
		f.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		loader.RegisterFunction(f);
	}
	{
		BoundCastInfo to_jsono(JsonoCastToJsono, nullptr, JsonoParseLocalState::InitCast);
		loader.RegisterCastFunction(LogicalType::VARCHAR, jsono_type, to_jsono.Copy(), -1);
		loader.RegisterCastFunction(LogicalType::JSON(), jsono_type, std::move(to_jsono), -1);
		loader.RegisterCastFunction(LogicalType::STRUCT({{"any", LogicalType::ANY}}), jsono_type, JsonoStructCastBind,
		                            -1);
		loader.RegisterCastFunction(jsono_type, LogicalType::JSON(), BoundCastInfo(JsonoCastToJson), -1);
		loader.RegisterCastFunction(jsono_type, LogicalType::VARCHAR, BoundCastInfo(JsonoCastToJson), -1);
	}
	// JSONO is physically STRUCT(slots, key_heap, string_heap, skips) with an
	// alias. Reading the column back from a plain Parquet file drops the
	// alias — we still want transform/to_json on it. Register a no-op
	// reinterpret cast both ways so the structural shape implicitly converts.
	child_list_t<LogicalType> raw_children;
	raw_children.emplace_back("jsono_slots", LogicalType::BLOB);
	raw_children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	raw_children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	raw_children.emplace_back("jsono_skips", LogicalType::BLOB);
	auto raw_struct = LogicalType::STRUCT(raw_children);
	loader.RegisterCastFunction(raw_struct, jsono_type, JsonoStructRetypeCast, 1);
	loader.RegisterCastFunction(jsono_type, raw_struct, JsonoStructRetypeCast, 1);
}

} // namespace duckdb
