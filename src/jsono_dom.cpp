#include "jsono_dom.hpp"
#include "jsono_number.hpp"

namespace duckdb {
namespace jsono_dom {

using namespace jsono;
using namespace duckdb_yyjson;

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

} // namespace jsono_dom
} // namespace duckdb
