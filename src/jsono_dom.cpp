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
	// An empty object has no keys to sort, dedup, cache or copy: emit_count stays 0 and the
	// whole permutation block is skipped. Beyond saving work, this avoids `indices.data()`/
	// `indices.begin() + offset` and the cache memcpy on an empty (null-data) vector, which is
	// undefined behavior the sanitizer build traps.
	size_t emit_count = 0;
	if (N > 0) {
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

// ===== Two-pass direct writer (see jsono_dom.hpp) =====

namespace {

constexpr uint64_t KEY_OFFSET_LIMIT = PAYLOAD_MASK >> HEAP_LEN_BITS;
constexpr size_t DIRECT_NO_SPAN = std::numeric_limits<size_t>::max();
// MakeExtSlot(NUMBER) marks the staged slots whose payload lives in
// string_heap (raw text re-read from the DOM) instead of nums.
constexpr uint64_t NUMBER_TEXT_SLOT = (tag::VAL_EXT << TAG_SHIFT) | ext_subtype::NUMBER;

void SizeDomElement(yyjson_val *element, DomDirectState &s, size_t depth);

// Decide what a trie leaf match means for its shred: capture-and-strip on an exact kind
// match, ResidualFill for a VARCHAR shred over a non-string scalar (the `->>` text needs
// the decoded value, so it is filled from the written residual), nothing otherwise (the
// shred stays NULL and the value stays in the residual).
void CaptureShredLeaf(yyjson_val *value, DomShredKind kind, DomShredCapture &cap) {
	switch (yyjson_get_type(value)) {
	case YYJSON_TYPE_STR:
		if (kind == DomShredKind::Varchar) {
			cap.state = DomShredCapture::State::String;
			cap.stripped = true;
			cap.text = nonstd::string_view(yyjson_get_str(value), yyjson_get_len(value));
		} else {
			// Present string at a typed shred path: stays in the residual, shred NULL (case B).
			cap.diverted_scalar = true;
		}
		return;
	case YYJSON_TYPE_RAW: {
		if (kind == DomShredKind::Varchar) {
			cap.state = DomShredCapture::State::ResidualFill;
			return;
		}
		if (kind != DomShredKind::Int64 && kind != DomShredKind::Uint64) {
			// A text number is never stored as kind Double; DOUBLE/BOOLEAN shreds stay NULL —
			// a present number they did not capture is a residual-diverted scalar (case B).
			cap.diverted_scalar = true;
			return;
		}
		auto classified = ClassifyNumberFromText(nonstd::string_view(yyjson_get_raw(value), yyjson_get_len(value)));
		bool int64_kind =
		    classified.slot == MakeSlot(tag::VAL_INT60, 0) || classified.slot == MakeExtSlot(ext_subtype::INT64);
		auto int_value = int64_t(classified.num);
		if (kind == DomShredKind::Int64 && int64_kind) {
			cap.state = DomShredCapture::State::Int;
			cap.stripped = true;
			cap.int_value = int_value;
		} else if (kind == DomShredKind::Uint64 && classified.slot == MakeExtSlot(ext_subtype::UINT64)) {
			cap.state = DomShredCapture::State::Uint;
			cap.stripped = true;
			cap.uint_value = classified.num;
		} else if (kind == DomShredKind::Uint64 && int64_kind && int_value >= 0) {
			cap.state = DomShredCapture::State::Uint;
			cap.stripped = true;
			cap.uint_value = uint64_t(int_value);
		} else {
			// Present number the int/uint shred could not capture (float, or out-of-range/sign):
			// kept in the residual, shred NULL (case B).
			cap.diverted_scalar = true;
		}
		return;
	}
	case YYJSON_TYPE_BOOL:
		if (kind == DomShredKind::Boolean) {
			cap.state = DomShredCapture::State::Bool;
			cap.stripped = true;
			cap.bool_value = yyjson_get_bool(value);
		} else if (kind == DomShredKind::Varchar) {
			cap.state = DomShredCapture::State::ResidualFill;
		} else {
			// Present boolean at a numeric shred path: stays in the residual, shred NULL (case B).
			cap.diverted_scalar = true;
		}
		return;
	default:
		// null literal or container: shred NULL, value stays in the residual. A `->>`+CAST read
		// yields NULL here too, so the bare struct_extract stays correct — not a diversion.
		return;
	}
}

void SizeDomContainerChecks(DomDirectState &s, size_t depth) {
	// `depth` counts open containers, mirroring the builder's
	// open_containers.size() check at EmitContainerStart.
	if (depth >= JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("jsono: nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	if (s.sz.container_count > std::numeric_limits<uint32_t>::max() || s.sz.container_count > CONTAINER_ID_MAX) {
		throw InvalidInputException("jsono: too many containers for storage");
	}
	s.sz.container_count++;
}

void SizeDomObject(yyjson_val *obj, DomDirectState &s, size_t depth, DomShredContext *shred = nullptr,
                   uint32_t trie_node = 0) {
	SizeDomContainerChecks(s, depth);

	auto kv_offset = s.kvs.size();
	auto idx_offset = s.indices.size();

	// Same collect+hash pass as EmitDomObject; the kvs stay retained for the
	// whole row so pass 2 can replay them.
	uint64_t shape_hash = HASH_SEED;
	yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
	yyjson_val *key;
	while ((key = yyjson_obj_iter_next(&iter))) {
		nonstd::string_view key_view(yyjson_get_str(key), yyjson_get_len(key));
		shape_hash = HashKey(shape_hash, key_view);
		s.kvs.emplace_back(key_view, yyjson_obj_iter_get_val(key));
	}
	size_t N = s.kvs.size() - kv_offset;
	s.indices.resize(idx_offset + N);

	// An empty object has no keys: emit_count stays 0, sorted_hash stays the empty-sequence
	// seed (what the miss branch would hash over zero keys), and the whole permutation block is
	// skipped — which also avoids `indices.data()`/`indices.begin() + offset` and the cache
	// memcpy on an empty (null-data) vector, undefined behavior the sanitizer build traps.
	size_t emit_count = 0;
	uint64_t sorted_hash = HASH_SEED;
	if (N > 0) {
		auto cached = s.ShapeCacheFind(shape_hash, uint32_t(N));
		if (cached) {
			emit_count = cached->perm.size();
			std::memcpy(s.indices.data() + idx_offset, cached->perm.data(), emit_count * sizeof(uint32_t));
			sorted_hash = cached->sorted_hash;
		} else {
			for (size_t i = 0; i < N; i++) {
				s.indices[idx_offset + i] = uint32_t(i);
			}
			// Tie-break equal keys by input index so the last duplicate sorts last in its run.
			std::sort(s.indices.begin() + idx_offset, s.indices.begin() + idx_offset + N,
			          [&s, kv_offset](uint32_t a, uint32_t c) {
				          int cmp = s.kvs[kv_offset + a].first.compare(s.kvs[kv_offset + c].first);
				          return cmp != 0 ? cmp < 0 : a < c;
			          });
			// Collapse runs of equal keys in place, keeping the last (last-wins).
			size_t write = idx_offset;
			for (size_t read = idx_offset; read < idx_offset + N;) {
				size_t run_end = read + 1;
				while (run_end < idx_offset + N &&
				       s.kvs[kv_offset + s.indices[run_end]].first == s.kvs[kv_offset + s.indices[read]].first) {
					run_end++;
				}
				s.indices[write++] = s.indices[run_end - 1];
				read = run_end;
			}
			emit_count = write - idx_offset;
			// The stored shape_hash fingerprints the sorted key sequence (what
			// HashObjectKeySlots computes from emitted KEY slots); caching it next
			// to the permutation makes it free on every subsequent hit.
			sorted_hash = HASH_SEED;
			for (size_t i = 0; i < emit_count; i++) {
				sorted_hash = HashKey(sorted_hash, s.kvs[kv_offset + s.indices[idx_offset + i]].first);
			}
			s.ShapeCacheInsert(shape_hash, uint32_t(N), s.indices.data() + idx_offset, uint32_t(emit_count),
			                   sorted_hash);
		}
	}
	// Drop dedup leftovers so the plan's perm slice is compact.
	s.indices.resize(idx_offset + emit_count);

	// Trie-matched object (shredded text write): capture shred leaves among the kept sorted
	// keys, strip the lossless ones out of the emit plan before any sizing, and note which
	// children descend the trie. The shape cache entry above stays unstripped — strips vary
	// per row with the leaf value's kind.
	std::vector<std::pair<size_t, uint32_t>> trie_children;
	if (shred) {
		auto &node = (*shred->nodes)[trie_node];
		std::vector<size_t> strip_pos;
		for (auto &edge : node.children) {
			nonstd::string_view edge_key(edge.first.data(), edge.first.size());
			size_t lo = 0;
			size_t hi = emit_count;
			size_t pos = emit_count;
			while (lo < hi) {
				size_t mid = lo + (hi - lo) / 2;
				int cmp = s.kvs[kv_offset + s.indices[idx_offset + mid]].first.compare(edge_key);
				if (cmp == 0) {
					pos = mid;
					break;
				}
				if (cmp < 0) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			if (pos == emit_count) {
				continue;
			}
			auto &child_node = (*shred->nodes)[edge.second];
			auto *value = s.kvs[kv_offset + s.indices[idx_offset + pos]].second;
			if (child_node.field >= 0) {
				auto &cap = shred->captures[size_t(child_node.field)];
				CaptureShredLeaf(value, shred->kinds[size_t(child_node.field)], cap);
				if (cap.stripped) {
					// A stripped leaf is a scalar; no trie descent continues through it.
					strip_pos.push_back(pos);
					continue;
				}
			}
			if (!child_node.children.empty() && yyjson_get_type(value) == YYJSON_TYPE_OBJ) {
				trie_children.emplace_back(pos, edge.second);
			}
		}
		if (!strip_pos.empty()) {
			std::sort(strip_pos.begin(), strip_pos.end());
			size_t write = 0;
			size_t drop = 0;
			for (size_t i = 0; i < emit_count; i++) {
				if (drop < strip_pos.size() && strip_pos[drop] == i) {
					drop++;
					continue;
				}
				for (auto &child : trie_children) {
					if (child.first == i) {
						child.first = write;
					}
				}
				s.indices[idx_offset + write] = s.indices[idx_offset + i];
				write++;
			}
			emit_count = write;
			s.indices.resize(idx_offset + emit_count);
			// The cached sorted_hash fingerprints the unstripped key set; rehash the kept keys.
			sorted_hash = HASH_SEED;
			for (size_t i = 0; i < emit_count; i++) {
				sorted_hash = HashKey(sorted_hash, s.kvs[kv_offset + s.indices[idx_offset + i]].first);
			}
		}
		std::sort(trie_children.begin(), trie_children.end());
	}

	if (emit_count > CONTAINER_CHILD_COUNT_MASK) {
		throw InvalidInputException("jsono: object has too many keys for storage");
	}
	for (size_t i = 0; i < emit_count; i++) {
		auto len = s.kvs[kv_offset + s.indices[idx_offset + i]].first.size();
		if (len > HEAP_LEN_MASK) {
			throw InvalidInputException("jsono: object key exceeds maximum length");
		}
		s.sz.key_bytes += len;
	}
	if (s.sz.key_bytes > KEY_OFFSET_LIMIT) {
		throw InvalidInputException("jsono: key heap exceeds storage limits");
	}

	auto start_slot_count = s.sz.slot_count;
	s.sz.slot_count += 2 + emit_count;
	if (emit_count > OBJECT_CHECKPOINT_STRIDE) {
		auto stride = emit_count >= LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT ? LARGE_OBJECT_CHECKPOINT_STRIDE
		                                                                    : OBJECT_CHECKPOINT_STRIDE;
		s.sz.checkpoint_index_count++;
		s.sz.checkpoint_count += (emit_count - 1) / stride;
	}

	auto plan_index = s.plans.size();
	s.plans.push_back(DomObjectPlan {sorted_hash, kv_offset, idx_offset, uint32_t(emit_count), false});

	bool has_container_child = false;
	if (trie_children.empty()) {
		for (size_t i = 0; i < emit_count; i++) {
			// Absolute indexing: recursion appends to kvs/indices and may reallocate.
			auto *value = s.kvs[kv_offset + s.indices[idx_offset + i]].second;
			auto value_type = yyjson_get_type(value);
			has_container_child |= value_type == YYJSON_TYPE_OBJ || value_type == YYJSON_TYPE_ARR;
			SizeDomElement(value, s, depth + 1);
		}
	} else {
		size_t next_child = 0;
		for (size_t i = 0; i < emit_count; i++) {
			auto *value = s.kvs[kv_offset + s.indices[idx_offset + i]].second;
			auto value_type = yyjson_get_type(value);
			has_container_child |= value_type == YYJSON_TYPE_OBJ || value_type == YYJSON_TYPE_ARR;
			if (next_child < trie_children.size() && trie_children[next_child].first == i) {
				// Collection above only descends into object values.
				SizeDomObject(value, s, depth + 1, shred, trie_children[next_child].second);
				next_child++;
			} else {
				SizeDomElement(value, s, depth + 1);
			}
		}
	}
	auto slot_span = s.sz.slot_count - start_slot_count;
	bool spanned =
	    emit_count > OBJECT_CHECKPOINT_STRIDE || (has_container_child && slot_span > OBJECT_CHECKPOINT_STRIDE);
	s.plans[plan_index].spanned = spanned;
	if (spanned) {
		s.sz.span_count++;
	}
}

void SizeDomArray(yyjson_val *arr, DomDirectState &s, size_t depth) {
	SizeDomContainerChecks(s, depth);
	if (yyjson_arr_size(arr) > 0) {
		s.sz.span_count++;
	}
	s.sz.slot_count += 2;
	yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
	yyjson_val *elem;
	while ((elem = yyjson_arr_iter_next(&iter))) {
		SizeDomElement(elem, s, depth + 1);
	}
}

void SizeDomElement(yyjson_val *element, DomDirectState &s, size_t depth) {
	switch (yyjson_get_type(element)) {
	case YYJSON_TYPE_OBJ:
		SizeDomObject(element, s, depth);
		return;
	case YYJSON_TYPE_ARR:
		SizeDomArray(element, s, depth);
		return;
	case YYJSON_TYPE_STR: {
		auto len = yyjson_get_len(element);
		if (len > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: string value exceeds storage limits");
		}
		s.sz.string_bytes += len;
		s.sz.length_count++;
		s.sz.slot_count++;
		return;
	}
	case YYJSON_TYPE_RAW: {
		auto classified = ClassifyNumberFromText(nonstd::string_view(yyjson_get_raw(element), yyjson_get_len(element)));
		s.staged_numbers.push_back(classified.slot);
		if (classified.consumes_num) {
			s.staged_numbers.push_back(classified.num);
			s.sz.num_count++;
		} else {
			auto len = yyjson_get_len(element);
			if (len > std::numeric_limits<uint32_t>::max()) {
				throw InvalidInputException("jsono: string value exceeds storage limits");
			}
			s.sz.string_bytes += len;
			s.sz.length_count++;
		}
		s.sz.slot_count++;
		return;
	}
	case YYJSON_TYPE_BOOL:
	default: // YYJSON_TYPE_NULL or unknown
		s.sz.slot_count++;
		return;
	}
}

// Pass-2 destination: raw cursors over the six exact-size result buffers.
// All stores go through memcpy because string buffers carry no alignment
// guarantee (readers load slot words the same way).
struct DirectDest {
	char *slots = nullptr; // first slot word, header already written
	char *key_heap = nullptr;
	char *string_heap = nullptr;
	char *lengths = nullptr;
	char *nums = nullptr;
	char *span_ids = nullptr;
	char *spans = nullptr;
	char *checkpoint_index = nullptr;
	char *checkpoints = nullptr;

	size_t slot_pos = 0;
	size_t key_pos = 0;
	size_t string_pos = 0;
	size_t length_pos = 0;
	size_t num_pos = 0;
	size_t span_pos = 0;
	size_t checkpoint_index_pos = 0;
	size_t checkpoint_pos = 0;
	uint64_t container_count = 0;
	size_t plan_pos = 0;
	size_t staged_pos = 0;

	void PushSlot(uint64_t slot) {
		std::memcpy(slots + slot_pos * sizeof(uint64_t), &slot, sizeof(slot));
		slot_pos++;
	}
	void PushLengthEntry(uint32_t len) {
		std::memcpy(lengths + length_pos * sizeof(uint32_t), &len, sizeof(len));
		length_pos++;
	}
	void PushNum(uint64_t num) {
		std::memcpy(nums + num_pos * sizeof(uint64_t), &num, sizeof(num));
		num_pos++;
	}
	void AppendKeyBytes(nonstd::string_view key) {
		std::memcpy(key_heap + key_pos, key.data(), key.size());
		key_pos += key.size();
	}
	void AppendStringBytes(const char *data, size_t len) {
		std::memcpy(string_heap + string_pos, data, len);
		string_pos += len;
	}
	// Allocates the next span_ids/spans pair and returns its index; the
	// ContainerSpan itself is patched at container end via StoreSpan.
	size_t AllocSpan(uint32_t container_id) {
		std::memcpy(span_ids + span_pos * sizeof(uint32_t), &container_id, sizeof(container_id));
		return span_pos++;
	}
	void StoreSpan(size_t index, const ContainerSpan &span) {
		std::memcpy(spans + index * sizeof(ContainerSpan), &span, sizeof(span));
	}
	void PushCheckpointIndexEntry(const ObjectCheckpointIndex &entry) {
		std::memcpy(checkpoint_index + checkpoint_index_pos * sizeof(ObjectCheckpointIndex), &entry, sizeof(entry));
		checkpoint_index_pos++;
	}
	void StoreCheckpoint(size_t index, const ObjectCursorCheckpoint &checkpoint) {
		std::memcpy(checkpoints + index * sizeof(ObjectCursorCheckpoint), &checkpoint, sizeof(checkpoint));
	}
};

struct DirectFrame {
	uint64_t id;
	size_t span_index;
};

void WriteDomElementDirect(yyjson_val *element, DomDirectState &s, DirectDest &d);

void WriteDomObjectDirect(DomDirectState &s, DirectDest &d) {
	const auto plan = s.plans[d.plan_pos++];
	DirectFrame frame {d.container_count++, DIRECT_NO_SPAN};

	auto checkpoint_offset = NO_OBJECT_CHECKPOINTS;
	uint32_t checkpoint_stride = 0;
	if (plan.spanned) {
		frame.span_index = d.AllocSpan(uint32_t(frame.id));
	}
	if (plan.emit_count > OBJECT_CHECKPOINT_STRIDE) {
		checkpoint_stride = plan.emit_count >= LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT ? LARGE_OBJECT_CHECKPOINT_STRIDE
		                                                                               : OBJECT_CHECKPOINT_STRIDE;
		checkpoint_offset = uint32_t(d.checkpoint_pos);
		d.PushCheckpointIndexEntry(
		    ObjectCheckpointIndex {uint32_t(frame.id), checkpoint_offset, uint16_t(checkpoint_stride), 0});
		d.checkpoint_pos += (plan.emit_count - 1) / checkpoint_stride;
	}

	auto start_slot = d.slot_pos;
	auto start_string = d.string_pos;
	auto start_length = d.length_pos;
	auto start_num = d.num_pos;

	d.PushSlot(MakeSlot(tag::OBJ_START, MakeContainerPayload(frame.id, plan.emit_count)));
	// kvs/indices are frozen after pass 1, so raw pointers stay valid across the recursion
	// below. An empty object emits no key/value slots, so offset into the (then null-data)
	// buffers only when there is something to read — `null + offset` is UB the sanitizer traps.
	const auto *kvs = plan.emit_count ? s.kvs.data() + plan.kv_offset : nullptr;
	const auto *perm = plan.emit_count ? s.indices.data() + plan.idx_offset : nullptr;
	for (uint32_t i = 0; i < plan.emit_count; i++) {
		auto key = kvs[perm[i]].first;
		d.PushSlot(MakeSlot(tag::KEY, MakeKeyPayload(d.key_pos, key.size())));
		d.AppendKeyBytes(key);
	}

	auto value_start = start_slot + 1 + plan.emit_count;
	for (uint32_t i = 0; i < plan.emit_count; i++) {
		if (checkpoint_offset != NO_OBJECT_CHECKPOINTS && i > 0 && i % checkpoint_stride == 0) {
			d.StoreCheckpoint(
			    checkpoint_offset + i / checkpoint_stride - 1,
			    ObjectCursorCheckpoint {uint32_t(d.slot_pos - value_start), uint32_t(d.string_pos - start_string),
			                            uint32_t(d.length_pos - start_length), uint32_t(d.num_pos - start_num)});
		}
		WriteDomElementDirect(kvs[perm[i]].second, s, d);
	}
	d.PushSlot(MakeSlot(tag::OBJ_END, 0));

	if (frame.span_index != DIRECT_NO_SPAN) {
		d.StoreSpan(frame.span_index,
		            ContainerSpan {uint32_t(d.slot_pos - start_slot), uint32_t(d.string_pos - start_string),
		                           plan.sorted_hash, uint32_t(d.length_pos - start_length),
		                           uint32_t(d.num_pos - start_num)});
	}
}

void WriteDomArrayDirect(yyjson_val *arr, DomDirectState &s, DirectDest &d) {
	DirectFrame frame {d.container_count++, DIRECT_NO_SPAN};
	if (yyjson_arr_size(arr) > 0) {
		frame.span_index = d.AllocSpan(uint32_t(frame.id));
	}

	auto start_slot = d.slot_pos;
	auto start_string = d.string_pos;
	auto start_length = d.length_pos;
	auto start_num = d.num_pos;

	d.PushSlot(MakeSlot(tag::ARR_START, MakeContainerPayload(frame.id, 0)));
	yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
	yyjson_val *elem;
	while ((elem = yyjson_arr_iter_next(&iter))) {
		WriteDomElementDirect(elem, s, d);
	}
	d.PushSlot(MakeSlot(tag::ARR_END, 0));

	if (frame.span_index != DIRECT_NO_SPAN) {
		d.StoreSpan(frame.span_index,
		            ContainerSpan {uint32_t(d.slot_pos - start_slot), uint32_t(d.string_pos - start_string), 0,
		                           uint32_t(d.length_pos - start_length), uint32_t(d.num_pos - start_num)});
	}
}

void WriteDomElementDirect(yyjson_val *element, DomDirectState &s, DirectDest &d) {
	switch (yyjson_get_type(element)) {
	case YYJSON_TYPE_OBJ:
		// Object keys/values come from the recorded plan, not the DOM handle.
		WriteDomObjectDirect(s, d);
		return;
	case YYJSON_TYPE_ARR:
		WriteDomArrayDirect(element, s, d);
		return;
	case YYJSON_TYPE_STR: {
		auto len = yyjson_get_len(element);
		d.AppendStringBytes(yyjson_get_str(element), len);
		d.PushLengthEntry(uint32_t(len));
		d.PushSlot(MakeSlot(tag::VAL_STR_HEAP, 0));
		return;
	}
	case YYJSON_TYPE_RAW: {
		auto slot = s.staged_numbers[d.staged_pos++];
		d.PushSlot(slot);
		if (slot == NUMBER_TEXT_SLOT) {
			auto len = yyjson_get_len(element);
			d.AppendStringBytes(yyjson_get_raw(element), len);
			d.PushLengthEntry(uint32_t(len));
		} else {
			d.PushNum(s.staged_numbers[d.staged_pos++]);
		}
		return;
	}
	case YYJSON_TYPE_BOOL:
		d.PushSlot(MakeSlot(yyjson_get_bool(element) ? tag::VAL_TRUE : tag::VAL_FALSE, 0));
		return;
	default: // YYJSON_TYPE_NULL or unknown
		d.PushSlot(MakeSlot(tag::VAL_NULL, 0));
		return;
	}
}

} // namespace

void EmitDomRowDirect(yyjson_val *root, DomDirectState &state, JsonoBodyWriter &writer, idx_t row,
                      DomShredContext *shred) {
	state.ResetRow();
	if (shred) {
		shred->captures.assign(shred->kinds.size(), DomShredCapture {});
		if (yyjson_get_type(root) == YYJSON_TYPE_OBJ) {
			SizeDomObject(root, state, 0, shred, 0);
		} else {
			// Non-object root: no object-key path locates, nothing strips.
			SizeDomElement(root, state, 0);
		}
		shred->manifest.clear();
		uint32_t stripped_count = 0;
		for (auto &cap : shred->captures) {
			if (cap.stripped) {
				stripped_count++;
			}
		}
		if (stripped_count > 0) {
			shred->manifest.append(reinterpret_cast<const char *>(&stripped_count), sizeof(stripped_count));
			for (idx_t f = 0; f < shred->captures.size(); f++) {
				if (shred->captures[f].stripped) {
					shred->manifest.append((*shred->manifest_entries)[f]);
				}
			}
		}
	} else {
		SizeDomElement(root, state, 0);
	}
	const auto &sz = state.sz;

	if (sz.span_count > std::numeric_limits<uint32_t>::max() ||
	    sz.checkpoint_index_count > std::numeric_limits<uint32_t>::max() ||
	    sz.checkpoint_count > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: skips blob exceeds storage limits");
	}
	auto manifest_bytes = shred ? shred->manifest.size() : 0;
	auto slots_total = JSONO_HEADER_SIZE + sz.slot_count * sizeof(uint64_t);
	auto skips_total = sizeof(ContainerMetadataHeader) + sz.span_count * sizeof(uint32_t) +
	                   sz.span_count * sizeof(ContainerSpan) +
	                   sz.checkpoint_index_count * sizeof(ObjectCheckpointIndex) +
	                   sz.checkpoint_count * sizeof(ObjectCursorCheckpoint) + manifest_bytes;
	auto lengths_total = sz.length_count * sizeof(uint32_t);
	auto nums_total = sz.num_count * sizeof(uint64_t);
	// string_t carries a u32 length; an overflowing stream would truncate
	// silently, so fail loud here (this bound also keeps every ContainerSpan
	// and checkpoint delta within u32).
	for (auto total : {slots_total, sz.key_bytes, sz.string_bytes, skips_total, lengths_total, nums_total}) {
		if (total > std::numeric_limits<uint32_t>::max()) {
			throw InvalidInputException("jsono: row exceeds storage limits");
		}
	}

	auto slots_str = StringVector::EmptyString(writer.Slots(), slots_total);
	auto keys_str = StringVector::EmptyString(writer.KeyHeap(), sz.key_bytes);
	auto strings_str = StringVector::EmptyString(writer.StringHeap(), sz.string_bytes);
	auto skips_str = StringVector::EmptyString(writer.Skips(), skips_total);
	auto lengths_str = StringVector::EmptyString(writer.Lengths(), lengths_total);
	auto nums_str = StringVector::EmptyString(writer.Nums(), nums_total);

	auto slots_dst = slots_str.GetDataWriteable();
	JsonoHeader header;
	header.magic = MAGIC;
	header.version = VERSION;
	header.flags = flags::SORTED_KEYS;
	header.reserved = 0;
	std::memcpy(slots_dst, &header, JSONO_HEADER_SIZE);

	auto skips_dst = skips_str.GetDataWriteable();
	ContainerMetadataHeader metadata_header;
	metadata_header.span_count = uint32_t(sz.span_count);
	metadata_header.checkpoint_index_count = uint32_t(sz.checkpoint_index_count);
	metadata_header.checkpoint_count = uint32_t(sz.checkpoint_count);
	std::memcpy(skips_dst, &metadata_header, sizeof(metadata_header));

	DirectDest d;
	d.slots = slots_dst + JSONO_HEADER_SIZE;
	d.key_heap = keys_str.GetDataWriteable();
	d.string_heap = strings_str.GetDataWriteable();
	d.lengths = lengths_str.GetDataWriteable();
	d.nums = nums_str.GetDataWriteable();
	d.span_ids = skips_dst + sizeof(ContainerMetadataHeader);
	d.spans = d.span_ids + sz.span_count * sizeof(uint32_t);
	d.checkpoint_index = d.spans + sz.span_count * sizeof(ContainerSpan);
	d.checkpoints = d.checkpoint_index + sz.checkpoint_index_count * sizeof(ObjectCheckpointIndex);

	WriteDomElementDirect(root, state, d);

	// Pass-1 sizing and pass-2 emission must agree exactly; a mismatch means
	// some buffer was written out of bounds — fail loud before publishing.
	if (d.slot_pos != sz.slot_count || d.key_pos != sz.key_bytes || d.string_pos != sz.string_bytes ||
	    d.length_pos != sz.length_count || d.num_pos != sz.num_count || d.span_pos != sz.span_count ||
	    d.checkpoint_index_pos != sz.checkpoint_index_count || d.checkpoint_pos != sz.checkpoint_count) {
		throw InternalException("jsono direct writer: pass-1 sizing does not match pass-2 emission");
	}

	if (manifest_bytes > 0) {
		std::memcpy(skips_dst + skips_total - manifest_bytes, shred->manifest.data(), manifest_bytes);
	}

	slots_str.Finalize();
	keys_str.Finalize();
	strings_str.Finalize();
	skips_str.Finalize();
	lengths_str.Finalize();
	nums_str.Finalize();
	writer.data[BODY_SLOTS][row] = slots_str;
	writer.data[BODY_KEY_HEAP][row] = keys_str;
	writer.data[BODY_STRING_HEAP][row] = strings_str;
	writer.data[BODY_SKIPS][row] = skips_str;
	writer.data[BODY_LENGTHS][row] = lengths_str;
	writer.data[BODY_NUMS][row] = nums_str;
}

} // namespace jsono_dom
} // namespace duckdb
