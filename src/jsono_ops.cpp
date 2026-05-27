#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

enum class JsonoNullTreatment : uint8_t { DeleteNulls, IgnoreNulls };

// One object child captured in a single pass: its key, the value's slot/string
// position, and the value's slot tag. Capturing the tag here lets the merge loop
// classify null/object children without re-reading slots.
struct MergeChild {
	nonstd::string_view key;
	size_t pos;
	size_t string_cursor;
	uint64_t tag;
};

enum class MergeSrc : uint8_t { A, B, Recurse };

struct MergePlanEntry {
	nonstd::string_view key;
	MergeSrc src;
	size_t ra;
	size_t rb;
};

struct JsonoOpsLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	std::vector<MergeChild> merge_children_a;
	std::vector<MergeChild> merge_children_b;
	std::vector<MergePlanEntry> merge_plan;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<JsonoOpsLocalState>();
	}
};

// ---- Cursor-based merge (the engine for jsono_merge_patch and jsono_group_merge) ----
// Walks source views directly into the builder with no intermediate DOM (no
// std::map, no per-key/value heap allocation). DeleteNulls implements RFC 7396
// merge_patch — the A side is the target (its nulls are kept), a B-side patch null
// deletes a key, and a patch object merged onto a non-object/absent value has its
// own nulls stripped. IgnoreNulls implements jsono_group_merge (null members
// dropped). Array values (and everything nested inside an array) are verbatim.

// Copy a scalar value at (pos, string_cursor) byte-for-byte into the builder,
// advancing pos/string_cursor. Equivalent to decoding the scalar and re-emitting
// it, but without reconstructing the value: every scalar encoding round-trips to
// the identical slot(s) (INT60/DEC60 re-encode to themselves; ext INT64/UINT64/
// DOUBLE re-store the same raw bits; STR_HEAP/NUMBER re-append the same heap
// bytes), so copying the source slot(s) and heap slice is the same result with
// none of the decode/re-classify work. Caller must have excluded container slots.
void EmitScalarVerbatim(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder) {
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	switch (slot_tag) {
	case tag::VAL_STR_HEAP: {
		auto len = size_t(StringLen(SlotPayload(slot)));
		auto s = view.StringAt(string_cursor, len);
		builder.string_heap.insert(builder.string_heap.end(), s.begin(), s.end());
		builder.slots.push_back(slot);
		string_cursor += len;
		pos++;
		return;
	}
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		auto slot_count = ExtSlotCount(subtype);
		if (slot_count > view.Slots() - pos) {
			throw InvalidInputException("malformed JSONO: extended scalar payload out of bounds");
		}
		if (subtype == ext_subtype::NUMBER) {
			auto len = size_t(NumberExtLen(slot));
			auto s = view.StringAt(string_cursor, len);
			builder.string_heap.insert(builder.string_heap.end(), s.begin(), s.end());
			builder.slots.push_back(slot);
			string_cursor += len;
			pos++;
			return;
		}
		builder.slots.push_back(slot);
		builder.slots.push_back(view.SlotAt(pos + 1));
		pos += slot_count;
		return;
	}
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		builder.slots.push_back(slot);
		pos++;
		return;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

// Push a scalar value's slot(s) verbatim WITHOUT copying its string bytes. Used by
// the merge value phase to emit a run of scalars whose string bytes are copied in
// one bulk insert (a contiguous slice of the source string_heap): per-child it
// only touches `slots` (cheap), and the heap is filled once. Caller must have
// excluded containers.
void PushScalarValueSlots(const JsonoView &view, size_t pos, JsonoBuilder &builder) {
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	switch (slot_tag) {
	case tag::VAL_STR_HEAP:
		builder.slots.push_back(slot);
		return;
	case tag::VAL_EXT: {
		auto subtype = ExtSubtype(slot);
		auto slot_count = ExtSlotCount(subtype);
		if (slot_count > view.Slots() - pos) {
			throw InvalidInputException("malformed JSONO: extended scalar payload out of bounds");
		}
		builder.slots.push_back(slot);
		if (subtype != ext_subtype::NUMBER) {
			builder.slots.push_back(view.SlotAt(pos + 1));
		}
		return;
	}
	case tag::VAL_INT60:
	case tag::VAL_DEC60:
	case tag::VAL_TRUE:
	case tag::VAL_FALSE:
	case tag::VAL_NULL:
		builder.slots.push_back(slot);
		return;
	default:
		throw InvalidInputException("malformed JSONO: non-value slot in value position");
	}
}

void EmitValueVerbatim(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		auto layout = ReadObjectLayout(view, pos);
		builder.EmitObjectStart(layout.key_count);
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
		size_t val_pos = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			builder.EmitObjectChildStart();
			EmitValueVerbatim(view, val_pos, string_cursor, builder, depth + 1);
		}
		builder.EmitObjectEnd();
		pos = layout.after_pos;
		return;
	}
	if (slot_tag == tag::ARR_START) {
		auto end_pos = ReadArrayEndPos(view, pos);
		builder.EmitArrayStart();
		pos++;
		while (pos < end_pos) {
			EmitValueVerbatim(view, pos, string_cursor, builder, depth + 1);
		}
		builder.EmitArrayEnd();
		pos = end_pos + 1;
		return;
	}
	EmitScalarVerbatim(view, pos, string_cursor, builder);
}

void EmitValueStrip(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder,
                    JsonoNullTreatment null_treatment, size_t depth);

// True if the value at `pos` keeps any content under IGNORE NULLS stripping: a non-null
// scalar or array survives; an object survives iff some member survives (recursively).
// Used to omit keys whose object value strips to empty — jsono_group_merge drops
// `key:{}`, unlike jsono_merge_patch (DeleteNulls) which keeps it. Read-only navigation;
// early-exits on the first surviving leaf so non-empty objects cost ~one probe.
bool ValueSurvivesIgnoreNulls(const JsonoView &view, size_t pos) {
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::VAL_NULL) {
		return false;
	}
	if (slot_tag != tag::OBJ_START) {
		return true;
	}
	auto layout = ReadObjectLayout(view, pos);
	size_t vp = layout.value_start;
	size_t sc = 0;
	for (size_t i = 0; i < layout.key_count; i++) {
		if (ValueSurvivesIgnoreNulls(view, vp)) {
			return true;
		}
		SkipValueFast(view, vp, sc);
	}
	return false;
}

// A member is kept if non-null and, under IGNORE NULLS, not an object that strips to
// empty. DeleteNulls keeps every non-null member (RFC 7396). The survival probe
// early-exits; EmitObjectStrip caches the result so it runs at most once per member.
inline bool MemberSurvivesStrip(const JsonoView &view, size_t vp, JsonoNullTreatment null_treatment) {
	if (SlotTag(view.SlotAt(vp)) == tag::VAL_NULL) {
		return false;
	}
	return null_treatment == JsonoNullTreatment::DeleteNulls || ValueSurvivesIgnoreNulls(view, vp);
}

void EmitObjectStrip(const JsonoView &view, size_t obj_pos, size_t obj_string_cursor, JsonoBuilder &builder,
                     JsonoNullTreatment null_treatment, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout = ReadObjectLayout(view, obj_pos);
	// Cache the per-member survival mask once. The IGNORE NULLS survival probe is
	// recursive, so recomputing it across the count/key/value passes is wasteful — and
	// with the mask the count and key passes need no slot walk at all (keys are addressed
	// by key_start + i). A stack mask avoids a per-object heap alloc; objects wider than
	// it (rare) fall back to recomputing the cheap predicate per pass.
	static constexpr size_t MASK_STACK = 128;
	char keep[MASK_STACK];
	bool cached = layout.key_count <= MASK_STACK;
	size_t count = 0;
	{
		size_t vp = layout.value_start;
		size_t sc = obj_string_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = MemberSurvivesStrip(view, vp, null_treatment);
			if (cached) {
				keep[i] = survives ? 1 : 0;
			}
			count += survives ? 1 : 0;
			SkipValueFast(view, vp, sc);
		}
	}
	builder.EmitObjectStart(count);
	if (cached) {
		for (size_t i = 0; i < layout.key_count; i++) {
			if (!keep[i]) {
				continue;
			}
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
		}
	} else {
		size_t vp = layout.value_start;
		size_t sc = obj_string_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			if (MemberSurvivesStrip(view, vp, null_treatment)) {
				auto key_slot = view.SlotAt(layout.key_start + i);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				builder.EmitKeySlot(view.KeyAt(SlotPayload(key_slot)));
			}
			SkipValueFast(view, vp, sc);
		}
	}
	{
		size_t vp = layout.value_start;
		size_t sc = obj_string_cursor;
		for (size_t i = 0; i < layout.key_count; i++) {
			bool survives = cached ? (keep[i] != 0) : MemberSurvivesStrip(view, vp, null_treatment);
			if (!survives) {
				SkipValueFast(view, vp, sc);
				continue;
			}
			builder.EmitObjectChildStart();
			EmitValueStrip(view, vp, sc, builder, null_treatment, depth + 1);
		}
	}
	builder.EmitObjectEnd();
}

void EmitValueStrip(const JsonoView &view, size_t &pos, size_t &string_cursor, JsonoBuilder &builder,
                    JsonoNullTreatment null_treatment, size_t depth) {
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		EmitObjectStrip(view, pos, string_cursor, builder, null_treatment, depth);
		auto span = CheckedContainerSpan(view, pos, tag::OBJ_START);
		pos += size_t(span.slot_span);
		string_cursor += size_t(span.string_byte_span);
		return;
	}
	if (slot_tag == tag::ARR_START) {
		EmitValueVerbatim(view, pos, string_cursor, builder, depth);
		return;
	}
	EmitScalarVerbatim(view, pos, string_cursor, builder);
}

int CompareJsonoKeys(nonstd::string_view a, nonstd::string_view b) {
	auto n = std::min(a.size(), b.size());
	if (n > 0) {
		auto c = std::memcmp(a.data(), b.data(), n);
		if (c != 0) {
			return c;
		}
	}
	if (a.size() < b.size()) {
		return -1;
	}
	if (a.size() > b.size()) {
		return 1;
	}
	return 0;
}

static constexpr uint32_t JSONO_ARRAY_CONTAINER = std::numeric_limits<uint32_t>::max();

bool IsValidJsonNumberText(nonstd::string_view text) {
	size_t i = 0;
	if (i < text.size() && text[i] == '-') {
		i++;
	}
	if (i >= text.size()) {
		return false;
	}
	if (text[i] == '0') {
		i++;
	} else if (text[i] >= '1' && text[i] <= '9') {
		do {
			i++;
		} while (i < text.size() && text[i] >= '0' && text[i] <= '9');
	} else {
		return false;
	}
	if (i < text.size() && text[i] == '.') {
		i++;
		if (i >= text.size() || text[i] < '0' || text[i] > '9') {
			return false;
		}
		do {
			i++;
		} while (i < text.size() && text[i] >= '0' && text[i] <= '9');
	}
	if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
		i++;
		if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
			i++;
		}
		if (i >= text.size() || text[i] < '0' || text[i] > '9') {
			return false;
		}
		do {
			i++;
		} while (i < text.size() && text[i] >= '0' && text[i] <= '9');
	}
	return i == text.size();
}

size_t ExpectedSkipsSize(const ContainerMetadataHeader &metadata) {
	return sizeof(ContainerMetadataHeader) + size_t(metadata.container_count) * sizeof(ContainerSpan) +
	       size_t(metadata.checkpoint_index_count) * sizeof(ObjectCheckpointIndex) +
	       size_t(metadata.checkpoint_count) * sizeof(ObjectCursorCheckpoint);
}

bool IsValidJsonoUtf8(nonstd::string_view text) {
	auto data = reinterpret_cast<const unsigned char *>(text.data());
	for (size_t i = 0; i < text.size(); i++) {
		if (data[i] & 0x80) {
			return Value::StringIsValid(text.data(), text.size());
		}
	}
	return true;
}

void ValidateScalar(const JsonoView &view, size_t &pos, size_t &string_cursor) {
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	switch (scalar.kind) {
	case JsonoScalarKind::String:
		if (!Value::StringIsValid(scalar.text.data(), scalar.text.size())) {
			throw InvalidInputException("malformed JSONO: string value is not valid UTF-8");
		}
		return;
	case JsonoScalarKind::NumberText:
		if (!IsValidJsonNumberText(scalar.text)) {
			throw InvalidInputException("malformed JSONO: raw number text is invalid");
		}
		return;
	case JsonoScalarKind::Double:
		if (!std::isfinite(scalar.double_value)) {
			throw InvalidInputException("malformed JSONO: non-finite double value");
		}
		return;
	case JsonoScalarKind::Dec60:
		if (scalar.dec_scale == 0 || !FitsDec60(scalar.dec_mantissa, scalar.dec_scale)) {
			throw InvalidInputException("malformed JSONO: invalid DEC60 payload");
		}
		return;
	case JsonoScalarKind::Null:
	case JsonoScalarKind::Bool:
	case JsonoScalarKind::Int64:
	case JsonoScalarKind::UInt64:
		return;
	}
}

void ValidateValue(const JsonoView &view, size_t &pos, size_t &string_cursor, size_t depth,
                   std::vector<uint32_t> &container_child_counts) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	if (pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: value position out of bounds");
	}
	auto slot = view.SlotAt(pos);
	auto slot_tag = SlotTag(slot);
	auto payload = SlotPayload(slot);
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto container_id = ContainerId(payload);
		if (container_id != container_child_counts.size()) {
			throw InvalidInputException("malformed JSONO: non-sequential container id");
		}
		auto layout = ReadObjectLayout(view, pos);
		container_child_counts.push_back(uint32_t(layout.key_count));
		nonstd::string_view previous_key;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			auto key = view.KeyAt(SlotPayload(key_slot));
			if (!IsValidJsonoUtf8(key)) {
				throw InvalidInputException("malformed JSONO: object key is not valid UTF-8");
			}
			if (i > 0 && CompareJsonoKeys(previous_key, key) >= 0) {
				throw InvalidInputException("malformed JSONO: object keys are not strictly sorted");
			}
			previous_key = key;
		}
		auto value_pos = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			ValidateValue(view, value_pos, string_cursor, depth + 1, container_child_counts);
		}
		if (value_pos != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		return;
	}
	case tag::ARR_START: {
		auto container_id = ContainerId(payload);
		if (container_id != container_child_counts.size()) {
			throw InvalidInputException("malformed JSONO: non-sequential container id");
		}
		if (ContainerChildCount(payload) != 0) {
			throw InvalidInputException("malformed JSONO: array child count payload must be zero");
		}
		auto end_pos = ReadArrayEndPos(view, pos);
		container_child_counts.push_back(JSONO_ARRAY_CONTAINER);
		pos++;
		while (pos < end_pos) {
			ValidateValue(view, pos, string_cursor, depth + 1, container_child_counts);
		}
		if (pos != end_pos) {
			throw InvalidInputException("malformed JSONO: array value span mismatch");
		}
		pos = end_pos + 1;
		return;
	}
	default:
		ValidateScalar(view, pos, string_cursor);
		return;
	}
}

void ValidateMetadata(const JsonoView &view, const std::vector<uint32_t> &container_child_counts) {
	auto &metadata = view.MetadataHeader();
	if (ExpectedSkipsSize(metadata) != view.SkipsSize()) {
		throw InvalidInputException("malformed JSONO: skips blob has trailing or missing bytes");
	}
	if (metadata.container_count != container_child_counts.size()) {
		throw InvalidInputException("malformed JSONO: container metadata count mismatch");
	}
	uint32_t previous_container_id = 0;
	uint32_t next_checkpoint_offset = 0;
	for (uint32_t i = 0; i < metadata.checkpoint_index_count; i++) {
		auto index = view.ObjectCheckpointIndexAt(i);
		if (index.reserved != 0) {
			throw InvalidInputException("malformed JSONO: object checkpoint index reserved field is not zero");
		}
		if (i > 0 && index.container_id <= previous_container_id) {
			throw InvalidInputException("malformed JSONO: object checkpoint index is not sorted");
		}
		if (index.container_id >= container_child_counts.size()) {
			throw InvalidInputException("malformed JSONO: object checkpoint container id out of bounds");
		}
		auto child_count = container_child_counts[index.container_id];
		if (child_count == JSONO_ARRAY_CONTAINER || child_count <= OBJECT_CHECKPOINT_STRIDE) {
			throw InvalidInputException(
			    "malformed JSONO: object checkpoint index references a non-checkpointed container");
		}
		auto expected_stride = child_count >= LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT ? LARGE_OBJECT_CHECKPOINT_STRIDE
		                                                                              : OBJECT_CHECKPOINT_STRIDE;
		if (index.checkpoint_stride != expected_stride) {
			throw InvalidInputException("malformed JSONO: object checkpoint stride mismatch");
		}
		auto checkpoint_count = (child_count - 1) / index.checkpoint_stride;
		if (index.checkpoint_offset != next_checkpoint_offset || index.checkpoint_offset > metadata.checkpoint_count ||
		    checkpoint_count > metadata.checkpoint_count - index.checkpoint_offset) {
			throw InvalidInputException("malformed JSONO: object checkpoint range mismatch");
		}
		auto span = view.ContainerSpanAt(index.container_id);
		auto value_slot_span = size_t(span.slot_span) - size_t(child_count) - 2;
		for (uint32_t checkpoint_idx = 0; checkpoint_idx < checkpoint_count; checkpoint_idx++) {
			auto checkpoint = view.ObjectCheckpointAt(index.checkpoint_offset + checkpoint_idx);
			if (checkpoint.slot_delta >= value_slot_span) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint slot out of bounds");
			}
			if (checkpoint.string_delta > span.string_byte_span) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint heap out of bounds");
			}
		}
		next_checkpoint_offset += checkpoint_count;
		previous_container_id = index.container_id;
	}
	if (next_checkpoint_offset != metadata.checkpoint_count) {
		throw InvalidInputException("malformed JSONO: unreferenced object cursor checkpoints");
	}
}

bool ValidateJsonoBlob(const JsonoBlobRow &blob) {
	try {
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			return false;
		}
		if (view.HeaderFlags() != flags::SORTED_KEYS || view.HeaderReserved() != 0) {
			return false;
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		std::vector<uint32_t> container_child_counts;
		ValidateValue(view, pos, string_cursor, 0, container_child_counts);
		if (pos != view.Slots() || string_cursor != view.StringHeapSize()) {
			return false;
		}
		ValidateMetadata(view, container_child_counts);
		return true;
	} catch (const InvalidInputException &) {
		return false;
	}
}

// Captures every child and returns the string cursor just past the last child, so
// the merge value phase can read child r's string-heap span as the gap between
// consecutive children's cursors (out[r+1].string_cursor, or this end for the last
// child) without re-decoding the value.
size_t CollectObjectChildren(const JsonoView &view, const ObjectLayout &layout, size_t obj_string_cursor,
                             std::vector<MergeChild> &out) {
	out.clear();
	out.reserve(layout.key_count);
	size_t vp = layout.value_start;
	size_t sc = obj_string_cursor;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: object key slot expected");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		auto value_slot = view.SlotAt(vp);
		out.push_back(MergeChild {key, vp, sc, SlotTag(value_slot)});
		SkipValueFastFromSlot(view, value_slot, vp, sc);
	}
	return sc;
}

void MergeTwoObjects(const JsonoView &va, size_t pos_a, size_t sc_a, const JsonoView &vb, size_t pos_b, size_t sc_b,
                     JsonoBuilder &builder, JsonoNullTreatment null_treatment, size_t depth);

// Merge two object views into the builder (B patches A), sorted-key linear merge.
void MergeTwoObjectsWithScratch(const JsonoView &va, size_t pos_a, size_t sc_a, const JsonoView &vb, size_t pos_b,
                                size_t sc_b, JsonoBuilder &builder, JsonoNullTreatment null_treatment, size_t depth,
                                std::vector<MergeChild> &children_a, std::vector<MergeChild> &children_b,
                                std::vector<MergePlanEntry> &plan) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto layout_a = ReadObjectLayout(va, pos_a);
	auto layout_b = ReadObjectLayout(vb, pos_b);
	size_t children_a_end = CollectObjectChildren(va, layout_a, sc_a, children_a);
	size_t children_b_end = CollectObjectChildren(vb, layout_b, sc_b, children_b);

	plan.clear();
	plan.reserve(children_a.size() + children_b.size());

	size_t ia = 0;
	size_t ib = 0;
	auto na = children_a.size();
	auto nb = children_b.size();
	while (ia < na || ib < nb) {
		int cmp;
		if (ib >= nb) {
			cmp = -1;
		} else if (ia >= na) {
			cmp = 1;
		} else {
			cmp = CompareJsonoKeys(children_a[ia].key, children_b[ib].key);
		}
		if (cmp < 0) {
			// A-only key. Under DeleteNulls (RFC 7396 merge_patch) A is the target
			// document, copied verbatim — its explicit nulls are values, not deletions.
			// Under IGNORE NULLS (jsono_group_merge) a null member is dropped.
			if (null_treatment == JsonoNullTreatment::DeleteNulls || children_a[ia].tag != tag::VAL_NULL) {
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, 0});
			}
			ia++;
		} else if (cmp > 0) {
			// B-only key: emit unless null, or (IGNORE NULLS) an object that strips to empty.
			if (children_b[ib].tag != tag::VAL_NULL && (null_treatment == JsonoNullTreatment::DeleteNulls ||
			                                            ValueSurvivesIgnoreNulls(vb, children_b[ib].pos))) {
				plan.push_back(MergePlanEntry {children_b[ib].key, MergeSrc::B, 0, ib});
			}
			ib++;
		} else {
			if (children_b[ib].tag != tag::VAL_NULL) {
				if (children_a[ia].tag == tag::OBJ_START && children_b[ib].tag == tag::OBJ_START) {
					// Recurse keeps A's surviving members, so the merged object is non-empty.
					plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::Recurse, ia, ib});
				} else if (null_treatment == JsonoNullTreatment::DeleteNulls ||
				           ValueSurvivesIgnoreNulls(vb, children_b[ib].pos)) {
					plan.push_back(MergePlanEntry {children_b[ib].key, MergeSrc::B, ia, ib});
				}
			} else if (null_treatment == JsonoNullTreatment::IgnoreNulls && children_a[ia].tag != tag::VAL_NULL) {
				// IGNORE NULLS: B's null must not overwrite A's accumulated value.
				plan.push_back(MergePlanEntry {children_a[ia].key, MergeSrc::A, ia, ib});
			}
			ia++;
			ib++;
		}
	}

	builder.EmitObjectStart(plan.size());
	// Key block. Output keys are a subset of one source's sorted keys, so a run of
	// plan keys that stay adjacent in that source's key_heap (no stripped key fell
	// between them) is one contiguous byte slice — copy it once and stamp each KEY
	// slot's relocated offset, instead of one key_heap insert per key. A run stays
	// in one key_heap: B-sourced keys live in vb's, the rest in va's, so the offset
	// arithmetic below stays within a single buffer.
	size_t key_plan_size = plan.size();
	size_t kk = 0;
	while (kk < key_plan_size) {
		size_t key_run_end = kk + 1;
		while (key_run_end < key_plan_size &&
		       (plan[key_run_end - 1].src == MergeSrc::B) == (plan[key_run_end].src == MergeSrc::B) &&
		       plan[key_run_end - 1].key.data() + plan[key_run_end - 1].key.size() == plan[key_run_end].key.data()) {
			key_run_end++;
		}
		const char *first_key = plan[kk].key.data();
		size_t key_base = builder.key_heap.size();
		size_t key_total = size_t(plan[key_run_end - 1].key.data() + plan[key_run_end - 1].key.size() - first_key);
		builder.key_heap.insert(builder.key_heap.end(), first_key, first_key + key_total);
		for (size_t i = kk; i < key_run_end; i++) {
			builder.PushKeySlot(key_base + size_t(plan[i].key.data() - first_key), plan[i].key.size());
		}
		kk = key_run_end;
	}
	// Value block. Consecutive scalar children from one source have their string
	// bytes in a contiguous slice of that source's string_heap (null-stripping
	// removes keys/slots but never string bytes), so a run is copied with one bulk
	// insert instead of one insert per value. A run stops at a source change, a
	// container child (emitted per-leaf), or a checkpoint-stride boundary. The
	// stride stop keeps the builder's per-child EmitObjectChildStart snapshots
	// correct: within a run only its first index can be a stride multiple, and that
	// snapshot reads the sizes the already-emitted prior run left behind.
	bool has_checkpoints;
	size_t stride;
	{
		// Capture by value: recursive child emission reallocates open_containers.
		auto &open = builder.open_containers.back();
		has_checkpoints = open.checkpoint_offset != NO_OBJECT_CHECKPOINTS;
		stride = has_checkpoints ? size_t(open.checkpoint_stride) : 0;
	}
	size_t plan_size = plan.size();
	size_t k = 0;
	while (k < plan_size) {
		auto src = plan[k].src;
		if (src == MergeSrc::Recurse) {
			builder.EmitObjectChildStart();
			MergeTwoObjects(va, children_a[plan[k].ra].pos, children_a[plan[k].ra].string_cursor, vb,
			                children_b[plan[k].rb].pos, children_b[plan[k].rb].string_cursor, builder, null_treatment,
			                depth + 1);
			k++;
			continue;
		}
		const JsonoView &vsrc = src == MergeSrc::A ? va : vb;
		auto &csrc = src == MergeSrc::A ? children_a : children_b;
		size_t csrc_end = src == MergeSrc::A ? children_a_end : children_b_end;
		size_t rank = src == MergeSrc::A ? plan[k].ra : plan[k].rb;
		auto value_tag = csrc[rank].tag;
		if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
			builder.EmitObjectChildStart();
			size_t p = csrc[rank].pos;
			size_t sc = csrc[rank].string_cursor;
			// A-side under DeleteNulls is the verbatim target (keep nested nulls); the
			// B-side patch and IGNORE NULLS still strip null members.
			if (src == MergeSrc::A && null_treatment == JsonoNullTreatment::DeleteNulls) {
				EmitValueVerbatim(vsrc, p, sc, builder, depth + 1);
			} else {
				EmitValueStrip(vsrc, p, sc, builder, null_treatment, depth + 1);
			}
			k++;
			continue;
		}
		// Extend the scalar run: same source, scalar value, before the next stride
		// boundary, and string-contiguous in the source heap. A child's string span
		// is the gap between its cursor and the next child's (csrc_end for the last),
		// so member `r` is contiguous with the run iff its cursor equals the position
		// just past the previous member. A base member stripped for being null leaves
		// no string bytes, so the slice spans it; a member stripped by a patch null
		// can carry bytes (string/number/object), which break the run there.
		size_t boundary = has_checkpoints ? (k / stride + 1) * stride : plan_size;
		size_t first_string_cursor = csrc[rank].string_cursor;
		size_t last_rank = rank;
		size_t run_end = k + 1;
		while (run_end < boundary && run_end < plan_size && plan[run_end].src == src) {
			size_t run_rank = src == MergeSrc::A ? plan[run_end].ra : plan[run_end].rb;
			auto run_tag = csrc[run_rank].tag;
			if (run_tag == tag::OBJ_START || run_tag == tag::ARR_START) {
				break;
			}
			size_t after_last = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].string_cursor : csrc_end;
			if (csrc[run_rank].string_cursor != after_last) {
				break;
			}
			last_rank = run_rank;
			run_end++;
		}
		// Advance the per-child cursor for the whole run first (only index k can hit
		// a stride boundary, and its snapshot reads the prior run's emitted sizes).
		for (size_t i = k; i < run_end; i++) {
			builder.EmitObjectChildStart();
		}
		for (size_t i = k; i < run_end; i++) {
			size_t run_rank = src == MergeSrc::A ? plan[i].ra : plan[i].rb;
			PushScalarValueSlots(vsrc, csrc[run_rank].pos, builder);
		}
		size_t run_string_end = last_rank + 1 < csrc.size() ? csrc[last_rank + 1].string_cursor : csrc_end;
		if (run_string_end > first_string_cursor) {
			auto slice = vsrc.StringAt(first_string_cursor, run_string_end - first_string_cursor);
			builder.string_heap.insert(builder.string_heap.end(), slice.begin(), slice.end());
		}
		k = run_end;
	}
	builder.EmitObjectEnd();
}

void MergeTwoObjects(const JsonoView &va, size_t pos_a, size_t sc_a, const JsonoView &vb, size_t pos_b, size_t sc_b,
                     JsonoBuilder &builder, JsonoNullTreatment null_treatment, size_t depth) {
	std::vector<MergeChild> children_a;
	std::vector<MergeChild> children_b;
	std::vector<MergePlanEntry> plan;
	MergeTwoObjectsWithScratch(va, pos_a, sc_a, vb, pos_b, sc_b, builder, null_treatment, depth, children_a, children_b,
	                           plan);
}

// A self-owned JSONO blob: the in-flight accumulator for the fold-based scalar
// jsono_merge_patch and the jsono_group_merge aggregate. Each fold runs the
// cursor-based MergeTwoObjects over contiguous bytes (no per-key std::string, no
// per-node heap alloc, keys compared in cache order) and re-serializes the merged
// builder back into the blob so the next fold can view it.
struct OwnedJsonoBlob {
	std::string slots;
	std::string key_heap;
	std::string string_heap;
	std::string skips;
};

void SerializeBuilderToBlob(const JsonoBuilder &builder, OwnedJsonoBlob &out) {
	auto slots_bytes = builder.slots.size() * sizeof(uint64_t);
	out.slots.resize(JSONO_HEADER_SIZE + slots_bytes);
	JsonoHeader header;
	header.magic = MAGIC;
	header.version = VERSION;
	header.flags = flags::SORTED_KEYS;
	header.reserved = 0;
	std::memcpy(&out.slots[0], &header, JSONO_HEADER_SIZE);
	if (slots_bytes > 0) {
		std::memcpy(&out.slots[JSONO_HEADER_SIZE], builder.slots.data(), slots_bytes);
	}
	out.key_heap.assign(builder.key_heap.data(), builder.key_heap.size());
	out.string_heap.assign(builder.string_heap.data(), builder.string_heap.size());

	auto spans_bytes = builder.skips.size() * sizeof(ContainerSpan);
	auto index_bytes = builder.object_checkpoint_index.size() * sizeof(ObjectCheckpointIndex);
	auto checkpoints_bytes = builder.object_checkpoints.size() * sizeof(ObjectCursorCheckpoint);
	out.skips.resize(sizeof(ContainerMetadataHeader) + spans_bytes + index_bytes + checkpoints_bytes);
	ContainerMetadataHeader meta;
	meta.container_count = uint32_t(builder.skips.size());
	meta.checkpoint_index_count = uint32_t(builder.object_checkpoint_index.size());
	meta.checkpoint_count = uint32_t(builder.object_checkpoints.size());
	size_t off = 0;
	std::memcpy(&out.skips[off], &meta, sizeof(meta));
	off += sizeof(meta);
	if (spans_bytes > 0) {
		std::memcpy(&out.skips[off], builder.skips.data(), spans_bytes);
		off += spans_bytes;
	}
	if (index_bytes > 0) {
		std::memcpy(&out.skips[off], builder.object_checkpoint_index.data(), index_bytes);
		off += index_bytes;
	}
	if (checkpoints_bytes > 0) {
		std::memcpy(&out.skips[off], builder.object_checkpoints.data(), checkpoints_bytes);
	}
}

JsonoView ViewOfBlob(const OwnedJsonoBlob &b) {
	return JsonoView(b.slots.data(), b.slots.size(), b.key_heap.data(), b.key_heap.size(), b.string_heap.data(),
	                 b.string_heap.size(), b.skips.data(), b.skips.size());
}

// One RFC 7396 fold step. Returns true when the result is SQL NULL (a SQL NULL
// patch argument replaces the result with SQL NULL). Otherwise the merged value is
// written into `builder`: an object patch deep-merges onto an object accumulator
// (the accumulator is the target — its nulls are kept; the patch's nulls delete
// keys), while onto a non-object or SQL NULL accumulator it is merged onto an empty
// object (its own nulls stripped). A non-object patch replaces the accumulator.
bool MergePatchFoldStep(bool acc_is_sqlnull, const JsonoView &acc_view, bool patch_present, const JsonoView &patch_view,
                        JsonoBuilder &builder, std::vector<MergeChild> &children_a, std::vector<MergeChild> &children_b,
                        std::vector<MergePlanEntry> &plan) {
	if (!patch_present) {
		return true;
	}
	builder.Reset();
	size_t p = 0;
	size_t sc = 0;
	if (SlotTag(patch_view.SlotAt(0)) != tag::OBJ_START) {
		EmitValueVerbatim(patch_view, p, sc, builder, 0);
		return false;
	}
	if (!acc_is_sqlnull && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START) {
		MergeTwoObjectsWithScratch(acc_view, 0, 0, patch_view, 0, 0, builder, JsonoNullTreatment::DeleteNulls, 0,
		                           children_a, children_b, plan);
	} else {
		EmitValueStrip(patch_view, p, sc, builder, JsonoNullTreatment::DeleteNulls, 0);
	}
	return false;
}

void EnsureListCapacity(Vector &result, idx_t needed) {
	if (needed <= ListVector::GetListCapacity(result)) {
		return;
	}
	auto next = std::max<idx_t>(needed, std::max<idx_t>(ListVector::GetListCapacity(result) * 2, 1));
	ListVector::Reserve(result, next);
}

void AppendListString(Vector &result, idx_t &length, const char *data, size_t size) {
	auto child_row = ListVector::GetListSize(result) + length;
	EnsureListCapacity(result, child_row + 1);
	auto &child = ListVector::GetEntry(result);
	FlatVector::GetData<string_t>(child)[child_row] = StringVector::AddString(child, data, size);
	length++;
}

void AppendListNull(Vector &result, idx_t &length) {
	auto child_row = ListVector::GetListSize(result) + length;
	EnsureListCapacity(result, child_row + 1);
	auto &child = ListVector::GetEntry(result);
	FlatVector::SetNull(child, child_row, true);
	length++;
}

void FinishListRow(Vector &result, idx_t row, idx_t start_offset, idx_t length) {
	ListVector::GetData(result)[row] = {start_offset, length};
	ListVector::SetListSize(result, start_offset + length);
}

void SetListRowNull(Vector &result, idx_t row) {
	FlatVector::SetNull(result, row, true);
	ListVector::GetData(result)[row] = {0, 0};
}

unique_ptr<FunctionData> JsonoMergePatchBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	(void)context;
	if (arguments.empty()) {
		throw BinderException("jsono_merge_patch() requires at least one argument");
	}
	for (auto &argument : arguments) {
		if (argument->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (argument->return_type.id() != LogicalTypeId::SQLNULL && !IsJsonoType(argument->return_type)) {
			throw BinderException("jsono_merge_patch() arguments must be JSONO");
		}
		bound_function.arguments.push_back(JsonoType());
	}
	return nullptr;
}

void JsonoMergePatchExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoOpsLocalState>();
	auto count = args.size();
	vector<JsonoVectorData> inputs(args.ColumnCount());
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		InitJsonoVectorData(args.data[i], count, inputs[i]);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	auto &slots_vec = *children[0];
	auto &key_heap_vec = *children[1];
	auto &string_heap_vec = *children[2];
	auto &skips_vec = *children[3];
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);

	idx_t ncols = args.ColumnCount();
	// jsono_merge_patch folds left to right (RFC 7396): the first argument is the
	// target document (kept verbatim — its nulls are values), the rest are patches.
	// Every step reuses the cursor MergeTwoObjects via MergePatchFoldStep, serializing
	// the running accumulator into a reused blob between steps; the dominant two-object
	// case runs a single merge straight into the output builder with no round-trip.
	static thread_local OwnedJsonoBlob acc_storage;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow target_blob {};
		bool acc_is_sqlnull = !ReadJsonoRow(inputs[0], row, target_blob);
		JsonoView acc_view = MakeJsonoView(target_blob);
		if (!acc_is_sqlnull && (!acc_view.ParseHeader() || acc_view.Slots() == 0)) {
			acc_is_sqlnull = true;
		}
		if (ncols == 1) {
			// A lone target is returned verbatim (its nulls survive).
			if (acc_is_sqlnull) {
				SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
			} else {
				lstate.builder.Reset();
				size_t p = 0;
				size_t sc = 0;
				EmitValueVerbatim(acc_view, p, sc, lstate.builder, 0);
				WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
				                 string_heap_out, skips_out, row, lstate.builder);
			}
			continue;
		}
		for (idx_t i = 1; i < ncols; i++) {
			JsonoBlobRow patch_blob {};
			bool patch_present = ReadJsonoRow(inputs[i], row, patch_blob);
			JsonoView patch_view = MakeJsonoView(patch_blob);
			if (patch_present && (!patch_view.ParseHeader() || patch_view.Slots() == 0)) {
				patch_present = false;
			}
			bool result_sqlnull =
			    MergePatchFoldStep(acc_is_sqlnull, acc_view, patch_present, patch_view, lstate.builder,
			                       lstate.merge_children_a, lstate.merge_children_b, lstate.merge_plan);
			if (i + 1 == ncols) {
				// Last step writes straight to the output builder — no serialize round-trip.
				if (result_sqlnull) {
					SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
				} else {
					WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
					                 string_heap_out, skips_out, row, lstate.builder);
				}
			} else if (result_sqlnull) {
				acc_is_sqlnull = true;
			} else {
				SerializeBuilderToBlob(lstate.builder, acc_storage);
				acc_view = ViewOfBlob(acc_storage);
				acc_view.ParseHeader();
				acc_is_sqlnull = false;
			}
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Walk from the document root to the slot addressed by `steps`, tracking the
// string heap cursor so nested string payloads resolve correctly. Returns false
// when a step misses (absent key, out-of-range index, or stepping into a
// non-container). Wildcard steps are rejected at bind, so they never reach here.
bool NavigateToPath(const JsonoView &view, const vector<PathStep> &steps, size_t &pos, size_t &string_cursor) {
	pos = 0;
	string_cursor = 0;
	for (auto &step : steps) {
		if (pos >= view.Slots()) {
			return false;
		}
		auto slot_tag = SlotTag(view.SlotAt(pos));
		if (step.kind == PathStepKind::Key) {
			if (slot_tag != tag::OBJ_START) {
				return false;
			}
			auto layout = ReadObjectLayout(view, pos);
			nonstd::string_view target(step.key);
			size_t lo = 0;
			size_t hi = layout.key_count;
			while (lo < hi) {
				auto mid = lo + (hi - lo) / 2;
				auto key_slot = view.SlotAt(layout.key_start + mid);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: object key slot expected");
				}
				if (view.KeyAt(SlotPayload(key_slot)) < target) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			if (lo >= layout.key_count || view.KeyAt(SlotPayload(view.SlotAt(layout.key_start + lo))) != target) {
				return false;
			}
			size_t value_pos = layout.key_start + layout.key_count;
			size_t value_string_cursor = string_cursor;
			for (size_t i = 0; i < lo; i++) {
				SkipValueFast(view, value_pos, value_string_cursor);
			}
			pos = value_pos;
			string_cursor = value_string_cursor;
		} else if (step.kind == PathStepKind::Index) {
			if (slot_tag != tag::ARR_START) {
				return false;
			}
			auto end_pos = ReadArrayEndPos(view, pos);
			size_t elem_pos = pos + 1;
			idx_t current = 0;
			while (elem_pos < end_pos && current < step.index) {
				SkipValueFast(view, elem_pos, string_cursor);
				current++;
			}
			if (elem_pos >= end_pos || current < step.index) {
				return false;
			}
			pos = elem_pos;
		} else {
			return false;
		}
	}
	return true;
}

struct JsonoPathBindData : public FunctionData {
	explicit JsonoPathBindData(vector<PathStep> steps_p) : steps(std::move(steps_p)) {
	}
	vector<PathStep> steps;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoPathBindData>(steps);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoPathBindData>();
		if (steps.size() != other.steps.size()) {
			return false;
		}
		for (idx_t i = 0; i < steps.size(); i++) {
			if (steps[i].kind != other.steps[i].kind || steps[i].index != other.steps[i].index ||
			    steps[i].key != other.steps[i].key) {
				return false;
			}
		}
		return true;
	}
};

unique_ptr<FunctionData> JsonoPathBind(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                       const char *function_name) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("%s: path must be constant", function_name);
	}
	auto path_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (path_value.IsNull()) {
		throw BinderException("%s: path must not be NULL", function_name);
	}
	auto steps = ParseJsonoPath(StringValue::Get(path_value), function_name);
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("%s: wildcard paths are not supported", function_name);
		}
	}
	return make_uniq<JsonoPathBindData>(std::move(steps));
}

unique_ptr<FunctionData> JsonoTypePathBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	(void)bound_function;
	return JsonoPathBind(context, arguments, "jsono_type");
}

unique_ptr<FunctionData> JsonoKeysPathBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	(void)bound_function;
	return JsonoPathBind(context, arguments, "jsono_keys");
}

const vector<PathStep> *PathStepsFromState(ExpressionState &state, idx_t column_count) {
	if (column_count < 2) {
		return nullptr;
	}
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	return &func_expr.bind_info->Cast<JsonoPathBindData>().steps;
}

const char *JsonoTypeName(const JsonoView &view, size_t pos, size_t string_cursor) {
	if (pos >= view.Slots()) {
		return nullptr;
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	if (slot_tag == tag::OBJ_START) {
		return "OBJECT";
	}
	if (slot_tag == tag::ARR_START) {
		return "ARRAY";
	}
	auto scalar = DecodeScalarAt(view, pos, string_cursor);
	return JsonoScalarTypeName(scalar.kind);
}

void JsonoTypeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		if (steps && !NavigateToPath(view, *steps, pos, string_cursor)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto type_name = JsonoTypeName(view, pos, string_cursor);
		if (!type_name) {
			FlatVector::SetNull(result, row, true);
		} else {
			result_data[row] = StringVector::AddString(result, type_name);
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoKeysExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);
	auto steps = PathStepsFromState(state, args.ColumnCount());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetListRowNull(result, row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetListRowNull(result, row);
			continue;
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		if (steps && !NavigateToPath(view, *steps, pos, string_cursor)) {
			SetListRowNull(result, row);
			continue;
		}
		if (SlotTag(view.SlotAt(pos)) != tag::OBJ_START) {
			SetListRowNull(result, row);
			continue;
		}
		auto layout = ReadObjectLayout(view, pos);
		auto start = ListVector::GetListSize(result);
		idx_t length = 0;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			auto key = view.KeyAt(SlotPayload(key_slot));
			AppendListString(result, length, key.data(), key.size());
		}
		FinishListRow(result, row, start, length);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

enum class JsonoEntriesKeyStyle : uint8_t { JsonPath, Dotted };

struct JsonoEntriesBindData : public FunctionData {
	explicit JsonoEntriesBindData(JsonoEntriesKeyStyle style_p) : style(style_p) {
	}
	JsonoEntriesKeyStyle style;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<JsonoEntriesBindData>(style);
	}
	bool Equals(const FunctionData &other_p) const override {
		return style == other_p.Cast<JsonoEntriesBindData>().style;
	}
};

unique_ptr<FunctionData> JsonoEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	(void)bound_function;
	auto style = JsonoEntriesKeyStyle::JsonPath;
	if (arguments.size() >= 2) {
		auto &style_arg = arguments[1];
		if (style_arg->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!style_arg->IsFoldable()) {
			throw BinderException("jsono_entries: key_style must be constant");
		}
		auto style_value = ExpressionExecutor::EvaluateScalar(context, *style_arg);
		if (style_value.IsNull()) {
			throw BinderException("jsono_entries: key_style must not be NULL");
		}
		auto name = StringValue::Get(style_value);
		StringUtil::Trim(name);
		name = StringUtil::Lower(name);
		if (name == "jsonpath") {
			style = JsonoEntriesKeyStyle::JsonPath;
		} else if (name == "dotted") {
			style = JsonoEntriesKeyStyle::Dotted;
		} else {
			throw BinderException("jsono_entries: key_style must be 'jsonpath' or 'dotted'");
		}
	}
	return make_uniq<JsonoEntriesBindData>(style);
}

// A bare JSONPath key segment is parseable only while it avoids the grammar's
// delimiters; anything else (or an empty key) must be quoted.
bool JsonPathKeyNeedsQuote(nonstd::string_view key) {
	if (key.empty()) {
		return true;
	}
	for (char c : key) {
		if (c == '.' || c == '[' || c == ']' || c == '"') {
			return true;
		}
	}
	return false;
}

void AppendKeySegment(std::string &path, nonstd::string_view key, JsonoEntriesKeyStyle style) {
	if (style == JsonoEntriesKeyStyle::JsonPath) {
		path.push_back('.');
		if (JsonPathKeyNeedsQuote(key)) {
			path.push_back('"');
			for (char c : key) {
				if (c == '"' || c == '\\') {
					path.push_back('\\');
				}
				path.push_back(c);
			}
			path.push_back('"');
		} else {
			path.append(key.data(), key.size());
		}
		return;
	}
	if (!path.empty()) {
		path.push_back('.');
	}
	path.append(key.data(), key.size());
}

void AppendIndexSegment(std::string &path, idx_t index, JsonoEntriesKeyStyle style) {
	if (style == JsonoEntriesKeyStyle::JsonPath) {
		path.push_back('[');
		path.append(std::to_string(index));
		path.push_back(']');
		return;
	}
	if (!path.empty()) {
		path.push_back('.');
	}
	path.append(std::to_string(index));
}

// Render a leaf scalar as bare VARCHAR (no JSON quoting). Returns true for JSON
// null, where the caller emits a SQL NULL value rather than the text "null".
bool JsonoScalarToText(const JsonoScalar &scalar, std::string &out) {
	switch (scalar.kind) {
	case JsonoScalarKind::Null:
		return true;
	case JsonoScalarKind::String:
	case JsonoScalarKind::NumberText:
		out.assign(scalar.text.data(), scalar.text.size());
		return false;
	case JsonoScalarKind::Int64:
		out = std::to_string(scalar.int_value);
		return false;
	case JsonoScalarKind::UInt64:
		out = std::to_string(scalar.uint_value);
		return false;
	case JsonoScalarKind::Double:
		EmitDouble(scalar.double_value, out);
		return false;
	case JsonoScalarKind::Dec60:
		AppendDec60Text(out, scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
		return false;
	case JsonoScalarKind::Bool:
		out = scalar.bool_value ? "true" : "false";
		return false;
	}
	return true;
}

// Output sink for jsono_entries. Resolves the LIST child (key/value) vectors once
// per execute instead of per leaf — the per-leaf GetEntry/GetEntries re-resolution
// dominated the profile. `next_child` is the running write cursor into the shared
// child vectors; data pointers are refreshed only when a capacity grow moves them.
// A single reused `scratch` renders non-text scalars, and text scalars are copied
// straight from the heap (no intermediate std::string per leaf).
struct EntriesSink {
	Vector &list;
	Vector *key_vec;
	Vector *value_vec;
	string_t *key_data;
	string_t *value_data;
	idx_t capacity;
	idx_t next_child;
	std::string scratch;

	explicit EntriesSink(Vector &list_p) : list(list_p), next_child(ListVector::GetListSize(list_p)) {
		auto &struct_entries = StructVector::GetEntries(ListVector::GetEntry(list_p));
		key_vec = struct_entries[0].get();
		value_vec = struct_entries[1].get();
		capacity = ListVector::GetListCapacity(list_p);
		key_data = FlatVector::GetData<string_t>(*key_vec);
		value_data = FlatVector::GetData<string_t>(*value_vec);
	}

	void Append(nonstd::string_view key, const JsonoScalar &scalar) {
		if (next_child >= capacity) {
			EnsureListCapacity(list, next_child + 1);
			capacity = ListVector::GetListCapacity(list);
			key_data = FlatVector::GetData<string_t>(*key_vec);
			value_data = FlatVector::GetData<string_t>(*value_vec);
		}
		auto child_row = next_child;
		key_data[child_row] = StringVector::AddString(*key_vec, key.data(), key.size());
		switch (scalar.kind) {
		case JsonoScalarKind::Null:
			FlatVector::SetNull(*value_vec, child_row, true);
			break;
		case JsonoScalarKind::String:
		case JsonoScalarKind::NumberText:
			// Already text in the heap — copy straight in, no intermediate std::string.
			value_data[child_row] = StringVector::AddString(*value_vec, scalar.text.data(), scalar.text.size());
			break;
		default:
			scratch.clear();
			JsonoScalarToText(scalar, scratch);
			value_data[child_row] = StringVector::AddString(*value_vec, scratch.data(), scratch.size());
			break;
		}
		next_child++;
	}
};

// Depth-first walk that materializes one (path, value) struct per leaf. Object
// values advance a value cursor while the shared string cursor consumes heap bytes
// in walk order.
void WalkEntries(const JsonoView &view, size_t &pos, size_t &string_cursor, std::string &path,
                 JsonoEntriesKeyStyle style, EntriesSink &sink, size_t depth) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	auto slot_tag = SlotTag(view.SlotAt(pos));
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto layout = ReadObjectLayout(view, pos);
		auto val_cursor = layout.value_start;
		for (size_t i = 0; i < layout.key_count; i++) {
			auto key_slot = view.SlotAt(layout.key_start + i);
			if (SlotTag(key_slot) != tag::KEY) {
				throw InvalidInputException("malformed JSONO: object key slot expected");
			}
			auto key = view.KeyAt(SlotPayload(key_slot));
			auto saved = path.size();
			AppendKeySegment(path, key, style);
			WalkEntries(view, val_cursor, string_cursor, path, style, sink, depth + 1);
			path.resize(saved);
		}
		if (val_cursor != layout.after_pos - 1) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		pos = layout.after_pos;
		return;
	}
	case tag::ARR_START: {
		auto end_pos = ReadArrayEndPos(view, pos);
		pos++;
		idx_t index = 0;
		while (pos < end_pos) {
			auto saved = path.size();
			AppendIndexSegment(path, index, style);
			WalkEntries(view, pos, string_cursor, path, style, sink, depth + 1);
			path.resize(saved);
			index++;
		}
		pos = end_pos + 1;
		return;
	}
	default: {
		auto scalar = DecodeScalarAt(view, pos, string_cursor);
		sink.Append(path, scalar);
		return;
	}
	}
}

void JsonoEntriesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto style = func_expr.bind_info->Cast<JsonoEntriesBindData>().style;

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	std::string path;
	EntriesSink sink(result);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			SetListRowNull(result, row);
			continue;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			SetListRowNull(result, row);
			continue;
		}
		auto start = sink.next_child;
		path.clear();
		if (style == JsonoEntriesKeyStyle::JsonPath) {
			path.push_back('$');
		}
		size_t pos = 0;
		size_t string_cursor = 0;
		WalkEntries(view, pos, string_cursor, path, style, sink, 0);
		FinishListRow(result, row, start, sink.next_child - start);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoValidateExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		result_data[row] = ValidateJsonoBlob(blob);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoStorageSizeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	auto slots_data = FlatVector::GetData<uint64_t>(*children[0]);
	auto key_heap_data = FlatVector::GetData<uint64_t>(*children[1]);
	auto string_heap_data = FlatVector::GetData<uint64_t>(*children[2]);
	auto skips_data = FlatVector::GetData<uint64_t>(*children[3]);
	auto total_data = FlatVector::GetData<uint64_t>(*children[4]);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			FlatVector::SetNull(result, row, true);
			for (auto &child : children) {
				FlatVector::SetNull(*child, row, true);
			}
			continue;
		}
		slots_data[row] = blob.slots.GetSize();
		key_heap_data[row] = blob.key_heap.GetSize();
		string_heap_data[row] = blob.string_heap.GetSize();
		skips_data[row] = blob.skips.GetSize();
		total_data[row] = slots_data[row] + key_heap_data[row] + string_heap_data[row] + skips_data[row];
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

struct GroupMergeBindData : public FunctionData {
	JsonoNullTreatment null_treatment;

	explicit GroupMergeBindData(JsonoNullTreatment null_treatment) : null_treatment(null_treatment) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<GroupMergeBindData>(null_treatment);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<GroupMergeBindData>();
		return null_treatment == other.null_treatment;
	}
};

struct GroupMergeState {
	OwnedJsonoBlob *acc;
	bool has_input;
};

struct GroupMergeFunction {
	static void Initialize(GroupMergeState &state) {
		state.acc = nullptr;
		state.has_input = false;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.acc;
		state.acc = nullptr;
		state.has_input = false;
	}
};

string NormalizeGroupMergeMode(string mode) {
	StringUtil::Trim(mode);
	return StringUtil::Lower(mode);
}

unique_ptr<FunctionData> JsonoGroupMergeBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	(void)function;
	if (arguments.size() != 2) {
		throw BinderException("jsono_group_merge() requires JSONO plus constant 'IGNORE NULLS'");
	}
	if (arguments[0]->return_type.id() != LogicalTypeId::SQLNULL && !IsJsonoType(arguments[0]->return_type)) {
		throw BinderException("jsono_group_merge() input must be JSONO");
	}
	auto &mode_arg = arguments[1];
	if (mode_arg->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!mode_arg->IsFoldable()) {
		throw BinderException("jsono_group_merge() null handling mode must be constant");
	}
	auto mode_value = ExpressionExecutor::EvaluateScalar(context, *mode_arg);
	if (mode_value.IsNull() || mode_value.type().id() != LogicalTypeId::VARCHAR ||
	    NormalizeGroupMergeMode(StringValue::Get(mode_value)) != "ignore nulls") {
		throw InvalidInputException("jsono_group_merge(): null handling mode must be IGNORE NULLS");
	}
	return make_uniq<GroupMergeBindData>(JsonoNullTreatment::IgnoreNulls);
}

// Fold one incoming value (a row blob, or a partial's accumulated blob) into the
// group state, matching jsono_group_merge IGNORE NULLS semantics: object+object →
// cursor merge (incoming wins per key, its nulls ignored, A kept on B-null, keys whose
// object value strips to empty omitted); a fresh object (no prior object accumulator)
// is seeded with nulls/empty-objects stripped; a non-object incoming replaces the
// accumulator wholesale (verbatim, including a top-level null) — the same rule
// jsono_merge_patch applies to a non-object patch. The merged builder is
// re-serialized so the next fold can view the accumulator.
void FoldIntoGroupState(GroupMergeState &state, const JsonoView &incoming) {
	static thread_local JsonoBuilder scratch;
	scratch.Reset();
	size_t pos = 0;
	size_t sc = 0;
	bool incoming_is_object = SlotTag(incoming.SlotAt(0)) == tag::OBJ_START;
	bool merged_objects = false;
	if (incoming_is_object && state.acc) {
		JsonoView acc_view = ViewOfBlob(*state.acc);
		if (acc_view.ParseHeader() && acc_view.Slots() > 0 && SlotTag(acc_view.SlotAt(0)) == tag::OBJ_START) {
			MergeTwoObjects(acc_view, 0, 0, incoming, 0, 0, scratch, JsonoNullTreatment::IgnoreNulls, 0);
			merged_objects = true;
		}
	}
	if (!merged_objects) {
		if (incoming_is_object) {
			EmitValueStrip(incoming, pos, sc, scratch, JsonoNullTreatment::IgnoreNulls, 0);
		} else {
			EmitValueVerbatim(incoming, pos, sc, scratch, 0);
		}
	}
	if (!state.acc) {
		state.acc = new OwnedJsonoBlob();
	}
	SerializeBuilderToBlob(scratch, *state.acc);
	state.has_input = true;
}

void ApplyGroupMergeFromBlob(GroupMergeState &state, const JsonoBlobRow &blob) {
	JsonoView view = MakeJsonoView(blob);
	if (!view.ParseHeader() || view.Slots() == 0) {
		return;
	}
	FoldIntoGroupState(state, view);
}

void JsonoGroupMergeSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 data_ptr_t state_ptr, idx_t count) {
	(void)aggr_input_data;
	(void)input_count;
	JsonoVectorData input;
	InitJsonoVectorData(inputs[0], count, input);
	auto &state = *reinterpret_cast<GroupMergeState *>(state_ptr);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			continue;
		}
		ApplyGroupMergeFromBlob(state, blob);
	}
}

void JsonoGroupMergeUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                           idx_t count) {
	(void)aggr_input_data;
	(void)input_count;
	JsonoVectorData input;
	InitJsonoVectorData(inputs[0], count, input);
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRow(input, row, blob)) {
			continue;
		}
		auto state_idx = RowIndex(state_fmt, row);
		ApplyGroupMergeFromBlob(*state_data[state_idx], blob);
	}
}

void JsonoGroupMergeCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	(void)aggr_input_data;
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<GroupMergeState *>(source_fmt);
	auto target_data = FlatVector::GetData<GroupMergeState *>(target);

	for (idx_t row = 0; row < count; row++) {
		auto &source_state = *source_data[RowIndex(source_fmt, row)];
		if (!source_state.has_input || !source_state.acc) {
			continue;
		}
		JsonoView source_view = ViewOfBlob(*source_state.acc);
		if (!source_view.ParseHeader() || source_view.Slots() == 0) {
			continue;
		}
		FoldIntoGroupState(*target_data[row], source_view);
	}
}

void JsonoGroupMergeFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	(void)aggr_input_data;
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<GroupMergeState *>(state_fmt);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	auto &slots_vec = *children[0];
	auto &key_heap_vec = *children[1];
	auto &string_heap_vec = *children[2];
	auto &skips_vec = *children[3];
	auto slots_out = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_out = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_out = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_out = FlatVector::GetData<string_t>(skips_vec);
	JsonoBuilder empty_builder;

	for (idx_t i = 0; i < count; i++) {
		auto rid = i + offset;
		auto &state = *state_data[RowIndex(state_fmt, i)];
		if (!state.has_input || !state.acc) {
			empty_builder.Reset();
			empty_builder.EmitObjectStart(0);
			empty_builder.EmitObjectEnd();
			WriteJsonoStruct(slots_vec, key_heap_vec, string_heap_vec, skips_vec, slots_out, key_heap_out,
			                 string_heap_out, skips_out, rid, empty_builder);
			continue;
		}
		// The accumulator is already a serialized blob; copy each component verbatim.
		slots_out[rid] = WriteBlobInto(slots_vec, state.acc->slots.data(), state.acc->slots.size());
		key_heap_out[rid] = WriteBlobInto(key_heap_vec, state.acc->key_heap.data(), state.acc->key_heap.size());
		string_heap_out[rid] =
		    WriteBlobInto(string_heap_vec, state.acc->string_heap.data(), state.acc->string_heap.size());
		skips_out[rid] = WriteBlobInto(skips_vec, state.acc->skips.data(), state.acc->skips.size());
	}
}

} // namespace

void RegisterJsonoOps(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{
		ScalarFunctionSet set("jsono_type");
		set.AddFunction(ScalarFunction({jsono_type}, LogicalType::VARCHAR, JsonoTypeExecute));
		set.AddFunction(ScalarFunction({jsono_type, LogicalType::VARCHAR}, LogicalType::VARCHAR, JsonoTypeExecute,
		                               JsonoTypePathBind));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_keys");
		set.AddFunction(ScalarFunction({jsono_type}, LogicalType::LIST(LogicalType::VARCHAR), JsonoKeysExecute));
		set.AddFunction(ScalarFunction({jsono_type, LogicalType::VARCHAR}, LogicalType::LIST(LogicalType::VARCHAR),
		                               JsonoKeysExecute, JsonoKeysPathBind));
		loader.RegisterFunction(set);
	}
	{
		auto entry_type =
		    LogicalType::LIST(LogicalType::STRUCT({{"key", LogicalType::VARCHAR}, {"value", LogicalType::VARCHAR}}));
		ScalarFunctionSet set("jsono_entries");
		set.AddFunction(ScalarFunction({jsono_type}, entry_type, JsonoEntriesExecute, JsonoEntriesBind));
		set.AddFunction(
		    ScalarFunction({jsono_type, LogicalType::VARCHAR}, entry_type, JsonoEntriesExecute, JsonoEntriesBind));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunction fun("jsono_validate", {jsono_type}, LogicalType::BOOLEAN, JsonoValidateExecute);
		loader.RegisterFunction(fun);
	}
	{
		auto storage_size_type = LogicalType::STRUCT({{"jsono_slots", LogicalType::UBIGINT},
		                                              {"jsono_key_heap", LogicalType::UBIGINT},
		                                              {"jsono_string_heap", LogicalType::UBIGINT},
		                                              {"jsono_skips", LogicalType::UBIGINT},
		                                              {"total", LogicalType::UBIGINT}});
		ScalarFunction fun("jsono_storage_size", {jsono_type}, storage_size_type, JsonoStorageSizeExecute);
		loader.RegisterFunction(fun);
	}
	{
		ScalarFunction fun("jsono_merge_patch", {}, jsono_type, JsonoMergePatchExecute, JsonoMergePatchBind, nullptr,
		                   nullptr, JsonoOpsLocalState::Init);
		fun.varargs = LogicalType::ANY;
		fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		loader.RegisterFunction(fun);
	}
	{
		AggregateFunction fun("jsono_group_merge", {jsono_type, LogicalType::VARCHAR}, jsono_type,
		                      AggregateFunction::StateSize<GroupMergeState>,
		                      AggregateFunction::StateInitialize<GroupMergeState, GroupMergeFunction>,
		                      JsonoGroupMergeUpdate, JsonoGroupMergeCombine, JsonoGroupMergeFinalize,
		                      FunctionNullHandling::DEFAULT_NULL_HANDLING, JsonoGroupMergeSimpleUpdate,
		                      JsonoGroupMergeBind,
		                      AggregateFunction::StateDestroy<GroupMergeState, GroupMergeFunction>);
		fun.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		loader.RegisterFunction(std::move(fun));
	}
}

} // namespace duckdb
