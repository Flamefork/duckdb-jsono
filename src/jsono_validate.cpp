#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

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
	return sizeof(ContainerMetadataHeader) + size_t(metadata.span_count) * (sizeof(uint32_t) + sizeof(ContainerSpan)) +
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

void ValidateScalar(const JsonoView &view, JsonoCursor &cursor) {
	auto scalar = DecodeScalarAt(view, cursor);
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

enum class ContainerSpanRequirement : uint8_t { Forbidden, Optional, Required };

// Validate one stored container span against the cursor deltas the walk actually
// produced. Spans are required for non-empty arrays and large objects; legacy small
// objects with container children and empty arrays may still carry optional spans.
// A spanned object additionally stores the shape_hash of its sorted key sequence.
void ValidateContainerSpan(const JsonoView &view, uint64_t container_id, ContainerSpanRequirement requirement,
                           uint64_t expected_hash, const JsonoCursor &start, const JsonoCursor &end,
                           size_t &validated_span_count) {
	ContainerSpan span;
	bool has_span = view.TryContainerSpan(container_id, span);
	if (has_span && requirement == ContainerSpanRequirement::Forbidden) {
		throw InvalidInputException("malformed JSONO: unexpected container span");
	}
	if (!has_span && requirement == ContainerSpanRequirement::Required) {
		throw InvalidInputException("malformed JSONO: container span missing");
	}
	if (!has_span) {
		return;
	}
	validated_span_count++;
	if (size_t(span.slot_span) != end.pos - start.pos ||
	    size_t(span.string_byte_span) != end.string_cursor - start.string_cursor ||
	    size_t(span.length_count) != end.length_cursor - start.length_cursor ||
	    size_t(span.num_count) != end.num_cursor - start.num_cursor) {
		throw InvalidInputException("malformed JSONO: container span does not match its subtree");
	}
	if (span.shape_hash != expected_hash) {
		throw InvalidInputException("malformed JSONO: container shape hash mismatch");
	}
}

// Returns whether the validated value is a container (the caller's span eligibility
// depends on its children's kinds).
bool ValidateValue(const JsonoView &view, JsonoCursor &cursor, size_t depth,
                   std::vector<uint32_t> &container_child_counts, size_t &validated_span_count) {
	if (depth > JSONO_MAX_NESTING_DEPTH) {
		throw InvalidInputException("JSONO nesting depth exceeds maximum of %llu",
		                            (unsigned long long)JSONO_MAX_NESTING_DEPTH);
	}
	if (cursor.pos >= view.Slots()) {
		throw InvalidInputException("malformed JSONO: value position out of bounds");
	}
	auto slot = view.SlotAt(cursor.pos);
	auto slot_tag = SlotTag(slot);
	auto payload = SlotPayload(slot);
	switch (slot_tag) {
	case tag::OBJ_START: {
		auto container_id = ContainerId(payload);
		if (container_id != container_child_counts.size()) {
			throw InvalidInputException("malformed JSONO: non-sequential container id");
		}
		auto layout = ReadObjectLayout(view, cursor.pos);
		container_child_counts.push_back(uint32_t(layout.key_count));
		JsonoCursor start = cursor;
		uint64_t shape_hash = HASH_SEED;
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
			shape_hash = HashKey(shape_hash, key);
			previous_key = key;
		}
		cursor.pos = layout.value_start;
		bool has_container_child = false;
		for (size_t i = 0; i < layout.key_count; i++) {
			has_container_child |= ValidateValue(view, cursor, depth + 1, container_child_counts, validated_span_count);
		}
		if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
			throw InvalidInputException("malformed JSONO: object value span mismatch");
		}
		cursor.pos++;
		auto span_requirement = layout.key_count > OBJECT_CHECKPOINT_STRIDE ? ContainerSpanRequirement::Required
		                        : has_container_child                       ? ContainerSpanRequirement::Optional
		                                                                    : ContainerSpanRequirement::Forbidden;
		ValidateContainerSpan(view, container_id, span_requirement, shape_hash, start, cursor, validated_span_count);
		return true;
	}
	case tag::ARR_START: {
		auto container_id = ContainerId(payload);
		if (container_id != container_child_counts.size()) {
			throw InvalidInputException("malformed JSONO: non-sequential container id");
		}
		if (ContainerChildCount(payload) != 0) {
			throw InvalidInputException("malformed JSONO: array child count payload must be zero");
		}
		auto end_pos = ReadArrayEndPos(view, cursor.pos);
		container_child_counts.push_back(JSONO_ARRAY_CONTAINER);
		JsonoCursor start = cursor;
		cursor.pos++;
		while (cursor.pos < end_pos) {
			ValidateValue(view, cursor, depth + 1, container_child_counts, validated_span_count);
		}
		if (cursor.pos != end_pos) {
			throw InvalidInputException("malformed JSONO: array value span mismatch");
		}
		cursor.pos = end_pos + 1;
		auto span_requirement =
		    cursor.pos == start.pos + 2 ? ContainerSpanRequirement::Optional : ContainerSpanRequirement::Required;
		ValidateContainerSpan(view, container_id, span_requirement, 0, start, cursor, validated_span_count);
		return true;
	}
	default:
		ValidateScalar(view, cursor);
		return false;
	}
}

void ValidateMetadata(const JsonoView &view, const std::vector<uint32_t> &container_child_counts,
                      size_t validated_span_count) {
	auto &metadata = view.MetadataHeader();
	if (ExpectedSkipsSize(metadata) != view.SkipsSize()) {
		// The only legal tail after the checkpoint sections is a shred manifest; parsing it
		// validates the framing (and throws on trailing garbage).
		auto tail = view.ManifestTail();
		ValidateShredManifestBytes(tail.data(), tail.size());
	}
	// The walk validated every span-eligible container against a stored span, so an
	// extra stored span is exactly a count mismatch. The sparse id index must be
	// strictly sorted (the readers binary-search it) and in container-id range.
	if (metadata.span_count != validated_span_count) {
		throw InvalidInputException("malformed JSONO: container metadata count mismatch");
	}
	for (uint32_t i = 0; i < metadata.span_count; i++) {
		auto id = view.SpanIdAt(i);
		if (id >= container_child_counts.size()) {
			throw InvalidInputException("malformed JSONO: span container id out of bounds");
		}
		if (i > 0 && id <= view.SpanIdAt(i - 1)) {
			throw InvalidInputException("malformed JSONO: span id index is not sorted");
		}
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
		if (index.checkpoint_stride != OBJECT_CHECKPOINT_STRIDE) {
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
			if (checkpoint.length_delta > span.length_count) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint lengths out of bounds");
			}
			if (checkpoint.num_delta > span.num_count) {
				throw InvalidInputException("malformed JSONO: object cursor checkpoint nums out of bounds");
			}
		}
		next_checkpoint_offset += checkpoint_count;
		previous_container_id = index.container_id;
	}
	if (next_checkpoint_offset != metadata.checkpoint_count) {
		throw InvalidInputException("malformed JSONO: unreferenced object cursor checkpoints");
	}
}

bool ValidateJsonoBlob(const JsonoBlobRow &blob, ShredManifestVerifier &verifier) {
	try {
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			return false;
		}
		if (view.HeaderFlags() != flags::SORTED_KEYS || view.HeaderReserved() != 0) {
			return false;
		}
		JsonoCursor cursor;
		std::vector<uint32_t> container_child_counts;
		size_t validated_span_count = 0;
		ValidateValue(view, cursor, 0, container_child_counts, validated_span_count);
		if (cursor.pos != view.Slots() || cursor.string_cursor != view.StringHeapSize() ||
		    cursor.length_cursor != view.LengthCount() || cursor.num_cursor != view.NumCount()) {
			return false;
		}
		ValidateMetadata(view, container_child_counts, validated_span_count);
		// A narrowed row (its manifest lists a shred the column type lost) is unreadable by
		// every reader; validate reports it as invalid rather than blessing the framing alone.
		verifier.Verify(view);
		return true;
	} catch (const InvalidInputException &) {
		return false;
	}
}

// One array shred's lockstep handle: the compiled skeleton path and the lane's list entries.
struct LockstepLane {
	vector<PathStep> steps;
	UnifiedVectorFormat fmt;
	const list_entry_t *entries = nullptr;
};

// Array-shred lanes are lockstep with the residual skeleton: readers throw when a non-NULL LIST
// lane's length disagrees with the skeleton array's element count. Validate enforces the same
// invariant so it never blesses a row every reader rejects. Lane VALUE integrity stays DuckDB's
// business; only the lockstep framing is jsono's.
void InitLockstepLanes(Vector &input, idx_t count, vector<LockstepLane> &lanes) {
	JsonoLayoutType layout;
	if (!TryParseJsonoLayoutType(input.GetType(), layout)) {
		return;
	}
	for (idx_t f = 0; f < layout.shreds.size(); f++) {
		if (ClassifyShredKind(layout.shreds[f].second) == ShredKind::Scalar) {
			continue;
		}
		LockstepLane lane;
		auto &name = layout.shreds[f].first;
		if (name.size() >= 2 && name[0] == '$' && name[1] == '.') {
			lane.steps = ParseJsonoPath(name, "jsono_validate shred");
		} else {
			lane.steps.push_back(PathStep {PathStepKind::Key, name, 0});
		}
		auto &lane_vec = JsonoShredVector(input, f);
		lane_vec.ToUnifiedFormat(count, lane.fmt);
		lane.entries = UnifiedVectorFormat::GetData<list_entry_t>(lane.fmt);
		lanes.push_back(std::move(lane));
	}
}

bool LanesLockstepValid(const JsonoBlobRow &blob, const vector<LockstepLane> &lanes, idx_t row) {
	JsonoView view = MakeJsonoView(blob);
	view.ParseHeader(); // already validated by ValidateJsonoBlob
	for (auto &lane : lanes) {
		auto idx = lane.fmt.sel->get_index(row);
		if (!lane.fmt.validity.RowIsValid(idx)) {
			continue; // NULL lane: the residual carries the value, nothing to disagree with
		}
		JsonoCursor cursor;
		bool found = true;
		for (idx_t depth = 0; depth < lane.steps.size(); depth++) {
			if (!LocatePathStep(nullptr, depth, view, lane.steps[depth], cursor)) {
				found = false;
				break;
			}
		}
		if (!found || SlotTag(view.SlotAt(cursor.pos)) != tag::ARR_START) {
			continue; // no skeleton at the path: readers ignore the lane (divert-robust)
		}
		auto skeleton_len = CountArrayElements(view, cursor);
		if (uint64_t(skeleton_len) != lane.entries[idx].length) {
			return false;
		}
	}
	return true;
}

void JsonoValidateExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	JsonoVectorData input;
	InitJsonoVectorData(args.data[0], count, input);

	ShredManifestVerifier verifier;
	verifier.InitFromType(args.data[0].GetType());
	vector<LockstepLane> lockstep;
	InitLockstepLanes(args.data[0], count, lockstep);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);

	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRowStrict(input, row, blob)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		result_data[row] = ValidateJsonoBlob(blob, verifier);
		if (result_data[row] && !lockstep.empty()) {
			result_data[row] = LanesLockstepValid(blob, lockstep, row);
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Accept ANY so a shredded value reaches bind; keep its shredded type (reconstruct_shredded=false)
// so Execute validates the residual blobs directly plus the array-lane lockstep against the
// residual skeleton. The typed lane VALUES stay DuckDB-native; only the framing readers enforce
// (manifest, lockstep lengths) is jsono's to assert.
unique_ptr<FunctionData> JsonoValidateBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, false);
	return nullptr;
}

} // namespace

void RegisterJsonoValidate(ExtensionLoader &loader) {
	ScalarFunction fun("jsono_validate", {LogicalType::ANY}, LogicalType::BOOLEAN, JsonoValidateExecute,
	                   JsonoValidateBind);
	loader.RegisterFunction(fun);
}

} // namespace duckdb
