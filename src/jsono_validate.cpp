#include "jsono_ops.hpp"
#include "jsono.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"

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

} // namespace

void RegisterJsonoValidate(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	ScalarFunction fun("jsono_validate", {jsono_type}, LogicalType::BOOLEAN, JsonoValidateExecute);
	loader.RegisterFunction(fun);
}

} // namespace duckdb
