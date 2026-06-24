#include "jsono.hpp"
#include "jsono_reader.hpp"
#include "jsono_shred.hpp"
#include "jsono_shred_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

#include <algorithm>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

struct StorageSizeSource {
	JsonoVectorData body;
	vector<ShredLane> shreds;
};

uint64_t ShredValueBytes(const UnifiedVectorFormat &fmt, JsonoScalarPrimitive kind, idx_t row) {
	if (!RowIsValid(fmt, row)) {
		return 0;
	}
	return JsonoPrimitiveVectorPayloadBytes(kind, fmt, RowIndex(fmt, row));
}

// Payload bytes an array shred occupies for one row, summed over the row's list span: a LIST<STRUCT>
// element contributes its lifted subfields' scalar bytes, a LIST<scalar> element its single lifted
// value's bytes (a NULL element / absent value is 0). Both array kinds expose their element value
// lanes through lane.sub_fmt/sub_kind (a scalar array has one), so the per-element sum is uniform.
uint64_t ArrayShredBytes(const ShredLane &lane, idx_t row) {
	if (!RowIsValid(lane.fmt, row)) {
		return 0;
	}
	auto entry = UnifiedVectorFormat::GetData<list_entry_t>(lane.fmt)[RowIndex(lane.fmt, row)];
	uint64_t bytes = 0;
	for (idx_t element = entry.offset; element < entry.offset + entry.length; element++) {
		for (idx_t j = 0; j < lane.sub_fmt.size(); j++) {
			bytes += ShredValueBytes(lane.sub_fmt[j], lane.sub_kind[j], element);
		}
	}
	return bytes;
}

uint64_t ShredLaneBytes(const ShredLane &lane, idx_t row) {
	switch (lane.kind) {
	case ShredKind::Scalar:
		return ShredValueBytes(lane.fmt, lane.scalar_kind, row);
	case ShredKind::Array:
	case ShredKind::ScalarArray:
		return ArrayShredBytes(lane, row);
	}
	return 0;
}

struct StorageSizeResult {
	uint64_t *slots;
	uint64_t *key_heap;
	uint64_t *string_heap;
	uint64_t *skips;
	uint64_t *lengths;
	uint64_t *nums;
	uint64_t *shreds;
	uint64_t *total;
};

StorageSizeResult InitStorageSizeResult(Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	return StorageSizeResult {FlatVector::GetData<uint64_t>(*children[0]), FlatVector::GetData<uint64_t>(*children[1]),
	                          FlatVector::GetData<uint64_t>(*children[2]), FlatVector::GetData<uint64_t>(*children[3]),
	                          FlatVector::GetData<uint64_t>(*children[4]), FlatVector::GetData<uint64_t>(*children[5]),
	                          FlatVector::GetData<uint64_t>(*children[6]), FlatVector::GetData<uint64_t>(*children[7])};
}

void SetStorageSizeRowNull(Vector &result, idx_t row) {
	FlatVector::SetNull(result, row, true);
	for (auto &child : StructVector::GetEntries(result)) {
		FlatVector::SetNull(*child, row, true);
	}
}

uint64_t ShredPayloadBytes(const StorageSizeSource &source, idx_t row) {
	uint64_t shred_bytes = 0;
	for (auto &lane : source.shreds) {
		shred_bytes += ShredLaneBytes(lane, row);
	}
	return shred_bytes;
}

void WriteStorageSizeRow(const StorageSizeSource &source, const JsonoBlobRow &blob, StorageSizeResult &output,
                         idx_t row) {
	auto shred_bytes = ShredPayloadBytes(source, row);
	output.slots[row] = blob.slots.GetSize();
	output.key_heap[row] = blob.key_heap.GetSize();
	output.string_heap[row] = blob.string_heap.GetSize();
	output.skips[row] = blob.skips.GetSize();
	output.lengths[row] = blob.lengths.GetSize();
	output.nums[row] = blob.nums.GetSize();
	output.shreds[row] = shred_bytes;
	output.total[row] = output.slots[row] + output.key_heap[row] + output.string_heap[row] + output.skips[row] +
	                    output.lengths[row] + output.nums[row] + shred_bytes;
}

void JsonoStorageSizeExecuteSingle(Vector &input_vec, idx_t count, Vector &result) {
	JsonoLayoutType layout;
	TryParseJsonoLayoutType(input_vec.GetType(), layout);
	StorageSizeSource source;
	InitJsonoVectorData(input_vec, count, source.body);
	InitShredLanes(input_vec, count, layout.shreds, source.shreds);
	auto output = InitStorageSizeResult(result);
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRowStrict(source.body, row, blob)) {
			SetStorageSizeRowNull(result, row);
			continue;
		}
		WriteStorageSizeRow(source, blob, output, row);
	}
}

void JsonoStorageSizeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	auto &input_vec = args.data[0];
	JsonoStorageSizeExecuteSingle(input_vec, count, result);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Accept ANY so a shredded value reaches bind; keep its shredded type (reconstruct_shredded=false)
// so Execute measures the real on-disk footprint — residual blobs plus shred payload — rather than a
// reconstructed plain blob.
unique_ptr<FunctionData> JsonoStorageSizeBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] =
	    JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, /*reconstruct_shredded=*/false);
	return nullptr;
}

// jsono_shred_manifest reads the residual's OWN manifest claim raw (ManifestTail walked with the
// bounds-safe collect sink), NOT through the verifying reader: introspection must show what a row
// claims it stripped — including a narrowed row — without raising, which the verifying read would do
// on exactly the rows a user most wants to inspect. A plain value (no manifest) yields an empty list,
// not NULL: "no paths were stripped" is a true answer. Only a SQL NULL / absent body row is NULL.
void JsonoShredManifestExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto count = args.size();
	auto &input_vec = args.data[0];
	JsonoVectorData body;
	InitJsonoVectorData(input_vec, count, body);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	auto &child = ListVector::GetEntry(result);
	auto &child_entries = StructVector::GetEntries(child);
	auto &path_vector = *child_entries[0];
	auto &type_vector = *child_entries[1];

	std::vector<ShredManifestEntry> entries;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (!ReadJsonoRowStrict(body, row, blob)) {
			SetListRowNull(result, row);
			continue;
		}
		view = MakeJsonoView(blob);
		if (!view.ParseHeader()) {
			SetListRowNull(result, row);
			continue;
		}
		auto start = ListVector::GetListSize(result);
		idx_t length = 0;
		if (view.HasShredManifest()) {
			entries.clear();
			ShredManifestCollectSink sink {entries};
			auto tail = view.ManifestTail();
			WalkShredManifestBytes(tail.data(), tail.size(), sink);
			length = entries.size();
			EnsureListCapacity(result, start + length);
			auto path_data = FlatVector::GetData<string_t>(path_vector);
			auto type_data = FlatVector::GetData<string_t>(type_vector);
			for (idx_t i = 0; i < length; i++) {
				auto child_row = start + i;
				path_data[child_row] =
				    StringVector::AddString(path_vector, entries[i].path.data(), entries[i].path.size());
				type_data[child_row] =
				    StringVector::AddString(type_vector, entries[i].type.data(), entries[i].type.size());
			}
		}
		FinishListRow(result, row, start, length);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterJsonoStorageSize(ExtensionLoader &loader) {
	// Per-BLOB byte counts share the body struct's field names, so derive them from
	// JsonoBodyStructType() instead of re-listing, then append `shreds` (shredded shred payload bytes,
	// 0 for a plain value) and `total`. The field order must stay slots/key_heap/string_heap/skips/
	// lengths/nums/shreds/total — Execute writes the child vectors by that positional order.
	auto body_struct = JsonoBodyStructType();
	child_list_t<LogicalType> size_children;
	for (auto &field : StructType::GetChildTypes(body_struct)) {
		size_children.emplace_back(field.first, LogicalType::UBIGINT);
	}
	size_children.emplace_back("shreds", LogicalType::UBIGINT);
	size_children.emplace_back("total", LogicalType::UBIGINT);
	auto storage_size_type = LogicalType::STRUCT(std::move(size_children));
	ScalarFunction fun("jsono_storage_size", {LogicalType::ANY}, storage_size_type, JsonoStorageSizeExecute,
	                   JsonoStorageSizeBind);
	loader.RegisterFunction(fun);

	auto manifest_type =
	    LogicalType::LIST(LogicalType::STRUCT({{"path", LogicalType::VARCHAR}, {"type", LogicalType::VARCHAR}}));
	ScalarFunction manifest_fun("jsono_shred_manifest", {LogicalType::ANY}, manifest_type, JsonoShredManifestExecute,
	                            JsonoStorageSizeBind);
	loader.RegisterFunction(manifest_fun);
}

} // namespace duckdb
