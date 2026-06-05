#include "jsono.hpp"
#include "jsono_reader.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

namespace {

using namespace jsono;

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

} // namespace

void RegisterJsonoStorageSize(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	// Per-BLOB byte counts share the physical layout's field names, so derive
	// them from JsonoRawStructType() instead of re-listing, then append `total`.
	// The field order must stay slots/key_heap/string_heap/skips — Execute writes
	// the child vectors by that positional order.
	auto raw_struct = JsonoRawStructType();
	child_list_t<LogicalType> size_children;
	for (auto &field : StructType::GetChildTypes(raw_struct)) {
		size_children.emplace_back(field.first, LogicalType::UBIGINT);
	}
	size_children.emplace_back("total", LogicalType::UBIGINT);
	auto storage_size_type = LogicalType::STRUCT(std::move(size_children));
	ScalarFunction fun("jsono_storage_size", {jsono_type}, storage_size_type, JsonoStorageSizeExecute);
	loader.RegisterFunction(fun);
}

} // namespace duckdb
