#include "jsono.hpp"
#include "jsono_reader.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

namespace {

using namespace jsono;

struct StorageSizeSource {
	JsonoVectorData body;
	vector<UnifiedVectorFormat> shred_fmt;
	vector<LogicalTypeId> shred_kind;
};

// Logical payload bytes a shred value occupies (the closed shred type set; a NULL shred is 0).
// Mirrors the BLOB fields, which report payload bytes, not vector/validity overhead.
uint64_t ShredValueBytes(const UnifiedVectorFormat &fmt, LogicalTypeId kind, idx_t row) {
	if (!RowIsValid(fmt, row)) {
		return 0;
	}
	auto idx = RowIndex(fmt, row);
	switch (kind) {
	case LogicalTypeId::VARCHAR:
		return UnifiedVectorFormat::GetData<string_t>(fmt)[idx].GetSize();
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::DOUBLE:
		return 8;
	case LogicalTypeId::BOOLEAN:
		return 1;
	default:
		throw InternalException("jsono_storage_size: unexpected shred type");
	}
}

void InitStorageSizeSource(Vector &input_vec, idx_t count, const child_list_t<LogicalType> &shreds,
                           StorageSizeSource &source) {
	InitJsonoVectorData(input_vec, count, source.body);
	if (shreds.empty()) {
		return;
	}
	source.shred_fmt.resize(shreds.size());
	for (idx_t k = 0; k < shreds.size(); k++) {
		JsonoShredVector(input_vec, k).ToUnifiedFormat(count, source.shred_fmt[k]);
		source.shred_kind.push_back(shreds[k].second.id());
	}
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
	for (idx_t k = 0; k < source.shred_fmt.size(); k++) {
		shred_bytes += ShredValueBytes(source.shred_fmt[k], source.shred_kind[k], row);
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
	InitStorageSizeSource(input_vec, count, layout.shreds, source);
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
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto &type = arguments[0]->return_type;
	JsonoRequireExtensionOptimizerForShredded(context, type, bound_function.name);
	if (IsShreddedJsonoType(type)) {
		bound_function.arguments[0] = type;
	} else if (type.id() == LogicalTypeId::SQLNULL || IsJsonoType(type)) {
		bound_function.arguments[0] = JsonoType();
	} else {
		throw BinderException("%s: argument must be JSONO", bound_function.name);
	}
	return nullptr;
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
}

} // namespace duckdb
