#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_dom.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"
#include "jsono_row_read.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;
using namespace jsono_dom;

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

// A strategy that emits a single scalar slot — true for the leaf scalars, false for the composite
// and deferred-render strategies. Shared by the list-child primitive check and the flat-scalar
// object gate.
bool IsPrimitiveConstructorStrategy(StructValueStrategy strategy) {
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
	uint64_t shape_hash = 0; // fingerprint of the sorted key set, constant per plan
	vector<uint64_t> string_object_slot_template;
	vector<JsonoStructPlan> children;
	idx_t key_cache_index = DConstants::INVALID_INDEX;
	bool flat_scalar_object = false;
	bool string_object = false;
	bool primitive_list = false;
	bool one_list_flat_scalar_object = false;
};

struct JsonoStructBindData : public FunctionData {
	JsonoStructPlan plan;
	// When non-empty, the constructor shreds these top-level scalar fields into shred
	// columns (result is a shredded JSONO struct).
	vector<std::pair<string, LogicalType>> shreds;
	// One-pass shredded write (ExecuteStructConstructorShredded): shred values copy straight
	// from the typed input children and the residual emits only the non-shred fields, so the
	// full plain value is never materialized. Eligible when every shred name is a literal
	// top-level key; a '$...'-named field addresses a nested path (ParseShredFieldPath) and
	// keeps the locate-and-strip path. `shred_fields` maps each shred to its input child,
	// `residual_fields` lists the non-shred children in input order, and `residual_plan`
	// is a constructor plan over just those fields (unset when every field is a shred).
	bool one_pass_shred = false;
	vector<idx_t> shred_fields;
	vector<idx_t> residual_fields;
	JsonoStructPlan residual_plan;

	explicit JsonoStructBindData(JsonoStructPlan plan_p) : plan(std::move(plan_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		// shreds must travel with the copy: the optimizer's shredded reconstruction copies the
		// constructor expression, and a copy that loses its shreds would take the plain path yet
		// still allocate the shredded return type, leaving the shred children uninitialized.
		auto copy = make_uniq<JsonoStructBindData>(plan);
		copy->shreds = shreds;
		copy->one_pass_shred = one_pass_shred;
		copy->shred_fields = shred_fields;
		copy->residual_fields = residual_fields;
		copy->residual_plan = residual_plan;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<JsonoStructBindData>();
		return plan.bound_type == other.plan.bound_type && shreds == other.shreds;
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
	// Verifies the shred manifest of embedded plain JSONO values (StructValueStrategy::Jsono):
	// the subtree re-emit drops the manifest, so a narrowed value must fail loud here instead
	// of laundering the loss. Plain values carry no shreds, so the signature set is empty.
	jsono::ShredManifestVerifier embedded_jsono_verifier;

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

bool IsJsonType(const LogicalType &type) {
	return type.id() == LogicalTypeId::VARCHAR && type.HasAlias() && type.GetAlias() == LogicalType::JSON_TYPE_NAME;
}

JsonoStructPlan BuildStructConstructorPlan(const LogicalType &source_type, const char *function_name) {
	JsonoStructPlan plan;

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
		// The format requires unique sorted object keys (readers binary-search, rank-cache and
		// shape_hash on that invariant). An unnamed struct would emit every field under the
		// duplicate key "", so reject it here, where both the constructor and the cast bind.
		if (StructType::IsUnnamed(source_type)) {
			throw BinderException("%s: unnamed STRUCT (row(...)) has no field names to use as object keys; "
			                      "use named fields like {'k': ...}",
			                      function_name);
		}
		auto &children = StructType::GetChildTypes(source_type);
		child_list_t<LogicalType> bound_children;
		plan.children.reserve(children.size());
		plan.field_names.reserve(children.size());
		plan.field_perm.reserve(children.size());
		plan.flat_scalar_object = true;
		plan.string_object = true;
		plan.one_list_flat_scalar_object = children.size() <= OBJECT_CHECKPOINT_STRIDE;
		idx_t list_flat_scalar_object_count = 0;
		for (idx_t i = 0; i < children.size(); i++) {
			auto child_plan = BuildStructConstructorPlan(children[i].second, function_name);
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
			} else if (!IsPrimitiveConstructorStrategy(child_plan.strategy)) {
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
		// Duplicate names are unreachable from SQL (DuckDB's binder rejects them in every struct
		// constructor), but type-level duplicates would corrupt the unique-key invariant, so fail
		// loud rather than silently dropping a field the caller passed.
		for (idx_t i = 1; i < plan.field_perm.size(); i++) {
			if (plan.field_names[plan.field_perm[i - 1]] == plan.field_names[plan.field_perm[i]]) {
				throw BinderException("%s: duplicate STRUCT field name '%s'", function_name,
				                      plan.field_names[plan.field_perm[i]]);
			}
		}
		for (auto field_idx : plan.field_perm) {
			auto &field_name = plan.field_names[field_idx];
			auto offset = plan.key_heap.size();
			plan.key_heap.insert(plan.key_heap.end(), field_name.begin(), field_name.end());
			plan.key_slots.push_back(MakeSlot(tag::KEY, MakeKeyPayload(uint64_t(offset), uint64_t(field_name.size()))));
		}
		plan.shape_hash = HashObjectKeySlots(plan.key_slots.data(), 0, plan.key_slots.size(), plan.key_heap.data());
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
		plan.children.push_back(BuildStructConstructorPlan(ListType::GetChildType(source_type), function_name));
		plan.primitive_list = IsPrimitiveConstructorStrategy(plan.children[0].strategy);
		plan.bound_type = LogicalType::LIST(plan.children[0].bound_type);
		return plan;
	}
	case LogicalTypeId::ARRAY: {
		plan.strategy = StructValueStrategy::List;
		plan.children.push_back(BuildStructConstructorPlan(ArrayType::GetChildType(source_type), function_name));
		plan.primitive_list = IsPrimitiveConstructorStrategy(plan.children[0].strategy);
		plan.bound_type = LogicalType::LIST(plan.children[0].bound_type);
		return plan;
	}
	default:
		throw BinderException("%s: unsupported value type '%s'", function_name, source_type.ToString());
	}
}

void AssignStructConstructorKeyCacheIndexes(JsonoStructPlan &plan, idx_t &next_key_cache_index) {
	if (plan.strategy == StructValueStrategy::Struct) {
		plan.key_cache_index = next_key_cache_index++;
	}
	for (auto &child : plan.children) {
		AssignStructConstructorKeyCacheIndexes(child, next_key_cache_index);
	}
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
		return !ReadJsonoRowStrict(data.jsono_data, row, ignored);
	}
	default:
		return !RowIsValid(data.fmt, row);
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

// The direct emitters' cursor snapshot at container start, paired with the sparse span
// slot the container reserved (JsonoBuilder::NO_SPAN for a small object).
struct DirectContainerStart {
	uint64_t id;
	uint64_t tag;
	size_t span_index;
	size_t start_slot;
	size_t start_string;
	size_t start_length;
	size_t start_num;
};

DirectContainerStart BeginConstructorDirectContainer(DomJsonoBuilder &builder, uint64_t container_tag,
                                                     uint32_t child_count) {
	if (child_count > CONTAINER_CHILD_COUNT_MASK) {
		throw InvalidInputException("jsono: object has too many keys for storage");
	}
	if (builder.container_count > std::numeric_limits<uint32_t>::max() || builder.container_count > CONTAINER_ID_MAX) {
		throw InvalidInputException("jsono: too many containers for storage");
	}
	auto id = builder.container_count++;
	auto span_index = JsonoBuilder::NO_SPAN;
	if (JsonoBuilder::ContainerStoresSpan(container_tag, child_count)) {
		builder.PushContainerSpanPlaceholder(id, span_index);
	}
	DirectContainerStart start {id,
	                            container_tag,
	                            span_index,
	                            builder.slots.size(),
	                            builder.string_heap.size(),
	                            builder.lengths.size(),
	                            builder.nums.size()};
	builder.slots.push_back(MakeSlot(container_tag, MakeContainerPayload(id, child_count)));
	return start;
}

// shape_hash is supplied by the caller from the (constant-per-plan) plan.shape_hash
// rather than recomputed from builder slots: the external_root_keys fast path writes
// KEY slots whose offsets point into an external key_heap, so builder.key_heap may be
// empty here. Arrays pass 0.
void FinishConstructorDirectContainer(DomJsonoBuilder &builder, const DirectContainerStart &start,
                                      uint64_t shape_hash) {
	if (start.span_index == JsonoBuilder::NO_SPAN) {
		return;
	}
	auto slot_span = builder.slots.size() - start.start_slot;
	auto string_byte_span = builder.string_heap.size() - start.start_string;
	auto length_count = builder.lengths.size() - start.start_length;
	auto num_count = builder.nums.size() - start.start_num;
	if (start.tag == tag::ARR_START && slot_span == 2) {
		if (start.span_index + 1 != builder.skips.size() || builder.span_ids[start.span_index] != start.id) {
			throw InternalException("jsono writer: empty array span is not the last sparse span");
		}
		builder.span_ids.pop_back();
		builder.skips.pop_back();
		return;
	}
	if (slot_span > std::numeric_limits<uint32_t>::max() || string_byte_span > std::numeric_limits<uint32_t>::max() ||
	    length_count > std::numeric_limits<uint32_t>::max() || num_count > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: container span exceeds storage limits");
	}
	builder.skips[start.span_index] = ContainerSpan {uint32_t(slot_span), uint32_t(string_byte_span), shape_hash,
	                                                 uint32_t(length_count), uint32_t(num_count)};
}

void EmitConstructorObjectKeys(const JsonoStructPlan &plan, DomJsonoBuilder &builder) {
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
			auto rebased_slot = MakeSlot(tag::KEY, MakeKeyPayload(KeyOffset(payload) + key_offset, KeyLen(payload)));
			builder.slots.push_back(rebased_slot);
		}
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
	auto start = BeginConstructorDirectContainer(builder, tag::OBJ_START, uint32_t(plan.field_perm.size()));
	EmitConstructorObjectKeys(plan, builder);
	for (auto field_idx : plan.field_perm) {
		if (!EmitConstructorScalarValue(data.children[field_idx], row, builder)) {
			throw InternalException("jsono constructor: non-scalar child in direct flat scalar object emitter");
		}
	}
	builder.slots.push_back(MakeSlot(tag::OBJ_END, 0));
	FinishConstructorDirectContainer(builder, start, plan.shape_hash);
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
	auto start = BeginConstructorDirectContainer(builder, tag::ARR_START, 0);
	auto child_all_valid = child.fmt.validity.AllValid();
	for (idx_t child_i = entry.offset; child_i < entry.offset + entry.length; child_i++) {
		if (!child_all_valid && ConstructorValueRowIsNull(child, child_i)) {
			builder.EmitNull();
		} else {
			EmitConstructorDirectFlatScalarObject(child, child_i, builder);
		}
	}
	builder.slots.push_back(MakeSlot(tag::ARR_END, 0));
	FinishConstructorDirectContainer(builder, start, 0);
}

void EmitConstructorOneListFlatScalarObject(const JsonoStructVectorData &data, idx_t row, DomJsonoBuilder &builder) {
	auto &plan = *data.plan;
	auto start = BeginConstructorDirectContainer(builder, tag::OBJ_START, uint32_t(plan.field_perm.size()));
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
	FinishConstructorDirectContainer(builder, start, plan.shape_hash);
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

	builder.EmitObjectStart(plan.field_perm.size());
	EmitConstructorObjectKeys(plan, builder);
	for (auto field_idx : plan.field_perm) {
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
		if (!ReadJsonoRowStrict(data.jsono_data, row, blob)) {
			builder.EmitNull();
			return;
		}
		JsonoView view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			builder.EmitNull();
			return;
		}
		lstate.embedded_jsono_verifier.Verify(view);
		JsonoCursor cursor;
		EmitValueVerbatim(view, cursor, builder, 0);
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

// Skips blob of a single root container: header only when the container stores no span
// (a small object), header + one sparse id + one span otherwise. No checkpoints — the
// direct string-object path never writes them (readers treat checkpoints as optional).
string_t WriteSingleContainerSkipsBlobInto(Vector &vec, bool spanned, const ContainerSpan &span) {
	auto total = sizeof(ContainerMetadataHeader) + (spanned ? sizeof(uint32_t) + sizeof(ContainerSpan) : 0);
	auto skips_str = StringVector::EmptyString(vec, total);
	auto skips_dst = skips_str.GetDataWriteable();

	ContainerMetadataHeader header;
	header.span_count = spanned ? 1 : 0;
	header.checkpoint_index_count = 0;
	header.checkpoint_count = 0;
	std::memcpy(skips_dst, &header, sizeof(header));
	if (spanned) {
		uint32_t id = 0;
		std::memcpy(skips_dst + sizeof(header), &id, sizeof(id));
		std::memcpy(skips_dst + sizeof(header) + sizeof(id), &span, sizeof(span));
	}

	skips_str.Finalize();
	return skips_str;
}

void WriteConstructorStringObjectDirect(const JsonoStructVectorData &data, idx_t row, JsonoBodyWriter &writer) {
	auto &plan = *data.plan;
	auto field_count = plan.field_perm.size();
	auto slot_count = uint32_t(2 + field_count * 2);
	auto slots_total = JSONO_HEADER_SIZE + size_t(slot_count) * sizeof(uint64_t);
	auto slots_str = StringVector::EmptyString(writer.Slots(), slots_total);
	auto slots_dst = slots_str.GetDataWriteable();

	WriteJsonoHeaderInto(reinterpret_cast<uint8_t *>(slots_dst), flags::SORTED_KEYS);
	std::memcpy(slots_dst + JSONO_HEADER_SIZE, plan.string_object_slot_template.data(),
	            plan.string_object_slot_template.size() * sizeof(uint64_t));

	std::array<string_t, 64> values;
	std::array<uint8_t, 64> value_valid;
	size_t values_size = 0;
	size_t non_null_count = 0;
	if (field_count <= values.size()) {
		for (idx_t value_idx = 0; value_idx < field_count; value_idx++) {
			auto &child = data.children[plan.field_perm[value_idx]];
			auto child_idx = RowIndex(child.fmt, row);
			if (child.fmt.validity.RowIsValid(child_idx)) {
				value_valid[value_idx] = 1;
				values[value_idx] = child.string_data[child_idx];
				values_size += values[value_idx].GetSize();
				non_null_count++;
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
				non_null_count++;
			}
		}
	}
	if (values_size > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: container string span exceeds storage limits");
	}
	auto values_str = StringVector::EmptyString(writer.StringHeap(), values_size);
	auto values_dst = values_str.GetDataWriteable();
	// One u32 length per non-null string value, in value order (walk order).
	auto lengths_str = StringVector::EmptyString(writer.Lengths(), non_null_count * sizeof(uint32_t));
	auto lengths_dst = lengths_str.GetDataWriteable();
	size_t value_offset = 0;
	size_t length_idx = 0;
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
		uint32_t value_length = uint32_t(value.GetSize());
		std::memcpy(lengths_dst + length_idx * sizeof(uint32_t), &value_length, sizeof(value_length));
		length_idx++;
		WriteSlotInto(slots_dst, value_start + value_idx, MakeSlot(tag::VAL_STR_HEAP, 0));
	}
	values_str.Finalize();
	lengths_str.Finalize();
	slots_str.Finalize();

	writer.data[BODY_STRING_HEAP][row] = values_str;
	writer.data[BODY_LENGTHS][row] = lengths_str;
	writer.data[BODY_NUMS][row] = WriteBlobInto(writer.Nums(), nullptr, 0);
	writer.data[BODY_SLOTS][row] = slots_str;
	auto spanned = JsonoBuilder::ContainerStoresSpan(tag::OBJ_START, uint32_t(field_count));
	ContainerSpan span {slot_count, uint32_t(values_size), plan.shape_hash, uint32_t(non_null_count), 0};
	writer.data[BODY_SKIPS][row] = WriteSingleContainerSkipsBlobInto(writer.Skips(), spanned, span);
}

void ExecuteStructConstructor(Vector &input, idx_t count, Vector &result, const JsonoStructPlan &plan,
                              JsonoStructLocalState &lstate) {
	JsonoStructVectorData input_data;
	InitStructConstructorVectorData(input, count, plan, input_data);

	JsonoBodyWriter writer;
	writer.Init(result);

	if (plan.strategy == StructValueStrategy::Jsono) {
		for (idx_t row = 0; row < count; row++) {
			JsonoBlobRow blob;
			if (!ReadJsonoRowStrict(input_data.jsono_data, row, blob)) {
				writer.SetRowNull(row);
				continue;
			}
			writer.data[BODY_SLOTS][row] = StringVector::AddStringOrBlob(writer.Slots(), blob.slots);
			writer.data[BODY_KEY_HEAP][row] = StringVector::AddStringOrBlob(writer.KeyHeap(), blob.key_heap);
			writer.data[BODY_STRING_HEAP][row] = StringVector::AddStringOrBlob(writer.StringHeap(), blob.string_heap);
			writer.data[BODY_SKIPS][row] = StringVector::AddStringOrBlob(writer.Skips(), blob.skips);
			writer.data[BODY_LENGTHS][row] = StringVector::AddStringOrBlob(writer.Lengths(), blob.lengths);
			writer.data[BODY_NUMS][row] = StringVector::AddStringOrBlob(writer.Nums(), blob.nums);
		}
		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	string_t static_keys;
	auto use_static_keys = plan.strategy == StructValueStrategy::Struct && plan.flat_scalar_object;
	auto use_direct_string_object = use_static_keys && plan.string_object;
	if (use_static_keys) {
		static_keys = WriteBlobInto(writer.KeyHeap(), plan.key_heap.data(), plan.key_heap.size());
	}

	auto &builder = lstate.builder;
	for (idx_t row = 0; row < count; row++) {
		if (ConstructorValueRowIsNull(input_data, row)) {
			writer.SetRowNull(row);
			continue;
		}
		builder.Reset();
		if (use_static_keys) {
			builder.external_root_keys = true;
		}
		if (use_direct_string_object) {
			writer.data[BODY_KEY_HEAP][row] = static_keys;
			WriteConstructorStringObjectDirect(input_data, row, writer);
			continue;
		}
		if (plan.strategy == StructValueStrategy::Struct) {
			EmitConstructorObject(input_data, row, lstate, builder);
		} else {
			EmitConstructorValue(input_data, row, lstate, builder);
		}
		if (use_static_keys) {
			writer.data[BODY_KEY_HEAP][row] = static_keys;
			writer.data[BODY_STRING_HEAP][row] =
			    WriteBlobInto(writer.StringHeap(), builder.string_heap.data(), builder.string_heap.size());
			// The root object's keys are external (external_root_keys), so FinishContainer
			// could not fingerprint them; set the root span's shape_hash from the plan.
			// The root (container id 0) stores a span only when it has > stride keys.
			if (!builder.span_ids.empty() && builder.span_ids[0] == 0) {
				builder.skips[0].shape_hash = plan.shape_hash;
			}
			writer.data[BODY_SLOTS][row] = WriteJsonoBlobInto(writer.Slots(), builder);
			writer.data[BODY_SKIPS][row] = WriteSkipsBlobInto(writer.Skips(), builder);
			writer.data[BODY_LENGTHS][row] = WriteLengthsBlobInto(writer.Lengths(), builder);
			writer.data[BODY_NUMS][row] = WriteNumsBlobInto(writer.Nums(), builder);
		} else {
			writer.WriteRow(row, builder);
		}
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Append `key` to a JSON path under `$`, quoting it when it carries a character ParseJsonoPath
// would otherwise treat as structural (a bare `.foo` step only spans up to the next . [ ] ").
void AppendShredPathStep(string &path, const string &key) {
	bool needs_quote = key.empty();
	for (char c : key) {
		if (c == '.' || c == '[' || c == ']' || c == '"' || c == '\\') {
			needs_quote = true;
			break;
		}
	}
	path.push_back('.');
	if (!needs_quote) {
		path.append(key);
		return;
	}
	path.push_back('"');
	for (char c : key) {
		if (c == '"' || c == '\\') {
			path.push_back('\\');
		}
		path.push_back(c);
	}
	path.push_back('"');
}

// Type-driven auto-shred: lift shred-eligible scalar fields into typed shreds. Top-level scalars
// keep their bare key name (the fast one-pass shred path); a scalar one level down inside a nested
// STRUCT is lifted under its `$.parent.child` JSON path so the existing path-aware two-pass shred
// writer captures it. The walk descends exactly one struct level: a leaf two or more levels deep
// stays in the residual. The schema is the only signal available at bind (the return type is a pure
// function of the input type, no per-row inference), and depth alone does not tell a hot path from a
// cold one — a deep schema would lift dozens of sparse leaves that are never queried while paying
// the full write and reconstruct cost. The one-level bound keeps the analytically addressed paths
// (a named object's direct fields) shredded without that blowup; frequency-aware budgeting of deeper
// paths belongs to a sampling ingest path, not the bind-time type walk.
void CollectAutoShreds(const LogicalType &struct_type, const string &path_prefix, bool top_level,
                       vector<pair<string, LogicalType>> &shreds) {
	for (auto &child : StructType::GetChildTypes(struct_type)) {
		// A top-level field named 'body' or the reserved shred-set marker would collide with a layout
		// field name, so it stays in the residual instead of becoming a shred (auto-shred must not
		// error on field names).
		if (top_level && (child.first == "body" || child.first == JsonoShredSetName())) {
			continue;
		}
		if (IsShredValueType(child.second)) {
			if (top_level) {
				shreds.emplace_back(child.first, child.second);
			} else {
				string path = path_prefix;
				AppendShredPathStep(path, child.first);
				shreds.emplace_back(std::move(path), child.second);
			}
			continue;
		}
		// A regular array shred — fixed-shape objects (LIST<STRUCT<scalars>>) lifting their element
		// subfields, or scalars (LIST<UBIGINT>, LIST<VARCHAR>, …) lifting each whole element — into a
		// parallel typed list shred, leaving a skeleton array in the residual. Lifted at the top level
		// by bare name and one struct level deep under `$.parent.items`, like a scalar leaf.
		if (IsShredListType(child.second)) {
			if (top_level) {
				shreds.emplace_back(child.first, child.second);
			} else {
				string path = path_prefix;
				AppendShredPathStep(path, child.first);
				shreds.emplace_back(std::move(path), child.second);
			}
			continue;
		}
		if (top_level && child.second.id() == LogicalTypeId::STRUCT) {
			string path = "$";
			AppendShredPathStep(path, child.first);
			CollectAutoShreds(child.second, path, false, shreds);
		}
	}
}

// A literal top-level key spelled as a JSON path ('$.a.b') and a nested scalar lifted to that same
// path resolve to one shred field name. A shredded STRUCT cannot carry duplicate child names:
// struct_extract would bind one child and leave the other value reachable only via to_json, not ->>.
// Drop every colliding shred back to the residual, where the plain emit keeps both values losslessly
// and path resolution stays correct. Mirrors the 'body' guard: auto-shred must not error or corrupt.
void DropCollidingAutoShreds(vector<pair<string, LogicalType>> &shreds) {
	vector<bool> colliding(shreds.size(), false);
	for (idx_t i = 0; i < shreds.size(); i++) {
		for (idx_t j = i + 1; j < shreds.size(); j++) {
			if (shreds[i].first == shreds[j].first) {
				colliding[i] = true;
				colliding[j] = true;
			}
		}
	}
	vector<pair<string, LogicalType>> kept;
	for (idx_t i = 0; i < shreds.size(); i++) {
		if (!colliding[i]) {
			kept.push_back(std::move(shreds[i]));
		}
	}
	shreds = std::move(kept);
}

unique_ptr<FunctionData> JsonoStructBindShared(ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments, bool allow_shred) {
	if (arguments.empty() || arguments[0]->return_type.id() != LogicalTypeId::STRUCT) {
		throw BinderException("jsono() STRUCT constructor requires a STRUCT argument");
	}
	auto &input_type = arguments[0]->return_type;
	auto plan = BuildStructConstructorPlan(input_type, "jsono()");
	idx_t next_key_cache_index = 0;
	AssignStructConstructorKeyCacheIndexes(plan, next_key_cache_index);
	// Keep the argument type the RAW input struct, not plan.bound_type. The shred set is selected
	// from the raw field types (below), and DuckDB re-binds this constructor from its serialized
	// argument type during plan verification. A promoted argument (INTEGER->BIGINT, UBIGINT->VARCHAR)
	// would flip shred eligibility on re-bind, producing a different shred manifest than the serialized
	// return type and a fatal STRUCT cast. The promotion the plan's strategies need is applied inside
	// execution instead (JsonoStructExecute), mirroring the cast entry point.
	bound_function.arguments[0] = input_type;
	auto bind_data = make_uniq<JsonoStructBindData>(std::move(plan));
	if (allow_shred) {
		CollectAutoShreds(input_type, string(), true, bind_data->shreds);
		DropCollidingAutoShreds(bind_data->shreds);
	}
	if (!bind_data->shreds.empty()) {
		// Canonical shred order (sorted by name) so the shredded type is a pure function of the shred
		// set, not of struct field order: the executor writes shreds by index, so the field write order
		// (bind_data->shreds) and the type's shred order must agree.
		std::sort(
		    bind_data->shreds.begin(), bind_data->shreds.end(),
		    [](const pair<string, LogicalType> &a, const pair<string, LogicalType> &b) { return a.first < b.first; });
		child_list_t<LogicalType> shred_types;
		for (auto &shred : bind_data->shreds) {
			shred_types.emplace_back(shred.first, shred.second);
		}
		bound_function.return_type = JsonoShreddedStructType(shred_types);

		bind_data->one_pass_shred = true;
		for (auto &shred : bind_data->shreds) {
			// A nested-path scalar ($-prefixed) or an array shred (LIST<STRUCT> / LIST<scalar>) cannot
			// use the scalar direct-copy one-pass writer; route through the materialize-then-shred
			// fallback, which lifts the array elements and builds the skeleton via JsonoShredFromSpec.
			if ((!shred.first.empty() && shred.first[0] == '$') || IsShredListType(shred.second)) {
				bind_data->one_pass_shred = false;
				break;
			}
		}
		if (bind_data->one_pass_shred) {
			auto &children = StructType::GetChildTypes(input_type);
			auto shred_for_name = [&](const string &name) {
				for (idx_t f = 0; f < bind_data->shreds.size(); f++) {
					if (bind_data->shreds[f].first == name) {
						return f;
					}
				}
				return idx_t(DConstants::INVALID_INDEX);
			};
			bind_data->shred_fields.resize(bind_data->shreds.size());
			child_list_t<LogicalType> residual_children;
			for (idx_t i = 0; i < children.size(); i++) {
				auto f = shred_for_name(children[i].first);
				if (f != DConstants::INVALID_INDEX) {
					bind_data->shred_fields[f] = i;
				} else {
					residual_children.push_back(children[i]);
					bind_data->residual_fields.push_back(i);
				}
			}
			if (!bind_data->residual_fields.empty()) {
				bind_data->residual_plan =
				    BuildStructConstructorPlan(LogicalType::STRUCT(std::move(residual_children)), "jsono()");
				AssignStructConstructorKeyCacheIndexes(bind_data->residual_plan, next_key_cache_index);
			}
		}
	}
	return std::move(bind_data);
}

unique_ptr<FunctionData> JsonoStructBind(ClientContext &context, ScalarFunction &bound_function,
                                         vector<unique_ptr<Expression>> &arguments) {
	return JsonoStructBindShared(bound_function, arguments, true);
}

// The optimizer's shredded reconstruction builds a JSONO patch via JsonoStructConstructorFunction;
// that patch must stay plain JSONO so the overlay merges plain residual + plain patch. This bind
// forces shred off (the patch's top-level fields are shred-typed scalars, which the always-on
// auto-shred would otherwise lift back into shreds and break the overlay).
unique_ptr<FunctionData> JsonoStructBindPlain(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	return JsonoStructBindShared(bound_function, arguments, false);
}

// Typed reads for one shred column: the value copies straight from the input child vector
// (no plain materialization, no per-row locate). A UBIGINT shred reads the RAW input child —
// the plan promotes UBIGINT to text for the residual number ladder, but the shred wants the
// native value.
struct ShredColumnSource {
	StructValueStrategy strategy = StructValueStrategy::Null;
	bool raw_uint = false;
	UnifiedVectorFormat fmt;
	const string_t *string_data = nullptr;
	const int64_t *int_data = nullptr;
	const double *double_data = nullptr;
	const bool *bool_data = nullptr;
	const uint64_t *uint_data = nullptr;
};

// One-pass shredded constructor: shred values copy from the typed input children, the residual
// emits only the non-shred fields, and the manifest is written along the way — the two-pass path
// (full plain emit, then per-row locate + strip + re-emit in JsonoShredFromSpec) is skipped
// entirely. Per row only the shred fields' NULL mask varies: every non-NULL shred field is
// losslessly captured by construction (a VARCHAR field is a real JSON string, BIGINT/DOUBLE/
// BOOLEAN/UBIGINT match their stored kind exactly) and is stripped from the residual; a NULL
// shred field stays in the residual as an explicit null, since a NULL shred means "value lives
// in the residual" to every reader.
void ExecuteStructConstructorShredded(Vector &raw_input, Vector &casted_input, idx_t count, Vector &result,
                                      const JsonoStructBindData &bind_data, JsonoStructLocalState &lstate) {
	auto &plan = bind_data.plan;
	auto &shreds = bind_data.shreds;
	auto shred_count = shreds.size();

	UnifiedVectorFormat parent_fmt;
	casted_input.ToUnifiedFormat(count, parent_fmt);
	auto &casted_children = StructVector::GetEntries(casted_input);

	vector<ShredColumnSource> sources(shred_count);
	vector<Vector *> shred_out(shred_count);
	for (idx_t f = 0; f < shred_count; f++) {
		auto field_idx = bind_data.shred_fields[f];
		auto &src = sources[f];
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
		if (shreds[f].second.id() == LogicalTypeId::UBIGINT) {
			auto &raw_child = *StructVector::GetEntries(raw_input)[field_idx];
			raw_child.ToUnifiedFormat(count, src.fmt);
			src.uint_data = UnifiedVectorFormat::GetData<uint64_t>(src.fmt);
			src.raw_uint = true;
			continue;
		}
		auto &child = *casted_children[field_idx];
		child.ToUnifiedFormat(count, src.fmt);
		src.strategy = plan.children[field_idx].strategy;
		switch (src.strategy) {
		case StructValueStrategy::String:
			src.string_data = UnifiedVectorFormat::GetData<string_t>(src.fmt);
			// The shred output references the input strings; keep their buffer alive.
			StringVector::AddHeapReference(*shred_out[f], child);
			break;
		case StructValueStrategy::Int:
			src.int_data = UnifiedVectorFormat::GetData<int64_t>(src.fmt);
			break;
		case StructValueStrategy::Double:
			src.double_data = UnifiedVectorFormat::GetData<double>(src.fmt);
			break;
		case StructValueStrategy::Bool:
			src.bool_data = UnifiedVectorFormat::GetData<bool>(src.fmt);
			break;
		default:
			throw InternalException("jsono constructor: unexpected shred field strategy");
		}
	}

	// Residual vector data over the non-shred children, addressed by the residual plan.
	JsonoStructVectorData residual_data;
	bool residual_empty = bind_data.residual_fields.empty();
	if (!residual_empty) {
		residual_data.plan = &bind_data.residual_plan;
		casted_input.ToUnifiedFormat(count, residual_data.fmt);
		residual_data.children.resize(bind_data.residual_fields.size());
		for (idx_t j = 0; j < bind_data.residual_fields.size(); j++) {
			InitStructConstructorVectorData(*casted_children[bind_data.residual_fields[j]], count,
			                                bind_data.residual_plan.children[j], residual_data.children[j]);
		}
	}

	// field_perm walk metadata for the partial-null rows: per plan field, its shred position
	// (INVALID for a residual field) and its position among the residual children.
	vector<idx_t> shred_pos(plan.children.size(), DConstants::INVALID_INDEX);
	vector<idx_t> residual_pos(plan.children.size(), DConstants::INVALID_INDEX);
	for (idx_t f = 0; f < shred_count; f++) {
		shred_pos[bind_data.shred_fields[f]] = f;
	}
	for (idx_t j = 0; j < bind_data.residual_fields.size(); j++) {
		residual_pos[bind_data.residual_fields[j]] = j;
	}

	// Manifest bytes: per-shred entry once, plus the full all-stripped manifest (the hot case).
	vector<JsonoShredManifestEntryBytes> manifest_entries(shred_count);
	std::string hot_manifest;
	for (idx_t f = 0; f < shred_count; f++) {
		manifest_entries[f] = JsonoShredManifestEntry(shreds[f].first, shreds[f].second);
	}
	JsonoAppendShredManifest(hot_manifest, manifest_entries);

	JsonoBodyWriter writer;
	writer.Init(result);
	// Type-driven auto-shred never diverts: each shred's type is its source field's type, so a
	// present value always fits the shred and a NULL shred is only an absent field. The value is
	// therefore always complete — which gives jsono(struct, shredding) the COALESCE-free typed
	// read on read-back for struct-ingest workloads.
	JsonoFillShredsComplete(result, count);
	auto &builder = lstate.builder;

	// All-stripped rows with an empty residual share constant blobs (header + {} + manifest).
	bool hot_blobs_ready = false;
	string_t hot_blobs[BODY_BLOB_COUNT];

	vector<uint8_t> stripped(shred_count);
	vector<idx_t> stripped_fields;
	std::string manifest;
	for (idx_t row = 0; row < count; row++) {
		if (!parent_fmt.validity.RowIsValid(parent_fmt.sel->get_index(row))) {
			writer.SetRowNull(row);
			JsonoSetRowMarkerNull(result, row);
			for (idx_t f = 0; f < shred_count; f++) {
				FlatVector::SetNull(*shred_out[f], row, true);
			}
			continue;
		}
		idx_t stripped_count = 0;
		stripped_fields.clear();
		for (idx_t f = 0; f < shred_count; f++) {
			auto &src = sources[f];
			auto idx = src.fmt.sel->get_index(row);
			if (!src.fmt.validity.RowIsValid(idx)) {
				FlatVector::SetNull(*shred_out[f], row, true);
				stripped[f] = 0;
				continue;
			}
			stripped[f] = 1;
			stripped_count++;
			stripped_fields.push_back(f);
			if (src.raw_uint) {
				FlatVector::GetData<uint64_t>(*shred_out[f])[row] = src.uint_data[idx];
				continue;
			}
			switch (src.strategy) {
			case StructValueStrategy::String:
				FlatVector::GetData<string_t>(*shred_out[f])[row] = src.string_data[idx];
				break;
			case StructValueStrategy::Int:
				FlatVector::GetData<int64_t>(*shred_out[f])[row] = src.int_data[idx];
				break;
			case StructValueStrategy::Double: {
				auto value = src.double_data[idx];
				// The value no longer passes through EmitDouble, but its gate still holds:
				// a non-finite double must fail loud, not slip into a shred.
				if (!std::isfinite(value)) {
					throw InvalidInputException("jsono: cannot store non-finite double value (NaN/Infinity)");
				}
				FlatVector::GetData<double>(*shred_out[f])[row] = value;
				break;
			}
			case StructValueStrategy::Bool:
				FlatVector::GetData<bool>(*shred_out[f])[row] = src.bool_data[idx];
				break;
			default:
				throw InternalException("jsono constructor: unexpected shred field strategy");
			}
		}
		FlatVector::Validity(result).SetValid(row);

		if (stripped_count == shred_count) {
			if (residual_empty) {
				if (!hot_blobs_ready) {
					builder.Reset();
					builder.EmitObjectStart(0);
					builder.EmitObjectEnd();
					hot_blobs[BODY_SLOTS] = WriteJsonoBlobInto(writer.Slots(), builder);
					hot_blobs[BODY_KEY_HEAP] = WriteBlobInto(writer.KeyHeap(), nullptr, 0);
					hot_blobs[BODY_STRING_HEAP] = WriteBlobInto(writer.StringHeap(), nullptr, 0);
					hot_blobs[BODY_SKIPS] = WriteSkipsBlobInto(writer.Skips(), builder, &hot_manifest);
					hot_blobs[BODY_LENGTHS] = WriteBlobInto(writer.Lengths(), nullptr, 0);
					hot_blobs[BODY_NUMS] = WriteBlobInto(writer.Nums(), nullptr, 0);
					hot_blobs_ready = true;
				}
				for (idx_t b = 0; b < BODY_BLOB_COUNT; b++) {
					writer.data[b][row] = hot_blobs[b];
				}
				continue;
			}
			builder.Reset();
			EmitConstructorObject(residual_data, row, lstate, builder);
			writer.WriteRow(row, builder, &hot_manifest);
			continue;
		}

		// Some shred fields are NULL this row: they stay in the residual as explicit nulls,
		// interleaved with the residual fields in sorted key order.
		builder.Reset();
		builder.EmitObjectStart(plan.field_perm.size() - stripped_count);
		for (auto field_idx : plan.field_perm) {
			auto f = shred_pos[field_idx];
			if (f != DConstants::INVALID_INDEX && stripped[f]) {
				continue;
			}
			auto &name = plan.field_names[field_idx];
			builder.EmitKeySlot(nonstd::string_view(name.data(), name.size()));
		}
		for (auto field_idx : plan.field_perm) {
			auto f = shred_pos[field_idx];
			if (f != DConstants::INVALID_INDEX && stripped[f]) {
				continue;
			}
			builder.EmitObjectChildStart();
			if (f != DConstants::INVALID_INDEX) {
				builder.EmitNull();
			} else {
				EmitConstructorValue(residual_data.children[residual_pos[field_idx]], row, lstate, builder);
			}
		}
		builder.EmitObjectEnd();
		const std::string *manifest_ptr = nullptr;
		if (stripped_count > 0) {
			manifest.clear();
			JsonoAppendShredManifest(manifest, manifest_entries, stripped_fields);
			manifest_ptr = &manifest;
		}
		writer.WriteRow(row, builder, manifest_ptr);
	}
}

void JsonoStructExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<JsonoStructLocalState>();
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = func_expr.bind_info->Cast<JsonoStructBindData>();
	auto count = args.size();

	// The argument keeps the raw input struct type (see JsonoStructBindShared); promote it to the
	// plan's bound type here, where the plan's strategies expect fixed widths.
	Vector *input = &args.data[0];
	Vector casted(bind_data.plan.bound_type, count);
	if (args.data[0].GetType() != bind_data.plan.bound_type) {
		VectorOperations::Cast(state.GetContext(), args.data[0], casted, count);
		input = &casted;
	}

	if (bind_data.shreds.empty()) {
		ExecuteStructConstructor(*input, count, result, bind_data.plan, lstate);
		return;
	}
	if (bind_data.one_pass_shred) {
		ExecuteStructConstructorShredded(args.data[0], *input, count, result, bind_data, lstate);
	} else {
		Vector plain(JsonoType(), count);
		ExecuteStructConstructor(*input, count, plain, bind_data.plan, lstate);
		JsonoShredFromSpec(plain, count, bind_data.shreds, result);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
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

// A plain JSONO value reaching the wildcard STRUCT(any)->JSONO cast: source and target are the
// same nested type, so reference it through. The corrupt check first: a valid top-level row whose
// layout/body/blobs are absent is the trace of a by-name struct cast NULL-filling a layout field
// (a set operation over differently-shredded values). ReadJsonoRowStrict owns that fail-loud:
// ReadJsonoRowImpl<true> raises ThrowCorruptJsonoRow ("body is NULL") on a present row with an
// absent body, so the strict read below throws on such a row rather than reading it as SQL NULL.
// The loop body is empty on purpose — its only job is to trigger that strict-read throw per row
// before the bytes are referenced through (a genuine all-NULL row reads false and is skipped).
bool JsonoPrefixReinterpretCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	jsono::JsonoVectorData residual;
	jsono::InitJsonoVectorData(source, count, residual);
	for (idx_t row = 0; row < count; row++) {
		jsono::JsonoBlobRow blob {};
		if (!jsono::ReadJsonoRowStrict(residual, row, blob)) {
			continue;
		}
	}
	result.Reference(source);
	return true;
}

// Cast a shredded JSONO struct to plain JSONO losslessly: overlay each row's shreds back onto
// its residual. This is what an implicit cast / INSERT into a plain JSONO column needs — the
// reinterpret path would silently drop the shred columns. The optimizer never reaches here for
// its residual extraction; it reinterprets the six-blob prefix directly.
bool JsonoShreddedReconstructCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	(void)parameters;
	JsonoReconstructToPlain(source, count, result);
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR && count > 0) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo JsonoStructCastBind(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	(void)input;
	(void)target;
	if (IsShreddedJsonoType(source)) {
		// A shredded value reaching a plain-JSONO context (cast or INSERT): reconstruct the
		// full value so the shred data is preserved rather than dropped.
		return BoundCastInfo(JsonoShreddedReconstructCast);
	}
	if (IsJsonoType(source)) {
		// A plain already-stored JSONO value (same nested type as the target): reference it through
		// rather than rebuilding from struct data.
		return BoundCastInfo(JsonoPrefixReinterpretCast);
	}
	auto plan = BuildStructConstructorPlan(source, "JSONO cast");
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

} // namespace

// The bare jsono(STRUCT) constructor overload, exposed so the optimizer's shredded
// reconstruction can build a JSONO patch from the shred columns without a catalog
// lookup. Mirrors the single-argument overload registered below.
ScalarFunction JsonoStructConstructorFunction() {
	ScalarFunction fun("jsono", {LogicalTypeId::STRUCT}, JsonoType(), JsonoStructExecute, JsonoStructBindPlain, nullptr,
	                   nullptr, JsonoStructLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

void RegisterJsonoStructConstructor(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();
	{
		ScalarFunctionSet set("jsono");
		ScalarFunction struct_ctor({LogicalTypeId::STRUCT}, jsono_type, JsonoStructExecute, JsonoStructBind, nullptr,
		                           nullptr, JsonoStructLocalState::Init);
		struct_ctor.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		struct_ctor.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
		set.AddFunction(struct_ctor);

		loader.RegisterFunction(set);
	}
	loader.RegisterCastFunction(LogicalType::STRUCT({{"any", LogicalType::ANY}}), jsono_type, JsonoStructCastBind, -1);
}

} // namespace duckdb
