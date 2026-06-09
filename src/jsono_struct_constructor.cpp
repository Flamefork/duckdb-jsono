#include "jsono.hpp"
#include "jsono_dom.hpp"
#include "jsono_number.hpp"
#include "jsono_reader.hpp"
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
	// When non-empty, the constructor builds a plain residual and then shreds these
	// top-level scalar fields into shred columns (result is a shredded JSONO struct).
	vector<std::pair<string, LogicalType>> shreds;

	explicit JsonoStructBindData(JsonoStructPlan plan_p) : plan(std::move(plan_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		// shreds must travel with the copy: the optimizer's shredded reconstruction copies the
		// constructor expression, and a copy that loses its shreds would take the plain path yet
		// still allocate the shredded return type, leaving the shred children uninitialized.
		auto copy = make_uniq<JsonoStructBindData>(plan);
		copy->shreds = shreds;
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
	jsono::SetJsonoStructNull(result, slots_vec, key_heap_vec, string_heap_vec, skips_vec, row);
}

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
		plan.primitive_list = IsPrimitiveConstructorListChild(plan.children[0].strategy);
		plan.bound_type = LogicalType::LIST(plan.children[0].bound_type);
		return plan;
	}
	case LogicalTypeId::ARRAY: {
		plan.strategy = StructValueStrategy::List;
		plan.children.push_back(BuildStructConstructorPlan(ArrayType::GetChildType(source_type), function_name));
		plan.primitive_list = IsPrimitiveConstructorListChild(plan.children[0].strategy);
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
	builder.skips.push_back(ContainerSpan {0, 0, 0});
	builder.slots.push_back(MakeSlot(container_tag, MakeContainerPayload(id, child_count)));
	return id;
}

// shape_hash is supplied by the caller from the (constant-per-plan) plan.shape_hash
// rather than recomputed from builder slots: the external_root_keys fast path writes
// KEY slots whose offsets point into an external key_heap, so builder.key_heap may be
// empty here. Arrays pass 0.
void FinishConstructorDirectContainer(DomJsonoBuilder &builder, uint64_t container_id, size_t start_slot,
                                      size_t start_string, uint64_t shape_hash) {
	auto slot_span = builder.slots.size() - start_slot;
	auto string_byte_span = builder.string_heap.size() - start_string;
	if (slot_span > std::numeric_limits<uint32_t>::max() || string_byte_span > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: container span exceeds storage limits");
	}
	builder.skips[container_id] = ContainerSpan {uint32_t(slot_span), uint32_t(string_byte_span), shape_hash};
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
	FinishConstructorDirectContainer(builder, container_id, start_slot, start_string, plan.shape_hash);
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
	FinishConstructorDirectContainer(builder, container_id, start_slot, start_string, 0);
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
	FinishConstructorDirectContainer(builder, container_id, start_slot, start_string, plan.shape_hash);
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

string_t WriteSingleContainerSkipsBlobInto(Vector &vec, uint32_t slot_span, uint32_t string_byte_span,
                                           uint64_t shape_hash) {
	auto total = sizeof(ContainerMetadataHeader) + sizeof(ContainerSpan);
	auto skips_str = StringVector::EmptyString(vec, total);
	auto skips_dst = skips_str.GetDataWriteable();

	ContainerMetadataHeader header;
	header.container_count = 1;
	header.checkpoint_index_count = 0;
	header.checkpoint_count = 0;
	std::memcpy(skips_dst, &header, sizeof(header));

	ContainerSpan span {slot_span, string_byte_span, shape_hash};
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
	skips_data[row] = WriteSingleContainerSkipsBlobInto(skips_vec, slot_count, uint32_t(values_size), plan.shape_hash);
}

void ExecuteStructConstructor(Vector &input, idx_t count, Vector &result, const JsonoStructPlan &plan,
                              JsonoStructLocalState &lstate) {
	JsonoStructVectorData input_data;
	InitStructConstructorVectorData(input, count, plan, input_data);

	Vector *slots_p, *key_heap_p, *string_heap_p, *skips_p;
	jsono::InitJsonoBodyWrite(result, slots_p, key_heap_p, string_heap_p, skips_p);
	auto &slots_vec = *slots_p;
	auto &key_heap_vec = *key_heap_p;
	auto &string_heap_vec = *string_heap_p;
	auto &skips_vec = *skips_p;
	auto slots_data = FlatVector::GetData<string_t>(slots_vec);
	auto key_heap_data = FlatVector::GetData<string_t>(key_heap_vec);
	auto string_heap_data = FlatVector::GetData<string_t>(string_heap_vec);
	auto skips_data = FlatVector::GetData<string_t>(skips_vec);

	if (plan.strategy == StructValueStrategy::Jsono) {
		for (idx_t row = 0; row < count; row++) {
			JsonoBlobRow blob;
			if (!ReadJsonoRowStrict(input_data.jsono_data, row, blob)) {
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
	auto use_static_keys = plan.strategy == StructValueStrategy::Struct && plan.flat_scalar_object;
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
			// The root object's keys are external (external_root_keys), so FinishContainer
			// could not fingerprint them; set the root span's shape_hash from the plan.
			if (!builder.skips.empty()) {
				builder.skips[0].shape_hash = plan.shape_hash;
			}
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
		for (auto &child : StructType::GetChildTypes(input_type)) {
			// A field named 'body' would collide with the layout's residual field, so it stays in
			// the residual instead of becoming a shred (auto-shred must not error on field names).
			if (child.first != "body" && IsShredValueType(child.second)) {
				bind_data->shreds.emplace_back(child.first, child.second);
			}
		}
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
	Vector plain(JsonoType(), count);
	ExecuteStructConstructor(*input, count, plain, bind_data.plan, lstate);
	JsonoShredFromSpec(plain, count, bind_data.shreds, result);
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
// (a set operation over differently-shredded values), which must throw rather than read as SQL NULL.
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
// its residual extraction; it reinterprets the four-blob prefix directly.
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
