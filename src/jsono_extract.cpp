#include "jsono_extract.hpp"
#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"

#include <string>
#include <vector>

namespace duckdb {
namespace {

using namespace jsono;

struct ExtractPathBindData : public FunctionData {
	ExtractPathBindData(string path_p, vector<PathStep> steps_p) : path(std::move(path_p)), steps(std::move(steps_p)) {
	}

	string path;
	vector<PathStep> steps;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ExtractPathBindData>(path, steps);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ExtractPathBindData>();
		if (path != other.path || steps.size() != other.steps.size()) {
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

struct ExtractLocalState : public FunctionLocalState {
	RankCache rank_cache;
	JsonoBuilder builder;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<ExtractLocalState>();
	}
};

struct Location {
	JsonoCursor cursor;
};

vector<PathStep> LiteralKeyPath(nonstd::string_view name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, string(name), 0});
	return path;
}

vector<PathStep> ArrayIndexPath(idx_t index) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Index, string(), index});
	return path;
}

unique_ptr<FunctionData> JsonoExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto function_name = bound_function.name;
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
	if (arguments[1]->return_type.IsIntegral()) {
		if (!path_value.DefaultTryCastAs(LogicalType::BIGINT)) {
			throw BinderException("%s: path must be BIGINT", function_name);
		}
		auto index = path_value.GetValue<int64_t>();
		if (index < 0) {
			throw BinderException("%s: array index must be non-negative", function_name);
		}
		return make_uniq<ExtractPathBindData>(std::to_string(index), ArrayIndexPath(idx_t(index)));
	}
	if (!path_value.DefaultTryCastAs(LogicalType::VARCHAR)) {
		throw BinderException("%s: path must be VARCHAR", function_name);
	}
	auto path = StringValue::Get(path_value);
	if (path.empty()) {
		ThrowInvalidPath(function_name.c_str(), path);
	}
	vector<PathStep> steps;
	if (path[0] == '$') {
		steps = ParseJsonoPath(path, function_name.c_str());
	} else {
		steps = LiteralKeyPath(path);
	}
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("%s: wildcard paths are not supported", function_name);
		}
	}
	return make_uniq<ExtractPathBindData>(std::move(path), std::move(steps));
}

bool LocatePath(ExtractLocalState &lstate, const vector<PathStep> &steps, const JsonoView &view, Location &location) {
	JsonoCursor cursor;
	if (!LocatePathSteps(&lstate.rank_cache, 0, steps, view, cursor)) {
		return false;
	}
	location.cursor = cursor;
	return true;
}

// Copy the container subtree at `start` into `builder` as a standalone value. The subtree
// is one contiguous range of every stream (slots, string_heap, lengths, nums), so the
// streams are bulk-copied; slots are rewritten only where they reference row-global state:
// KEY heap offsets are rebased (contiguous source key-byte runs are copied once, not per
// key) and container ids are renumbered to start at 0. The writer's counter assigns ids in
// walk order, so the subtree's containers occupy the contiguous id range
// [root_id, root_id + container_count) — its sparse span index, checkpoint index and
// checkpoint records are contiguous record runs in `skips`, copied with the same id/offset
// rebase (span contents and checkpoint deltas are subtree-local already). The result is
// byte-identical to a per-value re-emit of the same subtree.
void BulkEmitSubtree(const JsonoView &view, const JsonoCursor &start, JsonoBuilder &builder) {
	auto root_slot = view.SlotAt(start.pos);
	auto root_tag = SlotTag(root_slot);
	uint64_t root_id = ContainerId(SlotPayload(root_slot));
	ContainerSpan root_span;
	bool root_has_span = view.TryContainerSpan(root_id, root_span);
	size_t slot_span;
	size_t string_bytes;
	size_t length_count;
	size_t num_count;
	if (root_has_span) {
		if (root_span.slot_span < 2 || size_t(root_span.slot_span) > view.Slots() - start.pos) {
			throw InvalidInputException("malformed JSONO: container slot span out of bounds");
		}
		auto end_tag = SlotTag(view.SlotAt(start.pos + root_span.slot_span - 1));
		if (end_tag != (root_tag == tag::OBJ_START ? tag::OBJ_END : tag::ARR_END)) {
			throw InvalidInputException("malformed JSONO: container span end tag mismatch");
		}
		slot_span = root_span.slot_span;
		string_bytes = root_span.string_byte_span;
		length_count = root_span.length_count;
		num_count = root_span.num_count;
	} else {
		// Spanless root: a small flat object (an array or an object with a container child
		// always stores a span). One iterative pass below discovers its end and stream spans.
		if (root_tag != tag::OBJ_START) {
			throw InvalidInputException("malformed JSONO: container span missing");
		}
		slot_span = 0;
		string_bytes = 0;
		length_count = 0;
		num_count = 0;
	}

	if (root_has_span) {
		builder.slots.reserve(slot_span);
	}
	uint64_t container_count = 0;
	// Pending run of contiguous source key bytes; every key belongs to exactly one run and
	// the flush bounds-checks the union of its keys' ranges before any bytes are copied.
	uint64_t run_src_start = 0;
	uint64_t run_src_len = 0;
	size_t run_dest_base = 0;
	// With a root span the slot range and stream spans are known up front, so the pass only
	// copies and rewrites slots. Without one it also walks the stream cursors (LengthAt per
	// string value) and stops when the container depth returns to zero.
	size_t pos = start.pos;
	size_t depth = 0;
	auto length_cursor = start.length_cursor;
	do {
		auto slot = view.SlotAt(pos);
		auto slot_tag = SlotTag(slot);
		switch (slot_tag) {
		case tag::KEY: {
			auto payload = SlotPayload(slot);
			auto offset = KeyOffset(payload);
			auto len = KeyLen(payload);
			if (run_src_len > 0 && offset != run_src_start + run_src_len) {
				auto run = view.KeyHeapSlice(run_src_start, run_src_len);
				builder.key_heap.insert(builder.key_heap.end(), run.begin(), run.end());
				run_src_len = 0;
			}
			if (run_src_len == 0) {
				run_src_start = offset;
				run_dest_base = builder.key_heap.size();
			}
			builder.PushKeySlot(run_dest_base + run_src_len, len);
			run_src_len += len;
			break;
		}
		case tag::OBJ_START:
		case tag::ARR_START: {
			auto payload = SlotPayload(slot);
			if (container_count > 0 && ContainerId(payload) != root_id + container_count) {
				throw InvalidInputException("malformed JSONO: container ids are not sequential in walk order");
			}
			builder.slots.push_back(
			    MakeSlot(slot_tag, MakeContainerPayload(container_count, ContainerChildCount(payload))));
			container_count++;
			depth++;
			break;
		}
		case tag::OBJ_END:
		case tag::ARR_END:
			builder.slots.push_back(slot);
			depth--;
			break;
		default:
			if (!root_has_span) {
				switch (slot_tag) {
				case tag::VAL_STR_HEAP:
					string_bytes += view.LengthAt(length_cursor + length_count);
					length_count++;
					break;
				case tag::VAL_INT60:
				case tag::VAL_DEC60:
					num_count++;
					break;
				case tag::VAL_EXT: {
					auto subtype = ExtSubtype(slot);
					if (subtype == ext_subtype::NUMBER) {
						string_bytes += view.LengthAt(length_cursor + length_count);
						length_count++;
					} else if (subtype < ext_subtype::COUNT) {
						num_count++;
					} else {
						throw InvalidInputException("malformed JSONO: unknown VAL_EXT subtype");
					}
					break;
				}
				case tag::VAL_NULL:
				case tag::VAL_TRUE:
				case tag::VAL_FALSE:
					break;
				default:
					throw InvalidInputException("malformed JSONO: unknown slot tag");
				}
			}
			builder.slots.push_back(slot);
			break;
		}
		pos++;
	} while (root_has_span ? pos < start.pos + slot_span : depth > 0);
	if (run_src_len > 0) {
		auto run = view.KeyHeapSlice(run_src_start, run_src_len);
		builder.key_heap.insert(builder.key_heap.end(), run.begin(), run.end());
	}
	builder.container_count = container_count;

	auto strings = view.StringAt(start.string_cursor, string_bytes);
	builder.string_heap.insert(builder.string_heap.end(), strings.begin(), strings.end());
	if (length_count > 0) {
		auto bytes = view.LengthsBytes(start.length_cursor, length_count);
		builder.lengths.resize(length_count);
		std::memcpy(builder.lengths.data(), bytes, length_count * sizeof(uint32_t));
	}
	if (num_count > 0) {
		auto bytes = view.NumsBytes(start.num_cursor, num_count);
		builder.nums.resize(num_count);
		std::memcpy(builder.nums.data(), bytes, num_count * sizeof(uint64_t));
	}

	// A spanless root cannot have spanned descendants or checkpoints (it is a flat object),
	// so its metadata sections are empty and the searches below are skipped entirely.
	if (!root_has_span && container_count == 1) {
		return;
	}

	auto &metadata = view.MetadataHeader();
	size_t span_lo = 0;
	{
		size_t lo = 0;
		size_t hi = metadata.span_count;
		while (lo < hi) {
			auto mid = lo + (hi - lo) / 2;
			if (view.SpanIdAt(mid) < root_id) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		span_lo = lo;
	}
	size_t span_hi = span_lo;
	while (span_hi < metadata.span_count) {
		auto span_id = view.SpanIdAt(span_hi);
		if (span_id >= root_id + container_count) {
			break;
		}
		builder.span_ids.push_back(uint32_t(span_id - root_id));
		span_hi++;
	}
	if (span_hi > span_lo) {
		auto count = span_hi - span_lo;
		builder.skips.resize(count);
		std::memcpy(builder.skips.data(), view.SpanRecordsBytes(span_lo, count), count * sizeof(ContainerSpan));
	}

	size_t index_lo = 0;
	{
		size_t lo = 0;
		size_t hi = metadata.checkpoint_index_count;
		while (lo < hi) {
			auto mid = lo + (hi - lo) / 2;
			if (view.ObjectCheckpointIndexAt(uint32_t(mid)).container_id < root_id) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		index_lo = lo;
	}
	size_t index_hi = index_lo;
	uint32_t checkpoint_base = 0;
	while (index_hi < metadata.checkpoint_index_count) {
		auto entry = view.ObjectCheckpointIndexAt(uint32_t(index_hi));
		if (entry.container_id >= root_id + container_count) {
			break;
		}
		if (index_hi == index_lo) {
			checkpoint_base = entry.checkpoint_offset;
		} else if (entry.checkpoint_offset < checkpoint_base) {
			throw InvalidInputException("malformed JSONO: object checkpoint offsets are not ascending");
		}
		builder.object_checkpoint_index.push_back(ObjectCheckpointIndex {uint32_t(entry.container_id - root_id),
		                                                                 entry.checkpoint_offset - checkpoint_base,
		                                                                 entry.checkpoint_stride, 0});
		index_hi++;
	}
	if (index_hi > index_lo) {
		auto checkpoint_end = index_hi < metadata.checkpoint_index_count
		                          ? view.ObjectCheckpointIndexAt(uint32_t(index_hi)).checkpoint_offset
		                          : metadata.checkpoint_count;
		if (checkpoint_end < checkpoint_base) {
			throw InvalidInputException("malformed JSONO: object checkpoint offsets are not ascending");
		}
		auto count = size_t(checkpoint_end - checkpoint_base);
		builder.object_checkpoints.resize(count);
		std::memcpy(builder.object_checkpoints.data(), view.CheckpointRecordsBytes(checkpoint_base, count),
		            count * sizeof(ObjectCursorCheckpoint));
	}
}

// Write a scalar extract result directly into the row's six body blobs, skipping the
// JsonoBuilder: slots are the header plus one tag word, skips is the constant empty
// metadata header, and the value's single stream entry comes straight from the source
// view. Tag selection mirrors JsonoBuilder::Emit*, so the bytes match a re-emit.
void WriteScalarExtractRow(JsonoBodyWriter &writer, idx_t row, const JsonoView &view, JsonoCursor cursor) {
	auto scalar = DecodeScalarAt(view, cursor);
	uint64_t slot = 0;
	uint64_t num_word = 0;
	bool has_num = false;
	nonstd::string_view text;
	bool has_text = false;
	switch (scalar.kind) {
	case JsonoScalarKind::String:
		slot = MakeSlot(tag::VAL_STR_HEAP, 0);
		text = scalar.text;
		has_text = true;
		break;
	case JsonoScalarKind::NumberText:
		slot = MakeExtSlot(ext_subtype::NUMBER);
		text = scalar.text;
		has_text = true;
		break;
	case JsonoScalarKind::Int64:
		slot = FitsInt60(scalar.int_value) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64);
		num_word = uint64_t(scalar.int_value);
		has_num = true;
		break;
	case JsonoScalarKind::UInt64:
		slot =
		    scalar.uint_value <= uint64_t(INT60_MAX) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::UINT64);
		num_word = scalar.uint_value;
		has_num = true;
		break;
	case JsonoScalarKind::Double:
		if (!std::isfinite(scalar.double_value)) {
			throw InvalidInputException("jsono: cannot store non-finite double value (NaN/Infinity)");
		}
		slot = MakeExtSlot(ext_subtype::DOUBLE);
		std::memcpy(&num_word, &scalar.double_value, sizeof(num_word));
		has_num = true;
		break;
	case JsonoScalarKind::Dec60:
		slot = MakeSlot(tag::VAL_DEC60, 0);
		num_word = MakeDec60Payload(scalar.dec_negative, scalar.dec_mantissa, scalar.dec_scale);
		has_num = true;
		break;
	case JsonoScalarKind::Bool:
		slot = MakeSlot(scalar.bool_value ? tag::VAL_TRUE : tag::VAL_FALSE, 0);
		break;
	case JsonoScalarKind::Null:
		slot = MakeSlot(tag::VAL_NULL, 0);
		break;
	}
	char slots_buf[JSONO_HEADER_SIZE + sizeof(uint64_t)];
	JsonoHeader header;
	header.magic = MAGIC;
	header.version = VERSION;
	header.flags = flags::SORTED_KEYS;
	header.reserved = 0;
	std::memcpy(slots_buf, &header, JSONO_HEADER_SIZE);
	std::memcpy(slots_buf + JSONO_HEADER_SIZE, &slot, sizeof(slot));
	static constexpr ContainerMetadataHeader EMPTY_METADATA {0, 0, 0};
	writer.data[BODY_SLOTS][row] = WriteBlobInto(writer.Slots(), slots_buf, sizeof(slots_buf));
	writer.data[BODY_KEY_HEAP][row] = WriteBlobInto(writer.KeyHeap(), nullptr, 0);
	writer.data[BODY_STRING_HEAP][row] = WriteBlobInto(writer.StringHeap(), text.data(), text.size());
	writer.data[BODY_SKIPS][row] =
	    WriteBlobInto(writer.Skips(), reinterpret_cast<const char *>(&EMPTY_METADATA), sizeof(EMPTY_METADATA));
	if (has_text) {
		auto len = uint32_t(text.size());
		writer.data[BODY_LENGTHS][row] =
		    WriteBlobInto(writer.Lengths(), reinterpret_cast<const char *>(&len), sizeof(len));
	} else {
		writer.data[BODY_LENGTHS][row] = WriteBlobInto(writer.Lengths(), nullptr, 0);
	}
	if (has_num) {
		writer.data[BODY_NUMS][row] =
		    WriteBlobInto(writer.Nums(), reinterpret_cast<const char *>(&num_word), sizeof(num_word));
	} else {
		writer.data[BODY_NUMS][row] = WriteBlobInto(writer.Nums(), nullptr, 0);
	}
}

void JsonoExtractExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ExtractPathBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ExtractLocalState>();
	auto count = args.size();

	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);

	JsonoBodyWriter writer;
	writer.Init(result);

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			writer.SetRowNull(row);
			continue;
		}
		Location location;
		if (!LocatePath(lstate, bind_data.steps, view, location)) {
			reader.CheckPathMiss(view, bind_data.steps);
			writer.SetRowNull(row);
			continue;
		}
		auto cursor = location.cursor;
		auto found_tag = SlotTag(view.SlotAt(cursor.pos));
		if (found_tag == tag::OBJ_START || found_tag == tag::ARR_START) {
			reader.CheckContainerRead(view, bind_data.steps);
			lstate.builder.Reset();
			BulkEmitSubtree(view, cursor, lstate.builder);
			writer.WriteRow(row, lstate.builder);
		} else {
			WriteScalarExtractRow(writer, row, view, cursor);
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void JsonoExtractStringExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ExtractPathBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ExtractLocalState>();
	auto count = args.size();

	JsonoRowReader reader;
	reader.InitPointRead(args.data[0], count);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	std::string scratch;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		Location location;
		if (!LocatePath(lstate, bind_data.steps, view, location)) {
			reader.CheckPathMiss(view, bind_data.steps);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto cursor = location.cursor;
		auto slot_tag = SlotTag(view.SlotAt(cursor.pos));
		scratch.clear();
		if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
			reader.CheckContainerRead(view, bind_data.steps);
			AppendJsonValueText(view, cursor, scratch, 0);
			result_data[row] = StringVector::AddString(result, scratch.data(), scratch.size());
			continue;
		}
		auto scalar = DecodeScalarAt(view, cursor);
		if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
			result_data[row] = StringVector::AddString(result, scalar.text.data(), scalar.text.size());
		} else if (RenderExtractText(scalar, scratch)) {
			FlatVector::SetNull(result, row, true);
		} else {
			result_data[row] = StringVector::AddString(result, scratch.data(), scratch.size());
		}
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Bind for the public jsono_extract/jsono_extract_string sets, which accept ANY so a shredded
// value reaches bind: redeclare arg0 to plain JSONO (the binder inserts the lossless reconstruct
// cast) and reject non-JSONO loudly, then run the constant-path bind. The core-named "json_extract"
// and "->>" sets stay hard-typed on plain JSONO and use JsonoExtractBind directly.
unique_ptr<FunctionData> JsonoExtractBindAny(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	bound_function.arguments[0] = JsonoResolveJsonoArgument(context, *arguments[0], bound_function.name, true);
	return JsonoExtractBind(context, bound_function, arguments);
}

ScalarFunction MakeExtractFunction(string name, LogicalType path_type, LogicalType arg0_type = JsonoType(),
                                   bind_scalar_function_t bind = JsonoExtractBind) {
	ScalarFunction fun(std::move(name), {std::move(arg0_type), std::move(path_type)}, JsonoType(), JsonoExtractExecute,
	                   bind, nullptr, nullptr, ExtractLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

ScalarFunction MakeExtractStringFunction(string name, LogicalType path_type, LogicalType arg0_type = JsonoType(),
                                         bind_scalar_function_t bind = JsonoExtractBind) {
	ScalarFunction fun(std::move(name), {std::move(arg0_type), std::move(path_type)}, LogicalType::VARCHAR,
	                   JsonoExtractStringExecute, bind, nullptr, nullptr, ExtractLocalState::Init);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return fun;
}

} // namespace

// Native jsono extract functions, exposed so the optimizer can read a non-shred path
// straight off the shredded residual (a JSONO value) instead of routing it through
// core json's serialize-then-parse path.
ScalarFunction JsonoExtractStringFunction(const LogicalType &path_type) {
	return MakeExtractStringFunction("jsono_extract_string", path_type);
}

ScalarFunction JsonoExtractFunction(const LogicalType &path_type) {
	return MakeExtractFunction("jsono_extract", path_type);
}

void RegisterJsonoExtract(ExtensionLoader &loader) {
	ScalarFunctionSet extract("jsono_extract");
	extract.AddFunction(
	    MakeExtractFunction("jsono_extract", LogicalType::VARCHAR, LogicalType::ANY, JsonoExtractBindAny));
	extract.AddFunction(
	    MakeExtractFunction("jsono_extract", LogicalType::BIGINT, LogicalType::ANY, JsonoExtractBindAny));
	loader.RegisterFunction(std::move(extract));

	// json_extract is core-owned: keep it hard-typed on plain JSONO (an ANY overload would tie with
	// core's on cast cost and the first-registered — core's — would win, so ANY is unreliable here).
	ScalarFunctionSet core_extract("json_extract");
	core_extract.AddFunction(MakeExtractFunction("json_extract", LogicalType::VARCHAR));
	core_extract.AddFunction(MakeExtractFunction("json_extract", LogicalType::BIGINT));
	loader.RegisterFunction(std::move(core_extract));

	ScalarFunctionSet extract_string("jsono_extract_string");
	extract_string.AddFunction(
	    MakeExtractStringFunction("jsono_extract_string", LogicalType::VARCHAR, LogicalType::ANY, JsonoExtractBindAny));
	extract_string.AddFunction(
	    MakeExtractStringFunction("jsono_extract_string", LogicalType::BIGINT, LogicalType::ANY, JsonoExtractBindAny));
	loader.RegisterFunction(std::move(extract_string));

	ScalarFunctionSet arrow_string("->>");
	arrow_string.AddFunction(MakeExtractStringFunction("->>", LogicalType::VARCHAR));
	arrow_string.AddFunction(MakeExtractStringFunction("->>", LogicalType::BIGINT));
	loader.RegisterFunction(std::move(arrow_string));
}

} // namespace duckdb
