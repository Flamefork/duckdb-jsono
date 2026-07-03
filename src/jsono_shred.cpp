#include "jsono_shred.hpp"
#include "jsono.hpp"
#include "jsono_dom.hpp"
#include "jsono_locate.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_scalar_write.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "string_view.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace duckdb {

// Build the residual: `view` minus the located leaf of each path (defined in jsono_merge.cpp).
void JsonoEmitObjectStrippingPaths(const jsono::JsonoView &view,
                                   const std::vector<const std::vector<PathStep> *> &paths,
                                   jsono::JsonoBuilder &builder);

// Build the residual skeleton: `view` minus the located scalar leaves (`scalar_paths`) AND, for
// each array shred, the located array rebuilt with the lifted element subfields stripped per
// element (`array_specs`). The array stays in place as the skeleton — its length, element order,
// non-object/null elements and per-element tail carry the lossless reconstruct (jsono_merge.cpp).
void JsonoEmitResidualSkeleton(const jsono::JsonoView &view,
                               const std::vector<const std::vector<PathStep> *> &scalar_paths,
                               const std::vector<const JsonoArrayShredSpec *> &array_specs,
                               jsono::JsonoBuilder &builder);

namespace {

using namespace jsono;
using namespace duckdb_yyjson;

// One subfield lifted out of each array element into the LIST<STRUCT> shred column.
struct ShredArraySubfield {
	string name; // element subfield key (= struct field name)
	JsonoScalarPrimitive primitive;
};

struct ShredField {
	JsonoPathSpec path; // canonical path; also the shred struct field name
	// kind == Scalar: one leaf scalar lifted at `steps` (use `primitive`). kind == Array: `steps`
	// addresses a regular array; `subfields` are the element leaves lifted into a parallel LIST<STRUCT>
	// column. kind == ScalarArray: `steps` addresses a regular array; each whole scalar element lifts
	// into a parallel LIST<element_type> column (use `element_primitive`).
	ShredKind kind = ShredKind::Scalar;
	JsonoScalarPrimitive primitive;         // valid when kind == Scalar
	vector<ShredArraySubfield> subfields;   // valid when kind == Array, element-struct order
	JsonoScalarPrimitive element_primitive; // valid when kind == ScalarArray
};

struct ShredBindData : public FunctionData {
	vector<ShredField> fields;
	LogicalType return_type;
	// Single-pass reshred over a shredded input: each target field either copies its source shred
	// (keep_src holds the source shred index) or extracts a new path from the residual
	// (INVALID_INDEX). Source shreds the target drops or retypes (return_src) are folded back
	// into the residual by a vectorized partial overlay before the per-row pass.
	bool reshred_active = false;
	vector<idx_t> keep_src;
	vector<idx_t> return_src;
	// One-pass text write (JsonoShredFromTextExecute): a trie over the object-key shred paths
	// lets pass 1 of the direct DOM writer capture and strip the leaves while sizing — the
	// throwaway plain materialization and the per-row locate/strip/re-emit disappear. Every shred
	// path is an object-key chain (the bind rejects the rest), so every field enters the trie.
	// Duplicate paths in the spec (e.g. '$.a' and 'a') fall back to the two-pass path.
	bool one_pass_text = false;
	std::vector<jsono_dom::DomShredTrieNode> trie;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ShredBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		// return_type encodes every shred's name (the path) and type, so it fully identifies the
		// spec; the reshred plan is derived from the argument type, compared via keep_src.
		auto &other = other_p.Cast<ShredBindData>();
		return return_type == other.return_type && reshred_active == other.reshred_active &&
		       keep_src == other.keep_src && return_src == other.return_src;
	}
};

struct ShredLocalState : public FunctionLocalState {
	JsonoBuilder builder;
	std::string text;
	std::vector<const std::vector<PathStep> *> strip_paths;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		(void)state;
		(void)expr;
		(void)bind_data;
		return make_uniq<ShredLocalState>();
	}
};

// A shred path must be a non-empty pure object-key sequence: the residual emit removes a key at
// each step, and only an object key can be re-filled by the object overlay on reconstruction. An
// array-index or root '$' path is rejected at bind (ParseShredPathSpec) so a set lane always means
// the value was stripped from the residual — the value is stored exactly once.
bool IsObjectKeyPath(const vector<PathStep> &steps) {
	for (auto &step : steps) {
		if (step.kind != PathStepKind::Key) {
			return false;
		}
	}
	return !steps.empty();
}

JsonoPathSpec ParseShredPathSpec(const string &path) {
	// The spec DSL reserves a leading `$` for the JSONPath form: a `$`-leading spec key that is not
	// `$.`-rooted is rejected here, so a literal key like `$x` must be spelled `$."$x"`. This is the
	// spec-parse contract, distinct from ShredNamePath, which resolves already-canonical lane names
	// and reads a bare `$x` as a literal key.
	auto steps = path.size() > 0 && path[0] == '$' ? ParseJsonoPath(path, "jsono shred") : LiteralKeyPath(path);
	for (auto &step : steps) {
		if (step.kind == PathStepKind::Wildcard) {
			throw BinderException("jsono shred: wildcard paths are not supported ('%s')", path);
		}
	}
	if (!IsObjectKeyPath(steps)) {
		throw BinderException("jsono shred: shred path '%s' must be a non-empty object-key path "
		                      "(array-index and root '$' paths cannot be shredded)",
		                      path);
	}
	return JsonoPathSpec(path, std::move(steps));
}

LogicalType ShredFieldType(const ShredField &field) {
	switch (field.kind) {
	case ShredKind::Scalar:
		return JsonoScalarPrimitiveLogicalType(field.primitive);
	case ShredKind::Array: {
		child_list_t<LogicalType> children;
		for (auto &subfield : field.subfields) {
			children.emplace_back(subfield.name, JsonoScalarPrimitiveLogicalType(subfield.primitive));
		}
		return LogicalType::LIST(LogicalType::STRUCT(std::move(children)));
	}
	case ShredKind::ScalarArray:
		return LogicalType::LIST(JsonoScalarPrimitiveLogicalType(field.element_primitive));
	}
	throw InternalException("jsono: unhandled shred kind");
}

struct YyjsonDoc {
	explicit YyjsonDoc(yyjson_doc *doc_p) : doc(doc_p) {
	}
	~YyjsonDoc() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	YyjsonDoc(const YyjsonDoc &) = delete;
	YyjsonDoc &operator=(const YyjsonDoc &) = delete;

	yyjson_doc *doc;
};

// Populate a shred field's kind and shred-specific members (subfields / element type) from its
// logical type. Returns false when the type is not a recognizable shred type — the caller raises
// the exception that fits its context. The object-key-path gate likewise stays caller-side.
bool FillShredFieldFromType(const LogicalType &type, ShredField &field) {
	if (TypeToJsonoScalarPrimitive(type, field.primitive)) {
		field.kind = ShredKind::Scalar;
		return true;
	}
	if (IsShredArrayType(type)) {
		field.kind = ShredKind::Array;
		auto &element = ListType::GetChildType(type);
		for (auto &sub : StructType::GetChildTypes(element)) {
			ShredArraySubfield subfield;
			subfield.name = sub.first;
			subfield.primitive = JsonoScalarPrimitiveFromType(sub.second, "jsono shred bind");
			field.subfields.push_back(std::move(subfield));
		}
		return true;
	}
	if (IsShredScalarArrayType(type)) {
		field.kind = ShredKind::ScalarArray;
		field.element_primitive = JsonoScalarPrimitiveFromType(ListType::GetChildType(type), "jsono shred bind");
		return true;
	}
	return false;
}

void BindShredField(const string &path, const LogicalType &type, const string &type_name, ShredField &field) {
	JsonoValidateShredFieldName(path);
	field.path = ParseShredPathSpec(path);
	if (!FillShredFieldFromType(type, field)) {
		throw BinderException("jsono shred: unsupported shred type '%s'", type_name);
	}
}

// Resolve one spec entry's type string into a shred field: a scalar leaf shred, or a
// LIST<STRUCT<...>> array shred lifting the element subfields of the array at `path`.
void BindShredFieldType(const string &type_name, ClientContext &context, const string &path, ShredField &field) {
	LogicalType type;
	try {
		type = TransformStringToLogicalType(type_name, context);
	} catch (const std::exception &) {
		throw BinderException("jsono shred: unsupported shred type '%s'", type_name);
	}
	BindShredField(path, type, type_name, field);
}

// The shredding spec is a constant STRUCT mapping each path to its shred type string, e.g.
// {'$.kind': 'VARCHAR', '$.commit.seq': 'BIGINT'} — the field name is the path, the field
// value is the stringified shred type (same idea as read_json's `columns` and Parquet's
// SHREDDING option). An array shred names a LIST<STRUCT<...>> type, e.g.
// {'$.products': 'STRUCT(id UBIGINT, name VARCHAR)[]'}.
unique_ptr<ShredBindData> ParseShredSpec(const Value &spec, ClientContext &context) {
	if (spec.IsNull() || spec.type().id() != LogicalTypeId::STRUCT) {
		throw BinderException(
		    "jsono shred: shredding spec must be a non-NULL STRUCT mapping path to type, e.g. {'$.kind': 'VARCHAR'}");
	}
	auto &child_types = StructType::GetChildTypes(spec.type());
	auto &child_values = StructValue::GetChildren(spec);

	auto bind_data = make_uniq<ShredBindData>();

	for (idx_t i = 0; i < child_types.size(); i++) {
		auto &path = child_types[i].first;
		auto type_value = child_values[i];
		if (type_value.IsNull() || !type_value.DefaultTryCastAs(LogicalType::VARCHAR)) {
			throw BinderException("jsono shred: type for '%s' must be a type string", path);
		}
		auto type_name = StringValue::Get(type_value);
		ShredField field;
		BindShredFieldType(type_name, context, path, field);
		bind_data->fields.push_back(std::move(field));
	}
	if (bind_data->fields.empty()) {
		throw BinderException("jsono shred: empty shredding spec");
	}
	// Canonical shred order (sorted by path) makes the shredded type a pure function of the shred set,
	// not of spec order: the execution writes shreds by index, so the field write order and the type's
	// shred order must agree — sort both here. The shred-set marker hash is order-independent regardless.
	std::sort(bind_data->fields.begin(), bind_data->fields.end(),
	          [](const ShredField &a, const ShredField &b) { return a.path.text < b.path.text; });
	child_list_t<LogicalType> shreds;
	for (auto &field : bind_data->fields) {
		shreds.emplace_back(field.path.text, ShredFieldType(field));
	}
	// JsonoShreddedStructType builds the canonical layout type purely from the shred set, so a column
	// declared from the same shreds is byte-identical and a mismatched cast fails loud via the manifest.
	bind_data->return_type = JsonoShreddedStructType(shreds);

	bind_data->one_pass_text = true;
	bind_data->trie.emplace_back();
	for (idx_t f = 0; f < bind_data->fields.size(); f++) {
		auto &field = bind_data->fields[f];
		if (field.kind != ShredKind::Scalar) {
			// Array shreds (object or scalar) extract per element from the parsed value (the two-pass
			// plain shred path); the one-pass text DOM writer's trie addresses object-key scalar leaves only.
			bind_data->one_pass_text = false;
			continue;
		}
		uint32_t node = 0;
		for (auto &step : field.path.steps) {
			uint32_t next = std::numeric_limits<uint32_t>::max();
			for (auto &edge : bind_data->trie[node].children) {
				if (edge.first == step.key) {
					next = edge.second;
					break;
				}
			}
			if (next == std::numeric_limits<uint32_t>::max()) {
				bind_data->trie.emplace_back();
				next = uint32_t(bind_data->trie.size() - 1);
				bind_data->trie[node].children.emplace_back(step.key, next);
			}
			node = next;
		}
		if (bind_data->trie[node].field >= 0) {
			// Two spec entries name the same key path (e.g. '$.a' and 'a'); the per-row
			// locate-and-strip path handles them with the documented semantics.
			bind_data->one_pass_text = false;
			break;
		}
		bind_data->trie[node].field = int64_t(f);
	}
	return bind_data;
}

unique_ptr<FunctionData> JsonoShredBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->GetAlias() != "shredding") {
		throw BinderException("jsono(): unknown argument '%s' (pass shredding := {'<path>': '<type>', ...})",
		                      arguments[1]->GetAlias());
	}
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("jsono shred: shredding spec must be constant");
	}
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto spec_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	auto bind_data = ParseShredSpec(spec_value, context);
	bound_function.return_type = bind_data->return_type;
	auto &arg_type = arguments[0]->return_type;
	if (arg_type.id() == LogicalTypeId::STRUCT) {
		JsonoLayoutType src_layout;
		if (!TryParseJsonoLayoutType(arg_type, src_layout)) {
			throw BinderException("jsono shred: value must be JSON text or a jsono value");
		}
		// Reshred a shredded value in one pass: surviving shreds copy through, dropped/retyped
		// scalar shreds (return_src) fold back into the residual via a partial overlay, and new
		// scalar target paths extract from that residual. Array shreds can ride this path only when
		// they survive unchanged: their residual skeleton and list lane are copied as a pair.
		if (src_layout.kind == JsonoLayoutKind::Shredded) {
			bind_data->keep_src.assign(bind_data->fields.size(), DConstants::INVALID_INDEX);
			vector<bool> src_kept(src_layout.shreds.size(), false);
			for (idx_t f = 0; f < bind_data->fields.size(); f++) {
				auto field_type = ShredFieldType(bind_data->fields[f]);
				for (idx_t k = 0; k < src_layout.shreds.size(); k++) {
					if (src_layout.shreds[k].first == bind_data->fields[f].path.text &&
					    src_layout.shreds[k].second == field_type) {
						bind_data->keep_src[f] = k;
						src_kept[k] = true;
						break;
					}
				}
			}
			bool can_single_pass = true;
			for (idx_t f = 0; f < bind_data->fields.size(); f++) {
				if (bind_data->fields[f].kind != ShredKind::Scalar &&
				    bind_data->keep_src[f] == DConstants::INVALID_INDEX) {
					can_single_pass = false;
					break;
				}
			}
			for (idx_t k = 0; k < src_layout.shreds.size() && can_single_pass; k++) {
				if (IsShredListType(src_layout.shreds[k].second) && !src_kept[k]) {
					can_single_pass = false;
				}
			}
			if (!can_single_pass) {
				bind_data->keep_src.clear();
				bound_function.arguments[0] = JsonoType();
				return std::move(bind_data);
			}
			for (idx_t k = 0; k < src_layout.shreds.size(); k++) {
				if (!src_kept[k]) {
					bind_data->return_src.push_back(k);
				}
			}
			bind_data->reshred_active = true;
			bound_function.arguments[0] = arg_type;
		} else {
			bound_function.arguments[0] = JsonoType();
		}
	}
	return std::move(bind_data);
}

// Write a scalar shred's VALUE lane for a field and return whether the original value is losslessly
// captured by the shred type. Every shred path is an object-key chain (the bind rejects the rest),
// so this return value alone decides stripping: true means the path is stripped from the residual. A
// shred stores the value only when it round-trips through the shred type exactly (a real JSON string
// in a VARCHAR shred, a matching-kind scalar in a typed shred); a divert leaves the lane NULL and the
// value in the residual, and an absent path / explicit null leaves the lane NULL with nothing
// stripped — so a set lane always means the path is stripped, the value is stored exactly once.
// `complete` (optional out) is the per-shred, per-row flag a bare lane read may be trusted by: true
// when the lane holds the full `->>` answer (absent path, explicit null, a fitting scalar), false
// when the value stayed in the residual (divert, container) and the read must fall back / reconstruct.
bool WriteShred(const ShredField &field, const JsonoView &view, const JsonoPathLocation &location, Vector &shred,
                idx_t row, std::string &scratch, bool *complete = nullptr) {
	auto set_complete = [&](bool value) {
		if (complete) {
			*complete = value;
		}
	};
	if (!location.found) {
		// Absent path: `->>` is NULL, a bare read of the NULL lane is correct.
		FlatVector::SetNull(shred, row, true);
		set_complete(true);
		return false;
	}
	auto value_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		// A container never fits a scalar lane: it stays in the residual and the read
		// reconstructs / falls back.
		FlatVector::SetNull(shred, row, true);
		set_complete(false);
		return false;
	}
	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	if (scalar.kind == JsonoScalarKind::Null) {
		// Explicit JSON null: `->>` is NULL, a bare read of the NULL lane is correct.
		FlatVector::SetNull(shred, row, true);
		set_complete(true);
		return false;
	}
	if (JsonoScalarFitsPrimitive(scalar, field.primitive)) {
		WriteJsonoFittingScalarLane(shred, row, scalar, field.primitive, scratch);
		set_complete(true);
		return true;
	}
	// A present scalar that did not match its shred's kind gate: the value stays in the
	// residual, so a bare lane read would miss it — not complete.
	FlatVector::SetNull(shred, row, true);
	set_complete(false);
	return false;
}

// Shared array-lane shred scaffold: both shred lanes (LIST<STRUCT> objects, LIST<scalar>) reject a
// missing path / non-array / absent location with a NULL list, count elements via SkipValueFast,
// reserve list capacity, then walk each element once writing into list slot `child`. Only the
// per-element body differs between lanes, so it is passed as `write_element(elem, child) -> bool`
// returning whether that element was losslessly lifted; the OR of those drives the manifest return.
// `setup` runs once after capacity is reserved, lane-side, to flatten the lane's child vectors (the
// FLAT setup is lane-specific — LIST<STRUCT> must flatten every subfield, LIST<scalar> only the one
// child). The caller's NULL-slot handling for non-conforming elements lives inside write_element.
template <class Setup, class WriteElement>
bool WriteArrayLaneShred(const JsonoView &view, const JsonoPathLocation &location, Vector &list_vec, idx_t row,
                         Setup setup, WriteElement write_element) {
	if (!location.found || SlotTag(view.SlotAt(location.cursor.pos)) != tag::ARR_START) {
		SetListRowNull(list_vec, row);
		return false;
	}
	auto element_count = idx_t(CountArrayElements(view, location.cursor));
	JsonoCursor first = location.cursor;
	first.pos++; // ARR_START consumes no stream entries, so the stream cursors are the first element's
	auto start = ListVector::GetListSize(list_vec);
	EnsureListCapacity(list_vec, start + element_count);
	setup();
	bool stripped_any = false;
	JsonoCursor elem = first;
	for (idx_t i = 0; i < element_count; i++) {
		stripped_any |= write_element(elem, start + i);
		SkipValueFast(view, elem);
	}
	FinishListRow(list_vec, row, start, element_count);
	return stripped_any;
}

// Write the LIST<STRUCT> shred column for one row's array at `location` and return whether any
// element subfield was losslessly stripped (so the manifest must list the array path). A missing
// path, a non-array value, or an absent location writes a NULL list (the value stays whole in the
// residual). Each array element becomes one struct row: an object element's lifted subfields are
// written typed (NULL where absent/diverted/type-mismatched), a non-object/null element a NULL
// struct. The skeleton residual emit (jsono_merge.cpp) strips exactly the subfields whose write
// here reports strippable, since both gate on JsonoScalarFitsPrimitive.
bool WriteArrayShred(const ShredField &field, const JsonoView &view, const JsonoPathLocation &location,
                     Vector &list_vec, idx_t row, std::string &scratch) {
	auto &struct_vec = ListVector::GetEntry(list_vec);
	auto &subfield_vecs = StructVector::GetEntries(struct_vec);
	return WriteArrayLaneShred(
	    view, location, list_vec, row,
	    [&]() {
		    struct_vec.SetVectorType(VectorType::FLAT_VECTOR);
		    for (auto &sub : subfield_vecs) {
			    sub->SetVectorType(VectorType::FLAT_VECTOR);
		    }
	    },
	    [&](JsonoCursor elem, idx_t child) -> bool {
		    if (SlotTag(view.SlotAt(elem.pos)) != tag::OBJ_START) {
			    // Non-object element (null/scalar/array): no subfields, a NULL struct row.
			    FlatVector::SetNull(struct_vec, child, true);
			    for (auto &sub : subfield_vecs) {
				    FlatVector::SetNull(*sub, child, true);
			    }
			    return false;
		    }
		    FlatVector::Validity(struct_vec).SetValid(child);
		    bool stripped = false;
		    for (idx_t j = 0; j < field.subfields.size(); j++) {
			    auto &sub = field.subfields[j];
			    ShredField subfield;
			    subfield.path.steps.push_back(PathStep {PathStepKind::Key, sub.name, 0});
			    subfield.primitive = sub.primitive;
			    JsonoCursor probe = elem;
			    JsonoPathLocation subloc;
			    if (LocatePathStep(nullptr, 0, view, subfield.path.steps[0], probe)) {
				    subloc = JsonoPathLocation {probe, true};
			    }
			    stripped |= WriteShred(subfield, view, subloc, *subfield_vecs[j], child, scratch);
		    }
		    return stripped;
	    });
}

// Write the LIST<scalar> shred column for one row's array at `location` and return whether any
// element was losslessly lifted (so the manifest must list the array path). A missing path, a
// non-array value, or an absent location writes a NULL list (the value stays whole in the residual).
// Each array element becomes one list slot: a scalar that fits the lane type is written typed and
// reported lifted; a non-conforming scalar, a null/object/array element, or a type mismatch writes a
// NULL slot and stays verbatim in the residual skeleton. The skeleton emit (jsono_merge.cpp) drops
// exactly the elements lifted here, since both gate on JsonoScalarFitsPrimitive.
bool WriteScalarArrayShred(const ShredField &field, const JsonoView &view, const JsonoPathLocation &location,
                           Vector &list_vec, idx_t row, std::string &scratch) {
	auto &child_vec = ListVector::GetEntry(list_vec);
	return WriteArrayLaneShred(
	    view, location, list_vec, row, [&]() { child_vec.SetVectorType(VectorType::FLAT_VECTOR); },
	    [&](JsonoCursor elem, idx_t child) -> bool {
		    auto value_tag = SlotTag(view.SlotAt(elem.pos));
		    if (value_tag != tag::OBJ_START && value_tag != tag::ARR_START) {
			    JsonoCursor probe = elem;
			    auto scalar = DecodeScalarAt(view, probe);
			    if (JsonoScalarFitsPrimitive(scalar, field.element_primitive)) {
				    WriteJsonoFittingScalarLane(child_vec, child, scalar, field.element_primitive, scratch);
				    return true;
			    }
		    }
		    // Kept verbatim in the residual skeleton: a NULL slot disambiguates it from a lifted element.
		    FlatVector::SetNull(child_vec, child, true);
		    return false;
	    });
}

void ApplyShredFields(Vector &input_vec, idx_t count, const vector<ShredField> &fields, Vector &result,
                      ShredLocalState &lstate) {
	// The input is plain JSONO, which never carries a manifest of its own: the reader fails
	// loud on a narrowed row, whose re-emit below would otherwise replace the old manifest
	// with a fresh one and launder the loss into permanent undetectability.
	JsonoRowReader reader;
	reader.Init(input_vec, count);

	JsonoBodyWriter writer;
	writer.Init(result);
	// Shred-set marker: the layout hash on every value row (NULL only on a SQL-NULL row — divert is
	// per-shred, in each scalar shred's `complete` flag, written below).
	Vector *set_out = &JsonoShredSetVector(result);
	set_out->SetVectorType(VectorType::FLAT_VECTOR);
	auto layout_hash = int64_t(JsonoLayoutHashOf(result.GetType()));
	vector<Vector *> shred_children;
	vector<Vector *> complete_children; // parallel to fields; nullptr for array shreds
	shred_children.reserve(fields.size());
	complete_children.reserve(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_children.push_back(&JsonoShredVector(result, f));
		shred_children.back()->SetVectorType(VectorType::FLAT_VECTOR);
		if (fields[f].kind == ShredKind::Scalar) {
			complete_children.push_back(&JsonoShredCompleteVector(result, f));
			complete_children.back()->SetVectorType(VectorType::FLAT_VECTOR);
		} else {
			complete_children.push_back(nullptr);
		}
	}

	auto null_shred_row = [&](idx_t row) {
		writer.SetRowNull(row);
		for (auto &shred : shred_children) {
			FlatVector::SetNull(*shred, row, true);
		}
		JsonoSetRowMarkerNull(result, row);
	};

	// Per-field manifest entry bytes, built once: the per-row manifest concatenates the entries
	// of the paths actually stripped from that row's residual (fields are already in canonical
	// sorted order, so the manifest is too).
	vector<JsonoShredManifestEntryBytes> manifest_entries(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(fields[f].path.text, ShredFieldType(fields[f]));
	}

	// Array shred specs (path + lifted-element description), built once for the residual skeleton
	// emit. The skeleton emit strips, per array element, exactly what the WriteArrayShred /
	// WriteScalarArrayShred pass reports lifted (both gate on JsonoScalarFitsPrimitive), keeping the
	// array as the position carrier.
	bool has_array = false;
	vector<JsonoArrayShredSpec> array_specs;
	for (auto &field : fields) {
		if (field.kind == ShredKind::Scalar) {
			continue;
		}
		has_array = true;
		JsonoArrayShredSpec spec;
		spec.path = field.path.steps;
		spec.kind = field.kind;
		if (field.kind == ShredKind::ScalarArray) {
			spec.element_primitive = field.element_primitive;
		} else {
			for (auto &sub : field.subfields) {
				spec.subfields.emplace_back(sub.name, sub.primitive);
			}
		}
		array_specs.push_back(std::move(spec));
	}
	vector<const JsonoArrayShredSpec *> array_spec_ptrs;
	for (auto &spec : array_specs) {
		array_spec_ptrs.push_back(&spec);
	}

	std::string manifest;
	vector<idx_t> stripped_fields;

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			null_shred_row(row);
			continue;
		}

		FlatVector::Validity(result).SetValid(row);
		FlatVector::GetData<int64_t>(*set_out)[row] = layout_hash;
		lstate.strip_paths.clear();
		stripped_fields.clear();
		for (idx_t f = 0; f < fields.size(); f++) {
			auto &field = fields[f];
			auto location = LocatePath(view, field.path.steps);
			if (field.kind != ShredKind::Scalar) {
				// Array shreds carry no completeness flag: a per-element diversion has no
				// bare-struct_extract fast path to invalidate (the array is always reconstructed).
				bool stripped = field.kind == ShredKind::ScalarArray
				                    ? WriteScalarArrayShred(field, view, location, *shred_children[f], row, lstate.text)
				                    : WriteArrayShred(field, view, location, *shred_children[f], row, lstate.text);
				if (stripped) {
					stripped_fields.push_back(f);
				}
				continue;
			}
			bool complete = true;
			bool strippable = WriteShred(field, view, location, *shred_children[f], row, lstate.text, &complete);
			FlatVector::GetData<int8_t>(*complete_children[f])[row] = complete ? 1 : 0;
			if (strippable) {
				lstate.strip_paths.push_back(&field.path.steps);
				stripped_fields.push_back(f);
			}
		}

		lstate.builder.Reset();
		if (has_array) {
			JsonoEmitResidualSkeleton(view, lstate.strip_paths, array_spec_ptrs, lstate.builder);
		} else {
			JsonoEmitObjectStrippingPaths(view, lstate.strip_paths, lstate.builder);
		}
		const std::string *manifest_ptr = nullptr;
		if (!stripped_fields.empty()) {
			manifest.clear();
			JsonoAppendShredManifest(manifest, manifest_entries, stripped_fields);
			manifest_ptr = &manifest;
		}
		writer.WriteRow(row, lstate.builder, manifest_ptr);
	}
}

// Single-pass reshred of a shredded input whose shreds all survive into the target (see
// ShredBindData::reshred_active): kept shreds are copied vector-at-a-time, only the target's new
// paths are located in and stripped from the residual, and the manifest carries over the kept
// entries plus the newly stripped ones — no reconstruct-to-plain materialization, no re-locating
// of paths that were already shredded.
void ApplyReshredShredded(Vector &input_vec, idx_t count, const ShredBindData &bind_data, Vector &result,
                          ShredLocalState &lstate) {
	auto &fields = bind_data.fields;

	input_vec.Flatten(count);

	// Shreds the target drops (or retypes) fold back into the residual first, as one vectorized
	// partial overlay; the per-row pass then locates/strips against that merged residual, while
	// the manifest still reads from the ORIGINAL residual (the overlay output carries none).
	// Reading the original residual, the reader verifies each row's manifest against the
	// source's shreds — a narrowed input row fails loud here too; the merged residual is plain
	// and unmanifested by construction (the overlay verified the source itself).
	Vector returned(JsonoType(), count);
	JsonoRowReader reader;
	bool reads_merged = !bind_data.return_src.empty();
	if (reads_merged) {
		JsonoOverlayShredsToPlain(input_vec, count, bind_data.return_src, returned);
		reader.Init(returned, count);
	} else {
		reader.Init(input_vec, count);
	}

	JsonoBodyWriter writer;
	writer.Init(result);
	// Shred-set marker: the layout hash on every value row (NULL only on a SQL-NULL row). Per-shred
	// completeness lives in each scalar shred's `complete` flag — copied through for a kept shred,
	// recomputed by WriteShred for a newly-extracted one.
	Vector *set_out = &JsonoShredSetVector(result);
	set_out->SetVectorType(VectorType::FLAT_VECTOR);
	auto layout_hash = int64_t(JsonoLayoutHashOf(result.GetType()));
	vector<Vector *> shred_out(fields.size());
	vector<Vector *> complete_out(fields.size(), nullptr); // scalar shreds only
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
		if (fields[f].kind == ShredKind::Scalar) {
			complete_out[f] = &JsonoShredCompleteVector(result, f);
			complete_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
		}
	}

	for (idx_t f = 0; f < fields.size(); f++) {
		if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
			VectorOperations::Copy(JsonoShredVector(input_vec, bind_data.keep_src[f]), *shred_out[f], count, 0, 0);
			// Carry the kept shred's per-row completeness through verbatim (a source divert stays a
			// divert).
			if (complete_out[f]) {
				auto &src_complete = JsonoShredCompleteVector(input_vec, bind_data.keep_src[f]);
				if (src_complete.GetType().id() == complete_out[f]->GetType().id()) {
					VectorOperations::Copy(src_complete, *complete_out[f], count, 0, 0);
				} else {
					// A degraded-width `complete` (a value->SQL->value round-trip re-parsed the 1/0 flag
					// wider; the layout parser accepts it by IsIntegral) cannot be Copy'd into the
					// canonical TINYINT lane — cast the flag down, preserving its 1/0/NULL value.
					VectorOperations::DefaultCast(src_complete, *complete_out[f], count);
				}
			}
		}
	}

	vector<JsonoShredManifestEntryBytes> manifest_entries(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(fields[f].path.text, ShredFieldType(fields[f]));
	}

	const std::vector<ShredManifestEntry> *old_manifest = nullptr;
	std::vector<ShredManifestEntry> src_manifest;
	auto manifest_has_path = [&](const string &path) {
		for (auto &entry : *old_manifest) {
			if (entry.path == nonstd::string_view(path.data(), path.size())) {
				return true;
			}
		}
		return false;
	};

	auto null_row = [&](idx_t row) {
		writer.SetRowNull(row);
		for (auto *shred : shred_out) {
			FlatVector::SetNull(*shred, row, true);
		}
		JsonoSetRowMarkerNull(result, row);
	};

	JsonoVectorData input;
	InitJsonoVectorData(input_vec, count, input);
	std::string manifest;
	vector<idx_t> stripped_fields;
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			null_row(row);
			continue;
		}
		if (!reads_merged) {
			old_manifest = &reader.RowManifest(view);
		} else {
			// The merged residual carries no manifest; the kept shreds' stripped state lives in
			// the original row's manifest. The partial overlay already verified it.
			JsonoBlobRow src_blob;
			src_manifest.clear();
			if (ReadJsonoRowStrict(input, row, src_blob)) {
				JsonoView src_view = MakeJsonoView(src_blob);
				if (src_view.ParseHeader()) {
					src_view.ReadShredManifest(src_manifest);
				}
			}
			old_manifest = &src_manifest;
		}

		FlatVector::Validity(result).SetValid(row);
		FlatVector::GetData<int64_t>(*set_out)[row] = layout_hash;
		lstate.strip_paths.clear();
		stripped_fields.clear();
		for (idx_t f = 0; f < fields.size(); f++) {
			if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
				// The kept shred's stripped-or-not state, and its per-row completeness, carry over from
				// the input row (the manifest, and the copied `complete` lane).
				if (manifest_has_path(fields[f].path.text)) {
					stripped_fields.push_back(f);
				}
				continue;
			}
			auto &field = fields[f];
			auto location = LocatePath(view, field.path.steps);
			bool complete = true;
			bool strippable = WriteShred(field, view, location, *shred_out[f], row, lstate.text, &complete);
			if (complete_out[f]) {
				FlatVector::GetData<int8_t>(*complete_out[f])[row] = complete ? 1 : 0;
			}
			if (strippable) {
				lstate.strip_paths.push_back(&field.path.steps);
				stripped_fields.push_back(f);
			}
		}

		if (lstate.strip_paths.empty()) {
			// No path was stripped from this row's residual, so its blobs are unchanged. Copy them
			// verbatim instead of re-emitting through the builder. The original residual already
			// carries the right manifest (the kept entries are exactly the input row's manifest);
			// a merged residual carries none, so the kept entries are appended to its skips.
			writer.data[BODY_SLOTS][row] = WriteBlobInto(writer.Slots(), blob.slots.GetData(), blob.slots.GetSize());
			writer.data[BODY_KEY_HEAP][row] =
			    WriteBlobInto(writer.KeyHeap(), blob.key_heap.GetData(), blob.key_heap.GetSize());
			writer.data[BODY_STRING_HEAP][row] =
			    WriteBlobInto(writer.StringHeap(), blob.string_heap.GetData(), blob.string_heap.GetSize());
			writer.data[BODY_LENGTHS][row] =
			    WriteBlobInto(writer.Lengths(), blob.lengths.GetData(), blob.lengths.GetSize());
			writer.data[BODY_NUMS][row] = WriteBlobInto(writer.Nums(), blob.nums.GetData(), blob.nums.GetSize());
			if (reads_merged && !stripped_fields.empty()) {
				manifest.clear();
				manifest.append(blob.skips.GetData(), blob.skips.GetSize());
				JsonoAppendShredManifest(manifest, manifest_entries, stripped_fields);
				writer.data[BODY_SKIPS][row] = WriteBlobInto(writer.Skips(), manifest.data(), manifest.size());
			} else {
				writer.data[BODY_SKIPS][row] =
				    WriteBlobInto(writer.Skips(), blob.skips.GetData(), blob.skips.GetSize());
			}
			continue;
		}

		lstate.builder.Reset();
		JsonoEmitObjectStrippingPaths(view, lstate.strip_paths, lstate.builder);
		const std::string *manifest_ptr = nullptr;
		if (!stripped_fields.empty()) {
			manifest.clear();
			JsonoAppendShredManifest(manifest, manifest_entries, stripped_fields);
			manifest_ptr = &manifest;
		}
		writer.WriteRow(row, lstate.builder, manifest_ptr);
	}
}

void JsonoShredExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ShredBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ShredLocalState>();
	bool input_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR || expr.children[0]->IsFoldable();
	if (bind_data.reshred_active) {
		ApplyReshredShredded(args.data[0], args.size(), bind_data, result, lstate);
	} else {
		ApplyShredFields(args.data[0], args.size(), bind_data.fields, result, lstate);
	}
	if (input_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// jsono(text, shredding := spec): parse the JSON text and shred it in one pass — the direct
// DOM writer's sizing pass captures the shred leaves and strips the lossless ones from the
// emit plan, so the full plain value is never materialized and no path is located twice. A
// present-but-lossy shred (a number's `->>` text at a VARCHAR lane) is captured by the DOM pass
// itself as a diverted lane value; nothing is filled after the fact.
void JsonoShredFromTextExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = expr.bind_info->Cast<ShredBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<ShredLocalState>();
	auto count = args.size();
	bool input_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR || expr.children[0]->IsFoldable();
	if (!bind_data.one_pass_text) {
		Vector plain(JsonoType(), count);
		JsonoParseTextVector(args.data[0], count, plain);
		ApplyShredFields(plain, count, bind_data.fields, result, lstate);
		if (input_constant) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	auto &fields = bind_data.fields;
	UnifiedVectorFormat input_fmt;
	args.data[0].ToUnifiedFormat(count, input_fmt);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_fmt);

	JsonoBodyWriter writer;
	writer.Init(result);
	// one_pass_text requires every field to be a scalar object-key shred, so every field has a
	// complete flag — gather both lanes per field.
	vector<Vector *> shred_out(fields.size());
	vector<Vector *> complete_out(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
		complete_out[f] = &JsonoShredCompleteVector(result, f);
		complete_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}
	// Shred-set marker: the layout hash on every value row (NULL only on a SQL-NULL row). Per-shred
	// completeness lives in each scalar shred's `complete` flag, written below.
	Vector *set_out = &JsonoShredSetVector(result);
	set_out->SetVectorType(VectorType::FLAT_VECTOR);
	auto layout_hash = int64_t(JsonoLayoutHashOf(result.GetType()));

	vector<JsonoShredManifestEntryBytes> manifest_entries(fields.size());
	jsono_dom::DomShredContext ctx;
	ctx.nodes = &bind_data.trie;
	ctx.manifest_entries = &manifest_entries;
	ctx.kinds.resize(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		manifest_entries[f] = JsonoShredManifestEntry(fields[f].path.text, ShredFieldType(fields[f]));
		ctx.kinds[f] = fields[f].primitive;
	}

	jsono_dom::YyjsonAllocator parser;
	jsono_dom::DomDirectState dom;

	for (idx_t row = 0; row < count; row++) {
		auto idx = input_fmt.sel->get_index(row);
		if (!input_fmt.validity.RowIsValid(idx)) {
			writer.SetRowNull(row);
			for (idx_t f = 0; f < fields.size(); f++) {
				FlatVector::SetNull(*shred_out[f], row, true);
			}
			JsonoSetRowMarkerNull(result, row);
			continue;
		}
		auto input = inputs[idx];
		if (input.GetSize() == 0) {
			throw InvalidInputException("jsono: input is empty");
		}
		yyjson_read_err err;
		auto doc = jsono_dom::ReadJsonoDoc(input, parser.alc, err);
		if (!doc) {
			throw InvalidInputException("jsono: invalid JSON - %s", err.msg);
		}
		try {
			jsono_dom::EmitDomRowDirect(yyjson_doc_get_root(doc), dom, writer, row, &ctx);
			FlatVector::Validity(result).SetValid(row);
			FlatVector::GetData<int64_t>(*set_out)[row] = layout_hash;
			for (idx_t f = 0; f < fields.size(); f++) {
				auto &cap = ctx.captures[f];
				// A captured value (or an absent path) is complete — a bare lane read equals `->>`. A
				// divert (diverted_scalar: a mismatched scalar or a container) is not.
				bool complete = !cap.diverted_scalar;
				switch (cap.state) {
				case jsono_dom::DomShredCapture::State::Missing:
					FlatVector::SetNull(*shred_out[f], row, true);
					break;
				case jsono_dom::DomShredCapture::State::String:
					WriteJsonoStringLane(*shred_out[f], row, cap.text);
					break;
				case jsono_dom::DomShredCapture::State::Int:
					WriteJsonoBigintLane(*shred_out[f], row, cap.int_value);
					break;
				case jsono_dom::DomShredCapture::State::Uint:
					WriteJsonoUbigintLane(*shred_out[f], row, cap.uint_value);
					break;
				case jsono_dom::DomShredCapture::State::Bool:
					WriteJsonoBooleanLane(*shred_out[f], row, cap.bool_value);
					break;
				}
				FlatVector::GetData<int8_t>(*complete_out[f])[row] = complete ? 1 : 0;
			}
		} catch (...) {
			yyjson_doc_free(doc);
			throw;
		}
		yyjson_doc_free(doc);
	}
	if (input_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

bool IsShredValueType(const LogicalType &type) {
	JsonoScalarPrimitive primitive;
	return TypeToJsonoScalarPrimitive(type, primitive);
}

ShredKind ClassifyShredKind(const LogicalType &type) {
	JsonoScalarPrimitive primitive;
	if (TypeToJsonoScalarPrimitive(type, primitive)) {
		return ShredKind::Scalar;
	}
	if (IsShredArrayType(type)) {
		return ShredKind::Array;
	}
	if (IsShredScalarArrayType(type)) {
		return ShredKind::ScalarArray;
	}
	throw InternalException("jsono: shred column type '%s' is neither a scalar nor an array shred", type.ToString());
}

bool IsShredArrayType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::LIST) {
		return false;
	}
	auto &element = ListType::GetChildType(type);
	if (element.id() != LogicalTypeId::STRUCT || StructType::IsUnnamed(element)) {
		return false;
	}
	auto &fields = StructType::GetChildTypes(element);
	if (fields.empty()) {
		return false;
	}
	for (auto &field : fields) {
		if (!IsShredValueType(field.second)) {
			return false;
		}
	}
	return true;
}

bool IsShredScalarArrayType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::LIST) {
		return false;
	}
	return IsShredValueType(ListType::GetChildType(type));
}

bool IsShredListType(const LogicalType &type) {
	return IsShredArrayType(type) || IsShredScalarArrayType(type);
}

bool ShredPathsOverlap(const vector<PathStep> &a, const vector<PathStep> &b) {
	idx_t common = std::min(a.size(), b.size());
	for (idx_t i = 0; i < common; i++) {
		if (a[i].kind != PathStepKind::Key || b[i].kind != PathStepKind::Key || a[i].key != b[i].key) {
			return false;
		}
	}
	return true;
}

void JsonoValidateShredField(const string &path, const LogicalType &type) {
	ShredField field;
	BindShredField(path, type, type.ToString(), field);
}

namespace {

uint8_t ShredManifestCompactTypeCode(const string &type) {
	for (auto &entry : jsono::SHRED_MANIFEST_COMPACT_TYPES) {
		if (entry.name == nonstd::string_view(type.data(), type.size())) {
			return entry.code;
		}
	}
	return jsono::SHRED_MANIFEST_TYPE_EXTENDED;
}

void AppendManifestLV(std::string &out, const string &text) {
	if (text.size() > std::numeric_limits<uint16_t>::max()) {
		throw InvalidInputException("jsono shred: path or type name exceeds manifest limits");
	}
	uint16_t len = uint16_t(text.size());
	out.append(reinterpret_cast<const char *>(&len), sizeof(len));
	out.append(text);
}

template <class EntryAt>
void JsonoAppendShredManifestInternal(std::string &manifest, idx_t entry_count, EntryAt entry_at) {
	if (entry_count > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono shred: too many manifest entries");
	}
	size_t full_size = sizeof(uint32_t);
	size_t compact_size = sizeof(uint32_t) * 2;
	for (idx_t i = 0; i < entry_count; i++) {
		auto &entry = entry_at(i);
		full_size += entry.full.size();
		compact_size += entry.compact.size();
	}
	if (compact_size < full_size) {
		uint32_t marker = jsono::SHRED_MANIFEST_COMPACT_TYPE_MARKER;
		uint32_t stored_count = uint32_t(entry_count);
		manifest.append(reinterpret_cast<const char *>(&marker), sizeof(marker));
		manifest.append(reinterpret_cast<const char *>(&stored_count), sizeof(stored_count));
		for (idx_t i = 0; i < entry_count; i++) {
			manifest.append(entry_at(i).compact);
		}
		return;
	}
	uint32_t stored_count = uint32_t(entry_count);
	manifest.append(reinterpret_cast<const char *>(&stored_count), sizeof(stored_count));
	for (idx_t i = 0; i < entry_count; i++) {
		manifest.append(entry_at(i).full);
	}
}

} // namespace

JsonoShredManifestEntryBytes JsonoShredManifestEntry(const string &path, const LogicalType &type) {
	auto type_name = type.ToString();
	JsonoShredManifestEntryBytes entry;
	AppendManifestLV(entry.full, path);
	AppendManifestLV(entry.full, type_name);
	AppendManifestLV(entry.compact, path);
	auto type_code = ShredManifestCompactTypeCode(type_name);
	entry.compact.push_back(char(type_code));
	if (type_code == jsono::SHRED_MANIFEST_TYPE_EXTENDED) {
		AppendManifestLV(entry.compact, type_name);
	}
	return entry;
}

void JsonoAppendShredManifest(std::string &manifest, const vector<JsonoShredManifestEntryBytes> &entries) {
	JsonoAppendShredManifestInternal(manifest, entries.size(),
	                                 [&](idx_t i) -> const JsonoShredManifestEntryBytes & { return entries[i]; });
}

void JsonoAppendShredManifest(std::string &manifest, const vector<JsonoShredManifestEntryBytes> &entries,
                              const vector<idx_t> &entry_indices) {
	JsonoAppendShredManifestInternal(
	    manifest, entry_indices.size(), [&](idx_t i) -> const JsonoShredManifestEntryBytes & {
		    auto index = entry_indices[i];
		    if (index >= entries.size()) {
			    throw InternalException("jsono shred: manifest entry index is out of bounds");
		    }
		    return entries[index];
	    });
}

void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &shreds,
                        Vector &result) {
	vector<ShredField> fields;
	fields.reserve(shreds.size());
	for (auto &shred : shreds) {
		ShredField field;
		field.path = ParseShredPathSpec(shred.first);
		if (!FillShredFieldFromType(shred.second, field)) {
			throw InternalException("jsono shred field '%s' has a non-shred type", shred.first);
		}
		fields.push_back(std::move(field));
	}
	ShredLocalState lstate;
	ApplyShredFields(input, count, fields, result, lstate);
}

// The jsono(jsono, shredding := spec) overload, exposed so the optimizer's cast normalization
// can reshred a value to a target shred set without a catalog lookup. Declared on a generic
// STRUCT so a shredded value reaches the bind natively: when every source shred survives into
// the target the bind plans the single-pass reshred; otherwise it redeclares the argument as
// plain JSONO and the binder's reconstruct cast feeds the plain shred path. The injected
// expression binds through the regular JsonoShredBind (a constant STRUCT spec), so it re-binds
// identically on plan deserialization.
ScalarFunction JsonoShredFromJsonoFunction() {
	ScalarFunction from_jsono("jsono", {LogicalTypeId::STRUCT, LogicalType::ANY}, LogicalType::ANY, JsonoShredExecute,
	                          JsonoShredBind, nullptr, nullptr, ShredLocalState::Init);
	from_jsono.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	from_jsono.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return from_jsono;
}

// Shredding is exposed as a named argument on the jsono() constructor:
// `shredding := {'<path>': '<type>', ...}` (a constant STRUCT, like Parquet's SHREDDING).
// The primary form parses text and shreds in one pass; the secondary form shreds an existing
// plain jsono value. Both reuse the same bind (spec parse + return type) and shred writer. The
// spec slot is ANY so the STRUCT literal binds; the bind validates it is a STRUCT.
void RegisterJsonoShred(ExtensionLoader &loader) {
	ScalarFunctionSet set("jsono");

	// jsono(text, shredding := spec) — the primary entry point: parse + shred.
	ScalarFunction from_text({LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::ANY, JsonoShredFromTextExecute,
	                         JsonoShredBind, nullptr, nullptr, ShredLocalState::Init);
	from_text.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	from_text.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	set.AddFunction(from_text);

	// jsono(jsono, shredding := spec) — shred a value that is already plain jsono.
	set.AddFunction(JsonoShredFromJsonoFunction());

	loader.RegisterFunction(set);
}

} // namespace duckdb
