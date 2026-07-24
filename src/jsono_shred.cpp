#include "jsono_shred.hpp"
#include "jsono.hpp"
#include "jsono_copy.hpp"
#include "jsono_dom.hpp"
#include "jsono_locate.hpp"
#include "jsono_number.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_scalar_write.hpp"
#include "jsono_trie.hpp"
#include "jsono_trie_walk.hpp"
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
	// One-pass jsono write (ApplyShredFields over a plain jsono input): the same idea against the
	// binary body — one sorted-merge trie walk of the document captures and nulls every scalar shred,
	// replacing the O(shreds) per-row LocatePath loop with an O(document) walk. Eligible whenever every
	// shred is a scalar (array shreds keep the per-path locate loop, like one_pass_text). The residual
	// still emits through the shared strip-emit pass. Built from the same scalar object-key paths.
	bool one_pass_jsono = false;
	JsonoTrie jsono_trie;

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
	vector<idx_t> flat_strip_positions;
	vector<uint8_t> flat_residual_keep;
	// Per-node rank cache for the one-pass jsono trie walk; sized to the bind-time trie on first use
	// and reused across chunks (it survives in the local state, the whole point of the cache).
	JsonoTrieRankCache jsono_rank_cache;

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

	// One-pass jsono trie (the binary counterpart of the text trie above): eligible whenever every
	// shred is a scalar object-key path. A shared trie node carries a duplicate path's two fields as
	// two scalar leaves, so — unlike the text writer — duplicates need no fallback: both leaves read
	// the same located cursor, matching what two independent LocatePath calls would have written.
	bind_data->one_pass_jsono = true;
	bind_data->jsono_trie.Reset();
	bind_data->jsono_trie.PrepareFieldMetadata(bind_data->fields.size());
	for (idx_t f = 0; f < bind_data->fields.size(); f++) {
		auto &field = bind_data->fields[f];
		if (field.kind != ShredKind::Scalar) {
			bind_data->one_pass_jsono = false;
			continue;
		}
		auto leaf = bind_data->jsono_trie.WalkFieldToLeaf(f, field.path.steps);
		bind_data->jsono_trie.nodes[leaf].scalar_leaves.push_back(f);
	}
	if (bind_data->one_pass_jsono) {
		bind_data->jsono_trie.SortEdges();
		for (auto &node : bind_data->jsono_trie.nodes) {
			if (node.scalar_leaves.size() == 1 && node.key_edges.empty() && node.index_edges.empty() &&
			    node.wildcard_child == DConstants::INVALID_INDEX) {
				node.simple_scalar_leaf = node.scalar_leaves[0];
			}
		}
	} else {
		bind_data->jsono_trie.Reset();
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

// Write a scalar shred's lane for a field and return whether the original value is losslessly
// captured by the shred type. Every shred path is an object-key chain (the bind rejects the rest),
// so this return value alone decides stripping: true means the path is stripped from the residual. A
// shred stores the value only when it round-trips through the shred type exactly (a real JSON string
// in a VARCHAR shred, a matching-kind scalar in a typed shred); a divert leaves the lane NULL and the
// value in the residual, and an absent path / explicit null leaves the lane NULL with nothing
// stripped — so a set lane always means the path is stripped, the value is stored exactly once.
// `spill` (optional out) is the per-shred, per-row bit for the `$jsono$spill` bitmap: true when the
// value stayed in the residual (divert, container) so a bare lane read would miss it, false when the
// NULL/typed lane holds the full `->>` answer (absent path, explicit null, a fitting scalar).
bool WriteShred(const ShredField &field, const JsonoView &view, const JsonoPathLocation &location, Vector &shred,
                idx_t row, std::string &scratch, bool *spill = nullptr) {
	auto set_spill = [&](bool value) {
		if (spill) {
			*spill = value;
		}
	};
	if (!location.found) {
		// Absent path: `->>` is NULL, a bare read of the NULL lane is correct.
		FlatVector::SetNull(shred, row, true);
		set_spill(false);
		return false;
	}
	auto value_tag = SlotTag(view.SlotAt(location.cursor.pos));
	if (value_tag == tag::OBJ_START || value_tag == tag::ARR_START) {
		// A container never fits a scalar lane: it stays in the residual and the read
		// reconstructs / falls back.
		FlatVector::SetNull(shred, row, true);
		set_spill(true);
		return false;
	}
	auto cursor = location.cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	if (scalar.kind == JsonoScalarKind::Null) {
		// Explicit JSON null: `->>` is NULL, a bare read of the NULL lane is correct.
		FlatVector::SetNull(shred, row, true);
		set_spill(false);
		return false;
	}
	if (JsonoScalarFitsPrimitive(scalar, field.primitive)) {
		WriteJsonoFittingScalarLane(shred, row, scalar, field.primitive, scratch);
		set_spill(false);
		return true;
	}
	// A present scalar that did not match its shred's kind gate: the value stays in the
	// residual, so a bare lane read would miss it — a spill.
	FlatVector::SetNull(shred, row, true);
	set_spill(true);
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

// One-pass jsono trie walk state: the shred writer's per-row outputs, reached by field index. The
// trie walk positions a cursor at each present scalar leaf (WriteScalarLeaf) and reaches every
// absent one (NullScalarLeaf / NullNodeLeaves); both route through WriteShred so the lane/spill/
// strip semantics are byte-identical to the per-field LocatePath loop, one document walk instead of
// one descent per shred.
struct ShredTrieWalkState {
	const vector<ShredField> &fields;
	vector<Vector *> &shred_children;
	const vector<idx_t> &spill_ranks;
	JsonoSpillStamp &stamp;
	std::string &scratch;
	std::vector<const std::vector<PathStep> *> &strip_paths;
	vector<idx_t> &stripped_fields;
};

struct ShredTrieWalkPolicy {
	using State = ShredTrieWalkState;
	static constexpr bool supports_wildcard = false;

	static JSONO_ALWAYS_INLINE void WriteLeaf(State &state, idx_t field_index, const JsonoView &view,
	                                          const JsonoPathLocation &location, idx_t row) {
		auto &field = state.fields[field_index];
		bool spill = false;
		bool strippable =
		    WriteShred(field, view, location, *state.shred_children[field_index], row, state.scratch, &spill);
		if (spill) {
			state.stamp.SetBit(state.spill_ranks[field_index]);
		}
		if (strippable) {
			state.strip_paths.push_back(&field.path.steps);
			state.stripped_fields.push_back(field_index);
		}
	}

	static JSONO_ALWAYS_INLINE void WriteScalarLeaf(State &state, idx_t field_index, const JsonoView &view,
	                                                const JsonoCursor &cursor, idx_t row) {
		WriteLeaf(state, field_index, view, JsonoPathLocation {cursor, true}, row);
	}

	static JSONO_ALWAYS_INLINE void NullScalarLeaf(State &state, idx_t field_index, const JsonoView &view, idx_t row) {
		// Absent path: WriteShred writes a NULL lane, complete=1, strips nothing.
		WriteLeaf(state, field_index, view, JsonoPathLocation {}, row);
	}

	static JSONO_ALWAYS_INLINE void NullNodeLeaves(State &state, const JsonoTrieNode &node, const JsonoView &view,
	                                               idx_t row) {
		for (auto field_index : node.scalar_leaves) {
			NullScalarLeaf(state, field_index, view, row);
		}
	}
};

// The canonical spill-bit rank of each shred field (see JsonoSpillRanksOfNames): rank == index for
// the sorted field lists the binds produce, recomputed here so an unsorted caller list stays
// correct.
vector<idx_t> ShredFieldSpillRanks(const vector<ShredField> &fields) {
	vector<string> names;
	names.reserve(fields.size());
	for (auto &field : fields) {
		names.push_back(field.path.text);
	}
	return JsonoSpillRanksOfNames(names);
}

void ApplyShredFields(Vector &input_vec, idx_t count, const vector<ShredField> &fields, Vector &result,
                      ShredLocalState &lstate, const JsonoTrie *trie = nullptr) {
	// The input is plain JSONO, which never carries a manifest of its own: the reader fails
	// loud on a narrowed row, whose re-emit below would otherwise replace the old manifest
	// with a fresh one and launder the loss into permanent undetectability.
	JsonoRowReader reader;
	reader.Init(input_vec, count);

	JsonoBodyWriter writer;
	writer.Init(result);
	// Shred-set marker + spill columns: the two-valued clean/dirty stamp on every value row
	// (JsonoSpillStamp; all NULL only on a SQL-NULL row).
	JsonoSpillStamp stamp;
	stamp.Init(result);
	auto spill_ranks = ShredFieldSpillRanks(fields);
	vector<Vector *> shred_children;
	shred_children.reserve(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_children.push_back(&JsonoShredVector(result, f));
		shred_children.back()->SetVectorType(VectorType::FLAT_VECTOR);
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

	// The one-pass trie walk (trie != nullptr) reaches every scalar shred in one sorted-merge pass of
	// the document, replacing the per-shred LocatePath loop. Its rank cache is sized to the bind-time
	// trie on first use and reused across chunks.
	if (trie && lstate.jsono_rank_cache.entries.empty()) {
		lstate.jsono_rank_cache.Init(trie->nodes);
	}

	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			null_shred_row(row);
			continue;
		}

		FlatVector::Validity(result).SetValid(row);
		stamp.ResetRow();
		lstate.strip_paths.clear();
		stripped_fields.clear();
		if (trie) {
			ShredTrieWalkState walk_state {fields,      shred_children,     spill_ranks,    stamp,
			                               lstate.text, lstate.strip_paths, stripped_fields};
			JsonoTrieWalkContext<ShredTrieWalkPolicy> ctx {trie->nodes, lstate.jsono_rank_cache, walk_state};
			JsonoTrieApplyNode<ShredTrieWalkPolicy>(ctx, 0, view, JsonoCursor(), row);
			// The walk visits leaves in document order; the manifest must stay canonical (sorted by
			// field, = sorted by path), so restore that order before serializing it.
			std::sort(stripped_fields.begin(), stripped_fields.end());
		} else {
			for (idx_t f = 0; f < fields.size(); f++) {
				auto &field = fields[f];
				auto location = LocatePath(view, field.path.steps);
				if (field.kind != ShredKind::Scalar) {
					// Array shreds never set spill bits: a per-element diversion has no
					// bare-struct_extract fast path to invalidate (the array is always reconstructed).
					bool stripped =
					    field.kind == ShredKind::ScalarArray
					        ? WriteScalarArrayShred(field, view, location, *shred_children[f], row, lstate.text)
					        : WriteArrayShred(field, view, location, *shred_children[f], row, lstate.text);
					if (stripped) {
						stripped_fields.push_back(f);
					}
					continue;
				}
				bool spill = false;
				bool strippable = WriteShred(field, view, location, *shred_children[f], row, lstate.text, &spill);
				if (spill) {
					stamp.SetBit(spill_ranks[f]);
				}
				if (strippable) {
					lstate.strip_paths.push_back(&field.path.steps);
					stripped_fields.push_back(f);
				}
			}
		}
		stamp.StampRow(row);

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

// A widening reshred already located every removed top-level scalar. Reuse those slot positions and
// bulk-copy the surviving stream runs; nested residuals keep the recursive emitter below.
bool TryWriteFlatObjectStrippingPaths(const JsonoView &view, const std::vector<const std::vector<PathStep> *> &paths,
                                      const vector<idx_t> &positions, vector<uint8_t> &keep,
                                      const std::string *manifest, JsonoBodyWriter &writer, idx_t row) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return false;
	}
	for (auto path : paths) {
		if (path->size() != 1) {
			return false;
		}
	}

	auto layout = ReadObjectLayout(view, 0);
	keep.assign(layout.key_count, 1);
	for (auto position : positions) {
		if (position < layout.value_start || position >= layout.value_start + layout.key_count) {
			return false;
		}
		keep[position - layout.value_start] = 0;
	}
	auto key_at = [&](idx_t i) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		return view.KeyAt(SlotPayload(key_slot));
	};
	idx_t surviving = 0;
	size_t key_bytes = 0;
	size_t string_bytes = 0;
	size_t length_count = 0;
	size_t num_count = 0;
	JsonoCursor cursor;
	cursor.pos = layout.value_start;
	for (idx_t i = 0; i < layout.key_count; i++) {
		auto slot = view.SlotAt(cursor.pos);
		auto slot_tag = SlotTag(slot);
		if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
			return false;
		}
		bool survives = keep[i] != 0;
		if (survives) {
			surviving++;
			key_bytes += key_at(i).size();
		}
		switch (ClassifyRawScalarSlot(slot)) {
		case RawScalarValueKind::LengthHeap: {
			auto len = view.LengthAt(cursor.length_cursor);
			if (survives) {
				string_bytes += len;
				length_count++;
			}
			cursor.string_cursor += len;
			cursor.length_cursor++;
			break;
		}
		case RawScalarValueKind::Number:
			if (survives) {
				num_count++;
			}
			cursor.num_cursor++;
			break;
		case RawScalarValueKind::Literal:
			break;
		}
		cursor.pos++;
	}
	if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	if (surviving > CONTAINER_CHILD_COUNT_MASK || key_bytes > std::numeric_limits<uint32_t>::max() ||
	    string_bytes > std::numeric_limits<uint32_t>::max() || length_count > std::numeric_limits<uint32_t>::max() ||
	    num_count > std::numeric_limits<uint32_t>::max()) {
		throw InvalidInputException("jsono: flat residual exceeds storage limits");
	}

	auto slot_count = size_t(2 + surviving * 2);
	auto slots = StringVector::EmptyString(writer.Slots(), JSONO_HEADER_SIZE + slot_count * sizeof(uint64_t));
	auto slots_data = slots.GetDataWriteable();
	WriteJsonoHeaderInto(reinterpret_cast<uint8_t *>(slots_data), flags::SORTED_KEYS);
	auto write_slot = [&](idx_t index, uint64_t slot) {
		std::memcpy(slots_data + JSONO_HEADER_SIZE + index * sizeof(uint64_t), &slot, sizeof(slot));
	};
	write_slot(0, MakeSlot(tag::OBJ_START, MakeContainerPayload(0, surviving)));

	auto key_heap = StringVector::EmptyString(writer.KeyHeap(), key_bytes);
	auto key_data = key_heap.GetDataWriteable();
	size_t key_offset = 0;
	idx_t output_field = 0;
	uint64_t shape_hash = HASH_SEED;
	size_t key_run_source = 0;
	size_t key_run_output = 0;
	size_t key_run_size = 0;
	auto flush_key_run = [&]() {
		if (key_run_size > 0) {
			auto source = view.KeyHeapSlice(key_run_source, key_run_size);
			std::memcpy(key_data + key_run_output, source.data(), key_run_size);
			key_run_size = 0;
		}
	};
	for (idx_t i = 0; i < layout.key_count; i++) {
		if (!keep[i]) {
			continue;
		}
		auto key_slot = view.SlotAt(layout.key_start + i);
		auto key = view.KeyAt(SlotPayload(key_slot));
		auto source_offset = KeyOffset(SlotPayload(key_slot));
		if (!key.empty() && key_run_size == 0) {
			key_run_source = source_offset;
			key_run_output = key_offset;
		} else if (!key.empty() && source_offset != key_run_source + key_run_size) {
			flush_key_run();
			key_run_source = source_offset;
			key_run_output = key_offset;
		}
		key_run_size += key.size();
		write_slot(1 + output_field, MakeSlot(tag::KEY, MakeKeyPayload(key_offset, key.size())));
		shape_hash = HashKey(shape_hash, key);
		key_offset += key.size();
		output_field++;
	}
	flush_key_run();

	auto string_heap = StringVector::EmptyString(writer.StringHeap(), string_bytes);
	auto string_data = string_heap.GetDataWriteable();
	auto lengths = StringVector::EmptyString(writer.Lengths(), length_count * sizeof(uint32_t));
	auto lengths_data = lengths.GetDataWriteable();
	auto nums = StringVector::EmptyString(writer.Nums(), num_count * sizeof(uint64_t));
	auto nums_data = nums.GetDataWriteable();

	bool has_checkpoints = surviving > OBJECT_CHECKPOINT_STRIDE;
	auto checkpoint_count = has_checkpoints ? (surviving - 1) / OBJECT_CHECKPOINT_STRIDE : 0;
	auto manifest_size = manifest ? manifest->size() : 0;
	auto skips_size = sizeof(ContainerMetadataHeader) +
	                  (has_checkpoints ? sizeof(uint32_t) + sizeof(ContainerSpan) + sizeof(ObjectCheckpointIndex) +
	                                         checkpoint_count * sizeof(ObjectCursorCheckpoint)
	                                   : 0) +
	                  manifest_size;
	auto skips = StringVector::EmptyString(writer.Skips(), skips_size);
	auto skips_data = skips.GetDataWriteable();
	ContainerMetadataHeader metadata {has_checkpoints ? 1U : 0U, has_checkpoints ? 1U : 0U, uint32_t(checkpoint_count)};
	std::memcpy(skips_data, &metadata, sizeof(metadata));
	size_t skips_offset = sizeof(metadata);
	char *checkpoint_data = nullptr;
	if (has_checkpoints) {
		uint32_t root_id = 0;
		std::memcpy(skips_data + skips_offset, &root_id, sizeof(root_id));
		skips_offset += sizeof(root_id);
		ContainerSpan span {uint32_t(slot_count), uint32_t(string_bytes), shape_hash, uint32_t(length_count),
		                    uint32_t(num_count)};
		std::memcpy(skips_data + skips_offset, &span, sizeof(span));
		skips_offset += sizeof(span);
		ObjectCheckpointIndex index {0, 0, OBJECT_CHECKPOINT_STRIDE, 0};
		std::memcpy(skips_data + skips_offset, &index, sizeof(index));
		skips_offset += sizeof(index);
		checkpoint_data = skips_data + skips_offset;
		skips_offset += checkpoint_count * sizeof(ObjectCursorCheckpoint);
	}
	if (manifest_size > 0) {
		std::memcpy(skips_data + skips_offset, manifest->data(), manifest_size);
	}

	cursor = JsonoCursor();
	cursor.pos = layout.value_start;
	output_field = 0;
	size_t output_string = 0;
	size_t output_length = 0;
	size_t output_num = 0;
	size_t string_run_source = 0;
	size_t string_run_output = 0;
	size_t string_run_size = 0;
	size_t length_run_source = 0;
	size_t length_run_output = 0;
	size_t length_run_count = 0;
	size_t num_run_source = 0;
	size_t num_run_output = 0;
	size_t num_run_count = 0;
	auto flush_string_run = [&]() {
		if (string_run_size > 0) {
			auto source = view.StringAt(string_run_source, string_run_size);
			std::memcpy(string_data + string_run_output, source.data(), string_run_size);
			string_run_size = 0;
		}
	};
	auto flush_length_run = [&]() {
		if (length_run_count > 0) {
			std::memcpy(lengths_data + length_run_output * sizeof(uint32_t),
			            view.LengthsBytes(length_run_source, length_run_count), length_run_count * sizeof(uint32_t));
			length_run_count = 0;
		}
	};
	auto flush_num_run = [&]() {
		if (num_run_count > 0) {
			std::memcpy(nums_data + num_run_output * sizeof(uint64_t), view.NumsBytes(num_run_source, num_run_count),
			            num_run_count * sizeof(uint64_t));
			num_run_count = 0;
		}
	};
	for (idx_t i = 0; i < layout.key_count; i++) {
		auto slot = view.SlotAt(cursor.pos);
		bool survives = keep[i] != 0;
		if (survives && output_field > 0 && output_field % OBJECT_CHECKPOINT_STRIDE == 0) {
			ObjectCursorCheckpoint checkpoint {uint32_t(output_field), uint32_t(output_string), uint32_t(output_length),
			                                   uint32_t(output_num)};
			auto checkpoint_index = output_field / OBJECT_CHECKPOINT_STRIDE - 1;
			std::memcpy(checkpoint_data + checkpoint_index * sizeof(checkpoint), &checkpoint, sizeof(checkpoint));
		}
		if (survives) {
			write_slot(1 + surviving + output_field, slot);
		}
		switch (ClassifyRawScalarSlot(slot)) {
		case RawScalarValueKind::LengthHeap: {
			auto len = view.LengthAt(cursor.length_cursor);
			if (survives) {
				if (len > 0 && string_run_size == 0) {
					string_run_source = cursor.string_cursor;
					string_run_output = output_string;
				} else if (len > 0 && cursor.string_cursor != string_run_source + string_run_size) {
					flush_string_run();
					string_run_source = cursor.string_cursor;
					string_run_output = output_string;
				}
				string_run_size += len;
				if (length_run_count == 0) {
					length_run_source = cursor.length_cursor;
					length_run_output = output_length;
				} else if (cursor.length_cursor != length_run_source + length_run_count) {
					flush_length_run();
					length_run_source = cursor.length_cursor;
					length_run_output = output_length;
				}
				length_run_count++;
				output_string += len;
				output_length++;
			}
			cursor.string_cursor += len;
			cursor.length_cursor++;
			break;
		}
		case RawScalarValueKind::Number:
			if (survives) {
				if (num_run_count == 0) {
					num_run_source = cursor.num_cursor;
					num_run_output = output_num;
				} else if (cursor.num_cursor != num_run_source + num_run_count) {
					flush_num_run();
					num_run_source = cursor.num_cursor;
					num_run_output = output_num;
				}
				num_run_count++;
				output_num++;
			}
			cursor.num_cursor++;
			break;
		case RawScalarValueKind::Literal:
			break;
		}
		if (survives) {
			output_field++;
		}
		cursor.pos++;
	}
	flush_string_run();
	flush_length_run();
	flush_num_run();
	write_slot(slot_count - 1, MakeSlot(tag::OBJ_END, 0));

	slots.Finalize();
	key_heap.Finalize();
	string_heap.Finalize();
	skips.Finalize();
	lengths.Finalize();
	nums.Finalize();
	writer.data[BODY_SLOTS][row] = slots;
	writer.data[BODY_KEY_HEAP][row] = key_heap;
	writer.data[BODY_STRING_HEAP][row] = string_heap;
	writer.data[BODY_SKIPS][row] = skips;
	writer.data[BODY_LENGTHS][row] = lengths;
	writer.data[BODY_NUMS][row] = nums;
	return true;
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
	// Shred-set marker + spill columns: the two-valued clean/dirty stamp on every value row. A kept
	// shred's bit is translated from the input row's mask (bit numbering follows each set's own
	// sorted-name ranks), a newly-extracted shred's bit is recomputed by WriteShred.
	JsonoSpillStamp stamp;
	stamp.Init(result);
	auto out_ranks = ShredFieldSpillRanks(fields);
	vector<Vector *> shred_out(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}

	for (idx_t f = 0; f < fields.size(); f++) {
		if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
			VectorOperations::Copy(JsonoShredVector(input_vec, bind_data.keep_src[f]), *shred_out[f], count, 0, 0);
		}
	}

	// The input row's spill mask is readable only under per-row set identity (the row's marker equals
	// a variant of the input TYPE's hash): a raw by-name cast copies masks across shred sets in
	// foreign numbering. A non-identity row recomputes each kept scalar's bit by probing the residual
	// instead (a NULL lane whose path holds a non-null residual value is a spill — the same invariant
	// validate enforces). Degraded integer widths (a value->SQL->value round-trip re-parsed the
	// BIGINT fields wider; the layout parser accepts them by IsIntegral) take the same recompute
	// path, as do ranks beyond the input type's spill columns.
	JsonoLayoutType src_layout;
	TryParseJsonoLayoutType(input_vec.GetType(), src_layout);
	vector<string> src_names;
	src_names.reserve(src_layout.shreds.size());
	for (auto &shred : src_layout.shreds) {
		src_names.push_back(shred.first);
	}
	auto in_ranks = JsonoSpillRanksOfNames(src_names);
	auto in_type_hash = int64_t(JsonoLayoutHashOf(input_vec.GetType()));
	Vector &in_set_vec = JsonoShredSetVector(input_vec);
	bool in_mask_readable = in_set_vec.GetType().id() == LogicalTypeId::BIGINT;
	vector<Vector *> in_spill_vecs;
	vector<int64_t *> in_spill_data;
	for (idx_t column = 0; column < src_layout.spill_columns; column++) {
		auto &spill_vec = JsonoShredSpillVector(input_vec, column);
		in_mask_readable = in_mask_readable && spill_vec.GetType().id() == LogicalTypeId::BIGINT;
		in_spill_vecs.push_back(&spill_vec);
	}
	int64_t *in_set_data = nullptr;
	if (in_mask_readable) {
		in_set_data = FlatVector::GetData<int64_t>(in_set_vec);
		for (auto *spill_vec : in_spill_vecs) {
			in_spill_data.push_back(FlatVector::GetData<int64_t>(*spill_vec));
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
		bool row_identity = in_mask_readable && !FlatVector::IsNull(in_set_vec, row) &&
		                    (in_set_data[row] == in_type_hash ||
		                     in_set_data[row] == int64_t(uint64_t(in_type_hash) ^ JSONO_DIRTY_HASH_FLIP));
		stamp.ResetRow();
		lstate.strip_paths.clear();
		lstate.flat_strip_positions.clear();
		stripped_fields.clear();
		for (idx_t f = 0; f < fields.size(); f++) {
			if (bind_data.keep_src[f] != DConstants::INVALID_INDEX) {
				// The kept shred's stripped-or-not state carries over from the input row (the manifest);
				// its spill bit is translated between the two sets' rank numberings — or, on a
				// non-identity row (foreign mask) or a rank beyond the input's spill columns, recomputed
				// from the residual. The kept path cannot be a returned one (a retyped path is
				// re-extracted, not kept), so the merged residual answers the probe exactly like the
				// original.
				if (fields[f].kind == ShredKind::Scalar) {
					auto in_rank = in_ranks[bind_data.keep_src[f]];
					auto in_word = in_rank / JSONO_SPILL_BITS;
					bool bit;
					if (row_identity && in_word < in_spill_data.size()) {
						bit = !FlatVector::IsNull(*in_spill_vecs[in_word], row) &&
						      (uint64_t(in_spill_data[in_word][row]) >> (in_rank % JSONO_SPILL_BITS)) & 1;
					} else {
						auto location = LocatePath(view, fields[f].path.steps);
						bit = FlatVector::IsNull(*shred_out[f], row) && location.found &&
						      SlotTag(view.SlotAt(location.cursor.pos)) != tag::VAL_NULL;
					}
					if (bit) {
						stamp.SetBit(out_ranks[f]);
					}
				}
				if (manifest_has_path(fields[f].path.text)) {
					stripped_fields.push_back(f);
				}
				continue;
			}
			auto &field = fields[f];
			auto location = LocatePath(view, field.path.steps);
			bool spill = false;
			bool strippable = WriteShred(field, view, location, *shred_out[f], row, lstate.text, &spill);
			if (spill) {
				stamp.SetBit(out_ranks[f]);
			}
			if (strippable) {
				lstate.strip_paths.push_back(&field.path.steps);
				lstate.flat_strip_positions.push_back(location.cursor.pos);
				stripped_fields.push_back(f);
			}
		}
		stamp.StampRow(row);

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

		const std::string *manifest_ptr = nullptr;
		if (!stripped_fields.empty()) {
			manifest.clear();
			JsonoAppendShredManifest(manifest, manifest_entries, stripped_fields);
			manifest_ptr = &manifest;
		}
		if (TryWriteFlatObjectStrippingPaths(view, lstate.strip_paths, lstate.flat_strip_positions,
		                                     lstate.flat_residual_keep, manifest_ptr, writer, row)) {
			continue;
		}
		lstate.builder.Reset();
		JsonoEmitObjectStrippingPaths(view, lstate.strip_paths, lstate.builder);
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
		const JsonoTrie *trie = bind_data.one_pass_jsono ? &bind_data.jsono_trie : nullptr;
		ApplyShredFields(args.data[0], args.size(), bind_data.fields, result, lstate, trie);
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
	// one_pass_text requires every field to be a scalar object-key shred.
	vector<Vector *> shred_out(fields.size());
	for (idx_t f = 0; f < fields.size(); f++) {
		shred_out[f] = &JsonoShredVector(result, f);
		shred_out[f]->SetVectorType(VectorType::FLAT_VECTOR);
	}
	// Shred-set marker + spill columns: the two-valued clean/dirty stamp on every value row
	// (JsonoSpillStamp; all NULL only on a SQL-NULL row).
	JsonoSpillStamp stamp;
	stamp.Init(result);
	auto spill_ranks = ShredFieldSpillRanks(fields);

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
			stamp.ResetRow();
			for (idx_t f = 0; f < fields.size(); f++) {
				auto &cap = ctx.captures[f];
				// A captured value (or an absent path) needs no spill bit — a bare lane read equals
				// `->>`. A divert (diverted_scalar: a mismatched scalar or a container) sets the bit.
				if (cap.diverted_scalar) {
					stamp.SetBit(spill_ranks[f]);
				}
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
			}
			stamp.StampRow(row);
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
