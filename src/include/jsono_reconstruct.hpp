#pragma once

#include "jsono.hpp"
#include "jsono_path.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

using jsono::JsonoShredSignature;

// One lane's manifest signature as the plan carries it to __jsono_internal_checked_residual: what
// JsonoBuildShredSignatures derives from a reading type, spelled as a value. A STRUCT of fields and
// a nested list, not joined strings — a path and a JSON key may each contain any byte, so there is
// no separator that needs no escaping, and the channel carries structure natively.
inline LogicalType JsonoShredSignatureType() {
	child_list_t<LogicalType> subfield;
	subfield.emplace_back("key", LogicalType::VARCHAR);
	subfield.emplace_back("type", LogicalType::VARCHAR);
	child_list_t<LogicalType> signature;
	signature.emplace_back("path", LogicalType::VARCHAR);
	signature.emplace_back("type", LogicalType::VARCHAR);
	signature.emplace_back("subfields", LogicalType::LIST(LogicalType::STRUCT(std::move(subfield))));
	return LogicalType::STRUCT(std::move(signature));
}

// The signature list as a plan constant, and its inverse. The two live side by side so the shape the
// optimizer writes and the shape the executor reads cannot drift apart.
inline Value JsonoShredSignaturesToValue(const std::vector<JsonoShredSignature> &signatures) {
	auto signature_type = JsonoShredSignatureType();
	auto subfield_type = ListType::GetChildType(StructType::GetChildTypes(signature_type)[2].second);
	std::vector<Value> values;
	values.reserve(signatures.size());
	for (auto &signature : signatures) {
		std::vector<Value> subfields;
		subfields.reserve(signature.subfields.size());
		for (auto &subfield : signature.subfields) {
			child_list_t<Value> pair;
			pair.emplace_back("key", Value(subfield.first));
			pair.emplace_back("type", Value(subfield.second));
			subfields.push_back(Value::STRUCT(std::move(pair)));
		}
		child_list_t<Value> fields;
		fields.emplace_back("path", Value(signature.path));
		fields.emplace_back("type", Value(signature.type));
		fields.emplace_back("subfields", Value::LIST(subfield_type, std::move(subfields)));
		values.push_back(Value::STRUCT(std::move(fields)));
	}
	return Value::LIST(signature_type, std::move(values));
}

// Every string in the channel is required. __jsono_internal_checked_residual is in the catalog (the
// re-bind after plan deserialization needs it there), which also puts it within reach of a
// hand-written query, so a NULL here is a wrong argument and not a broken invariant: reading it
// through StringValue::Get would raise an INTERNAL error and abort the caller's transaction over it.
inline const std::string &JsonoRequiredSignatureString(const Value &value, const char *field) {
	if (value.IsNull()) {
		throw InvalidInputException("__jsono_internal_checked_residual: a shred signature's '%s' must not be NULL",
		                            field);
	}
	return StringValue::Get(value);
}

inline std::vector<JsonoShredSignature> JsonoShredSignaturesFromValue(const Value &value) {
	if (value.IsNull()) {
		throw InvalidInputException("__jsono_internal_checked_residual: the shred signature list must not be NULL");
	}
	std::vector<JsonoShredSignature> signatures;
	for (auto &element : ListValue::GetChildren(value)) {
		if (element.IsNull()) {
			throw InvalidInputException("__jsono_internal_checked_residual: a shred signature must not be NULL");
		}
		auto &fields = StructValue::GetChildren(element);
		JsonoShredSignature signature;
		signature.path = JsonoRequiredSignatureString(fields[0], "path");
		signature.type = JsonoRequiredSignatureString(fields[1], "type");
		if (fields[2].IsNull()) {
			throw InvalidInputException("__jsono_internal_checked_residual: a shred signature's 'subfields' must not "
			                            "be NULL; an object-array lane spells its element subfields out and every "
			                            "other lane spells an empty list");
		}
		for (auto &subfield : ListValue::GetChildren(fields[2])) {
			if (subfield.IsNull()) {
				throw InvalidInputException(
				    "__jsono_internal_checked_residual: an element subfield signature must not be NULL");
			}
			auto &pair = StructValue::GetChildren(subfield);
			signature.subfields.emplace_back(JsonoRequiredSignatureString(pair[0], "subfields.key"),
			                                 JsonoRequiredSignatureString(pair[1], "subfields.type"));
		}
		signatures.push_back(std::move(signature));
	}
	return signatures;
}

class Expression;
class ScalarFunction;

// One shred of a shredded type resolved for reading: the shred column index, its lane type, and the
// object-key path its value re-enters the document at. Shared by the reconstruct overlay (which
// resolves only scalar lanes here and keeps LIST lanes in its own array-shred struct) and the keyed
// group_merge direct fold (which stages LIST lanes in this struct too).
struct ReconShred {
	idx_t child;
	LogicalType type;
	vector<PathStep> steps; // object-key path; size 1 is a top-level key
	// Manifest name of the path, filled only by the keyed fold (which rewrites the residual's shred
	// manifest as lanes move); the reconstruct overlay leaves it empty.
	string manifest_path;
};

// Reconstruct a shredded JSONO `input` (six-BLOB residual + shred columns) into the
// lossless plain JSONO `result` by overlaying each row's shred values onto its residual.
// This is how a shredded value becomes usable where a plain JSONO is required (an implicit
// cast, an INSERT into a plain JSONO column) without dropping the shred data. An all-top-level
// shred set overlays as one flat patch; a set carrying any nested path folds top-level and nested
// together into ONE patch tree applied in a single overlay. Throws on a valid row whose residual
// blob is NULL: the writer never emits that, so the row is the trace of a struct cast NULL-filling
// the body.
void JsonoReconstructToPlain(Vector &input, idx_t count, Vector &result);

// Overlay only `shreds` (shred indices over the shred set) of the shredded `input` onto its
// residual, producing plain JSONO. The single-pass narrowing reshred folds the shreds the target
// type drops back into the residual with this, leaving the kept shreds untouched. Each row's shred
// manifest is still verified against ALL of the type's shreds, not just the overlaid ones — callers
// that fold a subset (the non-keyed group_merge array path) rely on that for their loud narrowing
// failure. `shreds` must be non-empty: an empty list overlays nothing and yields the bare residual,
// which is a narrowed document the manifest check cannot catch — callers branch on emptiness before
// reaching here rather than asking for a no-op overlay.
void JsonoOverlayShredsToPlain(Vector &input, idx_t count, const vector<idx_t> &shreds, Vector &result);

// Render a shredded JSONO carrying top-level LIST shreds directly to JSON text. Scalar shreds are
// first overlaid into the residual; list lanes then merge into their residual arrays by index,
// without materializing the reconstructed arrays as a plain JSONO blob. The optimizer binds this
// only where it already proved the shape, so a non-shredded input, an empty shred set, a list shred
// on a nested path, or two list shreds on one path are all InternalException.
void JsonoRenderShreddedListsToJson(Vector &input, idx_t count, Vector &result);

// Wrap a shredded JSONO argument expression in the internal __jsono_reconstruct scalar operator,
// whose executor is JsonoReconstructToPlain. Only jsono_transform's array-shred reconstruct path uses
// this wrapper: it would otherwise redeclare the argument as plain JSONO and let the binder insert an
// anonymous reconstruct cast, so the wrapper makes that 3-10x shredded->plain cost an explicit, named
// operator in EXPLAIN / EXPLAIN ANALYZE. The other reconstruct binds (collect, elements, diff fallback)
// still inject anonymous casts. Returns the wrapping expression (plain JSONO return type);
// `shredded_arg` must have a shredded JSONO type. Defined in jsono_optimizer.cpp, next to the other
// optimizer-injected internal operators that share its throwing plan-serialization callbacks.
unique_ptr<Expression> MakeJsonoReconstructExpression(unique_ptr<Expression> shredded_arg);

// The optimizer-injected manifest guards: __jsono_internal_checked_residual verifies a residual's
// shred manifest against the shredded type it was read out of (narrowed rows fail loud);
// __jsono_internal_strip_manifest drops the manifest tail for the soft residual reinterpret.
ScalarFunction JsonoCheckedResidualFunction();
ScalarFunction JsonoStripManifestFunction();

} // namespace duckdb
