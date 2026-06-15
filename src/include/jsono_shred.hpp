#pragma once

#include "jsono_path.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <string>
#include <utility>
#include <vector>

namespace duckdb {

class ExtensionLoader;
class ScalarFunction;

namespace jsono {
struct JsonoScalar;
}

// The scalar types a shred column can hold. The single closed set, shared by the shred writer and
// every native shred-lane reader (storage_size, entries, reconstruct). Readers switch over it with
// no `default`, so adding a scalar shred type is a compile error in each reader, not a runtime throw.
enum class ShredPrimitive : uint8_t { Varchar, Bigint, Ubigint, Double, Boolean };

// The category of a shred column: one scalar value, or a LIST<STRUCT> array shred. Readers switch
// over it with no `default`, so adding a future shred category (e.g. a map shred) is a compile error
// in each native reader rather than a silently-swallowed runtime `default`.
enum class ShredKind : uint8_t { Scalar, Array };

// Classify a scalar shred type into its ShredPrimitive; false if `type` is not a scalar a shred can
// hold (a JSON-aliased VARCHAR carrying a document, or any non-scalar). The single owner of the
// scalar shred set — every writer and reader classifies through here, never re-listing the type ids.
bool TypeToShredPrimitive(const LogicalType &type, ShredPrimitive &out);

// Classify a shred column type into its category. Throws InternalException if `type` is neither a
// scalar nor an array shred — the single fail-loud point, so readers carry no swallowing `default`.
// Bind validates shred types up front, so a miss here is a broken invariant, not user input.
ShredKind ClassifyShredKind(const LogicalType &type);

// True if a shred of this kind keeps a read-copy of a value that ALSO stays in the residual: only
// VARCHAR, which renders the `->>` text of any scalar for a fast typed read but strips just a real
// JSON string. A reader that enumerates logical leaves (jsono_entries) must treat such a cell as the
// residual's and not emit it twice. The single owner of the read-copy rule; see WriteShred.
bool ShredPrimitiveStoresReadCopy(ShredPrimitive kind);

void RegisterJsonoShred(ExtensionLoader &loader);

// The jsono(jsono, shredding := spec) overload, exposed so the optimizer's set-operation
// normalization can reshred a branch to the merged shred set without a catalog lookup.
ScalarFunction JsonoShredFromJsonoFunction();

// True if `type` is a scalar a shred value can hold losslessly (VARCHAR/BIGINT/UBIGINT/
// DOUBLE/BOOLEAN). The struct constructor lifts only such top-level fields into shreds;
// every other field (other scalars, nested objects/arrays) stays in the residual tape.
bool IsShredValueType(const LogicalType &type);

// True if `type` is an array shred column type: LIST<STRUCT<...>> whose every struct child is
// itself an IsShredValueType scalar. An array shred lifts the chosen leaf subfields of each
// element of a regular array (`$.products`) into a parallel typed LIST<STRUCT>, leaving the
// element tail in the residual skeleton — see docs/jsono_format.md "Array shreds".
bool IsShredArrayType(const LogicalType &type);

// The lossless strip gate for one located scalar against a shred column type: true iff the
// value round-trips through the shred type byte-for-byte (a JSON string in a VARCHAR shred, an
// integer in a BIGINT shred), so it may be removed from the residual. The single owner of the
// per-type kind gate, shared by the scalar shred writer (WriteShred) and the array residual
// skeleton emit so the strip decision cannot drift between write and skeleton.
bool JsonoScalarFitsShredType(const jsono::JsonoScalar &scalar, const LogicalType &type);

// One array shred for the residual-skeleton emit (write) and the reconstruct overlay (read): the
// object-key chain to the array, and the element subfields (name, scalar shred type) lifted into
// the LIST<STRUCT> column, in element-struct order.
struct JsonoArrayShredSpec {
	vector<PathStep> path;
	vector<std::pair<string, LogicalType>> subfields;
};

// Serialized shred-manifest entry bytes for one shred (u16 path_len, path, u16 type_len,
// type name); a row's manifest is a u32 entry count followed by the entries of the paths
// stripped from that row's residual.
std::string JsonoShredManifestEntry(const string &path, const LogicalType &type);

// Shred a plain JSONO `input` vector into the shredded `result` STRUCT (the four-BLOB
// residual prefix followed by one shred column per `shreds` entry, in order). Each shred is
// a top-level key named by `shreds[i].first` with type `shreds[i].second`; its value is
// lifted from the document and (when losslessly captured) stripped from the residual.
// Reuses the jsono_shred executor so the constructor and the shred function share strip
// and shred-write semantics.
void JsonoShredFromSpec(Vector &input, idx_t count, const vector<std::pair<string, LogicalType>> &shreds,
                        Vector &result);

} // namespace duckdb
