#include "jsono.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parser.hpp"

#include <algorithm>

namespace duckdb {

void RegisterJsonoParse(ExtensionLoader &loader);
void RegisterJsonoStructConstructor(ExtensionLoader &loader);
void RegisterJsonoToJson(ExtensionLoader &loader);

namespace {

void JsonoNullExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

void JsonoIdentityExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	result.Reference(args.data[0]);
}

// jsono_storage_type() -> the DDL string of the physical STRUCT that a jsono value is.
// DuckLake rejects user-defined type aliases (the reason jsono carries none), so writers
// declare storage columns with this struct; exposing it here keeps the layout owned by the
// extension.
void JsonoStorageTypeExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, Value(JsonoRawStructType().ToString()));
}

// jsono_version() -> the binary format version this extension reads/writes. A format bump
// leaves the physical STRUCT type unchanged, so a schema epoch derived from the type alone
// cannot see it; folding this into a schema hash triggers re-materialization on a bump
// before a version-mismatched read fails loud.
void JsonoVersionExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, Value::INTEGER(int32_t(jsono::VERSION)));
}

// jsono_storage_type(lanes) -> the shredded storage type's DDL: the 4-BLOB residual plus the
// given lane columns, so a schema can declare a shredded jsono column from a readable lane spec
// (e.g. 'event_name VARCHAR, n BIGINT') without computing the field-name fingerprint by hand. The
// lane DDL is parsed into (name, type) and re-emitted through JsonoShreddedStructType, so the
// declared column is byte-identical to a value built from the same lanes; declare lanes the same
// way in the column and the shredding spec, or a generic struct cast on INSERT fails loud.
void JsonoStorageTypeWithLanesExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t lanes) {
		auto columns = Parser::ParseColumnList(lanes.GetString());
		child_list_t<LogicalType> lane_types;
		for (auto &col : columns.Logical()) {
			lane_types.emplace_back(col.Name(), col.Type());
		}
		// Canonical lane order (sorted by name) so the DDL matches a value built from the same lanes
		// regardless of the order they are written in here vs the shredding spec.
		std::sort(lane_types.begin(), lane_types.end(),
		          [](const std::pair<string, LogicalType> &a, const std::pair<string, LogicalType> &b) {
			          return a.first < b.first;
		          });
		return StringVector::AddString(result, JsonoShreddedStructType(lane_types).ToString());
	});
}

// Order-independent fingerprint of a lane set, folding in lane types so a type change is also a
// mismatch. Stable across processes and builds (a fixed FNV-1a, not std::hash, whose seed varies):
// jsono_storage_type and the constructor must agree byte-for-byte on the suffix for the same lanes.
string JsonoLaneFingerprint(const child_list_t<LogicalType> &lanes) {
	vector<string> signatures;
	signatures.reserve(lanes.size());
	for (auto &lane : lanes) {
		signatures.push_back(lane.first + '\x01' + lane.second.ToString());
	}
	std::sort(signatures.begin(), signatures.end());
	uint64_t hash = 1469598103934665603ULL;
	for (auto &signature : signatures) {
		for (unsigned char byte : signature) {
			hash = (hash ^ byte) * 1099511628211ULL;
		}
		hash = (hash ^ 0xFFu) * 1099511628211ULL; // separator so {"ab"},{"a","b"} differ
	}
	static const char *hex = "0123456789abcdef";
	string suffix = "#";
	for (int shift = 60; shift >= 0; shift -= 4) {
		suffix += hex[(hash >> shift) & 0xF];
	}
	return suffix;
}

} // namespace

bool IsJsonoType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	// A jsono value is recognised purely by structure: the four jsono_* BLOB children.
	// There is no type alias to short-circuit on — a stored value (DuckDB-native, Parquet
	// or DuckLake) is just this STRUCT. JsonoRawStructType is the single field contract.
	return StructType::GetChildTypes(type) == StructType::GetChildTypes(JsonoRawStructType());
}

// The four residual BLOB field bases, in order. A jsono-prefixed type names them base+suffix,
// where the suffix is "" for a plain value and "#<fp>" for a shredded one (carried by every field).
static const char *const JSONO_BLOB_BASES[4] = {"jsono_slots", "jsono_key_heap", "jsono_string_heap", "jsono_skips"};

bool HasJsonoBlobPrefix(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructType::GetChildTypes(type);
	if (children.size() < 4) {
		return false;
	}
	// The suffix is whatever follows "jsono_slots" in the first field; all four blobs must then be
	// base+suffix and BLOB. This recognises both plain (suffix "") and shredded (suffix "#<fp>").
	const string base0 = JSONO_BLOB_BASES[0];
	const string &name0 = children[0].first;
	if (name0.size() < base0.size() || name0.compare(0, base0.size(), base0) != 0) {
		return false;
	}
	string suffix = name0.substr(base0.size());
	for (idx_t i = 0; i < 4; i++) {
		if (children[i].second.id() != LogicalTypeId::BLOB) {
			return false;
		}
		if (children[i].first != string(JSONO_BLOB_BASES[i]) + suffix) {
			return false;
		}
	}
	return true;
}

bool IsShreddedJsonoType(const LogicalType &type) {
	return HasJsonoBlobPrefix(type) && StructType::GetChildTypes(type).size() > 4;
}

LogicalType JsonoShreddedStructType(const child_list_t<LogicalType> &lanes) {
	auto suffix = JsonoLaneFingerprint(lanes);
	child_list_t<LogicalType> children;
	for (auto base : JSONO_BLOB_BASES) {
		children.emplace_back(base + suffix, LogicalType::BLOB);
	}
	for (auto &lane : lanes) {
		children.emplace_back(lane.first + suffix, lane.second);
	}
	return LogicalType::STRUCT(std::move(children));
}

string JsonoLaneSuffix(const LogicalType &type) {
	// field[0] is "jsono_slots" + suffix; everything past the base is the shared suffix.
	const string base0 = JSONO_BLOB_BASES[0];
	return StructType::GetChildTypes(type)[0].first.substr(base0.size());
}

string JsonoStripLaneSuffix(const string &field_name, const string &suffix) {
	if (!suffix.empty() && field_name.size() >= suffix.size() &&
	    field_name.compare(field_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
		return field_name.substr(0, field_name.size() - suffix.size());
	}
	return field_name;
}

LogicalType JsonoRawStructType() {
	child_list_t<LogicalType> children;
	children.emplace_back("jsono_slots", LogicalType::BLOB);
	children.emplace_back("jsono_key_heap", LogicalType::BLOB);
	children.emplace_back("jsono_string_heap", LogicalType::BLOB);
	children.emplace_back("jsono_skips", LogicalType::BLOB);
	return LogicalType::STRUCT(std::move(children));
}

LogicalType JsonoType() {
	// The JSONO type alias was dropped: a user-defined alias cannot round-trip through
	// Parquet (stripped on read) or DuckLake (rejected as an unsupported user-defined type),
	// so a stored jsono value is the physical STRUCT and nothing more. Function and cast
	// dispatch is structural (IsJsonoType). Kept as the accessor the registrations read
	// against; identical to JsonoRawStructType().
	return JsonoRawStructType();
}

void RegisterJsonoType(ExtensionLoader &loader) {
	auto jsono_type = JsonoType();

	{
		ScalarFunctionSet set("jsono");
		set.AddFunction(ScalarFunction({LogicalType::SQLNULL}, jsono_type, JsonoNullExecute));
		set.AddFunction(ScalarFunction({jsono_type}, jsono_type, JsonoIdentityExecute));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("try_jsono");
		set.AddFunction(ScalarFunction({LogicalType::SQLNULL}, jsono_type, JsonoNullExecute));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_storage_type");
		set.AddFunction(ScalarFunction({}, LogicalType::VARCHAR, JsonoStorageTypeExecute));
		set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, JsonoStorageTypeWithLanesExecute));
		loader.RegisterFunction(set);
	}
	{
		ScalarFunctionSet set("jsono_version");
		set.AddFunction(ScalarFunction({}, LogicalType::INTEGER, JsonoVersionExecute));
		loader.RegisterFunction(set);
	}

	RegisterJsonoStructConstructor(loader);
	RegisterJsonoParse(loader);
	RegisterJsonoToJson(loader);
}

} // namespace duckdb
