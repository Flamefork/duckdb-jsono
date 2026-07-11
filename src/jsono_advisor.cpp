#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_memory.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"
#include "jsono_row_read.hpp"
#include "jsono_scalar_write.hpp"
#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include "string_view.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace duckdb {

namespace {

using namespace jsono;

// The five lane primitives, in the preference order the advisor picks a scalar lane type in: a
// value fitting more than one candidate (a non-negative int64 fits both BIGINT and UBIGINT) is
// lifted to BIGINT; UBIGINT wins only when a value exceeds the int64 range so BIGINT no longer
// fits every present value. DOUBLE/BOOLEAN/VARCHAR are mutually exclusive with the integer lanes
// and each other, so at min_fit = 1.0 at most one of them can qualify; under a lowered min_fit a
// mixed path can clear the threshold for several, and this fixed order — not the larger fit
// count — picks the lane.
constexpr std::array<JsonoScalarPrimitive, 5> kPrimitivePreference = {
    JsonoScalarPrimitive::Bigint, JsonoScalarPrimitive::Ubigint, JsonoScalarPrimitive::Double,
    JsonoScalarPrimitive::Boolean, JsonoScalarPrimitive::Varchar};

// Distinct candidate paths one aggregate state may track before failing loud. A shredded column
// with thousands of distinct keys is a modelling mistake, not a shredding target — truncating the
// map silently would emit a partial spec that looks authoritative, so throw instead.
constexpr idx_t kMaxCandidatePaths = 8192;

// Per-lane-primitive counters keyed by the primitive's enum ordinal. Sized off kPrimitivePreference
// so a new primitive can never index past the array (`fit[uint8_t(primitive)]` out-of-bounds).
using FitCounts = std::array<idx_t, kPrimitivePreference.size()>;

// One object-array element subfield's fit tally: how many object elements carry it, and how many
// of those carry a scalar value fitting each lane primitive.
struct SubfieldStat {
	idx_t present = 0;
	FitCounts fit {};
};

// Everything the advisor needs to decide one candidate path's shred role and lane type at finalize.
// A path is observed as at most one of scalar / scalar-array / object-array in clean data; the
// counters keep the three interpretations separate so a mixed path drops under min_fit instead of
// silently picking one. Occurrences that carry the path but fit no lane role (an object value under
// a scalar key, or a nested `$.a.b` whose value is itself an object or array) land in
// nonrole_present: they inflate the min_fit denominator without feeding any role's fit numerator, so
// a path that is a scalar in some rows and an object in others drops at min_fit = 1.0.
struct SuggestCandidate {
	// Scalar leaf role (a top-level scalar key or a nested `$.a.b`(`.c`) scalar up to the auto-shred
	// depth cap).
	idx_t scalar_present = 0;
	FitCounts scalar_fit {};

	// Rows carrying the path with a value that maps to no liftable role (see the struct comment).
	idx_t nonrole_present = 0;

	// Array roles: present = the number of rows where the path is an array.
	idx_t array_rows = 0;
	bool saw_scalar_element = false;
	bool saw_object_element = false;
	// Scalar-array: a row fits primitive p when every element is a scalar fitting p (empty rows fit
	// vacuously, but saw_scalar_element gates whether the role has any evidence).
	FitCounts sarray_fit {};
	// Object-array: rows whose every element is an object, and the union of element subfields.
	idx_t oarray_object_rows = 0;
	idx_t object_elements = 0;
	std::map<string, SubfieldStat> subfields;
};

struct SuggestState {
	std::map<string, SuggestCandidate> *paths;
	idx_t non_null_rows;
	JsonoMemoryReservation mem;
};

// Approximate std::map (red-black tree) node bookkeeping per entry; the accounting only needs a faithful,
// monotone proxy for the map footprint, not byte-exactness.
constexpr idx_t kMapNodeOverhead = 48;

// Owned-heap footprint of the candidate-path map (each path entry plus its object-array subfield map).
// Bounded by kMaxCandidatePaths, but walked per Update/Combine call (via AccountDistinctStates) rather
// than per row.
idx_t SuggestFootprint(const SuggestState &state) {
	if (!state.paths) {
		return 0;
	}
	idx_t bytes = 0;
	for (auto &entry : *state.paths) {
		bytes += kMapNodeOverhead + entry.first.capacity() + sizeof(SuggestCandidate);
		for (auto &subfield : entry.second.subfields) {
			bytes += kMapNodeOverhead + subfield.first.capacity() + sizeof(SubfieldStat);
		}
	}
	return bytes;
}

struct SuggestBindData : public FunctionData {
	double min_presence;
	double min_fit;
	// Carried into Update/Combine so the candidate map's growth is accounted. Unlike the merge bind_data
	// this type has a serialize callback (named args do not survive re-bind), so it is re-fetched in the
	// deserialize path rather than by a re-run bind.
	BufferManager &buffer_manager;

	SuggestBindData(double min_presence_p, double min_fit_p, BufferManager &buffer_manager)
	    : min_presence(min_presence_p), min_fit(min_fit_p), buffer_manager(buffer_manager) {
	}
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SuggestBindData>(min_presence, min_fit, buffer_manager);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<SuggestBindData>();
		return min_presence == other.min_presence && min_fit == other.min_fit;
	}
};

// Plan serialization round-trips the two thresholds directly. Without these callbacks a plan
// deserialize would RE-BIND from the serialized children, whose named-argument aliases
// (min_presence :=, min_fit :=) do not survive expression serialization — the re-bind then fails
// loud on "unknown argument ''" under debug plan verification.
void SuggestSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                      const AggregateFunction &function) {
	auto &data = bind_data->Cast<SuggestBindData>();
	serializer.WriteProperty(100, "min_presence", data.min_presence);
	serializer.WriteProperty(101, "min_fit", data.min_fit);
}

unique_ptr<FunctionData> SuggestDeserialize(Deserializer &deserializer, AggregateFunction &function) {
	auto min_presence = deserializer.ReadProperty<double>(100, "min_presence");
	auto min_fit = deserializer.ReadProperty<double>(101, "min_fit");
	auto &context = deserializer.Get<ClientContext &>();
	return make_uniq<SuggestBindData>(min_presence, min_fit, BufferManager::GetBufferManager(context));
}

// Tally one present scalar into a lane's fit counters. A JSON null is losslessly captured by every
// typed lane (the shred writer keeps it complete: the NULL lane reads back as `->>` NULL and the
// value stays in the residual as null), so it counts as fitting every primitive — it never biases
// or disqualifies a lane. A typed scalar counts only where JsonoScalarFitsPrimitive accepts it.
void AccumulateScalarFit(FitCounts &fit, const JsonoScalar &scalar) {
	bool is_null = scalar.kind == JsonoScalarKind::Null;
	for (auto primitive : kPrimitivePreference) {
		if (is_null || JsonoScalarFitsPrimitive(scalar, primitive)) {
			fit[uint8_t(primitive)]++;
		}
	}
}

// Record one scalar leaf observation against a candidate path.
void RecordScalar(SuggestCandidate &candidate, const JsonoScalar &scalar) {
	candidate.scalar_present++;
	AccumulateScalarFit(candidate.scalar_fit, scalar);
}

// Fetch (or create) the candidate for `path`, failing loud when a new path would push the distinct
// count past the cap. The map grows only through this helper so the cap has a single owner.
SuggestCandidate &GetCandidate(std::map<string, SuggestCandidate> &paths, const string &path) {
	auto it = paths.find(path);
	if (it != paths.end()) {
		return it->second;
	}
	if (paths.size() >= kMaxCandidatePaths) {
		throw InvalidInputException("jsono_suggest_shredding: distinct candidate path count exceeds the cap of %llu; "
		                            "the input has too many distinct keys to be a shredding target",
		                            (unsigned long long)kMaxCandidatePaths);
	}
	return paths[path];
}

// Fetch (or create) one object-array subfield's stat, sharing GetCandidate's cap and loud-throw
// rationale: silently truncating the subfield union would emit a partial STRUCT spec that looks
// authoritative. The two subfield maps (accumulate and combine) grow only through this helper.
SubfieldStat &GetSubfieldStat(std::map<string, SubfieldStat> &subfields, const string &key) {
	auto it = subfields.find(key);
	if (it != subfields.end()) {
		return it->second;
	}
	if (subfields.size() >= kMaxCandidatePaths) {
		throw InvalidInputException("jsono_suggest_shredding: distinct object-array subfield count exceeds the cap "
		                            "of %llu; the array elements have too many distinct keys to be a shredding "
		                            "target",
		                            (unsigned long long)kMaxCandidatePaths);
	}
	return subfields[key];
}

// Append `key` as a `$`-rooted JSONPath step, quoting it when a bare `.key` step would be
// mis-parsed (empty key, or a delimiter/quote/backslash inside it). Mirrors jsono_entries'
// AppendKeySegment so a suggested nested path re-parses to the same steps.
void AppendJsonPathStep(string &path, nonstd::string_view key) {
	bool needs_quote = key.empty();
	for (char c : key) {
		if (c == '.' || c == '[' || c == ']' || c == '"' || c == '\\') {
			needs_quote = true;
			break;
		}
	}
	path.push_back('.');
	if (!needs_quote) {
		path.append(key.data(), key.size());
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

// Name a top-level key in the candidate map (and so in the emitted spec). The common case is the
// bare name; a key the spec parser would misread as a JSONPath (`$`-leading, see ParseShredPathSpec)
// or reject as a reserved layout name (`body`) is spelled through its quoted `$.`-rooted path form,
// which names the same top-level key.
string TopLevelSpecName(nonstd::string_view key) {
	if (key[0] != '$' && key != "body") {
		return string(key.data(), key.size());
	}
	string path = "$";
	AppendJsonPathStep(path, key);
	return path;
}

// Classify a top-level array value at `cursor` into the scalar-array / object-array counters. The
// cursor arrives at ARR_START with correct stream cursors and is advanced past the whole array. A
// row fits scalar-array primitive p when every element is a scalar fitting p; it is an object-array
// row when every element is an object, in which case each element's scalar subfields are tallied.
void ClassifyTopLevelArray(const JsonoView &view, JsonoCursor &cursor, SuggestCandidate &candidate) {
	candidate.array_rows++;
	auto end_pos = ReadArrayEndPos(view, cursor.pos);
	cursor.pos++; // ARR_START consumes no stream entries; the stream cursors are the first element's

	bool all_scalar = true;
	bool all_object = true;
	FitCounts row_scalar_fit;
	row_scalar_fit.fill(1); // "every element fits p" — flipped off as a non-fitting element appears
	std::map<string, SubfieldStat> row_subfields;
	idx_t element_count = 0;

	while (cursor.pos < end_pos) {
		element_count++;
		auto element_tag = SlotTag(view.SlotAt(cursor.pos));
		if (element_tag == tag::OBJ_START) {
			all_scalar = false;
			candidate.saw_object_element = true;
			auto layout = ReadObjectLayout(view, cursor.pos);
			cursor.pos = layout.value_start;
			for (size_t i = 0; i < layout.key_count; i++) {
				auto key_slot = view.SlotAt(layout.key_start + i);
				if (SlotTag(key_slot) != tag::KEY) {
					throw InvalidInputException("malformed JSONO: expected KEY slot");
				}
				auto sub_name = view.KeyAt(SlotPayload(key_slot));
				auto sub_tag = SlotTag(view.SlotAt(cursor.pos));
				if (sub_name.empty()) {
					SkipValueFast(view, cursor);
					continue;
				}
				auto &sub = row_subfields[string(sub_name.data(), sub_name.size())];
				sub.present++;
				if (sub_tag == tag::OBJ_START || sub_tag == tag::ARR_START) {
					// A non-scalar subfield is present but never liftable: it fits no lane primitive,
					// so leaving fit untouched drops it under min_fit.
					SkipValueFast(view, cursor);
				} else {
					auto scalar = DecodeScalarAt(view, cursor);
					AccumulateScalarFit(sub.fit, scalar);
				}
			}
			if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
				throw InvalidInputException("malformed JSONO: object value span mismatch");
			}
			cursor.pos++;
		} else if (element_tag == tag::ARR_START) {
			all_scalar = false;
			all_object = false;
			SkipValueFast(view, cursor);
		} else {
			all_object = false;
			candidate.saw_scalar_element = true;
			auto scalar = DecodeScalarAt(view, cursor);
			// A null element never disqualifies a lane: the scalar-array writer keeps a null element
			// verbatim in the residual skeleton, so the array still round-trips losslessly into any
			// element type (mirrors the complete-null handling for scalar leaves).
			if (scalar.kind != JsonoScalarKind::Null) {
				for (auto primitive : kPrimitivePreference) {
					if (!JsonoScalarFitsPrimitive(scalar, primitive)) {
						row_scalar_fit[uint8_t(primitive)] = 0;
					}
				}
			}
		}
	}
	cursor.pos = end_pos + 1;

	if (all_scalar) {
		for (auto primitive : kPrimitivePreference) {
			if (row_scalar_fit[uint8_t(primitive)]) {
				candidate.sarray_fit[uint8_t(primitive)]++;
			}
		}
	}
	if (all_object) {
		candidate.oarray_object_rows++;
		candidate.object_elements += element_count;
		for (auto &entry : row_subfields) {
			auto &agg = GetSubfieldStat(candidate.subfields, entry.first);
			agg.present += entry.second.present;
			for (idx_t p = 0; p < kPrimitivePreference.size(); p++) {
				agg.fit[p] += entry.second.fit[p];
			}
		}
	}
}

// Walk a nested object, recording each nested scalar leaf under its `$.parent.child` path and
// descending into further nested objects while `depth` (the 1-based key depth of the children being
// scanned) stays below JSONO_AUTO_SHRED_MAX_DEPTH — so it mirrors what auto-shred lifts. `base` is
// the already-built path prefix of this object (`$.URL` for the object at key URL). An object too
// deep to descend, or any nested array, is recorded as a nonrole occurrence and skipped. The cursor
// arrives at the nested OBJ_START and is advanced past the whole object.
void WalkNestedObject(const JsonoView &view, JsonoCursor &cursor, const string &base,
                      std::map<string, SuggestCandidate> &paths, idx_t depth) {
	auto layout = ReadObjectLayout(view, cursor.pos);
	cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		auto child_key = view.KeyAt(SlotPayload(key_slot));
		auto child_tag = SlotTag(view.SlotAt(cursor.pos));
		if (child_key.empty()) {
			// An empty nested key is out of the shred universe (its lane name would carry an empty quoted
			// path step) — it names no candidate at all, so it is not even a nonrole occurrence.
			SkipValueFast(view, cursor);
			continue;
		}
		string path = base;
		AppendJsonPathStep(path, child_key);
		if (child_tag == tag::OBJ_START && depth < JSONO_AUTO_SHRED_MAX_DEPTH) {
			// A nested object still within shred depth: record it as a nonrole occurrence of its own path
			// (so a path that is a scalar in some rows and an object in others drops at min_fit = 1.0) and
			// descend, exactly as AccumulateDocument does for a top-level object.
			GetCandidate(paths, path).nonrole_present++;
			WalkNestedObject(view, cursor, path, paths, depth + 1);
			continue;
		}
		if (child_tag == tag::OBJ_START || child_tag == tag::ARR_START) {
			// A too-deep object (out of the auto-shred universe) or any nested array carries `$.a.b` but
			// fits no scalar lane: count it into total_present without feeding the scalar role so a nested
			// path that is a scalar in some rows and a container in others drops at min_fit = 1.0.
			GetCandidate(paths, path).nonrole_present++;
			SkipValueFast(view, cursor);
			continue;
		}
		auto scalar = DecodeScalarAt(view, cursor);
		RecordScalar(GetCandidate(paths, path), scalar);
	}
	if (cursor.pos >= view.Slots() || SlotTag(view.SlotAt(cursor.pos)) != tag::OBJ_END) {
		throw InvalidInputException("malformed JSONO: object value span mismatch");
	}
	cursor.pos++;
}

// Accumulate one document's contributions over the auto-shred universe: top-level scalar keys
// (bare name), nested scalar keys (`$.parent.child`(`.grandchild`) up to JSONO_AUTO_SHRED_MAX_DEPTH),
// and top-level array keys (scalar-
// or object-array). A non-object document contributes nothing (no key to name a shred). Empty keys
// (at any level) are out of the universe: their lane name would carry an empty quoted path step
// (`$.""`), which the shredded readers do not support.
void AccumulateDocument(const JsonoView &view, std::map<string, SuggestCandidate> &paths) {
	if (view.Slots() == 0 || SlotTag(view.SlotAt(0)) != tag::OBJ_START) {
		return;
	}
	auto layout = ReadObjectLayout(view, 0);
	JsonoCursor cursor;
	cursor.pos = layout.value_start;
	for (size_t i = 0; i < layout.key_count; i++) {
		auto key_slot = view.SlotAt(layout.key_start + i);
		if (SlotTag(key_slot) != tag::KEY) {
			throw InvalidInputException("malformed JSONO: expected KEY slot");
		}
		auto key = view.KeyAt(SlotPayload(key_slot));
		auto value_tag = SlotTag(view.SlotAt(cursor.pos));
		if (key.empty()) {
			SkipValueFast(view, cursor);
		} else if (value_tag == tag::OBJ_START) {
			// The object value carries this top-level key but fits no scalar/array role (its children feed
			// the nested `$.key.child` candidates instead), so record it as a nonrole occurrence of the
			// parent path — a key that is a scalar in some rows and an object in others drops at min_fit.
			GetCandidate(paths, TopLevelSpecName(key)).nonrole_present++;
			string base = "$";
			AppendJsonPathStep(base, key);
			WalkNestedObject(view, cursor, base, paths, 2);
		} else if (value_tag == tag::ARR_START) {
			ClassifyTopLevelArray(view, cursor, GetCandidate(paths, TopLevelSpecName(key)));
		} else {
			auto scalar = DecodeScalarAt(view, cursor);
			RecordScalar(GetCandidate(paths, TopLevelSpecName(key)), scalar);
		}
	}
}

// Pick the best lane primitive whose fitting fraction over `present` meets `min_fit`, in the fixed
// preference order. Returns false when none qualifies (or there is nothing present).
bool PickLanePrimitive(const FitCounts &fit, idx_t present, double min_fit, JsonoScalarPrimitive &out) {
	if (present == 0) {
		return false;
	}
	double threshold = min_fit * double(present);
	for (auto primitive : kPrimitivePreference) {
		if (double(fit[uint8_t(primitive)]) >= threshold - 1e-9) {
			out = primitive;
			return true;
		}
	}
	return false;
}

// Append a struct-literal key, bare when it is a plain SQL identifier, else double-quoted with
// embedded quotes doubled (SQL identifier escaping — the spec must re-parse when pasted into
// jsono(value, shredding := {...})).
void AppendSpecKey(string &out, const string &key) {
	bool simple = !key.empty();
	for (size_t i = 0; simple && i < key.size(); i++) {
		char c = key[i];
		bool ident = (c == '_') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (i > 0 && c >= '0' && c <= '9');
		simple = ident;
	}
	if (simple) {
		out.append(key);
		return;
	}
	out.push_back('"');
	for (char c : key) {
		if (c == '"') {
			out.push_back('"');
		}
		out.push_back(c);
	}
	out.push_back('"');
}

// Build the object-array STRUCT(...)[] type string, or empty when no subfield qualifies.
string BuildObjectArrayType(const SuggestCandidate &candidate, double min_presence, double min_fit) {
	string fields;
	double subfield_presence_floor = min_presence * double(candidate.object_elements);
	for (auto &entry : candidate.subfields) {
		if (double(entry.second.present) < subfield_presence_floor - 1e-9) {
			continue;
		}
		JsonoScalarPrimitive primitive;
		if (!PickLanePrimitive(entry.second.fit, entry.second.present, min_fit, primitive)) {
			continue;
		}
		if (!fields.empty()) {
			fields.append(", ");
		}
		AppendSpecKey(fields, entry.first);
		fields.push_back(' ');
		fields.append(JsonoScalarPrimitiveTypeName(primitive));
	}
	if (fields.empty()) {
		return string();
	}
	return "STRUCT(" + fields + ")[]";
}

// Choose one candidate path's shred type string, or empty when no role qualifies. Roles compete by
// presence (the count of rows the path appears in that role); scalar beats scalar-array beats
// object-array on a tie. Each role must also clear min_presence over the group's non-NULL rows.
// min_fit is measured over the path's TOTAL present occurrences (every role plus nonrole_present): a
// role that fits its own rows but covers only part of a mixed path falls below min_fit, so a
// genuinely mixed path drops instead of a majority role silently winning.
string ChooseShredType(const SuggestCandidate &candidate, idx_t non_null_rows, double min_presence, double min_fit) {
	double presence_floor = min_presence * double(non_null_rows);
	// A row observes the path as exactly one of a scalar leaf, an array, or a nonrole value (an object,
	// or a nested container), so these three counts sum to the count of rows carrying this path.
	idx_t total_present = candidate.scalar_present + candidate.array_rows + candidate.nonrole_present;
	string best;
	idx_t best_present = 0;

	// Scalar leaf.
	{
		JsonoScalarPrimitive primitive;
		if (candidate.scalar_present > 0 && double(candidate.scalar_present) >= presence_floor - 1e-9 &&
		    PickLanePrimitive(candidate.scalar_fit, total_present, min_fit, primitive)) {
			best = string("'") + JsonoScalarPrimitiveTypeName(primitive) + "'";
			best_present = candidate.scalar_present;
		}
	}
	// Scalar array (strictly wins only on a larger presence than the scalar role).
	if (candidate.array_rows > 0 && candidate.saw_scalar_element &&
	    double(candidate.array_rows) >= presence_floor - 1e-9 && candidate.array_rows > best_present) {
		JsonoScalarPrimitive primitive;
		if (PickLanePrimitive(candidate.sarray_fit, total_present, min_fit, primitive)) {
			best = string("'") + JsonoScalarPrimitiveTypeName(primitive) + "[]'";
			best_present = candidate.array_rows;
		}
	}
	// Object array.
	if (candidate.array_rows > 0 && candidate.saw_object_element &&
	    double(candidate.array_rows) >= presence_floor - 1e-9 && candidate.array_rows > best_present &&
	    double(candidate.oarray_object_rows) >= min_fit * double(total_present) - 1e-9) {
		auto struct_type = BuildObjectArrayType(candidate, min_presence, min_fit);
		if (!struct_type.empty()) {
			best = string("'") + struct_type + "'";
			best_present = candidate.array_rows;
		}
	}
	return best;
}

struct SuggestAggregate {
	static void Initialize(SuggestState &state) {
		state.paths = nullptr;
		state.non_null_rows = 0;
		state.mem.reserved = 0;
		state.mem.buffer_manager = nullptr;
	}
	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		state.mem.Release();
		delete state.paths;
		state.paths = nullptr;
	}
};

unique_ptr<FunctionData> JsonoSuggestBind(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	double min_presence = 0.5;
	double min_fit = 1.0;
	bool seen_min_presence = false;
	bool seen_min_fit = false;
	for (idx_t i = 1; i < arguments.size(); i++) {
		auto &arg = arguments[i];
		auto alias = arg->GetAlias();
		if (alias != "min_presence" && alias != "min_fit") {
			throw BinderException("jsono_suggest_shredding: unknown argument '%s' (pass min_presence := <fraction>, "
			                      "min_fit := <fraction>)",
			                      alias);
		}
		auto &seen = alias == "min_presence" ? seen_min_presence : seen_min_fit;
		if (seen) {
			throw BinderException("jsono_suggest_shredding: duplicate argument '%s'", alias);
		}
		seen = true;
		if (arg->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arg->IsFoldable()) {
			throw BinderException("jsono_suggest_shredding: %s must be constant", alias);
		}
		auto value = ExpressionExecutor::EvaluateScalar(context, *arg);
		if (value.IsNull()) {
			throw BinderException("jsono_suggest_shredding: %s must not be NULL", alias);
		}
		auto fraction = value.GetValue<double>();
		if (!(fraction >= 0.0 && fraction <= 1.0)) {
			throw BinderException("jsono_suggest_shredding: %s must be in [0.0, 1.0], got %f", alias, fraction);
		}
		if (alias == "min_presence") {
			min_presence = fraction;
		} else {
			min_fit = fraction;
		}
	}
	auto &type = arguments[0]->return_type;
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (IsShreddedJsonoType(type)) {
		throw BinderException("jsono_suggest_shredding: input is already shredded; pass the plain jsono column it was "
		                      "shredded from (jsono_shred_stats reports on an existing shredded column)");
	}
	if (type.id() != LogicalTypeId::SQLNULL && !IsJsonoType(type)) {
		throw BinderException("jsono_suggest_shredding: input must be plain JSONO");
	}
	function.arguments[0] = JsonoType();
	return make_uniq<SuggestBindData>(min_presence, min_fit, BufferManager::GetBufferManager(context));
}

void SuggestAccumulate(SuggestState &state, Vector &input, idx_t count) {
	if (!state.paths) {
		state.paths = new std::map<string, SuggestCandidate>();
	}
	JsonoRowReader reader;
	reader.Init(input, count);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		state.non_null_rows++;
		AccumulateDocument(view, *state.paths);
	}
}

void JsonoSuggestSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, data_ptr_t state_ptr,
                              idx_t count) {
	auto &state = *reinterpret_cast<SuggestState *>(state_ptr);
	SuggestAccumulate(state, inputs[0], count);
	auto &bind_data = aggr_input_data.bind_data->Cast<SuggestBindData>();
	state.mem.Resize(bind_data.buffer_manager, SuggestFootprint(state));
}

void JsonoSuggestUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &states, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<SuggestBindData>();
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<SuggestState *>(state_fmt);
	// Group each row to its own state: one chunk-level reader, per-row map inserts.
	JsonoRowReader reader;
	reader.Init(inputs[0], count);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			continue;
		}
		auto &state = *state_data[state_fmt.sel->get_index(row)];
		if (!state.paths) {
			state.paths = new std::map<string, SuggestCandidate>();
		}
		state.non_null_rows++;
		AccumulateDocument(view, *state.paths);
	}
	// Account each distinct state once per chunk (the candidate-map footprint walk is too costly per row).
	AccountDistinctStates<SuggestState>(
	    count, bind_data.buffer_manager,
	    [&](idx_t row) -> SuggestState & { return *state_data[state_fmt.sel->get_index(row)]; },
	    [](const SuggestState &s) { return SuggestFootprint(s); });
}

void MergeSuggestCandidate(SuggestCandidate &target, const SuggestCandidate &source) {
	target.scalar_present += source.scalar_present;
	target.nonrole_present += source.nonrole_present;
	target.array_rows += source.array_rows;
	target.saw_scalar_element |= source.saw_scalar_element;
	target.saw_object_element |= source.saw_object_element;
	target.oarray_object_rows += source.oarray_object_rows;
	target.object_elements += source.object_elements;
	for (idx_t p = 0; p < kPrimitivePreference.size(); p++) {
		target.scalar_fit[p] += source.scalar_fit[p];
		target.sarray_fit[p] += source.sarray_fit[p];
	}
	for (auto &entry : source.subfields) {
		auto &agg = GetSubfieldStat(target.subfields, entry.first);
		agg.present += entry.second.present;
		for (idx_t p = 0; p < kPrimitivePreference.size(); p++) {
			agg.fit[p] += entry.second.fit[p];
		}
	}
}

void JsonoSuggestCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<SuggestBindData>();
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<SuggestState *>(source_fmt);
	auto target_data = FlatVector::GetData<SuggestState *>(target);
	for (idx_t row = 0; row < count; row++) {
		auto &source_state = *source_data[source_fmt.sel->get_index(row)];
		if (!source_state.paths) {
			continue;
		}
		auto &target_state = *target_data[row];
		if (!target_state.paths) {
			target_state.paths = new std::map<string, SuggestCandidate>();
		}
		target_state.non_null_rows += source_state.non_null_rows;
		for (auto &entry : *source_state.paths) {
			MergeSuggestCandidate(GetCandidate(*target_state.paths, entry.first), entry.second);
		}
	}
	// Only targets grow (sources are merged in, not moved out); account each distinct target once.
	AccountDistinctStates<SuggestState>(
	    count, bind_data.buffer_manager, [&](idx_t row) -> SuggestState & { return *target_data[row]; },
	    [](const SuggestState &s) { return SuggestFootprint(s); });
}

void JsonoSuggestFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                          idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<SuggestState *>(state_fmt);
	auto &bind_data = aggr_input_data.bind_data->Cast<SuggestBindData>();
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto rid = offset + i;
		auto &state = *state_data[state_fmt.sel->get_index(i)];
		if (!state.paths || state.non_null_rows == 0) {
			FlatVector::SetNull(result, rid, true);
			continue;
		}
		string spec = "{";
		bool first = true;
		// std::map iterates in sorted (canonical) path order.
		for (auto &entry : *state.paths) {
			auto type = ChooseShredType(entry.second, state.non_null_rows, bind_data.min_presence, bind_data.min_fit);
			if (type.empty()) {
				continue;
			}
			if (!first) {
				spec.append(", ");
			}
			first = false;
			AppendSpecKey(spec, entry.first);
			spec.append(": ");
			spec.append(type);
		}
		if (first) {
			FlatVector::SetNull(result, rid, true);
			continue;
		}
		spec.push_back('}');
		result_data[rid] = StringVector::AddString(result, spec);
	}
}

// ===== jsono_shred_stats: per-shred lane / divert / complete rates over a shredded column =====

struct ShredStatCounters {
	idx_t lane_present = 0;
	idx_t divert = 0;
	idx_t complete = 0;
};

struct ShredStatsState {
	std::vector<ShredStatCounters> *counters;
	idx_t non_null_rows;
	JsonoMemoryReservation mem;
};

// The counters vector is allocated once to the (fixed) shred count and never grows, so its footprint is a
// single constant term.
idx_t ShredStatsFootprint(const ShredStatsState &state) {
	if (!state.counters) {
		return 0;
	}
	return state.counters->capacity() * sizeof(ShredStatCounters);
}

// One shred of the input type: its output path/type strings plus the residual path steps used to
// detect a diverted value (lane NULL but the path is still present in the residual).
struct ShredStatDescriptor {
	string path;
	string type_name;
	ShredKind kind;
	vector<PathStep> steps;
};

struct ShredStatsBindData : public FunctionData {
	vector<ShredStatDescriptor> shreds;
	// Carried into Update/Combine to account the counters vector; re-captured on plan round-trips (no
	// serialize callback, so deserialize re-runs the bind).
	BufferManager &buffer_manager;

	explicit ShredStatsBindData(BufferManager &buffer_manager) : buffer_manager(buffer_manager) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ShredStatsBindData>(buffer_manager);
		copy->shreds = shreds;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ShredStatsBindData>();
		if (shreds.size() != other.shreds.size()) {
			return false;
		}
		for (idx_t i = 0; i < shreds.size(); i++) {
			if (shreds[i].path != other.shreds[i].path || shreds[i].type_name != other.shreds[i].type_name) {
				return false;
			}
		}
		return true;
	}
};

struct ShredStatsAggregate {
	static void Initialize(ShredStatsState &state) {
		state.counters = nullptr;
		state.non_null_rows = 0;
		state.mem.reserved = 0;
		state.mem.buffer_manager = nullptr;
	}
	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		state.mem.Release();
		delete state.counters;
		state.counters = nullptr;
	}
};

LogicalType ShredStatsResultType() {
	return LogicalType::LIST(LogicalType::STRUCT({{"path", LogicalType::VARCHAR},
	                                              {"type", LogicalType::VARCHAR},
	                                              {"lane_rate", LogicalType::DOUBLE},
	                                              {"divert_rate", LogicalType::DOUBLE},
	                                              {"complete_rate", LogicalType::DOUBLE}}));
}

unique_ptr<FunctionData> JsonoShredStatsBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw BinderException("jsono_shred_stats() requires a single shredded JSONO argument");
	}
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto &type = arguments[0]->return_type;
	JsonoRequireExtensionOptimizerForShredded(context, type, "jsono_shred_stats");
	if (!IsShreddedJsonoType(type)) {
		throw BinderException("jsono_shred_stats() input must be shredded JSONO; use jsono_suggest_shredding to "
		                      "propose shreds for a plain column");
	}
	JsonoLayoutType layout;
	TryParseJsonoLayoutType(type, layout);
	auto bind_data = make_uniq<ShredStatsBindData>(BufferManager::GetBufferManager(context));
	for (auto &shred : layout.shreds) {
		ShredStatDescriptor descriptor;
		descriptor.path = shred.first;
		descriptor.type_name = shred.second.ToString();
		descriptor.kind = ClassifyShredKind(shred.second);
		descriptor.steps = ShredNamePath(shred.first, "jsono_shred_stats shred");
		bind_data->shreds.push_back(std::move(descriptor));
	}
	function.arguments[0] = type;
	return std::move(bind_data);
}

// Per-chunk read handles: the residual reader plus each shred's value lane (and, for a scalar shred,
// its `complete` flag lane).
struct ShredStatsLanes {
	JsonoRowReader reader;
	vector<UnifiedVectorFormat> value_fmt;
	vector<UnifiedVectorFormat> complete_fmt; // valid only for scalar shreds
};

void InitShredStatsLanes(Vector &input, idx_t count, const ShredStatsBindData &bind_data, ShredStatsLanes &lanes) {
	// Whole-document policy: the divert probe interprets residual presence, and a diagnostic that is
	// natural to run over a suspect column must fail loud on a narrowed row (a manifest naming a shred
	// the type no longer carries) like every other reader — not report plausible rates over it. An
	// honest shredded column's manifest names exactly the type's shreds, so verification passes (and
	// is memoized to a byte-compare per row).
	lanes.reader.Init(input, count);
	auto nshreds = bind_data.shreds.size();
	lanes.value_fmt.resize(nshreds);
	lanes.complete_fmt.resize(nshreds);
	for (idx_t f = 0; f < nshreds; f++) {
		JsonoShredVector(input, f).ToUnifiedFormat(count, lanes.value_fmt[f]);
		if (bind_data.shreds[f].kind == ShredKind::Scalar) {
			JsonoShredCompleteVector(input, f).ToUnifiedFormat(count, lanes.complete_fmt[f]);
		}
	}
}

void AccumulateShredStatsRow(ShredStatsState &state, const ShredStatsBindData &bind_data, ShredStatsLanes &lanes,
                             idx_t row, JsonoRowState row_state, const JsonoView &view) {
	state.non_null_rows++;
	bool have_residual = row_state == JsonoRowState::Value;
	for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
		auto &counters = (*state.counters)[f];
		bool is_scalar = bind_data.shreds[f].kind == ShredKind::Scalar;
		auto value_idx = lanes.value_fmt[f].sel->get_index(row);
		bool lane_present = lanes.value_fmt[f].validity.RowIsValid(value_idx);
		if (lane_present) {
			counters.lane_present++;
		}
		// Array shreds carry no completeness flag; report complete_rate 1.0. A scalar shred's flag is 1
		// on a fit, an absent path, and an explicit null (a bare NULL-lane read is the full `->>` answer).
		bool complete = true;
		if (is_scalar) {
			auto complete_idx = lanes.complete_fmt[f].sel->get_index(row);
			complete = lanes.complete_fmt[f].validity.RowIsValid(complete_idx) &&
			           UnifiedVectorFormat::GetData<int8_t>(lanes.complete_fmt[f])[complete_idx] == 1;
		}
		if (complete) {
			counters.complete++;
		}
		// A divert is a value that did not fit its shred type and stayed in the residual — the signal
		// "the type is too narrow". For a scalar shred a complete NULL lane is an absent path or an
		// explicit null (lossless, not a diversion), so gate scalar diverts on an incomplete lane; only a
		// value that actually spilled counts. Array shreds have no completeness flag: their lane-NULL rows
		// where the path is present carry a whole non-array value in the residual — a genuine divert.
		bool diverted_lane = is_scalar ? (!lane_present && !complete) : !lane_present;
		if (diverted_lane && have_residual && LocatePath(view, bind_data.shreds[f].steps).found) {
			counters.divert++;
		}
	}
}

void JsonoShredStatsSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, data_ptr_t state_ptr,
                                 idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<ShredStatsBindData>();
	auto &state = *reinterpret_cast<ShredStatsState *>(state_ptr);
	if (!state.counters) {
		state.counters = new std::vector<ShredStatCounters>(bind_data.shreds.size());
	}
	ShredStatsLanes lanes;
	InitShredStatsLanes(inputs[0], count, bind_data, lanes);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		auto row_state = lanes.reader.Read(row, blob, view);
		if (row_state == JsonoRowState::Null) {
			continue;
		}
		AccumulateShredStatsRow(state, bind_data, lanes, row, row_state, view);
	}
	state.mem.Resize(bind_data.buffer_manager, ShredStatsFootprint(state));
}

void JsonoShredStatsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &states, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<ShredStatsBindData>();
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<ShredStatsState *>(state_fmt);
	ShredStatsLanes lanes;
	InitShredStatsLanes(inputs[0], count, bind_data, lanes);
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		JsonoBlobRow blob;
		auto row_state = lanes.reader.Read(row, blob, view);
		if (row_state == JsonoRowState::Null) {
			continue;
		}
		auto &state = *state_data[state_fmt.sel->get_index(row)];
		if (!state.counters) {
			state.counters = new std::vector<ShredStatCounters>(bind_data.shreds.size());
		}
		AccumulateShredStatsRow(state, bind_data, lanes, row, row_state, view);
	}
	AccountDistinctStates<ShredStatsState>(
	    count, bind_data.buffer_manager,
	    [&](idx_t row) -> ShredStatsState & { return *state_data[state_fmt.sel->get_index(row)]; },
	    [](const ShredStatsState &s) { return ShredStatsFootprint(s); });
}

void JsonoShredStatsCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<ShredStatsBindData>();
	UnifiedVectorFormat source_fmt;
	source.ToUnifiedFormat(count, source_fmt);
	auto source_data = UnifiedVectorFormat::GetData<ShredStatsState *>(source_fmt);
	auto target_data = FlatVector::GetData<ShredStatsState *>(target);
	for (idx_t row = 0; row < count; row++) {
		auto &source_state = *source_data[source_fmt.sel->get_index(row)];
		if (!source_state.counters) {
			continue;
		}
		auto &target_state = *target_data[row];
		if (!target_state.counters) {
			target_state.counters = new std::vector<ShredStatCounters>(bind_data.shreds.size());
		}
		target_state.non_null_rows += source_state.non_null_rows;
		for (idx_t f = 0; f < bind_data.shreds.size(); f++) {
			(*target_state.counters)[f].lane_present += (*source_state.counters)[f].lane_present;
			(*target_state.counters)[f].divert += (*source_state.counters)[f].divert;
			(*target_state.counters)[f].complete += (*source_state.counters)[f].complete;
		}
	}
	// A target's counters are allocated on first touch (fixed size); account each distinct target once.
	AccountDistinctStates<ShredStatsState>(
	    count, bind_data.buffer_manager, [&](idx_t row) -> ShredStatsState & { return *target_data[row]; },
	    [](const ShredStatsState &s) { return ShredStatsFootprint(s); });
}

void JsonoShredStatsFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	UnifiedVectorFormat state_fmt;
	states.ToUnifiedFormat(count, state_fmt);
	auto state_data = UnifiedVectorFormat::GetData<ShredStatsState *>(state_fmt);
	auto &bind_data = aggr_input_data.bind_data->Cast<ShredStatsBindData>();
	auto nshreds = bind_data.shreds.size();
	for (idx_t i = 0; i < count; i++) {
		auto rid = offset + i;
		auto &state = *state_data[state_fmt.sel->get_index(i)];
		if (!state.counters || state.non_null_rows == 0) {
			SetListRowNull(result, rid);
			continue;
		}
		auto start = ListVector::GetListSize(result);
		EnsureListCapacity(result, start + nshreds);
		auto &entry = ListVector::GetEntry(result);
		auto &child = StructVector::GetEntries(entry);
		auto path_data = FlatVector::GetData<string_t>(*child[0]);
		auto type_data = FlatVector::GetData<string_t>(*child[1]);
		auto lane_data = FlatVector::GetData<double>(*child[2]);
		auto divert_data = FlatVector::GetData<double>(*child[3]);
		auto complete_data = FlatVector::GetData<double>(*child[4]);
		double denom = double(state.non_null_rows);
		for (idx_t f = 0; f < nshreds; f++) {
			auto idx = start + f;
			auto &counters = (*state.counters)[f];
			path_data[idx] = StringVector::AddString(*child[0], bind_data.shreds[f].path);
			type_data[idx] = StringVector::AddString(*child[1], bind_data.shreds[f].type_name);
			lane_data[idx] = double(counters.lane_present) / denom;
			divert_data[idx] = double(counters.divert) / denom;
			complete_data[idx] = double(counters.complete) / denom;
		}
		FinishListRow(result, rid, start, nshreds);
	}
}

} // namespace

void RegisterJsonoAdvisor(ExtensionLoader &loader) {
	// jsono_suggest_shredding(j [, min_presence := DOUBLE] [, min_fit := DOUBLE]) -> VARCHAR
	AggregateFunctionSet suggest_set("jsono_suggest_shredding");
	for (auto &signature :
	     {vector<LogicalType> {LogicalType::ANY}, vector<LogicalType> {LogicalType::ANY, LogicalType::DOUBLE},
	      vector<LogicalType> {LogicalType::ANY, LogicalType::DOUBLE, LogicalType::DOUBLE}}) {
		AggregateFunction suggest(signature, LogicalType::VARCHAR, AggregateFunction::StateSize<SuggestState>,
		                          AggregateFunction::StateInitialize<SuggestState, SuggestAggregate>,
		                          JsonoSuggestUpdate, JsonoSuggestCombine, JsonoSuggestFinalize,
		                          // SPECIAL_HANDLING so a NULL min_presence/min_fit constant reaches the bind validator
		                          // instead of folding the whole aggregate call to NULL (the named-arg NULL-fold trap).
		                          FunctionNullHandling::SPECIAL_HANDLING, JsonoSuggestSimpleUpdate, JsonoSuggestBind,
		                          AggregateFunction::StateDestroy<SuggestState, SuggestAggregate>);
		suggest.serialize = SuggestSerialize;
		suggest.deserialize = SuggestDeserialize;
		suggest_set.AddFunction(suggest);
	}
	loader.RegisterFunction(suggest_set);

	// jsono_shred_stats(shredded) -> LIST<STRUCT(path, type, lane_rate, divert_rate, complete_rate)>
	AggregateFunction stats("jsono_shred_stats", {LogicalType::ANY}, ShredStatsResultType(),
	                        AggregateFunction::StateSize<ShredStatsState>,
	                        AggregateFunction::StateInitialize<ShredStatsState, ShredStatsAggregate>,
	                        JsonoShredStatsUpdate, JsonoShredStatsCombine, JsonoShredStatsFinalize,
	                        FunctionNullHandling::DEFAULT_NULL_HANDLING, JsonoShredStatsSimpleUpdate,
	                        JsonoShredStatsBind, AggregateFunction::StateDestroy<ShredStatsState, ShredStatsAggregate>);
	loader.RegisterFunction(stats);
}

} // namespace duckdb
