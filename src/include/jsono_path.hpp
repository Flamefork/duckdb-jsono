#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

#include "string_view.hpp"

#include <cctype>
#include <cstdint>
#include <limits>
#include <utility>

namespace duckdb {

// Shared JSONPath grammar for JSONO functions (jsono_transform navigation and
// jsono_keys/jsono_type path arguments). Supports dotted keys, quoted keys,
// numeric array indices, and a single [*] wildcard level.
enum class PathStepKind : uint8_t { Key, Index, Wildcard };

struct PathStep {
	PathStep() = default;
	PathStep(PathStepKind kind_p, string key_p, idx_t index_p) : kind(kind_p), key(std::move(key_p)), index(index_p) {
	}

	PathStepKind kind;
	string key;
	idx_t index = 0;
};

struct JsonoPathSpec {
	JsonoPathSpec() = default;
	JsonoPathSpec(string text_p, vector<PathStep> steps_p) : text(std::move(text_p)), steps(std::move(steps_p)) {
	}

	string text;
	vector<PathStep> steps;
};

[[noreturn]] inline void ThrowInvalidPath(const char *function_name, const string &path) {
	throw BinderException("%s: invalid path '%s'", function_name, path);
}

inline vector<PathStep> ParseJsonoPath(const string &path, const char *function_name) {
	if (path.empty() || path[0] != '$') {
		ThrowInvalidPath(function_name, path);
	}
	vector<PathStep> steps;
	idx_t wildcard_count = 0;
	size_t i = 1;
	while (i < path.size()) {
		if (path[i] == '.') {
			i++;
			if (i >= path.size()) {
				ThrowInvalidPath(function_name, path);
			}
			if (path[i] == '"') {
				i++;
				string key;
				bool closed = false;
				while (i < path.size()) {
					auto c = path[i++];
					if (c == '"') {
						closed = true;
						break;
					}
					if (c == '\\') {
						if (i >= path.size() || (path[i] != '"' && path[i] != '\\')) {
							ThrowInvalidPath(function_name, path);
						}
						key.push_back(path[i++]);
						continue;
					}
					key.push_back(c);
				}
				if (!closed) {
					ThrowInvalidPath(function_name, path);
				}
				steps.push_back(PathStep {PathStepKind::Key, std::move(key), 0});
				continue;
			}

			auto start = i;
			while (i < path.size() && path[i] != '.' && path[i] != '[' && path[i] != ']' && path[i] != '"') {
				i++;
			}
			if (i == start) {
				ThrowInvalidPath(function_name, path);
			}
			steps.push_back(PathStep {PathStepKind::Key, path.substr(start, i - start), 0});
			continue;
		}
		if (path[i] == '[') {
			if (steps.empty()) {
				ThrowInvalidPath(function_name, path);
			}
			i++;
			if (i >= path.size()) {
				ThrowInvalidPath(function_name, path);
			}
			if (path[i] == '*') {
				i++;
				if (i >= path.size() || path[i] != ']') {
					ThrowInvalidPath(function_name, path);
				}
				i++;
				wildcard_count++;
				if (wildcard_count > 1) {
					throw BinderException("%s: multiple wildcard levels are not supported", function_name);
				}
				steps.push_back(PathStep {PathStepKind::Wildcard, string(), 0});
				continue;
			}
			if (path[i] == '-') {
				throw BinderException("%s: negative array index is not supported", function_name);
			}
			if (!std::isdigit(static_cast<unsigned char>(path[i]))) {
				ThrowInvalidPath(function_name, path);
			}
			idx_t index = 0;
			while (i < path.size() && std::isdigit(static_cast<unsigned char>(path[i]))) {
				idx_t digit = idx_t(path[i] - '0');
				if (index > (std::numeric_limits<idx_t>::max() - digit) / 10) {
					throw BinderException("%s: array index out of range in path '%s'", function_name, path);
				}
				index = index * 10 + digit;
				i++;
			}
			if (i >= path.size() || path[i] != ']') {
				ThrowInvalidPath(function_name, path);
			}
			i++;
			steps.push_back(PathStep {PathStepKind::Index, string(), index});
			continue;
		}
		ThrowInvalidPath(function_name, path);
	}
	return steps;
}

// A bare key names a literal top-level object key, not a JSONPath expression: dots in the
// key (e.g. analytics "utm.source") must not be read as nesting.
inline vector<PathStep> LiteralKeyPath(const string &name) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Key, name, 0});
	return path;
}

// Whether `steps` is a path a shred lane may carry: a non-empty pure object-key chain. The residual
// emit removes one object key per step, and only an object key can be re-filled by the object
// overlay on reconstruction, so an array-index or root `$` lane could never be rebuilt.
inline bool IsObjectKeyPath(const vector<PathStep> &steps) {
	for (auto &step : steps) {
		if (step.kind != PathStepKind::Key) {
			return false;
		}
	}
	return !steps.empty();
}

// Append `key` to a `$`-rooted JSONPath as one `.key` step, quoting it exactly when a bare step
// would mis-parse: an empty key, or one carrying a character ParseJsonoPath treats as structural (a
// bare `.foo` step spans only up to the next . [ ] ", and a backslash is an escape inside a quoted
// key). The single owner of the quoting rule — the serializer below, the shred-spec emitters of the
// constructor and the advisor, and the jsono_entries key builder all call it, so a path this
// project prints always re-parses to the steps it came from.
inline void AppendJsonPathKey(string &path, nonstd::string_view key) {
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

inline vector<PathStep> ArrayIndexPath(idx_t index) {
	vector<PathStep> path;
	path.push_back(PathStep {PathStepKind::Index, string(), index});
	return path;
}

inline bool PathStepEquals(const PathStep &left, const PathStep &right) {
	return left.kind == right.kind && left.key == right.key && left.index == right.index;
}

inline bool PathStepsEqual(const vector<PathStep> &left, const vector<PathStep> &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (idx_t step_index = 0; step_index < left.size(); step_index++) {
		if (!PathStepEquals(left[step_index], right[step_index])) {
			return false;
		}
	}
	return true;
}

inline bool JsonoPathSpecEqual(const JsonoPathSpec &left, const JsonoPathSpec &right) {
	return left.text == right.text && PathStepsEqual(left.steps, right.steps);
}

// Serialize object-key / array-index steps back to a `$`-rooted JSONPath that ParseJsonoPath
// round-trips (the inverse of ParseJsonoPath). This is the project's single logical path form —
// `$.`-always, quoting per AppendJsonPathKey — shared by the spec DSL the constructor and the
// advisor emit and by every diagnostic that names a path. Fails for a leading non-key step or any
// wildcard, which ParseJsonoPath cannot root — the caller then declines rather than emit an
// unparseable path.
inline bool TryStepsToJsonPath(const vector<PathStep> &steps, string &out) {
	if (steps.empty() || steps[0].kind != PathStepKind::Key) {
		return false;
	}
	string path = "$";
	for (auto &step : steps) {
		switch (step.kind) {
		case PathStepKind::Key:
			AppendJsonPathKey(path, nonstd::string_view(step.key.data(), step.key.size()));
			break;
		case PathStepKind::Index:
			path.push_back('[');
			path.append(std::to_string(step.index));
			path.push_back(']');
			break;
		case PathStepKind::Wildcard:
			return false;
		}
	}
	out = std::move(path);
	return true;
}

// ===== The lane-name boundary =====
//
// A shred lane has two names. The LOGICAL path is the object-key chain it lifts out of the
// document: what reconstruct, render, jsono_entries and every diagnostic emit, and what the per-row
// shred manifest records. The PHYSICAL name is the STRUCT field the lane occupies inside `shreds`,
// and it is NOT that path as text — it is the path's STRUCTURE, serialized and encoded.
//
// DuckDB matches STRUCT field names case-insensitively over ASCII A-Z, so a lane named by its path
// verbatim collapses with another spelling of the same key: two sources writing `gclid` and `GCLID`
// merge into one lane, and the by-name cast then moves one key's values into the other key's lane —
// loud on the dropped spelling, silently wrong on the surviving one. The encoded alphabet has no
// upper case at all, so distinct paths can never produce names differing only in case: the collapse
// becomes unrepresentable rather than guarded against. The same bijection ends a second conflation —
// `gclid` and `$.gclid` are one path and now encode to one name, so how the public spec DSL spells
// a path stops leaking into the stored format.
//
// Two stages, both order-preserving, so sorting encoded names IS sorting paths:
//
//   1. serialize the steps — each key's raw bytes with 0x00 escaped as `00 FF`, then terminated by
//      `00 00`; keys concatenate in path order;
//   2. base32hex (RFC 4648, alphabet `0-9a-v`, lowercase, unpadded) over those bytes.
//
// The DOUBLED terminator is what makes stage 1 self-delimiting: 0x00 ALWAYS appears in a pair, so a
// decoder that has read a 0x00 decides on the single byte after it and never depends on what the
// next key starts with. A single-0x00 terminator is ambiguous the moment the next key starts with
// 0xFF: `61 00 FF 62 00` would then be both [`a`, `\xFFb`] and [`a\x00b`]. Doubled, they are
// `61 00 00 FF 62 00 00` and `61 00 FF 62 00 00` — distinct, which is why the pair is not optional.
//
// Order preservation is a property of THIS serialization, not of encoding in general (a
// length-prefixed variant breaks it: `a` sorts before `URL`→`path` as a path but after it once the
// length byte dominates). It is load-bearing for the two-pointer walkers, the sort-merges against
// the residual's byte-sorted document keys, and the binary searches of the merge fast path, so it
// is the first thing to re-verify if the serialization is ever revisited.

// Encode the lane path `steps` as its STRUCT field name. Every step must be an object key: an
// index or wildcard step has no lane to name (each step strips one object key from the residual),
// and admitting one here would mint a name whose decode contradicts the lane invariant.
inline string JsonoEncodeLaneName(const vector<PathStep> &steps) {
	if (steps.empty()) {
		throw InvalidInputException("jsono shred: a lane path must name at least one object key");
	}
	idx_t serialized_size = 0;
	for (auto &step : steps) {
		if (step.kind != PathStepKind::Key) {
			throw InvalidInputException("jsono shred: a lane path may only contain object keys");
		}
		serialized_size += step.key.size() + 2;
	}

	string serialized;
	serialized.reserve(serialized_size);
	for (auto &step : steps) {
		for (char c : step.key) {
			serialized.push_back(c);
			if (c == '\0') {
				serialized.push_back(static_cast<char>(0xFF));
			}
		}
		serialized.push_back('\0');
		serialized.push_back('\0');
	}

	const char digits[] = "0123456789abcdefghijklmnopqrstuv";
	string name;
	name.reserve((serialized.size() * 8 + 4) / 5);
	uint32_t buffer = 0;
	uint32_t bits = 0;
	for (char c : serialized) {
		buffer = (buffer << 8) | static_cast<unsigned char>(c);
		bits += 8;
		while (bits >= 5) {
			bits -= 5;
			name.push_back(digits[(buffer >> bits) & 0x1F]);
		}
	}
	if (bits > 0) {
		name.push_back(digits[(buffer << (5 - bits)) & 0x1F]);
	}
	return name;
}

inline bool JsonoBase32HexDigit(char c, uint8_t &value) {
	if (c >= '0' && c <= '9') {
		value = static_cast<uint8_t>(c - '0');
		return true;
	}
	// Upper case is rejected rather than folded: `C4000` decodes to the same bytes as `c4000`, so
	// accepting it would hand one lane two spellings — the very aliasing this codec removes.
	if (c >= 'a' && c <= 'v') {
		value = static_cast<uint8_t>(c - 'a' + 10);
		return true;
	}
	return false;
}

// Decode a lane name back to its path. Returns false — never throws — for any name that is not
// canonical, i.e. any name `JsonoEncodeLaneName` would not have produced: layout recognition runs
// this over arbitrary user structs and must classify them silently as "not JSONO" rather than fail
// the query. `steps` is assigned only on success.
inline bool JsonoTryDecodeLaneName(const string &name, vector<PathStep> &steps) {
	if (name.empty()) {
		return false;
	}
	string serialized;
	serialized.reserve(name.size() * 5 / 8);
	uint32_t buffer = 0;
	uint32_t bits = 0;
	for (char c : name) {
		uint8_t digit;
		if (!JsonoBase32HexDigit(c, digit)) {
			return false;
		}
		buffer = (buffer << 5) | digit;
		bits += 5;
		if (bits >= 8) {
			bits -= 8;
			serialized.push_back(static_cast<char>((buffer >> bits) & 0xFF));
		}
	}
	// One test covers both base32 rules: a name whose length is not a valid unpadded base32 length
	// leaves a whole character (>= 5 bits) decoding to nothing, and a name whose last character
	// carries non-zero unused bits leaves them set. Either way the bytes have a shorter spelling,
	// which is the one Encode writes, so this name is an alias and not canonical.
	if (bits >= 5 || (buffer & ((1u << bits) - 1)) != 0) {
		return false;
	}

	vector<PathStep> decoded;
	string key;
	idx_t i = 0;
	while (i < serialized.size()) {
		if (serialized[i] != '\0') {
			key.push_back(serialized[i]);
			i++;
			continue;
		}
		if (i + 1 >= serialized.size()) {
			return false;
		}
		auto partner = static_cast<unsigned char>(serialized[i + 1]);
		i += 2;
		if (partner == 0xFF) {
			key.push_back('\0');
			continue;
		}
		if (partner != 0x00) {
			return false;
		}
		decoded.push_back(PathStep {PathStepKind::Key, std::move(key), 0});
		key.clear();
	}
	if (!key.empty() || decoded.empty()) {
		return false;
	}
	steps = std::move(decoded);
	return true;
}

// Whether `name` is a name JsonoEncodeLaneName could have produced. The structural gate in layout
// recognition, which runs over arbitrary user structs and must classify them silently rather than
// fail the query. Canonicity subsumes the old object-key-path check: a decode yields nothing but a
// non-empty chain of Key steps.
inline bool JsonoLaneNameIsCanonical(const string &name) {
	vector<PathStep> steps;
	return JsonoTryDecodeLaneName(name, steps);
}

// Decode a lane's physical name into the logical path it lifts. Throws on a non-canonical name: no
// writer mints one and layout recognition refuses a type carrying one, so reaching this is a broken
// invariant rather than user input.
inline vector<PathStep> ShredNamePath(const string &name, const char *function_name) {
	vector<PathStep> steps;
	if (!JsonoTryDecodeLaneName(name, steps)) {
		throw InternalException("%s: shred lane name '%s' is not a canonical encoded path", function_name, name);
	}
	return steps;
}

// The lane's logical path in the project's one text form (`$.`-always, quoted per
// AppendJsonPathKey): what the per-row shred manifest stores and what every message naming a lane
// prints. The single producer of that text, so the manifest a writer emits and the signature a
// reader verifies against are the same bytes by construction.
inline string JsonoLaneLogicalPath(const string &name) {
	string path;
	if (!TryStepsToJsonPath(ShredNamePath(name, "jsono lane"), path)) {
		throw InternalException("jsono lane: path of '%s' cannot be serialized", name);
	}
	return path;
}

// True unless `read` and the object-key path `key_path` provably diverge — i.e. on some
// shared-depth step both are object keys that differ. A wildcard or index at a shared step is
// treated as a possible match (not provably disjoint), and one path being a prefix of the other
// also shares a branch. Used to decide, conservatively, whether a read could touch a shredded
// array path (read it, descend into it, or sit on a container subtree that holds it) — whose
// lifted element values are stripped from the residual and so demand a reconstruct.
inline bool PathStepsMayShareBranch(const vector<PathStep> &read, const vector<PathStep> &key_path) {
	idx_t shared = read.size() < key_path.size() ? read.size() : key_path.size();
	for (idx_t i = 0; i < shared; i++) {
		if (read[i].kind == PathStepKind::Key && key_path[i].kind == PathStepKind::Key &&
		    read[i].key != key_path[i].key) {
			return false;
		}
	}
	return true;
}

// True if a pure object-key path ends at / continues past `key` when matched at `depth`. The
// size check precedes the [depth] read, so a path shorter than depth+1 never indexes OOB. The
// path ref is std::vector so duckdb::vector<PathStep> binds too (derived-to-base). Shared by the
// array read overlay, the residual skeleton emit, and the LWW tree strip.
inline bool PathTerminatesOnKey(const std::vector<PathStep> &path, size_t depth, nonstd::string_view key) {
	return path.size() == depth + 1 && nonstd::string_view(path[depth].key.data(), path[depth].key.size()) == key;
}

inline bool PathContinuesPastKey(const std::vector<PathStep> &path, size_t depth, nonstd::string_view key) {
	return path.size() > depth + 1 && nonstd::string_view(path[depth].key.data(), path[depth].key.size()) == key;
}

// Multi-path variants: "does any active path terminate on `key`" and "collect the active paths
// that continue past `key`". Templated over the path-pointer container so both the std::vector
// skeleton side and the duckdb::vector LWW side share them; both delegate to the predicates above.
template <class Paths>
inline bool AnyPathTerminatesOnKey(const Paths &paths, size_t depth, nonstd::string_view key) {
	for (auto *path : paths) {
		if (PathTerminatesOnKey(*path, depth, key)) {
			return true;
		}
	}
	return false;
}

template <class Paths>
inline void CollectContinuingPaths(const Paths &paths, size_t depth, nonstd::string_view key, Paths &out) {
	out.clear();
	for (auto *path : paths) {
		if (PathContinuesPastKey(*path, depth, key)) {
			out.push_back(path);
		}
	}
}

} // namespace duckdb
