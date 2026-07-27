#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

#include "string_view.hpp"

#include <cctype>
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

// ===== The lane-name boundary =====
//
// A shred lane has two names: the LOGICAL path it lifts out of the document (a pure object-key
// chain — what reconstruct, render and jsono_entries emit, and what a diagnostic prints) and the
// PHYSICAL STRUCT field it occupies inside `shreds` (what the type carries, what the manifest and
// the canonical spill ranks are keyed by). Today they are one string: the physical name IS the path
// spelled as text. That conflation is the defect plan 056 fixes — it is why two spellings of one
// path (`gclid` and `$.gclid`) mint two lanes, and why DuckDB's case-insensitive STRUCT field
// matching collapses `gclid` and `GCLID` into one. Everything that crosses between the two names
// goes through the two functions below; nothing else may split a lane name on `$.` or hand a
// physical name to a path parser.

// Whether a lane name is spelled in the `$.`-rooted JSONPath form rather than as a bare literal
// top-level key. Only the `$.` prefix marks it: a literal top-level key may itself begin with `$`
// (e.g. `$x`), and the reserved `$jsono$set` marker begins with `$` too, so a bare-`$` test would
// misread both.
inline bool LaneNameIsPathForm(const string &name) {
	return name.size() >= 2 && name[0] == '$' && name[1] == '.';
}

// Decode a lane's physical name into the logical path it lifts. Throws on a name no writer could
// have minted (a malformed `$.`-rooted path); every recognizable lane decodes, because layout
// recognition already gated the name through ShredNameIsObjectKeyPath.
inline vector<PathStep> ShredNamePath(const string &name, const char *function_name) {
	if (LaneNameIsPathForm(name)) {
		return ParseJsonoPath(name, function_name);
	}
	return LiteralKeyPath(name);
}

// The non-throwing form of the decode above, plus the lane-path check: whether `name` decodes to a
// path a shred lane may carry. It is the structural gate in layout recognition, which runs over
// arbitrary user structs and must classify them silently rather than fail the query.
inline bool ShredNameIsObjectKeyPath(const string &name) {
	vector<PathStep> steps;
	try {
		steps = ShredNamePath(name, "jsono shred");
	} catch (const std::exception &) {
		return false;
	}
	return IsObjectKeyPath(steps);
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
