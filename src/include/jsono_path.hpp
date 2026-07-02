#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

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

// Resolve a shred lane name to its path steps. A lane name is either a bare literal top-level object
// key or a `$.`-rooted JSONPath naming a nested key. Only the `$.` prefix marks the JSONPath form:
// a literal top-level key may itself begin with `$` (e.g. `$x`), and the reserved `$jsono$set` marker
// begins with `$` too, so a bare-`$` test would misparse both. Every reader that turns a shred name
// into path steps must agree on this split, so they share this one classifier.
inline vector<PathStep> ShredNamePath(const string &name, const char *function_name) {
	if (name.size() >= 2 && name[0] == '$' && name[1] == '.') {
		return ParseJsonoPath(name, function_name);
	}
	return LiteralKeyPath(name);
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

} // namespace duckdb
