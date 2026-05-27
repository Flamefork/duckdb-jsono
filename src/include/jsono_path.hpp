#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

#include <cctype>
#include <limits>

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
				if (!closed || key.empty()) {
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

} // namespace duckdb
