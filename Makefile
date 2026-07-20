PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=jsono
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Default to the Ninja generator locally (faster configure + incremental rebuilds). CI already
# sets GEN=ninja in its environment, so ?= leaves CI untouched and only changes the local default.
GEN ?= ninja

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# extension-ci-tools ships the `relassert` build (RelWithDebInfo + FORCE_ASSERT=1) but no runner for
# its unittest binary; mirror test_release_internal against that build directory, behind the same
# SKIP_TESTS switch ci-tools puts in front of its own test targets.
TEST_RELASSERT_TARGET=test_relassert_internal
TEST_CONSTRUCTOR_MATRIX_RELASSERT_TARGET=test_constructor_matrix_relassert_internal

ifeq ($(SKIP_TESTS),1)
	TEST_RELASSERT_TARGET=tests_skipped
	TEST_CONSTRUCTOR_MATRIX_RELASSERT_TARGET=tests_skipped
endif

.PHONY: test_relassert test_relassert_internal test_constructor_matrix_relassert test_constructor_matrix_relassert_internal
test_relassert: $(TEST_RELASSERT_TARGET)

test_relassert_internal:
	./build/relassert/$(TEST_PATH) "$(TESTS_BASE_DIRECTORY)*"

test_constructor_matrix_relassert: $(TEST_CONSTRUCTOR_MATRIX_RELASSERT_TARGET)

test_constructor_matrix_relassert_internal:
	JSONO_DUCKDB_BIN=build/relassert/duckdb JSONO_EXTENSION=build/relassert/extension/jsono/jsono.duckdb_extension \
		uv run --frozen python -c "from test.property.jsono_property import test_struct_constructor_auto_shred_matrix, test_struct_constructor_auto_shred_parquet; test_struct_constructor_auto_shred_matrix(); test_struct_constructor_auto_shred_parquet()"

# Aggregate local pre-PR gate: build and run the SQLLogic suite twice — once against `release`, once
# against the assert-enabled `relassert` build — then run the deterministic constructor type/shape
# matrix against relassert and check formatting. The legs are recipe steps rather than prerequisites
# because make runs prerequisites in an unspecified order (and in parallel under -j), while each test
# leg must run the binary the build leg before it just produced. The relassert legs are not redundant
# coverage: D_ASSERT is compiled out of `release`, so DuckDB's internal contracts (vector types,
# validity, string_t lifetimes) are checked only there.
# (tidy-check is run separately; it needs the CI clang-tidy/compile-db setup.)
.PHONY: verify
verify:
	$(MAKE) release
	$(MAKE) test
	$(MAKE) relassert
	$(MAKE) test_relassert
	$(MAKE) test_constructor_matrix_relassert
	$(MAKE) format-check
