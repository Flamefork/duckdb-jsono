PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=jsono
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Default to the Ninja generator locally (faster configure + incremental rebuilds). CI already
# sets GEN=ninja in its environment, so ?= leaves CI untouched and only changes the local default.
GEN ?= ninja

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# extension-ci-tools ships the `relassert` build (RelWithDebInfo + FORCE_ASSERT=1) but no runner
# for its unittest binary; mirror test_release_internal against that build directory.
.PHONY: test_relassert
test_relassert:
	./build/relassert/$(TEST_PATH) "$(TESTS_BASE_DIRECTORY)*"

# Aggregate local pre-PR gate: build and run the SQLLogic suite twice — once against `release`,
# once against the assert-enabled `relassert` build — then check formatting. Order matters: each
# test target runs the binary the preceding build target just produced. The relassert leg is not
# redundant coverage: D_ASSERT is compiled out of `release`, so DuckDB's internal contracts
# (vector types, validity, string_t lifetimes) hold only there.
# (tidy-check is run separately; it needs the CI clang-tidy/compile-db setup.)
.PHONY: verify
verify: release test relassert test_relassert format-check
