PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=jsono
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Default to the Ninja generator locally (faster configure + incremental rebuilds). CI already
# sets GEN=ninja in its environment, so ?= leaves CI untouched and only changes the local default.
GEN ?= ninja

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
