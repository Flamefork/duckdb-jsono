# This file is included by DuckDB's build system. It specifies which extension to load

# Extension from this repo
duckdb_extension_load(jsono
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Any extra extensions that should be built
duckdb_extension_load(parquet)
duckdb_extension_load(json)
# icu: the TIMESTAMPTZ tests need a session TimeZone, which is what ICU registers
duckdb_extension_load(icu)
