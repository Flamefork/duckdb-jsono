# Tests

`test/sql/` holds [SQLLogicTests](https://duckdb.org/dev/sqllogictest/intro.html) —
the primary test format for this extension. One `.test` file per feature or
related cluster of behavior.

Run them from the repo root:

```bash
uv run make test          # release build
uv run make test_debug    # debug build
```

Each SQLLogic file must start with `# group: [sql]` and `require jsono`
(add `require parquet` only for Parquet-specific cases). Keep output
deterministic — sort results or use fixed inputs.

When changing `JSONO` storage or casts, cover bind-time errors, runtime
invalid-input behavior, NULL behavior, and Parquet round-trips. For bug fixes,
add or update a focused SQLLogic case before changing the implementation.

## Python guards (non-SQLLogic)

Some invariants are invisible to SQLLogic and run as standalone `uv` scripts
against the built CLI (`JSONO_DUCKDB_BIN` / `JSONO_EXTENSION` override the binary
and extension paths; both default to the release build):

```bash
uv run --frozen test/property/jsono_property.py   # hypothesis property/fuzz harness
uv run --frozen test/pruning/jsono_pruning.py     # row-group pruning regression guard
```

`test/pruning/jsono_pruning.py` writes a clustered shredded Parquet and asserts,
via the DuckDB row-group-scanned metrics, that a selective filter on a shred leaf
prunes row groups exactly like the native control column does — so a format or
optimizer change that silently breaks shred filter pushdown fails loudly. Both
guards run under sanitizers in CI (`.github/workflows/Sanitizer.yml`).
