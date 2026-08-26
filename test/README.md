# Tests

`test/sql/` holds [SQLLogicTests](https://duckdb.org/dev/sqllogictest/intro.html) —
the primary test format for this extension. One `.test` file per feature or
related cluster of behavior.

Run them from the repo root:

```bash
uv run --frozen make release
uv run --frozen make test

uv run --frozen make debug
uv run --frozen make test_debug
```

Each SQLLogic file must start with `# group: [sql]` and `require jsono`
(add `require parquet` only for Parquet-specific cases). Keep output
deterministic — sort results or use fixed inputs.

When changing `JSONO` storage or casts, cover bind-time errors, runtime
invalid-input behavior, NULL behavior, and Parquet round-trips. For bug fixes,
add or update a focused SQLLogic case before changing the implementation.

### Spelling the storage type

Two non-obvious rules govern how the `jsono` storage type is spelled in tests:

1. `jsono_storage_type()` returns a `VARCHAR`, so it cannot follow `::` (you
   cannot write `x::jsono_storage_type()`). When you need the raw 6-blob layout
   as a cast *target*, spell it out as the `::STRUCT(jsono STRUCT("body$2"
   STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB,
   nums BLOB)))` literal (the `$<N>` suffix is the layout revision — see
   `docs/jsono_format.md` → Layout revisions).
2. The storage-path anchor checks (`jsono_layout_golden.test`,
   `jsono_storage_type.test`, `jsono_roundtrip.test`, `jsono_parquet.test`)
   spell that literal raw *on purpose* — they are the anchor that pins the
   physical layout, so rewriting them to `= jsono_storage_type()` would make the
   assertion circular (the function reading from the same source it is meant to
   verify). Everywhere else prefer `typeof(x) = jsono_storage_type(spec)`.
3. `jsono_layout_golden.test` additionally pins the body BYTES, which is the
   only tripwire for a semantic layout change the type cannot show (spill bit
   numbering, marker value). When it goes red on purpose, follow the
   revision-close checklist in `docs/jsono_format.md` → Layout revisions;
   `jsono_layout_revision.test` is where the closed revision's fixture lands.

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
