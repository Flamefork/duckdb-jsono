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
