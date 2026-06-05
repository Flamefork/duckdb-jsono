# DuckDB JSONO Extension

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Main Extension Distribution Pipeline](https://github.com/Flamefork/duckdb-jsono/actions/workflows/MainDistributionPipeline.yml/badge.svg)](https://github.com/Flamefork/duckdb-jsono/actions/workflows/MainDistributionPipeline.yml)

Experimental DuckDB extension for JSON extraction and JSON-shaped intermediate storage built on top of [yyjson](https://github.com/ibireme/yyjson).

The extension has three main goals:

- spec-driven multi-field extraction from a JSON document into a typed `STRUCT` with `jsono_transform`;
- a pre-parsed `JSONO` representation for repeated reads over the same JSON column;
- path-shredded `JSONO` storage (`jsono_shred`) that lifts hot paths into typed lane columns for fast columnar reads while keeping `->>` and `to_json` working transparently.

**Warning:** this extension is experimental and maintained on a best-effort basis by a developer who doesn't write C++ professionally, so expect rough edges. It has not been hardened for production, and the `JSONO` binary format and SQL API are still being designed and may change between revisions. Validate behavior and performance on your own data before relying on it. Feedback and contributions are welcome via GitHub.

## Core Features

- `jsono(json)` parses JSON text or `JSON` into `JSONO`.
- `jsono(struct[, options])` constructs `JSONO` directly from typed DuckDB values.
- `try_jsono(json)` parses JSON text or `JSON` into `JSONO`, returning `NULL` for malformed input.
- `to_json(value)` serializes `JSONO` back to compact `JSON`.
- `jsono_extract(value, path)` / `value -> path` extracts one value as `JSONO`.
- `jsono_extract_string(value, path)` / `value ->> path` extracts one value as `VARCHAR`.
- `jsono_transform(value, spec)` extracts multiple fields into a typed `STRUCT`.
- `jsono_shred(value, spec)` rewrites `JSONO` into a shredded `STRUCT` whose hot paths are typed lane columns; `->>` and `to_json` stay transparent and read the lanes.
- `jsono_entries(value[, key_style])` flattens scalar leaves into `STRUCT(key VARCHAR, value VARCHAR)[]`.
- `jsono_group_merge(value, 'IGNORE NULLS' [ORDER BY ...])` aggregates JSON patches.
- `jsono_merge_patch(...)` applies RFC 7396-style merge patch to JSONO values.
- `jsono_overlay(...)` merges patches base-wins (fills only keys missing from the base).

## Quick Start

Build and load the extension first (see [Installation](#installation)), then:

```sql
SELECT jsono_transform(
    jsono('{"id": 42, "name": "duck", "active": true}'),
    '{"id":"BIGINT","name":"VARCHAR","active":"BOOLEAN"}'
);
-- {'id': 42, 'name': duck, 'active': true}
```

Convert to `JSONO` when the same JSON values will be read repeatedly:

```sql
CREATE TABLE events AS
SELECT jsono(payload) AS payload_jsono
FROM read_parquet('events.parquet');

SELECT (jsono_transform(payload_jsono, '{"user_id":"VARCHAR","event_ts":"BIGINT"}')).*
FROM events;
```

## Installation

> **Note:** This extension is not yet published in DuckDB's extension repository, and the `JSONO` binary format and SQL API are still being designed. Build it from source and load the produced local extension.

### CLI

```bash
uv run make release
./build/release/duckdb -unsigned
```

```sql
LOAD './build/release/extension/jsono/jsono.duckdb_extension';
```

### Python

```python
import duckdb

con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
con.execute("LOAD './build/release/extension/jsono/jsono.duckdb_extension'")
```

## Usage

### `jsono(json) -> JSONO`

Strictly parses JSON text or `JSON` into `JSONO`.

```sql
SELECT jsono('{"a":1,"b":"x"}');
```

Use SQL casts for the same strict parse behavior:

```sql
SELECT '{"a":1}'::JSONO;
SELECT CAST('{"a":1}' AS JSONO);
```

Malformed or empty input raises an error. Use `try_jsono` or `TRY_CAST` when malformed input should become `NULL`:

```sql
SELECT try_jsono('{not json') IS NULL;
SELECT TRY_CAST('{not json' AS JSONO) IS NULL;
```

`JSONO` physically stores as:

```text
STRUCT(jsono_slots BLOB, jsono_key_heap BLOB, jsono_string_heap BLOB, jsono_skips BLOB)
```

DuckDB storage can expose the physical struct shape after Parquet round-trips. The extension accepts that struct shape implicitly in `to_json`, `jsono_transform`, and JSONO functions.

The binary layout of each BLOB (slot encoding, tag table, string-heap cursor, and the `jsono_skips` navigation metadata) is specified in [docs/jsono_format.md](docs/jsono_format.md).

### `jsono(struct[, options]) -> JSONO`

Constructs `JSONO` directly from typed DuckDB values without serializing the whole input through JSON text first.

```sql
SELECT to_json(jsono({'b': 2, 'a': 'x', 'n': NULL, 'arr': ['c', NULL, 'd']}));
```
*Result:*
```json
{"a":"x","arr":["c",null,"d"],"b":2,"n":null}
```

`STRUCT` fields become JSON object keys, `LIST` values become arrays, `JSON` children are parsed as JSON subtrees, and `VARCHAR` children remain JSON strings. Exact integer and decimal values are preserved; non-finite `DOUBLE` values are rejected because JSON has no NaN/Infinity representation.

SQL `NULL` object fields are included as JSON nulls by default. Use a constant options struct to omit SQL `NULL` object fields:

```sql
SELECT to_json(jsono({'b': NULL, 'a': 1}, {'nulls': 'omit'}));
SELECT to_json(jsono({'b': NULL, 'a': 1}, {'omit_nulls': true}));
```
*Result:*
```json
{"a":1}
```

The `STRUCT -> JSONO` cast uses the same default constructor behavior:

```sql
SELECT to_json({'b': 2, 'a': 'x'}::JSONO);
```
*Result:*
```json
{"a":"x","b":2}
```

### `to_json(value) -> JSON`

Serializes `JSONO` back to compact `JSON`.

```sql
SELECT to_json(jsono('{"b":2,"a":1}'));
```
*Result:*
```json
{"a":1,"b":2}
```

Object keys are emitted in sorted byte order. Explicit casts are also available:

```sql
SELECT CAST(jsono('{"a":1}') AS JSON);
SELECT CAST(jsono('{"a":1}') AS VARCHAR);
```

The `JSON` cast and `to_json` produce DuckDB's core `JSON` logical type, which is provided by the bundled `json` extension that ships with every standard DuckDB distribution — no manual setup is needed.

### `jsono_extract(value, path) -> JSONO`

Extracts one value using a constant path. `value -> path` is an alias for the same operation. A bare string path names a literal top-level key, a string path starting with `$` uses JSONPath, and a `BIGINT` path indexes an array:

```sql
SELECT CAST(jsono_extract(jsono('{"a.b":"dot","a":{"b":"nested"}}'), 'a.b') AS VARCHAR);
-- "dot"

SELECT CAST(jsono('{"a.b":"dot","a":{"b":"nested"}}') -> '$.a' AS VARCHAR);
-- {"b":"nested"}

SELECT jsono('{"items":[{"name":"duck"}]}') -> 'items' -> 0 ->> 'name';
-- duck
```

JSON null at the selected path returns a `JSONO` null value. Missing paths return SQL `NULL`.

### `jsono_extract_string(value, path) -> VARCHAR`

Extracts one value using a constant path. `value ->> path` is an alias for the same operation. Path rules match `jsono_extract`:

```sql
SELECT jsono_extract_string(jsono('{"a.b":"dot","a":{"b":"nested"}}'), 'a.b');
-- dot

SELECT jsono('{"a.b":"dot","a":{"b":"nested"}}') ->> 'a.b';
-- dot

SELECT jsono_extract_string(jsono('{"a.b":"dot","a":{"b":"nested"}}'), '$.a.b');
-- nested
```

JSON strings return unquoted text, numbers and booleans return their JSON text, JSON null and missing paths return SQL `NULL`, and arrays/objects return compact JSON text.

### `jsono_transform(value, spec) -> struct`

Extracts fields from `JSONO` into a typed DuckDB `STRUCT`. `spec` must be a constant JSON object mapping each output field name to an extraction rule.

```sql
SELECT jsono_transform(
    jsono('{"a":42,"b":3.5,"c":"hi","d":true}'),
    '{"a":"BIGINT","b":"DOUBLE","c":"VARCHAR","d":"BOOLEAN"}'
);
```

A rule is either a type shorthand string or a wrapper object. Supported scalar types:

- `BIGINT`
- `UBIGINT`
- `DOUBLE`
- `VARCHAR`
- `BOOLEAN`

A shorthand reads the matching top-level key (path defaults to `$."field"`). A wrapper object reads from an explicit JSONPath:

```sql
SELECT jsono_transform(
    jsono('{"event":{"timestamp":1234567890}}'),
    '{"ts":{"type":"BIGINT","path":"$.event.timestamp"}}'
);
```

Paths support nested keys, array indices, quoted keys, and the `[*]` wildcard: `$.user.name`, `$.items[0].id`, `$."my-key"`, `$.tags[*]`.

Collect array elements into a `VARCHAR[]` with an array type, or join them into one string with `join_separator`:

```sql
SELECT jsono_transform(
    jsono('{"tags":["a","b","c"]}'),
    '{"tags":{"type":["VARCHAR"],"path":"$.tags[*]"}}'
);

SELECT jsono_transform(
    jsono('{"tags":["a","b","c"]}'),
    '{"tags":{"type":"VARCHAR","path":"$.tags[*]","join_separator":","}}'
);
```

The object keys `type`, `path`, and `join_separator` are reserved inside a wrapper object.

### `jsono_shred(value, spec) -> struct`

Rewrites a `JSONO` value into a *shredded* `STRUCT`: the four-BLOB `JSONO` residual followed by one typed lane column per path in `spec`. Reads stay transparent — `->>`, `jsono_extract_string`, `to_json`, and casts work on the shredded value exactly as on a plain `JSONO` column — but extracting a shredded path becomes a direct read of its typed lane instead of a per-row parse, which the planner can push down and prune on.

`spec` is a constant JSON object mapping each path to its lane type (same path grammar as `jsono_transform`, no wildcards; lane types `VARCHAR`, `BIGINT`, `UBIGINT`, `DOUBLE`, `BOOLEAN`):

```sql
CREATE TABLE events AS
SELECT jsono_shred(
    jsono(payload),
    '{"$.kind":"VARCHAR","$.did":"VARCHAR","$.time_us":"BIGINT"}'
) AS payload
FROM read_parquet('events.parquet');

-- queried like any JSONO column; the planner reads the lanes
SELECT payload ->> '$.kind' AS kind, count(*) AS n
FROM events GROUP BY kind;

SELECT max(CAST(payload ->> '$.time_us' AS BIGINT)) FROM events;
```

The conversion is lossless: `to_json(payload)` reconstructs the original document, and any path not in `spec` still reads from the residual. A value that does not fit its lane type (a string in a `BIGINT` lane, an explicit JSON `null`, a missing key) is kept in the residual unchanged, so reconstruction never loses or coerces data. Top-level shredded paths are stored once — in the lane rather than duplicated in the residual — so the shredded column is typically smaller than the plain `JSONO` column for the same data while answering hot-path queries from narrow typed columns.

```sql
SELECT to_json(jsono_shred(jsono('{"kind":"commit","time_us":1700,"extra":"e1"}'),
                           '{"$.kind":"VARCHAR","$.time_us":"BIGINT"}'));
-- {"extra":"e1","kind":"commit","time_us":1700}
```

Transparent reads over a shredded value go through the bundled `json` extension (present in every standard DuckDB distribution) and the extension's query optimizer. The shredded shape — four BLOBs plus the named lane columns — survives a plain Parquet round-trip and reads back transparently.

### `jsono_entries(value[, key_style]) -> STRUCT(key VARCHAR, value VARCHAR)[]`

Flattens scalar leaves of a `JSONO` document into a list of key/value structs. The default `key_style` is `'jsonpath'`; pass constant `'dotted'` for dot-separated keys:

```sql
SELECT unnest(jsono_entries(jsono('{"a":1,"b":{"c":"x"},"d":true}')));
-- {'key': $.a, 'value': 1}
-- {'key': $.b.c, 'value': x}
-- {'key': $.d, 'value': true}

SELECT unnest(jsono_entries(jsono('{"a":1,"b":{"c":"x"}}'), 'dotted'));
-- {'key': a, 'value': 1}
-- {'key': b.c, 'value': x}
```

JSON null leaves keep their key and return SQL `NULL` as `value`; empty objects and arrays do not produce entries. Dotted output can collide when a literal dotted key and a nested path render to the same string, for example `"a.b"` and `{"a":{"b":...}}`.

### `jsono_group_merge(value, 'IGNORE NULLS' [ORDER BY ...]) -> JSONO`

Aggregates a stream of JSON object patches with [RFC 7396](https://datatracker.ietf.org/doc/html/rfc7396) merge semantics: later patches in the ordered stream overwrite earlier keys and arrays replace wholesale. The current implementation requires the constant mode `'IGNORE NULLS'`, which drops `null` object members before merging so existing keys stay untouched. Provide an `ORDER BY` clause to make the fold deterministic.

```sql
WITH patches(patch, ts) AS (
    VALUES
        (jsono('{"a":1,"b":{"x":1}}'), 1),
        (jsono('{"a":null,"b":{"y":2}}'), 2),
        (jsono('{"c":3}'), 3)
)
SELECT to_json(jsono_group_merge(patch, 'IGNORE NULLS' ORDER BY ts))
FROM patches;
```
*Result:*
```json
{"a":1,"b":{"x":1,"y":2},"c":3}
```

Use `GROUP BY` for grouped state accumulation:

```sql
SELECT session_id,
       to_json(jsono_group_merge(patch, 'IGNORE NULLS' ORDER BY event_ts)) AS state
FROM session_patch_events
GROUP BY session_id;
```

Use it as a window function to materialize the running state after each patch:

```sql
SELECT id,
       to_json(jsono_group_merge(patch, 'IGNORE NULLS' ORDER BY id)
           OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS state
FROM (
    VALUES
        (1, jsono('{"theme":"dark"}')),
        (2, jsono('{"lang":"en"}')),
        (3, jsono('{"theme":"light"}'))
) AS t(id, patch)
ORDER BY id;
```
*Result:*
```text
1  {"theme":"dark"}
2  {"lang":"en","theme":"dark"}
3  {"lang":"en","theme":"light"}
```

### `jsono_merge_patch(target, patch[, ...]) -> JSONO`

Applies one or more [RFC 7396](https://datatracker.ietf.org/doc/html/rfc7396) merge patches left to right. The function is variadic and requires at least one argument; each `null` object member in a patch deletes that key from the result.

```sql
SELECT to_json(jsono_merge_patch(
    jsono('{"a":1,"b":2}'),
    jsono('{"b":null,"c":3}')
));
```
*Result:*
```json
{"a":1,"c":3}
```

Pass additional patches to fold them in sequence:

```sql
SELECT to_json(jsono_merge_patch(
    jsono('{"a":1}'),
    jsono('{"a":2,"b":2}'),
    jsono('{"c":3}')
));
```
*Result:*
```json
{"a":2,"b":2,"c":3}
```

Unlike `jsono_group_merge`, this is a scalar function over a fixed set of patch arguments rather than an aggregate over rows.

### `jsono_overlay(base, patch[, ...]) -> JSONO`

Overlays one or more patches onto a base value, base-wins: a patch only fills object keys absent from the base, a `null` patch member is a no-op (it does not delete), and the base's own values — including explicit nulls — are kept. This is the complement of `jsono_merge_patch`, where the patch wins and a `null` member deletes.

```sql
SELECT to_json(jsono_overlay(
    jsono('{"a":1,"b":2}'),
    jsono('{"b":99,"c":3}')
));
```
*Result:*
```json
{"a":1,"b":2,"c":3}
```

### Introspection

```sql
jsono_type(value[, path])     -> VARCHAR    -- OBJECT/ARRAY/VARCHAR/BIGINT/UBIGINT/DOUBLE/BOOLEAN/NULL
jsono_keys(value[, path])     -> VARCHAR[]  -- object keys
jsono_validate(value)         -> BOOLEAN    -- strict current-format validation
jsono_storage_size(value)     -> STRUCT     -- physical four-BLOB byte sizes
jsono_storage_type()          -> VARCHAR    -- DDL of the physical STRUCT backing JSONO
```

`jsono_type` and `jsono_keys` inspect the shape of unknown JSON before extracting it with `jsono_transform`. The optional `path` is a constant JSONPath (same grammar as `jsono_transform`, without wildcards) that points at a nested position; a missing path yields `NULL`.

```sql
SELECT jsono_keys(jsono('{"user":{"name":"a","age":1}}'), '$.user');
-- [age, name]
SELECT jsono_type(jsono('{"items":[1,2,3]}'), '$.items[0]');
-- BIGINT
```

`jsono_validate` checks the current binary format contract: header flags, sorted unique keys, slots, heaps, navigation metadata, UTF-8 strings, and raw number text. It returns `false` for malformed physical blobs and `NULL` for SQL `NULL`.

`jsono_storage_size` reports `jsono_slots`, `jsono_key_heap`, `jsono_string_heap`, `jsono_skips`, and `total` byte counts. It is a physical diagnostic and does not validate the value.

`jsono_storage_type` returns the DDL of the physical `STRUCT` backing `JSONO` with the alias dropped — the form stores that reject user-defined type aliases (DuckLake, plain Parquet) actually keep. Use it to declare storage columns without hardcoding the field names; a column of that exact shape binds to `jsono` ops and `to_json` without an explicit `::JSONO` cast.

```sql
SELECT jsono_storage_type();
-- STRUCT(jsono_slots BLOB, jsono_key_heap BLOB, jsono_string_heap BLOB, jsono_skips BLOB)
```

These helpers are primarily used by the JSONO workflows and tests. Treat their exact surface as less stable than `jsono_transform`, `jsono`, and `to_json`.

## Benchmarks

`bench/` is the comparable harness for `jsono` versions and the core DuckDB `json` baseline. Run a smoke benchmark:

```bash
uv run --frozen python bench/bench.py --filter group_merge/1k --runs 1
```

The operation set, version/core-json comparisons, field-sample scenarios, and result-reading contract are documented in [bench/README.md](bench/README.md); profiling is in [bench/PROFILING.md](bench/PROFILING.md).

## Development

```bash
uv run make release
uv run make test
uv run --frozen black --check bench
```

Useful commands:

```bash
uv run make debug
uv run make reldebug
uv run make clean
uv run --frozen python bench/run_benchmarks.py --list
uv run --frozen python bench/compare_results.py --save-baseline
uv run --frozen python bench/compare_results.py
```

The DuckDB submodule lives in `duckdb/`. Update it only via explicit submodule commands.

## Configuration

`JSONO_SHAPE_CACHE_SIZE` controls the JSONO writer shape cache size:

```bash
JSONO_SHAPE_CACHE_SIZE=16384 ./build/release/duckdb -unsigned
```

Use power-of-two values when comparing performance. The default is tuned for local benchmark workloads, but real data can differ.

## Error Handling and Caveats

- `jsono_transform` requires a constant spec string.
- `jsono_shred` requires a constant spec string and rejects wildcard paths; only top-level paths are physically stripped from the residual (nested shredded paths are read-accelerated but kept in the residual). Transparent `->>`/`to_json` over a shredded value need the query optimizer; reading one with the optimizer fully disabled (`PRAGMA disable_optimizer`) is not supported.
- `jsono_extract` / `->` / `jsono_extract_string` / `->>` require a constant path, do not support wildcard list extraction yet, and do not support negative array indexes.
- Unsupported scalar types in `jsono_transform` fail at bind time.
- `jsono_group_merge` currently requires constant `'IGNORE NULLS'`.
- JSON nested deeper than 1000 levels is rejected with an error (the limit is lowered to 50 under sanitizer builds).
- The binary JSONO format is strict-versioned. Incompatible storage changes must bump `jsono::VERSION`; validation rejects mismatched versions, while extraction/serialization helpers treat invalid headers as SQL `NULL`.
- Object keys inside `JSONO` output are sorted, so serialized JSON text may not preserve input object key order.
- Malformed input raises DuckDB errors unless `try_jsono` or `TRY_CAST(... AS JSONO)` is used.

## Known limitations

The current surface is intentionally small. The following are not implemented:

- No cast from `JSONO` to a scalar type (`jsono('42')::INTEGER` is unsupported). Project string values out with `->>` / `jsono_extract_string` or typed fields with `jsono_transform`.
- No cast from `JSONO` to an arbitrary `STRUCT`. Only the physical four-BLOB shape is interchangeable with `JSONO`; field extraction goes through `jsono_transform`.
- No dedicated table function that unnests a `JSONO` array or object into rows. Use `unnest(jsono_entries(...))` for flattened scalar leaves.
- Object key order is not preserved: keys are stored and emitted in sorted byte order (see [Error Handling and Caveats](#error-handling-and-caveats)).

## Contributing

Contributions and feedback are welcome. Please:

1. Open an issue first to discuss proposed changes.
2. Add or update SQLLogic tests in `test/sql/` for new behavior.
3. Run `uv run make release && uv run make test` before submitting a pull request.

See [GitHub Issues](https://github.com/Flamefork/duckdb-jsono/issues) for current tasks and feature requests.

## License

MIT. See [LICENSE](LICENSE).

For third-party components and their licenses, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---
*This extension is based on the [DuckDB Extension Template](https://github.com/duckdb/extension-template).*
