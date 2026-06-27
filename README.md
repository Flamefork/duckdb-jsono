# DuckDB JSONO

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Main Extension Distribution Pipeline](https://github.com/Flamefork/duckdb-jsono/actions/workflows/MainDistributionPipeline.yml/badge.svg)](https://github.com/Flamefork/duckdb-jsono/actions/workflows/MainDistributionPipeline.yml)

A DuckDB extension for analytics-optimized JSON storage and querying.

The extension has three main goals:

- spec-driven multi-field extraction from a JSON document into a typed `STRUCT` with `jsono_transform`;
- a pre-parsed JSONO representation for repeated reads over the same JSON column;
- path-shredded JSONO storage (`jsono(value, shredding := spec)`) that lifts hot paths into typed shred columns for fast columnar reads while keeping `->>` and `to_json` working transparently.

> [!WARNING]
> This extension is experimental and maintained on a best-effort basis by a developer who doesn't write C++ professionally, so expect rough edges. It has not been hardened for production, and the JSONO binary format and SQL API are still being designed and may change between revisions. Validate behavior and performance on your own data before relying on it. Feedback and contributions are welcome via GitHub.

## Functions

Build and convert:

- [`jsono(json)`](#jsonojson) / [`try_jsono(json)`](#jsonojson) — parse JSON text or `JSON` into JSONO (`try_` returns `NULL` on malformed input instead of raising).
- [`jsono(struct)`](#jsonostruct) — construct JSONO directly from typed DuckDB values, without serializing through JSON text first.
- [`to_json(value)`](#to_json) — serialize JSONO back to compact `JSON` (also available as a `CAST`).

Extract and project:

- [`jsono_extract(value, path)` / `value -> path`](#jsono_extract) — extract one value as JSONO.
- [`jsono_extract_string(value, path)` / `value ->> path`](#jsono_extract_string) — extract one value as `VARCHAR`.
- [`jsono_transform(value, spec[, on_type_mismatch])`](#jsono_transform) — project many fields into a typed `STRUCT` in one pass (`spec` is a `STRUCT` literal; type mismatches default to `convert`).
- [`jsono_entries(value[, key_style])`](#jsono_entries) — flatten scalar leaves into `STRUCT(key VARCHAR, value VARCHAR)[]`.

Merge and aggregate:

- [`jsono_merge_patch(target, patch[, ...])`](#jsono_merge_patch) — RFC 7396 merge patch (later patches win; a `null` member deletes the key).
- [`jsono_group_merge(value [ORDER BY ...])`](#jsono_group_merge) — aggregate a stream of patches into one value.
- [`jsono_group_merge_max(value, order_key)` / `jsono_group_merge_min(value, order_key)`](#jsono_group_merge_max--jsono_group_merge_min) — order-independent keyed merge: per leaf, the value from the row with the greatest (`_max`) or smallest (`_min`) `order_key` wins, without buffering the input.
- [`jsono_diff(prev, cur[, arrays])`](#jsono_diff) — structural diff of `prev → cur`; `arrays` selects how array changes render (`atomic` reverse merge-patch, `counts`, or `elements`).

Shredded storage:

- [`jsono(value, shredding := spec)`](#jsonovalue-shredding) — build a *shredded* `STRUCT` from JSON text or an existing JSONO, lifting hot paths into typed shred columns; `->>` and `to_json` stay transparent and read the shreds.

Inspect:

- [`jsono_type(value[, path])`](#introspection) — the value's JSON type as `VARCHAR` (`OBJECT`, `ARRAY`, a scalar type, or `NULL`).
- [`jsono_keys(value[, path])`](#introspection) — object keys as `VARCHAR[]`.
- [`jsono_validate(value)`](#introspection) — strict current-format validation as `BOOLEAN`.
- [`jsono_storage_size(value)`](#introspection) — physical byte sizes (body blobs, shreds, total) as a `STRUCT`.
- [`jsono_storage_type()`](#introspection) — DDL of the physical `STRUCT` backing a JSONO value.
- [`jsono_shred_manifest(value)`](#introspection) — paths stripped into shred columns and their types as a `LIST<STRUCT(path, type)>`.

## Quick Start

> [!NOTE]
> JSONO is a format, not a SQL type. A JSONO value is this extension's pre-parsed binary representation of a JSON document — physically a nested `STRUCT(jsono STRUCT(body STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB, nums BLOB)))`, with no registered type name. There is no `JSONO` to name in a `CAST` or a column definition. Instead: build a value with `jsono(...)` / `try_jsono(...)`, declare storage columns with `jsono_storage_type()`, and let the functions, the `->` / `->>` operators, and `to_json` recognise that `STRUCT` shape structurally — a value read back from Parquet or DuckLake works with no cast. `.body` (and, on a shredded value, the shred fields in its sibling `shreds` struct) are physical storage details, not a public access API.

Build and load the extension first (see [Installation](#installation)), then:

```sql
SELECT jsono_transform(
    jsono('{"id": 42, "name": "duck", "active": true}'),
    {id: 'BIGINT', name: 'VARCHAR', active: 'BOOLEAN'}
);
-- {'id': 42, 'name': duck, 'active': true}
```

Convert to JSONO when the same JSON values will be read repeatedly:

```sql
CREATE TABLE events AS
SELECT jsono(payload) AS payload_jsono
FROM read_parquet('events.parquet');

SELECT (jsono_transform(payload_jsono, {user_id: 'VARCHAR', event_ts: 'BIGINT'})).*
FROM events;
```

## Installation

> **Note:** This extension is not yet published in DuckDB's extension repository, and the JSONO binary format and SQL API are still being designed. Build it from source and load the produced local extension.

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

### `jsono(json)`

Strictly parses JSON text or `JSON` into JSONO.

```sql
SELECT jsono('{"a":1,"b":"x"}');
```

Malformed or empty input raises an error. Use `try_jsono` when malformed input should become `NULL`:

```sql
SELECT try_jsono('{not json') IS NULL;
```

To parse and shred hot paths into typed shred columns in one pass, supply a constant `shredding := {...}` spec — the same option documented under the shredding constructor below:

```sql
SELECT jsono('{"kind":"commit","time_us":1700}', shredding := {'$.kind': 'VARCHAR', '$.time_us': 'BIGINT'});
```

A JSONO value is physically this nested `STRUCT` — one `jsono` layout field wrapping the six body BLOBs every storage layer (DuckDB-native, Parquet, DuckLake) sees:

```text
STRUCT(jsono STRUCT(body STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB, nums BLOB)))
```

The JSONO functions (`to_json`, `jsono_transform`, …) recognise this STRUCT structurally, so a value read back from Parquet or DuckLake binds to them with no cast. `jsono_storage_type()` returns the DDL string above for declaring storage columns.

The binary layout of each body BLOB (slot encoding, tag table, string/key heaps, the `skips` navigation metadata, and the `lengths`/`nums` side columns) is specified in [jsono_format.md](docs/jsono_format.md).

### `jsono(struct)`

Constructs JSONO directly from typed DuckDB values without serializing the whole input through JSON text first.

```sql
SELECT to_json(jsono({'b': 2, 'a': 'x', 'n': NULL, 'arr': ['c', NULL, 'd']}));
```
*Result:*
```json
{"a":"x","arr":["c",null,"d"],"b":2,"n":null}
```

`STRUCT` fields become JSON object keys, `LIST` values become arrays, `JSON` children are parsed as JSON subtrees, and `VARCHAR` children remain JSON strings. Exact integer and decimal values are preserved; non-finite `DOUBLE` values are rejected because JSON has no NaN/Infinity representation. SQL `NULL` object fields are serialized as JSON nulls.

Because the input `STRUCT` already carries its field names and types, `jsono(struct)` shreds automatically — there is no `shredding` argument to pass. Every top-level field whose type is a scalar shred type (`BIGINT`, `UBIGINT`, `DOUBLE`, `BOOLEAN`, or `VARCHAR`) is lifted into a typed shred column named by the field, exactly as the [`shredding`](#jsonovalue-shredding) constructor does for text. Top-level `LIST<TYPE>` fields also become scalar-array shreds when `TYPE` is one of those scalar shred types, and top-level `LIST<STRUCT<...>>` fields become array shreds when the struct children are supported scalar shred types. One nested `STRUCT` level is eligible too, with shred leaves named by JSONPath-style paths such as `$.parent.child`. Unsupported lists, narrower integers like `INTEGER`, `DECIMAL`, `NULL`, `JSON` subtrees, and deeper or unsupported nested structures stay in the residual; cast a field to its shred type (`x::BIGINT`, `x::DOUBLE`) to lift it.

```sql
SELECT typeof(jsono({'kind': 'commit', 'time_us': 1700::BIGINT}));
-- a shredded STRUCT: a `jsono` layout wrapping the body blobs plus shreds "kind" VARCHAR and "time_us" BIGINT
```

### `to_json`

Serializes JSONO back to compact `JSON`.

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

### `jsono_extract`

Extracts one value using a constant path. `value -> path` is an alias for the same operation. A bare string path names a literal top-level key, a string path starting with `$` uses JSONPath, and a `BIGINT` path indexes an array:

```sql
SELECT CAST(jsono_extract(jsono('{"a.b":"dot","a":{"b":"nested"}}'), 'a.b') AS VARCHAR);
-- "dot"

SELECT CAST(jsono('{"a.b":"dot","a":{"b":"nested"}}') -> '$.a' AS VARCHAR);
-- {"b":"nested"}

SELECT jsono('{"items":[{"name":"duck"}]}') -> 'items' -> 0 ->> 'name';
-- duck
```

JSON null at the selected path returns a JSONO null value. Missing paths return SQL `NULL`.

### `jsono_extract_string`

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

> **Operator precedence.** As in core `json`, the `->` and `->>` operators bind *looser* than `AND`/`OR` — the arrow token is shared with [lambda syntax](https://duckdb.org/docs/stable/sql/functions/lambda.html), whose body must bind loosely. Inside a boolean conjunction you must therefore parenthesize the extraction: write `WHERE cond AND (e ->> '$.k') = 'v'`, not `WHERE cond AND e ->> '$.k' = 'v'` — the latter parses as `(cond AND e) ->> …` and fails trying to cast the JSONO struct to `BOOLEAN`. The named `jsono_extract` / `jsono_extract_string` functions need no parentheses. This is expected DuckDB behavior ([duckdb/duckdb#16970](https://github.com/duckdb/duckdb/issues/16970)).

### `jsono_transform`

Extracts fields from JSONO into a typed DuckDB `STRUCT`. `spec` must be a constant `STRUCT` mapping each output field name to an extraction rule.

```sql
SELECT jsono_transform(
    jsono('{"a":42,"b":3.5,"c":"hi","d":true}'),
    {a: 'BIGINT', b: 'DOUBLE', c: 'VARCHAR', d: 'BOOLEAN'}
);
```

A rule is either a type shorthand string or a wrapper `STRUCT`. Supported scalar types:

- `BIGINT`
- `UBIGINT`
- `DOUBLE`
- `VARCHAR`
- `BOOLEAN`

A shorthand reads the matching top-level key (path defaults to `$."field"`). A wrapper `STRUCT` reads from an explicit JSONPath:

```sql
SELECT jsono_transform(
    jsono('{"event":{"timestamp":1234567890}}'),
    {ts: {type: 'BIGINT', path: '$.event.timestamp'}}
);
```

Paths support nested keys, array indices, quoted keys, and the `[*]` wildcard: `$.user.name`, `$.items[0].id`, `$."my-key"`, `$.tags[*]`.

Type mismatch handling is controlled by the constant `on_type_mismatch` argument:

- `convert` (default): CAST-like scalar coercion to the requested type. Values that cannot be coerced become `NULL`.
- `null`: keep the legacy silent-NULL behavior. A scalar value is returned only when its stored JSON type already matches the requested projection type.
- `fail`: use the same scalar coercions as `convert`, but raise an error when a present value cannot be coerced.

Missing paths and explicit JSON `null` return SQL `NULL` in every mode. `on_type_mismatch` applies independently to each projected field and each wildcard list element.

```sql
SELECT jsono_transform(jsono({'a':'12345'}), {'a':'UBIGINT'});
-- {'a': 12345}

SELECT jsono_transform(jsono({'a':'12345'}), {'a':'UBIGINT'}, on_type_mismatch := 'null');
-- {'a': NULL}

SELECT jsono_transform(jsono({'a':'abc'}), {'a':'UBIGINT'}, on_type_mismatch := 'fail');
-- error: cannot convert value at $.a from VARCHAR to UBIGINT
```

Collect array elements into a typed list with a single-element list type, or join them into one string with `join_separator`:

```sql
SELECT jsono_transform(
    jsono('{"values":["1",2,true,"bad"]}'),
    {values: {type: ['UBIGINT'], path: '$.values[*]'}}
);
-- {'values': [1, 2, 1, NULL]}

SELECT jsono_transform(
    jsono('{"tags":["a","b","c"]}'),
    {tags: {type: 'VARCHAR', path: '$.tags[*]', join_separator: ','}}
);
```

The field names `type`, `path`, and `join_separator` are reserved inside a wrapper `STRUCT`.

### `jsono(value, shredding)`

The `shredding` named argument turns `jsono()` into a *shredding* constructor: it produces a *shredded* `STRUCT` — the six-BLOB JSONO residual plus a `shreds` struct carrying one typed shred per path in `spec`. The primary form takes JSON **text** and parses + shreds in one pass; passing an existing **JSONO** value shreds it directly. Those are the two accepted `value` inputs — there is no `jsono(struct, shredding := …)` overload, because a `STRUCT` built with [`jsono(struct)`](#jsonostruct) is already shredded by that constructor (so to shred a struct-built value, build it with `jsono({…})` directly rather than wrapping it). Reads stay transparent — `->>`, `jsono_extract_string`, `to_json`, and casts work on the shredded value exactly as on a plain JSONO column — but extracting a shredded path becomes a direct read of its typed shred instead of a per-row parse, which the planner can push down and prune on.

`spec` is a constant `STRUCT` mapping each path to its shred type string (the same shape as Parquet's `SHREDDING` option and `read_json`'s `columns`; same path grammar as `jsono_transform`, no wildcards). Scalar shreds use `VARCHAR`, `BIGINT`, `UBIGINT`, `DOUBLE`, or `BOOLEAN`; scalar-array shreds use `VARCHAR[]`, `BIGINT[]`, `UBIGINT[]`, `DOUBLE[]`, or `BOOLEAN[]`; object-array shreds use `STRUCT(<supported scalar children>)[]`:

```sql
CREATE TABLE events AS
SELECT jsono(payload, shredding := {'$.kind': 'VARCHAR', '$.did': 'VARCHAR', '$.time_us': 'BIGINT'}) AS payload
FROM read_parquet('events.parquet');

-- queried like any JSONO column; the planner reads the shreds
SELECT payload ->> '$.kind' AS kind, count(*) AS n
FROM events GROUP BY kind;

SELECT max(CAST(payload ->> '$.time_us' AS BIGINT)) FROM events;
```

Array-shred specs name the array's object-key path, not an indexed or wildcard element path. Elements or subfields that do not fit the requested shred type stay in the residual skeleton, so reconstruction never loses or coerces data.

```sql
SELECT to_json(jsono('{"products":[{"id":1,"name":"a"}]}',
                     shredding := {'$.products': 'STRUCT(id UBIGINT, name VARCHAR)[]'}));
-- {"products":[{"id":1,"name":"a"}]}
```

The conversion is lossless: `to_json(payload)` reconstructs the original document, and any path not in `spec` still reads from the residual. A value that does not fit its shred type (a string in a `BIGINT` shred, an explicit JSON `null`, a missing key) is kept in the residual unchanged, so reconstruction never loses or coerces data. Top-level shredded paths are stored once — in the shred rather than duplicated in the residual — so the shredded column is typically smaller than the plain JSONO column for the same data while answering hot-path queries from narrow typed columns.

```sql
SELECT to_json(jsono('{"kind":"commit","time_us":1700,"extra":"e1"}',
                     shredding := {'$.kind': 'VARCHAR', '$.time_us': 'BIGINT'}));
-- {"extra":"e1","kind":"commit","time_us":1700}
```

Transparent reads over a shredded value go through the bundled `json` extension (present in every standard DuckDB distribution) and the extension's query optimizer. The shredded shape — the `jsono` layout wrapping the `body` blobs and the named `shreds` columns — survives a plain Parquet round-trip and reads back transparently; the scalar shred leaves keep projection and filter pushdown.

Two things decide whether a filter actually **prunes** Parquet row groups (skips them on the per-row-group min/max statistics) rather than scanning the whole file:

- **Filter a typed shred in its own type.** A `VARCHAR` shred prunes on the plain `payload ->> '$.kind' = 'commit'`. A numeric/boolean shred prunes only when the comparison is in the shred's type — `CAST(payload ->> '$.time_us' AS BIGINT) = 1700`, not the string form `payload ->> '$.time_us' = '1700'`. `->>` is textual, so the string form compares the shred *rendered back to text* (a computed expression the scan cannot push down) and reads every row group. Compare with `CAST(... AS <shred type>)` (or project the typed value with `jsono_transform`) to get pruning on a typed shred.
- **Cluster the rows on the leaf at write time.** Pruning needs each row group to hold a disjoint range of the filtered path, so add `ORDER BY` on that path to the `CREATE TABLE … AS SELECT` (or the `COPY`). Without clustering every row group spans the whole domain and none can be skipped. Clustering also improves compression, so one `ORDER BY` on the dominant filter path pays off twice.

```sql
CREATE TABLE events AS
SELECT jsono(payload, shredding := {'$.kind': 'VARCHAR', '$.time_us': 'BIGINT'}) AS payload
FROM read_parquet('events.parquet')
ORDER BY CAST(payload ->> '$.time_us' AS BIGINT);   -- cluster on the hot filter path

-- prunes to the matching row groups (typed shred compared as BIGINT)
SELECT count(*) FROM events WHERE CAST(payload ->> '$.time_us' AS BIGINT) BETWEEN 1700 AND 1800;
```

Values shredded differently compose freely. A set operation, `CASE` or `COALESCE` over differently-shredded values merges into one shredded type on the union of the shred sets, and an `INSERT` into a column with any other shred set — fully disjoint included — lands losslessly: the optimizer rewrites the struct cast into a reshred against the column's shred set (single-pass when the target keeps every source shred), and a path with no shred column in the target simply stays in the residual.

The safety net for the raw struct cast (no extension optimizer: another process, `SET disabled_optimizers='extension'`) is the *shred manifest*: each shredded value records, inside its residual, which paths were stripped into shred columns and as what type. A cast that silently drops or retypes a shred column leaves that record contradicting the data, and every JSONO reader verifies it — reading such a narrowed row raises an error instead of silently returning partial data. The manifest lives in the value's own blobs, so the protection rides with the data through Parquet and DuckLake. Shred order does not matter: the shred field order is canonicalized (sorted by name), so the same shred set always yields the *identical* type — `jsono_storage_type(<shred DDL>)` and `CREATE TABLE … AS SELECT` produce the same column type regardless of the order shreds are written in.

### `jsono_entries`

Flattens scalar leaves of a JSONO document into a list of key/value structs. The default `key_style` is `'jsonpath'`; pass the named argument `key_style := 'dotted'` for dot-separated keys:

```sql
SELECT unnest(jsono_entries(jsono('{"a":1,"b":{"c":"x"},"d":true}')));
-- {'key': $.a, 'value': 1}
-- {'key': $.b.c, 'value': x}
-- {'key': $.d, 'value': true}

SELECT unnest(jsono_entries(jsono('{"a":1,"b":{"c":"x"}}'), key_style := 'dotted'));
-- {'key': a, 'value': 1}
-- {'key': b.c, 'value': x}
```

JSON null leaves keep their key and return SQL `NULL` as `value`; empty objects and arrays do not produce entries. Dotted output can collide when a literal dotted key and a nested path render to the same string, for example `"a.b"` and `{"a":{"b":...}}`.

The order of the returned entries is unspecified — do not rely on it. Over a shredded value the shred leaves are emitted after the residual leaves (the shreds are read straight from their columns), so the same logical document can flatten in a different order than its plain form, which lists leaves in sorted-key order. Sort the result yourself if you need a stable order.

### `jsono_group_merge`

Aggregates a stream of JSON object patches with [RFC 7396](https://datatracker.ietf.org/doc/html/rfc7396) merge semantics: later patches in the ordered stream overwrite earlier keys and arrays replace wholesale. `null` object members are dropped before merging so existing keys stay untouched. Provide an `ORDER BY` clause to make the fold deterministic.

```sql
WITH patches(patch, ts) AS (
    VALUES
        (jsono('{"a":1,"b":{"x":1}}'), 1),
        (jsono('{"a":null,"b":{"y":2}}'), 2),
        (jsono('{"c":3}'), 3)
)
SELECT to_json(jsono_group_merge(patch ORDER BY ts))
FROM patches;
```
*Result:*
```json
{"a":1,"b":{"x":1,"y":2},"c":3}
```

Use `GROUP BY` for grouped state accumulation:

```sql
SELECT session_id,
       to_json(jsono_group_merge(patch ORDER BY event_ts)) AS state
FROM session_patch_events
GROUP BY session_id;
```

Use it as a window function to materialize the running state after each patch:

```sql
SELECT id,
       to_json(jsono_group_merge(patch ORDER BY id)
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

### `jsono_group_merge_max` / `jsono_group_merge_min`

Order-independent variants of `jsono_group_merge` that take the ordering key as an ordinary second argument instead of an `ORDER BY` clause. Per leaf, the value from the row with the **greatest** `order_key` wins (`jsono_group_merge_max`) or the **smallest** wins (`jsono_group_merge_min`); `null` object members never overwrite, exactly like `jsono_group_merge`. They are drop-in replacements for the ordered form:

```text
jsono_group_merge_max(value, order_key)  ≡  jsono_group_merge(value ORDER BY order_key)
jsono_group_merge_min(value, order_key)  ≡  jsono_group_merge(value ORDER BY order_key DESC)
```

```sql
WITH hits(params, event_ts) AS (
    VALUES
        (jsono('{"utm":{"source":"google"},"page":"/a"}'), 1),
        (jsono('{"utm":{"medium":"cpc"}}'), 2),
        (jsono('{"page":"/b"}'), 3)
)
SELECT to_json(jsono_group_merge_max(params, event_ts)) AS latest_wins,
       to_json(jsono_group_merge_min(params, event_ts)) AS earliest_wins
FROM hits;
```
*Result:*
```text
latest_wins                                          earliest_wins
{"page":"/b","utm":{"medium":"cpc","source":"google"}}   {"page":"/a","utm":{"medium":"cpc","source":"google"}}
```

`order_key` is any comparable type; for a composite tie-break pass a `ROW(...)`/struct, which compares lexicographically:

```sql
SELECT to_json(jsono_group_merge_max(params, ROW(event_ts, hit_id)))
FROM hits;
```

**Why a separate function?** `jsono_group_merge(value ORDER BY key)` is order-dependent, so DuckDB buffers and sorts *every* input row before folding — `O(rows)` memory, which can exhaust memory on large groups. The keyed variants resolve conflicts per leaf by the key argument, so they are commutative and associative: DuckDB streams rows straight into the aggregate with no buffer and the state stays `O(distinct leaves per group)`. Use these when folding a large stream (e.g. collapsing event hits into sessions) where the ordered form runs out of memory.

A shredded input keeps its shredding in the output, just like `jsono_group_merge`.

These functions require each path to keep a consistent kind across rows (always an object, or always a scalar/array). A path that is an object in one row and a scalar/array in another makes per-leaf last-write-wins order-dependent, so it is rejected with a clear error — use `jsono_group_merge(value ORDER BY key)` for such structurally-inconsistent data.

### `jsono_merge_patch`

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

### `jsono_diff`

Returns the structural diff of `prev → cur`: a minimal document containing only the keys/paths that changed. Each changed value renders by the **`cur`-side type** (the diff describes how to reach `cur`); a removed key (or a value cleared to `null`) renders as `key: null`. Changed scalars keep their native JSON type.

```sql
SELECT to_json(jsono_diff(
    jsono('{"status":"new","total":10,"note":"x"}'),
    jsono('{"status":"paid","total":10}')
));
```
*Result:*
```json
{"note":null,"status":"paid"}
```

The optional `arrays` named parameter selects how an array-valued change renders (object/scalar handling is identical in all three modes):

| `arrays` | an array change renders as | |
|----------|----------------------------|---|
| `'atomic'` *(default)* | the whole new `cur` array | order-sensitive; the exact inverse of `jsono_merge_patch` |
| `'counts'` | `{"added": N, "removed": M}` | order-insensitive multiset cardinalities |
| `'elements'` | `{"added": [...], "removed": [...]}` | order-insensitive multiset elements |

```sql
SELECT to_json(jsono_diff(
    jsono('{"items":[{"sku":"a"},{"sku":"b"}]}'),
    jsono('{"items":[{"sku":"a"},{"sku":"c"}]}'),
    arrays := 'counts'
));
```
*Result:*
```json
{"items":{"added":1,"removed":1}}
```

The default `'atomic'` mode round-trips with `jsono_merge_patch` — `jsono_merge_patch(prev, jsono_diff(prev, cur)) = cur` (for `cur` with no explicit `null` at an object key, since `null` is the merge-patch delete sentinel). The `'counts'` and `'elements'` modes are change *reports*, not appliable patches. A `NULL` argument is treated as the empty document, so `jsono_diff(NULL, cur)` is "everything new" and `jsono_diff(prev, NULL)` is "everything removed" — handy for a `lag()` window over a snapshot stream:

```sql
SELECT seq, to_json(changed) AS changed
FROM (
    SELECT seq,
           jsono_diff(lag(snapshot) OVER (ORDER BY seq), snapshot, arrays := 'counts') AS changed
    FROM events
)
WHERE changed IS DISTINCT FROM jsono('{}');  -- drop no-op rows (a pure array reorder is also a no-op under counts/elements)
```

A number's representation is part of its identity, so `1`, `1.0` and `1e0` compare unequal (a snapshot whose serializer drifts a number's form reports a change — normalize upstream if that matters).

### Introspection

```sql
jsono_type(value[, path])     -> VARCHAR    -- OBJECT/ARRAY/VARCHAR/BIGINT/UBIGINT/DOUBLE/BOOLEAN/NULL
jsono_keys(value[, path])     -> VARCHAR[]  -- object keys
jsono_validate(value)         -> BOOLEAN    -- strict current-format validation
jsono_storage_size(value)     -> STRUCT     -- physical byte sizes (body blobs + shreds + total)
jsono_storage_type()          -> VARCHAR    -- DDL of the physical STRUCT backing JSONO
jsono_shred_manifest(value)   -> STRUCT[]   -- paths stripped into shred columns, with their types
```

`jsono_type` and `jsono_keys` inspect the shape of unknown JSON before extracting it with `jsono_transform`. The optional `path` is a constant JSONPath (same grammar as `jsono_transform`, without wildcards) that points at a nested position; a missing path yields `NULL`.

```sql
SELECT jsono_keys(jsono('{"user":{"name":"a","age":1}}'), '$.user');
-- [age, name]
SELECT jsono_type(jsono('{"items":[1,2,3]}'), '$.items[0]');
-- BIGINT
```

`jsono_validate` checks the current binary format contract: header flags, sorted unique keys, slots, heaps, navigation metadata, UTF-8 strings, and raw number text. It returns `false` for malformed physical blobs and `NULL` for SQL `NULL`. On a shredded value it validates the residual encoding; the typed shred columns are DuckDB-native and not jsono's to assert.

`jsono_storage_size` reports `slots`, `key_heap`, `string_heap`, `skips`, `shreds`, and `total` byte counts. `shreds` is the per-row shred payload (0 for a plain value); `total` sums the body blobs and the shreds. It is a physical diagnostic and does not validate the value.

`jsono_storage_type` returns the DDL of the physical `STRUCT` that a JSONO value is — the form stored everywhere (DuckDB-native, DuckLake, plain Parquet). Use it to declare storage columns without hardcoding the field names; a column of that exact shape binds to `jsono` ops and `to_json` structurally, with no cast.

```sql
SELECT jsono_storage_type();
-- STRUCT(jsono STRUCT(body STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB, nums BLOB)))
```

`jsono_shred_manifest` returns a `LIST<STRUCT(path VARCHAR, type VARCHAR)>` of the paths a shredded value stripped out of its residual into shred columns, each with the shred type it was written as, in canonical (sorted) order. A plain value returns an empty list (nothing was stripped); a SQL `NULL` returns `NULL`. It reads the row's own manifest claim and, unlike the other readers, does not raise on a narrowed row — pair it with `jsono_validate` to debug shredding, layout, or migration questions (which paths got lifted, did a cast drop one). A shred value that did not round-trip its declared type stays in the residual and is correctly absent from the manifest, so a short manifest is not "shredding did not happen".

```sql
SELECT jsono_shred_manifest(jsono('{"a":"x","b":1}', shredding := {'a':'VARCHAR','b':'BIGINT'}));
-- [{'path': a, 'type': VARCHAR}, {'path': b, 'type': BIGINT}]
```

These helpers are primarily used by the JSONO workflows and tests. Treat their exact surface as less stable than `jsono_transform`, `jsono`, and `to_json`.

## Benchmarks

`bench/` is the comparable harness for `jsono` versions and the core DuckDB `json` baseline. Run a smoke benchmark:

```bash
uv run --frozen python bench/bench.py --filter group_merge/1k --runs 1
```

The operation set, version/core-json comparisons, field-sample scenarios, and result-reading contract are documented in [bench/README.md](bench/README.md); profiling is in [bench/PROFILING.md](bench/PROFILING.md).

## Development

Run the full local pre-PR gate (build + tests + format check) in one command:

```bash
uv run make verify
```

Or run the gates individually:

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

- Object keys inside JSONO output are sorted, so serialized JSON text may not preserve input object key order.
- `jsono_transform` requires a constant `STRUCT` spec; its optional `on_type_mismatch` argument must also be constant and one of `convert`, `null`, or `fail`.
- `jsono(value, shredding := spec)` requires a constant `STRUCT` spec and rejects wildcard paths; object-key paths that round-trip through their shred type are physically stripped from the residual, while array-index paths and non-lossless shred values stay in the residual.
- Shredded values and core json interop: the jsono-named functions (`jsono_extract`, `jsono_extract_string`, `jsono_type`, `jsono_keys`, `jsono_validate`, `jsono_storage_size`), `jsono_group_merge`, the `::VARCHAR` cast, and the shredded→plain JSONO reconstruct are bind-correct — they accept a shredded value directly and work even with the extension optimizer disabled. The operators and casts that resolve to the bundled core `json` extension — `->>`, `->`, `json_extract`, `::JSON`, and `to_json` — are correct over a shredded value only with the extension optimizer enabled (the default): the optimizer rewrites them to read the shreds and residual. With the optimizer fully disabled (`PRAGMA disable_optimizer`) those core-routed operations bind into core json and serialize the physical fields instead. This is a structural limit: core json owns the `STRUCT → JSON` cast and DuckDB's cast registry keeps the first registration, so the extension cannot intercept it.
- Declare a shredded storage column with `CREATE TABLE … AS SELECT jsono(…, shredding := …)` or from `jsono_storage_type(<shred DDL>)`, and access a shred through `->>`/`to_json` (the optimizer reads the shred) rather than by struct member name — `.body` and the shred fields are physical storage details. Inserting a value shredded on any other shred set — fully disjoint included — lands losslessly (the optimizer reshredds it to the column's shred set). A raw struct cast that drops or retypes a shred column (only possible without the extension optimizer) is caught at read time by the shred manifest: reading the narrowed row raises an error instead of silently losing the stripped value.
- `jsono_extract` / `->` / `jsono_extract_string` / `->>` require a constant path, do not support wildcard list extraction yet, and do not support negative array indexes.
- Unsupported scalar types in `jsono_transform` fail at bind time.
- JSON nested deeper than 1000 levels is rejected with an error (the limit is lowered to 50 under sanitizer builds).
- The binary JSONO format is strict-versioned (current `version = 4`, reported by `jsono_version()`). Incompatible storage changes must bump `jsono::VERSION`. A present-but-unreadable header — wrong magic, a different format version, misaligned slots — fails loud on read rather than silently reading as SQL `NULL`; only an absent value (a slots blob too short to hold a header) reads as `NULL`, and `jsono_validate` reports a corrupt blob as `false`.
- Malformed input raises DuckDB errors unless `try_jsono` is used.

## Known limitations

The current surface is intentionally small. The following are not implemented:

- No cast from JSONO to a scalar type (`jsono('42')::INTEGER` is unsupported). Project string values out with `->>` / `jsono_extract_string` or typed fields with `jsono_transform`.
- No cast from JSONO to an arbitrary `STRUCT`. Only the physical six-BLOB shape is interchangeable with JSONO; field extraction goes through `jsono_transform`.
- No dedicated table function that unnests a JSONO array or object into rows. Use `unnest(jsono_entries(...))` for flattened scalar leaves.
- Object key order is not preserved: keys are stored and emitted in sorted byte order (see [Error Handling and Caveats](#error-handling-and-caveats)).

## Contributing

Contributions and feedback are welcome. Please:

1. Open an issue first to discuss proposed changes.
2. Add or update SQLLogic tests in `test/sql/` for new behavior.
3. Run `uv run make verify` (build + tests + format check) before submitting a pull request. CI additionally runs `uv run make tidy-check`, which needs a local clang-tidy + compile-database setup.

See [GitHub Issues](https://github.com/Flamefork/duckdb-jsono/issues) for current tasks and feature requests.

## License

MIT. See [LICENSE](LICENSE).

For third-party components and their licenses, see [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

---
*This extension is based on the [DuckDB Extension Template](https://github.com/duckdb/extension-template).*
