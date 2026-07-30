# DuckDB JSONO

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Main Extension Distribution Pipeline](https://github.com/Flamefork/duckdb-jsono/actions/workflows/MainDistributionPipeline.yml/badge.svg)](https://github.com/Flamefork/duckdb-jsono/actions/workflows/MainDistributionPipeline.yml)

A DuckDB extension for analytics-optimized JSON storage and queries.

- **Parse one time, read many times.** `jsono(payload)` converts a JSON document into a pre-parsed binary value. Queries navigate the binary structure and do not parse text again.
- **Columnar reads of hot paths.** With `shredding := spec`, the same constructor also shreds hot paths (the paths that queries read most) into typed columns. Filters and extractions on these paths read narrow typed columns and can prune Parquet row groups. `->>` and `to_json` do not change, and the conversion is lossless.
- **A typed SQL toolset.** One-pass projection into a typed `STRUCT` (`jsono_transform`), RFC 7396 merge aggregates, structural diff, and introspection of the document, the row, and the type — see the [Function reference](#function-reference).

> [!WARNING]
> This extension is experimental. The maintainer does not write C++ professionally and maintains the extension on a best-effort basis. Rough edges are possible. The extension is not hardened for production. Do tests of the behavior and the performance on your own data before you use the extension. Give feedback and contributions through GitHub.

> [!NOTE]
> **Stability.** Stored values stay readable across upgrades: a newer build reads the values that an older release wrote. The SQL API is not frozen: function names and signatures can change while the version is `0.x`.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [The JSONO format](#the-jsono-format)
- [Function reference](#function-reference)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [Interop and limits](#interop-and-limits)
- [Build from source](#build-from-source)
- [Development](#development)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Installation

Install the extension from the [DuckDB Community Extensions](https://duckdb.org/community_extensions/) repository:

```sql
INSTALL jsono FROM community;
LOAD jsono;
```

Community builds are signed. No special flags are necessary.

For development, or for a platform that has no community build, see [Build from source](#build-from-source).

## Quick Start

> [!NOTE]
> JSONO is a format, not a SQL type. There is no `JSONO` name for a `CAST` or a column definition — see [The JSONO format](#the-jsono-format).

```sql
SELECT jsono_transform(
    jsono('{"id": 42, "name": "duck", "active": true}'),
    {id: 'BIGINT', name: 'VARCHAR', active: 'BOOLEAN'}
);
-- {'id': 42, 'name': duck, 'active': true}
```

Convert a column to JSONO when queries will read the same JSON values many times:

```sql
CREATE TABLE events AS
SELECT jsono(payload) AS payload_jsono
FROM read_parquet('events.parquet');

SELECT (jsono_transform(payload_jsono, {user_id: 'VARCHAR', event_ts: 'BIGINT'})).*
FROM events;
```

## The JSONO format

JSONO is a format. It is not a SQL type. A JSONO value is the extension's pre-parsed binary version of a JSON document, carried in a plain nested `STRUCT`:

```text
STRUCT                             -- a JSONO value
└─ jsono                           -- the layout anchor field
   ├─ body$2                       -- the document body
   │  ├─ slots          BLOB
   │  ├─ key_heap       BLOB
   │  ├─ string_heap    BLOB
   │  ├─ skips          BLOB
   │  ├─ lengths        BLOB
   │  └─ nums           BLOB
   └─ shreds$2                     -- shredded values only
      ├─ $jsono$set                -- reserved fields (marker, spill bitmap)
      ├─ $jsono$spill$…
      └─ <encoded path>  <TYPE>    -- one typed lane per shredded path
```

Each storage layer (DuckDB-native, Parquet, DuckLake) sees this same shape.

There is no `JSONO` type name to use in a `CAST` or in a column definition. Make values with `jsono(...)`; declare columns with `jsono_storage_type()`. The functions, the `->` / `->>` operators, and `to_json` identify the `STRUCT` shape itself, so a value that comes back from Parquet or DuckLake binds with no cast.

The field names are physical storage details, not a public API. A shred field name is the encoded form of its path; read the lane set with `jsono_layout_lanes(value)`. The `$N` suffixes are layout revisions. A value that this build cannot decode — a foreign revision, an unreadable header — fails loudly on read, never silently incorrect; only an absent value (a slots blob too short for a header) reads as SQL `NULL`.

The binary layout of the body BLOBs, the lane-name encoding, and the revision history are specified in [jsono_format.md](docs/jsono_format.md).

### Shredded storage

With `shredding := spec`, [`jsono()`](#jsonovalue-shredding) fills the `shreds$2` branch above: each path in the spec gets a typed lane column, and the residual keeps the rest of the document. Reads do not change — `->>`, `jsono_extract_string`, `to_json`, and the casts operate the same as on a plain value. But an extraction of a shredded path becomes a direct read of its typed lane, which the planner can push down and prune on.

The usual workflow has three steps:

1. Aggregate the plain column with [`jsono_suggest_shredding`](#jsono_suggest_shredding) to get a spec string.
2. Construct the shredded column with [`jsono(value, shredding := spec)`](#jsonovalue-shredding).
3. Measure the lanes with [`jsono_shred_stats`](#jsono_shred_stats).

The conversion is lossless. `to_json` reconstructs the initial document, and a path that is not in the spec still reads from the residual. A value that does not fit its shred type (a string in a `BIGINT` lane, an explicit JSON `null`, a missing key) stays in the residual with no change. A top-level shredded path is stored one time — in the lane, not also in the residual — so the shredded column is usually smaller than the plain column for the same data.

```sql
SELECT to_json(jsono('{"kind":"commit","time_us":1700,"extra":"e1"}',
                     shredding := '{"$.kind": "VARCHAR", "$.time_us": "BIGINT"}'));
-- {"extra":"e1","kind":"commit","time_us":1700}
```

The shredded shape survives a plain Parquet round trip, and the scalar lanes keep projection and filter pushdown. Values with different shred sets combine freely: a set operation, a `CASE`, or a `COALESCE` merges them on the union of the shred sets, and an `INSERT` into a column with a different shred set — including a fully different one — reshreds losslessly (a path with no lane in the target stays in the residual). The lane fields are stored in canonical order (sorted by name), so the same shred set always makes the identical column type.

To change the shred set of an existing column, rewrite the column through `jsono(value, shredding := spec)` with the new spec. Do not edit the shred fields with DDL — a field name that the codec did not make refuses JSON reads (see [Troubleshooting](#a-jsono-column-answers-null-or-refuses-reads)).

Most reads use the lanes directly. A read that requires the full logical value — the usual example is a `jsono_transform` field that goes down into an array shred — first reconstructs the value to plain JSONO, at a 3–10× per-row cost. The plan shows this: see [Is my read pushed down?](#is-my-read-pushed-down).

The *shred manifest* protects the data when the extension optimizer is not available (a different process, `SET disabled_optimizers='extension'`). Each shredded value records in its residual which of its paths are shredded, and their types. A raw struct cast that removes or retypes a lane makes the record disagree with the data, and a read of such a narrowed row causes an error instead of silent partial data. The record is in the value's own blobs, so the protection rides through Parquet and DuckLake.

### Get row-group pruning

A filter over shredded Parquet data prunes row groups (skips them on the per-row-group min/max statistics) only under the two conditions below. Without them, the scan reads the full file.

- **Filter a typed shred in its own type.** A `VARCHAR` shred prunes on the plain `payload ->> '$.kind' = 'commit'`. A numeric or boolean shred prunes only when the comparison is in the shred's type: `CAST(payload ->> '$.time_us' AS BIGINT) = 1700`, not the string form `payload ->> '$.time_us' = '1700'`. `->>` is textual. So the string form compares the shred after a conversion back to text. That is a computed expression: the scan cannot push it down and reads each row group. To get pruning on a typed shred, compare with `CAST(... AS <shred type>)`, or project the typed value with `jsono_transform`.
- **Cluster the rows on the leaf at write time.** Pruning is possible only when each row group holds a separate range of the filtered path. So add `ORDER BY` on that path to the `CREATE TABLE … AS SELECT` (or to the `COPY`). Without this order, each row group contains the full domain, and the scan cannot skip a row group. This order also improves compression. So one `ORDER BY` on the primary filter path helps two times.

```sql
CREATE TABLE events AS
SELECT jsono(payload, shredding := '{"$.kind": "VARCHAR", "$.time_us": "BIGINT"}') AS payload
FROM read_parquet('events.parquet')
ORDER BY CAST(payload ->> '$.time_us' AS BIGINT);   -- cluster on the hot filter path

-- prunes to the matching row groups (typed shred compared as BIGINT)
SELECT count(*) FROM events WHERE CAST(payload ->> '$.time_us' AS BIGINT) BETWEEN 1700 AND 1800;
```

**For reads across many Parquet files, use `union_by_name := true`.** A plain multi-file scan (`read_parquet('dir/*.parquet')`, `read_parquet([...])`, or a DuckLake table over many data files) returns no per-file column statistics to the optimizer. Then a shred read becomes a `COALESCE(shred, residual)`. This read uses the lane but cannot prune row groups. With `union_by_name := true`, the engine supplies merged statistics. For files with different shred sets, the engine also recovers statistics. The shred is then total, and the read becomes a bare `struct_extract_at` that can prune — the same plan as a single-file read. Make this option the default for multi-file shredded facts:

```sql
-- prunes across files; without union_by_name this keeps a non-prunable COALESCE
SELECT count(*) FROM read_parquet('events/*.parquet', union_by_name := true)
WHERE CAST(payload ->> '$.time_us' AS BIGINT) BETWEEN 1700 AND 1800;
```

## Function reference

Build and convert:

- [`jsono(json)` / `try_jsono(json)`](#jsonojson) — parse JSON text into JSONO.
- [`jsono(struct)`](#jsonostruct) — construct JSONO from typed DuckDB values.
- [`to_json(value)`](#to_json) — serialize JSONO back to `JSON`.

Extract and project:

- [`jsono_extract(value, path)` / `value -> path`](#jsono_extract) — one value as JSONO.
- [`jsono_extract_string(value, path)` / `value ->> path`](#jsono_extract_string) — one value as `VARCHAR`.
- [`jsono_transform(value, spec[, on_type_mismatch])`](#jsono_transform) — many fields as a typed `STRUCT`, in one pass.
- [`jsono_entries(value[, key_style, array_style])`](#jsono_entries) — scalar leaves as a key/value list.
- [`jsono_array_elements(value[, path])`](#jsono_array_elements) — array elements as a `LIST` of JSONO values.

Merge and aggregate:

- [`jsono_merge_patch(target, patch[, ...])`](#jsono_merge_patch) — RFC 7396 merge patch.
- [`jsono_group_merge(value [ORDER BY ...])`](#jsono_group_merge) — aggregate a stream of patches.
- [`jsono_group_merge_max(value, order_key)` / `jsono_group_merge_min(value, order_key)`](#jsono_group_merge_max--jsono_group_merge_min) — keyed order-independent merge.
- [`jsono_diff(prev, cur[, arrays])`](#jsono_diff) — structural difference of two documents.
- [`jsono_group_array(value [ORDER BY ...])` / `jsono_group_object(key, value)`](#jsono_group_array--jsono_group_object) — collect a group into one array or object.

Shredded storage:

- [`jsono(value, shredding := spec)`](#jsonovalue-shredding) — the shredding constructor.
- [`jsono_suggest_shredding(value[, min_presence, min_fit])`](#jsono_suggest_shredding) — suggest a spec from a plain column.
- [`jsono_shred_stats(value)`](#jsono_shred_stats) — lane and divert rates of a shredded column.

[Introspection](#introspection) — three question groups: the document, this row, the type:

- document: [`jsono_type`, `jsono_keys`, `jsono_array_length`](#introspection)
- row: [`jsono_validate`, `jsono_storage_size`, `jsono_shred_manifest`](#introspection)
- type: [`jsono_layout_diagnose`, `jsono_layout_lanes`, `jsono_storage_type`](#introspection)

### `jsono(json)`

`jsono(json)` parses JSON text or a `JSON` value into JSONO. The parse is strict.

```sql
SELECT jsono('{"a":1,"b":"x"}');
```

Invalid or empty input causes an error. Use `try_jsono` if invalid input must become `NULL`:

```sql
SELECT try_jsono('{not json') IS NULL;
```

`try_jsono` never causes an error. A limit violation on valid JSON (nesting past the depth limit, a key or a string that is too large) also becomes `NULL`, and you cannot know the difference from invalid input. Use the strict `jsono` when a limit overflow must cause an error and must not drop the row.

To parse the text and shred the hot paths in one pass, give a constant `shredding := '{…}'` specification — see [`jsono(value, shredding)`](#jsonovalue-shredding):

```sql
SELECT jsono('{"kind":"commit","time_us":1700}', shredding := '{"$.kind": "VARCHAR", "$.time_us": "BIGINT"}');
```

### `jsono(struct)`

`jsono(struct)` makes JSONO directly from typed DuckDB values. It does not serialize the input through JSON text first.

```sql
SELECT to_json(jsono({'b': 2, 'a': 'x', 'n': NULL, 'arr': ['c', NULL, 'd']}));
```
*Result:*
```json
{"a":"x","arr":["c",null,"d"],"b":2,"n":null}
```

`STRUCT` fields become JSON object keys. `LIST` values become arrays. The function parses `JSON` children as JSON subtrees. `VARCHAR` children stay JSON strings. Exact integer and decimal values do not change.

JSON has no NaN or Infinity representation, so the function refuses non-finite `DOUBLE` values. SQL `NULL` object fields become JSON nulls.

Input types and their JSON output:

| Input type | JSON |
|---|---|
| `BOOLEAN` | `true` / `false` |
| `TINYINT` … `BIGINT`, unsigned narrow ints, `HUGEINT`/`UHUGEINT`/`UBIGINT`, `VARINT`, `DECIMAL` | number, exact digits |
| `FLOAT`, `DOUBLE` | number (the function refuses non-finite values) |
| `VARCHAR` | string |
| `JSON` | parsed subtree |
| `DATE`, `TIME`(`_NS`), `TIMETZ`, `TIMESTAMP`(`_S`/`_MS`/`_NS`/`TZ`), `UUID`, `ENUM`, `INTERVAL`, `BLOB`, `BIT` | string, identical to the core `to_json` output |
| `STRUCT`, `MAP` | object |
| `LIST`, `ARRAY` | array |
| a `jsono` value | a subtree, with no text round trip (a shredded value is reconstructed first) |

A `MAP` builds an object whose keys are per-row data: the keys become text and are **sorted** (core `to_json` keeps insertion order; the pairs are the same). Two typed keys that make the same text would collapse into one entry, so that case is an error. `UNION` is not supported and gives a bind error.

`TIMESTAMPTZ` values become UTC text (`2024-01-01 12:00:00+00`), independent of the session `TimeZone`: the same instant must make the same bytes — and the same lane statistics — in each session. Core `to_json` writes the same UTC text.

The document root does not have to be an object. `jsono(MAP {...})`, `jsono([1, 2, 3])`, and `jsono(array_value(1, 2))` make object or array documents. The related casts let you `INSERT` a `MAP`, a `LIST`, or an `ARRAY` directly into a JSONO column. A `VARCHAR` root is the text parser by contract. So you cannot construct a scalar root — put the scalar in a list or in an object.

The input `STRUCT` has its field names and types, so `jsono(struct)` shreds automatically — the same operation that the [`shredding`](#jsonovalue-shredding) constructor does for text, with no `shredding` argument. The lane per field type (`to_json` output does not change):

| Field type | Lane |
|---|---|
| `BIGINT`, `UBIGINT`, `DOUBLE`, `BOOLEAN`, `VARCHAR` | the same type |
| `TINYINT`/`SMALLINT`/`INTEGER`, `UTINYINT`/`USMALLINT`/`UINTEGER` | `BIGINT` |
| `FLOAT` | `DOUBLE` |
| string-rendered types (`TIMESTAMP`, `DATE`, `UUID`, `ENUM`, `BLOB`, …) | `VARCHAR`, exactly the text that the document holds |
| `LIST<TYPE>` / `ARRAY<TYPE>`, with `TYPE` from the rows above | scalar-array lane |
| `LIST<STRUCT<...>>` with supported scalar children | object-array lane |
| nested `STRUCT` leaves (`$.parent.child`, `$.URL.query_params.utm_source`, …) down to a high depth limit | one lane per leaf |
| `HUGEINT`/`UHUGEINT`/`VARINT`, `DECIMAL`, `MAP`, `NULL`, `JSON` subtrees, unsupported lists, leaves past the depth limit | none — the value stays in the residual |

The depth limit exists only to keep a pathologically deep whole-value reconstruction from super-linear cost; a deeper leaf can still be shredded explicitly with `shredding := '{…}'`. Shredding applies to named top-level object fields, which only a `STRUCT` root has — `MAP`, `LIST`, `ARRAY`, and `jsono` roots make plain JSONO, and an embedded `jsono` field is a document, not a struct to walk: it stays in the residual as a whole.

```sql
SELECT typeof(jsono({'kind': 'commit', 'time_us': 1700::BIGINT}));
-- a shredded STRUCT: a `jsono` layout with the body blobs plus one lane per field. Each lane name is
-- the encoding of its path (`ddkmsp0000` VARCHAR for `kind`, `ehkmqpavelpg000` BIGINT for `time_us`).
-- Read the paths back with jsono_layout_lanes(value).
```

### `to_json`

`to_json` serializes JSONO back to compact `JSON`.

```sql
SELECT to_json(jsono('{"b":2,"a":1}'));
```
*Result:*
```json
{"a":1,"b":2}
```

The output shows object keys in sorted byte order. Explicit casts are also available:

```sql
SELECT CAST(jsono('{"a":1}') AS JSON);
SELECT CAST(jsono('{"a":1}') AS VARCHAR);
```

The `JSON` cast and `to_json` make DuckDB's core `JSON` logical type. The bundled `json` extension supplies this type. Each standard DuckDB distribution includes this extension, so no manual setup is necessary.

### `jsono_extract`

`jsono_extract` extracts one value at a constant path. `value -> path` is an alias for the same operation. A plain string path identifies a literal top-level key. A string path that starts with `$` uses JSONPath. A `BIGINT` path is an index into an array. Wildcard list extraction and negative array indexes are not supported.

> [!IMPORTANT]
> In a boolean condition, put an operator extraction in parentheses: write `WHERE cond AND (e ->> '$.k') = 'v'`. The `->` and `->>` operators bind more loosely than `AND` and `OR`, the same as in core `json` — the arrow token is shared with the [lambda syntax](https://duckdb.org/docs/stable/sql/functions/lambda.html), and a lambda body must bind loosely. Without parentheses, DuckDB parses `cond AND e ->> '$.k' = 'v'` as `(cond AND e) ->> …` and fails on a cast of the JSONO struct to `BOOLEAN`. The named functions `jsono_extract` and `jsono_extract_string` do not need parentheses. This is the expected DuckDB behavior ([duckdb/duckdb#16970](https://github.com/duckdb/duckdb/issues/16970)).

```sql
SELECT CAST(jsono_extract(jsono('{"a.b":"dot","a":{"b":"nested"}}'), 'a.b') AS VARCHAR);
-- "dot"

SELECT CAST(jsono('{"a.b":"dot","a":{"b":"nested"}}') -> '$.a' AS VARCHAR);
-- {"b":"nested"}

SELECT jsono('{"items":[{"name":"duck"}]}') -> 'items' -> 0 ->> 'name';
-- duck
```

A JSON null at the path returns a JSONO null value. A missing path returns SQL `NULL`.

### `jsono_extract_string`

`jsono_extract_string` extracts one value at a constant path. `value ->> path` is an alias for the same operation. The path rules are the same as for `jsono_extract`:

```sql
SELECT jsono_extract_string(jsono('{"a.b":"dot","a":{"b":"nested"}}'), 'a.b');
-- dot

SELECT jsono('{"a.b":"dot","a":{"b":"nested"}}') ->> 'a.b';
-- dot

SELECT jsono_extract_string(jsono('{"a.b":"dot","a":{"b":"nested"}}'), '$.a.b');
-- nested
```

A JSON string returns text without quotation marks. A number or a boolean returns its JSON text. A JSON null or a missing path returns SQL `NULL`. An array or an object returns compact JSON text.

### `jsono_transform`

`jsono_transform` extracts fields from JSONO into a typed DuckDB `STRUCT`. The `spec` argument must be a constant `STRUCT`. It maps each output field name to an extraction rule.

```sql
SELECT jsono_transform(
    jsono('{"a":42,"b":3.5,"c":"hi","d":true}'),
    {a: 'BIGINT', b: 'DOUBLE', c: 'VARCHAR', d: 'BOOLEAN'}
);
```

A rule is a type shorthand string or a wrapper `STRUCT`. These scalar types are supported:

- `BIGINT`
- `UBIGINT`
- `DOUBLE`
- `VARCHAR`
- `BOOLEAN`

A scalar type that is not in this list causes a bind-time error.

A shorthand reads the top-level key with the same name (the path default is `$."field"`). A wrapper `STRUCT` reads from an explicit JSONPath:

```sql
SELECT jsono_transform(
    jsono('{"event":{"timestamp":1234567890}}'),
    {ts: {type: 'BIGINT', path: '$.event.timestamp'}}
);
```

Paths support nested keys, array indices, quoted keys, and the `[*]` wildcard: `$.user.name`, `$.items[0].id`, `$."my-key"`, `$.tags[*]`.

The constant `on_type_mismatch` argument controls the behavior for a type mismatch:

- `convert` (default): the function does a CAST-type scalar conversion to the requested type. A value that it cannot convert becomes `NULL`.
- `null`: the function keeps the legacy silent-`NULL` behavior. It returns a scalar value only when the stored JSON type is the same as the requested projection type.
- `fail`: the function does the same scalar conversions as `convert`, but it stops with an error when it cannot convert a present value.

A missing path and an explicit JSON `null` return SQL `NULL` in each mode. `on_type_mismatch` applies independently to each projected field and to each wildcard list element.

```sql
SELECT jsono_transform(jsono({'a':'12345'}), {'a':'UBIGINT'});
-- {'a': 12345}

SELECT jsono_transform(jsono({'a':'12345'}), {'a':'UBIGINT'}, on_type_mismatch := 'null');
-- {'a': NULL}

SELECT jsono_transform(jsono({'a':'abc'}), {'a':'UBIGINT'}, on_type_mismatch := 'fail');
-- error: cannot convert value at $.a from VARCHAR to UBIGINT
```

To collect array elements into a typed list, use a list type with one element. To join the elements into one string, use `join_separator`:

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

The field names `type`, `path`, and `join_separator` are reserved in a wrapper `STRUCT`.

### `jsono(value, shredding)`

The named argument `shredding` changes `jsono()` into a *shredding* constructor. It produces a *shredded* `STRUCT` — the JSONO residual plus one typed shred per path in `spec`. [Shredded storage](#shredded-storage) describes the read behavior, pruning, and maintenance; this section gives the input contract and the spec grammar.

The primary form takes JSON **text**. It parses and shreds in one pass. If you give an existing **JSONO** value, the function shreds it directly. These are the only two permitted `value` inputs. There is no `jsono(struct, shredding := …)` overload, because [`jsono(struct)`](#jsonostruct) already shreds its result. To shred a struct-built value, build it with `jsono({…})` directly. Do not put it in a wrapper.

`spec` is a constant string that holds a JSON object. The object maps each path to its shred type string. This is the same carrier that core json's `json_transform` uses for its spec: JSON keys stay byte-exact, so two case-spellings of one key declare two lanes. The path grammar is the same as in `jsono_transform`, without wildcards. Scalar shreds use `VARCHAR`, `BIGINT`, `UBIGINT`, `DOUBLE`, or `BOOLEAN`. Scalar-array shreds use `VARCHAR[]`, `BIGINT[]`, `UBIGINT[]`, `DOUBLE[]`, or `BOOLEAN[]`. Object-array shreds use `STRUCT(<supported scalar children>)[]`.

A `$` at the start of a path starts a `$.`-rooted path. To shred a document key that starts with `$`, write the key as a quoted step below the root: `'{"$.\"$foo\"": "VARCHAR"}'` shreds the key `$foo`. This is the only reserved spelling in the spec grammar. The storage format itself reserves no key.

```sql
CREATE TABLE events AS
SELECT jsono(payload, shredding := '{"$.kind": "VARCHAR", "$.did": "VARCHAR", "$.time_us": "BIGINT"}') AS payload
FROM read_parquet('events.parquet');

-- queried like any JSONO column; the planner reads the shreds
SELECT payload ->> '$.kind' AS kind, count(*) AS n
FROM events GROUP BY kind;

SELECT max(CAST(payload ->> '$.time_us' AS BIGINT)) FROM events;
```

An array-shred spec gives the object-key path of the array. It does not use an indexed or wildcard element path. Elements or subfields that do not fit the requested shred type stay in the residual skeleton. So reconstruction never loses data and never converts data.

```sql
SELECT to_json(jsono('{"products":[{"id":1,"name":"a"}]}',
                     shredding := '{"$.products": "STRUCT(id UBIGINT, name VARCHAR)[]"}'));
-- {"products":[{"id":1,"name":"a"}]}
```

### `jsono_suggest_shredding`

`jsono_suggest_shredding` aggregates a **plain** JSONO column and returns a `shredding :=` spec string that you can paste. You do not have to select shred paths and types manually.

The function walks each document over the set that the constructor auto-shreds: top-level scalar keys; nested scalar keys (`$.parent.child`, `$.a.b.c` — down to the same high auto-shred depth limit, so web-event leaves such as `$.URL.query_params.utm_source` are included); and top-level arrays (as scalar-array `TYPE[]` or object-array `STRUCT(...)[]`). For each path, the function selects the narrowest lane type that holds each present value losslessly. Nested arrays (`$.parent.items`) are not in that set, although the constructor shreds them. So a nested list path never appears in the suggested spec.

```sql
SELECT jsono_suggest_shredding(payload) FROM events;
-- {"$.meta.seq": "BIGINT", "kind": "VARCHAR", "tags": "BIGINT[]"}

-- paste the result straight into the shredding constructor
CREATE TABLE shredded AS
SELECT jsono(payload, shredding := '{"$.meta.seq": "BIGINT", "kind": "VARCHAR", "tags": "BIGINT[]"}') AS payload
FROM events;
```

Two named arguments adjust the thresholds. Both are fractions in `[0.0, 1.0]`:

- `min_presence` (default `0.5`) — the function keeps a path only when the path is present in at least this fraction of the non-`NULL` rows. Sparse fields do not appear in the result.
- `min_fit` (default `1.0`, fully lossless) — a lane type must hold at least this fraction of the rows that have the path. Rows whose value is an object count in the denominator, so a scalar-or-object key does not qualify at `1.0`. A lower value accepts a majority type, and the off-type values stay in the residual. An explicit JSON `null` fits every lane and never blocks a shred.

```sql
SELECT jsono_suggest_shredding(payload, min_presence := 0.9, min_fit := 0.99) FROM events;
```

Number lanes use the `BIGINT` > `DOUBLE` > `BOOLEAN` > `VARCHAR` preference. The function selects `UBIGINT` only when a value is out of the `int64` range. A path with no applicable lane is not in the result. If no path qualifies, or if the column is all `NULL`, the result is `NULL`. The input must be the **plain** column. To examine a column that is already shredded, use `jsono_shred_stats`.

One output is not paste-able: an object-array whose element keys differ only in letter case. Its `STRUCT(...)` element type is refused by the SQL type parser — see the paste-back note under `jsono_layout_lanes` in [Introspection](#introspection).

### `jsono_shred_stats`

`jsono_shred_stats` aggregates an existing **shredded** column. For each shred, it reports how well the shred set operates. The result is a `LIST<STRUCT(path VARCHAR, type VARCHAR, lane_rate DOUBLE, divert_rate DOUBLE)>`, in the lane order of the type (the same order that `jsono_layout_lanes` reports):

- `lane_rate` — the fraction of non-`NULL` rows whose typed lane holds a shredded value.
- `divert_rate` — the fraction of non-`NULL` rows where the lane is `NULL` because the value did not fit its shred type and stayed in the residual. A high divert rate means that the shred type is too narrow. An explicit JSON `null` (a complete NULL lane) is lossless and does **not** count as a divert.

```sql
SELECT to_json(jsono_shred_stats(payload)) FROM shredded;
-- [{"path":"$.kind","type":"VARCHAR","lane_rate":1.0,"divert_rate":0.0},
--  {"path":"$.time_us","type":"BIGINT","lane_rate":0.98,"divert_rate":0.02}]
```

The function refuses plain (non-shredded) input. Use `jsono_suggest_shredding` to get shred suggestions for a plain column.

### `jsono_entries`

`jsono_entries` flattens the scalar leaves of a JSONO document into a list of key/value structs. The default `key_style` is `'jsonpath'`. Use the named argument `key_style := 'dotted'` for keys with dot separators:

```sql
SELECT unnest(jsono_entries(jsono('{"a":1,"b":{"c":"x"},"d":true}')));
-- {'key': $.a, 'value': 1}
-- {'key': $.b.c, 'value': x}
-- {'key': $.d, 'value': true}

SELECT unnest(jsono_entries(jsono('{"a":1,"b":{"c":"x"}}'), key_style := 'dotted'));
-- {'key': a, 'value': 1}
-- {'key': b.c, 'value': x}
```

A `'jsonpath'` key parses back to the same keys, so an ambiguous key is quoted and escaped: `{"a.b":1}` gives `$."a.b"` (one key, not two levels). `'dotted'` keys are display text and are not quoted.

The named argument `array_style` selects where the walk stops at an array. The default `'indexed_elements'` goes into arrays and gives each element an indexed key (`arr[0]`, `arr.0`, …). Use `array_style := 'whole_json'` to stop at the array boundary. Then each array becomes one leaf whose value is the full array as JSON text:

```sql
SELECT unnest(jsono_entries(jsono('{"items":[{"sku":"a"},{"sku":"b"}],"name":"x"}'), array_style := 'whole_json'));
-- {'key': $.items, 'value': [{"sku":"a"},{"sku":"b"}]}
-- {'key': $.name,  'value': x}
```

`array_style` and `key_style` are independent, and their argument order has no effect. In `'whole_json'`, the `value` of an array leaf is a JSON literal (parse it as JSON), and a scalar leaf stays bare text. An empty array appears as a present `[]` value. A document whose top level is an array is refused at run time — the mode is defined for object documents whose array-valued fields appear whole.

A JSON null leaf keeps its key and returns SQL `NULL` as `value`. Empty objects make no entries. With `indexed_elements`, empty arrays also make no entries. Dotted output can collide when a literal dotted key and a nested path make the same string, for example `"a.b"` and `{"a":{"b":...}}`.

The order of the returned entries is unspecified — a shredded value can flatten in a different order than its plain form. If you require a stable order, sort the result.

### `jsono_array_elements`

`jsono_array_elements` returns the elements of a JSON array as a `LIST` of complete JSONO values. Each element becomes a full JSONO document of its own. So `unnest` of the list gives one native JSONO value per row, with no `to_json` + reparse round trip:

```sql
SELECT unnest(jsono_array_elements(jsono('[{"name":"a"},{"name":"b"}]'))) ->> 'name';
-- a
-- b
```

The optional `path` gives a nested array. It has the same grammar as `jsono_extract`: a constant JSONPath (`'$.items'`), a bare top-level key (`'items'`), or a non-negative array index:

```sql
SELECT to_json(jsono_array_elements(jsono('{"items":[7,8,9]}'), '$.items')[2]);
-- 8
```

An empty array gives an empty `LIST` (`[]`). A non-array value (an object, a scalar, or a JSON `null`), a missing path, and a SQL `NULL` input each give SQL `NULL`. A JSON `null` *element* becomes a JSONO null document, in its position. The function reconstructs a shredded value through the validated read path before it splits out the elements.

### `jsono_group_merge`

`jsono_group_merge` aggregates a stream of JSON object patches with [RFC 7396](https://datatracker.ietf.org/doc/html/rfc7396) merge semantics. In the ordered stream, later patches overwrite earlier keys, and an array replaces the earlier array as a whole. The function removes `null` object members before the merge, so existing keys do not change. Give an `ORDER BY` clause to make the fold deterministic.

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

Use `GROUP BY` for state accumulation per group, or a window frame (`OVER (ORDER BY … ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`) to get the running state after each patch.

### `jsono_group_merge_max` / `jsono_group_merge_min`

`jsono_group_merge_max` and `jsono_group_merge_min` are order-independent versions of `jsono_group_merge`. They take the order key as a usual second argument, not as an `ORDER BY` clause. For each leaf, the value from the row with the **largest** `order_key` wins (`jsono_group_merge_max`), or the value from the row with the **smallest** `order_key` wins (`jsono_group_merge_min`). `null` object members never overwrite, exactly as in `jsono_group_merge`. The functions are direct replacements for the ordered form:

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

`order_key` can have each comparable type. For a composite tie-break, use a `ROW(...)` or a struct — the comparison is lexicographic:

```sql
SELECT to_json(jsono_group_merge_max(params, ROW(event_ts, hit_id)))
FROM hits;
```

**Why a separate function?** The ordered form buffers and sorts each input row before the fold — `O(rows)` memory, which a large group can exhaust. The keyed versions are commutative and associative, so DuckDB streams the rows with no buffer and the state stays `O(distinct leaves per group)`. Use them when the ordered form runs out of memory.

A shredded input keeps its shredding in the output, the same as with `jsono_group_merge`.

These functions require a consistent kind for each path across the rows: always an object, or always a scalar/array. A path that is an object in one row and a scalar or an array in a different row makes per-leaf last-write-wins order-dependent. So the functions refuse it with a clear error. For such structurally-inconsistent data, use `jsono_group_merge(value ORDER BY key)`.

### `jsono_merge_patch`

`jsono_merge_patch` applies one or more [RFC 7396](https://datatracker.ietf.org/doc/html/rfc7396) merge patches from left to right. The function is variadic and requires at least one argument. Each `null` object member in a patch deletes that key from the result.

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

Give more patches to fold them in sequence:

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

`jsono_group_merge` is an aggregate over rows. `jsono_merge_patch` is different: it is a scalar function over a fixed set of patch arguments.

### `jsono_diff`

`jsono_diff` returns the structural difference of `prev → cur`: a minimal document that contains only the keys and paths that changed. Each changed value appears with its **`cur`-side type** (the diff tells how to get to `cur`). A removed key, or a value that changed to `null`, appears as `key: null`. A changed scalar keeps its native JSON type.

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

The optional named parameter `arrays` selects the format of an array-valued change. The handling of objects and scalars is identical in all three modes:

| `arrays` | an array change appears as | |
|----------|----------------------------|---|
| `'atomic'` *(default)* | the whole new `cur` array | order-sensitive; the exact inverse of `jsono_merge_patch` |
| `'counts'` | `{"added": N, "removed": M}` | multiset cardinalities; element order has no effect |
| `'elements'` | `{"added": [...], "removed": [...]}` | multiset elements; element order has no effect |

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

The default `'atomic'` mode makes a round trip with `jsono_merge_patch`: `jsono_merge_patch(prev, jsono_diff(prev, cur)) = cur`. This holds for `cur` with no explicit `null` at an object key, because `null` is the merge-patch delete signal. The `'counts'` and `'elements'` modes are change *reports*, not patches that you can apply (a pure array reorder is a no-op there). The function uses the empty document for a `NULL` argument: `jsono_diff(NULL, cur)` is "all new", `jsono_diff(prev, NULL)` is "all removed" — useful with a `lag()` window over a stream of snapshots.

The representation of a number is part of its identity. So `1`, `1.0`, and `1e0` compare as not equal. If a serializer changes the form of a number, the snapshot reports a change — normalize upstream if this is important.

### `jsono_group_array` / `jsono_group_object`

These aggregates collect the values of a group into one JSONO array or one JSONO object. They are the JSONO equivalents of core JSON's `json_group_array` and `json_group_object`.

```sql
-- Collect values into an array, in the ORDER BY fold order:
SELECT to_json(jsono_group_array(j ORDER BY ts))
FROM (VALUES (jsono('1'), 1), (jsono('"x"'), 2), (jsono('[3]'), 3)) t(j, ts);
-- [1,"x",[3]]

-- Collect key -> value pairs into an object:
SELECT to_json(jsono_group_object(k, j))
FROM (VALUES ('b', jsono('2')), ('a', jsono('1'))) t(k, j);
-- {"a":1,"b":2}
```

`jsono_group_array` keeps a `NULL` input row as a JSON `null` element (the same as `json_group_array`). It returns SQL `NULL` for an empty group. `jsono_group_object` casts the key to `VARCHAR` and stops with a loud error for a `NULL` key. JSONO stores unique sorted keys. So `jsono_group_object` writes the keys in **sorted** order, and for duplicate keys it keeps the **last** value in fold order. This is different from `json_group_object`, which keeps each duplicate in input order. A shredded input is accepted (the function reconstructs the full document), and the result is always plain JSONO.

### Introspection

The introspection functions answer three different questions, and which question a function answers is the first thing to know about it: the JSON *document*, the bytes of this *row*, or the *type* that the column declares. Only the functions in the row group read data.

```sql
-- the document
jsono_type(value[, path])         -> VARCHAR    -- OBJECT/ARRAY/VARCHAR/BIGINT/UBIGINT/DOUBLE/BOOLEAN/NULL
jsono_keys(value[, path])         -> VARCHAR[]  -- object keys
jsono_array_length(value[, path]) -> BIGINT     -- array element count, NULL for a non-array

-- this row (reads the value, per row)
jsono_validate(value)             -> BOOLEAN    -- strict current-format validation
jsono_storage_size(value)         -> STRUCT     -- physical byte sizes (body blobs + shreds + total)
jsono_shred_manifest(value)       -> STRUCT[]   -- the shredded paths of THIS ROW, with their types

-- the type (answered at bind, reads no bytes)
jsono_layout_diagnose(value)      -> STRUCT     -- what this build sees the TYPE as, and why not JSONO when it is not
jsono_layout_lanes(value)         -> STRUCT[]   -- the shred lanes the TYPE declares, as logical paths
jsono_storage_type()              -> VARCHAR    -- DDL of the physical STRUCT backing plain JSONO
jsono_storage_type(<spec>)        -> VARCHAR    -- same, shredded: residual plus the spec's shred lanes
```

A lane that the *type* declares and a lane that a *row* used are different facts. On a healthy column, `jsono_shred_manifest` is a subset of `jsono_layout_lanes`, and an anti-join of the two on `(path, type)` answers which declared lanes a row did not use.

`jsono_type` and `jsono_keys` examine the shape of unknown JSON before you extract it with `jsono_transform`. The optional `path` is a constant JSONPath that points at a nested position. It has the same grammar as in `jsono_transform`, without wildcards. A missing path gives `NULL`.

```sql
SELECT jsono_keys(jsono('{"user":{"name":"a","age":1}}'), '$.user');
-- [age, name]
SELECT jsono_type(jsono('{"items":[1,2,3]}'), '$.items[0]');
-- BIGINT
```

`jsono_array_length` returns the number of elements in the array at the root or at `path`. A non-array value (an object, a scalar, or a JSON `null`), a missing path, and a SQL `NULL` each give `NULL`. Core DuckDB `json_array_length` is different: it returns `0` for a non-array. jsono uses the `NULL`-on-mismatch convention of `jsono_type` and `jsono_keys`.

```sql
SELECT jsono_array_length(jsono('[10,20,30]'));
-- 3
SELECT jsono_array_length(jsono('{"items":[1,2,3,4]}'), '$.items');
-- 4
SELECT jsono_array_length(jsono('{"a":1}'));
-- NULL
```

`jsono_validate` examines the current binary format contract: the header flags, sorted unique keys, the slots, the heaps, the navigation metadata, UTF-8 strings, and raw number text. It returns `false` for damaged physical blobs and `NULL` for SQL `NULL`. On a shredded value, it validates the residual encoding. It also compares the length of each non-`NULL` array lane with the element count in the residual skeleton, and a mismatch returns `false`. Each reader enforces this lockstep framing, so jsono must assert it. The *values* in a typed shred lane are DuckDB-native and are not jsono's to check.

`jsono_storage_size` reports the `slots`, `key_heap`, `string_heap`, `skips`, `lengths`, `nums`, `shreds`, and `total` byte counts. `shreds` is the shred payload of the row (0 for a plain value). `total` is the sum of the body blobs and the shreds. The function is a physical diagnostic and does not validate the value.

`jsono_storage_type` returns the DDL of the physical `STRUCT` that a JSONO value is — the form in each storage layer (DuckDB-native, DuckLake, plain Parquet). Use it to declare storage columns without hardcoded field names. A column of that exact shape binds to the jsono operations and to `to_json` structurally, with no cast.

```sql
SELECT jsono_storage_type();
-- STRUCT(jsono STRUCT("body$2" STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB, nums BLOB)))
```

With a shredding spec argument — the same JSON object string that the constructor's `shredding :=` takes — the function returns the *shredded* storage type with those lanes. This type is byte-identical to the type of a value that the same spec constructs. This is also the way to get the encoded physical field name of a lane, which the [Troubleshooting](#a-jsono-column-answers-null-or-refuses-reads) recovery needs.

```sql
SELECT jsono_storage_type('{"event_name": "VARCHAR", "$.commit.seq": "BIGINT"}');
-- STRUCT(jsono STRUCT("body$2" STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB, nums BLOB), "shreds$2" STRUCT("$jsono$set" BIGINT, "$jsono$spill$0" BIGINT, cdnmqrb9eg000sr5e4000 BIGINT, clr6arjkbtn62rb50000 VARCHAR)))
```

`jsono_shred_manifest` returns a `LIST<STRUCT(path VARCHAR, type VARCHAR)>`. It lists the shredded paths of the value — the paths whose values are in shred columns, not in the residual — each with the written shred type, in canonical (sorted) order. A plain value returns an empty list (nothing is shredded). A SQL `NULL` returns `NULL`. The function reads the row's own manifest record and, unlike the other readers, does not stop on a narrowed row. Use it together with `jsono_validate` to examine shredding, layout, or migration questions — which paths are shredded, or did a cast remove one. A shred value that did not fit its declared type stays in the residual, and the manifest correctly does not show it. So a short manifest does not mean "shredding did not occur".

The answers describe the *row*, not the type, and the two can put items in different orders. The manifest stores the element subfields of an object-array lane sorted by key, and `jsono_shred_manifest` reports them in that order. `jsono_layout_lanes` reports the element struct in the field order of the type. For paste-back into a `shredding :=` spec, use the latter: the field order of the element struct is part of the type.

```sql
SELECT jsono_shred_manifest(jsono('{"a":"x","b":1}', shredding := '{"a": "VARCHAR", "b": "BIGINT"}'));
-- [{'path': $.a, 'type': VARCHAR}, {'path': $.b, 'type': BIGINT}]
```

`jsono_layout_diagnose` tells what this build sees the argument's **type** as, and, if that type is not a JSONO value, which rule failed. The question is about the type, and the answer comes at bind time. The function never reads the bytes of the value, and a `NULL` value gets the same diagnosis as other values. The answer is a `STRUCT`:

| field | |
|---|---|
| `kind VARCHAR` | `plain`, `shredded`, `foreign` (a layout revision that this build does not read), or `not jsono` |
| `reason VARCHAR` | the refusal text from the grammar; `NULL` for the two readable kinds — so `reason IS NULL` *is* "this build can read it" |
| `body_revision UBIGINT` | the layout revision that the value was written under, after the anchor parsed one |
| `shreds_revision UBIGINT` | the same for the shred set; `NULL` for a plain value, which has no shred set |
| `spill_columns UBIGINT` | the number of spill bitmap columns in the type; `NULL` unless the grammar read the full shred set |

Each item that the grammar did not reach is `NULL`, not `0`. For the lane count, use `len(jsono_layout_lanes(value))`. `reason` is the same string that the refused reads throw.

The function exists for one specific failure mode. A type of the current revision that fails the layout grammar is classified as "not JSONO" *silently* — a refusal in recognition would break legal write paths that narrow types. Reads of such a column fall through to core json (`->>` reads `NULL`, `to_json` serializes the raw physical struct), and this is the function that says why. One miss is loud instead: a lane name that the codec did not make refuses JSON reads — see [Troubleshooting](#a-jsono-column-answers-null-or-refuses-reads).

```sql
SELECT jsono_layout_diagnose(jsono('{"a":1,"b":2}', shredding := '{"a": "BIGINT"}'));
-- {'kind': shredded, 'reason': NULL, 'body_revision': 2, 'shreds_revision': 2, 'spill_columns': 1}
SELECT jsono_layout_diagnose({'a': 1}).reason;
-- the value's single field is named 'a', not the layout anchor 'jsono'

-- a column that should be JSONO and is not, without parsing any prose
SELECT jsono_layout_diagnose(payload).kind <> 'shredded' AS lost_its_shreds FROM events LIMIT 1;
```

The function *diagnoses* a value of a foreign layout revision. It does not refuse it, although each other consumer of that value throws. A diagnostic that fails on the value that you diagnose is not a diagnostic.

`jsono_layout_lanes` lists the shred lanes of the argument's **type** as a `LIST<STRUCT(path VARCHAR, type VARCHAR)>`, in the type's own field order. Like `jsono_layout_diagnose`, the answer comes at bind time, and the function reads no bytes. The empty list means one thing only: a plain JSONO value, which really has no lanes. A foreign revision or a non-JSONO argument is a bind error, because an answer of `[]` there would look the same as the plain answer. `jsono_layout_diagnose` is the question that always answers.

You cannot read the lane set from the type text: a lane's field name is the base32hex encoding of its path (see [lane names](docs/jsono_format.md#lane-names)), which keeps case-colliding keys such as `gclid` / `GCLID` apart. This function is the way to read a shredded schema, and its answers are spec DSL: paste one back into `jsono(value, shredding := '{…}')` or `jsono_storage_type('{…}')` and you get the same lane.

One exception has no paste-back: an object-array lane whose element subfields differ only in letter case. The spec carries the element struct as SQL type text, and DuckDB's type parser folds the two spellings into a `Duplicate STRUCT type argument name` refusal. Such a lane itself is correct — there is just no spec spelling for it.

```sql
SELECT jsono_layout_lanes(jsono('{"URL":{"path":"/x"},"n":1}', shredding := '{"$.URL.path": "VARCHAR", "n": "BIGINT"}'));
-- [{'path': $.URL.path, 'type': VARCHAR}, {'path': $.n, 'type': BIGINT}]
SELECT jsono_layout_lanes(jsono('{"a":1}'));
-- []
```

The primary users of these helpers are the JSONO workflows and the tests. Their exact surface is less stable than `jsono_transform`, `jsono`, and `to_json`.

## Troubleshooting

### Is my read pushed down?

The plan shows if a read touches a shred lane (fast, and prunable per row group) or rebuilds the document. Run `EXPLAIN <your query>` (or `EXPLAIN ANALYZE` for timings) and read the `physical_plan`. If the only jsono operation over a shredded column is a read of its `.shreds$2.<lane>` column, the read is pushed down. If one of the residual or reconstruct tokens below is in the plan, the read is not pushed down.

- **Pushed** — the plan reads a typed shred column, prunable on its min/max statistics:
  - a bare `.shreds$2.<lane>` — a total shred, with no `COALESCE`. A projection shows it as a scan projection column (`ep.jsono.shreds$2.<lane>`). A filter or an aggregate shows it as `struct_extract_at(...shreds$2.<lane>...)`. `<lane>` is the encoded path. Compare it with the output of `jsono_layout_lanes`. Do not read it as text.
  - `COALESCE(struct_extract_at(...shreds...), jsono_extract_string(...))` — a partial shred. The plan reads the lane (prunable) and keeps a per-row residual fallback for diverted rows. The `divert_rate` from `jsono_shred_stats` shows how frequently that fallback operates.
- **Residual** — the plan parses the six body blobs for each row, with no pruning:
  - `jsono_extract_string` / `jsono_extract` — one extraction on a path that has no shred.
  - `__jsono_internal_project` — two or more residual extractions on the same column, fused into one document walk.
- **Reconstruct** — the plan rebuilds the full document (a 3–10× cost per row):
  - `__jsono_reconstruct` — explicit (a `jsono_transform` field that goes down into an array shred).
  - `jsono_overlay` / `__jsono_shred_patch` — a whole-value read as JSON (`to_json`, `::JSON`).
  - an anonymous `CAST(… AS STRUCT("jsono" …))` — `jsono_array_elements`, the collect aggregates, and `jsono_diff` when its lane-aware fast path declines (array shreds, or the two sides typed differently).

If a read that must be fast shows a residual or a reconstruct token, shred the exact path. The plan says if a read *can* push down; [`jsono_shred_stats`](#jsono_shred_stats) says how well an existing lane serves the read. Then apply the rules in [Get row-group pruning](#get-row-group-pruning): filter the typed shred in its own type, `ORDER BY` the hot leaf at write time, and use `union_by_name := true` across many Parquet files. Then the lane stays a bare `struct_extract_at` that can prune, not a `COALESCE`.

### A JSONO column answers `NULL` or refuses reads

When a JSONO column starts to answer `NULL` for paths that you know are there, or a read stops with a layout error, find the cause with [`jsono_layout_diagnose`](#introspection):

1. Run `SELECT jsono_layout_diagnose(col) FROM t LIMIT 1`. The answer comes at bind time and reads no bytes.
2. If `reason IS NULL`, this build reads the type, and the layout is not the problem. Examine the rows instead: `jsono_validate` and `jsono_shred_manifest` describe each row.
3. If `kind` is `'not jsono'` and the reason gives a lane name, DDL changed a shred field to a name that the codec did not make — see the recovery below.
4. If `kind` is `'foreign'`, the value was written under a layout revision that this build does not read — see [jsono_format.md §Layout revisions](docs/jsono_format.md#layout-revisions).
5. If the extension optimizer is disabled, reads fall through to core json: `->>` reads `NULL`, and `to_json` serializes the raw physical struct — see [Interop and limits](#interop-and-limits).

**Recovery from a renamed lane.** The data is intact: JSON reads refuse loudly with the grammar's reason, and passthrough (`SELECT *`, `COPY`, the raw `::VARCHAR` render) stays open. Use `ALTER TABLE … RENAME COLUMN` back to the encoded lane name — `jsono_storage_type(<spec>)` prints it. The rename restores each read in place. `ALTER TABLE … DROP COLUMN` recovers only a lane that held no shredded values: the shred manifest refuses the rows whose values were shredded into a dropped lane, and a drop of the only lane makes the struct silently stop being JSONO.

An error on one specific row, not on the type, usually comes from the shred manifest: a raw struct cast removed or retyped a shred column, and the read of the narrowed row refuses to return partial data — see [Shredded storage](#shredded-storage).

## Configuration

`JSONO_SHAPE_CACHE_SIZE` controls the size of the JSONO writer shape cache:

```bash
JSONO_SHAPE_CACHE_SIZE=16384 duckdb
```

Use power-of-two values for performance comparisons. The default is adjusted for the local benchmark workloads, but real data can be different.

## Interop and limits

**Shredded values and core json.** The jsono-named functions (`jsono_extract`, `jsono_extract_string`, `jsono_type`, `jsono_keys`, `jsono_validate`, `jsono_storage_size`), `jsono_group_merge`, the `::VARCHAR` cast, and the shredded→plain JSONO reconstruction are bind-correct: they accept a shredded value directly and operate correctly with the extension optimizer off. The operators and casts that go to the bundled core `json` extension — `->>`, `->`, `json_extract`, `::JSON`, and `to_json` — are correct over a shredded value only with the extension optimizer on (the default). The optimizer changes them to read the shreds and the residual. With the optimizer fully off (`PRAGMA disable_optimizer`), those core-routed operations bind into core json and serialize the physical fields. This is a structural DuckDB limit: core json owns the `STRUCT → JSON` cast, and the extension cannot intercept it.

**Chained operators over a constant literal.** Over a shredded **column**, `value -> 'a' ->> 'b'` becomes one deep-path shred read, the same as `value ->> '$.a.b'`. Over a **fully-constant `jsono({…})` literal**, the same chain returns `NULL` — DuckDB folds it before the extension optimizer runs. For a constant literal, use the JSONPath form `jsono({…}) ->> '$.a.b'`.

**Nesting depth.** The parser refuses JSON that has more than 1000 levels (512 on macOS, where the worker-thread stacks are too small for the deeper recursion). Sanitizer builds decrease the limit only where their larger frames make it necessary: 32 with AddressSanitizer (each platform), and 32 with UndefinedBehaviorSanitizer on macOS. The UndefinedBehaviorSanitizer build on Linux keeps the full 1000.

**Not implemented.** The current surface is intentionally small:

- There is no cast from JSONO to a scalar type (`jsono('42')::INTEGER` is unsupported). Project string values with `->>` / `jsono_extract_string`, or typed fields with `jsono_transform`.
- There is no cast from JSONO to an arbitrary `STRUCT`. Only the physical six-BLOB shape is interchangeable with JSONO. Field extraction goes through `jsono_transform`.
- There is no dedicated table function that changes a JSONO array or object into rows. Use `unnest(jsono_entries(...))` for flattened scalar leaves, or `unnest(jsono_array_elements(...))` for array elements as rows.
- Object key order is not kept: keys are stored and written in sorted byte order.

## Build from source

A local build is the alternative to the community build. Use it for development, or for a platform that has no community build. A local build is unsigned, so DuckDB must permit unsigned extensions.

Prerequisites:

- [`uv`](https://docs.astral.sh/uv/) — controls the build and the Python tools.
- A C++17 toolchain and CMake.
- [Ninja](https://ninja-build.org/) — the build uses the Ninja generator by default. Use `GEN=make` to use Make.
- `ccache` (optional) — makes incremental builds faster.

Build and load (CLI):

1. Build the release extension:

   ```bash
   uv run make release
   ```

2. Start DuckDB with unsigned extensions permitted:

   ```bash
   ./build/release/duckdb -unsigned
   ```

3. Load the extension:

   ```sql
   LOAD './build/release/extension/jsono/jsono.duckdb_extension';
   ```

Python:

```python
import duckdb

con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
con.execute("LOAD './build/release/extension/jsono/jsono.duckdb_extension'")
```

## Development

Run the full local pre-PR gate with one command. The gate does the release build and its tests, the assert-enabled build with its tests and the constructor matrix, and the format check:

```bash
uv run make verify
```

Or run the gates one at a time:

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

The DuckDB submodule is in `duckdb/`. Update it only with explicit submodule commands.

## Benchmarks

The `bench/` directory is the harness for comparisons between `jsono` versions and the core DuckDB `json` baseline. Run a smoke benchmark:

```bash
uv run --frozen python bench/bench.py --filter group_merge/1k --runs 1
```

[bench/README.md](bench/README.md) documents the operation set, the version and core-json comparisons, the field-sample scenarios, and the result-reading contract. [bench/PROFILING.md](bench/PROFILING.md) documents profiling.

## Contributing

Contributions and feedback are welcome. Please:

1. Open an issue first to discuss the changes.
2. Add or update SQLLogic tests in `test/sql/` for new behavior.
3. Run `uv run make verify` before you send a pull request. The command does the release build and its tests, the assert-enabled build with its tests and the constructor matrix, and the format check. CI also runs `uv run make tidy-check`, which requires a local clang-tidy and compile-database setup.

See [GitHub Issues](https://github.com/Flamefork/duckdb-jsono/issues) for current tasks and feature requests.

## License

MIT. See [LICENSE](LICENSE).

For third-party components and their licenses, see [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

---
*This extension is based on the [DuckDB Extension Template](https://github.com/duckdb/extension-template).*
