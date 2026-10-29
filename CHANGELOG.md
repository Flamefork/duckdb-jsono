# Changelog

One entry per tag. Each entry records the SQL-surface changes, whether the
layout revisions (`body$N` / `shreds$M`) moved, and — for a moved revision —
the migration note.

## v0.1.4 — 2026-10-29

A correctness release for `jsono_group_merge_max` / `jsono_group_merge_min`,
built against DuckDB `v1.5.6`. Values written by `v0.1.0`–`v0.1.3` stay
readable.

- SQL surface: no function, signature, or return type changed. The keyed
  group merges change behavior:
  - A path that changes kind between rows (an object in one row, a scalar or
    an array in another) no longer fails. The result is the RFC 7396 fold in
    key order, increasing for `_max` and decreasing for `_min`: a scalar or an
    array erases what the rows applied before it wrote at its path and under
    it, and an object after it starts fresh.
  - Rows with an equal `order_key` make one shared patch: their objects merge,
    a scalar or an array wins over an object at the same path, and a fixed
    order of the values selects between two scalars or arrays. Before, each
    leaf was resolved on its own, which could give a result that no order of
    the rows gives.
  - Rows with a SQL `NULL` `order_key` are applied first in both functions:
    a row with a key overwrites them where it writes, and their other members
    stay.
- Format: no revision change. The layout stays `body$2` / `shreds$2`.
- Migration: none.

## v0.1.3 — 2026-10-24

A correctness release built against DuckDB `v1.5.6` (was `v1.5.5`). Values
written by `v0.1.0`–`v0.1.2` stay readable.

- SQL surface: no change. No function, signature, return type, or documented
  option changed.
- Correctness:
  - A path read that lands on an empty array (`->`, `jsono_extract`) now
    returns `[]` instead of failing with `malformed JSONO: container span
    missing`, in plain and shredded columns alike.
  - A shredded value cast to the plain JSONO storage type and compared with a
    constant (`=`, `IN`) now compares documents. DuckDB's expression rewriter
    used to move the cast onto the constant, so the same document in two
    encodings could compare as different.
- Documentation: the README now names when a core json read over a fully
  constant shredded value returns `NULL`, and the jsono-native workaround.
- Format: no revision change. The layout stays `body$2` / `shreds$2`.
- Migration: none.

## v0.1.2 — 2026-08-27

A correctness and read-path performance release. Values written by `v0.1.0`
and `v0.1.1` stay readable.

- SQL surface: no change. No function, signature, return type, or documented
  option changed.
- Correctness:
  - Path reads from a raw-cast narrowed row now fail only if a removed shred
    can change the requested result.
  - Windows builds now use canonical unsigned-byte order for mixed ASCII and
    UTF-8 keys in `MAP` construction and keyed merge paths.
  - Full-manifest consumers now reject unsorted or duplicate entries. Point
    reads still reject malformed framing and loss that affects their result.
- Performance:
  - `jsono_keys(value)` and `jsono_array_length(value[, path])` now read
    shredded values without reconstructing the full document.
  - Manifest checks and reshredding now use sorted walks instead of repeated
    lane scans.
  - DOM shape caches now allocate memory only after the first cache insertion.
- Format: no revision change. The layout stays `body$2` / `shreds$2`.
  Manifest framing, type codes, marker values, and spill-bit numbering do not
  change. The MSVC `MAP` writer can produce different residual bytes because
  it now obeys the existing canonical key order.
- Migration: none.

## v0.1.1 — 2026-08-06

An internal cleanup release. Values written by `v0.1.0` stay readable.

- SQL surface: no change. No function was added, removed, or changed its
  signature or its return type.
- Format: no change. The layout revisions stay `body$2` / `shreds$2`, and the
  bytes inside the residual blobs are the same as in `v0.1.0`.
- Migration: none.

## v0.1.0 — 2026-07-31

The first tagged release.

- SQL surface: the initial public set — see the
  [Function reference](README.md#function-reference).
- Format: layout revisions `body$2` / `shreds$2`. The two field names version
  the format whole — `body$N` covers the byte encoding inside the residual
  blobs, there is no separate byte version. This release is the storage
  compatibility anchor: from here, a newer release reads the values that this
  release wrote.
- Migration: values written by pre-release builds (the field names
  `body` / `shreds`, or `body$1` / `shreds$1`) are not readable by this
  release. The way forward for each pre-release shape is in
  [docs/jsono_format.md § Revision history](docs/jsono_format.md#revision-history).
