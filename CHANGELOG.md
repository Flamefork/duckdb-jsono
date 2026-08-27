# Changelog

One entry per tag. Each entry records the SQL-surface changes, whether the
layout revisions (`body$N` / `shreds$M`) moved, and — for a moved revision —
the migration note.

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
