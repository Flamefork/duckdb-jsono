# Changelog

One entry per tag. Each entry records the SQL-surface changes, whether the
layout revisions (`body$N` / `shreds$M`) moved, and — for a moved revision —
the migration note.

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
