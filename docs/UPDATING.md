# Updating the DuckDB target

This extension is built against DuckDB's internal C++ API, which is not stable
across releases. When moving to a newer DuckDB, update the pinned version in
each place that participates in local builds or CI:

- Bump submodules
  - `duckdb/` → the target tagged release.
  - `extension-ci-tools/` → the patch branch matching that release; it can trail
    the DuckDB tag, so fall back to the newest patch branch that exists (e.g.
    DuckDB `v1.5.5` pairs with `extension-ci-tools` `v1.5.5`).
- Bump versions in `.github/workflows/MainDistributionPipeline.yml`
  - reusable workflow refs for `duckdb-stable-build` and `code-quality-check`.
  - `duckdb_version` and `ci_tools_version` inputs in both jobs.
- Pin the Python package to the same exact DuckDB version in `pyproject.toml`
  and refresh `uv.lock` in the same change.

`.github/workflows/NextDuckDB.yml` needs no bump: it tracks DuckDB `main` on
purpose, and a red run there is the signal that the next release will need the
work below.

Update `duckdb/` only via explicit `git submodule` commands.

## What a published extension adds

The extension is distributed through
[DuckDB Community Extensions](https://duckdb.org/community_extensions/), where
the descriptor `extensions/jsono/description.yml` pins one source ref and the
community CI builds it against the current stable DuckDB.

- If the extension compiles against the upcoming DuckDB, **no action is needed**
  for the release itself: every community extension is rebuilt as part of a
  DuckDB release.
- After a DuckDB bump lands here, open a pull request against
  `duckdb/community-extensions` that moves `repo.ref` to the new commit or tag.
- If the upcoming DuckDB needs source changes, they must be ready *before* the
  release: DuckDB starts a ~2-week feature freeze with a `vX.Y-<codename>`
  branch, and the extension needs a branch of the same name carrying the fixes,
  pointed at by `repo.ref_next` in the descriptor. Miss that window and the
  extension is absent on release day, and stays out of the automatic rebuild
  until it compiles again.

## When the build breaks after a bump

A DuckDB bump can break the extension because the internal C++ API changed.
DuckDB does not publish a dedicated changelog for these changes; to figure out
what moved:

- DuckDB [release notes](https://github.com/duckdb/duckdb/releases)
- DuckDB [core extension patches](https://github.com/duckdb/duckdb/commits/main/.github/patches/extensions)
- the git history of the relevant DuckDB C++ header
