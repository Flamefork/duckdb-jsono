# Updating the DuckDB target

This extension is built against DuckDB's internal C++ API, which is not stable
across releases. When moving to a newer DuckDB, update the pinned version in
each place that participates in local builds or CI:

- Bump submodules
  - `duckdb/` → the target tagged release.
  - `extension-ci-tools/` → the branch matching that release (for DuckDB `vX.Y.Z`
    there is a branch `vX.Y.Z` in `extension-ci-tools`).
- Bump versions in `.github/workflows/MainDistributionPipeline.yml`
  - reusable workflow refs for `duckdb-stable-build` and `code-quality-check`.
  - `duckdb_version` and `ci_tools_version` inputs in both jobs.

Update `duckdb/` only via explicit `git submodule` commands.

## When the build breaks after a bump

A DuckDB bump can break the extension because the internal C++ API changed.
DuckDB does not publish a dedicated changelog for these changes; to figure out
what moved:

- DuckDB [release notes](https://github.com/duckdb/duckdb/releases)
- DuckDB [core extension patches](https://github.com/duckdb/duckdb/commits/main/.github/patches/extensions)
- the git history of the relevant DuckDB C++ header
