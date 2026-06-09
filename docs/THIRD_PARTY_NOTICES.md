# Third-Party Notices

This project is licensed under MIT. The following third-party components are
included in the source tree or in release binaries.

Source distributions should include this file and the referenced or embedded
license notices. Binary distributions should include this file together with
the referenced license files from the build tree.

## Source Tree

### DuckDB Extension Template

- Upstream: https://github.com/duckdb/extension-template
- Files: repository build and extension scaffolding
- License: MIT

Copyright 2018-2025 Stichting DuckDB Foundation.

### string-view-lite

- Upstream: https://github.com/martinmoene/string-view-lite
- Version: v1.8.0
- File: `third_party/string-view-lite/string_view.hpp` (vendored single header)
- License: Boost Software License 1.0 (BSL-1.0)
- License text: embedded in the header

Copyright 2017-2020 Martin Moene.

## Release Binaries

DuckDB loadable extension binaries are linked with DuckDB build artifacts and
may include DuckDB third-party components. Notices for DuckDB itself are in the
DuckDB submodule.

### DuckDB

- Upstream: https://github.com/duckdb/duckdb
- Version used by this repository: submodule `duckdb`
- License: MIT
- License file: `duckdb/LICENSE`

Copyright 2018-2025 Stichting DuckDB Foundation.

### RE2

- Upstream: https://github.com/google/re2
- Source in DuckDB: `duckdb/third_party/re2`
- License: BSD-style
- License file: `duckdb/third_party/re2/LICENSE`

Copyright 2009 The RE2 Authors.

### yyjson

- Upstream: https://github.com/ibireme/yyjson
- Source in DuckDB: `duckdb/third_party/yyjson` (not vendored here; the extension
  links yyjson symbols exported by DuckDB)
- License: MIT
- License file: `duckdb/third_party/yyjson/LICENSE`

Copyright 2020 YaoYuan.
