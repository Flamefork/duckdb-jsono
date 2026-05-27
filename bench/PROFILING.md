# Profiling

`run_benchmarks.py --profile` enables DuckDB JSON profiling for the selected cases and writes files to
`bench/results/profiles/<case>/query_profile.json`.

Example:

```bash
uv run --frozen python bench/run_benchmarks.py \
  --profile \
  --filter current/transform/1k/flat_core
```

In profile mode the number of runs is automatically reduced to `1` unless `--runs` is set explicitly.
The JSONO input for `jsono` is prepared before the profiled SQL, so the profile reflects the operation
itself rather than `jsono`.

For a CPU flamegraph it is more convenient to run the same filter through an external profiler (`samply`, Instruments, perf).
`analyze_profile.py` parses `samply` JSON when the profile is recorded with presymbolization.
By default it aggregates self-time across all execution threads — the honest view under
DuckDB's intra-query parallelism, where a single thread misleads (the main thread mixes in
coordination overhead, a lone worker over-weights its slice). Pass `--thread <name>` for a
single-thread per-row view; `--min-thread-samples` sets the idle-thread floor (default 50).

`profile_driver.py` is the standalone `samply` driver: it resolves one `jsono` benchmark case through the
same registry as `run_benchmarks.py`, prepares the input table once, then runs that case's timed SQL in a
tight loop so the flamegraph is dominated by the extension's C++ hotspots instead of DuckDB driver overhead.

```bash
samply record -- uv run --frozen python bench/profile_driver.py \
  build/release/extension/jsono/jsono.duckdb_extension \
  group_merge_jsono/100k/many_groups_ignore_nulls \
  150 8
```

For local field-sample cases, pass the same flag used by the benchmark runner:

```bash
samply record -- uv run --frozen python bench/profile_driver.py \
  --include-field-sample \
  build/release/extension/jsono/jsono.duckdb_extension \
  merge_patch/245760/retail_sample_nested \
  80 8
```

Arguments are `<extension_path> <filter> [iterations=150] [threads]`. The filter uses the same substring
matching as `run_benchmarks.py`, but it must match exactly one `jsono` case; ambiguous filters fail with the
matched case list. It pairs with the git-ignored `bench/optimization-log/jsono_review_regression_map.md` for
regression triage.
