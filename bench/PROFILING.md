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

## Row-group pruning

`run_benchmarks.py --row-groups` prints the parquet row groups a timed query actually scanned next to
the timing, so shred-leaf filter pushdown is measurable directly rather than inferred from wall-clock:

```bash
uv run --frozen python bench/run_benchmarks.py --row-groups --threads 1 --filter prune_filter
# current/prune_filter/100k/synthetic_shredded [t1]... 0.7ms, …, row_groups 1/9
# current/prune_filter/100k/synthetic_native   [t1]... 0.4ms, …, row_groups 1/9
```

`row_groups <scanned>/<total>` is `OPERATOR_ROW_GROUPS_SCANNED` / `OPERATOR_TOTAL_ROW_GROUPS_TO_SCAN`
on the scan operator. `scanned < total` means the filter pruned row groups; `scanned == total` means
no pruning; `—` means the timed query had no direct table scan (e.g. it filtered an in-memory temp
table that was answered without a scan node). The two `prune_filter` scenarios write one shredded,
`event_ts`-clustered Parquet (untimed) and time a selective band filter over it — `synthetic_shredded`
filters the typed shred leaf, `synthetic_native` the plain control column, and they prune identically,
proving the shred carries usable per-row-group statistics. `--row-groups` also works for any other case;
a scan of an in-memory temp table reports `N/N` (one native row group, no pruning).

The metric is persisted per result in `latest.json` as `row_groups: [{operator, scanned, total}]`.

### Manual recipe

These two operator metrics (populated by the parquet reader and the native table scan since DuckDB
**1.5.4**) are **not** in the default/`detailed` profiling set, so they must be requested explicitly,
and they do **not** appear in the `query_tree` / `EXPLAIN ANALYZE` text box — only in the JSON profile:

```sql
PRAGMA enable_profiling='json';
PRAGMA custom_profiling_settings='{"OPERATOR_NAME":"true","OPERATOR_TYPE":"true","OPERATOR_ROW_GROUPS_SCANNED":"true","OPERATOR_TOTAL_ROW_GROUPS_TO_SCAN":"true"}';
PRAGMA profiling_output='/tmp/prof.json';
-- the query under test, e.g.
SELECT count(*) FROM 'shredded_clustered.parquet'
WHERE CAST(t->>'$.event_ts' AS BIGINT) BETWEEN 1717000000 AND 1717120000;
PRAGMA disable_profiling;
```

Then walk the JSON tree for nodes with `operator_type == "TABLE_SCAN"` and read
`operator_row_groups_scanned` / `operator_total_row_groups_to_scan` (`parse_row_group_scans` in
`run_benchmarks.py` does exactly this). Caveats: a no-filter `SELECT count(*)` over Parquet is answered
from file metadata with no scan operator (no metric); pruning only works through typed shred leaves
(a residual `->>` path reads an opaque blob with no per-row-group stats and never prunes); and the data
must be clustered on the leaf (`ORDER BY` on write) for row-group ranges to be disjoint.
