# Benchmark harness

This directory serves two purposes:

- compare the current `jsono` build with other local `jsono` builds;
- compare comparable `jsono` operations against the core DuckDB `json` extension as a baseline.

## Quick start

```bash
uv run --frozen python bench/bench.py --filter merge_patch --runs 1
```

`bench.py` generates any missing parquet files in `bench/data/`, runs a sanity check
plus a correctness gate, and launches the benchmark.
The result is written to `bench/results/latest.json`.

The core DuckDB `json` baseline (target label `json`) is always run for the operations
that have a comparable core-json form; it loads via `INSTALL json; LOAD json` and needs
no extension file.

## Comparing jsono versions

```bash
uv run --frozen python bench/run_benchmarks.py \
  --target old=/path/to/old/jsono.duckdb_extension \
  --filter transform/10k \
  --runs 5
```

By default the `current` target from `build/release/extension/jsono/jsono.duckdb_extension` is always run.
The `--target LABEL=PATH` flag adds another `jsono` build; it can be repeated.

## Thread modes

Every case runs once per DuckDB thread count so single-thread vs parallel scaling is
visible side by side. The default is `--threads 1,8`; pass any comma-separated list to
override (e.g. `--threads 1` for a single mode, or `--threads 1,4,8`).

```bash
uv run --frozen python bench/run_benchmarks.py --filter group_merge_jsono --threads 1,8
```

The results table pivots `min_ms` into one column per thread count plus a
`speedup_t{lo}_t{hi}` column (slowest÷fastest thread count): values above `1` got faster
with more threads, below `1` got slower with more threads. The `scaling_note` column marks
`small_case_overhead` when the case has fewer than 64 DuckDB-sized 2048-row chunks, where
fixed parallel materialization cost can dominate the operation itself.

For three-way regression triage of a `before`/`after`/`current` results file, `cmp3.py` prints a pivot of
`min_ms` per `operation/scenario/size` with the `after/before`, `current/after` and `current/before` deltas,
flagging cases where `current` moved by ≥3 %:

```bash
uv run --frozen python bench/cmp3.py bench/results/latest.json
```

It pairs with the git-ignored `bench/optimization-log/jsono_review_regression_map.md` for regression triage.
For CPU flamegraphs, `profile_driver.py` runs one selected benchmark case in a tight loop using the same
case registry; see `bench/PROFILING.md`.

## Operation set

JSONO-only operations:

- `parse`: `jsono(json_text)`;
- `parse_shred`: `jsono(json_text, shredding := spec)` — end-to-end shred-from-text ingest;
- `parse_copy`: `parse` + writing the result to Parquet (end-to-end path with storage size);
- `scan_text`: diagnostic lower bound on scan/output cost; compare against `parse` when optimizing JSONO creation;
- `parse_struct`: `jsono(STRUCT)` on a pre-materialized typed STRUCT input;
- `parse_struct_roundtrip`: control path `jsono(to_json(STRUCT)::VARCHAR)` for the same typed input;
- `parse_struct_json_roundtrip`: control path `jsono(to_json(STRUCT))` for the same typed input;
- `object_jsono`: `jsono(STRUCT)` on a scalar object payload;
- `object_json`: `json_object(key, value, ...)` on the same scalar object payload for comparison;
- `validate`: `jsono_validate` on a pre-materialized `jsono` input, excluding parse cost;
- `storage_size`: `jsono_storage_size` on a pre-materialized `jsono` input, excluding parse cost;
- `transform`: `jsono_transform`;
- `group_merge`: `to_json(jsono_group_merge(...))`;
- `group_merge_jsono`: `jsono_group_merge`, the JSONO-native path.
- `group_merge_keyed_max_jsono` / `group_merge_keyed_min_jsono`:
  `jsono_group_merge_max/min(value, row(k_ts, k_secondary))`, the order-independent
  keyed JSONO-native path used to model wide keyed group merges.
- `group_merge_keyed_pair_jsono`: one `jsono_group_merge_min(wide_payload, key)`
  and one `jsono_group_merge_max(detail_payload, key)` in the same `GROUP BY`,
  matching a two-keyed-aggregate wide/detail shape.

The `keyed_pair` generator writes explicit Parquet row groups. For parallel
triage, verify row groups with `parquet_metadata()`; the benchmark summary's
`chunks` column is DuckDB 2048-row vector chunks, not Parquet row groups.

Scenarios may set a `shredding` spec: the materialized `jsono` input (and the `parse_shred`
timed call) is then shredded with it. Scenario names carry `shred` so they are easy to filter.

Operations paired with a core DuckDB `json` baseline (target `json`):

- `merge_patch`: `jsono_merge_patch(jsono(base), jsono(STRUCT patch))` vs
  `json_merge_patch(base, patch)` — a small constant patch object merged into the wide
  `json_nested` base. The `jsono` base input is materialized before the timed section;
  the patch is a typed constant JSONO value. The patch has no null members, so the jsono
  RFC 7396 null-stripping does not diverge from core `json_merge_patch`.
- `entries`: `jsono_entries(jsono(obj), 'dotted')` vs `json_transform(obj, '"map(string, string)"').map_entries()`
  — flatten an object to a key/value list. Run over `json_flat` (already leaf-flat), so the
  jsono leaf-flatten matches the core shallow map. The `jsono` input is materialized before the timed section.
- `extract`: `jsono_transform(jsono(obj), spec)` vs `json_transform(obj, spec)` with the same
  `VARCHAR`/`BIGINT` schema over top-level `json_nested` paths. The `jsono` input is materialized
  before the timed section.
- `extract_jsono`: `jsono_extract(jsono(obj), path)` vs `json_extract(obj, path)` for one constant
  path. This covers the JSONO-returning executor used by `jsono_extract`, `json_extract(JSONO, ...)`,
  and `->`; aliases are not benchmarked separately.
- `extract_string`: `jsono_extract_string(jsono(obj), path)` vs `json_extract_string(obj, path)` for
  one constant path. This covers the VARCHAR-returning executor used by `jsono_extract_string` and
  `->>`; aliases are not benchmarked separately.
- `project_paths`: filterless multi-extract projection `SELECT j->>'p1', ..., j->>'pN'` (no WHERE)
  vs the same `->>` list over core JSON text. The `jsono` input is materialized before the timed
  section; the shredded variant is JSONO-only.
- `prune_filter`: row-group pruning over a shredded, `event_ts`-clustered Parquet (written untimed in
  prepare). The timed query is a selective band filter that scans that Parquet directly; the
  `synthetic_shredded`/`synthetic_native` pair filters the typed shred leaf vs the plain control column
  and prunes identically. JSONO-only; run with `--row-groups` (see `bench/PROFILING.md`) to print
  `scanned/total` row groups next to the timing.

## Correctness gate

Before any timing, `sanity_checks.py` asserts each `jsono` operation produces output semantically
equal to its core `json` baseline over the full dataset, failing loudly with a sample diff on any
mismatch:

- `merge_patch`: per-row `to_json(jsono_merge_patch(jsono(base), jsono(STRUCT patch)))` vs
  `to_json(jsono(json_merge_patch(...)::VARCHAR))`;
- `entries`: per-row `list_sort` of the key/value entries, comparing the (key, value) set order-independently;
- `extract`: per-row struct compared field-by-field (`IS DISTINCT FROM`).
- `extract_jsono`: per-row extracted JSON compared as core JSON;
- `extract_string`: per-row extracted string compared directly.

For JSONO-native operations the input `JSONO` is materialized before the timed section.
If you need the end-to-end path with parse cost, combine the matching `parse` scenario with the JSONO operation.
For `parse_struct` field-sample scenarios the typed `STRUCT` is also materialized before the timed section:
the timed part measures the JSONO constructor itself, not DuckDB `json_transform`.

## Field-sample scenarios

Field-sample benchmarks use local gitignored parquet and rules files under
`bench/data/field_sample/` and are enabled with the `--include-field-sample` flag:

```bash
# creating JSONO from field-sample nested JSON
uv run --frozen python bench/bench.py --include-field-sample --filter field_sample_nested --runs 3

# parse case only
uv run --frozen python bench/bench.py --include-field-sample --filter parse/245760/field_sample_nested --runs 5

# end-to-end path with writing to Parquet
uv run --frozen python bench/bench.py --include-field-sample --filter parse_copy/245760/field_sample_nested --runs 1
```

## Reading results

The result of each run is written to `bench/results/latest.json`. Generated
data and results are git-ignored (`bench/results/baseline.json` is an intentional
exception for regression comparisons).

- Compare relative values from a single run, not absolute values across different sessions.
- The stored baseline drifts with machine state — use it as a regression hint, not as absolute truth.

## Profiling

Use `run_benchmarks.py --profile` for DuckDB operator profiles. This path works for both `current`
JSONO cases and core `json` baselines:

```bash
uv run --frozen python bench/run_benchmarks.py --profile --filter current/extract_jsono/10k/nested_top_key --runs 1 --threads 1
uv run --frozen python bench/run_benchmarks.py --profile --filter json/extract_jsono/10k/nested_top_key --runs 1 --threads 1
```

Use `profile_driver.py` for external CPU profilers. It intentionally loops only the `current` JSONO
target, so the filter must resolve to exactly one JSONO case:

```bash
uv run --frozen python bench/profile_driver.py build/release/extension/jsono/jsono.duckdb_extension extract_jsono/10k/nested_top_key 150 1
```

## Useful commands

```bash
uv run --frozen python bench/run_benchmarks.py --list
uv run --frozen python bench/generate_data.py --kind events --size 10k
uv run --frozen python bench/generate_data.py --kind wide_flat
uv run --frozen python bench/generate_data.py --kind keyed_pair
uv run --frozen python bench/compare_results.py --save-baseline
uv run --frozen python bench/compare_results.py
uv run --frozen python bench/run_benchmarks.py --profile --filter current/transform/1k/flat_core
```
