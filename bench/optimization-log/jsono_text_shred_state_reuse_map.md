# Text-shred state reuse optimization map

Goal: test whether retaining the yyjson allocator and direct DOM writer state across DuckDB chunks reduces end-to-end `jsono(text, shredding := spec)` cost on the production-shaped field sample without changing output or taxing adjacent paths.

## Methodology

- The baseline and candidate used the same local data, machine, session, commands, release build type, and one DuckDB thread.
- The primary metric was `parse_shred/245760/field_sample_nested`, the existing end-to-end text-to-shredded case that exercises `JsonoShredFromTextExecute`.
- Controls were all existing `reshred/100k/` cases, `parse/245760/field_sample_nested`, `parse_struct/1M/typed_object_array`, and `merge_patch/100k/wide_shredded_base_nested_shred_patch`.
- The primary and controls used seven timed runs in both target orders. RSS used a separate process and three timed runs. DuckDB profiling confirmed query shape; presymbolicated `samply` profiles inspected native executor, writer, parser, allocator, and destructor costs.
- Keep required identical primary checksum tuples in both orders, at least 5% primary median improvement in AB and BA, no control regression above 5% in either order, candidate peak RSS no more than 10% above baseline, allocator/destructor self share below 1%, no allocator/destructor inclusive-share increase of at least one percentage point, and the expected one-pass native symbols.
- A failed timing was repeated once in a fresh process. A repeated failure rejected the implementation.
- Timing deltas are `(candidate / baseline) - 1`; negative is faster. AB ran candidate then baseline. BA ran baseline then candidate.

## Current Baseline

- Date: 2026-08-26. The frozen baseline is characterization commit `8983a0a`; plan 059 is present as kept commit `1f24aa8`.
- Baseline extension: `bench/results/workers/062-baseline/jsono.duckdb_extension`, SHA-256 `fbddc1aa2806b2b4b59fc3dd77aa0b6dd5a5079b0bbb675afebd5d0e4c44ab90`.
- Source identity: `bench/data/field_sample/events_nested.parquet` resolves to `/Users/flamefork/Development/ff/duckdb-json-tools/bench/data/real_prod/events_nested.parquet` from sibling `duckdb-json-tools` commit `5e944ca5a1aadd3a6ce1243b7b5d70213027f2f6`.
- Live source metadata: 245,760 rows, two 122,880-row groups, ZSTD for every column chunk.
- The ignored `events_nested_jsono.parquet` materialization is not the primary input. It has eight row groups (`7 × 32,768 + 16,384`) and is used by other field-sample cases.
- Independent seven-run baseline medians: primary 3884.16 ms; reshred same-set 40.64 ms, widening 106.77 ms, narrowing 166.14 ms; plain parse 3725.33 ms; typed object array 404.87 ms; merge patch 610.03 ms.

## Recommendation

Reject H2. The candidate did not reach the 5% primary improvement in either repeated target order: AB regressed by 2.41%, while BA improved by only 4.51%. Widening reshred also regressed by 30.83% in repeated AB and 31.30% in BA. The experimental implementation and its revert were collapsed out of final history; the checksum characterization remains.

The native profile explains the missing primary gain. Before the patch, `dyn_malloc` plus `dyn_free` occupied only 0.075% of self samples and 0.154% inclusive. The candidate reduced them to 0.021% self and 0.026% inclusive, but that work was far below the 5% keep threshold. Parser and writer shares stayed effectively unchanged. Retaining these direct objects also broadens their lifetime to every executor that uses `ShredLocalState`; the repeatable widening control regression makes that broader cost unacceptable. Its lower-level mechanism was not needed to decide the already-failed hypothesis and was not pursued with another patch.

## Relation To Existing Ideas

Plan 059 made `DomDirectState` shape-cache allocation lazy and removed per-slot reservation. That prerequisite prevented the old eager shape-cache memory defect, but it did not make state retention profitable here. The normal text parser retains the same state in its text-only local state; `ShredLocalState` is broader because reshred paths also use it.

Keyed group-merge's `JsonoShredFromLayout` fallback has no existing benchmark witness. This experiment added no scenario for it.

## Hypothesis Map

| ID | Idea | Evidence | Status | Primary result | Worst control | Complexity | Correctness surface | Decision |
|---|---|---|---|---|---|---|---|---|
| H2 | Retain direct `YyjsonAllocator` and `DomDirectState` fields in `ShredLocalState` | Baseline allocator/destructor samples were only 0.075% self and 0.154% inclusive | rejected and reverted | repeated AB +2.41%; repeated BA −4.51% | widening +30.83% AB / +31.30% BA | one implementation commit; `+4/−5` in one file | checksums passed; broad local-state timing failed | reject |

## Experiment Log

### Benchmark characterization and baseline

`bench/run_benchmarks.py` now reuses `collect_struct_constructor_checksum` for `parse_shred`. The focused harness test proves that `run_benchmarks` stores the returned `{rows, hash_sum, json_bytes}` tuple. No helper, operation, or scenario was added.

Commands:

```text
uv run --frozen python -m unittest bench.test_bench
uv run --frozen python bench/run_benchmarks.py --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 3 --threads 1 --output bench/results/062-characterization.json
ccache -M 30G
uv run --frozen make release
mkdir -p bench/results/workers/062-baseline bench/results/samply
cp build/release/extension/jsono/jsono.duckdb_extension bench/results/workers/062-baseline/jsono.duckdb_extension
shasum -a 256 bench/results/workers/062-baseline/jsono.duckdb_extension
uv run --frozen python bench/run_benchmarks.py --include-field-sample --list --filter parse_shred/245760/field_sample_nested
uv run --frozen python bench/run_benchmarks.py --list --filter reshred/100k/
uv run --frozen python bench/run_benchmarks.py --list --filter parse_struct/1M/typed_object_array
uv run --frozen python bench/run_benchmarks.py --list --filter merge_patch/100k/wide_shredded_base_nested_shred_patch
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-baseline-primary.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --filter reshred/100k/ --runs 7 --threads 1 --output bench/results/062-baseline-reshred.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-baseline-parse.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --filter parse_struct/1M/typed_object_array --runs 7 --threads 1 --output bench/results/062-baseline-parse-struct.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --filter merge_patch/100k/wide_shredded_base_nested_shred_patch --runs 7 --threads 1 --output bench/results/062-baseline-merge-patch.json
/usr/bin/time -l uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 3 --threads 1 --output bench/results/062-rss-baseline.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --profile --runs 1 --threads 1 --output bench/results/062-duckdb-profile-baseline.json
samply record --save-only --unstable-presymbolicate --output bench/results/samply/062-baseline-text-shred.json.gz -- uv run --frozen python bench/profile_driver.py --include-field-sample bench/results/workers/062-baseline/jsono.duckdb_extension parse_shred/245760/field_sample_nested 5 1
uv run --frozen python bench/analyze_profile.py bench/results/samply/062-baseline-text-shred.json.gz --top 80
```

Characterization checksum and every primary checksum in both target orders were identical:

```text
rows       245760
hash_sum   2265287782386176543464656
json_bytes 1381819034
```

The DuckDB profile showed one `CREATE TABLE AS` projection over the source Parquet with a direct `jsono(event_properties::VARCHAR, shredding := four-scalar-spec)` expression. The four scalar object-key paths satisfy the bind-time one-pass eligibility. The native profile contained `JsonoShredFromTextExecute`, `EmitDomRowDirect`, and `yyjson_read_opts`.

### Candidate implementation and correctness

The experimental implementation added direct `parser` and `dom` fields to the existing `ShredLocalState`, removed the executor-local pair, and used `lstate.parser` and `lstate.dom` directly. It changed no control flow or adjacent state.

```text
src/jsono_shred.cpp | 9 ++++-----
1 file changed, 4 insertions(+), 5 deletions(-)
numstat: 4  5  src/jsono_shred.cpp
```

Commands:

```text
rg -n "YyjsonAllocator parser|DomDirectState dom" src/jsono_shred.cpp
git diff --check
git diff --stat
git diff --numstat
uv run --frozen make release
uv run --frozen make test
uv run --frozen python bench/run_benchmarks.py --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 3 --threads 1 --output bench/results/062-candidate-checksum.json
```

The release SQL suite passed 3,964 assertions in 98 cases. The candidate checksum matched baseline. The plan's global `rg` expectation was imprecise: a pre-existing bind-time `YyjsonAllocator parser` remains in `JsonoShredSpecEntries`. The actual executor invariant passed: the retained pair was declared in `ShredLocalState`, and `JsonoShredFromTextExecute` had no stack-local pair. The unrelated shredding-spec parser was not changed.

### AB/BA timing

Primary commands, including the required fresh-process repeats:

```text
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-primary-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-primary-ab-repeat.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --target candidate=build/release/extension/jsono/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-primary-ba.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --target candidate=build/release/extension/jsono/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-primary-ba-repeat.json
```

| Primary order | Baseline ms | Candidate ms | Delta | Verdict |
|---|---:|---:|---:|---|
| AB initial | 3869.10 | 3936.63 | +1.75% | fail |
| AB repeat | 3787.14 | 3878.28 | +2.41% | repeated fail |
| BA initial | 3915.78 | 4156.82 | +6.16% | fail |
| BA repeat | 3920.80 | 3743.97 | −4.51% | repeated fail: below 5% gain |

Control commands:

```text
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --filter reshred/100k/ --runs 7 --threads 1 --output bench/results/062-controls-reshred-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --filter reshred/100k/ --runs 7 --threads 1 --output bench/results/062-controls-reshred-ab-repeat.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --target candidate=build/release/extension/jsono/jsono.duckdb_extension --filter reshred/100k/ --runs 7 --threads 1 --output bench/results/062-controls-reshred-ba.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --include-field-sample --filter parse/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-control-parse-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --target candidate=build/release/extension/jsono/jsono.duckdb_extension --include-field-sample --filter parse/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/062-control-parse-ba.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --filter parse_struct/1M/typed_object_array --runs 7 --threads 1 --output bench/results/062-control-parse-struct-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --target candidate=build/release/extension/jsono/jsono.duckdb_extension --filter parse_struct/1M/typed_object_array --runs 7 --threads 1 --output bench/results/062-control-parse-struct-ba.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target baseline=bench/results/workers/062-baseline/jsono.duckdb_extension --filter merge_patch/100k/wide_shredded_base_nested_shred_patch --runs 7 --threads 1 --output bench/results/062-control-merge-patch-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/062-baseline/jsono.duckdb_extension --target candidate=build/release/extension/jsono/jsono.duckdb_extension --filter merge_patch/100k/wide_shredded_base_nested_shred_patch --runs 7 --threads 1 --output bench/results/062-control-merge-patch-ba.json
```

| Control | AB delta | BA delta | Verdict |
|---|---:|---:|---|
| reshred same-set | +3.54% repeated | −4.72% | pass |
| reshred widening | +30.83% repeated | +31.30% | fail |
| reshred narrowing | +1.73% repeated | +2.09% | pass |
| plain parse field sample | −1.29% | +0.77% | pass |
| typed object array | −1.87% | +0.16% | pass |
| wide shredded merge patch | −1.17% | +1.39% | pass |

The first widening AB result was +34.91%; the fresh-process repeat remained a failure at +30.83%.

### RSS and profiles

Commands:

```text
/usr/bin/time -l uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --runs 3 --threads 1 --output bench/results/062-rss-candidate.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --include-field-sample --filter parse_shred/245760/field_sample_nested --profile --runs 1 --threads 1 --output bench/results/062-duckdb-profile-candidate.json
samply record --save-only --unstable-presymbolicate --output bench/results/samply/062-candidate-text-shred.json.gz -- uv run --frozen python bench/profile_driver.py --include-field-sample build/release/extension/jsono/jsono.duckdb_extension parse_shred/245760/field_sample_nested 5 1
uv run --frozen python bench/analyze_profile.py bench/results/samply/062-candidate-text-shred.json.gz --top 80
```

- Peak RSS: baseline 6,153,895,936 bytes; candidate 6,258,802,688 bytes; +1.70%, within the 10% gate.
- DuckDB query/source shape was unchanged.
- `JsonoShredFromTextExecute` inclusive: 59.91% baseline, 60.17% candidate.
- `EmitDomRowDirect` inclusive: 37.84% baseline, 37.62% candidate.
- `yyjson_read_opts` inclusive: 40.89% baseline, 41.62% candidate.
- `dyn_malloc` plus `dyn_free`: 0.075% self / 0.154% inclusive baseline; 0.021% self / 0.026% inclusive candidate. Candidate self was below 1%, and inclusive share decreased by 0.129 percentage points.
- Expected one-pass executor, writer, and parser symbols remained present.

### Revert and final gates

The timing contract rejected the candidate. The implementation was reverted without changing the checksum commit, then the implementation/revert pair was collapsed out of final history.

```text
uv run --frozen python -m unittest bench.test_bench
uv run --frozen make verify
```

Final results: 31 benchmark harness tests passed; release and relassert each passed 3,964 assertions in 98 cases; constructor matrix, Parquet property check, C++/SQL format check, and Python Black check passed. DuckLake and layout-revision gates were not required because the rejected candidate left no storage/layout change.
