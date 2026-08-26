# DOM shape cache allocation optimization map

Goal: remove memory allocated for unused DOM shape-cache slots while preserving
the existing cache capacity, collision policy, binary layout, and SQL behavior.

## Methodology

- Baseline and candidate use the same machine, session, data, command shape,
  preserved baseline extension, seven timing runs, and one DuckDB thread.
- H1 changes only allocation timing: both DOM cache tables start empty, allocate
  on first insert, and no longer reserve 64 permutation elements in every slot.
- Main metric: peak RSS of a ten-expression typed-constructor query with the
  default cache versus `JSONO_SHAPE_CACHE_SIZE=1`.
- Memory stop gate: do not patch if the baseline default-minus-size-1 RSS gap is
  below 10 MB.
- Memory keep gates: all ten checksums match; candidate default RSS is no more
  than 5 MB above candidate size-1 RSS; at least 80% of the baseline gap is
  removed.
- Performance controls: `parse/100k/{flat,nested}`, six
  `parse_struct/1M/typed_*` cases, and the local real-production-shaped
  `parse/245760/field_sample_nested` case. `number_heavy` is diagnostic only.
- Performance keep gate: candidate median time is no more than 5% above the
  preserved baseline in both AB and BA target orders for every required case.
  One outlier is repeated once; a repeated failure rejects H1.
- Correctness surfaces: benchmark checksums, cache miss/first insert/hit paths,
  state reset behavior, release and relassert SQL suites, constructor matrix,
  and format checks.
- Complexity is recorded with `git diff --stat` and `git diff --numstat`; no
  alternative mode, fallback, helper, or cache-policy change is permitted.

## Current Baseline

- Date: 2026-08-26.
- Source: `9ff2ba51d1599aed38c0390c633771782de28571` with a clean worktree
  before creating this ignored map.
- Real-production-shaped input:
  `bench/data/field_sample/events_nested.parquet`, 245,760 rows, two
  122,880-row groups, ZSTD compression, 54,920,338 bytes after dereferencing
  the local symlink.
- Non-default timing configuration: `--runs 7 --threads 1`.
- Cache-size isolation: baseline and candidate RSS runs explicitly unset
  `JSONO_SHAPE_CACHE_SIZE` or set it to `1`.
- Frozen extension:
  `bench/results/workers/059-baseline/jsono.duckdb_extension`. The plan's
  proposed renamed filename could not load because DuckDB derives the C++
  entrypoint name from the extension basename; the user approved preserving
  the required `jsono.duckdb_extension` basename in an ignored directory.
  SHA-256: `2f3194ccae3ebb1a92e03cbcc5dc81fa82879553e8656caea15921b41f76e78a`.
- RSS: default 50,839,552 bytes; size 1 25,657,344 bytes; gap 25,182,208
  bytes (24.02 MiB). The ten checksum values are identical between runs.
- Timing medians: flat 211.11 ms; nested 207.46 ms; diagnostic number-heavy
  69.08 ms; typed top scalar 258.82 ms; scalar array 370.71 ms; promoted
  scalar array 465.12 ms; object array 404.91 ms; nested scalar 419.11 ms;
  mixed all 691.08 ms; field sample nested 3,552.05 ms.

## Recommendation

KEEP H1. The memory benefit passes both thresholds, every required timing case
passes the 5% AB/BA gate after the prescribed single repeat of the field-sample
outlier, and the full repository verification gate passes.

## Relation To Existing Ideas

The direct-mapped cache, its default capacity, salted lookup fingerprint, and
collision acceptance already exist at the baseline. H1 does not test a smaller
logical cache. It tests whether allocating the table and every permutation
buffer before the first object is the measured source of memory overhead.
Plan 062 depends on H1 being kept because it retains direct-writer state across
chunks.

## Hypothesis Map

| ID | Idea | Status | Main metric | Delta | Complexity | Correctness surface | Decision |
|---|---|---|---|---:|---|---|---|
| H1 | Allocate both DOM cache tables on first insert and remove per-slot permutation reservation | kept | ten-expression peak RSS gap | 99.02% of baseline gap removed | 16 insertions, 20 deletions | first miss/insert/hit, reset, checksums, release/relassert | KEEP |

## Experiment Log

### Baseline checkpoint

Contract frozen before implementation. Live case discovery returned
`parse/100k/{flat,nested,number_heavy}`, six `parse_struct/1M/typed_*` cases,
and `parse/245760/field_sample_nested`. The benchmark harness passed 25 tests.

Commands:

```text
uv run --frozen make release
uv run --frozen python -m unittest bench.test_bench
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/059-baseline/jsono.duckdb_extension --filter parse/100k/ --runs 7 --threads 1 --output bench/results/059-baseline-parse.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/059-baseline/jsono.duckdb_extension --filter parse_struct/1M/typed_ --runs 7 --threads 1 --output bench/results/059-baseline-typed.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/059-baseline/jsono.duckdb_extension --include-field-sample --filter parse/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/059-baseline-field-sample.json
/usr/bin/time -l env -u JSONO_SHAPE_CACHE_SIZE build/release/duckdb -unsigned ...
/usr/bin/time -l env JSONO_SHAPE_CACHE_SIZE=1 build/release/duckdb -unsigned ...
```

The RSS query and full output are preserved in
`bench/results/059-rss-baseline-{default,size1}-{stdout,stderr}.log`. Its
default-minus-size-1 gap is above the 10 MB stop threshold, so H1 may proceed.

### H1 memory result — PASS

Implementation commit: `1f24aa8` (`Allocate DOM shape caches lazily`). Both
DOM state constructors now use empty cache vectors; each lookup reports a miss
while empty, and each insert allocates the unchanged `ShapeCacheSize()` table
before indexing it. Both eager per-slot permutation reservations are gone.

Candidate extension SHA-256:
`7ff5b5ec5c66b50499abad53a6cb692921c4b17ca86c46e9742c76154398b112`.

| Configuration | Baseline RSS | Candidate RSS |
|---|---:|---:|
| Default cache | 50,839,552 bytes | 25,739,264 bytes |
| Cache size 1 | 25,657,344 bytes | 25,493,504 bytes |
| Default-minus-size-1 gap | 25,182,208 bytes | 245,760 bytes |

The candidate gap is 0.23 MiB, below the 5 MB ceiling, and 99.02% of the
baseline gap disappeared, above the 80% threshold. All ten checksum values
match across baseline default, baseline size 1, candidate default, and
candidate size 1.

Diff complexity:

```text
src/include/jsono_dom.hpp | 36 ++++++++++++++++--------------------
1 file changed, 16 insertions(+), 20 deletions(-)
16  20  src/include/jsono_dom.hpp
```

### H1 AB/BA timing — PASS

Commands:

```text
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target before=bench/results/workers/059-baseline/jsono.duckdb_extension --filter parse/100k/ --runs 7 --threads 1 --output bench/results/059-parse-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/059-baseline/jsono.duckdb_extension --target after=build/release/extension/jsono/jsono.duckdb_extension --filter parse/100k/ --runs 7 --threads 1 --output bench/results/059-parse-ba.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target before=bench/results/workers/059-baseline/jsono.duckdb_extension --filter parse_struct/1M/typed_ --runs 7 --threads 1 --output bench/results/059-typed-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/059-baseline/jsono.duckdb_extension --target after=build/release/extension/jsono/jsono.duckdb_extension --filter parse_struct/1M/typed_ --runs 7 --threads 1 --output bench/results/059-typed-ba.json
uv run --frozen python bench/run_benchmarks.py --current-extension build/release/extension/jsono/jsono.duckdb_extension --target before=bench/results/workers/059-baseline/jsono.duckdb_extension --include-field-sample --filter parse/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/059-field-sample-ab.json
uv run --frozen python bench/run_benchmarks.py --current-extension bench/results/workers/059-baseline/jsono.duckdb_extension --target after=build/release/extension/jsono/jsono.duckdb_extension --include-field-sample --filter parse/245760/field_sample_nested --runs 7 --threads 1 --output bench/results/059-field-sample-ba.json
```

Seven-run, one-thread medians:

| Required scenario | AB baseline | AB candidate | AB delta | BA baseline | BA candidate | BA delta |
|---|---:|---:|---:|---:|---:|---:|
| parse flat | 210.50 ms | 214.07 ms | +1.70% | 211.82 ms | 215.11 ms | +1.55% |
| parse nested | 213.27 ms | 212.28 ms | -0.46% | 211.84 ms | 213.42 ms | +0.75% |
| typed top scalar | 255.84 ms | 257.10 ms | +0.49% | 257.58 ms | 258.60 ms | +0.40% |
| typed scalar array | 364.94 ms | 376.15 ms | +3.07% | 369.26 ms | 364.32 ms | -1.34% |
| typed promoted scalar array | 471.90 ms | 464.17 ms | -1.64% | 468.74 ms | 466.32 ms | -0.52% |
| typed object array | 406.67 ms | 405.24 ms | -0.35% | 407.09 ms | 405.26 ms | -0.45% |
| typed nested scalar | 422.35 ms | 421.56 ms | -0.19% | 418.76 ms | 420.74 ms | +0.47% |
| typed mixed all | 674.22 ms | 677.92 ms | +0.55% | 677.45 ms | 676.14 ms | -0.19% |
| field sample nested, repeated | 3,463.40 ms | 3,487.20 ms | +0.69% | 3,459.20 ms | 3,505.70 ms | +1.34% |

The first field-sample pair was the only outlier: +6.90% AB and +5.48% BA.
The prescribed one-time repeat is stored in
`059-field-sample-{ab,ba}-repeat.json` and passed both orders. The worst
accepted required deltas are +3.07% AB and +1.55% BA. Diagnostic
`number_heavy` measured -0.24% AB and +2.45% BA and does not affect KEEP.
Typed constructor result checksums match in both target orders; the parse
families passed the harness correctness gate before timing.

### Final verification — PASS

```text
uv run --frozen python -m unittest bench.test_bench
uv run --frozen make verify
rg -n "perm\.reserve\(64\)|shape_cache\.resize" src/include/jsono_dom.hpp
```

The benchmark harness passed 25 tests. Release and relassert each passed 3,887
assertions in 98 SQLLogic cases. The constructor matrix and constructor Parquet
round-trip passed. The C++ format check and Black check passed. The structural
search returns exactly two `shape_cache.resize` calls, both in insert methods,
and no `perm.reserve(64)` call.
