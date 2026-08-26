# jsono_array_length residual-native optimization map

Goal: answer shredded `jsono_array_length` from the verified residual skeleton without reconstructing the logical document, while preserving whole-document narrowing failures.

## Methodology

- Baseline and candidate use the same machine, data, extension basename, session shape, seven timing runs, and one DuckDB thread.
- Main metric: `array_length/100k/wide_root_shredded`, the synthetic wide-lane case that isolates reconstruct removal.
- Secondary acceptance metric: `array_length/100k/ecom_object_array_shredded`, which exercises object-array skeleton cardinality at `$.products`.
- Controls: plain e-commerce, the local production-shaped field sample, and a direct-Parquet wide diagnostic.
- Correctness uses SQLLogic edge cases, property parity, vector-representation checks, narrowing failures, benchmark checksums, and optimizer plan locks.
- The candidate is one implementation patch after separate characterization and red plan-lock commits.
- Candidate complexity is recorded with `git diff --stat` and `git diff --numstat`; the correctness surface is whole-document manifest verification plus residual path location.

## Current Baseline

- Date: 2026-08-26.
- Base SHA: `4feea30` (characterization commit).
- Plan-lock SHA: `8ee0b4e`.
- Baseline extension: `bench/results/workers/061-baseline/jsono.duckdb_extension`.
- Baseline extension SHA-256: `1df3814513a7005978d36b1fdb32c0b5a82108011ef7813830848f3a850aa55d`.
- `wide_root_shredded` uses `wide_flat_100k.parquet` and `WIDE_FLAT_SHREDDING_SPEC` (synthetic wide scalar lanes).
- The e-commerce cases use `ecom_100k.parquet`; the shredded case stores `$.products` in the existing object-array lane and the plain case is its unaffected control.
- `field_sample_root_shredded` resolves to `/Users/flamefork/Development/ff/duckdb-json-tools/bench/data/real_prod/events_nested.parquet`, 245,760 rows, two ZSTD row groups of 122,880 rows each, with the existing field-sample shred specification.
- The direct-Parquet case writes current-revision shredded JSONO during prepare with the same schema, row count, compression, and row-group configuration for both targets, then times a direct scan.
- Current DuckDB projects the stored `j` struct as a whole. Body-only nested projection cannot preserve the required ancestor validity today, so `Projections: j` is a known DuckDB-core limitation and not an acceptance condition.

## Recommendation

Keep H1. Both acceptance cases exceed their thresholds in both target orders, every checksum is identical, and the plain control stays within the 5% budget. The field-sample and direct-Parquet controls improve substantially. The candidate removes reconstruct from the executor plan, but DuckDB still projects the full stored `j` struct from Parquet; body-only I/O remains a separate DuckDB-core follow-up.

## Relation To Existing Ideas

Plan 060 introduced an operator-lifetime shared-signature reader overload for native shredded `jsono_keys`. This hypothesis reuses only that row-reader contract. It deliberately adds a dedicated array-length bind data, executor, and rewrite branch instead of extending the keys algorithm or the generic introspection rewrite.

## Hypothesis Map

| ID | Idea | Status | Main metric | Keep threshold | Complexity | Correctness surface | Decision |
|---|---|---|---:|---:|---|---|---|
| H1 | Count the verified residual array skeleton in a dedicated optimizer-only scalar | kept | `array_length/100k/wide_root_shredded` | at least 50% faster in both orders; e-commerce shredded at least 30% faster | one implementation commit, 105 added lines in one file | parity, narrowing, vectors, checksums, and no-reconstruct plan lock passed | keep |

## Experiment Log

### Characterization checkpoint

Commit `4feea30` added public NULL semantics, scalar and array shreds, empty skeletons, more than 63 shreds, constant/dictionary/nested-shuffle vectors, exact and unrelated narrowing, property parity, benchmark query placement, and scalar checksums. The pre-implementation SQLLogic suites, 300-example property suite, and 30 benchmark-harness tests passed. The captured root/path plans contained the public `jsono_array_length(CAST(j AS plain JSONO), ...)` path and `Projections: j`.

### Plan-lock checkpoint

Commit `8ee0b4e` added the test-only locks. The focused optimizer suite reached 132 passing assertions and then failed on the first new lock because the actual root plan still contained the public cast and lacked `__jsono_internal_shredded_array_length`. The implementation turned the same suite green with 265 assertions; plain JSONO does not use the internal scalar.

### H1 implementation

Implementation SHA: `db23ae2`.

The implementation adds a dedicated bind data, executor, factory, and rewrite branch in `src/jsono_optimizer.cpp`. Its diff is `1 file changed, 105 insertions(+)`; numstat is `105 0 src/jsono_optimizer.cpp`. The repository compiles as C++11, so the planned value `std::optional<JsonoPathSpec>` is unavailable. The implementation uses an owned nullable `unique_ptr<JsonoPathSpec>`: root is absence, the path overload deep-copies the public bind data, and `Equals` compares path and signature contents rather than pointer identities.

AB medians, candidate then baseline:

| Case | Baseline ms | Candidate ms | Delta |
|---|---:|---:|---:|
| `array_length/100k/wide_root_shredded` | 825.16 | 56.59 | -93.14% |
| `array_length/100k/ecom_object_array_shredded` | 469.42 | 36.12 | -92.31% |
| `array_length/100k/ecom_object_array_plain` | 24.95 | 25.39 | +1.76% |
| `array_length/245760/field_sample_root_shredded` | 1522.30 | 63.20 | -95.85% |
| `array_length_scan/100k/wide_root_shredded` | 838.14 | 106.39 | -87.31% |

BA medians, baseline then candidate:

| Case | Baseline ms | Candidate ms | Delta |
|---|---:|---:|---:|
| `array_length/100k/wide_root_shredded` | 812.69 | 55.68 | -93.15% |
| `array_length/100k/ecom_object_array_shredded` | 467.48 | 35.27 | -92.46% |
| `array_length/100k/ecom_object_array_plain` | 24.71 | 24.68 | -0.12% |
| `array_length/245760/field_sample_root_shredded` | 1511.29 | 62.19 | -95.88% |
| `array_length_scan/100k/wide_root_shredded` | 832.11 | 103.89 | -87.51% |

Checksums matched in both orders:

- wide materialized and direct scan: 100,000 rows, 0 non-NULL rows, NULL result sum, hash sum `1378784879315654392900000`;
- e-commerce plain and shredded: 100,000 rows, 100,000 non-NULL rows, result sum `499665`, hash sum `781484586252764726929940`;
- field sample: 245,760 rows, 0 non-NULL rows, NULL result sum, hash sum `3388501719406152235991040`.

Baseline and candidate direct-Parquet artifacts are identical at the declared storage level: 100,000 rows, 109 schema leaves, zero schema-diff rows, ZSTD compression, four row groups (three of 32,768 rows and one of 1,696 rows), and 14,423,993 compressed metadata bytes. The benchmark JSON records the same 14,857,941-byte file size for both targets.

Candidate root/path plans contain `__jsono_internal_shredded_array_length`, contain no public reconstruct cast, and show `Projections: j`. The executor expression references only the residual `body$2` reinterpret; it does not execute lane, marker, or spill reads.

Focused SQLLogic, the 300-example property suite, and 30 benchmark-harness tests passed. `uv run --frozen make verify` passed release and relassert with 3,964 assertions in 98 suites each, the relassert constructor matrix/Parquet checks, and both format checks. No timing repeat was needed.
