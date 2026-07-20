from config import (
    DATA_DIR,
    DIFF_PAIRS_SIZES,
    ECOM_PRODUCTS_SHRED,
    ECOM_SIZES,
    KEYED_PAIR_SIZES,
    KEYED_PAIR_WIDE_SHREDDING_SPEC,
    EXTRACT_CORE_SPEC,
    MULTIFILE_EXTRACT_COPY_PATH_A,
    MULTIFILE_EXTRACT_COPY_PATH_B,
    MULTIFILE_EXTRACT_SHREDDING_A,
    MULTIFILE_EXTRACT_SHREDDING_B,
    SETOP_EXTRACT_SHREDDING_SPEC,
    FIELD_SAMPLE_JSONO_COMPRESSION,
    FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL,
    FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE,
    NUMBERS_PARSE_COPY_PATH,
    NUMBERS_SIZES,
    PLUCK_CORE_SPEC,
    PRUNE_FILTER_BAND,
    PRUNE_FILTER_COPY_PATH,
    PRUNE_FILTER_NATIVE_COLUMN,
    PRUNE_FILTER_ROW_GROUP_SIZE,
    PRUNE_FILTER_SHRED_LEAF,
    PRUNE_FILTER_SHRED_TYPE,
    TRANSFORM_CORE_SPEC,
    TRANSFORM_ECOMMERCE_JOIN_SPEC,
    WIDE_FLAT_NESTED_PATCH,
    WIDE_FLAT_NESTED_SHRED_PATCH,
    WIDE_FLAT_SHREDDING_SPEC,
    WIDE_FLAT_SIZES,
    WIDE_FLAT_SMALL_PATCH,
)

SCENARIOS = [
    {
        "operation": "parse",
        "scenario": "flat",
        "json_column": "json_flat",
        "targets": ["jsono"],
    },
    {
        "operation": "parse",
        "scenario": "nested",
        "json_column": "json_nested",
        "targets": ["jsono"],
    },
    {
        "operation": "validate",
        "scenario": "flat",
        "json_column": "json_flat",
        "targets": ["jsono"],
    },
    {
        "operation": "validate",
        "scenario": "nested",
        "json_column": "json_nested",
        "targets": ["jsono"],
    },
    {
        "operation": "storage_size",
        "scenario": "nested",
        "json_column": "json_nested",
        "targets": ["jsono"],
    },
    {
        "operation": "parse_struct",
        "scenario": "nested_typed",
        "typed_shape": "nested_typed",
        "size": "100k",
        "row_count": 100_000,
        "targets": ["jsono"],
    },
    {
        "operation": "parse_struct_roundtrip",
        "scenario": "nested_typed",
        "typed_shape": "nested_typed",
        "size": "100k",
        "row_count": 100_000,
        "targets": ["jsono"],
    },
    {
        "operation": "parse_struct_json_roundtrip",
        "scenario": "nested_typed",
        "typed_shape": "nested_typed",
        "size": "100k",
        "row_count": 100_000,
        "targets": ["jsono"],
    },
    {
        "operation": "transform",
        "scenario": "nested_core",
        "json_column": "json_nested",
        "spec": TRANSFORM_CORE_SPEC,
        "current_only": True,
        "targets": ["jsono"],
    },
    {
        "operation": "transform",
        "scenario": "ecommerce_join",
        "json_column": "json_nested",
        "spec": TRANSFORM_ECOMMERCE_JOIN_SPEC,
        "current_only": True,
        "targets": ["jsono"],
    },
    {
        "operation": "transform",
        "scenario": "flat_core",
        "json_column": "json_flat",
        "spec": PLUCK_CORE_SPEC,
        "targets": ["jsono"],
    },
    {
        "operation": "shape_plan_recovery_transform",
        "scenario": "stable_only",
        "size": "100k",
        "row_count": 100_000,
        "shape_stream": "stable_only",
        "targets": ["jsono"],
    },
    {
        "operation": "shape_plan_recovery_transform",
        "scenario": "flood_tail",
        "size": "100k",
        "row_count": 100_000,
        "shape_stream": "flood_tail",
        "prefix_rows": 6_144,
        "targets": ["jsono"],
    },
    {
        "operation": "shape_plan_recovery_transform",
        "scenario": "flood_only",
        "size": "100k",
        "row_count": 100_000,
        "shape_stream": "flood_only",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge",
        "scenario": "few_groups_ignore_nulls",
        "json_column": "json_flat",
        "group_col": "g1e1",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_jsono",
        "scenario": "few_groups_ignore_nulls",
        "json_column": "json_flat",
        "group_col": "g1e1",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge",
        "scenario": "many_groups_ignore_nulls",
        "json_column": "json_flat",
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_jsono",
        "scenario": "many_groups_ignore_nulls",
        "json_column": "json_flat",
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "optimizer_project",
        "scenario": "repeated_filter_project",
        "size": "100k",
        "json_column": "json_nested",
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "nested_small_patch",
        "size": "10k",
        "json_column": "json_nested",
        "patch_object": {"event_name": "page_view"},
        "targets": ["jsono", "json"],
    },
    {
        "operation": "reshred",
        "scenario": "same_set",
        "size": "100k",
        "json_column": "json_flat",
        "source_spec": {"event_name": "VARCHAR", "user_id": "VARCHAR"},
        "target_spec": {"event_name": "VARCHAR", "user_id": "VARCHAR"},
        "targets": ["jsono"],
    },
    {
        "operation": "reshred",
        "scenario": "widening",
        "size": "100k",
        "json_column": "json_flat",
        "source_spec": {"event_name": "VARCHAR", "user_id": "VARCHAR"},
        "target_spec": {
            "event_name": "VARCHAR",
            "user_id": "VARCHAR",
            "session_id": "VARCHAR",
            "event_ts": "BIGINT",
        },
        "targets": ["jsono"],
    },
    {
        "operation": "reshred",
        "scenario": "narrowing",
        "size": "100k",
        "json_column": "json_flat",
        "source_spec": {
            "event_name": "VARCHAR",
            "user_id": "VARCHAR",
            "session_id": "VARCHAR",
            "event_ts": "BIGINT",
        },
        "target_spec": {"event_name": "VARCHAR", "user_id": "VARCHAR"},
        "targets": ["jsono"],
    },
    {
        "operation": "entries",
        "scenario": "flat_dotted",
        "size": "10k",
        "json_column": "json_flat",
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract",
        "scenario": "nested_core",
        "size": "10k",
        "json_column": "json_nested",
        "spec": EXTRACT_CORE_SPEC,
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract_jsono",
        "scenario": "nested_top_key",
        "size": "10k",
        "json_column": "json_nested",
        "path": "event_name",
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract_jsono",
        "scenario": "nested_jsonpath_object",
        "size": "10k",
        "json_column": "json_nested",
        "path": "$.page",
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract_jsono",
        "scenario": "nested_array_index",
        "size": "10k",
        "json_column": "json_nested",
        "base_path": "$.ec_items",
        "path": 0,
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract_string",
        "scenario": "nested_top_key",
        "size": "10k",
        "json_column": "json_nested",
        "path": "event_name",
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract_string",
        "scenario": "nested_jsonpath_scalar",
        "size": "10k",
        "json_column": "json_nested",
        "path": "$.page.page_url",
        "targets": ["jsono", "json"],
    },
    {
        "operation": "extract_string",
        "scenario": "nested_array_index",
        "size": "10k",
        "json_column": "json_nested",
        "base_path": "$.ec_items",
        "path": 0,
        "targets": ["jsono", "json"],
    },
    # Type-reconciliation boundaries (plan 033): the same lane extract measured across a
    # facade UNION ALL (shredded ∪ plain) and a heterogeneous union_by_name multi-file read.
    {
        "operation": "setop_extract_string",
        "scenario": "facade_shredded_plain",
        "size": "100k",
        "json_column": "json_nested",
        "shredding": SETOP_EXTRACT_SHREDDING_SPEC,
        "path": "event_name",
        "targets": ["jsono"],
    },
    {
        "operation": "multifile_extract_string",
        "scenario": "hetero_lane_both_files",
        "size": "100k",
        "json_column": "json_nested",
        "shredding": MULTIFILE_EXTRACT_SHREDDING_A,
        "shredding_b": MULTIFILE_EXTRACT_SHREDDING_B,
        "copy_path_a": MULTIFILE_EXTRACT_COPY_PATH_A,
        "copy_path_b": MULTIFILE_EXTRACT_COPY_PATH_B,
        "path": "event_name",
        "targets": ["jsono"],
    },
    # Wide-flat page_view-class merge shape (~100 schema-stable scalar keys):
    # merge-family performance is shape-dependent, the nested merge scenarios
    # above do not represent it. Generate the dataset with:
    # uv run python bench/generate_data.py --kind wide_flat
    {
        "operation": "merge_patch",
        "scenario": "wide_flat_small_patch",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "patch_object": {"event_name": "page_view_merged"},
        "targets": ["jsono", "json"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_flat_patch_count_1",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "patches": [{"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}}],
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_flat_patch_count_2",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "patches": [
            {"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}},
            {"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}},
        ],
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_flat_patch_count_4",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "patches": [
            {"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}},
            {"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}},
            {"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}},
            {"plain": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}},
        ],
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_jsono",
        "scenario": "wide_flat_few_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "group_col": "g1e1",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_jsono",
        "scenario": "wide_flat_many_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_jsono",
        "scenario": "wide_shredded_few_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "group_col": "g1e1",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_jsono",
        "scenario": "wide_shredded_many_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_max_jsono",
        "scenario": "wide_flat_keyed_many_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_min_jsono",
        "scenario": "wide_flat_keyed_many_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_max_jsono",
        "scenario": "wide_shredded_keyed_many_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_min_jsono",
        "scenario": "wide_shredded_keyed_many_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "group_col": "g1e4",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_max_jsono",
        "scenario": "wide_shredded_keyed_few_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "group_col": "g1e1",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_min_jsono",
        "scenario": "wide_shredded_keyed_few_groups",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "group_col": "g1e1",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_pair_jsono",
        "scenario": "wide_keyed_pair_medium_group_rows",
        "size": "1M",
        "row_count": KEYED_PAIR_SIZES["1M"],
        "data_file": DATA_DIR / "keyed_pair_1M.parquet",
        "wide_json_column": "json_wide_payload",
        "wide_shredding": KEYED_PAIR_WIDE_SHREDDING_SPEC,
        "detail_json_column": "json_detail_payload",
        "group_col": "g_medium_group_rows",
        "targets": ["jsono"],
    },
    {
        "operation": "group_merge_keyed_pair_jsono",
        "scenario": "wide_keyed_pair_few_group_rows",
        "size": "1M",
        "row_count": KEYED_PAIR_SIZES["1M"],
        "data_file": DATA_DIR / "keyed_pair_1M.parquet",
        "wide_json_column": "json_wide_payload",
        "wide_shredding": KEYED_PAIR_WIDE_SHREDDING_SPEC,
        "detail_json_column": "json_small_payload",
        "group_col": "g_few_group_rows",
        "targets": ["jsono"],
    },
    # Wide SHREDDED base (~107 lanes) + small patches. The base is shredded outside
    # timing; the timed merge builds the merged shredded document per row. Variants
    # isolate which executor path is taken (jsono-only — shredded has no core-json
    # equivalent):
    #   - mixed_patch: plain nested patch (payload holds a JSONO value) + small
    #     shredded patch. All top-level shreds -> the fast path copies the ~107
    #     base lanes through and folds only the small residuals.
    #   - shredded_patch_only: all-shredded inputs, the fast path baseline ceiling.
    #   - plain_patch_only: a single plain nested patch, isolates the plain-input fast
    #     path (the relaxation H1 enables).
    #   - nested_shred_patch: jsono({'params':{...}}) auto-shreds nested scalars into
    #     '$.params.*' lanes; a nested shred disqualifies the fast path, so this stays
    #     on the reshred fallback even after H1 — it isolates that separate bottleneck.
    {
        "operation": "merge_patch",
        "scenario": "wide_shredded_base_mixed_patch",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "patches": [WIDE_FLAT_NESTED_PATCH, WIDE_FLAT_SMALL_PATCH],
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_shredded_base_shredded_patch_only",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "patches": [WIDE_FLAT_SMALL_PATCH],
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_shredded_base_plain_patch_only",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "patches": [WIDE_FLAT_NESTED_PATCH],
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_shredded_base_poisoned_patch",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "bench_row_number": True,
        "patches_sql": [
            """CASE
                WHEN bench_row_number % 2048 = 1 THEN jsono('{"event_name":null}')
                ELSE jsono('{"payload":{"detail_goal_a":"1042","detail_goal_b":"5530"}}')
            END"""
        ],
        "targets": ["jsono"],
    },
    {
        "operation": "merge_patch",
        "scenario": "wide_shredded_base_nested_shred_patch",
        "size": "100k",
        "row_count": WIDE_FLAT_SIZES["100k"],
        "data_file": DATA_DIR / "wide_flat_100k.parquet",
        "json_column": "json_wide_flat",
        "shredding": WIDE_FLAT_SHREDDING_SPEC,
        "patches": [WIDE_FLAT_NESTED_SHRED_PATCH, WIDE_FLAT_SMALL_PATCH],
        "targets": ["jsono"],
    },
    {
        "operation": "parse",
        "scenario": "number_heavy",
        "size": "100k",
        "row_count": NUMBERS_SIZES["100k"],
        "data_file": DATA_DIR / "numbers_100k.parquet",
        "json_column": "json_numbers",
        "targets": ["jsono"],
    },
    {
        "operation": "parse_copy",
        "scenario": "number_heavy",
        "size": "100k",
        "row_count": NUMBERS_SIZES["100k"],
        "data_file": DATA_DIR / "numbers_100k.parquet",
        "json_column": "json_numbers",
        "copy_path": NUMBERS_PARSE_COPY_PATH,
        "copy_compression": FIELD_SAMPLE_JSONO_COMPRESSION,
        "copy_compression_level": FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL,
        "copy_row_group_size": FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE,
        "targets": ["jsono"],
    },
    {
        "operation": "validate",
        "scenario": "number_heavy",
        "size": "100k",
        "row_count": NUMBERS_SIZES["100k"],
        "data_file": DATA_DIR / "numbers_100k.parquet",
        "json_column": "json_numbers",
        "targets": ["jsono"],
    },
    {
        "operation": "storage_size",
        "scenario": "number_heavy",
        "size": "100k",
        "row_count": NUMBERS_SIZES["100k"],
        "data_file": DATA_DIR / "numbers_100k.parquet",
        "json_column": "json_numbers",
        "targets": ["jsono"],
    },
    # Row-group pruning: both scenarios write the same shredded, event_ts-clustered
    # Parquet (untimed prepare) and time a selective band filter over it. The
    # shredded variant filters the typed shred leaf; the native variant filters the
    # plain BIGINT control column. Run with --row-groups to see scanned/total: the
    # shred-leaf filter prunes identically to the native column (e.g. 1/9), proving
    # the shred carries usable per-row-group stats.
    {
        "operation": "prune_filter",
        "scenario": "synthetic_shredded",
        "size": "100k",
        "row_count": 100_000,
        "json_column": "json_nested",
        "shred_leaf": PRUNE_FILTER_SHRED_LEAF,
        "shred_leaf_type": PRUNE_FILTER_SHRED_TYPE,
        "native_column": PRUNE_FILTER_NATIVE_COLUMN,
        "filter_kind": "shred",
        "filter_band": PRUNE_FILTER_BAND,
        "copy_path": PRUNE_FILTER_COPY_PATH,
        "copy_compression": FIELD_SAMPLE_JSONO_COMPRESSION,
        "copy_compression_level": FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL,
        "copy_row_group_size": PRUNE_FILTER_ROW_GROUP_SIZE,
        "targets": ["jsono"],
    },
    {
        "operation": "prune_filter",
        "scenario": "synthetic_native",
        "size": "100k",
        "row_count": 100_000,
        "json_column": "json_nested",
        "shred_leaf": PRUNE_FILTER_SHRED_LEAF,
        "shred_leaf_type": PRUNE_FILTER_SHRED_TYPE,
        "native_column": PRUNE_FILTER_NATIVE_COLUMN,
        "filter_kind": "native",
        "filter_band": PRUNE_FILTER_BAND,
        "copy_path": PRUNE_FILTER_COPY_PATH,
        "copy_compression": FIELD_SAMPLE_JSONO_COMPRESSION,
        "copy_compression_level": FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL,
        "copy_row_group_size": PRUNE_FILTER_ROW_GROUP_SIZE,
        "targets": ["jsono"],
    },
]


# jsono_diff over per-transaction ordered cumulative snapshots (diff_pairs dataset; generate with
# `uv run python bench/generate_data.py --kind diff_pairs --size 100k`). jsono-only: core json has no
# diff op, and the SQL-emulation mode below IS the comparator. Read RELATIVE within one run:
#   - isolated_atomic ≈ isolated_merge_patch_proxy (the two-in/one-out C++ floor); isolated_counts /
#     isolated_elements expose the array-multiset cost on top of it;
#   - windowed_counts vs sql_emulation is the headline before→after (the superlinear flatten+sort the
#     §11 asset replaced) — run at 10k and 100k to see jsono_diff stay linear while the baseline does not;
#   - parse_counts adds the end-to-end jsono(text) parse cost.
DIFF_PAIRS_MODES = [
    ("isolated_atomic", "isolated_atomic"),
    ("isolated_counts", "isolated_counts"),
    ("isolated_elements", "isolated_elements"),
    ("isolated_merge_patch_proxy", "isolated_merge_patch_proxy"),
    ("windowed_counts", "windowed_counts"),
    ("sql_emulation", "sql_emulation_counts"),
    ("parse_counts", "parse_counts"),
]
SCENARIOS += [
    {
        "operation": "diff",
        "scenario": scenario_label,
        "size": size_name,
        "row_count": num_rows,
        "data_file": DATA_DIR / f"diff_pairs_{size_name}.parquet",
        "json_column": "snapshot_json",
        "mode": mode,
        "targets": ["jsono"],
    }
    for size_name, num_rows in DIFF_PAIRS_SIZES.items()
    for mode, scenario_label in DIFF_PAIRS_MODES
]

# The same isolated diff over shredded snapshot pairs (the rick/data storage shape): both sides
# share one all-scalar shredded type, so the direct lane diff applies — pairwise typed lanes plus
# the residual walk, reconstruct only for lane-presence-mismatch rows. Compare against
# isolated_atomic from the same run for the shredded-input overhead.
DIFF_PAIRS_SNAPSHOT_SHREDDING = {
    "transaction_id": "VARCHAR",
    "created_at": "BIGINT",
    "currency": "VARCHAR",
    "owner_id": "VARCHAR",
    "pipeline": "VARCHAR",
    "source": "VARCHAR",
    "utm_source": "VARCHAR",
    "utm_medium": "VARCHAR",
    "utm_campaign": "VARCHAR",
    "contact_id": "VARCHAR",
    "company_id": "VARCHAR",
    "region": "VARCHAR",
    "status": "VARCHAR",
    "stage": "VARCHAR",
    "amount": "BIGINT",
    "probability": "BIGINT",
    "score": "BIGINT",
    "priority": "VARCHAR",
    "last_event": "VARCHAR",
}
SCENARIOS += [
    {
        "operation": "diff",
        "scenario": "isolated_atomic_shredded",
        "size": size_name,
        "row_count": num_rows,
        "data_file": DATA_DIR / f"diff_pairs_{size_name}.parquet",
        "json_column": "snapshot_json",
        "mode": "isolated_atomic",
        "shredding": DIFF_PAIRS_SNAPSHOT_SHREDDING,
        "targets": ["jsono"],
    }
    for size_name, num_rows in DIFF_PAIRS_SIZES.items()
]

# jsono_entries array_style: the same array-heavy ecom doc flattened both ways. indexed_elements
# (default) explodes each `products` array into per-element leaves; whole_json emits one leaf per
# array. JSONO-only (core json has no equivalent contract); both run so a single run gives the
# relative cost. Needs the ecom dataset (`generate_data.py --kind ecom`).
SCENARIOS += [
    {
        "operation": "entries",
        "scenario": scenario_label,
        "size": size_name,
        "row_count": num_rows,
        "data_file": DATA_DIR / f"ecom_{size_name}.parquet",
        "json_column": "json_ecom",
        "array_style": array_style,
        "targets": ["jsono"],
    }
    for size_name, num_rows in ECOM_SIZES.items()
    for array_style, scenario_label in (
        ("indexed_elements", "ecom_indexed"),
        ("whole_json", "ecom_whole"),
    )
]

TYPED_STRUCT_OPTIMIZATION_SHAPES = (
    "typed_top_scalar",
    "typed_scalar_array",
    "typed_promoted_scalar_array",
    "typed_object_array",
    "typed_nested_scalar",
    "typed_mixed_all",
)
SCENARIOS += [
    {
        "operation": operation,
        "scenario": shape,
        "typed_shape": shape,
        "size": "1M",
        "row_count": 1_000_000,
        "targets": ["jsono"],
    }
    for operation in ("parse_struct", "parse_struct_plain")
    for shape in TYPED_STRUCT_OPTIMIZATION_SHAPES
]

# Same flatten over a products-shredded input — the production event_properties shape. whole_json
# here goes through reconstruct-to-plain (the shredded read path), so the shredded-vs-plain gap is
# the reconstruct cost. Paired indexed control reads the LIST<STRUCT> lanes element-by-element.
SCENARIOS += [
    {
        "operation": "entries",
        "scenario": scenario_label,
        "size": size_name,
        "row_count": num_rows,
        "data_file": DATA_DIR / f"ecom_{size_name}.parquet",
        "json_column": "json_ecom",
        "shredding": ECOM_PRODUCTS_SHRED,
        "array_style": array_style,
        "targets": ["jsono"],
    }
    for size_name, num_rows in ECOM_SIZES.items()
    for array_style, scenario_label in (
        ("indexed_elements", "ecom_shred_indexed"),
        ("whole_json", "ecom_shred_whole"),
    )
]
