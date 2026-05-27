import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))

spec = importlib.util.spec_from_file_location("bench_script", BENCH_DIR / "bench.py")
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench)

run_spec = importlib.util.spec_from_file_location(
    "run_benchmarks_script", BENCH_DIR / "run_benchmarks.py"
)
run_benchmarks = importlib.util.module_from_spec(run_spec)
assert run_spec.loader is not None
run_spec.loader.exec_module(run_benchmarks)

compare_spec = importlib.util.spec_from_file_location(
    "compare_results_script", BENCH_DIR / "compare_results.py"
)
compare_results = importlib.util.module_from_spec(compare_spec)
assert compare_spec.loader is not None
compare_spec.loader.exec_module(compare_results)

profile_spec = importlib.util.spec_from_file_location(
    "profile_driver_script", BENCH_DIR / "profile_driver.py"
)
profile_driver = importlib.util.module_from_spec(profile_spec)
assert profile_spec.loader is not None
profile_spec.loader.exec_module(profile_driver)


class FieldSampleJSONOMetadataTest(unittest.TestCase):
    def test_metadata_rejects_storage_parameter_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "events_nested.parquet"
            extension_path = temp_path / "jsono.duckdb_extension"
            output_path = temp_path / "events_nested_jsono.parquet"
            source_path.write_text("source")
            extension_path.write_text("extension")
            output_path.write_text("output")

            with (
                patch.object(bench, "JSONO_EXTENSION_PATH", extension_path),
                patch.object(bench, "FIELD_SAMPLE_JSONO_EVENTS_PATH", output_path),
                patch.object(bench, "FIELD_SAMPLE_JSONO_COMPRESSION", "zstd"),
                patch.object(bench, "FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL", 8),
                patch.object(bench, "FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE", 32_768),
                patch.object(bench, "FIELD_SAMPLE_ROW_COUNT", 245_760),
            ):
                bench.write_field_sample_jsono_metadata(source_path)
                self.assertTrue(bench.field_sample_jsono_metadata_matches(source_path))

                with patch.object(bench, "FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL", 1):
                    self.assertFalse(
                        bench.field_sample_jsono_metadata_matches(source_path)
                    )

                with patch.object(bench, "FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE", 8_192):
                    self.assertFalse(
                        bench.field_sample_jsono_metadata_matches(source_path)
                    )


class ProfileDriverCaseResolutionTest(unittest.TestCase):
    def test_resolves_exact_core_case(self) -> None:
        target = profile_driver.Target(
            "current", "jsono", Path("jsono.duckdb_extension")
        )

        _, size, scenario_config = profile_driver.resolve_profile_case(
            target, "merge_patch/10k/nested_small_patch", False
        )

        self.assertEqual(size, "10k")
        self.assertEqual(scenario_config["operation"], "merge_patch")
        self.assertEqual(scenario_config["scenario"], "nested_small_patch")

    def test_resolves_extract_jsono_case(self) -> None:
        target = profile_driver.Target(
            "current", "jsono", Path("jsono.duckdb_extension")
        )

        _, size, scenario_config = profile_driver.resolve_profile_case(
            target, "extract_jsono/10k/nested_top_key", False
        )

        self.assertEqual(size, "10k")
        self.assertEqual(scenario_config["operation"], "extract_jsono")
        self.assertEqual(scenario_config["scenario"], "nested_top_key")

    def test_rejects_core_json_target_filter(self) -> None:
        target = profile_driver.Target(
            "current", "jsono", Path("jsono.duckdb_extension")
        )

        with self.assertRaisesRegex(ValueError, "only current JSONO target"):
            profile_driver.resolve_profile_case(
                target, "json/extract_jsono/10k/nested_top_key", False
            )

    def test_rejects_ambiguous_filter(self) -> None:
        target = profile_driver.Target(
            "current", "jsono", Path("jsono.duckdb_extension")
        )

        with self.assertRaisesRegex(ValueError, "must match exactly one"):
            profile_driver.resolve_profile_case(target, "group_merge", False)

    def test_field_sample_requires_flag(self) -> None:
        target = profile_driver.Target(
            "current", "jsono", Path("jsono.duckdb_extension")
        )

        with self.assertRaisesRegex(ValueError, "no jsono benchmark case matches"):
            profile_driver.resolve_profile_case(
                target, "merge_patch/245760/retail_sample_nested", False
            )

        _, size, scenario_config = profile_driver.resolve_profile_case(
            target, "merge_patch/245760/retail_sample_nested", True
        )

        self.assertEqual(size, "245760")
        self.assertEqual(scenario_config["scenario"], "retail_sample_nested")


class ExtractBenchmarkQueryTest(unittest.TestCase):
    def test_jsono_merge_patch_query_uses_constant_typed_patch(self) -> None:
        query = run_benchmarks.build_jsono_query(
            {
                "operation": "merge_patch",
                "scenario": "nested_small_patch",
                "json_column": "json_nested",
                "patch_object": {"event_name": "page_view"},
                "targets": ["jsono", "json"],
            },
            Path("events.parquet"),
        )

        self.assertNotIn("AS patch", query.prepare_sql[0])
        self.assertIn(
            "SELECT jsono_merge_patch(base, jsono({'event_name': 'page_view'})) AS r",
            query.timed_sql,
        )

    def test_jsono_extract_jsono_query_uses_single_path_api(self) -> None:
        query = run_benchmarks.build_jsono_query(
            {
                "operation": "extract_jsono",
                "scenario": "nested_top_key",
                "json_column": "json_nested",
                "path": "event_name",
                "targets": ["jsono", "json"],
            },
            Path("events.parquet"),
        )

        self.assertIn("SELECT jsono_extract(t, 'event_name') AS r", query.timed_sql)

    def test_jsono_extract_string_query_uses_index_path_api(self) -> None:
        query = run_benchmarks.build_jsono_query(
            {
                "operation": "extract_string",
                "scenario": "nested_array_index",
                "json_column": "json_nested",
                "base_path": "$.ec_items",
                "path": 0,
                "targets": ["jsono", "json"],
            },
            Path("events.parquet"),
        )

        self.assertIn(
            "SELECT jsono_extract_string(jsono_extract(t, '$.ec_items'), 0) AS r",
            query.timed_sql,
        )

    def test_core_extract_string_query_uses_core_json_baseline(self) -> None:
        query = run_benchmarks.build_json_query(
            {
                "operation": "extract_string",
                "scenario": "nested_array_index",
                "json_column": "json_nested",
                "base_path": "$.ec_items",
                "path": 0,
                "targets": ["jsono", "json"],
            },
            Path("events.parquet"),
        )

        self.assertIn(
            "SELECT json_extract_string(json_extract(json_nested, '$.ec_items'), 0) AS r",
            query.timed_sql,
        )


class CompareResultsCompatibilityTest(unittest.TestCase):
    def test_schema2_baseline_matches_highest_thread_latest_result(self) -> None:
        diff = compare_results.generate_diff(
            {
                "schema_version": 2,
                "results": [
                    {
                        "id": "current/parse/10k/flat",
                        "median_ms": 100.0,
                    }
                ],
            },
            {
                "schema_version": 4,
                "results": [
                    {
                        "id": "current/parse/10k/flat/t1",
                        "threads": 1,
                        "median_ms": 200.0,
                    },
                    {
                        "id": "current/parse/10k/flat/t8",
                        "threads": 8,
                        "median_ms": 95.0,
                    },
                ],
            },
            0.0,
            0.01,
        )

        self.assertEqual(diff["summary"]["missing_in_baseline"], 0)
        self.assertEqual(diff["summary"]["missing_in_latest"], 0)
        self.assertEqual(len(diff["cases"]), 1)
        self.assertEqual(diff["cases"][0]["id"], "current/parse/10k/flat")
        self.assertEqual(diff["cases"][0]["current_ms"], 95.0)


if __name__ == "__main__":
    unittest.main()
