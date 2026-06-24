#!/usr/bin/env python3
"""Sanity checks for benchmark data files.

Beyond schema/row-count checks, this also runs a correctness gate that asserts
each jsono operation produces output semantically equal to its core DuckDB json
baseline before any timing happens. The gate compares normalized output over the
full dataset and fails loudly with a sample diff on any mismatch.
"""

import sys
from pathlib import Path

import duckdb

from config import DATA_DIR, JSONO_EXTENSION_PATH, SCENARIOS, SIZES, URL_SIZES
from sql_literals import sql_json, sql_string, sql_typed_literal

REQUIRED_EVENT_COLUMNS = ["json_nested", "json_flat", "g1e1", "g1e3", "g1e4"]
REQUIRED_URL_COLUMNS = ["url_short", "url_marketing", "url_wide_query", "url_mixed"]


def extract_path_sql(path: object) -> str:
    if isinstance(path, str):
        return sql_string(path)
    if isinstance(path, int):
        return str(path)
    raise ValueError(f"extract path must be a string or integer, got: {path!r}")


def extract_call_sql(function_name: str, value_sql: str, path: object) -> str:
    return f"{function_name}({value_sql}, {extract_path_sql(path)})"


def get_filter_arg(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "--filter" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--filter="):
            return arg.split("=", 1)[1]
    return None


def gate_columns(scenario_config: dict) -> tuple[str, str]:
    """Return (jsono_expr, core_json_expr) producing row-comparable values.

    Each expression normalizes away semantic-but-not-structural differences
    (key order, leaf vs nested representation) so a row-wise IS DISTINCT FROM
    detects only genuine divergence.
    """
    operation = scenario_config["operation"]
    json_column = scenario_config["json_column"]

    if operation == "merge_patch":
        patch = sql_json(scenario_config["patch_object"])
        typed_patch = sql_typed_literal(scenario_config["patch_object"])
        jsono_expr = f"to_json(jsono_merge_patch(jsono({json_column}::VARCHAR), jsono({typed_patch})))"
        core_expr = f"to_json(jsono(json_merge_patch({json_column}, {patch})::VARCHAR))"
        return jsono_expr, core_expr

    if operation == "entries":
        jsono_expr = f"list_sort(jsono_entries(jsono({json_column}::VARCHAR), key_style := 'dotted'))"
        core_expr = (
            f"list_sort(map_entries("
            f"json_transform({json_column}, '\"map(string, string)\"')))"
        )
        return jsono_expr, core_expr

    if operation == "extract":
        # jsono_transform takes a STRUCT-literal spec; core json_transform takes the JSON-string spec.
        jsono_expr = (
            f"jsono_transform(jsono({json_column}::VARCHAR), "
            f"{sql_typed_literal(scenario_config['spec'])})"
        )
        core_expr = (
            f"json_transform({json_column}, {sql_json(scenario_config['spec'])})"
        )
        return jsono_expr, core_expr

    if operation == "extract_jsono":
        jsono_value = f"jsono({json_column}::VARCHAR)"
        core_value = json_column
        if "base_path" in scenario_config:
            jsono_value = extract_call_sql(
                "jsono_extract", jsono_value, scenario_config["base_path"]
            )
            core_value = extract_call_sql(
                "json_extract", core_value, scenario_config["base_path"]
            )
        jsono_extract_expr = extract_call_sql(
            "jsono_extract", jsono_value, scenario_config["path"]
        )
        jsono_expr = f"to_json({jsono_extract_expr})"
        core_expr = extract_call_sql(
            "json_extract", core_value, scenario_config["path"]
        )
        return jsono_expr, core_expr

    if operation == "extract_string":
        jsono_value = f"jsono({json_column}::VARCHAR)"
        core_value = json_column
        if "base_path" in scenario_config:
            jsono_value = extract_call_sql(
                "jsono_extract", jsono_value, scenario_config["base_path"]
            )
            core_value = extract_call_sql(
                "json_extract", core_value, scenario_config["base_path"]
            )
        jsono_expr = extract_call_sql(
            "jsono_extract_string", jsono_value, scenario_config["path"]
        )
        core_expr = extract_call_sql(
            "json_extract_string", core_value, scenario_config["path"]
        )
        return jsono_expr, core_expr

    raise ValueError(f"no correctness gate defined for operation: {operation}")


def gate_scenarios(filter_arg: str | None) -> list[dict]:
    selected = []
    for scenario_config in SCENARIOS:
        targets = scenario_config["targets"]
        if "jsono" not in targets or "json" not in targets:
            continue
        size = scenario_config["size"]
        case_id = f"{scenario_config['operation']}/{size}/{scenario_config['scenario']}"
        if filter_arg is None or filter_arg in case_id:
            selected.append(scenario_config)
    return selected


def run_correctness_gate(args: list[str]) -> list[str]:
    """Assert jsono op output == core-json op output. Returns error messages."""
    scenarios = gate_scenarios(get_filter_arg(args))
    if not scenarios:
        return []

    if not JSONO_EXTENSION_PATH.exists():
        return [f"jsono extension not found: {JSONO_EXTENSION_PATH}"]

    conn = duckdb.connect(config={"allow_unsigned_extensions": True})
    errors: list[str] = []
    checked: list[str] = []
    try:
        conn.execute(f"LOAD {sql_string(str(JSONO_EXTENSION_PATH))}")
        conn.execute("INSTALL json")
        conn.execute("LOAD json")

        for scenario_config in scenarios:
            size = scenario_config["size"]
            if "data_file" in scenario_config:
                data_path = scenario_config["data_file"]
            else:
                data_path = DATA_DIR / f"events_{size}.parquet"
            if not data_path.exists():
                continue

            operation = scenario_config["operation"]
            scenario = scenario_config["scenario"]
            case_id = f"{operation}/{size}/{scenario}"
            checked.append(case_id)
            jsono_expr, core_expr = gate_columns(scenario_config)
            source = f"read_parquet({sql_string(str(data_path))})"

            paired = f"""
                SELECT {jsono_expr} AS j, {core_expr} AS c
                FROM {source}
            """
            mismatches = conn.execute(
                f"SELECT count(*) FROM ({paired}) WHERE j IS DISTINCT FROM c"
            ).fetchone()[0]

            if mismatches == 0:
                continue

            sample = conn.execute(
                f"SELECT j, c FROM ({paired}) WHERE j IS DISTINCT FROM c LIMIT 1"
            ).fetchone()
            errors.append(
                f"correctness gate failed for {case_id}: "
                f"{mismatches:,} row(s) differ between jsono and core json\n"
                f"      jsono: {sample[0]}\n"
                f"      json : {sample[1]}"
            )
    finally:
        conn.close()

    if checked and not errors:
        print(f"correctness gate: OK ({', '.join(checked)})")

    return errors


def check_row_count(
    conn: duckdb.DuckDBPyConnection, path: Path, expected: int
) -> str | None:
    """Check row count matches expected. Returns error message or None."""
    result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()
    actual = result[0]
    if actual != expected:
        return f"File `{path.name}` contains {actual:,} rows, expected {expected:,}"
    return None


def check_schema(
    conn: duckdb.DuckDBPyConnection, path: Path, required_columns: list[str]
) -> str | None:
    """Check required columns exist. Returns error message or None."""
    result = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    columns = {row[0] for row in result}
    missing = set(required_columns) - columns
    if missing:
        return f"File `{path.name}` missing columns: {', '.join(sorted(missing))}"
    return None


def run_sanity_checks(args: list[str]) -> int:
    """Run all sanity checks. Returns exit code (0=ok, 2=error)."""
    conn = duckdb.connect()
    errors = []

    for size_name, expected_rows in SIZES.items():
        path = DATA_DIR / f"events_{size_name}.parquet"

        if not path.exists():
            continue  # Missing files handled by bench.py

        # Row count
        if err := check_row_count(conn, path, expected_rows):
            errors.append(err)

        # Schema
        if err := check_schema(conn, path, REQUIRED_EVENT_COLUMNS):
            errors.append(err)

    for size_name, expected_rows in URL_SIZES.items():
        path = DATA_DIR / f"urls_{size_name}.parquet"

        if not path.exists():
            continue

        if err := check_row_count(conn, path, expected_rows):
            errors.append(err)

        if err := check_schema(conn, path, REQUIRED_URL_COLUMNS):
            errors.append(err)

    conn.close()

    if errors:
        print("Sanity check failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\nRegenerate data: uv run python bench/generate_data.py", file=sys.stderr
        )
        return 2

    gate_errors = run_correctness_gate(args)
    if gate_errors:
        print("Correctness gate failed:", file=sys.stderr)
        for err in gate_errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(run_sanity_checks(sys.argv[1:]))
