#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Literal

import duckdb

from config import (
    DATA_DIR,
    DEFAULT_PROFILE_RUNS,
    DEFAULT_RUNS,
    JSONO_EXTENSION_PATH,
    PROFILES_DIR,
    FIELD_SAMPLE_SCENARIOS,
    RESULTS_DIR,
    SCENARIOS,
    SCHEMA_VERSION,
    SIZES,
    URL_SIZES,
)
from environment import collect_environment
from sql_literals import sql_identifier, sql_json, sql_string, sql_typed_literal

DEFAULT_SEED = 42
TargetKind = Literal["jsono", "json"]


@dataclass(frozen=True)
class Target:
    label: str
    kind: TargetKind
    extension_path: Path


@dataclass(frozen=True)
class BenchmarkQuery:
    prepare_sql: tuple[str, ...]
    timed_sql: str


def table_sql(data_path: Path) -> str:
    return f"read_parquet({sql_string(str(data_path))})"


def get_data_path(size: str, scenario_config: dict) -> Path:
    if "data_file" in scenario_config:
        return scenario_config["data_file"]
    return DATA_DIR / f"events_{size}.parquet"


def get_scenario_sizes(scenario_config: dict) -> list[str]:
    if "size" in scenario_config:
        return [scenario_config["size"]]
    if "sizes" in scenario_config:
        return scenario_config["sizes"]
    return list(SIZES.keys())


def get_row_count(size: str, scenario_config: dict) -> int | None:
    if "row_count" in scenario_config:
        return scenario_config["row_count"]
    if size in SIZES:
        return SIZES[size]
    if size in URL_SIZES:
        return URL_SIZES[size]
    if size.endswith("k") and size[:-1].isdigit():
        return int(size[:-1]) * 1_000
    if size.endswith("M") and size[:-1].isdigit():
        return int(size[:-1]) * 1_000_000
    return None


def collect_rule_key_regexes(value: object) -> list[str]:
    if isinstance(value, dict):
        regexes: list[str] = []
        if "key_regex" in value:
            key_regex = value["key_regex"]
            if not isinstance(key_regex, str):
                raise ValueError("rules key_regex must be a string")
            regexes.append(key_regex)
        for child in value.values():
            regexes.extend(collect_rule_key_regexes(child))
        return regexes

    if isinstance(value, list):
        regexes = []
        for item in value:
            regexes.extend(collect_rule_key_regexes(item))
        return regexes

    return []


def load_rule_key_regexes(rules_path: Path) -> list[str]:
    if not rules_path.exists():
        raise FileNotFoundError(f"field_sample rules file not found: {rules_path}")

    with rules_path.open() as file:
        document = json.load(file)

    if not isinstance(document, dict) or "rules" not in document:
        raise ValueError("rules file must contain a top-level rules array")
    if not isinstance(document["rules"], list):
        raise ValueError("rules file must contain a top-level rules array")

    seen: set[str] = set()
    regexes = []
    for key_regex in collect_rule_key_regexes(document["rules"]):
        if key_regex in seen:
            continue
        seen.add(key_regex)
        regexes.append(key_regex)

    if not regexes:
        raise ValueError("rules file does not contain any key_regex values")

    return regexes


def exact_root_key_from_regex(key_regex: str) -> str | None:
    if not key_regex.startswith("^") or not key_regex.endswith("$"):
        return None

    key = key_regex[1:-1]
    if key and all(char.isalnum() or char == "_" for char in key):
        return key

    return None


def transform_field_from_rule_key_regex(key_regex: str) -> tuple[str, object]:
    root_key = exact_root_key_from_regex(key_regex)
    if root_key is not None:
        return root_key, "VARCHAR"

    if key_regex == r"^products.\d+.category$":
        return (
            "product_category",
            {
                "type": "VARCHAR",
                "path": "$.products[*].category",
                "join_separator": "\n",
            },
        )

    raise ValueError(f"unsupported field_sample nested key regex: {key_regex}")


def load_transform_spec_from_rules(rules_path: Path) -> dict:
    spec = {}
    for key_regex in load_rule_key_regexes(rules_path):
        field_name, field_spec = transform_field_from_rule_key_regex(key_regex)
        if field_name in spec:
            raise ValueError(f"duplicate output field from rules: {field_name}")
        spec[field_name] = field_spec
    return spec


def get_transform_spec(scenario_config: dict) -> dict:
    if "rules_file" in scenario_config:
        return load_transform_spec_from_rules(scenario_config["rules_file"])
    return scenario_config["spec"]


def get_case_id(target: Target, operation: str, size: str, scenario: str) -> str:
    return f"{target.label}/{operation}/{size}/{scenario}"


def validate_label(label: str) -> None:
    if not label:
        raise ValueError("target label must not be empty")
    if any(char in label for char in "/= \t\n"):
        raise ValueError(f"target label contains unsupported characters: {label}")


def parse_jsono_target(value: str) -> Target:
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH target, got: {value}")
    label, path = value.split("=", 1)
    validate_label(label)
    return Target(label=label, kind="jsono", extension_path=Path(path).expanduser())


def create_targets(args: argparse.Namespace) -> list[Target]:
    targets = [
        Target(
            label="current",
            kind="jsono",
            extension_path=args.current_extension.expanduser(),
        ),
    ]
    targets.extend(parse_jsono_target(value) for value in args.target)

    targets.append(
        Target(
            label="json",
            kind="json",
            extension_path=Path("json"),
        )
    )

    seen: set[str] = set()
    for target in targets:
        if target.label in seen:
            raise ValueError(f"duplicate target label: {target.label}")
        seen.add(target.label)
        if target.kind == "json":
            continue
        if not target.extension_path.exists():
            raise FileNotFoundError(
                f"extension for target {target.label} not found: {target.extension_path}"
            )

    return targets


def create_connection(target: Target) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(config={"allow_unsigned_extensions": True})
    if target.kind == "json":
        conn.execute("INSTALL json")
        conn.execute("LOAD json")
        return conn
    conn.execute(f"LOAD {sql_string(str(target.extension_path))}")
    return conn


def jsono_value_sql(scenario_config: dict) -> str:
    json_column = scenario_config["json_column"]
    if "shredding" in scenario_config:
        shredding = sql_typed_literal(scenario_config["shredding"])
        return f"jsono({json_column}::VARCHAR, shredding := {shredding})"
    return f"jsono({json_column}::VARCHAR)"


def jsono_prepare_jsono(scenario_config: dict, data_path: Path) -> str:
    return f"""
        CREATE OR REPLACE TEMP TABLE _bench_in AS
        SELECT {jsono_value_sql(scenario_config)} AS t
        FROM {table_sql(data_path)}
    """


def jsono_prepare_jsono_with_group(scenario_config: dict, data_path: Path) -> str:
    group_col = scenario_config["group_col"]
    return f"""
        CREATE OR REPLACE TEMP TABLE _bench_in AS
        SELECT {jsono_value_sql(scenario_config)} AS t, {group_col}
        FROM {table_sql(data_path)}
    """


def jsono_prepare_jsono_with_group_and_key(
    scenario_config: dict, data_path: Path
) -> str:
    group_col = scenario_config["group_col"]
    return f"""
        CREATE OR REPLACE TEMP TABLE _bench_in AS
        WITH source AS (
            SELECT *, row_number() OVER ()::UBIGINT AS row_num
            FROM {table_sql(data_path)}
        )
        SELECT
            {jsono_value_sql(scenario_config)} AS t,
            {group_col} AS g,
            row_num AS k_ts,
            (row_num % 1000000)::UBIGINT AS k_secondary
        FROM source
    """


def jsono_prepare_jsono_pair_with_group_and_key(
    scenario_config: dict, data_path: Path
) -> str:
    wide_shredding = sql_typed_literal(scenario_config["wide_shredding"])
    group_col = scenario_config["group_col"]
    return f"""
        CREATE OR REPLACE TEMP TABLE _bench_in AS
        WITH source AS (
            SELECT *, row_number() OVER ()::UBIGINT AS row_num
            FROM {table_sql(data_path)}
        )
        SELECT
            jsono({scenario_config["wide_json_column"]}::VARCHAR, shredding := {wide_shredding}) AS wide_payload,
            jsono({scenario_config["detail_json_column"]}::VARCHAR) AS detail_payload,
            {group_col} AS g,
            row_num AS k_ts,
            (row_num % 1000000)::UBIGINT AS k_secondary
        FROM source
    """


def typed_struct_payload_sql() -> str:
    return """
        struct_pack(
            event_name := 'event_' || (i % 10)::VARCHAR,
            event_ts := i,
            score := i * 0.5,
            ok := i % 2 = 0,
            nested := struct_pack(
                a := i,
                b := 'x' || i::VARCHAR
            ),
            arr := [i, NULL, i + 1]
        )
    """


def jsono_prepare_typed_struct(scenario_config: dict, data_path: Path) -> str:
    if "struct_spec" in scenario_config:
        json_column = scenario_config["json_column"]
        struct_spec = scenario_config["struct_spec"]
        field_columns = ",\n                ".join(
            f"struct_extract(payload, {sql_string(name)}) AS {sql_identifier(name)}"
            for name in struct_spec.keys()
        )
        return f"""
            CREATE OR REPLACE TEMP TABLE _bench_struct_in AS
            SELECT payload, {field_columns}
            FROM (
                SELECT json_transform({json_column}, {sql_json(struct_spec)}) AS payload
                FROM {table_sql(data_path)}
            )
        """

    row_count = scenario_config["row_count"]
    return f"""
        CREATE OR REPLACE TEMP TABLE _bench_struct_in AS
        SELECT {typed_struct_payload_sql()} AS payload
        FROM range({row_count}) t(i)
    """


def object_args_sql(field_names: list[str]) -> str:
    arguments = []
    for field_name in field_names:
        arguments.append(sql_string(field_name))
        arguments.append(sql_identifier(field_name))
    return ",\n".join(arguments)


def extract_path_sql(path: object) -> str:
    if isinstance(path, str):
        return sql_string(path)
    if isinstance(path, int):
        return str(path)
    raise ValueError(f"extract path must be a string or integer, got: {path!r}")


def extract_call_sql(function_name: str, value_sql: str, path: object) -> str:
    return f"{function_name}({value_sql}, {extract_path_sql(path)})"


def build_jsono_parse_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    json_column = scenario_config["json_column"]
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono({json_column}::VARCHAR) AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_jsono_parse_shred_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    # End-to-end shred-from-text ingest: parse + shred in one timed call, the
    # production write path of a shredded column (W-1/W-3 frame).
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {jsono_value_sql(scenario_config)} AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_jsono_parse_struct_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_typed_struct(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono(payload) AS r
            FROM _bench_struct_in
        """,
    )


def build_jsono_parse_struct_roundtrip_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_typed_struct(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono(to_json(payload)::VARCHAR) AS r
            FROM _bench_struct_in
        """,
    )


def build_jsono_parse_struct_json_roundtrip_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_typed_struct(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono(to_json(payload)) AS r
            FROM _bench_struct_in
        """,
    )


def build_jsono_object_jsono_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_typed_struct(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono(payload) AS r
            FROM _bench_struct_in
        """,
    )


def build_jsono_object_json_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    field_names = list(scenario_config["struct_spec"].keys())
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_typed_struct(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT json_object(
                {object_args_sql(field_names)}
            ) AS r
            FROM _bench_struct_in
        """,
    )


def build_jsono_parse_copy_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    json_column = scenario_config["json_column"]
    copy_path = scenario_config["copy_path"]
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            COPY (
                SELECT jsono({json_column}::VARCHAR) AS r
                FROM {table_sql(data_path)}
            )
            TO {sql_string(str(copy_path))}
            (
                FORMAT parquet,
                COMPRESSION {scenario_config["copy_compression"]},
                COMPRESSION_LEVEL {scenario_config["copy_compression_level"]},
                ROW_GROUP_SIZE {scenario_config["copy_row_group_size"]},
                OVERWRITE_OR_IGNORE true
            )
        """,
    )


def build_jsono_scan_text_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    json_column = scenario_config["json_column"]
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {json_column}::VARCHAR AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_jsono_validate_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    if "jsono_column" in scenario_config:
        return BenchmarkQuery(
            prepare_sql=(),
            timed_sql=f"""
                CREATE OR REPLACE TEMP TABLE _bench_out AS
                SELECT jsono_validate({scenario_config["jsono_column"]}) AS r
                FROM {table_sql(data_path)}
            """,
        )
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_validate(t) AS r
            FROM _bench_in
        """,
    )


def build_jsono_storage_size_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    if "jsono_column" in scenario_config:
        return BenchmarkQuery(
            prepare_sql=(),
            timed_sql=f"""
                CREATE OR REPLACE TEMP TABLE _bench_out AS
                SELECT jsono_storage_size({scenario_config["jsono_column"]}) AS r
                FROM {table_sql(data_path)}
            """,
        )
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_storage_size(t) AS r
            FROM _bench_in
        """,
    )


def build_jsono_transform_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    spec = get_transform_spec(scenario_config)
    if "jsono_column" in scenario_config:
        return BenchmarkQuery(
            prepare_sql=(),
            timed_sql=f"""
                CREATE OR REPLACE TEMP TABLE _bench_out AS
                SELECT jsono_transform({scenario_config["jsono_column"]}, {sql_typed_literal(spec)}) AS r
                FROM {table_sql(data_path)}
            """,
        )
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_transform(t, {sql_typed_literal(spec)}) AS r
            FROM _bench_in
        """,
    )


def build_jsono_group_merge_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono_with_group(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT to_json(jsono_group_merge(t)) AS r
            FROM _bench_in
            GROUP BY {scenario_config["group_col"]}
        """,
    )


def build_jsono_group_merge_jsono_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono_with_group(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_group_merge(t) AS r
            FROM _bench_in
            GROUP BY {scenario_config["group_col"]}
        """,
    )


def build_jsono_group_merge_keyed_query(
    scenario_config: dict, data_path: Path, function_name: str
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(
            jsono_prepare_jsono_with_group_and_key(scenario_config, data_path),
        ),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {function_name}(t, row(k_ts, k_secondary)) AS r
            FROM _bench_in
            GROUP BY g
        """,
    )


def build_jsono_group_merge_keyed_pair_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(
            jsono_prepare_jsono_pair_with_group_and_key(scenario_config, data_path),
        ),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT
                jsono_group_merge_min(wide_payload, row(k_ts, k_secondary)) AS wide_payload,
                jsono_group_merge_max(detail_payload, row(k_ts, k_secondary)) AS detail_payload
            FROM _bench_in
            GROUP BY g
        """,
    )


def build_jsono_optimizer_project_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT
                t->>'$.user_id' AS user_id,
                MIN(CAST(t->>'$.event_ts' AS BIGINT)) AS first_event_ts,
                MAX(CAST(t->>'$.event_ts' AS BIGINT)) AS last_event_ts,
                count() AS count
            FROM _bench_in
            WHERE (t->>'$.event_name' = 'page_view')
              AND (t->>'$.device_type' = 'mobile')
              AND (t->>'$.geo_country' IN ['US', 'DE', 'ES'])
            GROUP BY user_id
        """,
    )


def project_paths_columns_sql(value_sql: str, paths: list[str]) -> str:
    return ",\n                ".join(
        f"{value_sql}->>{sql_string(path)} AS p{index}"
        for index, path in enumerate(paths)
    )


def build_jsono_project_paths_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    # Filterless multi-extract projection: no WHERE, so the matcher-gated
    # projector fuse does not fire today (O-1); over shredded input each path is
    # rewritten independently (O-2).
    columns = project_paths_columns_sql("t", scenario_config["paths"])
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {columns}
            FROM _bench_in
        """,
    )


def filter_paths_predicates_sql(value_sql: str, predicates: list[dict]) -> str:
    clauses = []
    for predicate in predicates:
        path = sql_string(predicate["path"])
        values = predicate["values"]
        if len(values) == 1:
            clauses.append(f"({value_sql}->>{path} = {sql_string(values[0])})")
        else:
            value_list = ", ".join(sql_string(value) for value in values)
            clauses.append(f"({value_sql}->>{path} IN [{value_list}])")
    return "\n              AND ".join(clauses)


def build_jsono_filter_paths_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    # Filtered multi-extract read: a WHERE with >=2 constant `->>` equality/IN predicates over
    # one column. The matcher fuses them into a single __jsono_internal_match (one residual read
    # + one manifest check over a shredded column; O-2). The aggregate count keeps the result
    # tiny so timing reflects the filter, not the sink.
    predicates_sql = filter_paths_predicates_sql("t", scenario_config["predicates"])
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT count(*) AS matched
            FROM _bench_in
            WHERE {predicates_sql}
        """,
    )


def merge_patch_one_patch_sql(patch: object) -> str:
    # A tagged patch picks how it is built (and thus the shredded type the merge sees):
    # {"plain": v} parses text -> a plain jsono value; {"shredded"/"auto_shred": v} runs the
    # struct constructor -> jsono({...}) auto-shreds (top-level scalars -> lanes, nested scalars
    # -> '$.path' lanes). A bare value is treated as the struct form for backward compatibility.
    if isinstance(patch, dict) and set(patch) == {"plain"}:
        return f"jsono({sql_json(patch['plain'])})"
    if (
        isinstance(patch, dict)
        and set(patch) <= {"shredded", "auto_shred"}
        and len(patch) == 1
    ):
        return f"jsono({sql_typed_literal(next(iter(patch.values())))})"
    return f"jsono({sql_typed_literal(patch)})"


def merge_patch_patches_sql(scenario_config: dict) -> list[str]:
    patches = scenario_config.get("patches", [scenario_config.get("patch_object")])
    return [merge_patch_one_patch_sql(patch) for patch in patches]


def build_jsono_merge_patch_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    # The base is materialized outside timing. With a 'shredding' key it becomes a
    # wide shredded value (jsono_value_sql), so the timed merge exercises the
    # shredded executor (fast path or reshred fallback). Patches are tagged plain vs
    # shredded so the scenario controls which executor path the merge takes.
    patch_args = ", ".join(merge_patch_patches_sql(scenario_config))
    return BenchmarkQuery(
        prepare_sql=(
            f"""
            CREATE OR REPLACE TEMP TABLE _bench_in AS
            SELECT {jsono_value_sql(scenario_config)} AS base
            FROM {table_sql(data_path)}
            """,
        ),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_merge_patch(base, {patch_args}) AS r
            FROM _bench_in
        """,
    )


def build_jsono_reshred_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    # The INSERT-into-a-differently-shredded-column write path: the optimizer rewrites that
    # cast into exactly this jsono(value, shredding := target) call, so timing the call times
    # the path. source==target exercises the shred copy-through, a superset target the
    # single-pass reshred, and a narrowing target the reconstruct+reshred fallback.
    json_column = scenario_config["json_column"]
    source_spec = sql_typed_literal(scenario_config["source_spec"])
    target_spec = sql_typed_literal(scenario_config["target_spec"])
    return BenchmarkQuery(
        prepare_sql=(
            f"""
            CREATE OR REPLACE TEMP TABLE _bench_in AS
            SELECT jsono({json_column}::VARCHAR, shredding := {source_spec}) AS t
            FROM {table_sql(data_path)}
            """,
        ),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono(t, shredding := {target_spec}) AS r
            FROM _bench_in
        """,
    )


def build_jsono_entries_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql="""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_entries(t, key_style := 'dotted') AS r
            FROM _bench_in
        """,
    )


def build_jsono_extract_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    spec = scenario_config["spec"]
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT jsono_transform(t, {sql_typed_literal(spec)}) AS r
            FROM _bench_in
        """,
    )


def build_jsono_extract_jsono_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    value_sql = "t"
    if "base_path" in scenario_config:
        value_sql = extract_call_sql(
            "jsono_extract", value_sql, scenario_config["base_path"]
        )
    result_sql = extract_call_sql("jsono_extract", value_sql, scenario_config["path"])
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {result_sql} AS r
            FROM _bench_in
        """,
    )


def build_jsono_extract_string_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    value_sql = "t"
    if "base_path" in scenario_config:
        value_sql = extract_call_sql(
            "jsono_extract", value_sql, scenario_config["base_path"]
        )
    result_sql = extract_call_sql(
        "jsono_extract_string", value_sql, scenario_config["path"]
    )
    return BenchmarkQuery(
        prepare_sql=(jsono_prepare_jsono(scenario_config, data_path),),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {result_sql} AS r
            FROM _bench_in
        """,
    )


def build_jsono_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    operation = scenario_config["operation"]

    match operation:
        case "parse":
            return build_jsono_parse_query(scenario_config, data_path)
        case "parse_shred":
            return build_jsono_parse_shred_query(scenario_config, data_path)
        case "parse_struct":
            return build_jsono_parse_struct_query(scenario_config, data_path)
        case "parse_struct_roundtrip":
            return build_jsono_parse_struct_roundtrip_query(scenario_config, data_path)
        case "parse_struct_json_roundtrip":
            return build_jsono_parse_struct_json_roundtrip_query(
                scenario_config, data_path
            )
        case "object_jsono":
            return build_jsono_object_jsono_query(scenario_config, data_path)
        case "object_json":
            return build_jsono_object_json_query(scenario_config, data_path)
        case "parse_copy":
            return build_jsono_parse_copy_query(scenario_config, data_path)
        case "scan_text":
            return build_jsono_scan_text_query(scenario_config, data_path)
        case "validate":
            return build_jsono_validate_query(scenario_config, data_path)
        case "storage_size":
            return build_jsono_storage_size_query(scenario_config, data_path)
        case "transform":
            return build_jsono_transform_query(scenario_config, data_path)
        case "group_merge":
            return build_jsono_group_merge_query(scenario_config, data_path)
        case "group_merge_jsono":
            return build_jsono_group_merge_jsono_query(scenario_config, data_path)
        case "group_merge_keyed_max_jsono":
            return build_jsono_group_merge_keyed_query(
                scenario_config, data_path, "jsono_group_merge_max"
            )
        case "group_merge_keyed_min_jsono":
            return build_jsono_group_merge_keyed_query(
                scenario_config, data_path, "jsono_group_merge_min"
            )
        case "group_merge_keyed_pair_jsono":
            return build_jsono_group_merge_keyed_pair_query(scenario_config, data_path)
        case "optimizer_project":
            return build_jsono_optimizer_project_query(scenario_config, data_path)
        case "project_paths":
            return build_jsono_project_paths_query(scenario_config, data_path)
        case "filter_paths":
            return build_jsono_filter_paths_query(scenario_config, data_path)
        case "merge_patch":
            return build_jsono_merge_patch_query(scenario_config, data_path)
        case "reshred":
            return build_jsono_reshred_query(scenario_config, data_path)
        case "entries":
            return build_jsono_entries_query(scenario_config, data_path)
        case "extract":
            return build_jsono_extract_query(scenario_config, data_path)
        case "extract_jsono":
            return build_jsono_extract_jsono_query(scenario_config, data_path)
        case "extract_string":
            return build_jsono_extract_string_query(scenario_config, data_path)
        case _:
            raise ValueError(f"jsono target does not support operation: {operation}")


def build_core_merge_patch_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    json_column = scenario_config["json_column"]
    patch_object = scenario_config["patch_object"]
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT json_merge_patch({json_column}, {sql_json(patch_object)}) AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_core_entries_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    json_column = scenario_config["json_column"]
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT
                map_entries(
                    json_transform({json_column}, '"map(string, string)"')
                ) AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_core_extract_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    json_column = scenario_config["json_column"]
    spec = scenario_config["spec"]
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT json_transform({json_column}, {sql_json(spec)}) AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_core_extract_jsono_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    value_sql = scenario_config["json_column"]
    if "base_path" in scenario_config:
        value_sql = extract_call_sql(
            "json_extract", value_sql, scenario_config["base_path"]
        )
    result_sql = extract_call_sql("json_extract", value_sql, scenario_config["path"])
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {result_sql} AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_core_extract_string_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    value_sql = scenario_config["json_column"]
    if "base_path" in scenario_config:
        value_sql = extract_call_sql(
            "json_extract", value_sql, scenario_config["base_path"]
        )
    result_sql = extract_call_sql(
        "json_extract_string", value_sql, scenario_config["path"]
    )
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {result_sql} AS r
            FROM {table_sql(data_path)}
        """,
    )


def build_core_project_paths_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    columns = project_paths_columns_sql(
        scenario_config["json_column"], scenario_config["paths"]
    )
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT {columns}
            FROM {table_sql(data_path)}
        """,
    )


def build_core_filter_paths_query(
    scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    predicates_sql = filter_paths_predicates_sql(
        scenario_config["json_column"], scenario_config["predicates"]
    )
    return BenchmarkQuery(
        prepare_sql=(),
        timed_sql=f"""
            CREATE OR REPLACE TEMP TABLE _bench_out AS
            SELECT count(*) AS matched
            FROM {table_sql(data_path)}
            WHERE {predicates_sql}
        """,
    )


def build_json_query(scenario_config: dict, data_path: Path) -> BenchmarkQuery:
    operation = scenario_config["operation"]

    match operation:
        case "merge_patch":
            return build_core_merge_patch_query(scenario_config, data_path)
        case "project_paths":
            return build_core_project_paths_query(scenario_config, data_path)
        case "filter_paths":
            return build_core_filter_paths_query(scenario_config, data_path)
        case "entries":
            return build_core_entries_query(scenario_config, data_path)
        case "extract":
            return build_core_extract_query(scenario_config, data_path)
        case "extract_jsono":
            return build_core_extract_jsono_query(scenario_config, data_path)
        case "extract_string":
            return build_core_extract_string_query(scenario_config, data_path)
        case _:
            raise ValueError(f"json target does not support operation: {operation}")


def build_query(
    target: Target, scenario_config: dict, data_path: Path
) -> BenchmarkQuery:
    if target.kind == "jsono":
        return build_jsono_query(scenario_config, data_path)
    if target.kind == "json":
        return build_json_query(scenario_config, data_path)
    raise ValueError(f"unknown target kind: {target.kind}")


def prepare_case(conn: duckdb.DuckDBPyConnection, query: BenchmarkQuery) -> None:
    for statement in query.prepare_sql:
        conn.execute(statement).fetchall()


def collect_duckdb_profile(
    conn: duckdb.DuckDBPyConnection, query: BenchmarkQuery, profile_path: Path
) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_case(conn, query)

    conn.execute("PRAGMA enable_profiling='json'")
    conn.execute("PRAGMA profiling_mode='detailed'")
    conn.execute(f"PRAGMA profiling_output={sql_string(str(profile_path))}")
    conn.execute(query.timed_sql).fetchall()
    conn.execute("PRAGMA disable_profiling")


def run_single_benchmark(
    conn: duckdb.DuckDBPyConnection, query: BenchmarkQuery, runs: int
) -> dict:
    prepare_case(conn, query)
    conn.execute(query.timed_sql).fetchall()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        conn.execute(query.timed_sql).fetchall()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "min_ms": round(min(times), 2),
        "median_ms": round(median(times), 2),
        "max_ms": round(max(times), 2),
    }


def target_metadata(target: Target) -> dict:
    build_type = "unknown"
    path_text = str(target.extension_path)
    if "/release/" in path_text:
        build_type = "release"
    elif "/reldebug/" in path_text:
        build_type = "reldebug"
    elif "/debug/" in path_text:
        build_type = "debug"

    return {
        "label": target.label,
        "kind": target.kind,
        "extension_path": path_text,
        "build_type": build_type,
    }


def run_benchmarks(
    targets: list[Target],
    cases_to_run: list[tuple[Target, str, dict]],
    runs: int,
    thread_modes: list[int],
    profile: bool = False,
) -> list[dict]:
    results = []
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    connections: dict[str, duckdb.DuckDBPyConnection] = {}

    try:
        for target, size, scenario_config in cases_to_run:
            data_path = get_data_path(size, scenario_config)
            if not data_path.exists():
                print(f"FAILED: benchmark data file not found: {data_path}")
                if "data_file" in scenario_config:
                    print(
                        "This benchmark uses an explicit data_file; it is not generated by bench/generate_data.py"
                    )
                else:
                    print(
                        f"Generate it with: uv run python bench/generate_data.py --kind events --size {size}"
                    )
                raise SystemExit(2)

            operation = scenario_config["operation"]
            scenario = scenario_config["scenario"]
            case_id = get_case_id(target, operation, size, scenario)

            conn = connections.get(target.label)
            if conn is None:
                conn = create_connection(target)
                connections[target.label] = conn

            query = build_query(target, scenario_config, data_path)
            row_count = get_row_count(size, scenario_config)

            for threads in thread_modes:
                # Set per timed run: the connection is reused across thread modes, so the
                # last SET wins and must be re-applied each pass.
                conn.execute(f"SET threads={threads}")
                print(f"Running {case_id} [t{threads}]...", end=" ", flush=True)

                if profile:
                    profile_dir = (
                        PROFILES_DIR / f"{case_id.replace('/', '_')}_t{threads}"
                    )
                    duckdb_profile_path = profile_dir / "query_profile.json"
                    print("profile...", end=" ", flush=True)
                    collect_duckdb_profile(conn, query, duckdb_profile_path)
                    print(f"{duckdb_profile_path}")

                timing = run_single_benchmark(conn, query, runs)
                rows_per_second = None
                if row_count is not None and timing["min_ms"] > 0:
                    rows_per_second = round(row_count / (timing["min_ms"] / 1000))
                throughput_text = (
                    f", {rows_per_second:,} rows/s"
                    if rows_per_second is not None
                    else ""
                )
                output_bytes = None
                copy_path = scenario_config.get("copy_path")
                if copy_path is not None and Path(copy_path).exists():
                    output_bytes = Path(copy_path).stat().st_size
                size_text = (
                    f", {output_bytes / 1_000_000:.2f}MB out"
                    if output_bytes is not None
                    else ""
                )
                print(f"{timing['median_ms']:.1f}ms{throughput_text}{size_text}")

                results.append(
                    {
                        "target": target.label,
                        "target_kind": target.kind,
                        "operation": operation,
                        "scenario": scenario,
                        "size": size,
                        "threads": threads,
                        "min_ms": timing["min_ms"],
                        "median_ms": timing["median_ms"],
                        "max_ms": timing["max_ms"],
                        "runs": runs,
                        "row_count": row_count,
                        "timestamp": timestamp,
                        "rows_per_second": rows_per_second,
                        "output_bytes": output_bytes,
                    }
                )
    finally:
        for conn in connections.values():
            conn.close()

    return results


def save_results_json(
    results: list[dict],
    path: Path,
    runs: int,
    environment: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (
            results[0]["timestamp"]
            if results
            else datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "environment": environment,
        "config": {
            "runs": runs,
            "warmup": True,
        },
        "results": [
            {
                "id": f"{r['target']}/{r['operation']}/{r['size']}/{r['scenario']}/t{r['threads']}",
                "target": r["target"],
                "target_kind": r["target_kind"],
                "operation": r["operation"],
                "scenario": r["scenario"],
                "size": r["size"],
                "threads": r["threads"],
                "min_ms": r["min_ms"],
                "median_ms": r["median_ms"],
                "max_ms": r["max_ms"],
                "runs": r["runs"],
                "row_count": r["row_count"],
                "rows_per_second": r["rows_per_second"],
                "output_bytes": r["output_bytes"],
            }
            for r in results
        ],
    }

    with path.open("w") as f:
        json.dump(output, f, indent=2)


def show_results(results_path: Path) -> None:
    with results_path.open() as f:
        data = json.load(f)

    results = data["results"]
    thread_modes = sorted({r["threads"] for r in results})
    duckdb_vector_size = 2048
    small_parallel_chunks = 64

    conn = duckdb.connect()
    try:
        rows = [
            (
                r["target"],
                r["operation"],
                r["scenario"],
                r["size"],
                r["threads"],
                r["min_ms"],
                r["median_ms"],
                r.get("row_count"),
                r.get("rows_per_second"),
            )
            for r in results
        ]
        conn.execute(
            """
            CREATE TABLE results (
                target VARCHAR,
                operation VARCHAR,
                scenario VARCHAR,
                size VARCHAR,
                threads INTEGER,
                min_ms DOUBLE,
                median_ms DOUBLE,
                row_count BIGINT,
                rows_per_second BIGINT
            )
            """
        )
        conn.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

        if len(thread_modes) < 2:
            conn.sql(
                """
                SELECT target, operation, scenario, size, threads,
                       round(min_ms, 1) AS min_ms,
                       round(median_ms, 1) AS median_ms,
                       row_count,
                       rows_per_second
                FROM results
                ORDER BY operation, scenario, size, target, threads
                """
            ).show(max_width=160)
            return

        # Pivot min_ms per thread mode plus a scaling speedup. speedup > 1 means
        # the case got faster with more threads. speedup < 1 on small inputs is
        # usually fixed parallel output overhead, not an operation-level regression.
        lo, hi = thread_modes[0], thread_modes[-1]
        per_mode = ",\n                   ".join(
            f"round(max(min_ms) FILTER (threads = {n}), 1) AS t{n}_ms"
            for n in thread_modes
        )
        conn.sql(
            f"""
            SELECT target, operation, scenario, size,
                   max(row_count) AS rows,
                   ceil(max(row_count)::DOUBLE / {duckdb_vector_size})::BIGINT AS chunks,
                   {per_mode},
                   round(
                       max(min_ms) FILTER (threads = {lo})
                       / nullif(max(min_ms) FILTER (threads = {hi}), 0), 2
                   ) AS speedup_t{lo}_t{hi},
                   CASE
                       WHEN max(row_count) IS NOT NULL
                            AND ceil(max(row_count)::DOUBLE / {duckdb_vector_size}) < {small_parallel_chunks}
                            AND max(min_ms) FILTER (threads = {lo})
                                / nullif(max(min_ms) FILTER (threads = {hi}), 0) < 1
                           THEN 'small_case_overhead'
                       WHEN max(min_ms) FILTER (threads = {lo})
                                / nullif(max(min_ms) FILTER (threads = {hi}), 0) < 1
                           THEN 'investigate'
                       ELSE ''
                   END AS scaling_note
            FROM results
            GROUP BY target, operation, scenario, size
            ORDER BY operation, scenario, size, target
            """
        ).show(max_width=200)
    finally:
        conn.close()


def build_cases(
    targets: list[Target], filter_text: str | None, include_field_sample: bool
) -> list[tuple[Target, str, dict]]:
    cases: list[tuple[Target, str, dict]] = []
    scenarios = [*SCENARIOS]
    if include_field_sample:
        scenarios.extend(FIELD_SAMPLE_SCENARIOS)

    for target in targets:
        for scenario_config in scenarios:
            if target.kind not in scenario_config["targets"]:
                continue
            if scenario_config.get("current_only") and target.label != "current":
                continue
            for size in get_scenario_sizes(scenario_config):
                case_id = get_case_id(
                    target,
                    scenario_config["operation"],
                    size,
                    scenario_config["scenario"],
                )
                if filter_text is None or filter_text in case_id:
                    cases.append((target, size, scenario_config))

    return cases


def print_case_list(cases: list[tuple[Target, str, dict]]) -> None:
    for target, size, scenario_config in cases:
        print(
            get_case_id(
                target, scenario_config["operation"], size, scenario_config["scenario"]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run jsono benchmarks against core DuckDB json baselines"
    )
    parser.add_argument(
        "--filter",
        help="Filter cases by substring, for example transform/1k/flat_core",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of timed runs (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--threads",
        default="1,8",
        help="Comma-separated DuckDB thread counts; every case runs once per count "
        "so single-thread vs parallel scaling is visible (default: 1,8)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Collect DuckDB query profiles"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected benchmark cases without running them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "latest.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--current-extension",
        type=Path,
        default=JSONO_EXTENSION_PATH,
        help="Current jsono extension path",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Additional jsono target as LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--include-field-sample",
        action="store_true",
        help="Include local field_sample benchmarks from bench/data/field_sample",
    )
    args = parser.parse_args()

    try:
        thread_modes = [int(part) for part in args.threads.split(",") if part.strip()]
    except ValueError:
        print(
            f"Error: --threads must be comma-separated integers, got {args.threads!r}"
        )
        raise SystemExit(2)
    if not thread_modes or any(count < 1 for count in thread_modes):
        print(f"Error: --threads must list positive integers, got {args.threads!r}")
        raise SystemExit(2)

    try:
        targets = create_targets(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        raise SystemExit(2)

    cases_to_run = build_cases(targets, args.filter, args.include_field_sample)
    if not cases_to_run:
        print(f"No cases match filter: {args.filter}")
        return

    if args.list:
        print_case_list(cases_to_run)
        return

    modes_text = ", ".join(f"t{count}" for count in thread_modes)
    print(
        f"Running {len(cases_to_run)} benchmark case(s) across {len(targets)} target(s) "
        f"at {modes_text}"
    )

    runs = args.runs
    if args.profile and args.runs == DEFAULT_RUNS:
        runs = DEFAULT_PROFILE_RUNS
        print(f"Profile mode: reducing runs to {runs}")

    results = run_benchmarks(
        targets, cases_to_run, runs, thread_modes, profile=args.profile
    )
    if results:
        sizes_used = sorted(set(size for _, size, _ in cases_to_run))
        environment = collect_environment(
            DEFAULT_SEED, sizes_used, [target_metadata(target) for target in targets]
        )
        save_results_json(results, args.output, runs, environment=environment)
        print(f"\nResults saved to {args.output}")

        print()
        show_results(args.output)


if __name__ == "__main__":
    main()
