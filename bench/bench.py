#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import duckdb

from config import (
    DATA_DIR,
    JSONO_EXTENSION_PATH,
    FIELD_SAMPLE_EVENTS_NESTED_PATH,
    FIELD_SAMPLE_JSONO_COMPRESSION,
    FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL,
    FIELD_SAMPLE_JSONO_EVENTS_PATH,
    FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE,
    FIELD_SAMPLE_ROW_COUNT,
    FIELD_SAMPLE_SCENARIOS,
    SIZES,
)


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def get_filter_arg(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "--filter" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--filter="):
            return arg.split("=", 1)[1]
    return None


def get_required_sizes(args: list[str]) -> list[str]:
    filter_arg = get_filter_arg(args)
    if filter_arg is None:
        return list(SIZES)

    sizes = [size for size in SIZES if size in filter_arg]
    if sizes:
        return sizes

    return list(SIZES)


def include_field_sample(args: list[str]) -> bool:
    return "--include-field-sample" in args


def get_jsono_target_labels(args: list[str]) -> list[str]:
    labels = ["current"]
    for index, arg in enumerate(args):
        if arg == "--target" and index + 1 < len(args):
            labels.append(args[index + 1].split("=", 1)[0])
        elif arg.startswith("--target="):
            labels.append(arg.split("=", 1)[1].split("=", 1)[0])
    return labels


def needs_field_sample_jsono(args: list[str]) -> bool:
    if not include_field_sample(args):
        return False

    filter_arg = get_filter_arg(args)
    if filter_arg is None:
        return True

    for scenario_config in FIELD_SAMPLE_SCENARIOS:
        if (
            "data_file" in scenario_config
            and scenario_config["data_file"] == FIELD_SAMPLE_JSONO_EVENTS_PATH
        ):
            case_id = (
                f"{scenario_config['operation']}/"
                f"{scenario_config['size']}/"
                f"{scenario_config['scenario']}"
            )
            for label in get_jsono_target_labels(args):
                if filter_arg in f"{label}/{case_id}":
                    return True

    return False


def field_sample_jsono_metadata_path() -> Path:
    return FIELD_SAMPLE_JSONO_EVENTS_PATH.with_suffix(
        f"{FIELD_SAMPLE_JSONO_EVENTS_PATH.suffix}.metadata.json"
    )


def expected_field_sample_jsono_metadata(source_path: Path) -> dict[str, object]:
    return {
        "source_path": str(source_path),
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "extension_path": str(JSONO_EXTENSION_PATH),
        "extension_mtime_ns": JSONO_EXTENSION_PATH.stat().st_mtime_ns,
        "row_count": FIELD_SAMPLE_ROW_COUNT,
        "row_group_size": FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE,
        "compression": FIELD_SAMPLE_JSONO_COMPRESSION,
        "compression_level": FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL,
    }


def field_sample_jsono_metadata_matches(source_path: Path) -> bool:
    metadata_path = field_sample_jsono_metadata_path()
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False

    return metadata == expected_field_sample_jsono_metadata(source_path)


def write_field_sample_jsono_metadata(source_path: Path) -> None:
    field_sample_jsono_metadata_path().write_text(
        json.dumps(expected_field_sample_jsono_metadata(source_path), indent=2) + "\n"
    )


def ensure_data_exists(args: list[str]) -> None:
    required_sizes = get_required_sizes(args)
    missing = [
        size
        for size in required_sizes
        if not (DATA_DIR / f"events_{size}.parquet").exists()
    ]
    if not missing:
        return

    print(f"Generating missing benchmark data: {', '.join(missing)}")
    for size in missing:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "generate_data.py"),
                "--kind",
                "events",
                "--size",
                size,
            ],
            check=True,
        )


def field_sample_jsono_is_current() -> bool:
    if not FIELD_SAMPLE_JSONO_EVENTS_PATH.exists():
        return False
    source_path = FIELD_SAMPLE_EVENTS_NESTED_PATH
    if not source_path.exists() or not JSONO_EXTENSION_PATH.exists():
        return False
    if not field_sample_jsono_metadata_matches(source_path):
        return False
    if FIELD_SAMPLE_JSONO_EVENTS_PATH.stat().st_mtime < max(
        source_path.stat().st_mtime, JSONO_EXTENSION_PATH.stat().st_mtime
    ):
        return False

    conn = duckdb.connect()
    try:
        rows, max_row_group_size, compression = conn.execute(f"""
            SELECT
                count(DISTINCT row_group_id),
                max(row_group_num_rows),
                string_agg(DISTINCT compression, ',')
            FROM parquet_metadata({sql_string(str(FIELD_SAMPLE_JSONO_EVENTS_PATH))})
            WHERE starts_with(path_in_schema, 'event_properties')
            """).fetchone()
        row_count = conn.execute(
            f"SELECT count(*) FROM read_parquet({sql_string(str(FIELD_SAMPLE_JSONO_EVENTS_PATH))})"
        ).fetchone()[0]
    finally:
        conn.close()

    return (
        row_count == FIELD_SAMPLE_ROW_COUNT
        and rows > 1
        and max_row_group_size
        == min(FIELD_SAMPLE_ROW_COUNT, FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE)
        and compression == FIELD_SAMPLE_JSONO_COMPRESSION.upper()
    )


def ensure_field_sample_jsono_exists(args: list[str]) -> None:
    if not needs_field_sample_jsono(args) or field_sample_jsono_is_current():
        return

    source_path = FIELD_SAMPLE_EVENTS_NESTED_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"field_sample source file not found: {source_path}")
    if not JSONO_EXTENSION_PATH.exists():
        raise FileNotFoundError(f"jsono extension not found: {JSONO_EXTENSION_PATH}")

    print(
        "Generating field_sample JSONO data: "
        f"{FIELD_SAMPLE_JSONO_EVENTS_PATH} "
        f"({FIELD_SAMPLE_JSONO_COMPRESSION} level {FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL}, "
        f"row group {FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE})",
        flush=True,
    )
    FIELD_SAMPLE_JSONO_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(config={"allow_unsigned_extensions": True})
    try:
        conn.execute(f"LOAD {sql_string(str(JSONO_EXTENSION_PATH))}")
        conn.execute(f"""
            COPY (
                SELECT
                    part_id,
                    date_utc,
                    row_id,
                    event_timestamp,
                    jsono(event_properties::VARCHAR) AS event_properties
                FROM read_parquet({sql_string(str(source_path))})
            )
            TO {sql_string(str(FIELD_SAMPLE_JSONO_EVENTS_PATH))}
            (
                FORMAT parquet,
                COMPRESSION {FIELD_SAMPLE_JSONO_COMPRESSION},
                COMPRESSION_LEVEL {FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL},
                ROW_GROUP_SIZE {FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE},
                OVERWRITE_OR_IGNORE true
            )
            """)
        write_field_sample_jsono_metadata(source_path)
    finally:
        conn.close()


def run_sanity_checks(args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "sanity_checks.py"), *args],
        check=True,
    )


def run_benchmarks(args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "run_benchmarks.py"), *args],
        check=True,
    )


def main() -> None:
    ensure_data_exists(sys.argv[1:])
    ensure_field_sample_jsono_exists(sys.argv[1:])
    run_sanity_checks(sys.argv[1:])
    run_benchmarks(sys.argv[1:])


if __name__ == "__main__":
    main()
