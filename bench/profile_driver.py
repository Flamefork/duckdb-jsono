import argparse
from pathlib import Path

from run_benchmarks import (
    Target,
    build_cases,
    build_query,
    case_uses_data_path,
    create_connection,
    get_case_id,
    get_data_path,
    prepare_case,
)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {parsed}")
    return parsed


def resolve_profile_case(
    target: Target, filter_text: str, include_field_sample: bool
) -> tuple[Target, str, dict]:
    if filter_text.startswith("json/"):
        raise ValueError(
            "external profile driver supports only current JSONO target; "
            "use bench/run_benchmarks.py --profile for core json profiles"
        )

    cases = build_cases([target], filter_text, include_field_sample)
    if len(cases) == 1:
        return cases[0]
    if not cases:
        raise ValueError(f"no jsono benchmark case matches filter: {filter_text}")

    matches = "\n".join(
        f"  {get_case_id(case_target, scenario_config['operation'], size, scenario_config['scenario'])}"
        for case_target, size, scenario_config in cases
    )
    raise ValueError(
        "profile filter must match exactly one jsono case; matched:\n" + matches
    )


def run_profile_loop(
    extension_path: Path,
    filter_text: str,
    iterations: int,
    threads: int | None,
    include_field_sample: bool,
) -> None:
    target = Target("current", "jsono", extension_path.expanduser())
    case_target, size, scenario_config = resolve_profile_case(
        target, filter_text, include_field_sample
    )
    data_path = get_data_path(size, scenario_config)
    if case_uses_data_path(scenario_config) and not data_path.exists():
        raise FileNotFoundError(f"benchmark data file not found: {data_path}")

    query = build_query(case_target, scenario_config, data_path)
    conn = create_connection(case_target)
    try:
        if threads is not None:
            conn.execute(f"SET threads={threads}")
        prepare_case(conn, query)
        conn.execute(query.timed_sql).fetchall()
        for _ in range(iterations):
            conn.execute(query.timed_sql).fetchall()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one benchmark case in a tight loop for external profilers"
    )
    parser.add_argument("extension_path", type=Path)
    parser.add_argument(
        "filter", help="Benchmark filter that must match one jsono case"
    )
    parser.add_argument("iterations", nargs="?", type=positive_int, default=150)
    parser.add_argument("threads", nargs="?", type=positive_int)
    parser.add_argument(
        "--include-field-sample",
        action="store_true",
        help="Include local field_sample benchmark cases",
    )
    args = parser.parse_args()

    try:
        run_profile_loop(
            args.extension_path,
            args.filter,
            args.iterations,
            args.threads,
            args.include_field_sample,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
