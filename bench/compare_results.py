#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    RESULTS_DIR,
    DEFAULT_TOLERANCE_PCT,
    SCHEMA_VERSION,
    DEFAULT_MIN_EFFECT_MS,
    PROFILES_DIR,
)


def load_results(path: Path) -> dict | None:
    """Load results from JSON file."""
    json_path = path.with_suffix(".json")

    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    return None


def classify_case(
    baseline_ms: float | None,
    current_ms: float | None,
    tolerance_pct: float,
    min_effect_ms: float,
) -> str:
    """Classify a case as FASTER/SLOWER/UNCHANGED/MISSING_IN_*."""
    if baseline_ms is None:
        return "MISSING_IN_BASELINE"
    if current_ms is None:
        return "MISSING_IN_LATEST"

    diff_ms = current_ms - baseline_ms
    diff_pct = (diff_ms / baseline_ms) * 100 if baseline_ms != 0 else 0

    # UNCHANGED if below min_effect_ms OR within tolerance
    if abs(diff_ms) < min_effect_ms or abs(diff_pct) <= tolerance_pct:
        return "UNCHANGED"

    if diff_pct > tolerance_pct:
        return "SLOWER"

    if diff_pct < -tolerance_pct:
        return "FASTER"

    return "UNCHANGED"


def diff_environment(baseline_env: dict | None, latest_env: dict | None) -> dict:
    """Compare environment sections, return differences."""
    if baseline_env is None or latest_env is None:
        return {}

    diffs = {}

    # Flat keys to compare
    flat_keys = ["build_type", "os", "arch", "cpu_model", "cpu_cores", "duckdb_version"]
    for key in flat_keys:
        baseline_val = baseline_env.get(key)
        latest_val = latest_env.get(key)
        if baseline_val != latest_val:
            diffs[key] = {"baseline": baseline_val, "current": latest_val}

    # Dataset nested comparison
    baseline_ds = baseline_env.get("dataset", {})
    latest_ds = latest_env.get("dataset", {})
    for key in ["seed", "sizes", "schema", "complexity_weights"]:
        baseline_val = baseline_ds.get(key)
        latest_val = latest_ds.get(key)
        if baseline_val != latest_val:
            diffs[f"dataset.{key}"] = {"baseline": baseline_val, "current": latest_val}

    return diffs


def format_multiplier(baseline_ms: float, current_ms: float) -> str:
    """Format speed change as multiplier (e.g., '2.00× faster', '1.50× slower')."""
    if current_ms > baseline_ms:
        mult = current_ms / baseline_ms
        return f"{mult:.2f}× slower"
    else:
        mult = baseline_ms / current_ms
        return f"{mult:.2f}× faster"


def get_top_cases(cases: list[dict], status: str, n: int = 3) -> list[dict]:
    """Get top N cases by multiplier magnitude for given status."""
    filtered = [c for c in cases if c["status"] == status and "multiplier" in c]
    sorted_cases = sorted(filtered, key=lambda c: c["multiplier"], reverse=True)
    return [
        {
            "id": c["id"],
            "multiplier": c["multiplier"],
            "multiplier_str": c["multiplier_str"],
        }
        for c in sorted_cases[:n]
    ]


def get_profile_path(case_id: str) -> Path | None:
    """Return profile path if exists for case."""
    profile_dir = PROFILES_DIR / case_id.replace("/", "_")
    profile_file = profile_dir / "query_profile.json"
    if profile_file.exists():
        return profile_file
    return None


def strip_thread_suffix(case_id: str) -> str:
    prefix, separator, suffix = case_id.rpartition("/")
    if separator and len(suffix) > 1 and suffix[0] == "t" and suffix[1:].isdigit():
        return prefix
    return case_id


def result_thread(result: dict) -> int:
    if "threads" in result:
        return int(result["threads"])
    prefix, separator, suffix = result["id"].rpartition("/")
    if separator and prefix and len(suffix) > 1 and suffix[0] == "t" and suffix[1:].isdigit():
        return int(suffix[1:])
    return 0


def latest_index_for_schema2_baseline(results: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    indexed_threads: dict[str, int] = {}
    for result in results:
        case_id = strip_thread_suffix(result["id"])
        thread = result_thread(result)
        if case_id not in indexed or thread > indexed_threads[case_id]:
            indexed[case_id] = result
            indexed_threads[case_id] = thread
    return indexed


def generate_diff(
    baseline: dict,
    latest: dict,
    tolerance_pct: float,
    min_effect_ms: float,
) -> dict:
    """Generate diff between baseline and latest results."""
    baseline_version = baseline["schema_version"] if "schema_version" in baseline else 0
    latest_version = latest["schema_version"] if "schema_version" in latest else 0
    if baseline_version < 4 and latest_version >= 4:
        baseline_by_id = {strip_thread_suffix(r["id"]): r for r in baseline["results"]}
        latest_by_id = latest_index_for_schema2_baseline(latest["results"])
    else:
        baseline_by_id = {r["id"]: r for r in baseline["results"]}
        latest_by_id = {r["id"]: r for r in latest["results"]}

    all_ids = set(baseline_by_id.keys()) | set(latest_by_id.keys())

    cases = []
    for case_id in sorted(all_ids):
        b = baseline_by_id.get(case_id)
        l = latest_by_id.get(case_id)

        baseline_ms = b["median_ms"] if b else None
        current_ms = l["median_ms"] if l else None

        status = classify_case(baseline_ms, current_ms, tolerance_pct, min_effect_ms)

        case = {
            "id": case_id,
            "status": status,
            "baseline_ms": baseline_ms,
            "current_ms": current_ms,
        }

        if baseline_ms is not None and current_ms is not None and baseline_ms != 0:
            if current_ms > baseline_ms:
                case["multiplier"] = round(current_ms / baseline_ms, 2)
            else:
                case["multiplier"] = round(baseline_ms / current_ms, 2)
            case["multiplier_str"] = format_multiplier(baseline_ms, current_ms)

        cases.append(case)

    # Summary
    statuses = [c["status"] for c in cases]
    regressions = statuses.count("SLOWER")
    improvements = statuses.count("FASTER")

    worst_regression_mult = 1.0
    for c in cases:
        if c["status"] == "SLOWER" and c.get("multiplier", 1.0) > worst_regression_mult:
            worst_regression_mult = c["multiplier"]

    env_diff = diff_environment(baseline.get("environment"), latest.get("environment"))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {
            "tolerance_pct": tolerance_pct,
            "min_effect_ms": min_effect_ms,
        },
        "environment_diff": env_diff,
        "summary": {
            "total": len(cases),
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": statuses.count("UNCHANGED"),
            "missing_in_baseline": statuses.count("MISSING_IN_BASELINE"),
            "missing_in_latest": statuses.count("MISSING_IN_LATEST"),
            "worst_regression_multiplier": worst_regression_mult,
            "top_regressions": get_top_cases(cases, "SLOWER", 3),
            "top_improvements": get_top_cases(cases, "FASTER", 3),
        },
        "cases": cases,
    }


def print_diff(diff: dict, baseline: dict, latest: dict) -> None:
    """Print human-readable diff summary."""
    print(
        f"Comparing latest ({latest.get('generated_at', 'unknown')}) vs baseline ({baseline.get('generated_at', 'unknown')})"
    )
    print(
        f"Thresholds: tolerance={diff['thresholds']['tolerance_pct']}%, min_effect={diff['thresholds']['min_effect_ms']}ms"
    )
    print()

    # Show environment differences
    env_diff = diff.get("environment_diff", {})
    if env_diff:
        print("Environment differences:")
        for key, val in env_diff.items():
            print(f"  - {key}: {val['baseline']} -> {val['current']}")
        print()

    # Regressions (sorted by multiplier, most impactful first)
    regressions = [c for c in diff["cases"] if c["status"] == "SLOWER"]
    regressions.sort(key=lambda c: c.get("multiplier", 1.0), reverse=True)
    if regressions:
        print("Regressions:")
        for c in regressions:
            profile_path = get_profile_path(c["id"])
            profile_str = f"  [profile: {profile_path}]" if profile_path else ""
            print(f"  {c['id']:<50} {c['multiplier_str']}{profile_str}")
        print()

    # Improvements (sorted by multiplier, most impactful first)
    improvements = [c for c in diff["cases"] if c["status"] == "FASTER"]
    improvements.sort(key=lambda c: c.get("multiplier", 1.0), reverse=True)
    if improvements:
        print("Improvements:")
        for c in improvements:
            profile_path = get_profile_path(c["id"])
            profile_str = f"  [profile: {profile_path}]" if profile_path else ""
            print(f"  {c['id']:<50} {c['multiplier_str']}{profile_str}")
        print()

    # New/removed cases
    new_cases = [c for c in diff["cases"] if c["status"] == "MISSING_IN_BASELINE"]
    removed_cases = [c for c in diff["cases"] if c["status"] == "MISSING_IN_LATEST"]
    if new_cases or removed_cases:
        print("New/removed:")
        for c in new_cases:
            print(f"  {c['id']:<50} (new)")
        for c in removed_cases:
            print(f"  {c['id']:<50} (removed)")
        print()

    s = diff["summary"]
    print(f"Summary: {s['regressions']} regression(s), {s['improvements']} improvement(s), {s['unchanged']} unchanged")


def save_diff(diff: dict, path: Path) -> None:
    """Save diff to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(diff, f, indent=2)


def compare_results(
    baseline_path: Path,
    latest_path: Path,
    output_path: Path,
    tolerance_pct: float,
    min_effect_ms: float,
) -> int:
    """Compare results and return exit code."""
    latest = load_results(latest_path)
    if latest is None:
        print(f"Error: Latest results not found at {latest_path}")
        print("Run benchmarks first: uv run python bench/run_benchmarks.py")
        return 2

    baseline = load_results(baseline_path)
    if baseline is None:
        print(f"Error: Baseline not found at {baseline_path}")
        print("Save current results as baseline: uv run python bench/compare_results.py --save-baseline")
        return 2

    # Check schema version
    baseline_version = baseline.get("schema_version", 0)
    if baseline_version > SCHEMA_VERSION:
        print(f"Error: Incompatible schema version {baseline_version}, expected {SCHEMA_VERSION}")
        return 2

    diff = generate_diff(baseline, latest, tolerance_pct, min_effect_ms)

    print_diff(diff, baseline, latest)

    save_diff(diff, output_path)
    print(f"Diff written to {output_path}")

    # Return exit code based on regressions
    if diff["summary"]["regressions"] > 0:
        print(f"\nExit code: 1 (regression detected)")
        return 1

    print(f"\nExit code: 0 (no regressions)")
    return 0


def save_baseline(latest_path: Path, baseline_path: Path) -> int:
    """Copy latest to baseline."""
    latest_json = latest_path.with_suffix(".json")

    if not latest_json.exists():
        print(f"Error: No results found at {latest_json}")
        print("Run benchmarks first: uv run python bench/run_benchmarks.py")
        return 2

    baseline_json = baseline_path.with_suffix(".json")
    shutil.copy(latest_json, baseline_json)
    print(f"Saved {baseline_json}")
    print("Don't forget to commit the new baseline!")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results with baseline")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=RESULTS_DIR / "baseline",
        help="Path to baseline (without extension)",
    )
    parser.add_argument(
        "--latest",
        type=Path,
        default=RESULTS_DIR / "latest",
        help="Path to latest (without extension)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "diff.json",
        help="Path to output diff.json",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PCT,
        help=f"Regression threshold percentage (default: {DEFAULT_TOLERANCE_PCT})",
    )
    parser.add_argument(
        "--min-effect-ms",
        type=float,
        default=DEFAULT_MIN_EFFECT_MS,
        help=f"Minimum effect in ms to consider (default: {DEFAULT_MIN_EFFECT_MS})",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as new baseline",
    )
    args = parser.parse_args()

    if args.save_baseline:
        exit_code = save_baseline(args.latest, args.baseline)
    else:
        exit_code = compare_results(
            args.baseline,
            args.latest,
            args.output,
            args.tolerance,
            args.min_effect_ms,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
