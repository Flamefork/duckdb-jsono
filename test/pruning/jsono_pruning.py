# Row-group pruning regression guard. Shred-leaf filter pushdown is a perf
# property the SQLLogic suite cannot see (SQLLogic has no access to the profiler
# metrics), and it is fragile: a format change that makes a shred non-leaf, or a
# change to the `->>` -> struct_extract optimizer rewrite, would silently turn a
# pruned scan back into a full scan with no test failure. This drives the built
# CLI (same artifact the property harness uses, no python duckdb / ABI pin),
# writes a clustered shredded Parquet, and asserts via the DuckDB 1.5.4
# OPERATOR_ROW_GROUPS_SCANNED / OPERATOR_TOTAL_ROW_GROUPS_TO_SCAN metrics that a
# selective filter on a shred leaf prunes row groups exactly like the native
# control column does. See bench/PROFILING.md for the manual recipe.

from json import loads as json_loads
from os import environ
from pathlib import Path
from subprocess import run
from sys import stderr
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(environ.get("JSONO_DUCKDB_BIN", REPO_ROOT / "build" / "release" / "duckdb"))
EXTENSION = Path(
    environ.get(
        "JSONO_EXTENSION",
        REPO_ROOT / "build" / "release" / "extension" / "jsono" / "jsono.duckdb_extension",
    )
)

ROWS = 120_000
ROW_GROUP_SIZE = 12_000  # -> 10 row groups
SELECTIVE_VALUE = 65_000  # mid-group, away from any row-group boundary

# These two operator metrics are not in the default/detailed profiling set, so
# they must be requested explicitly; they appear only in the JSON profile.
CUSTOM_SETTINGS = (
    '{"OPERATOR_TYPE":"true","OPERATOR_ROW_GROUPS_SCANNED":"true","OPERATOR_TOTAL_ROW_GROUPS_TO_SCAN":"true"}'
)


def run_sql(sql: str) -> None:
    result = run([str(BINARY), "-unsigned", "-batch"], input=sql, text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"duckdb exited {result.returncode}:\n{result.stderr.strip()}")


def scan_metric(profile_path: Path) -> tuple[int, int]:
    found: list[tuple[int, int]] = []

    def walk(node: dict) -> None:
        if node.get("operator_type") == "TABLE_SCAN":
            scanned = node.get("operator_row_groups_scanned")
            total = node.get("operator_total_row_groups_to_scan")
            if scanned is not None and total is not None:
                found.append((int(scanned), int(total)))
        for child in node.get("children", []):
            walk(child)

    walk(json_loads(profile_path.read_text()))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one parquet scan in profile, got {found}")
    return found[0]


def profile_query(parquet: Path, profile_path: Path, query: str) -> tuple[int, int]:
    run_sql(
        f"LOAD '{EXTENSION}';\n"
        f"PRAGMA enable_profiling='json';\n"
        f"PRAGMA custom_profiling_settings='{CUSTOM_SETTINGS}';\n"
        f"PRAGMA profiling_output='{profile_path}';\n"
        f"{query};\n"
    )
    return scan_metric(profile_path)


def main() -> None:
    if not BINARY.exists():
        raise SystemExit(f"duckdb binary not found: {BINARY} (build it first)")
    if not EXTENSION.exists():
        raise SystemExit(f"jsono extension not found: {EXTENSION} (build it first)")

    with TemporaryDirectory(prefix="jsono_prune_") as tmp_name:
        tmp = Path(tmp_name)
        parquet = tmp / "clustered.parquet"
        profile = tmp / "profile.json"

        # Numeric $.seq (BIGINT shred) + string $.kind (VARCHAR shred, zero-padded
        # so lexicographic order matches numeric order) + a plain native control
        # column, all clustered by the monotonic key so each row group holds a
        # disjoint range.
        run_sql(
            f"LOAD '{EXTENSION}';\n"
            "COPY (\n"
            "  SELECT\n"
            "    jsono('{\"seq\":' || i || ',\"kind\":\"' || printf('%08d', i) || '\"}',\n"
            "          shredding := {'$.seq':'BIGINT','$.kind':'VARCHAR'}) AS t,\n"
            "    i AS seq_native\n"
            f"  FROM range({ROWS}) r(i)\n"
            "  ORDER BY i\n"
            f") TO '{parquet}' (FORMAT parquet, ROW_GROUP_SIZE {ROW_GROUP_SIZE}, OVERWRITE_OR_IGNORE true);\n"
        )

        padded = f"{SELECTIVE_VALUE:08d}"
        typed_scanned, typed_total = profile_query(
            parquet,
            profile,
            f"SELECT count(*) FROM '{parquet}' WHERE CAST(t->>'$.seq' AS BIGINT) = {SELECTIVE_VALUE}",
        )
        varchar_scanned, varchar_total = profile_query(
            parquet,
            profile,
            f"SELECT count(*) FROM '{parquet}' WHERE t->>'$.kind' = '{padded}'",
        )
        native_scanned, native_total = profile_query(
            parquet,
            profile,
            f"SELECT count(*) FROM '{parquet}' WHERE seq_native = {SELECTIVE_VALUE}",
        )
        full_scanned, full_total = profile_query(
            parquet,
            profile,
            f"SELECT count(t->>'$.kind') FROM '{parquet}'",
        )

        failures: list[str] = []

        # The data must actually span multiple row groups, or "pruning" is vacuous.
        if full_total < 2:
            failures.append(f"test data has only {full_total} row group(s); raise ROWS or lower ROW_GROUP_SIZE")

        # A typed shred prunes only when filtered in its own type.
        if not typed_scanned < typed_total:
            failures.append(
                f"typed shred CAST(... AS BIGINT) filter did not prune: scanned {typed_scanned}/{typed_total}"
            )
        # A VARCHAR shred prunes on the plain textual `->>`.
        if not varchar_scanned < varchar_total:
            failures.append(f"VARCHAR shred ->> filter did not prune: scanned {varchar_scanned}/{varchar_total}")
        # The native control proves the data is prunable at all.
        if not native_scanned < native_total:
            failures.append(f"native control filter did not prune: scanned {native_scanned}/{native_total}")
        # The shred must prune *identically* to the native column it mirrors.
        if typed_scanned != native_scanned:
            failures.append(
                f"shred pruning diverged from native control: typed {typed_scanned} vs native {native_scanned}"
            )
        # No prunable filter -> the whole file is read; this also confirms the
        # metric reflects real scanning rather than always reading nothing.
        if full_scanned != full_total:
            failures.append(f"unfiltered scan unexpectedly pruned: scanned {full_scanned}/{full_total}")

        print(
            f"typed shred {typed_scanned}/{typed_total}, "
            f"varchar shred {varchar_scanned}/{varchar_total}, "
            f"native {native_scanned}/{native_total}, "
            f"full scan {full_scanned}/{full_total}"
        )

        if failures:
            print("\nROW-GROUP PRUNING REGRESSION:", file=stderr)
            for failure in failures:
                print(f"  - {failure}", file=stderr)
            raise SystemExit(1)

        print("Row-group pruning guard passed.")


if __name__ == "__main__":
    main()
