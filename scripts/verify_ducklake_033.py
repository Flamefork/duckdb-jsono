#!/usr/bin/env python3
"""DuckLake verification for plan 033 (A3 + part-B open question).

Not a SQLLogic test because DuckLake cannot be vendored here: INSTALL needs the
network, so this runs locally against the release build:

    uv run --frozen python scripts/verify_ducklake_033.py

Checks, over a DuckLake table of shredded jsono rows evolved with
`ALTER TABLE ... ADD COLUMN j.jsono.shreds."$.b" ...`:

- A3: the OLD lane's extract folds back to a bare struct_extract (no residual
  COALESCE) after the ALTER — schema identity is broken by the evolved read
  type, so this is exactly the null-aware per-shred totality path.
- Soundness (the §5 open question): the ALTER-ADDED lane must NOT fold, even
  after new files carry it complete — old files lack the lane and their values
  live in the residual, so a fold would silently read NULLs. The value check
  fails loudly if DuckLake ever reports misleading no-NULL stats for it.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DUCKDB = REPO_ROOT / "build" / "release" / "duckdb"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def run_sql(statements: str) -> str:
    result = subprocess.run(
        [str(DUCKDB), "-unsigned"],
        input=statements,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"duckdb failed:\n{result.stderr}")
    return result.stdout


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        marker = "SELECT 'SECTION_BREAK_9f3a' AS marker;"
        out = run_sql(f"""
            INSTALL ducklake;
            LOAD ducklake;
            ATTACH 'ducklake:{tmp}/meta.ducklake' AS dl (DATA_PATH '{tmp}/data');
            CREATE TABLE dl.t AS
            SELECT jsono('{{"a":' || i || ',"b":"x' || i || '"}}',
                         shredding := {{'$.a':'BIGINT'}}) AS j
            FROM range(1000) t(i);
            PRAGMA explain_output='physical_only';
            {marker}
            EXPLAIN SELECT MIN(CAST(j->>'$.a' AS BIGINT)) FROM dl.t;
            {marker}
            ALTER TABLE dl.t ADD COLUMN j.jsono.shreds."$.b" STRUCT("value" VARCHAR, complete TINYINT);
            EXPLAIN SELECT MIN(CAST(j->>'$.a' AS BIGINT)) FROM dl.t;
            {marker}
            EXPLAIN SELECT MIN(j->>'$.b') FROM dl.t;
            {marker}
            INSERT INTO dl.t
            SELECT jsono('{{"a":' || (i + 1000) || ',"b":"y' || i || '"}}',
                         shredding := {{'$.a':'BIGINT','$.b':'VARCHAR'}})
            FROM range(1000) t(i);
            EXPLAIN SELECT MIN(j->>'$.b') FROM dl.t;
            {marker}
            SELECT MIN(CAST(j->>'$.a' AS BIGINT)) || '/' || MAX(CAST(j->>'$.a' AS BIGINT))
                   || '/' || COUNT(j->>'$.b') AS values_check FROM dl.t;
            {marker}
            SELECT column_id, contains_null, min_value, max_value
            FROM __ducklake_metadata_dl.ducklake_table_column_stats
            ORDER BY column_id;
            """)
        sections = out.split("SECTION_BREAK_9f3a")
        check(
            "pre-ALTER old-lane fold (homogeneous identity)",
            "COALESCE" not in sections[1],
        )
        check(
            "A3: post-ALTER old-lane fold (null-aware totality)",
            "COALESCE" not in sections[2],
        )
        check(
            "post-ALTER new-lane read keeps COALESCE (no rows carry it yet)",
            "COALESCE" in sections[3],
        )
        check(
            "new-lane read keeps COALESCE after new complete files (soundness)",
            "COALESCE" in sections[4],
        )
        values = re.search(r"0/1999/2000", sections[5])
        check(
            "values: old lane full range, new-lane count spans old residuals",
            values is not None,
            sections[5].strip().splitlines()[-2] if values is None else "",
        )
        print()
        print("ducklake stats for the ALTER-added lane (evidence):")
        print(sections[6].strip())

    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("\nAll DuckLake plan-033 checks passed.")


if __name__ == "__main__":
    main()
