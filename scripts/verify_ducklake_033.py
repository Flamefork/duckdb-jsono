#!/usr/bin/env python3
"""DuckLake verification for plan 033 (A3 + part-B open question).

Not a SQLLogic test because DuckLake cannot be vendored here: INSTALL needs the
network, so this runs locally against the release build:

    uv run --frozen python scripts/verify_ducklake_033.py

Checks, over a DuckLake table of shredded jsono rows evolved with
`ALTER TABLE ... ADD COLUMN j.jsono."shreds$2".c8000 VARCHAR`:

- A3: the OLD lane's extract folds back to a bare struct_extract (no residual
  COALESCE) after the ALTER — schema identity is broken by the evolved read
  type (the `$jsono$set` marker of the old files hashes the one-shred set), so
  this is exactly the null-aware per-shred totality path.
- Soundness (the §5 open question): the ALTER-ADDED lane must NOT fold, even
  after new files carry it complete — old files lack the lane and their values
  live in the residual, so a fold would silently read NULLs. The value check
  fails loudly if DuckLake ever reports misleading no-NULL stats for it.

Layout note (plan 054 + 055 + 056): a scalar shred is a BARE typed lane, so the
ALTER adds `VARCHAR`, not a value/complete pair; per-row divert information lives
in the `$jsono$spill$0` bitmap column that the table already carries, and the
layout fields are revisioned (`body$1` / `shreds$2`). A lane's field name is the
base32hex encoding of its path (`c8000` is `$.b`), so the ALTER also exercises the
plan-056 naming across the catalog. The spill bits of the old files are numbered
for the one-shred set they were written under, so after the ALTER the marker no
longer matches and they are conservatively ignored — which is why this scenario
exercises lane statistics rather than the bitmap.

The last section is the plan-056 boundary check: a JSON key containing a NUL byte
used to be unrepresentable as a lane, because a NUL in a DuckDB field name breaks
the DuckLake catalog. The codec escapes `00`, so the lane NAME is plain ASCII
(`c40fuog000` is the one-step path `a\0b`) while the raw key never reaches a field
name at all — the column crosses the catalog, the decoded path comes back exact,
and the document round-trips.
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
            ALTER TABLE dl.t ADD COLUMN j.jsono."shreds$2".c8000 VARCHAR;
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
            {marker}
            CREATE TABLE dl.nul AS
            SELECT jsono('{{"a\\u0000b":"v","z":1}}', shredding := {{'z':'BIGINT'}}) AS j;
            ALTER TABLE dl.nul ADD COLUMN j.jsono."shreds$2".c40fuog000 VARCHAR;
            SELECT (SELECT count(*) FROM (SELECT unnest(jsono_layout_lanes(j)) AS l FROM dl.nul)
                    WHERE l.path = ('$.a' || chr(0) || 'b')) AS decoded_exact,
                   to_json(j)::VARCHAR AS rt
            FROM dl.nul;
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
        decoded = re.search("│\\s+1\\s+│", sections[7]) is not None
        check(
            "NUL-bearing key: encoded lane crosses the catalog, path decodes exact",
            decoded,
            "" if decoded else sections[7].strip(),
        )
        round_tripped = '{"a\\u0000b":"v","z":1}' in sections[7]
        check(
            "NUL-bearing key: document round-trips through the evolved column",
            round_tripped,
            "" if round_tripped else sections[7].strip(),
        )

    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("\nAll DuckLake plan-033 checks passed.")


if __name__ == "__main__":
    main()
