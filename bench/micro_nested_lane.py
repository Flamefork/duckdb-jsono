# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb"]
# ///
"""Self-contained micro-benchmark for nested-shred-lane read speed.

Builds a synthetic ~1M-row jsono column with a top-level scalar field (`pageViewID`, a BIGINT shred)
and a NESTED struct field (`$.URL.scheme`, a VARCHAR shred), writes it to zstd Parquet, and times a
filter scan on each. It reproduces the gap that motivated the nested-lane optimization and validates
the fix: a nested shredded path served by `->>` should scan at top-level-scalar-lane speed.

Realistic wide-data shape: a SEPARATE typed shred (`n`) spills on a handful of rows (a value that
cannot fit its BIGINT shred stays in the residual) — exactly what happens on real ~85-field telemetry
where some field, somewhere, fails its typed shred. Under the old single global value-complete marker
one such spill flipped the whole column off, demoting EVERY non-null-free lane (including
`$.URL.scheme`, absent on some rows) to a per-row `jsono_extract_string` over the reconstructed
residual; with the per-shred `complete` flag the spill only touches `n`, so the nested
VARCHAR lane stays a bare columnar read.

Run:  uv run --frozen python bench/micro_nested_lane.py
"""

import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION = (
    REPO_ROOT / "build" / "release" / "extension" / "jsono" / "jsono.duckdb_extension"
)
PARQUET = REPO_ROOT / "bench" / "data" / "micro_nested_lane.parquet"
ROWS = 1_000_000
HOT_RUNS = 7

# Each case: (label, predicate, expect_lane). `count(*)` over the predicate isolates the scan cost.
# `expect_lane` marks the reads that SHOULD lower to a bare columnar lane scan; the controls reach
# into the residual / reconstruct the whole value and are expected to be much slower.
CASES = [
    (
        "scalar lane     ->>'pageViewID'",
        "(root_params->>'pageViewID')::bigint = 424242",
        True,
    ),
    (
        "nested lane     ->>'$.URL.scheme'",
        "(root_params->>'$.URL.scheme') in ('goal', 'autogoal')",
        True,
    ),
    # Controls: the slow paths a nested read takes when it does NOT hit the lane.
    (
        "nested two-step (->'URL')->>",
        "((root_params->'URL')->>'scheme') in ('goal', 'autogoal')",
        False,
    ),
    ("full reconstruct ::varchar", "length(root_params::varchar) = 0", False),
]

# The reconstruct / residual machinery a non-lane read compiles to; absence of all of these means the
# scan reads the lane column directly.
RESIDUAL_MARKERS = (
    "COALESCE",
    "jsono_extract_string",
    "__jsono_internal",
    "strip_manifest",
)


def connect() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(config={"allow_unsigned_extensions": True})
    conn.execute(f"LOAD '{EXTENSION}'")
    return conn


def build(conn: duckdb.DuckDBPyConnection) -> None:
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(
        f"""
        COPY (
            SELECT jsono(
                json::VARCHAR,
                shredding := {{'pageViewID': 'BIGINT', 'n': 'BIGINT', '$.URL.scheme': 'VARCHAR'}}
            ) AS root_params
            FROM (
                SELECT json_object(
                    'pageViewID', i + 1,
                    -- `n` spills on a few rows (a string where a BIGINT shred is declared), clearing
                    -- only its own per-shred `complete` flag, never touching the nested lane.
                    'n', CASE WHEN i % 100000 = 7 THEN to_json('oops') ELSE to_json(i) END,
                    -- $.URL.scheme present on ~70% of rows (URL object omitted on the rest).
                    'URL', CASE WHEN (i * 2654435761) % 10 < 7
                        THEN json_object('scheme', CASE WHEN i % 3 = 0 THEN 'goal' ELSE 'https' END, 'path', '/p')
                    END,
                    'tag', 'x'
                ) AS json
                FROM range({ROWS}) t(i)
            )
        ) TO '{PARQUET}' (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 8);
        """
    )


def is_bare_lane(conn: duckdb.DuckDBPyConnection, predicate: str) -> bool:
    plan = "\n".join(
        row[1]
        for row in conn.execute(
            f"EXPLAIN SELECT count(*) FROM read_parquet('{PARQUET}') WHERE {predicate}"
        ).fetchall()
    )
    return not any(marker in plan for marker in RESIDUAL_MARKERS)


def bench(conn: duckdb.DuckDBPyConnection, predicate: str) -> float:
    query = f"SELECT count(*) FROM read_parquet('{PARQUET}') WHERE {predicate}"
    conn.execute(query).fetchone()  # warm
    times = []
    for _ in range(HOT_RUNS):
        start = time.perf_counter()
        conn.execute(query).fetchone()
        times.append((time.perf_counter() - start) * 1000.0)
    return min(times)


def main() -> int:
    if not EXTENSION.exists():
        print(
            f"extension not built: {EXTENSION}\nrun `uv run make release` first",
            file=sys.stderr,
        )
        return 1
    conn = connect()
    build(conn)
    print(
        f"rows={ROWS:,}  parquet={PARQUET.stat().st_size / 1e6:.1f} MB  (min of {HOT_RUNS} hot runs)\n"
    )
    scalar_ms = None
    not_lowered = False
    for label, predicate, expect_lane in CASES:
        ms = bench(conn, predicate)
        if scalar_ms is None:
            scalar_ms = ms
        ratio = ms / scalar_ms if scalar_ms else 1.0
        if expect_lane:
            bare = is_bare_lane(conn, predicate)
            not_lowered = not_lowered or not bare
            tag = "lane" if bare else "NOT LOWERED TO LANE"
        else:
            tag = "control: residual/reconstruct"
        print(f"{label:34s} {ms:9.1f} ms   {ratio:6.1f}x scalar   [{tag}]")
    return 1 if not_lowered else 0


if __name__ == "__main__":
    raise SystemExit(main())
