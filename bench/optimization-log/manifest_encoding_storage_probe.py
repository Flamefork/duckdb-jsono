"""Decisive compact-vs-bitset manifest comparison on IDENTICAL data, no rebuild needed.

A direct `jsono(struct)` emits full/compact manifests; keyed group_merge emits indexed/bitset.
So shred the same 50k distinct-field-presence-mask rows two ways and compare the `skips`
(manifest-carrying) parquet column compressed size:
  - direct  = compact  (what manifest 4->2 would make group_merge produce)
  - merged  = bitset/indexed (current group_merge)
Each row i uses mask = i over 20 fields → ~50k distinct dense manifests (adversarial: no
dictionary dedup). This is the one workload the roadmap flagged as the simplification's risk.
"""

from pathlib import Path

import duckdb

EXT = "build/release/extension/jsono/jsono.duckdb_extension"
ROWS = 50_000

specs = []
for n in range(8):
    specs.append((f"f_str_{n}", f"('s{n}_' || (i % {37 + n * 11}))"))
for n in range(6):
    specs.append((f"f_int_{n}", f"(i % {500 + n * 250})::BIGINT"))
for n in range(4):
    specs.append((f"f_uint_{n}", f"(i % {300 + n * 150})::UBIGINT"))
for n in range(2):
    specs.append((f"f_bool_{n}", f"((i + {n}) % 2 = 0)"))
fields = [
    f"'{name}': CASE WHEN ((i >> {bit}) & 1) = 1 THEN {expr} END"
    for bit, (name, expr) in enumerate(specs)
]
struct_literal = "{" + ", ".join(fields) + "}"

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{EXT}'")
con.execute("PRAGMA threads=4")
con.execute(
    f"CREATE TABLE src AS SELECT i AS g, 0 AS ts, jsono({struct_literal}) AS v "
    f"FROM range({ROWS}) t(i)"
)
con.execute(
    "CREATE TABLE merged AS SELECT jsono_group_merge_max(v, ts) AS v FROM src GROUP BY g"
)


def skips_size(table: str, expr: str) -> tuple[int, int, str, int]:
    path = Path(f"/tmp/mcmp_{table}.parquet")
    if path.exists():
        path.unlink()
    con.execute(
        f"COPY (SELECT {expr} AS v FROM {table}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)"
    )
    rows = con.execute(
        f"""SELECT total_compressed_size, total_uncompressed_size
            FROM parquet_metadata('{path}')
            WHERE replace(path_in_schema,' ','') LIKE '%,skips'"""
    ).fetchall()
    comp = sum(r[0] for r in rows)
    uncomp = sum(r[1] for r in rows)
    return comp, uncomp, path.stat().st_size


d_comp, d_uncomp, d_file = skips_size("src", "v")
m_comp, m_uncomp, m_file = skips_size("merged", "v")

print(
    f"rows={ROWS} (each a distinct 20-field presence mask → distinct dense manifest)\n"
)
print(f"{'variant':<22}{'skips_comp':>12}{'skips_uncomp':>14}{'file':>12}")
print(f"{'direct (compact)':<22}{d_comp:>12,}{d_uncomp:>14,}{d_file:>12,}")
print(f"{'merged (bitset/idx)':<22}{m_comp:>12,}{m_uncomp:>14,}{m_file:>12,}")
print(
    f"\ncompact vs bitset skips: {d_comp:,} vs {m_comp:,} compressed "
    f"({'+' if d_comp >= m_comp else ''}{100 * (d_comp - m_comp) / m_comp:.1f}%)"
)
print(
    f"compact manifest tail bytes ~ {d_uncomp / ROWS:.0f}B/row, bitset ~ {m_uncomp / ROWS:.0f}B/row"
)
