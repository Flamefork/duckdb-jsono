import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_h1b.json"
metric = "min_ms"
rows = json.load(open(path))["results"]

# key -> {target: value}
table = {}
for r in rows:
    key = (r["operation"], r["scenario"], r["size"])
    table.setdefault(key, {})[r["target"]] = r[metric]


def pct(new, base):
    if base is None or new is None or base == 0:
        return None
    return (new - base) / base * 100.0


def fmt(x):
    return f"{x:+.1f}%" if x is not None else "  -  "


hdr = f"{'operation/scenario/size':<48} {'before':>9} {'after':>9} {'cand':>9} {'aft/bef':>8} {'cnd/aft':>8} {'cnd/bef':>8}"
print(hdr)
print("-" * len(hdr))
for key in sorted(table):
    t = table[key]
    before, after, cand = t.get("before"), t.get("after"), t.get("current")
    name = "/".join(key)
    reg = pct(after, before)
    f6 = pct(cand, after)
    net = pct(cand, before)
    flag = ""
    if f6 is not None and f6 <= -3:
        flag = "  <= cand faster"
    elif f6 is not None and f6 >= 3:
        flag = "  !! cand slower"
    bs = f"{before:>9.1f}" if before is not None else f"{'-':>9}"
    af = f"{after:>9.1f}" if after is not None else f"{'-':>9}"
    cd = f"{cand:>9.1f}" if cand is not None else f"{'-':>9}"
    print(f"{name:<48} {bs} {af} {cd} {fmt(reg):>8} {fmt(f6):>8} {fmt(net):>8}{flag}")
