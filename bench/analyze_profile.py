#!/usr/bin/env python3
"""Analyze samply profiles programmatically.

By default self-time is aggregated across all execution threads, which is the honest
view under DuckDB's intra-query parallelism. Pass --thread <name> for a single-thread
(per-row) view of one worker.

Usage:
    python bench/analyze_profile.py bench/results/samply/<case>.json.gz [--thread <name>]

Requires --unstable-presymbolicate when recording:
    samply record --save-only --unstable-presymbolicate --output <file>.json.gz -- <command>
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def load_profile(profile_path: Path):
    with gzip.open(profile_path, "rt") as f:
        return json.load(f)


def load_symbols(syms_path: Path):
    with open(syms_path, "r") as f:
        return json.load(f)


def build_symbol_lookup(syms: dict) -> dict:
    """Build address -> symbol lookup per library."""
    string_table = syms.get("string_table", [])
    lib_symbols = {}
    for entry in syms.get("data", []):
        debug_name = entry.get("debug_name")
        symbols = []
        for s in entry.get("symbol_table", []):
            rva = s["rva"]
            size = s.get("size", 0)
            sym_idx = s["symbol"]
            sym_name = (
                string_table[sym_idx] if sym_idx < len(string_table) else f"<{sym_idx}>"
            )
            symbols.append((rva, rva + size, sym_name))
        symbols.sort(key=lambda x: x[0])
        lib_symbols[debug_name] = symbols
    return lib_symbols


def lookup_symbol(symbols: list, addr: int) -> str | None:
    for rva_start, rva_end, name in symbols:
        if rva_start <= addr < rva_end:
            return name
    return None


def analyze_thread(
    thread: dict, lib_symbols: dict, lib_debug_names: dict
) -> tuple[dict, dict]:
    """Analyze a thread and return (self_samples, inclusive_samples) dicts."""
    samples = thread.get("samples", {})
    stack_table = thread.get("stackTable", {})
    frame_table = thread.get("frameTable", {})
    func_table = thread.get("funcTable", {})
    resource_table = thread.get("resourceTable", {})

    sample_stacks = samples.get("stack", [])
    stack_frame = stack_table.get("frame", [])
    stack_prefix = stack_table.get("prefix", [])
    frame_func = frame_table.get("func", [])
    frame_address = frame_table.get("address", [])
    func_resource = func_table.get("resource", [])
    res_lib = resource_table.get("lib", [])

    # Build frame_idx -> symbol_name lookup
    frame_symbols = {}
    for frame_idx in range(len(frame_func)):
        func_idx = frame_func[frame_idx]
        addr = frame_address[frame_idx] if frame_idx < len(frame_address) else None
        if func_idx < len(func_resource):
            res_idx = func_resource[func_idx]
            if res_idx is not None and res_idx < len(res_lib):
                lib_idx = res_lib[res_idx]
                debug_name = lib_debug_names.get(lib_idx)
                if debug_name and debug_name in lib_symbols and addr is not None:
                    sym = lookup_symbol(lib_symbols[debug_name], addr)
                    if sym:
                        frame_symbols[frame_idx] = sym
                        continue
        frame_symbols[frame_idx] = f"<frame:{frame_idx}>"

    # Count samples
    self_samples = defaultdict(int)
    inclusive_samples = defaultdict(int)

    for stack_idx in sample_stacks:
        if stack_idx is None:
            continue
        seen = set()
        current = stack_idx
        is_leaf = True
        while current is not None:
            frame_idx = stack_frame[current]
            name = frame_symbols.get(frame_idx, f"<unknown:{frame_idx}>")
            if is_leaf:
                self_samples[name] += 1
                is_leaf = False
            if name not in seen:
                inclusive_samples[name] += 1
                seen.add(name)
            current = stack_prefix[current]

    return dict(self_samples), dict(inclusive_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze samply profiles")
    parser.add_argument("profile", type=Path, help="Path to profile.json.gz")
    parser.add_argument(
        "--thread",
        default=None,
        help="Analyze a single named thread instead of the all-thread aggregate "
        "(use for a per-row view of one worker)",
    )
    parser.add_argument(
        "--min-thread-samples",
        type=int,
        default=50,
        help="Threads with fewer samples are treated as idle/infra and excluded from "
        "the aggregate (default: 50)",
    )
    parser.add_argument(
        "--top", type=int, default=30, help="Number of top functions to show"
    )
    parser.add_argument(
        "--filter", default=None, help="Filter functions containing this string"
    )
    args = parser.parse_args()

    syms_path = args.profile.with_suffix("").with_suffix(".json.syms.json")
    if not syms_path.exists():
        print(f"Error: syms file not found: {syms_path}")
        print("Re-record with --unstable-presymbolicate")
        raise SystemExit(2)

    profile = load_profile(args.profile)
    syms = load_symbols(syms_path)
    lib_symbols = build_symbol_lookup(syms)

    libs = profile.get("libs", [])
    lib_debug_names = {
        i: lib.get("debugName", lib.get("name", f"lib{i}"))
        for i, lib in enumerate(libs)
    }

    threads = profile.get("threads", [])
    by_count = sorted(
        (
            (t.get("name", "<unnamed>"), len(t.get("samples", {}).get("stack", [])), t)
            for t in threads
        ),
        key=lambda entry: -entry[1],
    )

    # Default to aggregating self-time across every execution thread. Reading a single
    # thread misleads under DuckDB's intra-query parallelism: the main thread mixes in
    # GIL/coordination overhead, while a lone worker over-weights its own slice. samply,
    # uv and idle threads carry only a handful of samples, so the floor excludes them.
    if args.thread:
        thread = next((t for t in threads if t.get("name") == args.thread), None)
        if thread is None:
            print(f"Error: thread '{args.thread}' not found")
            raise SystemExit(2)
        analyzed = [thread]
        scope = f"thread {args.thread}"
    else:
        analyzed = [t for _, count, t in by_count if count >= args.min_thread_samples]
        if not analyzed and by_count:
            analyzed = [by_count[0][2]]
        scope = (
            f"aggregate of {len(analyzed)} thread(s) "
            f"with >= {args.min_thread_samples} samples"
        )

    analyzed_ids = {id(t) for t in analyzed}
    print("Threads (samples; * = in aggregate):")
    for name, count, t in by_count:
        mark = "*" if id(t) in analyzed_ids else " "
        print(f"  {mark} {name}: {count}")

    self_samples: dict = defaultdict(int)
    inclusive_samples: dict = defaultdict(int)
    for t in analyzed:
        thread_self, thread_inclusive = analyze_thread(t, lib_symbols, lib_debug_names)
        for name, count in thread_self.items():
            self_samples[name] += count
        for name, count in thread_inclusive.items():
            inclusive_samples[name] += count

    total = sum(self_samples.values())
    print(f"\nAnalyzing {scope}")
    print(f"Total self samples: {total}\n")

    def shorten(name: str, maxlen: int = 70) -> str:
        short = name.split("(")[0] if "(" in name else name
        return short[: maxlen - 3] + "..." if len(short) > maxlen else short

    def matches_filter(name: str) -> bool:
        if not args.filter:
            return True
        return args.filter.lower() in name.lower()

    print(f"=== Self time (top {args.top}) ===")
    for name, count in sorted(self_samples.items(), key=lambda x: -x[1])[: args.top]:
        if not matches_filter(name):
            continue
        pct = 100 * count / total if total else 0
        print(f"{pct:5.1f}%  {count:5d}  {shorten(name)}")

    print(f"\n=== Inclusive time (top {args.top}) ===")
    for name, count in sorted(inclusive_samples.items(), key=lambda x: -x[1])[
        : args.top
    ]:
        if not matches_filter(name):
            continue
        pct = 100 * count / total if total else 0
        print(f"{pct:5.1f}%  {count:5d}  {shorten(name)}")


if __name__ == "__main__":
    main()
