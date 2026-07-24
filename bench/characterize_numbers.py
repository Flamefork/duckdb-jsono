#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb"]
# ///
"""One-off: characterize the JSON number population in field_sample event_properties.

Classifies every JSON number *token* (raw text, as yyjson NUMBER_AS_RAW would see
it) against the JSONO numeric ladder from docs/jsono_format.md:

  integer:
    |v| < 2^59                 -> VAL_INT60   (hot, 1 slot)
    2^59 .. 2^63-1             -> VAL_EXT/INT64
    2^63 .. 2^64-1             -> VAL_EXT/UINT64
    > 2^64-1 / < -2^63         -> VAL_EXT/NUMBER (bignum)
  fractional (from text):
    |mantissa| < 2^53 and scale <= 22 -> VAL_DEC60 (hot, 1 slot)
    otherwise                          -> VAL_EXT/NUMBER (high-precision)

Reads raw JSON text from the parquet (no jsono extension needed) and scans number
literals with a JSON-number regex. This is deliberately text-level: it is the
exact input the parser classifier will face.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import duckdb

FIELD_SAMPLE_EVENTS = Path(__file__).parent / "data" / "field_sample" / "events_nested.parquet"

# JSON number token: optional '-', int part, optional fraction, optional exponent.
# Anchored to value positions: preceded by [:,\[ \t\n] or start, to avoid matching
# digits inside string contents we strip strings first.
NUMBER_RE = re.compile(rb"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Remove JSON string literals (including escapes) so digits inside strings/keys
# are not counted as numbers.
STRING_RE = re.compile(rb'"(?:[^"\\]|\\.)*"')

TWO_59 = 2**59
TWO_63 = 2**63
TWO_64 = 2**64
TWO_53 = 2**53


def classify_token(tok: bytes) -> str:
    text = tok.decode("ascii")
    has_dot = "." in text
    has_exp = "e" in text or "E" in text
    if not has_dot and not has_exp:
        # integer literal
        v = int(text)
        a = abs(v)
        if a < TWO_59:
            return "int60"
        if v >= 0:
            if v < TWO_63:
                return "int64"
            if v < TWO_64:
                return "uint64"
            return "bignum"
        # negative
        if a <= TWO_63:
            return "int64"
        return "bignum"
    # fractional / exponent form: decompose to mantissa * 10^-scale via Decimal-free path
    return classify_fractional(text)


def classify_fractional(text: str) -> str:
    neg = text.startswith("-")
    if neg:
        text = text[1:]
    if "e" in text or "E" in text:
        mant_str, exp_str = re.split("[eE]", text)
        exp = int(exp_str)
    else:
        mant_str, exp = text, 0
    if "." in mant_str:
        int_part, frac_part = mant_str.split(".")
    else:
        int_part, frac_part = mant_str, ""
    digits = (int_part + frac_part).lstrip("0") or "0"
    # scale before applying exponent: number = digits / 10^len(frac_part)
    scale = len(frac_part) - exp
    mantissa = int(digits)
    # normalize trailing zeros in mantissa against positive scale
    while scale > 0 and mantissa % 10 == 0 and mantissa != 0:
        mantissa //= 10
        scale -= 1
    if scale < 0:
        # integer-valued after exponent; treat as integer ladder
        mantissa *= 10 ** (-scale)
        scale = 0
        a = mantissa
        if a < TWO_59:
            return "int60_via_exp"
        if a < TWO_63:
            return "int64_via_exp"
        if a < TWO_64:
            return "uint64_via_exp"
        return "bignum_via_exp"
    if mantissa < TWO_53 and scale <= 22:
        return "dec60"
    return "highprec"


def main() -> int:
    if not FIELD_SAMPLE_EVENTS.exists():
        print(
            f"BLOCKER: field_sample parquet not found: {FIELD_SAMPLE_EVENTS}",
            file=sys.stderr,
        )
        return 2

    conn = duckdb.connect()
    rows = conn.execute(f"SELECT event_properties::VARCHAR FROM read_parquet('{FIELD_SAMPLE_EVENTS}')").fetchall()
    conn.close()

    counts: Counter[str] = Counter()
    total_numbers = 0
    sample_highprec: list[str] = []
    sample_big: list[str] = []
    for (text,) in rows:
        if text is None:
            continue
        stripped = STRING_RE.sub(b"", text.encode("utf-8"))
        for m in NUMBER_RE.finditer(stripped):
            tok = m.group(0)
            if tok in (b"", b"-"):
                continue
            cls = classify_token(tok)
            counts[cls] += 1
            total_numbers += 1
            if cls == "highprec" and len(sample_highprec) < 10:
                sample_highprec.append(tok.decode())
            if cls.startswith("bignum") and len(sample_big) < 10:
                sample_big.append(tok.decode())

    print(f"rows scanned: {len(rows):,}")
    print(f"total number tokens: {total_numbers:,}")
    print()
    order = [
        "int60",
        "int60_via_exp",
        "int64",
        "int64_via_exp",
        "uint64",
        "uint64_via_exp",
        "dec60",
        "highprec",
        "bignum",
        "bignum_via_exp",
    ]
    print(f"{'class':<18}{'count':>14}{'pct':>10}")
    for cls in order:
        c = counts.get(cls, 0)
        pct = (c / total_numbers * 100) if total_numbers else 0.0
        print(f"{cls:<18}{c:>14,}{pct:>9.4f}%")
    leftover = set(counts) - set(order)
    for cls in sorted(leftover):
        c = counts[cls]
        pct = (c / total_numbers * 100) if total_numbers else 0.0
        print(f"{cls:<18}{c:>14,}{pct:>9.4f}%  (unexpected)")
    print()

    int_total = sum(
        counts.get(k, 0)
        for k in (
            "int60",
            "int60_via_exp",
            "int64",
            "int64_via_exp",
            "uint64",
            "uint64_via_exp",
        )
    )
    frac_total = counts.get("dec60", 0) + counts.get("highprec", 0)
    big_total = counts.get("bignum", 0) + counts.get("bignum_via_exp", 0)
    print(f"integers:      {int_total:>14,} ({int_total/total_numbers*100:.2f}%)")
    print(f"fractional:    {frac_total:>14,} ({frac_total/total_numbers*100:.2f}%)")
    print(f"  dec60-fit:   {counts.get('dec60',0):>14,}")
    print(f"  high-prec:   {counts.get('highprec',0):>14,}")
    print(f"bignum>2^64:   {big_total:>14,} ({big_total/total_numbers*100:.4f}%)")
    if sample_highprec:
        print("\nhigh-precision samples:", sample_highprec)
    if sample_big:
        print("bignum samples:", sample_big)
    return 0


if __name__ == "__main__":
    sys.exit(main())
