# /// script
# requires-python = ">=3.11"
# dependencies = ["hypothesis>=6.100"]
# ///

from decimal import Decimal
from json import dumps as json_dumps
from json import loads as json_loads
from os import environ
from pathlib import Path
from subprocess import DEVNULL
from subprocess import PIPE
from subprocess import Popen
from sys import stderr
from typing import Any

from hypothesis import HealthCheck
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build" / "release"
EXTENSION_DIR = BUILD_DIR / "extension" / "jsono"
DEFAULT_BINARY = BUILD_DIR / "duckdb"
DEFAULT_EXTENSION = EXTENSION_DIR / "jsono.duckdb_extension"
JSONO_HEADER_SIZE = 8
JSONO_STRUCT_FIELDS = [
    "jsono_slots BLOB",
    "jsono_key_heap BLOB",
    "jsono_string_heap BLOB",
    "jsono_skips BLOB",
]
JSONO_STRUCT_TYPE = "STRUCT(" + ", ".join(JSONO_STRUCT_FIELDS) + ")"
BlobHex = tuple[str, str, str, str]

# Statement boundaries are marked by a sentinel SELECT, not a `.print`: the CLI's
# dot-command output is not ordered against the result renderer, so a `.print`
# can surface before the preceding query's rows. Routing the marker through the
# same renderer keeps it strictly after the rows. A missing marker (the pipe
# closed mid-query) is the crash signal the fuzz tests rely on.
SENTINEL_TOKEN = "JSONO_PROPERTY_SENTINEL_9f3a"
SENTINEL_LINE = f'[{{"jsono_sentinel":"{SENTINEL_TOKEN}"}}]'

MAX_EXAMPLES = int(environ.get("JSONO_PROPERTY_MAX_EXAMPLES", "300"))


class JsonoCrash(Exception):
    pass


class JsonoSession:
    # One long-lived CLI process: each query is a pipe round-trip rather than a
    # fresh process, so hypothesis can drive thousands of examples (and shrink
    # failures) without paying spawn cost per case. A crash restarts the process
    # so the next shrink attempt runs against a live session. The pipe is bytes,
    # decoded with replacement, so a stray non-UTF-8 byte in output never raises
    # on the Python side and masks a real C++ defect.
    def __init__(self, binary: Path, extension: Path) -> None:
        self.binary = binary
        self.extension = extension
        self.proc: Popen[bytes] | None = None
        self._start()

    def _start(self) -> None:
        self.proc = Popen(
            [str(self.binary), "-unsigned", "-batch", "-json"],
            stdin=PIPE,
            stdout=PIPE,
            stderr=DEVNULL,
            bufsize=0,
        )
        self._send(".mode json")
        self._send(f"LOAD '{self.extension}';")
        self._drain()

    def _send(self, line: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def _drain(self) -> list[str]:
        assert self.proc is not None and self.proc.stdout is not None
        self._send(f"SELECT '{SENTINEL_TOKEN}' AS jsono_sentinel;")
        lines: list[str] = []
        while True:
            raw = self.proc.stdout.readline()
            if raw == b"":
                raise JsonoCrash("CLI process exited before sentinel")
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line == SENTINEL_LINE:
                return lines
            lines.append(line)

    def restart(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
        self._start()

    # Evaluate one scalar expression and return its text, or None for SQL NULL /
    # empty result. The expression is cast to VARCHAR so a JSON-typed result comes
    # back as a single JSON string (the renderer would otherwise inline a JSON
    # column as a nested object, which is not round-trippable through json_loads).
    # Property queries are written to return data (never to raise), so a JsonoCrash
    # here means a genuine process death, not a SQL error.
    def value(self, expr: str) -> str | None:
        if self.proc is None or self.proc.poll() is not None:
            self.restart()
        try:
            self._send(f"SELECT CAST(({expr}) AS VARCHAR) AS v;")
            rows = self._drain()
        except JsonoCrash:
            self.restart()
            raise
        payload = "".join(rows).strip()
        if not payload or payload == "[]":
            return None
        parsed = json_loads(payload)
        if not parsed:
            return None
        (cell,) = parsed[0].values()
        return None if cell is None else str(cell)


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def jsono_blob_hex_expr(text: str) -> str:
    struct_expr = f"jsono({sql_literal(text)})::{JSONO_STRUCT_TYPE}"
    return " || '|' || ".join(
        [
            f"hex(({struct_expr}).jsono_slots)",
            f"hex(({struct_expr}).jsono_key_heap)",
            f"hex(({struct_expr}).jsono_string_heap)",
            f"hex(({struct_expr}).jsono_skips)",
        ]
    )


def jsono_struct_sql(blobs: BlobHex) -> str:
    slots, key_heap, string_heap, skips = blobs
    fields = [
        f"'jsono_slots': unhex('{slots}')",
        f"'jsono_key_heap': unhex('{key_heap}')",
        f"'jsono_string_heap': unhex('{string_heap}')",
        f"'jsono_skips': unhex('{skips}')",
    ]
    return "{" + ", ".join(fields) + "}"


def mutate_jsono_blobs(blobs: BlobHex, mutation: str) -> BlobHex:
    slots = bytearray(bytes.fromhex(blobs[0]))
    key_heap = bytearray(bytes.fromhex(blobs[1]))
    string_heap = bytearray(bytes.fromhex(blobs[2]))
    skips = bytearray(bytes.fromhex(blobs[3]))

    def fallback() -> None:
        if len(slots) > 4:
            slots[4] = 2

    if mutation == "header_version":
        fallback()
    elif mutation == "header_flags":
        if len(slots) > 5:
            slots[5] = 0
        else:
            fallback()
    elif mutation == "header_reserved":
        if len(slots) > 6:
            slots[6] = 1
        else:
            fallback()
    elif mutation == "first_slot_tag":
        if len(slots) >= JSONO_HEADER_SIZE + 8:
            slots[JSONO_HEADER_SIZE + 7] = (slots[JSONO_HEADER_SIZE + 7] & 0x0F) | 0x90
        else:
            fallback()
    elif mutation == "slots_truncate":
        if len(slots) > JSONO_HEADER_SIZE:
            slots = slots[:-1]
        else:
            fallback()
    elif mutation == "slots_extra":
        slots.append(0)
    elif mutation == "key_heap_truncate":
        if key_heap:
            key_heap = key_heap[:-1]
        else:
            fallback()
    elif mutation == "string_heap_truncate":
        if string_heap:
            string_heap = string_heap[:-1]
        else:
            fallback()
    elif mutation == "string_heap_bad_number":
        if string_heap:
            string_heap[0] = ord("x")
        else:
            fallback()
    elif mutation == "skips_truncate":
        if skips:
            skips = skips[:-1]
        else:
            fallback()
    elif mutation == "span_zero":
        if len(skips) >= 16:
            skips[12:16] = b"\x00\x00\x00\x00"
        else:
            fallback()
    elif mutation == "checkpoint_stride":
        if len(skips) >= 12:
            container_count = int.from_bytes(skips[0:4], "little")
            checkpoint_index_count = int.from_bytes(skips[4:8], "little")
            index_start = 12 + container_count * 8
            if checkpoint_index_count > 0 and len(skips) >= index_start + 12:
                skips[index_start + 8 : index_start + 10] = b"\x01\x00"
            else:
                fallback()
        else:
            fallback()
    else:
        fallback()

    slots_hex = slots.hex().upper()
    key_heap_hex = key_heap.hex().upper()
    string_heap_hex = string_heap.hex().upper()
    skips_hex = skips.hex().upper()
    return (slots_hex, key_heap_hex, string_heap_hex, skips_hex)


def numbers_as_decimal(text: str) -> Any:
    # Compare values, not their textual form: Decimal keeps full precision so a
    # DECIMAL-through-DOUBLE coercion surfaces as a value mismatch, while
    # equivalent spellings (1e3 vs 1000.0) still compare equal.
    return json_loads(text, parse_float=Decimal, parse_int=int)


def keys_in_byte_order(value: Any) -> bool:
    if isinstance(value, dict):
        keys = list(value.keys())
        if keys != sorted(keys, key=lambda key: key.encode("utf-8")):
            return False
        return all(keys_in_byte_order(child) for child in value.values())
    if isinstance(value, list):
        return all(keys_in_byte_order(child) for child in value)
    return True


SESSION = JsonoSession(
    Path(environ.get("JSONO_DUCKDB_BIN", str(DEFAULT_BINARY))),
    Path(environ.get("JSONO_EXTENSION", str(DEFAULT_EXTENSION))),
)

PROPERTY_SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
VALIDISH_PROPERTY_SETTINGS = settings(
    max_examples=max(50, MAX_EXAMPLES // 2),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

interesting_json_keys = [
    "a.b",
    "utm.source",
    "page.title",
    "$.x",
    "[0]",
    "",
    "ключ",
    "🦆",
]
json_keys = st.one_of(st.text(), st.sampled_from(interesting_json_keys))

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    # Span the i60 / int64 / uint64 / bignum boundaries the format switches on.
    st.integers(min_value=-(2**80), max_value=2**80),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(),
)

json_documents = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(json_keys, children, max_size=6),
    ),
    max_leaves=20,
).map(lambda value: json_dumps(value))

wide_object_text = "{" + ",".join(f'"k{i:02d}":{i}' for i in range(17)) + "}"
validish_json_documents = st.one_of(
    json_documents,
    st.sampled_from([wide_object_text, '{"n":1e3}', '{"s":"hello"}']),
)
validish_mutations = st.sampled_from(
    [
        "header_version",
        "header_flags",
        "header_reserved",
        "first_slot_tag",
        "slots_truncate",
        "slots_extra",
        "key_heap_truncate",
        "string_heap_truncate",
        "string_heap_bad_number",
        "skips_truncate",
        "span_zero",
        "checkpoint_stride",
    ]
)


@settings(PROPERTY_SETTINGS)
@example(text='{"a.b":1,"a":{"b":2}}')
@example(text='{"u":18446744073709551615,"big":9223372036854775000}')
@example(text='{"d":123456789012345678901234567890123456.12}')
@example(text='{"b":2,"a":1,"nested":{"y":2,"x":1}}')
@given(text=json_documents)
def test_round_trip_idempotent(text: str) -> None:
    once = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    assert once is not None
    twice = SESSION.value(f"to_json(jsono({sql_literal(once)}))")
    # Canonical output (sorted keys, normalized whitespace) must be a fixed point.
    assert once == twice, f"idempotency broke: {text!r} -> {once!r} -> {twice!r}"


@settings(PROPERTY_SETTINGS)
@example(text='{"a.b":"flat","a":{"b":"nested"}}')
@example(text='{"d":0.1,"e":123456789012345678901234567890123456.12}')
@example(text='[1,2,"x",true,null,-9223372036854775000]')
@given(text=json_documents)
def test_value_parity(text: str) -> None:
    out = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    assert out is not None
    # jsono must preserve the RFC 8259 value of the input: same structure, same
    # string content, same numeric value at full precision.
    actual = numbers_as_decimal(out)
    expected = numbers_as_decimal(text)
    assert actual == expected, f"value drift: {text!r} -> {out!r}"


@settings(PROPERTY_SETTINGS)
@given(text=json_documents)
def test_keys_sorted(text: str) -> None:
    out = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    assert out is not None
    assert keys_in_byte_order(json_loads(out)), f"object keys not byte-sorted: {out!r}"


# Mutations of valid JSON text: a corrupted document must yield a value, a NULL,
# or a clean SQL error — never a crash. Under an ASan/UBSan build this is where a
# missing bound check in the parser surfaces.
corrupt_text = st.builds(
    lambda base, idx, char: base[:idx] + char + base[idx + 1 :] if base else char,
    json_documents,
    st.integers(min_value=0, max_value=64),
    st.sampled_from(['"', "{", "}", "[", "]", "\\", ":", ",", "", "x", " "]),
)


@settings(PROPERTY_SETTINGS)
@given(text=corrupt_text)
def test_fuzz_text_no_crash(text: str) -> None:
    # try_jsono swallows malformed input as NULL; the call returning at all (the
    # sentinel came back) is the property — a crash raises JsonoCrash.
    SESSION.value(f"to_json(try_jsono({sql_literal(text)}))")


hex_blob = st.text(alphabet="0123456789ABCDEF", min_size=0, max_size=48).map(
    lambda value: value if len(value) % 2 == 0 else value + "0"
)


@settings(PROPERTY_SETTINGS)
@given(slots=hex_blob, key=hex_blob, string=hex_blob, skips=hex_blob)
def test_fuzz_blob_no_crash(slots: str, key: str, string: str, skips: str) -> None:
    # The four-BLOB physical shape is reachable from Parquet, so arbitrary bytes
    # must be rejected by the reader's bound checks, not dereferenced blindly.
    struct = (
        f"{{'jsono_slots': unhex('{slots}'), 'jsono_key_heap': unhex('{key}'), "
        f"'jsono_string_heap': unhex('{string}'), 'jsono_skips': unhex('{skips}')}}"
    )
    SESSION.value(f"to_json({struct})")


@settings(VALIDISH_PROPERTY_SETTINGS)
@example(text=wide_object_text, mutation="checkpoint_stride")
@example(text='{"n":1e3}', mutation="string_heap_bad_number")
@given(text=validish_json_documents, mutation=validish_mutations)
def test_fuzz_validish_blob_no_crash(text: str, mutation: str) -> None:
    encoded = SESSION.value(jsono_blob_hex_expr(text))
    assert encoded is not None
    parts = encoded.split("|")
    assert len(parts) == 4
    blobs = (parts[0], parts[1], parts[2], parts[3])
    struct = jsono_struct_sql(mutate_jsono_blobs(blobs, mutation))
    SESSION.value(f"jsono_validate({struct})")
    SESSION.value(f"to_json({struct})")


PROPERTIES = [
    test_round_trip_idempotent,
    test_value_parity,
    test_keys_sorted,
    test_fuzz_text_no_crash,
    test_fuzz_blob_no_crash,
    test_fuzz_validish_blob_no_crash,
]


def main() -> int:
    failures = 0
    for prop in PROPERTIES:
        name = prop.__name__
        try:
            prop()
        except JsonoCrash as crash:
            failures += 1
            print(f"CRASH  {name}: {crash}", file=stderr)
        except AssertionError as failure:
            failures += 1
            print(f"FAIL   {name}: {failure}", file=stderr)
        else:
            print(f"ok     {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
