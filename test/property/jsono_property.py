from decimal import Decimal
from json import dumps as json_dumps
from json import loads as json_loads
from os import environ
from pathlib import Path
from select import select
from subprocess import PIPE
from subprocess import Popen
from sys import stderr
from tempfile import TemporaryFile
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
BlobHex = tuple[str, str, str, str]

# Statement boundaries are marked by a sentinel SELECT, not a `.print`: the CLI's
# dot-command output is not ordered against the result renderer, so a `.print`
# can surface before the preceding query's rows. Routing the marker through the
# same renderer keeps it strictly after the rows. A missing marker (the pipe
# closed mid-query) is the crash signal the fuzz tests rely on.
SENTINEL_TOKEN = "JSONO_PROPERTY_SENTINEL_9f3a"
SENTINEL_LINE = f'[{{"jsono_sentinel":"{SENTINEL_TOKEN}"}}]'

MAX_EXAMPLES = int(environ.get("JSONO_PROPERTY_MAX_EXAMPLES", "300"))

# Ceiling on how long one statement may go without producing output. A property query runs in
# milliseconds, so this only fires on a genuine stall: a CLI that treats the sent SQL as an
# incomplete statement (it blocks reading stdin for a continuation it never gets while we block
# reading stdout) deadlocks the pipe round-trip forever. The timeout converts that into a bounded
# JsonoCrash with the stuck statement and stderr tail, which restarts the process — the same
# recovery path as an outright process death.
DRAIN_TIMEOUT_SECONDS = float(environ.get("JSONO_PROPERTY_DRAIN_TIMEOUT", "30"))


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
        self.stderr_file: Any = None
        self._start()

    def _start(self) -> None:
        # stderr goes to a temp file, not DEVNULL: SQL errors are still dropped from the
        # result stream (only stdout is parsed), but when the process dies the file holds
        # its last words — an ASan/UBSan stack trace under the sanitizer build — which the
        # crash message then surfaces. Detaching it entirely turned every sanitizer crash
        # into an undiagnosable "exited before sentinel" with no trace.
        self.stderr_file = TemporaryFile()
        self.proc = Popen(
            [str(self.binary), "-unsigned", "-batch", "-json"],
            stdin=PIPE,
            stdout=PIPE,
            stderr=self.stderr_file,
            bufsize=0,
        )
        self._send(".mode json")
        self._send(f"LOAD '{self.extension}';")
        self._drain()

    def _stderr_tail(self) -> str:
        if self.stderr_file is None:
            return ""
        self.stderr_file.seek(0)
        text = self.stderr_file.read().decode("utf-8", "replace")
        # The CLI's stderr also carries the pre-crash SQL errors of a long-lived session;
        # the actionable part is the final sanitizer report. A report begins with its header
        # and prints the top access stack *before* the long shadow-byte dump and SUMMARY, so
        # a fixed tail can drop the most useful frames. Anchor on the last report header and
        # return everything from there — the report is self-bounded, the noise before it is not.
        markers = ("ERROR: AddressSanitizer", "ERROR: LeakSanitizer", "runtime error:")
        start = max(text.rfind(marker) for marker in markers)
        if start < 0:
            return text[-8000:]
        return text[text.rfind("\n", 0, start) + 1 :]

    def _send(self, line: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def _drain(self, context: str = "") -> list[str]:
        assert self.proc is not None and self.proc.stdout is not None
        self._send(f"SELECT '{SENTINEL_TOKEN}' AS jsono_sentinel;")
        lines: list[str] = []
        while True:
            ready, _, _ = select([self.proc.stdout], [], [], DRAIN_TIMEOUT_SECONDS)
            if not ready:
                tail = self._stderr_tail()
                detail = f"\n--- CLI stderr tail ---\n{tail}" if tail.strip() else ""
                stmt = f"\n--- stuck statement ---\n{context}" if context else ""
                raise JsonoCrash(
                    f"CLI produced no output for {DRAIN_TIMEOUT_SECONDS:.0f}s "
                    f"(hang or incomplete statement){stmt}{detail}"
                )
            raw = self.proc.stdout.readline()
            if raw == b"":
                tail = self._stderr_tail()
                detail = f"\n--- CLI stderr tail ---\n{tail}" if tail.strip() else ""
                raise JsonoCrash(f"CLI process exited before sentinel{detail}")
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
            stmt = f"SELECT CAST(({expr}) AS VARCHAR) AS v;"
            self._send(stmt)
            rows = self._drain(stmt)
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

    # Run one non-SELECT statement (DDL/DML/SET) for its side effect; an error is dropped
    # (stderr is detached) — properties that need to observe one read through value().
    def statement(self, sql: str) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self.restart()
        try:
            self._send(sql)
            self._drain(sql)
        except JsonoCrash:
            self.restart()
            raise


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def jsono_blob_hex_expr(text: str) -> str:
    body_expr = f'jsono({sql_literal(text)})."jsono".body'
    return " || '|' || ".join(
        [
            f"hex(({body_expr}).slots)",
            f"hex(({body_expr}).key_heap)",
            f"hex(({body_expr}).string_heap)",
            f"hex(({body_expr}).skips)",
        ]
    )


def jsono_struct_sql(blobs: BlobHex) -> str:
    slots, key_heap, string_heap, skips = blobs
    body_fields = [
        f"'slots': unhex('{slots}')",
        f"'key_heap': unhex('{key_heap}')",
        f"'string_heap': unhex('{string_heap}')",
        f"'skips': unhex('{skips}')",
    ]
    body = "{" + ", ".join(body_fields) + "}"
    return "{'jsono': {'body': " + body + "}}"


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


def shredded_blob_hex_expr(text: str, spec_sql: str) -> str:
    body_expr = f'jsono({sql_literal(text)}, shredding := {spec_sql})."jsono".body'
    return " || '|' || ".join(
        [
            f"hex(({body_expr}).slots)",
            f"hex(({body_expr}).key_heap)",
            f"hex(({body_expr}).string_heap)",
            f"hex(({body_expr}).skips)",
        ]
    )


def manifest_offset(skips: bytes) -> int:
    # Mirrors ParseHeader: the manifest is the skips tail after the checkpoint sections.
    if len(skips) < 12:
        return len(skips)
    container_count = int.from_bytes(skips[0:4], "little")
    checkpoint_index_count = int.from_bytes(skips[4:8], "little")
    checkpoint_count = int.from_bytes(skips[8:12], "little")
    offset = 12 + container_count * 16 + checkpoint_index_count * 12 + checkpoint_count * 8
    return min(offset, len(skips))


def mutate_manifest(skips_hex: str, mutation: str) -> str:
    skips = bytearray(bytes.fromhex(skips_hex))
    offset = manifest_offset(bytes(skips))
    if mutation == "count_inflate" and len(skips) >= offset + 4:
        count = int.from_bytes(skips[offset : offset + 4], "little")
        skips[offset : offset + 4] = ((count + 1) & 0xFFFFFFFF).to_bytes(4, "little")
    elif mutation == "count_huge" and len(skips) >= offset + 4:
        skips[offset : offset + 4] = b"\xff\xff\xff\xff"
    elif mutation == "path_len_huge" and len(skips) >= offset + 6:
        skips[offset + 4 : offset + 6] = b"\xff\xff"
    elif mutation == "truncate_mid" and len(skips) > offset + 5:
        skips = skips[: offset + 5]
    elif mutation == "truncate_tail" and len(skips) > offset:
        skips = skips[:-1]
    elif mutation == "append_junk":
        skips.extend(b"\x07garbage")
    elif len(skips) > offset:
        skips[-1] ^= 0xFF
    return skips.hex().upper()


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
# Repeated/nested empty objects drive the DOM writer's shape-cache replay with zero keys
# against a still-empty index buffer — the format-v4 null-pointer-offset UB.
@example(text="[{},{}]")
@example(text='{"a":{},"b":{}}')
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
@example(text='{"a":{},"b":{}}')
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
        f"{{'jsono': {{'body': {{'slots': unhex('{slots}'), 'key_heap': unhex('{key}'), "
        f"'string_heap': unhex('{string}'), 'skips': unhex('{skips}')}}}}}}"
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


# Shredding properties run on object documents whose top-level keys are plain identifiers
# (shred paths are spec field names; quoting rules are not the property under test).
shred_keys = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8).filter(lambda key: key != "body")
shred_documents = st.dictionaries(shred_keys, json_scalars, min_size=1, max_size=6)
shred_types = st.sampled_from(["VARCHAR", "BIGINT", "DOUBLE", "BOOLEAN"])


def shred_spec_sql(spec: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{path}': '{stype}'" for path, stype in spec.items()) + "}"


@settings(PROPERTY_SETTINGS)
@given(doc=shred_documents, data=st.data())
def test_shred_lossless(doc: dict[str, Any], data: Any) -> None:
    # Shredding must never change the logical value, for any document and any spec —
    # including paths absent from the document and shred types the values do not fit
    # (the lossless gate keeps those in the residual).
    text = json_dumps(doc)
    keys = sorted(doc.keys())
    paths = data.draw(
        st.lists(
            st.sampled_from(keys + ["missing_path"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    spec = {path: data.draw(shred_types) for path in paths}
    plain = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    shredded = SESSION.value(f"to_json(jsono({sql_literal(text)}, shredding := {shred_spec_sql(spec)}))")
    assert plain == shredded, f"shredding changed the value: {text!r} spec {spec!r}: {plain!r} -> {shredded!r}"


@settings(PROPERTY_SETTINGS)
@given(doc=shred_documents, data=st.data())
def test_reshred_lossless(doc: dict[str, Any], data: Any) -> None:
    # Reshredding a shredded value to another spec (the single-pass superset path and the
    # reconstruct fallback alike) must preserve the logical value.
    text = json_dumps(doc)
    keys = sorted(doc.keys())
    first_paths = data.draw(st.lists(st.sampled_from(keys), min_size=1, max_size=3, unique=True))
    second_paths = data.draw(
        st.lists(
            st.sampled_from(keys + ["missing_path"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    first = shred_spec_sql({path: data.draw(shred_types) for path in first_paths})
    second = shred_spec_sql({path: data.draw(shred_types) for path in second_paths})
    plain = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    reshredded = SESSION.value(
        f"to_json(jsono(jsono({sql_literal(text)}, shredding := {first}), shredding := {second}))"
    )
    assert plain == reshredded, f"reshred changed the value: {text!r} {first} -> {second}: {plain!r} -> {reshredded!r}"


# merge_patch over a SHREDDED base with PLAIN patch arguments must fold identically to the same
# PLAIN base with the same patches. This fuzzes every way a plain patch can interact with a lane:
# disjoint append, a value colliding into the
# residual, an RFC 7396 null that deletes a lane (no trace in the folded residual), and a non-object
# patch that replaces the whole document. The plain-base fold is the RFC 7396 reference.
merge_patch_keys = st.sampled_from(["a", "b", "c", "d", "e"])
merge_patch_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(min_value=-100, max_value=100), st.text(max_size=4)
)
merge_patch_objects = st.dictionaries(
    merge_patch_keys,
    st.one_of(merge_patch_scalars, st.dictionaries(merge_patch_keys, merge_patch_scalars, max_size=2)),
    min_size=0,
    max_size=4,
)
merge_patch_args = st.lists(
    st.one_of(merge_patch_objects, merge_patch_scalars, st.lists(merge_patch_scalars, max_size=3)),
    min_size=1,
    max_size=3,
)
merge_patch_base = st.dictionaries(merge_patch_keys, merge_patch_scalars, min_size=1, max_size=5)


@settings(PROPERTY_SETTINGS)
@example(base={"a": 1}, patches=[{"a": None}], spec_keys=["a"])  # RFC 7396 null deletes a lane
@example(base={"a": 1}, patches=[5], spec_keys=["a"])  # non-object patch replaces the document
@example(base={"a": 1, "b": 2}, patches=[{"c": 3}], spec_keys=["a", "b"])  # disjoint append, fast path
@example(base={"a": 1}, patches=[{"a": 2}], spec_keys=["a"])  # value collides into the residual
@example(base={"a": 1}, patches=[{"a": None}, {"a": 9}], spec_keys=["a"])  # delete then re-add a lane
@given(
    base=merge_patch_base,
    patches=merge_patch_args,
    spec_keys=st.lists(merge_patch_keys, min_size=1, max_size=5, unique=True),
)
def test_merge_patch_shredded_plain_parity(base: dict[str, Any], patches: list[Any], spec_keys: list[str]) -> None:
    base_text = json_dumps(base)
    spec_keys = [key for key in spec_keys if key in base]
    if not spec_keys:
        return
    spec = shred_spec_sql({key: "VARCHAR" for key in spec_keys})
    patch_args = ", ".join(f"jsono({sql_literal(json_dumps(patch))})" for patch in patches)
    reference = SESSION.value(f"to_json(jsono_merge_patch(jsono({sql_literal(base_text)}), {patch_args}))")
    candidate = SESSION.value(
        f"to_json(jsono_merge_patch(jsono({sql_literal(base_text)}, shredding := {spec}), {patch_args}))"
    )
    assert reference == candidate, (
        f"shredded-base merge diverged from plain: base {base_text!r} patches {patches!r} "
        f"spec {spec_keys!r}: plain {reference!r} != shredded {candidate!r}"
    )


# A jsono({...}) patch with a NESTED object auto-shreds its scalars into '$.path' lanes.
# Merging it onto a shredded base puts nested shreds in the
# merged set, which the fast path copies through as lanes overlaid at their paths. The same merge
# expressed with PLAIN text patches over a plain base is the RFC 7396 reference; the two must agree
# (disjoint nested add, nested value collision, a scalar replacing the nested parent, a deeper nest).
def struct_literal_sql(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return sql_literal(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{sql_literal(k)}: {struct_literal_sql(v)}" for k, v in value.items()) + "}"
    raise ValueError(f"unsupported struct literal value: {value!r}")


# A key may be a scalar in one patch and a one-level object in another, drawn from one shared pool:
# this exercises scalar↔object transitions, where the merged shred set acquires prefix-overlapping
# paths (a scalar `a` AND a nested `$.a.x`) — a valid sparse multi-shape layout. The to_json overlay
# reconstructs such overlapping paths via the independent-overlay fallback, so the shredded fold must
# still agree with the plain RFC 7396 reference.
merge_patch_struct_key = st.sampled_from(["a", "b", "c"])
merge_patch_struct_scalars = st.one_of(st.booleans(), st.integers(min_value=-100, max_value=100), st.text(max_size=4))
merge_patch_struct_patch = st.dictionaries(
    merge_patch_struct_key,
    st.one_of(
        merge_patch_struct_scalars,
        st.dictionaries(merge_patch_struct_key, merge_patch_struct_scalars, min_size=1, max_size=3),
    ),
    min_size=1,
    max_size=4,
)


@settings(PROPERTY_SETTINGS)
@example(base={"a": 1}, patches=[{"a": {"a": 1, "b": 2}}], spec_keys=["a"])  # disjoint nested add
@example(base={"a": 1}, patches=[{"a": {"a": 1}}, {"a": 9}], spec_keys=["a"])  # nested then top-level lane
@example(base={"a": 1}, patches=[{"a": {"a": 1}}, {"a": {"a": 2}}], spec_keys=["a"])  # nested collision
@example(base={"a": None}, patches=[{"a": {"a": False}}, {"a": False}], spec_keys=["a"])  # scalar↔object↔scalar
@given(
    base=merge_patch_base,
    patches=st.lists(merge_patch_struct_patch, min_size=1, max_size=3),
    spec_keys=st.lists(merge_patch_struct_key, min_size=1, max_size=3, unique=True),
)
def test_merge_patch_auto_shred_patch_parity(
    base: dict[str, Any], patches: list[dict[str, Any]], spec_keys: list[str]
) -> None:
    base_text = json_dumps(base)
    spec_keys = [key for key in spec_keys if key in base]
    if not spec_keys:
        return
    spec = shred_spec_sql({key: "VARCHAR" for key in spec_keys})
    plain_args = ", ".join(f"jsono({sql_literal(json_dumps(patch))})" for patch in patches)
    shred_args = ", ".join(f"jsono({struct_literal_sql(patch)})" for patch in patches)
    reference = SESSION.value(f"to_json(jsono_merge_patch(jsono({sql_literal(base_text)}), {plain_args}))")
    candidate = SESSION.value(
        f"to_json(jsono_merge_patch(jsono({sql_literal(base_text)}, shredding := {spec}), {shred_args}))"
    )
    assert reference == candidate, (
        f"auto-shred nested patch merge diverged from plain: base {base_text!r} patches {patches!r} "
        f"spec {spec_keys!r}: plain {reference!r} != shredded {candidate!r}"
    )


# Array shredding: a regular array's element subfields lift into a parallel LIST<STRUCT> column, the
# element tail staying in the residual skeleton. Elements are heterogeneous (object/null/scalar/
# nested array); an element object draws its subfields from a small shared name set so a schema can
# over- or under-cover them (missing subfield, tail key) and pick any shred type (matching or not).
array_subfield_names = ["id", "name", "price", "qty", "sku"]
array_element = st.one_of(
    st.dictionaries(st.sampled_from(array_subfield_names), json_scalars, min_size=0, max_size=5),
    st.none(),
    json_scalars,
    st.lists(st.integers(min_value=0, max_value=5), max_size=2),
)
array_documents = st.fixed_dictionaries({"items": st.lists(array_element, max_size=6)})
array_shred_types = st.sampled_from(["VARCHAR", "BIGINT", "UBIGINT", "DOUBLE", "BOOLEAN"])
# A LIST<STRUCT> element schema: a non-empty subset of the subfield names, each with any shred type.
# The field order is the dict's, so it is frequently unsorted (the overlay patch must re-sort it).
array_element_schemas = st.dictionaries(
    st.sampled_from(array_subfield_names), array_shred_types, min_size=1, max_size=5
).map(lambda fields: "STRUCT(" + ", ".join(f"{name} {stype}" for name, stype in fields.items()) + ")[]")


# Every native way to read a shredded value must yield the same result as a plain parse of the same
# document. Each reader is one entry in ARRAY_READER_PARITY: a function taking the shredded and plain
# `jsono(...)` SQL expressions and returning a full SELECT that evaluates to 'true' iff the reader
# agrees on both. Adding a shred-aware reader (or a new way to read a shredded value) is one line.
def _to_json_parity(shredded: str, plain: str) -> str:
    # Whole-value reconstruct: the residual skeleton overlaid with the shred columns.
    return f"SELECT (to_json({shredded}) IS NOT DISTINCT FROM to_json({plain}))"


def _descend_reads_parity(shredded: str, plain: str) -> str:
    # Reads that descend INTO the shredded array ($.items[i].field, and the whole $.items): the lifted
    # subfields are stripped from the skeleton residual, so the optimizer must reconstruct rather than
    # read a silent NULL/wrong text. Covers the edge doubles / control chars that once crossed a
    # core-JSON render boundary.
    reads = [f"$.items[{i}].{f}" for i in (0, 1, 2, 9) for f in array_subfield_names]
    reads.append("$.items")
    cond = " AND ".join(f"(sj ->> '{r}') IS NOT DISTINCT FROM (pj ->> '{r}')" for r in reads)
    return f"SELECT ({cond}) FROM (SELECT {shredded} sj, {plain} pj)"


def _entries_parity(shredded: str, plain: str) -> str:
    # jsono_entries flattens to leaf (key, value) structs; order is not canonical, so compare the two
    # flattenings as a multiset via an empty symmetric difference.
    return (
        f"SELECT (SELECT count(*) FROM (SELECT unnest(jsono_entries({shredded})) "
        f"EXCEPT ALL SELECT unnest(jsono_entries({plain})))) = 0 "
        f"AND (SELECT count(*) FROM (SELECT unnest(jsono_entries({plain})) "
        f"EXCEPT ALL SELECT unnest(jsono_entries({shredded})))) = 0"
    )


ARRAY_READER_PARITY = [
    ("to_json", _to_json_parity),
    ("descend_reads", _descend_reads_parity),
    ("entries", _entries_parity),
]


@settings(PROPERTY_SETTINGS)
@example(doc={"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}, schema="STRUCT(id BIGINT, name VARCHAR)[]")
@example(doc={"items": []}, schema="STRUCT(id BIGINT)[]")
@example(doc={"items": [None, {"id": 1}, 5, "x"]}, schema="STRUCT(id BIGINT, name VARCHAR)[]")
@example(doc={"items": [{"id": 1, "name": None, "sku": "s"}]}, schema="STRUCT(id BIGINT, name VARCHAR)[]")
@example(doc={"items": [{"qty": 2, "sku": "s1"}]}, schema="STRUCT(sku VARCHAR, qty BIGINT)[]")
@example(doc={"items": [{"id": 1, "name": "a", "sku": "tail"}, None, 5]}, schema="STRUCT(id BIGINT, name VARCHAR)[]")
@example(doc={"items": [{"id": None, "name": "b"}, {"id": "x"}]}, schema="STRUCT(id BIGINT, name VARCHAR)[]")
@example(doc={"items": [{"id": False}]}, schema="STRUCT(id VARCHAR)[]")
@example(doc={"items": [{"price": 1e16}]}, schema="STRUCT(price DOUBLE)[]")
@given(doc=array_documents, schema=array_element_schemas)
def test_array_reader_parity(doc: dict[str, Any], schema: str) -> None:
    # A shredded array value must read identically to a plain parse across every reader in
    # ARRAY_READER_PARITY (to_json reconstruct, `->>` descend, jsono_entries leaf multiset). Fuzzes
    # the full scalar space over heterogeneous elements, missing / type-mismatched / explicit-null
    # subfields, tail keys, empty arrays, VARCHAR read-copies of non-string scalars, and edge doubles.
    text = json_dumps(doc)
    shredded = f"jsono({sql_literal(text)}, shredding := {{'$.items': '{schema}'}})"
    plain = f"jsono({sql_literal(text)})"
    for name, build_query in ARRAY_READER_PARITY:
        ok = SESSION.value(build_query(shredded, plain))
        assert ok == "true", f"array {name} differs shredded vs plain: {text!r} schema {schema!r}"


# Scalar-array shredding: a regular array's whole scalar elements lift into a parallel LIST<scalar>
# column, a VAL_NULL placeholder per lifted element staying in the residual skeleton. Elements are
# heterogeneous (scalar / null / object / nested array) so the per-element lossless gate is exercised
# the same way the object-array test exercises subfield gates.
scalar_array_element = st.one_of(
    json_scalars,
    st.dictionaries(st.sampled_from(array_subfield_names), json_scalars, min_size=0, max_size=3),
    st.lists(st.integers(min_value=0, max_value=5), max_size=2),
)
scalar_array_documents = st.fixed_dictionaries({"items": st.lists(scalar_array_element, max_size=6)})
scalar_array_shred_types = st.sampled_from(["VARCHAR[]", "BIGINT[]", "UBIGINT[]", "DOUBLE[]", "BOOLEAN[]"])


def _scalar_descend_reads_parity(shredded: str, plain: str) -> str:
    # Reads that descend INTO the shredded scalar array ($.items[i], and the whole $.items): the lifted
    # elements are stripped to placeholders in the skeleton residual, so the optimizer must reconstruct
    # rather than read the placeholder NULL.
    reads = [f"$.items[{i}]" for i in (0, 1, 2, 9)]
    reads.append("$.items")
    cond = " AND ".join(f"(sj ->> '{r}') IS NOT DISTINCT FROM (pj ->> '{r}')" for r in reads)
    return f"SELECT ({cond}) FROM (SELECT {shredded} sj, {plain} pj)"


SCALAR_ARRAY_READER_PARITY = [
    ("to_json", _to_json_parity),
    ("descend_reads", _scalar_descend_reads_parity),
    ("entries", _entries_parity),
]


@settings(PROPERTY_SETTINGS)
@example(doc={"items": [1, 2, 3]}, schema="UBIGINT[]")
@example(doc={"items": []}, schema="UBIGINT[]")
@example(doc={"items": [1, "two", 3, None, True, -5]}, schema="UBIGINT[]")
@example(doc={"items": ["x", 5, "z"]}, schema="VARCHAR[]")
@example(doc={"items": [1.5, 2.0, 3]}, schema="DOUBLE[]")
@example(doc={"items": [True, False, 1]}, schema="BOOLEAN[]")
@example(doc={"items": [1, None, 2]}, schema="UBIGINT[]")
@example(doc={"items": [{"id": 1}, 5, "x", None]}, schema="BIGINT[]")
@example(doc={"items": [1e16]}, schema="DOUBLE[]")
@given(doc=scalar_array_documents, schema=scalar_array_shred_types)
def test_scalar_array_reader_parity(doc: dict[str, Any], schema: str) -> None:
    # A shredded scalar-array value must read identically to a plain parse across every reader in
    # SCALAR_ARRAY_READER_PARITY (to_json reconstruct, `->>` descend, jsono_entries leaf multiset).
    # Fuzzes the full scalar space over heterogeneous elements (the lossless gate keeps non-conforming
    # scalars / null / object / nested-array elements in the residual skeleton).
    text = json_dumps(doc)
    shredded = f"jsono({sql_literal(text)}, shredding := {{'$.items': '{schema}'}})"
    plain = f"jsono({sql_literal(text)})"
    for name, build_query in SCALAR_ARRAY_READER_PARITY:
        ok = SESSION.value(build_query(shredded, plain))
        assert ok == "true", f"scalar array {name} differs shredded vs plain: {text!r} schema {schema!r}"


# Typed scalars for the STRUCT-input auto-shred test: each carries an explicit DuckDB type so the
# constructor's bind sees the same eligibility it would from a read_json_auto-detected schema.
typed_struct_scalars = st.one_of(
    st.builds(lambda value: (value, "VARCHAR"), st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", max_size=8)),
    st.builds(lambda value: (value, "BIGINT"), st.integers(min_value=-(2**62), max_value=2**62)),
    st.builds(lambda value: (value, "BOOLEAN"), st.booleans()),
)
typed_struct_documents = st.recursive(
    typed_struct_scalars,
    lambda children: st.dictionaries(shred_keys, children, min_size=1, max_size=4),
    max_leaves=8,
).filter(lambda value: isinstance(value, dict))


def typed_struct_sql(doc: dict[str, Any]) -> str:
    fields = []
    for key, value in doc.items():
        if isinstance(value, dict):
            fields.append(f"'{key}': {typed_struct_sql(value)}")
        else:
            literal, stype = value
            if stype == "VARCHAR":
                fields.append(f"'{key}': {sql_literal(literal)}")
            elif stype == "BOOLEAN":
                fields.append(f"'{key}': {'true' if literal else 'false'}")
            else:
                fields.append(f"'{key}': {literal}::{stype}")
    return "{" + ", ".join(fields) + "}"


def typed_struct_json(doc: dict[str, Any]) -> Any:
    out: dict[str, Any] = {}
    for key, value in doc.items():
        out[key] = typed_struct_json(value) if isinstance(value, dict) else value[0]
    return out


@settings(PROPERTY_SETTINGS)
@example(doc={"did": ("d1", "VARCHAR"), "commit": {"collection": ("coll", "VARCHAR"), "rev": (1, "BIGINT")}})
@example(doc={"a": ("x", "VARCHAR"), "b": {"c": (1, "BIGINT"), "d": ("y", "VARCHAR")}})
@given(doc=typed_struct_documents)
def test_struct_auto_shred_lossless(doc: dict[str, Any]) -> None:
    # The type-driven auto-shred (jsono over a typed STRUCT) lifts every scalar field — top-level
    # ones by bare name, nested ones under a `$.parent.child` path — into a typed shred. Whatever
    # the shred set, the logical value must equal the plain parse of the same document.
    if not isinstance(doc, dict) or not doc:
        return
    struct_sql = typed_struct_sql(doc)
    text = json_dumps(typed_struct_json(doc))
    auto = SESSION.value(f"to_json(jsono({struct_sql}))")
    plain = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    assert auto == plain, f"auto-shred changed the value: {struct_sql} : {auto!r} != plain {plain!r}"


# Constructor (native STRUCT) auto-shred of a top-level LIST<STRUCT> field: the lifted array must
# reconstruct to the same JSON as parsing the equivalent document. The element subfield types are
# fixed (a DuckDB list is homogeneous); values and null elements are fuzzed.
array_struct_leaf = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=-(2**40), max_value=2**40),
        "name": st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", max_size=6),
        "flag": st.booleans(),
    }
)
array_struct_elements = st.lists(st.one_of(array_struct_leaf, st.none()), max_size=5)


def array_struct_sql(elements: list[Any]) -> str:
    parts = []
    for el in elements:
        if el is None:
            parts.append("NULL")
        else:
            flag = "true" if el["flag"] else "false"
            parts.append(f"{{'id': {el['id']}::BIGINT, 'name': {sql_literal(el['name'])}, 'flag': {flag}}}")
    return "[" + ", ".join(parts) + "]::STRUCT(id BIGINT, \"name\" VARCHAR, flag BOOLEAN)[]"


@settings(PROPERTY_SETTINGS)
@example(elements=[{"id": 1, "name": "a", "flag": True}, None])
@example(elements=[])
@given(elements=array_struct_elements)
def test_struct_array_auto_shred_lossless(elements: list[Any]) -> None:
    list_sql = array_struct_sql(elements)
    from_struct = SESSION.value(f"to_json(jsono({{'items': {list_sql}}}))")
    json_doc = {
        "items": [None if el is None else {"id": el["id"], "name": el["name"], "flag": el["flag"]} for el in elements]
    }
    plain = SESSION.value(f"to_json(jsono({sql_literal(json_dumps(json_doc))}))")
    assert from_struct == plain, f"struct array auto-shred changed the value: {list_sql} : {from_struct!r} != {plain!r}"


# Constructor (native STRUCT) auto-shred of a top-level LIST<scalar> field: the lifted array must
# reconstruct to the same JSON as parsing the equivalent document. A DuckDB list is homogeneous and
# typed, so each draw fixes one element type and fuzzes values plus NULL elements.
# A binary DOUBLE from the struct constructor renders through jsono's own formatter
# (10000000000000000.0), which legitimately differs from the JSON text a plain parse preserves
# (1e+16) — orthogonal to shredding (a non-shredded constructor double renders the same way). The
# auto-shred lossless property compares the lifted value against a text parse, so it fuzzes the types
# whose text and value coincide; the DOUBLE lane's reconstruct is covered by the deterministic SQL
# test (clean doubles) and the object-array DOUBLE-subfield property.
scalar_list_specs = [
    ("UBIGINT", st.integers(min_value=0, max_value=2**63)),
    ("BIGINT", st.integers(min_value=-(2**62), max_value=2**62)),
    ("BOOLEAN", st.booleans()),
    ("VARCHAR", st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", max_size=6)),
]


@st.composite
def scalar_list_draw(draw: Any) -> tuple[str, list[Any]]:
    stype, element_strategy = draw(st.sampled_from(scalar_list_specs))
    values = draw(st.lists(st.one_of(element_strategy, st.none()), max_size=5))
    return stype, values


def scalar_list_element_sql(value: Any, stype: str) -> str:
    if value is None:
        return "NULL"
    if stype == "BOOLEAN":
        return "true" if value else "false"
    if stype == "VARCHAR":
        return sql_literal(value)
    return f"{value}::{stype}"


@settings(PROPERTY_SETTINGS)
@example(spec=("UBIGINT", [1, 2, None]))
@example(spec=("UBIGINT", []))
@example(spec=("VARCHAR", ["a", None, "b"]))
@example(spec=("BIGINT", [-5, 10, None]))
@given(spec=scalar_list_draw())
def test_scalar_array_auto_shred_lossless(spec: tuple[str, list[Any]]) -> None:
    # The type-driven auto-shred (jsono over a typed STRUCT with a LIST<scalar> field) lifts the whole
    # array into a typed LIST<scalar> shred; whatever the values and NULL elements, the logical value
    # must equal the plain parse of the same document.
    stype, values = spec
    list_sql = "[" + ", ".join(scalar_list_element_sql(v, stype) for v in values) + f"]::{stype}[]"
    from_struct = SESSION.value(f"to_json(jsono({{'item_ids': {list_sql}}}))")
    plain = SESSION.value(f"to_json(jsono({sql_literal(json_dumps({'item_ids': values}))}))")
    assert from_struct == plain, f"scalar array auto-shred changed the value: {list_sql} : {from_struct!r} != {plain!r}"


@settings(VALIDISH_PROPERTY_SETTINGS)
@given(doc=shred_documents, data=st.data())
def test_narrowing_fails_loud(doc: dict[str, Any], data: Any) -> None:
    # A raw narrowing struct cast (extension optimizer disabled) drops a shred; every read of
    # such a row must fail loud — silently returning partial data is the defect class the
    # shred manifest exists to catch. A shred whose value stayed in the residual (the lossless
    # gate rejected it) is not in the manifest, so narrowing it away is harmless; the property
    # therefore shreds string values as VARCHAR (always stripped).
    str_keys = sorted(key for key, value in doc.items() if isinstance(value, str) and value)
    if len(str_keys) < 2:
        return
    dropped = data.draw(st.sampled_from(str_keys))
    kept = [key for key in str_keys if key != dropped]
    text = json_dumps(doc)
    wide_spec = shred_spec_sql({path: "VARCHAR" for path in str_keys})
    narrow_spec = shred_spec_sql({path: "VARCHAR" for path in kept})
    SESSION.statement("DROP TABLE IF EXISTS narrow_prop;")
    SESSION.statement(f"CREATE TABLE narrow_prop AS SELECT jsono('{{}}', shredding := {narrow_spec}) AS e WHERE false;")
    SESSION.statement("SET disabled_optimizers='extension';")
    SESSION.statement(f"INSERT INTO narrow_prop SELECT jsono({sql_literal(text)}, shredding := {wide_spec});")
    SESSION.statement("RESET disabled_optimizers;")
    read = SESSION.value("(SELECT to_json(e)::VARCHAR FROM narrow_prop LIMIT 1)")
    # The narrowed row must error (the session returns None for an errored query), never a
    # partial document.
    assert read is None, f"narrowed row read silently: {text!r} dropped {dropped!r} -> {read!r}"


manifest_mutations = st.sampled_from(
    [
        "count_inflate",
        "count_huge",
        "path_len_huge",
        "truncate_mid",
        "truncate_tail",
        "append_junk",
        "flip_last",
    ]
)
# (text, shred spec) pairs whose values pass the lossless gate, so the residual carries a
# non-empty manifest for the mutations to target.
manifest_documents = st.sampled_from(
    [
        ('{"s":"hello","t":"x"}', "{'s': 'VARCHAR'}"),
        ('{"a":"1","b":"2","c":3}', "{'a': 'VARCHAR', 'b': 'VARCHAR'}"),
        ('{"n":7,"o":{"x":1}}', "{'n': 'BIGINT'}"),
        (wide_object_text, "{'k00': 'BIGINT', 'k16': 'VARCHAR'}"),
    ]
)


@settings(VALIDISH_PROPERTY_SETTINGS)
@given(doc=manifest_documents, mutation=manifest_mutations)
def test_fuzz_manifest_no_crash(doc: tuple[str, str], mutation: str) -> None:
    # A corrupted shred-manifest tail (reachable from Parquet like any other blob bytes) must
    # yield a value, a NULL, or a clean SQL error from the manifest framing checks — never a
    # crash. The mutated residual is read as a plain value: jsono_validate parses the manifest
    # framing directly, and an extract miss consults the manifest for the narrowing check.
    text, spec_sql = doc
    encoded = SESSION.value(shredded_blob_hex_expr(text, spec_sql))
    assert encoded is not None
    parts = encoded.split("|")
    assert len(parts) == 4
    struct = jsono_struct_sql((parts[0], parts[1], parts[2], mutate_manifest(parts[3], mutation)))
    SESSION.value(f"jsono_validate({struct})")
    SESSION.value(f"jsono_extract_string({struct}, '$.no_such_key')")
    SESSION.value(f"to_json({struct})")


# Rows for the keyed group-merge invariant. Every path is kind-consistent (scalar leaves under
# fixed keys; an optional nested object under "n" is never a scalar), so the order-independent
# jsono_group_merge_max/_min must reproduce the sequential ORDER BY fold exactly.
lww_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=4),
)
lww_leaf_keys = st.sampled_from(["a", "b", "c"])
lww_rows = st.lists(
    st.builds(
        lambda flat, nested, has_nested: {**flat, **({"n": nested} if has_nested else {})},
        st.dictionaries(lww_leaf_keys, lww_scalars, max_size=3),
        st.dictionaries(lww_leaf_keys, lww_scalars, max_size=3),
        st.booleans(),
    ),
    min_size=1,
    max_size=8,
)


@settings(PROPERTY_SETTINGS)
@example(rows=[{"a": 1, "n": {"x": 1}}, {"a": None, "n": {"y": 2}}, {"b": 3}], data=None)
@example(rows=[{"a": 1}, {"a": 2}, {"a": 3}], data=None)
@given(rows=lww_rows, data=st.data())
def test_group_merge_keyed_parity(rows: list[dict[str, Any]], data: Any) -> None:
    # Distinct per-row keys (a permutation) so the ORDER BY fold is itself deterministic and the
    # keyed result has a single well-defined target to match.
    order_keys = list(range(len(rows))) if data is None else data.draw(st.permutations(list(range(len(rows)))))
    values = ", ".join(f"(jsono({sql_literal(json_dumps(row))}), {key})" for row, key in zip(rows, order_keys))
    cte = f"WITH t(j, ok) AS (VALUES {values})"
    keyed_max = SESSION.value(f"({cte} SELECT to_json(jsono_group_merge_max(j, ok)) FROM t)")
    ordered_asc = SESSION.value(f"({cte} SELECT to_json(jsono_group_merge(j ORDER BY ok)) FROM t)")
    assert keyed_max == ordered_asc, f"max != ORDER BY: {rows!r} -> {keyed_max!r} vs {ordered_asc!r}"

    keyed_min = SESSION.value(f"({cte} SELECT to_json(jsono_group_merge_min(j, ok)) FROM t)")
    ordered_desc = SESSION.value(f"({cte} SELECT to_json(jsono_group_merge(j ORDER BY ok DESC)) FROM t)")
    assert keyed_min == ordered_desc, f"min != ORDER BY DESC: {rows!r} -> {keyed_min!r} vs {ordered_desc!r}"

    # Order independence: the physical row order fed to the aggregate must not change the result.
    keyed_reordered = SESSION.value(
        f"({cte} SELECT to_json(jsono_group_merge_max(j, ok)) FROM (SELECT * FROM t ORDER BY ok DESC))"
    )
    assert keyed_max == keyed_reordered, f"order-dependent: {rows!r} -> {keyed_max!r} vs {keyed_reordered!r}"


PROPERTIES = [
    test_round_trip_idempotent,
    test_value_parity,
    test_keys_sorted,
    test_shred_lossless,
    test_reshred_lossless,
    test_merge_patch_shredded_plain_parity,
    test_merge_patch_auto_shred_patch_parity,
    test_array_reader_parity,
    test_struct_auto_shred_lossless,
    test_struct_array_auto_shred_lossless,
    test_narrowing_fails_loud,
    test_group_merge_keyed_parity,
    test_fuzz_text_no_crash,
    test_fuzz_blob_no_crash,
    test_fuzz_validish_blob_no_crash,
    test_fuzz_manifest_no_crash,
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
