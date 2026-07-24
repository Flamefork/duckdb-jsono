from collections import Counter
from decimal import Decimal
from json import dumps as json_dumps
from json import loads as json_loads
from os import environ
from pathlib import Path
from select import select
from subprocess import PIPE
from subprocess import Popen
from sys import stderr
from tempfile import mkdtemp
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
BlobHex = tuple[str, str, str, str, str, str]

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

    # A SQL error raised by the evaluated expression. `value()` collapses this to None (its
    # contract is "the value, or None for NULL/error"), but the 3-state `read()` returns it so a
    # fails-loud property can assert a LOUD ERROR specifically — not merely "not a value". `message`
    # is the CLI's stderr text for this one statement (the error goes to stderr, never the result
    # stream), so a property can pin the expected error to its real substring.
    class Errored:
        def __init__(self, message: str) -> None:
            self.message = message

    # Evaluate one scalar expression and return one of three states: an Errored (the SQL raised),
    # None (the result was SQL NULL or empty), or the value's text. The expression is cast to VARCHAR
    # so a JSON-typed result comes back as a single JSON string (the renderer would otherwise inline a
    # JSON column as a nested object, which is not round-trippable through json_loads). A JsonoCrash
    # here is a genuine process death, distinct from a SQL error (which leaves the process alive and
    # surfaces as Errored). An errored statement emits no result row, so its payload is empty; a NULL
    # result emits a row whose cell is null — the two are told apart by the payload, not stderr.
    def read(self, expr: str) -> "JsonoSession.Errored | str | None":
        if self.proc is None or self.proc.poll() is not None:
            self.restart()
        assert self.stderr_file is not None
        self.stderr_file.seek(0, 2)
        stderr_before = self.stderr_file.tell()
        try:
            stmt = f"SELECT CAST(({expr}) AS VARCHAR) AS v;"
            self._send(stmt)
            rows = self._drain(stmt)
        except JsonoCrash:
            self.restart()
            raise
        payload = "".join(rows).strip()
        parsed = json_loads(payload) if payload and payload != "[]" else None
        if not parsed:
            self.stderr_file.seek(stderr_before)
            return JsonoSession.Errored(self.stderr_file.read().decode("utf-8", "replace"))
        (cell,) = parsed[0].values()
        return None if cell is None else str(cell)

    # Evaluate one scalar expression and return its text, or None for SQL NULL / error / empty
    # result. Property queries that are written to always return data use this; a property that must
    # distinguish a raised error from a NULL uses read() instead. A JsonoCrash here means a genuine
    # process death, not a SQL error.
    def value(self, expr: str) -> str | None:
        result = self.read(expr)
        return None if isinstance(result, JsonoSession.Errored) else result

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
    # A SQL expression evaluating to exactly `text`. The persistent CLI reads stdin as
    # newline-delimited C-strings (local_getline), so a raw NUL byte on the wire terminates the
    # line early and the statement is never seen as complete — the pipe round-trip then deadlocks
    # (a bare newline is harmless: the shell reassembles the statement across lines). A NUL reaches
    # here only from raw st.text() values that skip json_dumps' u0000 escaping (struct literals);
    # splice it back in with chr(0) so the wire carries no raw NUL while the value is preserved.
    if "\x00" not in text:
        return "'" + text.replace("'", "''") + "'"
    runs = ["'" + run.replace("'", "''") + "'" for run in text.split("\x00")]
    return "(" + " || chr(0) || ".join(runs) + ")"


BLOB_FIELDS = ["slots", "key_heap", "string_heap", "skips", "lengths", "nums"]


def blob_hex_expr(body_expr: str) -> str:
    return " || '|' || ".join(f"hex(({body_expr}).{field})" for field in BLOB_FIELDS)


def jsono_blob_hex_expr(text: str) -> str:
    body_expr = f'jsono({sql_literal(text)})."jsono".body'
    return blob_hex_expr(body_expr)


def jsono_struct_sql(blobs: BlobHex) -> str:
    slots, key_heap, string_heap, skips, lengths, nums = blobs
    body_fields = [
        f"'slots': unhex('{slots}')",
        f"'key_heap': unhex('{key_heap}')",
        f"'string_heap': unhex('{string_heap}')",
        f"'skips': unhex('{skips}')",
        f"'lengths': unhex('{lengths}')",
        f"'nums': unhex('{nums}')",
    ]
    body = "{" + ", ".join(body_fields) + "}"
    return "{'jsono': {'body': " + body + "}}"


def mutate_jsono_blobs(blobs: BlobHex, mutation: str) -> BlobHex:
    slots = bytearray(bytes.fromhex(blobs[0]))
    key_heap = bytearray(bytes.fromhex(blobs[1]))
    string_heap = bytearray(bytes.fromhex(blobs[2]))
    skips = bytearray(bytes.fromhex(blobs[3]))
    lengths = bytearray(bytes.fromhex(blobs[4]))
    nums = bytearray(bytes.fromhex(blobs[5]))

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
    elif mutation == "lengths_truncate":
        if lengths:
            lengths = lengths[:-1]
        else:
            fallback()
    elif mutation == "lengths_extra_entry":
        lengths.extend(b"\x00\x00\x00\x00")
    elif mutation == "lengths_misalign":
        lengths.append(0)
    elif mutation == "nums_truncate":
        if nums:
            nums = nums[:-1]
        else:
            fallback()
    elif mutation == "nums_extra_entry":
        nums.extend(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    elif mutation == "nums_misalign":
        nums.append(0)
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
    lengths_hex = lengths.hex().upper()
    nums_hex = nums.hex().upper()
    return (slots_hex, key_heap_hex, string_heap_hex, skips_hex, lengths_hex, nums_hex)


def shredded_blob_hex_expr(text: str, spec_sql: str) -> str:
    body_expr = f'jsono({sql_literal(text)}, shredding := {spec_sql})."jsono".body'
    return blob_hex_expr(body_expr)


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
# The Parquet round-trip property writes and reads a file per example, so it runs on a smaller budget
# than the pure in-memory properties (file I/O is the expensive part, not the SQL).
PARQUET_PROPERTY_SETTINGS = settings(
    max_examples=max(25, MAX_EXAMPLES // 6),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
# One temp dir for the Parquet round-trip property; a single file is overwritten each example (COPY TO
# replaces the target), so nothing accumulates across examples.
PARQUET_ROUNDTRIP_DIR = mkdtemp(prefix="jsono_parquet_prop_")

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
        "lengths_truncate",
        "lengths_extra_entry",
        "lengths_misalign",
        "nums_truncate",
        "nums_extra_entry",
        "nums_misalign",
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


# Every SQL entry point that decodes a JSONO value, exercised over hostile bytes. One reader is
# drawn per example (rotating instead of running all of them keeps the example budget constant);
# the dict covers the walkers the original fuzz missed: the merge family's contiguity-gated bulk
# memcpy (corrupt value on either side), the diff recursion, entries in both array styles,
# transform's shape-plan walk, the introspection walks, the collect/array-elements emitters, and the
# shredding advisor's plain-tape AccumulateDocument (jsono_suggest_shredding walks slots per row).
# Also covered: the `->` subtree extract (its OWN bulk ContainerSpan / checkpoint-records memcpy,
# distinct from the scalar extract_string reader), jsono_overlay (the shredded-reconstruction
# primitive), the keyed last-write-wins group_merge_max/_min state fold, the shred-manifest tail
# walk, and the storage_size blob/manifest reader. jsono_shred_stats folds only a SHREDDED input, so
# over these plain forged structs it exercises the bind-time rejection rather than the decoder (the
# harness forges plain layouts only; a shredded forge would be a harness change). jsono_storage_type
# is deliberately absent: it has no jsono-value overload (only `jsono_storage_type()` and the
# `(VARCHAR)` shreds-spec form), so it can never receive forged value bytes.
FUZZ_READER_CALLS = {
    "to_json": lambda s: f"to_json({s})",
    "validate": lambda s: f"jsono_validate({s})",
    "extract_string": lambda s: f"jsono_extract_string({s}, '$.a')",
    "extract": lambda s: f"to_json(jsono_extract({s}, '$.a'))",
    "type": lambda s: f"jsono_type({s})",
    "keys": lambda s: f"jsono_keys({s})",
    "array_length": lambda s: f"jsono_array_length({s}, '$.a')",
    "array_elements": lambda s: f"jsono_array_elements({s})",
    "entries": lambda s: f"jsono_entries({s})",
    "entries_whole": lambda s: f"jsono_entries({s}, array_style := 'whole_json')",
    "transform": lambda s: f"jsono_transform({s}, {{a: 'VARCHAR'}})",
    "merge_patch_target": lambda s: f"to_json(jsono_merge_patch({s}, jsono('{{\"k\":1}}')))",
    "merge_patch_patch": lambda s: f"to_json(jsono_merge_patch(jsono('{{\"k\":1}}'), {s}))",
    "overlay": lambda s: f"to_json(jsono_overlay({s}, jsono('{{\"k\":1}}')))",
    "diff_prev": lambda s: f"to_json(jsono_diff({s}, jsono('{{}}')))",
    "diff_cur": lambda s: f"to_json(jsono_diff(jsono('{{}}'), {s}))",
    "group_merge": lambda s: f"(SELECT to_json(jsono_group_merge(v)) FROM (VALUES ({s}), (jsono('{{\"k\":1}}'))) t(v))",
    "group_merge_max": lambda s: f"(SELECT to_json(jsono_group_merge_max(v, k)) FROM (VALUES ({s}, 1), (jsono('{{\"k\":1}}'), 2)) t(v, k))",
    "group_merge_min": lambda s: f"(SELECT to_json(jsono_group_merge_min(v, k)) FROM (VALUES ({s}, 1), (jsono('{{\"k\":1}}'), 2)) t(v, k))",
    "group_array": lambda s: f"(SELECT to_json(jsono_group_array(v)) FROM (VALUES ({s}), (jsono('{{}}'))) t(v))",
    "group_object": lambda s: f"(SELECT to_json(jsono_group_object(k, v)) FROM (VALUES ('a', {s}), ('b', jsono('{{}}'))) t(k, v))",
    "suggest_shredding": lambda s: f"(SELECT jsono_suggest_shredding(v) FROM (VALUES ({s}), (jsono('{{\"k\":1}}'))) t(v))",
    "shred_manifest": lambda s: f"jsono_shred_manifest({s})",
    "shred_stats": lambda s: f"(SELECT jsono_shred_stats(v) FROM (VALUES ({s}), (jsono('{{\"k\":1}}'))) t(v))",
    "storage_size": lambda s: f"jsono_storage_size({s})",
}
fuzz_readers = st.sampled_from(sorted(FUZZ_READER_CALLS))


@settings(PROPERTY_SETTINGS)
@given(
    slots=hex_blob,
    key=hex_blob,
    string=hex_blob,
    skips=hex_blob,
    lengths=hex_blob,
    nums=hex_blob,
    reader=fuzz_readers,
)
def test_fuzz_blob_no_crash(
    slots: str, key: str, string: str, skips: str, lengths: str, nums: str, reader: str
) -> None:
    # The six-BLOB physical shape is reachable from Parquet, so arbitrary bytes
    # must be rejected by every reader's bound checks, not dereferenced blindly.
    struct = jsono_struct_sql((slots, key, string, skips, lengths, nums))
    SESSION.value(f"to_json({struct})")
    SESSION.value(FUZZ_READER_CALLS[reader](struct))


@settings(VALIDISH_PROPERTY_SETTINGS)
@example(
    text='{"s":"abcdefghijklmnopqrstuvwxyz"}',
    mutation="lengths_truncate",
    reader="merge_patch_target",
)
@example(
    text='{"s":"abcdefghijklmnopqrstuvwxyz"}',
    mutation="lengths_extra_entry",
    reader="diff_prev",
)
@example(
    text='{"s":"abcdefghijklmnopqrstuvwxyz"}',
    mutation="lengths_misalign",
    reader="entries",
)
@example(text='{"u":18446744073709551615}', mutation="nums_truncate", reader="transform")
@example(text='{"u":18446744073709551615}', mutation="nums_extra_entry", reader="group_merge")
@example(text='{"u":18446744073709551615}', mutation="nums_misalign", reader="group_array")
@example(text=wide_object_text, mutation="checkpoint_stride", reader="entries_whole")
@example(text='{"n":1e3}', mutation="string_heap_bad_number", reader="array_elements")
@given(text=validish_json_documents, mutation=validish_mutations, reader=fuzz_readers)
def test_fuzz_validish_blob_no_crash(text: str, mutation: str, reader: str) -> None:
    encoded = SESSION.value(jsono_blob_hex_expr(text))
    assert encoded is not None
    parts = encoded.split("|")
    assert len(parts) == 6
    blobs = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
    struct = jsono_struct_sql(mutate_jsono_blobs(blobs, mutation))
    SESSION.value(f"jsono_validate({struct})")
    SESSION.value(f"to_json({struct})")
    SESSION.value(FUZZ_READER_CALLS[reader](struct))


# Shredding properties run on object documents whose top-level keys are plain identifiers
# (shred paths are spec field names; quoting rules are not the property under test).
shred_keys = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8).filter(lambda key: key != "body")
shred_documents = st.dictionaries(shred_keys, json_scalars, min_size=1, max_size=6)
shred_types = st.sampled_from(["VARCHAR", "BIGINT", "DOUBLE", "BOOLEAN"])


def shred_spec_sql(spec: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{path}': '{stype}'" for path, stype in spec.items()) + "}"


PLAIN_JSONO_TYPE_SQL = "STRUCT(jsono STRUCT(body STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB, lengths BLOB, nums BLOB)))"


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


@settings(PARQUET_PROPERTY_SETTINGS)
@given(doc=shred_documents, data=st.data())
def test_parquet_round_trip_to_json_parity(doc: dict[str, Any], data: Any) -> None:
    # A jsono column — both a plain value and one shredded spec — survives a real Parquet write/read:
    # the six body blobs (and the shred lanes) must reassemble to the byte-identical struct, so to_json
    # over the Parquet-backed value equals to_json over the in-memory original. This exercises the
    # physical vector shapes Parquet produces (per-column null masks, blob children) that the in-memory
    # constructor never builds — the rest of the harness only simulates the physical shape.
    text = json_dumps(doc)
    keys = sorted(doc.keys())
    paths = data.draw(st.lists(st.sampled_from(keys), min_size=1, max_size=4, unique=True))
    spec = shred_spec_sql({path: data.draw(shred_types) for path in paths})
    expected = SESSION.value(f"to_json(jsono({sql_literal(text)}))")
    path = f"{PARQUET_ROUNDTRIP_DIR}/round_trip.parquet"
    SESSION.statement(
        f"COPY (SELECT jsono({sql_literal(text)}) AS plain, "
        f"jsono({sql_literal(text)}, shredding := {spec}) AS shredded) "
        f"TO '{path}' (FORMAT parquet);"
    )
    plain_back = SESSION.value(f"(SELECT to_json(plain) FROM read_parquet('{path}'))")
    shredded_back = SESSION.value(f"(SELECT to_json(shredded) FROM read_parquet('{path}'))")
    assert (
        plain_back == expected
    ), f"plain Parquet round-trip changed the value: {text!r}: {expected!r} -> {plain_back!r}"
    assert (
        shredded_back == expected
    ), f"shredded Parquet round-trip changed the value: {text!r} spec {spec!r}: {expected!r} -> {shredded_back!r}"


# merge_patch over a SHREDDED base with PLAIN patch arguments must fold identically to the same
# PLAIN base with the same patches. This fuzzes every way a plain patch can interact with a lane:
# disjoint append, a value colliding into the
# residual, an RFC 7396 null that deletes a lane (no trace in the folded residual), and a non-object
# patch that replaces the whole document. The plain-base fold is the RFC 7396 reference.
merge_patch_keys = st.sampled_from(["a", "b", "c", "d", "e"])
merge_patch_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-100, max_value=100),
    st.text(max_size=4),
)
merge_patch_objects = st.dictionaries(
    merge_patch_keys,
    st.one_of(
        merge_patch_scalars,
        st.dictionaries(merge_patch_keys, merge_patch_scalars, max_size=2),
    ),
    min_size=0,
    max_size=4,
)
merge_patch_args = st.lists(
    st.one_of(
        merge_patch_objects,
        merge_patch_scalars,
        st.lists(merge_patch_scalars, max_size=3),
    ),
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


# EVERY input SHREDDED with the same spec must fold identically to the same all-PLAIN fold, for both
# merge_patch (last-wins) and overlay (base-authoritative). This drives the all-shredded fast path's
# PER-ROW lane winner: a shred key present in one input and absent (or explicit-null) in another,
# where a single per-INPUT winner would drop the surviving value or keep a deleted one. Patches are
# objects so their keys interact with the lanes (a None value is an RFC 7396 delete / overlay no-op).
all_shred_patches = st.lists(merge_patch_objects, min_size=1, max_size=3)


@settings(PROPERTY_SETTINGS)
@example(base={"a": 1}, patches=[{}], spec_keys=["a"])  # patch omits the key: base lane survives
@example(base={"a": 1}, patches=[{"a": None}], spec_keys=["a"])  # present-null deletes the base lane
@example(base={"a": 1, "b": 2}, patches=[{"b": 3}], spec_keys=["a", "b"])  # winner differs per shred
@example(base={"a": 1}, patches=[{"a": 2}, {}], spec_keys=["a"])  # a later patch omits: earlier wins
@given(
    base=merge_patch_base,
    patches=all_shred_patches,
    spec_keys=st.lists(merge_patch_keys, min_size=1, max_size=5, unique=True),
)
def test_merge_all_shredded_parity(base: dict[str, Any], patches: list[Any], spec_keys: list[str]) -> None:
    spec_keys = [key for key in spec_keys if key in base]
    if not spec_keys:
        return
    spec = shred_spec_sql({key: "VARCHAR" for key in spec_keys})
    base_lit = sql_literal(json_dumps(base))
    plain_args = ", ".join(f"jsono({sql_literal(json_dumps(patch))})" for patch in patches)
    shred_args = ", ".join(f"jsono({sql_literal(json_dumps(patch))}, shredding := {spec})" for patch in patches)
    for fn in ("jsono_merge_patch", "jsono_overlay"):
        reference = SESSION.value(f"to_json({fn}(jsono({base_lit}), {plain_args}))")
        candidate = SESSION.value(f"to_json({fn}(jsono({base_lit}, shredding := {spec}), {shred_args}))")
        assert reference == candidate, (
            f"all-shredded {fn} diverged from plain: base {base!r} patches {patches!r} "
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


def descend_reads_parity(shredded: str, plain: str, reads: list[str]) -> str:
    cond = " AND ".join(f"(sj ->> '{r}') IS NOT DISTINCT FROM (pj ->> '{r}')" for r in reads)
    return f"SELECT ({cond}) FROM (SELECT {shredded} sj, {plain} pj)"


def _descend_reads_parity(shredded: str, plain: str) -> str:
    # Reads that descend INTO the shredded array ($.items[i].field, and the whole $.items): the lifted
    # subfields are stripped from the skeleton residual, so the optimizer must reconstruct rather than
    # read a silent NULL/wrong text. Covers the edge doubles / control chars that once crossed a
    # core-JSON render boundary.
    reads = [f"$.items[{i}].{f}" for i in (0, 1, 2, 9) for f in array_subfield_names]
    reads.append("$.items")
    return descend_reads_parity(shredded, plain, reads)


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
@example(
    doc={"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]},
    schema="STRUCT(id BIGINT, name VARCHAR)[]",
)
@example(doc={"items": []}, schema="STRUCT(id BIGINT)[]")
@example(doc={"items": [None, {"id": 1}, 5, "x"]}, schema="STRUCT(id BIGINT, name VARCHAR)[]")
@example(
    doc={"items": [{"id": 1, "name": None, "sku": "s"}]},
    schema="STRUCT(id BIGINT, name VARCHAR)[]",
)
@example(doc={"items": [{"qty": 2, "sku": "s1"}]}, schema="STRUCT(sku VARCHAR, qty BIGINT)[]")
@example(
    doc={"items": [{"id": 1, "name": "a", "sku": "tail"}, None, 5]},
    schema="STRUCT(id BIGINT, name VARCHAR)[]",
)
@example(
    doc={"items": [{"id": None, "name": "b"}, {"id": "x"}]},
    schema="STRUCT(id BIGINT, name VARCHAR)[]",
)
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
    return descend_reads_parity(shredded, plain, reads)


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
    st.builds(
        lambda value: (value, "VARCHAR"),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", max_size=8),
    ),
    st.builds(
        lambda value: (value, "BIGINT"),
        st.integers(min_value=-(2**62), max_value=2**62),
    ),
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
@example(
    doc={
        "did": ("d1", "VARCHAR"),
        "commit": {"collection": ("coll", "VARCHAR"), "rev": (1, "BIGINT")},
    }
)
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


# One row per constructor source-type class. `lane_type` is the auto-shred lane after promotion;
# None means the value must stay in the residual. The matrix below crosses every row with scalar,
# scalar-list/array, nested, and object-list/array shapes, both alone and beside an independent nested shred.
CONSTRUCTOR_AUTO_SHRED_TYPE_CASES: list[tuple[str, str, str | None]] = [
    ("boolean", "true", "BOOLEAN"),
    ("tinyint", "-7::TINYINT", "BIGINT"),
    ("smallint", "-300::SMALLINT", "BIGINT"),
    ("integer", "-70000::INTEGER", "BIGINT"),
    ("bigint", "-5000000000::BIGINT", "BIGINT"),
    ("utinyint", "200::UTINYINT", "BIGINT"),
    ("usmallint", "60000::USMALLINT", "BIGINT"),
    ("uinteger", "4000000000::UINTEGER", "BIGINT"),
    ("ubigint", "18446744073709551615::UBIGINT", "UBIGINT"),
    ("float", "1.25::FLOAT", "DOUBLE"),
    ("double", "2.5::DOUBLE", "DOUBLE"),
    ("varchar", "'text'::VARCHAR", "VARCHAR"),
    ("date", "DATE '2024-01-02'", "VARCHAR"),
    ("time", "TIME '12:34:56'", "VARCHAR"),
    ("time_ns", "TIME_NS '12:34:56.123456789'", "VARCHAR"),
    ("timetz", "TIMETZ '12:34:56+02'", "VARCHAR"),
    ("timestamp_s", "TIMESTAMP_S '2024-01-02 12:34:56'", "VARCHAR"),
    ("timestamp_ms", "TIMESTAMP_MS '2024-01-02 12:34:56.789'", "VARCHAR"),
    ("timestamp", "TIMESTAMP '2024-01-02 12:34:56.123456'", "VARCHAR"),
    ("timestamp_ns", "TIMESTAMP_NS '2024-01-02 12:34:56.123456789'", "VARCHAR"),
    ("timestamptz", "TIMESTAMPTZ '2024-01-02 12:34:56+02'", "VARCHAR"),
    ("uuid", "UUID '4ac7a9e9-607c-4c8a-84f3-843f0191e3fd'", "VARCHAR"),
    ("enum", "CAST('alpha' AS ENUM('alpha','beta'))", "VARCHAR"),
    ("interval", "INTERVAL '3 days 2 hours'", "VARCHAR"),
    ("blob", "'\\xAA\\xBB'::BLOB", "VARCHAR"),
    ("bit", "'101'::BIT", "VARCHAR"),
    ("hugeint", "170141183460469231731687303715884105727::HUGEINT", None),
    ("uhugeint", "340282366920938463463374607431768211455::UHUGEINT", None),
    ("bignum", "1234567890123456789012345678901234567890::BIGNUM", None),
    ("decimal", "123456789012345678901234567890123456.12::DECIMAL(38,2)", None),
    ("json", "'{\"x\":1}'::JSON", None),
]


def constructor_auto_shred_shapes(value_sql: str, lane_type: str | None) -> list[tuple[str, str, dict[str, str], str]]:
    lane = {} if lane_type is None else {"value": lane_type}
    list_lane = {} if lane_type is None else {"values": f"{lane_type}[]"}
    nested_lane = {} if lane_type is None else {"$.parent.value": lane_type}
    nested_list_lane = {} if lane_type is None else {"$.parent.values": f"{lane_type}[]"}
    object_list_lane = {} if lane_type is None else {"items": f"STRUCT(value {lane_type})[]"}
    sibling_lane = {"$.sibling.kind": "VARCHAR"}
    value_list = f"[{value_sql}, NULL]"
    value_array = f"array_value({value_sql}, NULL)"
    object_list = f"[struct_pack(value := {value_sql}), NULL]"
    object_array = f"array_value(struct_pack(value := {value_sql}), NULL)"
    return [
        ("scalar", f"{{'value': {value_sql}}}", lane, "value"),
        (
            "scalar+nested",
            f"{{'value': {value_sql}, 'sibling': {{'kind': 'x'}}}}",
            lane | sibling_lane,
            "value",
        ),
        ("list", f"{{'values': {value_list}}}", list_lane, "values"),
        (
            "list+nested",
            f"{{'values': {value_list}, 'sibling': {{'kind': 'x'}}}}",
            list_lane | sibling_lane,
            "values",
        ),
        ("array", f"{{'values': {value_array}}}", list_lane, "values"),
        (
            "array+nested",
            f"{{'values': {value_array}, 'sibling': {{'kind': 'x'}}}}",
            list_lane | sibling_lane,
            "values",
        ),
        ("nested-scalar", f"{{'parent': {{'value': {value_sql}}}}}", nested_lane, "$.parent.value"),
        (
            "nested-list",
            f"{{'parent': {{'values': {value_list}}}}}",
            nested_list_lane,
            "$.parent.values",
        ),
        ("object-list", f"{{'items': {object_list}}}", object_list_lane, "$.items[0].value"),
        (
            "object-list+nested",
            f"{{'items': {object_list}, 'sibling': {{'kind': 'x'}}}}",
            object_list_lane | sibling_lane,
            "$.items[0].value",
        ),
        ("object-array", f"{{'items': {object_array}}}", object_list_lane, "$.items[0].value"),
        (
            "object-array+nested",
            f"{{'items': {object_array}, 'sibling': {{'kind': 'x'}}}}",
            object_list_lane | sibling_lane,
            "$.items[0].value",
        ),
    ]


def test_struct_constructor_auto_shred_matrix() -> None:
    for type_name, value_sql, lane_type in CONSTRUCTOR_AUTO_SHRED_TYPE_CASES:
        for shape_name, struct_sql, spec, read_path in constructor_auto_shred_shapes(value_sql, lane_type):
            auto = f"jsono({struct_sql})"
            plain = f"CAST(({struct_sql}) AS {PLAIN_JSONO_TYPE_SQL})"
            generic = plain if not spec else f"jsono({plain}, shredding := {shred_spec_sql(spec)})"
            parity = SESSION.value(
                f"(typeof({auto}) = typeof({generic})) AND "
                + f"(({auto}) IS NOT DISTINCT FROM ({generic})) AND "
                + f"(to_json({auto}) IS NOT DISTINCT FROM to_json({generic})) AND "
                + f"((({auto}) ->> '{read_path}') IS NOT DISTINCT FROM (({generic}) ->> '{read_path}'))"
            )
            assert parity == "true", f"{type_name}/{shape_name}: auto-shred differs from generic writer"


def test_struct_constructor_auto_shred_parquet() -> None:
    struct_sql = "{'values': [1::UBIGINT, NULL, 18446744073709551615::UBIGINT], " + "'sibling': {'kind': 'x'}}"
    auto = f"jsono({struct_sql})"
    path = f"{PARQUET_ROUNDTRIP_DIR}/struct_constructor.parquet"
    SESSION.statement(f"COPY (SELECT {auto} AS j) TO '{path}' (FORMAT parquet);")
    parity = SESSION.value(
        f"(SELECT (j IS NOT DISTINCT FROM {auto}) AND "
        + f"(to_json(j) IS NOT DISTINCT FROM to_json({auto})) AND "
        + f"((j ->> 'values') IS NOT DISTINCT FROM (({auto}) ->> 'values')) "
        + f"FROM read_parquet('{path}'))"
    )
    assert parity == "true", "UBIGINT list + nested shred changed across Parquet round-trip"


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
    return "[" + ", ".join(parts) + ']::STRUCT(id BIGINT, "name" VARCHAR, flag BOOLEAN)[]'


@settings(PROPERTY_SETTINGS)
@example(elements=[{"id": 1, "name": "a", "flag": True}, None])
@example(elements=[])
@given(elements=array_struct_elements)
def test_struct_array_auto_shred_lossless(elements: list[Any]) -> None:
    list_sql = array_struct_sql(elements)
    items = [(None if el is None else {"id": el["id"], "name": el["name"], "flag": el["flag"]}) for el in elements]
    for nested_sql, nested_json in (("", {}), (", 'nested': {'kind': 'x'}", {"nested": {"kind": "x"}})):
        from_struct = SESSION.value(f"to_json(jsono({{'items': {list_sql}{nested_sql}}}))")
        plain = SESSION.value(f"to_json(jsono({sql_literal(json_dumps({'items': items, **nested_json}))}))")
        assert from_struct == plain, (
            f"struct array auto-shred changed the value: {list_sql}{nested_sql} : " + f"{from_struct!r} != {plain!r}"
        )


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
    for nested_sql, nested_json in (("", {}), (", 'nested': {'kind': 'x'}", {"nested": {"kind": "x"}})):
        from_struct = SESSION.value(f"to_json(jsono({{'item_ids': {list_sql}{nested_sql}}}))")
        plain = SESSION.value(f"to_json(jsono({sql_literal(json_dumps({'item_ids': values, **nested_json}))}))")
        assert from_struct == plain, (
            f"scalar array auto-shred changed the value: {list_sql}{nested_sql} : " + f"{from_struct!r} != {plain!r}"
        )


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
    result = SESSION.read("(SELECT to_json(e)::VARCHAR FROM narrow_prop LIMIT 1)")
    # The narrowed row must fail LOUD — a raised SQL error, not a partial document and not a silent
    # NULL (which read() reports distinctly from an error). Pin the error to the manifest-guard's
    # real message so a regression that swaps the throw for a NULL/empty result is caught.
    assert isinstance(
        result, JsonoSession.Errored
    ), f"narrowed row did not fail loud: {text!r} dropped {dropped!r} -> {result!r}"
    assert (
        "was narrowed by a raw struct cast and cannot be read losslessly" in result.message
    ), f"narrowed row errored with an unexpected message: {text!r} dropped {dropped!r} -> {result.message!r}"


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
    assert len(parts) == 6
    struct = jsono_struct_sql(
        (
            parts[0],
            parts[1],
            parts[2],
            mutate_manifest(parts[3], mutation),
            parts[4],
            parts[5],
        )
    )
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
        lambda flat, nested, has_nested: {
            **flat,
            **({"n": nested} if has_nested else {}),
        },
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


# jsono_transform parity: the shared JsonoTrieWalk path (normal edges for named fields and wildcard
# edges for the list field) must agree, value-by-value, with two independent oracles: (A) running
# the same spec over the SAME document once shredded and once plain (the shred-lane read and the
# residual walk are different code paths) and (B) the equivalent single-path `->>` extraction per
# VARCHAR field (the operator path, not the multi-field walk). The generated cross-product spans
# the branches the walk dispatches on: named AND wildcard fields in one spec; objects both <=16 keys
# (linear walk) and >16 keys (checkpoint walk); present AND absent keys (JsonoTrieNullEdge /
# JsonoTrieAppendWildcardMissing); nested-path AND top-level fields; typed coercion AND lenient
# string extraction.
transform_leaf_keys = ["a", "b", "c", "d"]
transform_nest_keys = ["x", "y"]
transform_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**62), max_value=2**62),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(alphabet="abcdefghij0123456789", max_size=5),
)
# A wildcard item is heterogeneous: an object carrying the '$.items[*].v' leaf (present / absent /
# null / non-string scalar / container), a bare scalar, or null — so the wildcard list field walks
# present, missing, JSON-null, and type-mismatch element cases.
transform_item = st.one_of(
    st.dictionaries(st.sampled_from(["v", "w"]), transform_scalars, max_size=2),
    st.none(),
    transform_scalars,
    st.fixed_dictionaries({"v": st.lists(st.integers(min_value=0, max_value=3), max_size=2)}),
)


@st.composite
def transform_documents(draw: Any) -> tuple[dict[str, Any], int]:
    doc: dict[str, Any] = draw(st.dictionaries(st.sampled_from(transform_leaf_keys), transform_scalars, max_size=4))
    if draw(st.booleans()):
        doc["nest"] = draw(
            st.dictionaries(
                st.sampled_from(transform_nest_keys),
                transform_scalars,
                min_size=1,
                max_size=2,
            )
        )
    if draw(st.booleans()):
        doc["items"] = draw(st.lists(transform_item, max_size=5))
    # Padding keys straddle OBJECT_CHECKPOINT_STRIDE (16): with the up-to-6 semantic keys above, a
    # width of 0..18 puts the top-level object both at or below 16 keys (the linear merge walk) and
    # above 16 (JsonoTrieWalkObjectCheckpoints). The pad keys are never named by the spec, so they
    # only change the object width the walk traverses, not the expected field values.
    width = draw(st.integers(min_value=0, max_value=18))
    for i in range(width):
        doc[f"pad{i:02d}"] = i
    return doc, width


# Every VARCHAR scalar field in the spec, paired with its JSONPath, so the per-field `->>` oracle
# (leg B) can rebuild the expected value independently of the transform walk. Each is an
# explicit-path wrapper field so the field's value IS the value at that JSONPath: a bare scalar
# shorthand instead names a literal top-level key (e.g. 'a_str' would read the key "a_str", not
# $.a), which the dotted-key `.test` case covers separately and the `->>` oracle would not match.
TRANSFORM_VARCHAR_FIELDS = [
    ("a_str", "$.a"),
    ("missing_str", "$.zzz_absent"),
    ("nest_x", "$.nest.x"),
]


def transform_spec_sql() -> str:
    # One spec mixing: explicit-path VARCHAR fields, a present and an absent key, a nested-path
    # field, typed coercion targets (BIGINT/DOUBLE/BOOLEAN), and a wildcard list field over
    # '$.items[*].v' (JsonoTrieWildcardEdgePolicy + emit_index_tail;
    # JsonoTrieAppendWildcardMissing fires on items lacking 'v'). The 'd_flag' field uses bare
    # scalar shorthand so the literal-top-level-key resolution (the simple_scalar_leaf fast edge) is
    # exercised alongside the path wrappers.
    return (
        "{'a_str':{'type':'VARCHAR','path':'$.a'},"
        "'missing_str':{'type':'VARCHAR','path':'$.zzz_absent'},"
        "'nest_x':{'type':'VARCHAR','path':'$.nest.x'},"
        "'b_int':{'type':'BIGINT','path':'$.b'},"
        "'c_dbl':{'type':'DOUBLE','path':'$.c'},"
        "'d':'BOOLEAN',"
        "'item_vs':{'type':['VARCHAR'],'path':'$.items[*].v'}}"
    )


def transform_varchar_oracle_sql(plain: str, transformed: str) -> str:
    # Leg B: each VARCHAR field equals the single-path `->>` of the same JSONPath on the plain value,
    # except a container leaf (OBJECT/ARRAY) where the scalar transform yields NULL while `->>` would
    # render the container text — that divergence is by design (a container is a scalar mismatch), so
    # the oracle nulls it the same way. jsono_type returns NULL for a missing/JSON-null leaf, which is
    # not a container, so those fall through to `->>` (also NULL) and match.
    conds = []
    for field_name, path in TRANSFORM_VARCHAR_FIELDS:
        expected = (
            f"CASE WHEN jsono_type({plain}, '{path}') IN ('OBJECT', 'ARRAY') "
            f"THEN NULL ELSE {plain} ->> '{path}' END"
        )
        conds.append(f"(({transformed}).{field_name} IS NOT DISTINCT FROM ({expected}))")
    return "SELECT (" + " AND ".join(conds) + ")"


@settings(PROPERTY_SETTINGS)
# Force a >16-key object (checkpoint walk) carrying present named fields, a nested path, an absent
# key, and a wildcard array whose items mix present / absent / null / container '$.items[*].v'.
@example(
    generated=(
        {f"pad{i:02d}": i for i in range(17)}
        | {
            "a": "hi",
            "b": 5,
            "c": 1.5,
            "d": True,
            "nest": {"x": "deep"},
            "items": [{"v": "p"}, {"w": 1}, {"v": None}, {"v": [1]}],
        },
        17,
    )
)
# A <=16-key object (linear merge walk) with the same field shapes.
@example(generated=({"a": "x", "b": 2, "nest": {"x": "n"}, "items": [{"v": "q"}]}, 0))
# Empty wildcard array and a fully absent items container both exercise JsonoTrieAppendWildcardMissing.
@example(generated=({"a": "y", "items": []}, 0))
@example(generated=({"a": "z"}, 0))
@given(generated=transform_documents())
def test_transform_walk_parity(generated: tuple[dict[str, Any], int]) -> None:
    document, _ = generated
    text = json_dumps(document)
    spec = transform_spec_sql()
    plain = f"jsono({sql_literal(text)})"
    transformed_plain = f"jsono_transform({plain}, {spec})"

    # Leg A: the same multi-field spec over a SHREDDED build of the same document must equal the plain
    # build. The shred set covers the scalar fields the spec reads (so the shred-lane read path runs);
    # values that do not fit a typed shred stay in the residual (the lossless gate), exercising the
    # residual-first / shred-fallback split inside the walk.
    shred_spec = "{'$.a': 'VARCHAR', '$.b': 'BIGINT', '$.c': 'DOUBLE', '$.d': 'BOOLEAN'}"
    shredded = f"jsono({sql_literal(text)}, shredding := {shred_spec})"
    transformed_shredded = f"jsono_transform({shredded}, {spec})"
    leg_a = SESSION.value(f"SELECT (to_json({transformed_shredded}) IS NOT DISTINCT FROM to_json({transformed_plain}))")
    assert leg_a == "true", f"transform shredded != plain: {text!r}"

    # Leg B: each VARCHAR field matches the independent single-path `->>` extraction on the same value.
    leg_b = SESSION.value(transform_varchar_oracle_sql(plain, transformed_plain))
    assert leg_b == "true", f"transform field != ->> extraction: {text!r}"


def _object_reachable_null(value: Any) -> bool:
    # A JSON null sitting at an object key reachable through object chains only. merge_patch deletes on
    # such a null (it is the RFC 7396 delete sentinel), so the reverse-merge-patch round-trip cannot
    # reach it — the documented domain carve-out (§8). A null inside an array is preserved, because
    # merge_patch replaces an array wholesale, so arrays are not descended into here.
    if isinstance(value, dict):
        for child in value.values():
            if child is None:
                return True
            if isinstance(child, dict) and _object_reachable_null(child):
                return True
    return False


# The atomic mode is the exact inverse of jsono_merge_patch: applying the diff back onto prev must
# reproduce cur. The law holds for an object cur with no object-reachable null leaf (the carve-out);
# other shapes still exercise the diff walk (no crash) without the equality assertion.
@settings(PROPERTY_SETTINGS)
@example(prev_text='{"a":1,"b":{"x":1,"y":2}}', cur_text='{"a":2,"b":{"x":1},"c":[1,2]}')
@example(prev_text='{"a":5}', cur_text='{"a":{}}')  # scalar -> empty object: patch must carry key:{}
@example(prev_text="{}", cur_text='{"a":1,"b":[1,2,3],"c":{"d":true}}')  # everything new
@example(prev_text="5", cur_text='{"a":1}')  # non-object prev replaced by an object
@given(prev_text=json_documents, cur_text=json_documents)
def test_diff_atomic_round_trip(prev_text: str, cur_text: str) -> None:
    diff = f"jsono_diff(jsono({sql_literal(prev_text)}), jsono({sql_literal(cur_text)}))"
    cur_value = json_loads(cur_text)
    if not isinstance(cur_value, dict) or _object_reachable_null(cur_value):
        SESSION.value(f"to_json({diff})")  # exercise the path; the round-trip is out of the law's domain
        return
    patched = SESSION.value(f"to_json(jsono_merge_patch(jsono({sql_literal(prev_text)}), {diff}))")
    expected = SESSION.value(f"to_json(jsono({sql_literal(cur_text)}))")
    assert (
        patched == expected
    ), f"atomic diff round-trip failed: prev {prev_text!r} cur {cur_text!r}: {patched!r} != {expected!r}"


# The counts/elements array reports must obey their laws: counts == element-list lengths, the multiset
# reconstruction multiset(prev) ⊎ added == multiset(cur) ⊎ removed, and the symmetry
# diff(a,b).added == diff(b,a).removed (per array). Items are small scalars so collisions and dups occur.
diff_array_scalars = st.lists(
    st.one_of(
        st.integers(min_value=-3, max_value=3),
        st.sampled_from(["x", "y", "z"]),
        st.none(),
        st.booleans(),
    ),
    max_size=6,
)


@settings(PROPERTY_SETTINGS)
@example(prev_items=[1, 1, 2], cur_items=[1, 2, 2])
@example(prev_items=[1, 2], cur_items=[2, 1])  # pure reorder -> omitted
@example(prev_items=[None, 1], cur_items=[1])  # null is an ordinary array element
@given(prev_items=diff_array_scalars, cur_items=diff_array_scalars)
def test_diff_array_multiset_laws(prev_items: list[Any], cur_items: list[Any]) -> None:
    prev = json_dumps({"items": prev_items})
    cur = json_dumps({"items": cur_items})

    def diff(left: str, right: str, mode: str) -> Any:
        return json_loads(
            SESSION.value(
                f"to_json(jsono_diff(jsono({sql_literal(left)}), jsono({sql_literal(right)}), arrays:={sql_literal(mode)}))"
            )
        )

    counts = diff(prev, cur, "counts")
    elements = diff(prev, cur, "elements")
    if "items" not in counts:
        # multiset-equal (incl. pure reorder): both reports omit the key.
        assert "items" not in elements, f"counts omitted items but elements did not: prev {prev!r} cur {cur!r}"
        return
    added = elements["items"]["added"]
    removed = elements["items"]["removed"]
    assert counts["items"]["added"] == len(added), f"counts.added != |elements.added|: prev {prev!r} cur {cur!r}"
    assert counts["items"]["removed"] == len(
        removed
    ), f"counts.removed != |elements.removed|: prev {prev!r} cur {cur!r}"

    prev_rendered = json_loads(SESSION.value(f"to_json(jsono({sql_literal(prev)}))"))["items"]
    cur_rendered = json_loads(SESSION.value(f"to_json(jsono({sql_literal(cur)}))"))["items"]
    key = lambda items: Counter(json_dumps(item) for item in items)
    assert key(prev_rendered) + key(added) == key(cur_rendered) + key(
        removed
    ), f"multiset reconstruction failed: prev {prev!r} cur {cur!r}"

    reverse = diff(cur, prev, "elements")
    reverse_removed = reverse.get("items", {}).get("removed", [])
    assert key(added) == key(
        reverse_removed
    ), f"symmetry diff(a,b).added != diff(b,a).removed: prev {prev!r} cur {cur!r}"


# ===== Shredded lane fuzz: mutate the DuckDB-native side of a shredded value =====
#
# The residual fuzz above mutates the six blobs; this lane mutates the typed shred columns —
# truncating/extending an array lane against its skeleton, nulling lanes, flipping the shred-set
# marker and the spill bitmap, and swapping a scalar lane's value. Two properties per example:
# no reader crashes, and jsono_validate never blesses a row that to_json then rejects
# (validate=true implies readable — the lockstep contract).
LANE_MUTATIONS = [
    "arr_truncate",
    "arr_extend",
    "arr_null",
    "obj_arr_truncate",
    "obj_arr_extend",
    "obj_arr_null",
    "scalar_swap",
    "scalar_null",
    "spill_garbage",
    "spill_null",
    "marker_null",
    "marker_flip",
    "intact",
]


def lane_mutant_sql(doc_sql: str, mutation: str) -> str:
    j = f"jsono({doc_sql}, shredding := {{'$.arr':'BIGINT[]', k:'BIGINT', '$.items':'STRUCT(n BIGINT)[]'}})"
    marker = f'({j})."jsono".shreds."$jsono$set"'
    spill = f'({j})."jsono".shreds."$jsono$spill$0"'
    arr = f'({j})."jsono".shreds."$.arr"'
    items = f'({j})."jsono".shreds."$.items"'
    k_value = f'({j})."jsono".shreds."k"'
    if mutation == "arr_truncate":
        arr = f"list_slice({arr}, 1, greatest(len({arr}) - 1, 0))"
    elif mutation == "arr_extend":
        arr = f"list_append(coalesce({arr}, []::BIGINT[]), 99)"
    elif mutation == "arr_null":
        arr = "NULL::BIGINT[]"
    elif mutation == "obj_arr_truncate":
        items = f"list_slice({items}, 1, greatest(len({items}) - 1, 0))"
    elif mutation == "obj_arr_extend":
        items = f"list_append(coalesce({items}, []::STRUCT(n BIGINT)[]), {{'n': 99}}::STRUCT(n BIGINT))"
    elif mutation == "obj_arr_null":
        items = "NULL::STRUCT(n BIGINT)[]"
    elif mutation == "scalar_swap":
        k_value = f"coalesce({k_value}, 0) + 7"
    elif mutation == "scalar_null":
        k_value = "NULL::BIGINT"
    elif mutation == "spill_garbage":
        spill = "96::BIGINT"
    elif mutation == "spill_null":
        spill = "NULL::BIGINT"
    elif mutation == "marker_null":
        marker = "NULL::BIGINT"
    elif mutation == "marker_flip":
        marker = f"{marker} + 1"
    return (
        f'struct_pack("jsono" := struct_pack(body := ({j})."jsono".body, '
        f'shreds := struct_pack("$jsono$set" := {marker}, '
        f'"$jsono$spill$0" := {spill}, '
        f'"$.arr" := {arr}, '
        f'"$.items" := {items}, '
        f'"k" := {k_value})))'
    )


lane_documents = st.builds(
    lambda arr, k, items, extra: {"arr": arr, **({"k": k} if k is not None else {}), "items": items, **extra},
    st.lists(st.integers(min_value=-1000, max_value=1000), max_size=5),
    st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000)),
    st.lists(st.fixed_dictionaries({"n": st.integers(min_value=-1000, max_value=1000)}), max_size=4),
    st.dictionaries(shred_keys, json_scalars, max_size=2),
)


@settings(VALIDISH_PROPERTY_SETTINGS)
@example(doc={"arr": [1, 2, 3], "k": 5}, mutation="arr_truncate")
@example(doc={"arr": [1, 2, 3], "k": 5}, mutation="arr_extend")
@example(doc={"arr": [], "k": 5}, mutation="arr_extend")
@example(doc={"arr": [1], "k": 5}, mutation="marker_flip")
@example(doc={"arr": [1], "k": None}, mutation="spill_garbage")
@example(doc={"arr": [1], "k": 5, "items": [{"n": 1}, {"n": 2}]}, mutation="obj_arr_truncate")
@example(doc={"arr": [1], "k": 5, "items": [{"n": 1}, {"n": 2}]}, mutation="obj_arr_extend")
@example(doc={"arr": [1], "k": 5, "items": []}, mutation="obj_arr_extend")
@example(doc={"arr": [1], "k": 5, "items": [{"n": 1}]}, mutation="obj_arr_null")
@given(doc=lane_documents, mutation=st.sampled_from(LANE_MUTATIONS))
def test_fuzz_shredded_lanes_no_lie(doc: dict[str, Any], mutation: str) -> None:
    mutant = lane_mutant_sql(sql_literal(json_dumps(doc)), mutation)
    verdict = SESSION.read(f"jsono_validate({mutant})")
    assert not isinstance(verdict, JsonoSession.Errored), f"validate errored: {verdict!r}"
    rendered = SESSION.read(f"to_json({mutant})")
    SESSION.value(f"jsono_extract_string({mutant}, '$.k')")
    SESSION.value(f"(SELECT count(*) FROM (SELECT unnest(jsono_entries({mutant}))))")
    SESSION.value(f"(SELECT to_json(jsono_shred_stats(v)) FROM (VALUES ({mutant})) t(v))")
    if verdict == "true":
        assert not isinstance(
            rendered, JsonoSession.Errored
        ), f"validate blessed an unreadable row: doc={doc!r} mutation={mutation} -> {rendered!r}"


PROPERTIES = [
    test_round_trip_idempotent,
    test_value_parity,
    test_keys_sorted,
    test_shred_lossless,
    test_reshred_lossless,
    test_parquet_round_trip_to_json_parity,
    test_merge_patch_shredded_plain_parity,
    test_merge_all_shredded_parity,
    test_merge_patch_auto_shred_patch_parity,
    test_array_reader_parity,
    test_scalar_array_reader_parity,
    test_struct_auto_shred_lossless,
    test_struct_constructor_auto_shred_matrix,
    test_struct_constructor_auto_shred_parquet,
    test_struct_array_auto_shred_lossless,
    test_scalar_array_auto_shred_lossless,
    test_narrowing_fails_loud,
    test_group_merge_keyed_parity,
    test_transform_walk_parity,
    test_diff_atomic_round_trip,
    test_diff_array_multiset_laws,
    test_fuzz_text_no_crash,
    test_fuzz_blob_no_crash,
    test_fuzz_validish_blob_no_crash,
    test_fuzz_manifest_no_crash,
    test_fuzz_shredded_lanes_no_lie,
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
