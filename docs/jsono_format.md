# JSONO — binary format specification

JSONO stores JSON as a sequence of 64-bit slots plus two byte heaps and a
navigation block. The logical type is `JSONO`; physically it is a DuckDB STRUCT
of four BLOBs:

```
STRUCT(
  jsono_slots       BLOB,   -- [Header 8 bytes][Slots n × 8 bytes]
  jsono_key_heap    BLOB,   -- key bytes, addressed by KEY slots
  jsono_string_heap BLOB,   -- string bytes, consumed by a cursor in walk order
  jsono_skips       BLOB    -- navigation metadata (see below)
)
```

The `JSONO` alias survives round-trips through DuckDB-native storage (catalog
metadata) but is lost through `read_parquet`, which delivers the structurally
compatible STRUCT with the same four `jsono_*` BLOB fields.

## Compatibility policy

The on-disk binary format is versioned by the `version` byte in `jsono_slots`.
Current `JSONO` files use `version = 1`.

This extension is still experimental, so compatibility is intentionally strict:
an incompatible change to slot tags, payload semantics, heap layout, navigation
metadata, or required header flags must bump `jsono::VERSION`. Readers accept
only the current version and do not attempt a silent best-effort decode.
Public helpers expose a version/header mismatch through their normal invalid
blob behavior: `jsono_validate` returns `false`, while extraction/serialization
helpers that treat malformed headers as absent values return SQL `NULL`.
Backward readers or migration functions should be added explicitly when
persisted old JSONO files become a supported contract.

## Data model

JSONO stores a JSON value losslessly per the RFC 8259 data model, but it is not a
byte-exact copy of the source text: insignificant serialization details are
normalized (object key order, duplicate keys, insignificant whitespace). This is
legal — RFC 8259 §4 defines an object as an unordered collection of pairs, so key
order carries no information at the value level. JSONO stores each object's keys
sorted by bytes (`SORTED_KEYS` flag in the Header), and duplicate object keys
collapse to the last occurrence before sorting. Array element order is
significant (RFC 8259 §5, ordered sequence) and is preserved. Number text is
preserved when it is stored as NUMBER or a DEC60-compatible decimal; integer
forms are numeric values, so `-0` normalizes to `0` (see
[Number model](#number-model)).

JSONO is a representation of JSON, not a typed VARIANT. Only the seven JSON
values are supported (object, array, string, number, true, false, null); types
outside JSON (date, uuid, blob, decimal(p,s), etc.) are deliberately not
introduced.

Heap deduplication is not done at the format level: heaps may contain duplicate
strings and keys. That is delegated to Parquet column-dictionary encoding — see
[split-heap rationale](#why-two-heaps-split-heap).

## Header

8 bytes at the start of `jsono_slots`, little-endian:

| field | type | value |
|---|---|---|
| `magic` | u32 | `'JSNO'` = `0x4F4E534A` |
| `version` | u8 | `1` |
| `flags` | u8 | bit0 = `SORTED_KEYS` |
| `reserved` | u16 | `0` |

Header parsing fails if `magic`/`version` do not match, or if the slot length
minus the Header is not a multiple of 8. Public functions then follow the invalid
blob behavior described in [Compatibility policy](#compatibility-policy).

## Slot

Slots follow the Header, 8 bytes each, little-endian:

```
slot = [4-bit tag][60-bit payload]
TAG_SHIFT = 60
PAYLOAD_MASK = (1 << 60) - 1
```

Payload semantics depend on the tag.

### Tag table

| hex | bin | name | group | slots | payload |
|---|---|---|---|---|---|
| `0x0` | `0000` | OBJ_START | structure | 1 | `container_id`(36) ‹‹ 24 \| `child_count`(24) |
| `0x1` | `0001` | OBJ_END | structure | 1 | — |
| `0x2` | `0010` | ARR_START | structure | 1 | `container_id`(36) ‹‹ 24 \| 0 |
| `0x3` | `0011` | ARR_END | structure | 1 | — |
| `0x4` | `0100` | KEY | structure | 1 | `key_offset`(36) ‹‹ 24 \| `key_len`(24) |
| `0x5` | `0101` | VAL_NULL | literal | 1 | — |
| `0x6` | `0110` | VAL_TRUE | literal | 1 | — |
| `0x7` | `0111` | VAL_FALSE | literal | 1 | — |
| `0x8` | `1000` | VAL_STR_HEAP | value (hot) | 1 | `str_len`(60), offset implicit |
| `0x9` | `1001` | — | reserved | | free |
| `0xA` | `1010` | VAL_INT60 | value (hot) | 1 | signed int60 inline |
| `0xB` | `1011` | VAL_DEC60 | value (hot) | 1 | sign + scale + mantissa inline |
| `0xC` | `1100` | — | reserved | | free |
| `0xD` | `1101` | — | reserved | | free |
| `0xE` | `1110` | — | reserved | | free |
| `0xF` | `1111` | VAL_EXT | escape | 1+ | subtype byte(8) + payload |

Field widths are fixed in `jsono.hpp`: `CONTAINER_CHILD_COUNT_BITS = 24`,
`HEAP_LEN_BITS = 24`. `container_id` and `key_offset` take the remaining 36
payload bits.

### Tag groups

**Structure (0x0–0x4).** Containers (`OBJ_*`, `ARR_*`) and keys (`KEY`).
`container_id` indexes the `ContainerSpan` table in `jsono_skips` (see
[Navigation](#navigation-jsono_skips)). An object's `child_count` is the number
of key/value pairs; arrays write 0. KEY references `jsono_key_heap` via explicit
offset+len.

**Literals (0x5–0x7).** `VAL_NULL`, `VAL_TRUE`, `VAL_FALSE` — no payload, one
slot.

**Hot values (0x8, 0xA, 0xB).** Encodings for values that occur en masse on the
main (text) parse path:

- **VAL_STR_HEAP** (0x8) — a string whose bytes live in `jsono_string_heap`. The
  payload carries only the length; the byte offset is **implicit** (see
  [Read model](#read-model-string-cursor)).
- **VAL_INT60** (0xA) — primary for integers. A signed integer inline in the
  60-bit payload (sign-extended from bit 59), range `INT60_MIN..INT60_MAX` =
  `−2^59 .. 2^59−1` (≈±5.76e17, 18 digits). Read without decoding.
- **VAL_DEC60** (0xB) — primary for fractionals. `value = mantissa / 10^scale`.
  Packing and lossless bounds are [below](#val_dec60--packing-and-lossless-bounds).

Slot `0x9` is reserved (free within the value region).

**Escape (0xF) — VAL_EXT.** An extensible escape tag. The low 8 bits of the
payload are the subtype byte, selecting a concrete encoding. The whole rare
numeric tail and any future storage encodings go here without spending scarce hot
tags.

| subtype | code | purpose | storage |
|---|---|---|---|
| INT64 | 0 | signed integer outside int60, within int64 | +1 slot, 64 raw bits |
| UINT64 | 1 | unsigned `2^63..2^64−1` | +1 slot, 64 raw bits |
| DOUBLE | 2 | IEEE double bits (`jsono(STRUCT)` path, no source text) | +1 slot, 64 bits |
| NUMBER | 3 | raw bignum / high-precision decimal, byte-exact | text in `string_heap`, no trailing slot |

The number of trailing slots is stored in the `EXT_SUBTYPE_TRAILING_SLOTS` table
so the reader counts slots from data rather than by branching. NUMBER puts its
text in `jsono_string_heap` (payload = `text_len ‹‹ 8 | subtype`) and is consumed
by the same cursor as VAL_STR_HEAP — hence no trailing slot.

## Container layout

Inside an object, keys and values are stored as **separate blocks**, not
interleaved. After `OBJ_START` come all `child_count` `KEY` slots (sorted by
bytes), then all `child_count` values in the same order, then `OBJ_END`:

```
OBJ_START                                  ← pos
  KEY[0] KEY[1] … KEY[N-1]                  ← key_start = pos + 1
  VALUE[0] VALUE[1] … VALUE[N-1]            ← value_start = key_start + N
OBJ_END                                     ← after_pos − 1
```

`VALUE[i]` is the subtree of the i-th key: one slot for a scalar, or a whole
nested container. Arrays are simpler — no key block:

```
ARR_START
  VALUE[0] VALUE[1] …
ARR_END
```

Separate key and value blocks give: (a) a predictable value-block offset
(`value_start = OBJ_START + 1 + N`), (b) cheap key lookup over the sorted key
block, (c) the basis for object checkpoints (see below).

## Read model: string cursor

A string's byte offset into `jsono_string_heap` is **not stored in the slot** —
it is implicit. The reader keeps a single cursor and, during the walk, advances
it by the length of each consumed string value:

- `VAL_STR_HEAP` — cursor += `str_len`;
- `VAL_EXT/NUMBER` — cursor += `text_len` (the same shared cursor);
- other tags consume no string bytes.

The cursor advances **in tree-walk order** (depth-first; an object's values come
after its key block). This removes the 36-bit offset from every string slot, at
the cost that random access to the k-th value needs the accumulated cursor. The
navigation block removes that cost.

## Navigation (`jsono_skips`)

`jsono_skips` is a separate BLOB that lets a reader skip subtrees and jump to the
k-th child of an object without replaying the whole string cursor. Layout (all
sections little-endian, fixed-size records):

```
[ContainerMetadataHeader]            12 bytes
[ContainerSpan        × container_count]        8 bytes each
[ObjectCheckpointIndex × checkpoint_index_count] 12 bytes each
[ObjectCursorCheckpoint × checkpoint_count]      8 bytes each
```

`ContainerMetadataHeader { u32 container_count; u32 checkpoint_index_count;
u32 checkpoint_count; }` gives the sizes of the three sections; the reader checks
that their combined size fits within the blob.

### ContainerSpan

One record per container, indexed by the `container_id` from the
`OBJ_START`/`ARR_START` payload:

```
ContainerSpan { u32 slot_span; u32 string_byte_span; }
```

- `slot_span` — how many slots the whole container occupies, including `*_START`
  and `*_END`. Jump past the entire subtree: `pos += slot_span`.
- `string_byte_span` — how many `jsono_string_heap` bytes this subtree consumes.
  Jump the string cursor: `string_cursor += string_byte_span`.

Together they give O(1) `SkipValueFast` — skipping any value without descending
into it.

### Object checkpoints

`ContainerSpan` skips a whole subtree, but does not help jumping **into** a large
object to a specific key: the string cursor would have to be accumulated by
replaying all preceding values. For that, objects with
`child_count > OBJECT_CHECKPOINT_STRIDE` (= 16) get checkpoints.

`ObjectCheckpointIndex` — one record per such object, **sorted by
`container_id`** (the reader binary-searches it, `ObjectCheckpointIndexForContainer`):

```
ObjectCheckpointIndex {
  u32 container_id;
  u32 checkpoint_offset;   // offset into the ObjectCursorCheckpoint array
  u16 checkpoint_stride;   // step (ranks between checkpoints)
  u16 reserved;
}
```

`ObjectCursorCheckpoint` — a snapshot of the cursors at every `stride`-th value
boundary, as deltas **relative to the start of the object's value block**:

```
ObjectCursorCheckpoint { u32 slot_delta; u32 string_delta; }
```

To reach the value at rank `r`: take the nearest checkpoint
`checkpoint_rank = r − r % stride`, position at `value_start + slot_delta` /
`object_string_cursor + string_delta`, then skip the remaining `< stride` values
with `SkipValueFast` (`MoveObjectValueCursorToRank`, `jsono_reader.hpp`). The
object cursor only moves forward.

`stride` is currently fixed: `OBJECT_CHECKPOINT_STRIDE = 16` for ordinary
objects, `LARGE_OBJECT_CHECKPOINT_STRIDE = 16` for objects with
`child_count ≥ LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT` (= 64). The two values
currently coincide; they are kept as separate constants for future tuning of
checkpoint density on large objects.

## Why two heaps (split-heap)

Keys (byte-identical across rows that share a schema) live in `jsono_key_heap`,
while per-row string values live in `jsono_string_heap`. For shape-stable data,
Parquet column-dictionary encoding collapses `jsono_key_heap` to a single entry
per column chunk, and the byte patterns of `KEY` slots become identical across
rows (the offset of key `"browser"` is always the same byte position while the
key set is the same). That is why `KEY` carries offset+len explicitly (cheap
after dict-encoding) while `VAL_STR_HEAP` carries only len (offset implicit via
the cursor).

## Number model

The full encoding ladder for a JSON number (`jsono_number.cpp`,
`JsonoBuilder::Emit*`):

```
INTEGER:
  -2^59 .. 2^59-1                    -> VAL_INT60       (1 slot, hot)
  -2^63 .. -2^59-1 / 2^59 .. 2^63-1  -> VAL_EXT/INT64    (2 slots)
  2^63 .. 2^64-1                     -> VAL_EXT/UINT64   (2 slots)
  < -2^63 / > 2^64-1                 -> VAL_EXT/NUMBER   (raw text)

FRACTIONAL (from text):
  fits-decimal           -> VAL_DEC60           (1 slot, hot)
  else (exponent, scale>22, mantissa>=2^53) -> VAL_EXT/NUMBER  (raw text)

FRACTIONAL (binary double from the struct constructor, no text):
                         -> VAL_EXT/DOUBLE      (2 slots)
```

The integer/DEC60 classifier does no float arithmetic — it reads digits out of
the text. Integer text is stored by numeric value, so JSON's valid `-0` integer
form is emitted back as `0`. DEC60 is chosen only when the canonical
mantissa/scale → text reconstruction is byte-identical to the input; otherwise
the number falls through to NUMBER and is preserved byte-exact.

Allocation principle: mass values on the text path → a hot tag; rare values →
VAL_EXT. DOUBLE under VAL_EXT by default; promoting it to a hot tag is a
benchmark-decidable decision (justified only for double-heavy data).

### VAL_DEC60 — packing and lossless bounds

Payload (60 bits):

```
bit 59      : sign (1 = negative mantissa)
bits 54..58 : scale (0..22, 5 bits)
bits 0..53  : |mantissa| (< 2^53, 54-bit field)
```

`value = mantissa / 10^scale`. Reconstruction at read time is done by
**division** by an exact power of ten, never multiplication:

- OK: `mantissa / pow10[scale]` — `10^scale` and `mantissa` are exact as double,
  IEEE division is correctly-rounded → the result is bit-equal to
  `parse("<text>")`.
- NO: `mantissa * 10^(-scale)` — `10^(-scale)` is not exactly representable in
  double → error is introduced and amplified, the round-trip breaks.

The encoding is valid (lossless) only within the bounds (`FitsDec60`):

- `|mantissa| < 2^53` (integer exact as double);
- `scale <= 22` (`10^22` is the last exactly representable power of ten).

Pure integers never take the DEC60 path (`scale >= 1`). Outside the bounds a
fractional goes to `VAL_EXT/NUMBER` (raw text).

## Reserved (0x9, 0xC–0xE)

Not implemented. Four free tags for encodings whose frequency must be proven by a
benchmark on real data (do not bake them in blindly):

- **SHORT_STRING** — a short string inline in the payload (≤7 bytes): enums,
  codes, statuses.
- **STRING_REF** — a reference into a pool of deduplicated values (repeated
  enumerations).
- **FLOAT60** — a double with a zero low nibble, lossless via `bits >> 4` /
  `bits << 4` (for float32-heavy data).
- **DOUBLE as a hot tag** — if the data profile turns out to be double-heavy.
