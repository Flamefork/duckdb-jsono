# JSONO — binary format specification

JSONO stores JSON as a sequence of 64-bit slots plus two byte heaps, two
fixed-width payload streams, and a navigation block. There is no logical type
alias; physically a JSONO value is a nested DuckDB STRUCT with exactly one
layout field, `jsono`, wrapping the six-BLOB `body`:

```
STRUCT(
  jsono STRUCT(
    body STRUCT(
      slots       BLOB,   -- [Header 8 bytes][Slots n × 8 bytes]
      key_heap    BLOB,   -- key bytes, addressed by KEY slots
      string_heap BLOB,   -- string/number-text bytes, consumed by a cursor in walk order
      skips       BLOB,   -- navigation metadata + shred manifest (see below)
      lengths     BLOB,   -- u32 LE byte lengths of string/number-text values, walk order
      nums        BLOB    -- u64 LE numeric payloads, walk order
    )
  )
)
```

There is no user-defined type alias (it could not survive `read_parquet` or
DuckLake); a stored value is just this nested STRUCT, recognised structurally.
The six body BLOBs round-trip through every storage layer unchanged.

Splitting the variable payloads out of the slots is deliberate: a v3 slot
carried its payload inline (string length, inline int60), so two rows with the
same shape still produced different slot bytes. In v4 every value slot is a
shape-constant tag word — rows sharing a schema produce byte-identical `slots`
blobs — and the per-row variability is concentrated in the small fixed-width
`lengths`/`nums` streams, which general-purpose compression (Parquet + zstd)
handles far better.

## Shredded layout

For fast columnar reads of hot paths, a JSONO value can be stored *shredded*:
the `jsono` layout field carries the `body` plus one typed *shred* per
chosen path, as the `body`'s sibling fields (produced by
`jsono(value, shredding := spec)`).

```
STRUCT(
  jsono STRUCT(
    body STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB,
                lengths BLOB, nums BLOB),
    "<path1>" <type1>, -- shred: value at <path1>, typed, named by the path
    "<path2>" <type2>,
    …
  )
)
```

Shredding does **not** change the binary slot format. The six `body` blobs are
an ordinary JSONO value — the *residual* — so everything else in this spec
applies to them unchanged, including `version`. Each *shred* is the
value at its canonical path (`$.kind`, `$.commit.operation`, …) materialized as
a plain typed DuckDB column (`VARCHAR`, `BIGINT`, `UBIGINT`, `DOUBLE`,
`BOOLEAN`), with the path as the field name. Nested shred paths are allowed. No
JSON key is reserved: a JSON key named `body` can still be shredded through its
`$.`-prefixed path form (`$.body`), which is a different field name.

The single layout name (`jsono` for both plain and shredded) and the shreds
sitting beside `body` are both deliberate. DuckDB reconciles struct types by
field name, so a set operation, CASE or COALESCE over differently-shredded
values merges them into one ordinary shredded type carrying the union of the
shred sets — the struct cast `NULL`-fills a branch's missing shreds, which is
exactly the writer's own "value stayed in the residual" encoding, so the merged
value reads losslessly by construction. And because `body` is a field every two
JSONO types share, DuckDB's by-name struct cast *binds* between any two of them
— fully disjoint shred sets included — so the extension optimizer can rewrite
every such cast into a lossless reconstruct + reshred. The remaining flip side —
a raw by-name cast silently *dropping* a shred when the optimizer is not running
— is caught at read time by the shred manifest (below), not by the type system.

Shreds are emitted in canonical order (sorted by name), so a constructed
type is a pure function of the shred *set*: `jsono_storage_type(<shred DDL>)` and
the constructor produce the identical type for the same shreds regardless of the
order they are listed in, so a column declared from one accepts a value built
from the other. (A type DuckDB merges out of differently-shredded branches lists
the left branch's shreds first — same shred set, branch-dependent field order.)
Shreds stay scalar columns, so Parquet/DuckLake projection and filter
pushdown act on them directly.

**Lossless invariant: the residual holds everything a shred cannot reproduce
exactly.** A value is removed from the residual and kept only in its shred when it
round-trips losslessly through the shred type (a JSON string in a `VARCHAR` shred,
an integer in a `BIGINT` shred). A value that does not fit — a string in a `BIGINT`
shred, an explicit JSON `null`, a missing key, or a number whose shred type would
re-encode it differently — stays in the residual, and its shred is a redundant
read copy. Pure object-key paths that round-trip through their shred type are
removed from the residual; array-index paths and non-lossless shred values stay in
the residual.

The original value is recovered by overlaying the shreds onto the residual,
**residual-authoritative**: a path absent from the residual is filled from its
shred, and a path present in the residual wins (its shred is the redundant copy).
Since the removed values are exactly those their shred reproduces byte-for-byte,
the overlay reconstructs the value losslessly.

### Shred manifest

The `skips` blob of a shredded residual ends with a *shred manifest*: the
write-time record of which paths were stripped out of *this row's* residual,
with the shred type each was written as. After the checkpoint sections (see
[Navigation](#navigation-skips)) the tail is:

```
u32 entry_count
per entry: u16 path_len, path bytes, u16 type_len, type bytes
```

Entries are sorted by path (the shred writer emits fields in canonical order)
and list only the paths actually stripped from this row (a value kept in the
residual by the lossless gate is not listed). A plain value writes no manifest —
its `skips` blob ends at the checkpoints, which reads as zero entries.

The manifest is what makes a raw by-name struct cast safe to detect: if a cast
drops a shred (the target type carries fewer shreds) or converts one to a
different type, the residual cannot reproduce the value — and the manifest
proves it. Every reader of a shredded residual verifies the row's manifest
against the shreds actually available and fails loud on a mismatch instead of
silently returning partial data. Extra shreds beyond the manifest are legal (a
widening cast `NULL`-fills them; readers fall back to the residual).

## Compatibility policy

The on-disk binary format is versioned by the `version` byte in `slots`.
Current `JSONO` files use `version = 4`:

- version 2 added the per-object `shape_hash` field to `ContainerSpan` (see
  [Navigation](#navigation-skips));
- version 3 added the shred manifest tail to `skips` (see
  [Shred manifest](#shred-manifest));
- version 4 moved the variable slot payloads into the `lengths`/`nums` streams
  (slots are pure tags), made `ContainerSpan`/`ObjectCursorCheckpoint` carry
  per-stream counts/deltas, and made span storage *sparse*: only arrays,
  objects with `child_count > OBJECT_CHECKPOINT_STRIDE`, and objects whose
  immediate children include a container store a span, addressed through a
  sorted sparse container-id index.

This extension is still experimental, so compatibility is intentionally strict:
an incompatible change to slot tags, payload semantics, heap layout, navigation
metadata, or required header flags must bump `jsono::VERSION`. Readers accept
only the current version and do not attempt a silent best-effort decode; the
current version is also exposed in SQL as `jsono_version()`.

Public helpers distinguish an *absent* value from a *corrupt* one. A slots blob
too short to hold a header is an absent value and reads as SQL `NULL`. A present
blob whose header is unreadable — wrong `magic`, a different `version`, misaligned
slots, or a metadata blob shorter than its declared spans — is corruption or
non-JSONO bytes bound to a jsono op, so extraction/serialization helpers raise an
error rather than silently reading as `NULL`. `jsono_validate` recovers a boolean
from these errors and returns `false`. Backward readers or migration functions
should be added explicitly when persisted old JSONO files become a supported
contract.

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

8 bytes at the start of `slots`, little-endian:

| field | type | value |
|---|---|---|
| `magic` | u32 | `'JSNO'` = `0x4F4E534A` |
| `version` | u8 | `4` |
| `flags` | u8 | bit0 = `SORTED_KEYS` |
| `reserved` | u16 | `0` |

A slots blob too short to hold a header is an absent value (SQL `NULL`).
Otherwise header parsing raises if `magic`/`version` do not match, if the slot
length minus the Header is not a multiple of 8, if `lengths`/`nums` are not whole
numbers of 4-/8-byte entries, or if the metadata blob is shorter than its
declared spans — see the invalid blob behavior in
[Compatibility policy](#compatibility-policy).

## Slot

Slots follow the Header, 8 bytes each, little-endian:

```
slot = [4-bit tag][60-bit payload]
TAG_SHIFT = 60
PAYLOAD_MASK = (1 << 60) - 1
```

Payload semantics depend on the tag. Since v4, value slots carry **no
variable data**: lengths live in `lengths`, numeric words in `nums`, both
consumed by walk-order cursors. This makes value slots shape-constant: two rows
with the same JSON shape produce byte-identical slot words.

### Tag table

| hex | bin | name | group | payload | stream entry |
|---|---|---|---|---|---|
| `0x0` | `0000` | OBJ_START | structure | `container_id`(36) ‹‹ 24 \| `child_count`(24) | — |
| `0x1` | `0001` | OBJ_END | structure | — | — |
| `0x2` | `0010` | ARR_START | structure | `container_id`(36) ‹‹ 24 \| 0 | — |
| `0x3` | `0011` | ARR_END | structure | — | — |
| `0x4` | `0100` | KEY | structure | `key_offset`(36) ‹‹ 24 \| `key_len`(24) | — |
| `0x5` | `0101` | VAL_NULL | literal | — | — |
| `0x6` | `0110` | VAL_TRUE | literal | — | — |
| `0x7` | `0111` | VAL_FALSE | literal | — | — |
| `0x8` | `1000` | VAL_STR_HEAP | value (hot) | 0 | one `lengths` u32 (byte length) |
| `0x9` | `1001` | — | reserved | | free |
| `0xA` | `1010` | VAL_INT60 | value (hot) | 0 | one `nums` u64 (two's-complement int64) |
| `0xB` | `1011` | VAL_DEC60 | value (hot) | 0 | one `nums` u64 (packed sign+scale+mantissa) |
| `0xC` | `1100` | — | reserved | | free |
| `0xD` | `1101` | — | reserved | | free |
| `0xE` | `1110` | — | reserved | | free |
| `0xF` | `1111` | VAL_EXT | escape | subtype byte(8) | one `nums` u64 or one `lengths` u32 |

Field widths are fixed in `jsono.hpp`: `CONTAINER_CHILD_COUNT_BITS = 24`,
`HEAP_LEN_BITS = 24`. `container_id` and `key_offset` take the remaining 36
payload bits.

### Tag groups

**Structure (0x0–0x4).** Containers (`OBJ_*`, `ARR_*`) and keys (`KEY`).
`container_id` is a sequential per-row counter (every container gets one); it
addresses the sparse `ContainerSpan` table in `skips` (see
[Navigation](#navigation-skips)). An object's `child_count` is the number
of key/value pairs; arrays write 0. KEY references `key_heap` via explicit
offset+len.

**Literals (0x5–0x7).** `VAL_NULL`, `VAL_TRUE`, `VAL_FALSE` — no payload, no
stream entry.

**Hot values (0x8, 0xA, 0xB).** Encodings for values that occur en masse on the
main (text) parse path. The slot is a pure tag; the payload is one stream entry:

- **VAL_STR_HEAP** (0x8) — a string whose bytes live in `string_heap`. Its byte
  length is the next `lengths` entry; the byte offset is **implicit** (see
  [Read model](#read-model-stream-cursors)). The fixed u32 length entry caps a
  single string value at 4 GiB (the fixed width is what keeps `lengths`
  random-accessible for checkpoint jumps).
- **VAL_INT60** (0xA) — primary for integers in `−2^59 .. 2^59−1` (the historical
  inline range that names the tag). The value is the next `nums` entry, a full
  64-bit two's-complement word.
- **VAL_DEC60** (0xB) — primary for fractionals. The next `nums` entry holds the
  packed `sign + scale + mantissa` word (top four bits zero); `value = mantissa /
  10^scale`. Packing and lossless bounds are
  [below](#val_dec60--packing-and-lossless-bounds).

Slot `0x9` is reserved (free within the value region).

**Escape (0xF) — VAL_EXT.** An extensible escape tag. The low 8 bits of the
payload are the subtype byte, selecting a concrete encoding. The whole rare
numeric tail and any future storage encodings go here without spending scarce hot
tags. Every VAL_EXT value is a **single slot**:

| subtype | code | purpose | stream entry |
|---|---|---|---|
| INT64 | 0 | signed integer outside int60, within int64 | one `nums` u64 (raw bits) |
| UINT64 | 1 | unsigned `2^63..2^64−1` | one `nums` u64 (raw bits) |
| DOUBLE | 2 | IEEE double bits (`jsono(STRUCT)` path, no source text) | one `nums` u64 (raw bits) |
| NUMBER | 3 | raw bignum / high-precision decimal, byte-exact | one `lengths` u32; text in `string_heap` |

NUMBER puts its text in `string_heap` and is consumed by the same cursors as
VAL_STR_HEAP.

## Container layout

Inside an object, keys and values are stored as **separate blocks**, not
interleaved. After `OBJ_START` come all `child_count` `KEY` slots (sorted by
bytes), then all `child_count` values in the same order, then `OBJ_END`:

```
OBJ_START                                  ← pos
  KEY[0] KEY[1] … KEY[N-1]                  ← key_start = pos + 1
  VALUE[0] VALUE[1] … VALUE[N-1]            ← value_start = key_start + N
OBJ_END
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

## Read model: stream cursors

None of the per-value payload locations are stored — they are implicit. The
reader keeps one cursor per stream (`JsonoCursor`: slot position, `string_heap`
byte offset, `lengths` entry index, `nums` entry index) and, during the walk,
advances them together:

- `VAL_STR_HEAP` / `VAL_EXT/NUMBER` — `len = lengths[length_cursor++]`; bytes at
  `string_heap[string_cursor .. +len)`; `string_cursor += len`;
- `VAL_INT60` / `VAL_DEC60` / `VAL_EXT INT64/UINT64/DOUBLE` —
  `word = nums[num_cursor++]`;
- literals, keys and container brackets consume no stream entries.

The cursors advance **in tree-walk order** (depth-first; an object's values come
after its key block). This removes all offsets and lengths from the value slots,
at the cost that random access to the k-th value needs the accumulated cursors.
The navigation block removes that cost: `ContainerSpan` and
`ObjectCursorCheckpoint` both carry per-stream counts/deltas, so a subtree skip
or a checkpoint jump moves the whole cursor in O(1).

## Navigation (`skips`)

`skips` is a separate BLOB that lets a reader skip subtrees and jump to the
k-th child of an object without replaying the stream cursors. Layout (all
sections little-endian, fixed-size records):

```
[ContainerMetadataHeader]                        12 bytes
[u32 container_id        × span_count]            4 bytes each, sorted ascending
[ContainerSpan           × span_count]           24 bytes each
[ObjectCheckpointIndex   × checkpoint_index_count] 12 bytes each
[ObjectCursorCheckpoint  × checkpoint_count]     16 bytes each
[shred manifest]                                  (shredded residuals only)
```

`ContainerMetadataHeader { u32 span_count; u32 checkpoint_index_count;
u32 checkpoint_count; }` gives the sizes of the sections; the reader checks
that their combined size fits within the blob.

### ContainerSpan — sparse, skip-cost driven

Spans are stored **only** for containers whose skip cannot be done cheaply by a
walk:

- every **array** — no inline child count, so a walk would not know where to
  stop without paying attention to nesting;
- every **object with `child_count > OBJECT_CHECKPOINT_STRIDE`** (= 16) —
  exactly the objects that also get cursor checkpoints;
- every **object whose immediate children include a container** (object or
  array) — without a span, skipping it would walk the whole child subtree
  instead of jumping.

Small **flat** objects (`child_count <= OBJECT_CHECKPOINT_STRIDE`, scalar
children only) store nothing; a reader skips one by walking its `child_count`
value slots — all scalars, so the walk touches only the slot words already in
cache. On shape-stable data this removes the dominant per-container metadata
cost: most real leaf objects are small and flat, while every skip that crosses
a subtree stays O(1).

The sorted `container_id` array before the spans is the sparse index; a reader
resolves it in `JsonoView::TryContainerSpan`. The ids are strictly ascending,
so `ids[i] >= i`: the entry for container `t`, if stored, sits at index
`<= t`. Near-root ids (the hottest lookups — the row root is always id 0) are
found by a short downward scan from `min(t, count−1)`; larger ids fall back to
a binary search, the same way object checkpoints are looked up. Container ids
stay sequential across *all* containers (the writer's counter), so they remain
usable as identity in validation; only the span table is sparse.

```
ContainerSpan {
  u32 slot_span;          // slots the whole container occupies, incl. *_START/*_END
  u32 string_byte_span;   // string_heap bytes the subtree consumes
  u64 shape_hash;         // objects: fingerprint of the sorted key sequence; arrays: 0
  u32 length_count;       // `lengths` entries the subtree consumes
  u32 num_count;          // `nums` entries the subtree consumes
}
```

- Jump past the entire subtree: `pos += slot_span`, `string_cursor +=
  string_byte_span`, `length_cursor += length_count`, `num_cursor += num_count` —
  O(1) `SkipValueFast` for spanned containers.
- `shape_hash` — a 64-bit fingerprint of the object's sorted, length-prefixed key
  sequence (`HashObjectKeySlots`). Two objects with equal `shape_hash` have the
  same sorted key set (collision risk ~5e-20), so a reader's per-object key-rank
  cache can trust a cached rank by an int compare instead of re-reading the key
  heap. Every spanned object stores its hash; the rank-cache fast path uses it
  only for the checkpointed ones (where the rank cache matters) and falls back
  to key validation for small objects, whose key blocks are at most stride-wide
  anyway.

### Object checkpoints

`ContainerSpan` skips a whole subtree, but does not help jumping **into** a large
object to a specific key: the stream cursors would have to be accumulated by
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
ObjectCursorCheckpoint { u32 slot_delta; u32 string_delta; u32 length_delta; u32 num_delta; }
```

To reach the value at rank `r`: take the nearest checkpoint
`checkpoint_rank = r − r % stride`, position every cursor at its value-block
base plus the checkpoint delta, then skip the remaining `< stride` values
with `SkipValueFast` (`MoveObjectValueCursorToRank`, `jsono_reader.hpp`). The
object cursor only moves forward.

`stride` is currently fixed: `OBJECT_CHECKPOINT_STRIDE = 16` for ordinary
objects, `LARGE_OBJECT_CHECKPOINT_STRIDE = 16` for objects with
`child_count ≥ LARGE_OBJECT_CHECKPOINT_MIN_CHILD_COUNT` (= 64). The two values
currently coincide; they are kept as separate constants for future tuning of
checkpoint density on large objects.

## Why two heaps (split-heap)

Keys (byte-identical across rows that share a schema) live in `key_heap`,
while per-row string values live in `string_heap`. For shape-stable data,
Parquet column-dictionary encoding collapses `key_heap` to a single entry
per column chunk, and the byte patterns of `KEY` slots become identical across
rows (the offset of key `"browser"` is always the same byte position while the
key set is the same). That is why `KEY` carries offset+len explicitly (cheap
after dict-encoding) while string values carry nothing in the slot at all (length
in the `lengths` stream, offset implicit via the cursor).

## Number model

The full encoding ladder for a JSON number (`jsono_number.cpp`,
`JsonoBuilder::Emit*`):

```
INTEGER:
  -2^59 .. 2^59-1                    -> VAL_INT60       (hot; value in nums)
  -2^63 .. -2^59-1 / 2^59 .. 2^63-1  -> VAL_EXT/INT64    (value in nums)
  2^63 .. 2^64-1                     -> VAL_EXT/UINT64   (value in nums)
  < -2^63 / > 2^64-1                 -> VAL_EXT/NUMBER   (raw text)

FRACTIONAL (from text):
  fits-decimal           -> VAL_DEC60           (hot; packed word in nums)
  else (exponent, scale>22, mantissa>=2^53) -> VAL_EXT/NUMBER  (raw text)

FRACTIONAL (binary double from the struct constructor, no text):
                         -> VAL_EXT/DOUBLE      (bits in nums)
```

The integer/DEC60 classifier does no float arithmetic — it reads digits out of
the text. Integer text is stored by numeric value, so JSON's valid `-0` integer
form is emitted back as `0`. DEC60 is chosen only when the canonical
mantissa/scale → text reconstruction is byte-identical to the input; otherwise
the number falls through to NUMBER and is preserved byte-exact.

Allocation principle: mass values on the text path → a hot tag; rare values →
VAL_EXT. DOUBLE under VAL_EXT by default; promoting it to a hot tag is a
benchmark-decidable decision (justified only for double-heavy data). The
INT60/INT64 distinction is purely a tag-level classification since v4 (both
store a full 64-bit word in `nums`); the ladder is kept so the tag remains an
honest type witness for readers and statistics.

### VAL_DEC60 — packing and lossless bounds

The packed word (one `nums` entry, top four bits zero):

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
