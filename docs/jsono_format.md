# JSONO — binary format specification

JSONO stores JSON as a sequence of 64-bit slots plus two byte heaps, two
fixed-width payload streams, and a navigation block. There is no logical type
alias; physically a JSONO value is a nested DuckDB STRUCT with exactly one
layout field, `jsono`, wrapping the six-BLOB residual `body$1`:

```
STRUCT(
  jsono STRUCT(                -- the anchor: never revisioned
    "body$1" STRUCT(           -- residual column layout, revision 1
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
the `jsono` layout field carries the residual `body$1` plus a sibling
`shreds$2` STRUCT — a reserved shred-set marker followed by one typed *shred*
per chosen path
(produced by `jsono(value, shredding := spec)`).

```
STRUCT(
  jsono STRUCT(
    "body$1" STRUCT(slots BLOB, key_heap BLOB, string_heap BLOB, skips BLOB,
                    lengths BLOB, nums BLOB),
    "shreds$2" STRUCT(         -- shred layout, revision 2
      "$jsono$set" BIGINT,       -- shred-set marker (see below)
      "$jsono$spill$0" BIGINT,   -- spill bitmap column(s): "$jsono$spill$1", … for sets past 63 shreds
      "<lane1>" <type1>,         -- scalar shred: the bare typed lane
      "<lane2>" <type2>,
      "<arrlane>" <list_type>,  -- array shred: a bare LIST
      …
    )
  )
)
```

Shredding does **not** change the binary slot format. The six `body` blobs are
an ordinary JSONO value — the *residual* — so everything else in this spec
applies to them unchanged, including `version`. A plain (unshredded) value has
**no** `shreds` field (`STRUCT(jsono STRUCT("body$1" …))`); shredded-ness is
exactly the presence of `shreds$2`. Each scalar *shred* is the value at its
canonical path (`$.kind`, `$.commit.operation`, …) materialized as a plain typed
DuckDB column (`VARCHAR`, `BIGINT`, `UBIGINT`, `DOUBLE`, `BOOLEAN`). Nested shred
paths are allowed.

### Lane names

A lane's STRUCT field name is **not** its path as text. It is the path's
*structure*, serialized and encoded:

1. **Serialize the steps.** Every step of a lane path is an object key (a lane
   lifts keys, never array positions). Each key contributes its raw bytes with
   `00` escaped as `00 FF`, terminated by `00 00`; keys concatenate in path
   order.
2. **Encode.** Those bytes as unpadded base32hex over `0-9a-v` — RFC 4648's
   base32hex alphabet lowercased. The lower case is not cosmetic: it is what
   keeps a name out of the case-folding that this encoding exists to escape, so
   an upper-case digit is *rejected* on decode rather than folded.

| Path | Lane name |
|------|-----------|
| `gclid` | `cthmoqb40000` |
| `GCLID` | `8t1koia40000` |
| `$.URL.fragment` (two keys) | `al94o000cpp62prdcln78000` |
| `URL.fragment` (one key with a dot) | `al94obj6e9gmerb5dpq0000` |

The doubled terminator is load-bearing: it keeps `00` always in a *pair* (`00 FF`
= escaped NUL, `00 00` = end of key), so a decoder decides on the single byte
after a `00` and never depends on what the next key starts with. With a single
`00` terminator, `61 00 FF 62 00` would be both `[a, \xFFb]` and `[a\x00b]`.

Why the encoding exists: DuckDB compares STRUCT field names **case-insensitively**
over ASCII `A-Z`. A lane named by its path verbatim therefore collapses with
another spelling of the same key — two sources writing `gclid` and `GCLID` merge
into one lane, and the by-name cast then moves one key's values into the other's,
loud on the dropped spelling and silently wrong on the surviving one. The encoded
alphabet has no upper case at all, so distinct paths can never produce names that
differ only in case: the collapse is unrepresentable rather than guarded against.
The same bijection ends a second conflation — `gclid` and `$.gclid` are one path,
so they are one lane, and how the public spec DSL spells a path no longer reaches
the stored format. Declaring both spellings in one spec is refused.

Two consequences fall out. **No JSON key is reserved by the format**: a name
drawn from `0-9a-v` can be neither the layout's `body` nor anything under the
`$jsono$` prefix, so a key spelled like a layout field shreds like any other.
(The spec DSL has one reservation of its own, unrelated to the stored form: a
leading `$` starts a `$.`-rooted path, so a key literally named `$foo` is
declared as `$."$foo"`.) And **the encoding is order-preserving** at both stages,
so sorting lane names *is* sorting paths. A length-prefixed serialization would
break that (the length byte dominates), so this is the first invariant to
re-check if the serialization is ever revisited; the property is pinned by
`test_lane_name_codec` in `test/property/jsono_property.py`, which is also the
answer to "what depends on it" — every dependent at once, without a hand-written
list of them to fall out of date.

An object-array lane's element subfield names are encoded the same way (a
subfield is a one-step path). The price is that the type text is unreadable;
`jsono_layout_lanes(value)` answers "which paths are shredded here" in logical
form, and is the intended way to read a shredded schema.

A name is canonical when it is the one spelling its bytes have — `Encode(Decode(
name)) == name` — *and* the path it spells is one a writer could have produced.
The decoder enforces both: it refuses a bad character, an invalid length,
non-zero padding bits, a truncated escape or an upper-case digit, and it refuses
a key whose bytes are not valid UTF-8, since every lane path is a chain of JSON
keys and no writer can mint a key the parser would not accept. (An embedded NUL
*is* valid UTF-8 and stays supported.) A non-canonical name never decodes at all,
so such a struct is silently *not JSONO* to recognition, the way it treats every
unknown struct — though JSON reads and JSONO conversions of a fully-anchored type
carrying such a name are refused loudly (see [Layout revisions](#layout-revisions)).
`jsono_layout_diagnose(value)` explains which name failed and why.

The single layout name (`jsono` for both plain and shredded) and the nested
`shreds` struct are both deliberate. DuckDB reconciles struct types by field
name, so a set operation, CASE or COALESCE over differently-shredded values
merges them into one ordinary shredded type carrying the union of the shred sets
— the struct cast `NULL`-fills a branch's missing shreds, which is exactly the
writer's own "value stayed in the residual" encoding, so the merged value reads
losslessly by construction. The `$jsono$set` marker is the mandatory shared
member inside `shreds`, so DuckDB's by-name struct cast *binds* between any two
shred sets — fully disjoint ones included — and the extension optimizer can
rewrite every such cast into a lossless reconstruct + reshred. The nested struct
also keeps the marker and the shreds inside one field of the layout struct, so a
merge cannot interleave them with `body`. The remaining flip side — a raw
by-name cast silently *dropping* a shred when the optimizer is not running — is
caught at read time by the shred manifest (below), not by the type system.

Shreds are emitted in canonical order (sorted by name), so a constructed
type is a pure function of the shred *set*: `jsono_storage_type(<spec>)` and
the constructor produce the identical type for the same shreds regardless of the
order they are listed in, so a column declared from one accepts a value built
from the other.
Shred lanes stay bare scalar columns, so Parquet/DuckLake projection and
filter pushdown act on them directly.

**A merged type's field order is not canonical.** DuckDB merges two branches'
`shreds` structs by name, keeping the left branch's own field order and
appending the fields unique to the right branch at the **end**. When the two
branches straddle the 63-shred spill boundary the right branch alone carries a
`$jsono$spill$1`, so that reserved column lands *after* the left branch's
shreds: in a merged type the marker, the spill columns and the shreds are
neither contiguous nor in their canonical relative order. Recognition
(`MatchJsonoLayoutField`) and every accessor therefore locate all three **by
name**, never by field position — a positional rule silently misclassifies such
a union as a plain user struct, at which point `->>` reads `NULL`, `::json`
serializes the physical layout and only a `COPY … TO … (FORMAT CSV)` fails
loudly. Shred *k* is the *k*-th field that is neither the marker nor a spill
column, in the order the struct lists them; that ordering is what
`JsonoLayoutType::shreds` records, so parse and read agree by construction.

### Shred-set marker and spill bitmap

Two reserved fields let the optimizer decide, per shred, whether a bare
`struct_extract` of the lane is the full `->>` answer (a zone-map-pushed
columnar read) or whether it must fall back to / reconstruct from the residual.
All proofs are plain min/max questions, so they survive every statistics
backend alike — Parquet zone-maps, native storage, and DuckLake's global
min/max strings (which carry no value counts, the reason nothing here depends
on a null-count or has-valid-values signal):

- **`shreds.$jsono$set`** (`BIGINT`, one per value): the canonical layout hash of
  the shred *set* (paths + types) the row was written under — its `uint64` hash
  bits reinterpreted as a signed `BIGINT` — stamped **clean** (the hash itself)
  on a row whose spill bitmap is empty and **dirty** (the hash with its lowest
  bit flipped) on a row where at least one shred diverted. It is the *schema
  identity* across files, two-valued so the zone-map alone separates the cases:
  `min == max ==` the clean hash proves the scan all-clean under exactly the
  read shred set — every scalar shred reads bare at once; marker values within
  the `{clean, dirty}` pair prove the identity with some dirty rows, whose spill
  statistics then decide per shred (below). A multi-file read that unions
  narrower/retyped shred sets (`union_by_name`, DuckLake schema evolution)
  carries an unrelated hash per file, which fails both patterns and keeps the
  residual fallback — unless the per-lane no-`NULL` proof below applies — so a
  reader-`NULL`-filled lane is read from the residual, not as a silent `NULL`.
  The hash is **order-independent** (the marker identifies the set, not the
  field order, so a set-op/reorder cast that permutes the shreds still matches).
  It is `NULL` only on a SQL-NULL row. It is a signed `BIGINT`, not `UBIGINT`,
  because DuckDB's Parquet writer omits min/max stats for unsigned integers — a
  `UBIGINT` marker would carry no zone-map on a Parquet scan.
- **`shreds.$jsono$spill$0`** (`BIGINT` bitmap column(s), per row): bit
  `rank % 63` of column `rank / 63` is set when the scalar shred at canonical
  *rank* diverted on this row (the value stayed in the residual). Canonical rank
  is the shred's position in the byte-wise **sorted name list** of the shred
  set — not the physical field position, which a reorder cast may permute while
  the order-independent marker still matches. A set of N shreds carries
  `⌈N/63⌉` columns, named `$jsono$spill$0`, `$jsono$spill$1`, `$jsono$spill$2`, …
  (63 bits per column — the sign bit stays clear so masks order as non-negative
  numbers in every zone-map). **All columns are `NULL` on a clean row** (an
  all-zero mask is never stored) **and all non-`NULL` on a dirty one** (zeroes
  allowed per column), so min/max statistics describe the *dirty rows alone*:
  under set identity, a column whose `min == max == M` decodes exactly — every
  dirty row carries the same mask word, so a shred whose bit is clear in `M` is
  total. A systematic divert (one source consistently missing a lane type)
  thereby quarantines to its own shreds instead of demoting the whole
  row-group. Mixed dirty masks (`min != max`) keep a one-sided bound: masks are
  non-negative, so a bit set anywhere forces `max >= 2^bit` — every bit with
  `2^bit > max` is still provably clean. An out-of-domain word (negative, or
  bits at/above the shred count) marks a hand-built row and is not trusted at
  all. Array shreds get ranks but never set bits (they are always
  reconstructed, never bare-read).
- **Per-lane no-`NULL` proof**, independent of identity: a lane whose statistics
  prove no `NULL` at all has nothing hiding behind it — no divert, no absent
  path, and no reconciliation `NULL`-fill — so it reads bare even when the
  marker cannot prove identity (`union_by_name` / DuckLake-evolved files). A
  reconciliation that `NULL`-fills a lane on rows whose writing shred set lacked
  it breaks the proof on exactly those scans, keeping the fallback correct.

Interpretation of the spill bitmap is **identity-gated**: a raw by-name cast
copies the bitmap across shred sets, where the foreign marker hash fails the
identity patterns and the foreign bit numbering is conservatively ignored. A
set-op merged type carries the union of both branches' shreds but only the
by-name union of their spill columns, which can under-provision a crossing
union's set (`⌈N/63⌉` columns needed, fewer present): such a type stays fully
readable — bits for ranks beyond its columns simply do not exist, so those
shreds are never proven total and every read keeps the residual `COALESCE`.

The spill bit by case, for a scalar shred at path `P`:

- absent path / explicit JSON `null`: lane `NULL`, `->>` = `NULL` → no bit (a
  bare `NULL` read is correct).
- a scalar that round-trips through the shred type: lane set → no bit.
- a divert (a string at a `BIGINT` lane, a number at a `VARCHAR` lane, …) or a
  container at `P`: lane `NULL`, the value stays in the residual → bit set, so
  the read falls back to the residual / reconstructs.

**Single-storage invariant: every value is stored exactly once, and a set lane
means the path is stripped from the residual.** Every shred path is a non-empty
object-key chain — array-index and root `$` paths are rejected at bind and are
not recognized by the layout parser, so a lane never shadows a value the residual
also keeps. A value is removed from the residual and kept only in its shred when
it round-trips losslessly through the shred type (a JSON string in a `VARCHAR`
shred, an integer in a `BIGINT` shred). A value that does not fit — a string in a
`BIGINT` shred, a number in a `VARCHAR` shred, a container, or a number whose
shred type would re-encode it differently — stays in the residual and leaves the
lane `NULL`. An explicit JSON `null` or a missing key also leaves the lane `NULL`
(and stores nothing extra). A whole regular array is a separate case — see
[Array shreds](#array-shreds-liststruct), which lift element subfields (object
arrays) or whole scalar elements
([scalar arrays](#scalar-array-shreds-listtype)) while leaving a skeleton array in
the residual.

The original value is recovered by overlaying the shreds onto the residual,
**residual-authoritative**: a path absent from the residual is filled from its
shred, and a path present in the residual wins (a divert's lane is `NULL`, so the
overlay never faces a live conflict on writer-produced data). Since the removed
values are exactly those their shred reproduces byte-for-byte, the overlay
reconstructs the value losslessly.

### Shred manifest

The `skips` blob of a shredded residual ends with a *shred manifest*: the
write-time record of which paths were stripped out of *this row's* residual,
with the shred type each was written as. After the checkpoint sections (see
[Navigation](#navigation-skips)) the tail is:

```
u32 entry_count
per entry: u16 path_len, path bytes, u8 type_code
           [if type_code = 11: u16 subfield_count,
              per subfield: u16 key_len, key bytes, u8 type_code]
```

Codes 1–10 are the scalar shred types and their scalar-array forms (`VARCHAR`,
`BIGINT`, `UBIGINT`, `DOUBLE`, `BOOLEAN`, then `VARCHAR[]` … `BOOLEAN[]`). Code
11 is an object-array lane, and it is the only one whose entry continues: its
element subfields follow as their own list, keys ascending, each with the code of
its scalar type. The grammar admits no other shred type, so there is no escape
code — a type without one is an internal error, not a string to fall back on.

Spelling an object array's subfields out is what makes a *widened* element struct
readable. The entry is the write-time claim "these subfields were stripped from
this row"; a subfield the reading type carries beyond them is a column of NULLs
whose values are still in the residual skeleton, which is where every reader
already looks when a lane is NULL. Rendering the element struct as one type
string instead made "my subfield was dropped" and "another file's subfield was
added beside mine" the same inequality, so a reader had to refuse both — and
`read_parquet(…, union_by_name := true)` over files whose elements carry
different subfields unions those structs, so such a scan was unreadable.

An entry's path is the lane's **logical** path in the project's one text form
(`$.`-always, quoted per key where a bare step would mis-parse) — never the
lane's encoded field name. The manifest is a per-row statement about the
*document*, while the encoding is a transport artifact of a DuckDB limitation at
the *type* level; keeping the two apart is what lets the naming codec change
without rewriting a single stored row. Readers decode their own type's lane names
once per reader init and compare bytes from there. For the same reason an
object-array entry's subfield list stores the elements' JSON keys, not their
encoded names.

Entries are self-describing — each carries its path and type inline — so a reader
needs no external shred list to decode them. (Earlier revisions also
defined layout-hash-keyed `indexed` and `bitset` tails that referenced the
reader's own shred list; they were removed because they never beat this
entry-list form on disk — their per-row bit-packing trades zstd-friendly
repetition for high entropy, so Parquet + zstd compress the entry list smaller.)

Entries are sorted by logical path, which is **not** the type's field order
(that follows the encoded name, and the two genuinely differ: `$.a-c` sorts before
`$.a.b` as text while the nested path sorts first structurally), and list only the paths
actually stripped from this row (a value kept in the residual by the lossless
gate is not listed). A plain value writes no manifest — its `skips` blob ends at
the checkpoints, which reads as zero entries.

The manifest is what makes a raw by-name struct cast safe to detect: if a cast
drops a shred (the target type carries fewer shreds) or converts one to a
different type, the residual cannot reproduce the value — and the manifest
proves it. Every reader of a shredded residual verifies the row's manifest
against the shreds actually available and fails loud on a mismatch instead of
silently returning partial data. Extra shreds beyond the manifest are legal (a
widening cast `NULL`-fills them; readers fall back to the residual).

### Array shreds (`LIST<STRUCT>`)

A shred type may be a `LIST<STRUCT<…>>` whose every struct child is one of the
five scalar shred types. Such an *array shred* lifts the chosen leaf subfields of
every element of a regular array — addressed by an object-key path like
`$.products`, not `$.products[*].name` — into one parallel typed `LIST<STRUCT>`
column (`jsono(value, shredding :=
'{"$.products": "STRUCT(name VARCHAR, id UBIGINT, …)[]"}')`). On shape-stable data this is a large storage win: Parquet
stores each element subfield as its own low-cardinality dictionary column,
where the whole-document residual would store them in one mixed heap.

The array is **not** removed from the residual. It stays in place as a
*skeleton array*: every element keeps its tail (the keys not in the shred
schema) verbatim, but the lifted subfields are stripped per element (the same
per-scalar lossless gate as object-key shreds, applied to each element). The
skeleton carries everything reconstruction needs with no new format primitive:

- the **array length** is the skeleton's element count (authoritative — a
  non-`NULL` shred list of a different length is a corrupt row and fails loud);
- the **element order** is the skeleton's order, in lockstep with the shred
  `LIST`;
- a **non-object / null / scalar element** stays verbatim in the skeleton, its
  shred-`LIST` slot a `NULL` struct, so heterogeneous arrays round-trip;
- the three-way distinction per subfield — *lifted* (absent from the skeleton,
  present in the shred), *explicit JSON null* (a `VAL_NULL` in the skeleton, never
  lifted), and *absent* (absent from both) — falls out of the same
  residual-authoritative overlay, with no extra bits.

The shred `LIST` has one struct per skeleton element (`NULL`-struct for the
non-object ones), so its length equals the skeleton array length. Reconstruction
walks the two in lockstep: each object element's present subfields are overlaid
back onto its skeleton tail (residual-authoritative — the tail and any kept
explicit-null subfield win), and every other element is emitted verbatim.

The manifest records the array's object-key path when at least one element's
subfield was lifted, exactly as for a scalar shred — so a raw cast that drops or
retypes the array shred is caught the same way (the skeleton elements are
missing their lifted subfields and the residual alone cannot reproduce them).
An object-array entry is the one kind that spells its element subfields out
rather than naming a type; the entry's exact framing is in
[Shred manifest](#shred-manifest), which is where it is specified.

A skeleton residual is an ordinary value, and the shred `LIST<STRUCT>` is a
DuckDB/Parquet column inside the `shreds` struct rather than part of the blob.
An old reader lacking array-shred support rejects the unknown `LIST<STRUCT>`
shred field at bind or fails loud on the manifest rather than silently dropping
data.

### Scalar array shreds (`LIST<TYPE>`)

A shred type may be a `LIST<TYPE>` whose element `TYPE` is one of the five scalar
shred types (`LIST<UBIGINT>`, `LIST<VARCHAR>`, `LIST<DOUBLE>`, `LIST<BIGINT>`,
`LIST<BOOLEAN>`). Such a *scalar array shred* lifts each whole scalar element of a
regular array — addressed by an object-key path like `$.item_ids`, not
`$.item_ids[*]` — into one parallel typed `LIST<TYPE>` column
(`jsono(value, shredding := '{"$.item_ids": "UBIGINT[]"}')`, or the `jsono(STRUCT)`
auto-shred of a top-level `LIST<scalar>` field). It is the strictly simpler case of
the object array above: the element is a scalar, not a struct, so there are no
per-element subfields to overlay — each element is either *lifted* whole or *kept*
whole.

Like the object array, the array is **not** removed from the residual: it stays as
a *skeleton array* the same length as the source, carrying length, element order
and every non-lifted element verbatim. An element is lifted only if it round-trips
losslessly through the lane type — the same per-scalar
[lossless gate](#shredded-layout) as an object-key shred, applied to each element —
so a string in a `UBIGINT[]` lane, an explicit JSON `null`, a number in a
`VARCHAR[]` lane, an object or a nested array all stay verbatim, element-wise.

Where an object array leaves the element object in place minus its lifted subfields,
a scalar array has no sub-structure to keep, so a lifted element leaves a **`VAL_NULL`
placeholder** in the skeleton (one slot, no stream entry — the value's bytes move
into the typed `LIST` column, which is the storage win). The parallel shred `LIST`
has exactly one slot per skeleton element (its length equals the skeleton array
length, authoritative — a different length is a corrupt row and fails loud), and the
slot's *validity* is what disambiguates the two element states on reconstruction:

- a **non-`NULL` shred slot** is a lifted element — its `VAL_NULL` placeholder in the
  skeleton is ignored and the typed value is emitted from the shred;
- a **`NULL` shred slot** is a kept element — the skeleton value (which may itself be
  an explicit JSON `null`, a non-conforming scalar, an object or an array) is emitted
  verbatim.

So a lifted element's placeholder `VAL_NULL` and an explicit JSON `null` element are
never confused: the former always pairs with a non-`NULL` shred slot, the latter
always with a `NULL` one. A `VARCHAR[]` lane lifts (and so marks its slot
non-`NULL`) only a real JSON string, like every lane — keeping that invariant exact.

The manifest records the array's object-key path when at least one element was
lifted, exactly as for the other shreds, so a raw cast that drops or retypes the
scalar array shred is caught the same way (the placeholders are missing their values
and the residual alone cannot reproduce them); a scalar-array lane carries its own
type code, like every non-object-array lane. `VAL_NULL` is an existing slot tag, and
the shred `LIST<TYPE>` is a DuckDB/Parquet column inside the `shreds` struct rather
than part of the blob.

## Layout revisions

Three things are versioned independently, because they fail in different ways
and invalidate different data:

| What | Where the version lives | How to read the current one |
|------|-------------------------|-----------------------------|
| The bytes **inside** the body blobs | `version` byte in `slots` | `jsono_version()`, and [Compatibility policy](#compatibility-policy) |
| The residual's **column** layout (the set, names and types of the body blobs) | the field name `body$<N>` | the value's own type |
| The **shred** layout (reserved fields inside `shreds`, lane naming, lane shape, spill bit numbering, marker semantics) | the field name `shreds$<M>` | the value's own type |

The current numbers are deliberately not restated here. Each already has one
authority that a reader can query — the version byte through `jsono_version()`,
the two layout revisions through the field names in any type printout — and a
second copy of a number that changes is a copy that eventually disagrees with
the first. Where a revision bump happened and what the closed form looked like
is in [Revision history](#revision-history).

The layout field name `jsono` is the **anchor** and never carries a revision.
Recognition is anchor-first: a top-level STRUCT with exactly one field named
`jsono` whose own field 0 is named `body` or `body$<digits>` **and is a STRUCT of
nothing but BLOBs** is a JSONO value of *some* revision. (The blob requirement is
revision-neutral — every revision's residual is blob columns — and is what keeps
an ordinary user struct that happens to spell `{'jsono': {'body': …}}` from being
refused as JSONO data.) A value whose revision is **known but different** is
refused loudly (`JsonoRejectForeignLayout`), never silently treated as an
ordinary struct — that silence was the actual failure mode before revisions
existed: an older shredded value bound to core `json`'s `->>`/`to_json`, which
returned `NULL` or serialized the raw blob struct.

A layout struct carrying **more than one** revisioned stem of a kind (`body` and
`body$1` side by side, say) is a *mixture* of revisions and is refused the same
way. That is what a multi-file scan produces: `read_parquet(…, union_by_name :=
true)` merges the per-file schemas by name, so an old file beside a current one
yields `body$1, shreds$2, body, shreds` in whichever order the files were
listed. Nothing can be done to such a scan as a whole (there is no single
revision to upgrade from) — upgrade the old files separately and `UNION ALL` the
results. The refusal names a *foreign* revision rather than whichever sorted
first, so it reads the same in both orders.

A value carrying the *current* revision that nevertheless fails the grammar
stays non-JSONO in *recognition* — recognition runs over every struct in every
plan and never throws. This is deliberate: such a type also arises on a legal
write path, where a generic value→SQL→value round-trip (DuckLake's inlined-data
flush) narrows an all-NULL spill column to `SQLNULL` and a small lane to
`INTEGER` on its way to the declared column type. Refusing that would break
writing in order to catch a hand-built struct. The near-misses split into two
classes from there:

- a **type**-level miss (the narrowing class above) stays fully silent — the
  cost is bounded: a column *declared* with a near-miss shred lane (`a INTEGER`,
  which the constructor itself rejects as a shred type) accepts a valid
  current-revision value silently, and every read of that column then answers
  `NULL`; the bytes are intact and one explicit cast to the correct lane type
  (`a BIGINT`) brings the whole document back. Declare storage columns with
  `jsono_storage_type(...)` and this cannot happen;
- a **name**-level miss — a lane name (or an array lane's element subfield
  name) that does not decode, inside a fully anchored type (exact residual,
  `shreds$2`, marker and spill present) — is a shape no legal write narrows
  into: generic round-trips change types and keep names. The optimizer
  therefore *refuses the JSON reads* (`->>`, `->`, `to_json`, `::JSON`) of
  such a type with the grammar's own reason instead of letting the whole
  column silently answer `NULL`, and the constructor and cast binds refuse
  the JSONO conversions (`jsono()`, a cast or `INSERT` into a JSONO column)
  that would grind the document into a rebuild of its raw layout fields.
  Passthrough (`SELECT *`, `COPY`), the raw `::VARCHAR` render and DDL stay
  open, so the bytes can be moved and the offending field dropped
  (`test/sql/jsono_malformed_lane_read.test` pins both halves).

Splitting `body$N` from `shreds$M` is what keeps a shred-layout change from
invalidating plain values, which are the bulk of stored data.

**What a foreign value can still do.** Interpretation fails; *binary* transport
lives. Moving the bytes works — a bare column reference in `SELECT *`, `INSERT …
SELECT`, `CREATE TABLE … AS`, `UNION ALL`, `GROUP BY`/`ORDER BY`, `COPY … TO`
Parquet, `EXPORT DATABASE (FORMAT PARQUET)` — so an old value can always be
carried to a build that reads it. Everything that reads or *renders* the value
refuses, including rendering to text: `->>`, `to_json`, `::JSON`, `::VARCHAR`,
every `jsono_*` function, and also `COPY … TO … (FORMAT CSV)`, `EXPORT DATABASE`
with its default CSV format, and simply printing the column in a shell — those
all go through the same VARCHAR cast, which would otherwise write an
unrecoverable dump of the raw blobs with no error at all. Composition refuses
too: `CASE`, `COALESCE`, `IS NOT NULL`, `=`, window functions, wrapping the value
in a struct or list.

Two boundaries are known and not caught:

- a foreign JSONO value **nested inside another struct** (`STRUCT(payload
  <foreign>, id INT)`) is invisible to the plan walk, which inspects the type of
  an expression's child, not the types nested inside it. `to_json(x)` on the
  outer struct serializes the raw blobs; `to_json(x.payload)` refuses. Tightening
  this would refuse `x.id` as well — reading a sibling field is not an
  interpretation of the JSONO value — so the walk stays at the top level;
- under `SET disabled_optimizers='extension'` the plan walk does not run at all,
  and the core-json paths (`->>`, `to_json`, `::JSON`) go quiet again. The casts
  and our own binders still refuse, since they do not depend on the optimizer.

**Bump `body$N`** when a body blob column is added, removed, renamed or
retyped. **Bump `shreds$M`** when the reserved field set inside `shreds`, the
lane naming (the path → field-name codec), the lane shape, the canonical rank
numbering, the marker semantics or the spill bit encoding change — including purely *semantic* changes the type cannot show.
Neither is bumped by the user's shred set or by the ⌈N/63⌉ spill column count:
those are data under a fixed layout.

Closing a revision is one commit with three parts:

1. bump the name (`JSONO_BODY_REVISION` / `JSONO_SHREDS_REVISION` in
   `src/include/jsono.hpp`);
2. add a closed-revision fixture — a struct literal over a live body, as in
   `test/sql/jsono_layout_revision.test` — asserting the loud refusal. Do **not**
   carry the golden bytes into it: a foreign revision is decided by the field
   *name* before a single blob is read, so bytes in a refusal fixture assert
   nothing;
3. add one row to the revision map below: the commit range that wrote the closed
   shape, and how it differs from the new current one. Not its full layout — the
   build that wrote it, and the golden bytes it wrote, are in git, and a second
   copy here can only drift from them. What a user holding old files needs is the
   way back: the commit range, plus the statement that moves the data forward
   whenever a rebuild of the layout struct is enough.

### Revision history

There is no compat reader and none is planned: this build refuses old data, it
does not upgrade it. So what follows is not an archive of closed layouts — those
live in git, in the build that wrote them and in the golden bytes of
`test/sql/jsono_layout_golden.test` at that commit. It is the way back to your
bytes: which commit range wrote which shape, and what it takes to move it
forward.

**Shreds revision 1** (`bf99fb2`…`172d892`) named a lane by its **path spelled as
text** — `gclid`, `$.commit.operation` — rather than by the encoding of the path.
Nothing else about it differs from revision 2: same reserved fields, same lane
shape, same spill numbering, same marker. That naming is the defect it was closed
for: DuckDB matches STRUCT field names case-insensitively, so two spellings of one
JSON key (`gclid` / `GCLID`) collapsed into one lane on a type merge and the
by-name cast then misfiled one key's values into the other's. The manifest of such
a value additionally stored the lane's *physical* name, so `gclid` appears there
where revision 2 writes `$.gclid`.

Renaming the field to `shreds$2` is **not** an upgrade: every lane inside it must
be re-spelled as the encoded path (see [Lane names](#lane-names)) and every row's
manifest rewritten. Both are per-row work no rebuild of the layout struct can do,
so a shredded revision-1 value is moved forward by reading it with a build from
that range and re-ingesting through `jsono(value, shredding := …)`. A **plain**
revision-1 value is unaffected — that is why the two stems version separately.

The rename is worse than useless, and it is worth knowing exactly how. Before it,
the value is refused loudly on every read and render, including the cast that
stands in front of `COPY … TO … (FORMAT CSV)`. After it, the revision check
passes and the lane names do not, so the value is not JSONO at all — and *that*
verdict is silent, which means the CSV dump of raw blobs is written with no error.
Recognition cannot close this: the type carries nothing that separates a renamed
old value from any other hand-built struct with a non-lane field, and those must
stay silent (a DuckLake inlined-data flush produces them on a legal write path).
So the rename is not a step of the upgrade — it is the one move that turns a loud
refusal into a quiet one. `jsono_layout_diagnose` is what tells you which side of
it you are on.

**Revision 0** is everything written before layout revisions existed: the
unrevisioned field names `body` and `shreds`. Four shapes shipped under it.

| Shape | Written by | Difference from revision 1 |
|-------|------------|----------------------------|
| 0.d | `cc7d8cb`…`31ba960` | the two field names, nothing else |
| 0.c | `f450a5b`…`cc7d8cb`~ | …and spill column 0 spelled `"$jsono$spill"`, the rest `"$jsono$spill1"`, `"$jsono$spill2"`, … |
| 0.b | `ada4d30`…`f450a5b`~ | …and exactly one spill column, `"$jsono$spill"`, capping a shred set at 63 |
| 0.a | before `ada4d30` | scalar shreds were `STRUCT(value <T>, complete TINYINT)` pairs, no spill bitmap, and the marker was the clean hash on every row with no dirty flip |

Stored files overwhelmingly carry 0.d: it lived from `cc7d8cb` until revisions
landed in `bf99fb2`.

**0.d, 0.c and 0.b upgrade in place, in SQL — plain values only.** Nothing inside
the blobs changed across them, so for a plain value the whole migration is a
rebuild of the layout struct under the current field names:

```sql
-- The extension optimizer refuses struct_extract over a foreign value — which is
-- exactly the read this rebuild needs — so it is off for this one statement.
SET disabled_optimizers='extension';
CREATE TABLE upgraded AS
SELECT {'jsono': {'body$1': v.jsono.body}} AS v
FROM old;
RESET disabled_optimizers;
```

A **shredded** revision-0 value carries text lane names and physical-name manifest
entries, so it needs the same per-row work as shreds revision 1 above: read it
with a build from its range and re-ingest. (Renaming the struct fields alone
leaves a lane set the grammar does not accept, and that verdict is *silent*: the
value is simply not JSONO, so an extract reads NULL and `to_json` serializes the
physical struct. Nothing announces the half-migration — check the result with
`jsono_layout_diagnose`, not by watching for an error.) Read the result back with
the optimizer on: a value that reads is a value that upgraded.

**0.a needs the build that wrote it.** Its per-row divert information lived in
the `complete` flags, which the current lane shape does not have, and the
conversion (lane = the pair's `value`, spill bit `r` = `NOT complete`, marker
flipped when the resulting mask is non-zero) is not expressible as a rename.
Rather than reimplement it here, read the data with the build that understands
it — `git checkout ada4d30~`, build, `COPY … TO 'x.json'` — and re-ingest with
`jsono(...)`. That old build is the compat reader, which is the other half of
why closed layouts are not restated in this document.

## Compatibility policy

The on-disk binary format is versioned by the `version` byte in `slots`.
Current `JSONO` files use `version = 5`:

- version 2 added the per-object `shape_hash` field to `ContainerSpan` (see
  [Navigation](#navigation-skips));
- version 3 added the shred manifest tail to `skips` (see
  [Shred manifest](#shred-manifest));
- version 4 moved the variable slot payloads into the `lengths`/`nums` streams
  (slots are pure tags), made `ContainerSpan`/`ObjectCursorCheckpoint` carry
  per-stream counts/deltas, and made span storage *sparse*: non-empty arrays
  and objects with `child_count > OBJECT_CHECKPOINT_STRIDE` store a span,
  addressed through a sorted sparse container-id index;
- version 5 made the shred manifest one framing instead of two (the full and
  compact forms collapsed into the type-code form, and its `0xffffffff` marker
  is gone), and gave an object-array lane its own type code whose entry spells
  the element subfields out one by one instead of rendering the element struct
  as a type string.

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
| `version` | u8 | the current format version — see [Compatibility policy](#compatibility-policy) |
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

Spans are required for containers whose skip cannot be done cheaply from the
slot stream alone:

- every **non-empty array** — no inline child count, so a walk would not know
  where to stop without replaying nested structure; an empty array is the
  adjacent `ARR_START, ARR_END` pair and needs no span;
- every **object with `child_count > OBJECT_CHECKPOINT_STRIDE`** (= 16) —
  exactly the objects that also get cursor checkpoints.

Small objects (`child_count <= OBJECT_CHECKPOINT_STRIDE`) may carry an optional
span when their subtree is no longer cheap to replay. The DOM writer stores one
when the object has a container child and its total `slot_span` exceeds
`OBJECT_CHECKPOINT_STRIDE`; tiny skeleton objects stay spanless. This keeps the
dominant per-container metadata cost out of leaf/skeleton objects, while late-key
navigation can still jump past compact-key, large-subtree values in O(1). Readers
accept and validate optional spans for small objects with container children and
for empty arrays.

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

`stride` is a single fixed constant `OBJECT_CHECKPOINT_STRIDE = 16` for every
checkpointed object. Each object records its own stride in `ObjectCheckpointIndex`,
so a future variable-density scheme can be introduced writer-side without a format
change.

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
