#pragma once

#include "jsono.hpp"
#include "jsono_locate.hpp"
#include "jsono_path.hpp"
#include "jsono_reader.hpp"
#include "jsono_render.hpp"

#include "duckdb/common/types/vector.hpp"

#include "string_view.hpp"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {
namespace jsono {

// Outcome of reading one row through JsonoRowReader. Null is an absent value (a SQL NULL row);
// Empty is a present row whose slots blob holds no value (readers map it to SQL NULL too, except
// the shredded readers that still emit shred-only output); Value hands out a parsed view.
enum class JsonoRowState : uint8_t { Null, Empty, Value };

// The (logical path, type-string) pairs of a reading type's shreds. Building them decodes every lane
// name out of its base32 form and renders every lane type — O(lanes) with an allocation per lane —
// which must not repeat on each chunk: a wide shredded merge inits a reader per input per chunk, so
// that build lands on every input of every chunk and is measurable against the whole operator. An
// operator therefore keeps one of these per input in its local state, across chunks.
//
// Identity is the LogicalType itself, compared by value (for a STRUCT the fast case is still one
// pointer compare on the shared type info). Keying on the ExtraTypeInfo ADDRESS was wrong for every
// type whose aux info is null — SQLNULL, VARCHAR, any scalar — where two different such types
// compared equal; that answered correctly only through a second file's reasoning (no aux ⇒ not
// shredded ⇒ both signature sets empty), which a new caller had no way to know it depended on.
//
// A miss builds a NEW vector and hands out shared ownership: whatever a verifier captured from a
// previous answer stays alive and immutable for as long as it is held, so re-resolving an input can
// never swap the lanes out from under an un-reset memo.
class JsonoShredSignatures {
public:
	const shared_ptr<const std::vector<JsonoShredSignature>> &For(const LogicalType &type) {
		if (!signatures_ || type != cached_type_) {
			auto built = make_shared_ptr<std::vector<JsonoShredSignature>>();
			JsonoBuildShredSignatures(type, *built);
			signatures_ = std::move(built);
			cached_type_ = type;
		}
		return signatures_;
	}

private:
	shared_ptr<const std::vector<JsonoShredSignature>> signatures_;
	LogicalType cached_type_;
};

// Memoized shred-manifest verification (the check itself is VerifyShredManifestEntries).
// Vectors are overwhelmingly manifest-homogeneous — the rows of one column chunk were shredded
// with the same spec — so after the first row a verification is one byte-compare of the row's
// manifest tail against the last verified tail.
class ShredManifestVerifier {
public:
	// Signatures = the shreds the reading type carries, as (logical path, type-string) pairs. The
	// manifest records a lane's path, not its encoded field name, so the names are decoded here —
	// once per reader init, leaving VerifyShredManifestEntries a byte comparison and the tail
	// memoization untouched. Operators on a hot path pass a JsonoShredSignatures instead, so the
	// decode happens once rather than per chunk.
	void InitFromType(const LogicalType &type) {
		ResetMemo();
		external_.reset();
		JsonoBuildShredSignatures(type, owned_);
		order_ = BuildShredSignatureOrder(owned_);
	}

	// Signatures shared with a longer-lived cache, HELD: JsonoShredSignatures::For allocates a new
	// vector on every miss, so what this verifier captured stays alive and immutable however the
	// cache moves on — there is no window where a re-resolved input silently swaps the lanes under
	// this verifier's memo.
	void InitSignaturesRef(shared_ptr<const std::vector<JsonoShredSignature>> signatures) {
		ResetMemo();
		owned_.clear();
		external_ = std::move(signatures);
		order_ = BuildShredSignatureOrder(*external_);
	}

	// Signatures supplied by the caller (__jsono_internal_checked_residual receives them as plan
	// constants); the paths must be the same canonical logical form InitFromType decodes to.
	void InitSignatures(std::vector<JsonoShredSignature> signatures) {
		ResetMemo();
		external_.reset();
		owned_ = std::move(signatures);
		order_ = BuildShredSignatureOrder(owned_);
	}

	// The hot path is a row without a manifest (a plain residual, the overwhelming majority):
	// one offset compare and return, inlined into the caller's row loop so it pays no call. The
	// manifest parse/verify is out-of-line so it does not bloat that loop.
	JSONO_ALWAYS_INLINE void Verify(const JsonoView &view) {
		if (!view.HasShredManifest()) {
			return;
		}
		auto tail = view.ManifestTail();
		if (verified_ && TailMatches(tail)) {
			return;
		}
		VerifyManifested(tail);
	}

	// The manifest entries of the row `Verify` just accepted, as views into verifier-owned
	// bytes (stable until a row with a different manifest is verified). Empty for a row
	// without a manifest.
	const std::vector<ShredManifestEntry> &VerifiedEntries(const JsonoView &view) {
		if (!view.HasShredManifest()) {
			static const std::vector<ShredManifestEntry> no_entries;
			return no_entries;
		}
		return entries_;
	}

private:
	// Every Init begins here, so "initialized" means the same thing however it was reached. The
	// memoization keys on the manifest tail ALONE — two rows with byte-equal tails skip re-verifying —
	// which is sound only within one set of signatures. Leaving it across an Init would bless a row of
	// the new input against the lanes of the old one, silently, in the mechanism that exists to fail
	// loud.
	void ResetMemo() {
		tail_.clear();
		entries_.clear();
		verified_ = false;
	}

	// The signatures in force: an external cache when one was supplied, this reader's own otherwise.
	// Held as "external or own" rather than as one pointer that may aim at a member, so copying or
	// moving a reader cannot leave it verifying against the source's signatures. Before any Init the
	// own set is empty, which is the strictest reading: every manifest entry fails to match.
	const std::vector<JsonoShredSignature> &Signatures() const {
		return external_ ? *external_ : owned_;
	}

	void VerifyManifested(nonstd::string_view tail) {
		verified_ = false;
		tail_.assign(tail.data(), tail.size());
		ParseShredManifestBytes(tail_.data(), tail_.size(), entries_);
		VerifyShredManifestEntries(entries_, Signatures(), order_);
		verified_ = true;
	}

	bool TailMatches(nonstd::string_view tail) const {
		return tail.size() == tail_.size() && std::memcmp(tail.data(), tail_.data(), tail.size()) == 0;
	}

	shared_ptr<const std::vector<JsonoShredSignature>> external_;
	std::vector<JsonoShredSignature> owned_;
	std::vector<uint32_t> order_;
	std::string tail_;
	std::vector<ShredManifestEntry> entries_;
	bool verified_ = false;
};

// Manifest check for a point read over a residual that carries a shred manifest. `found_container`
// = false: the read path was not found — an exact manifest match (or a manifest leaf inside the
// missing subtree) means the value was stripped into a shred this reading context does not have,
// so fail loud instead of returning a silent NULL. `found_container` = true: the read returns a
// whole found container — a manifest leaf strictly inside it means the result would be missing
// that leaf. An honest shredded read never reaches either case (the optimizer reads shreds
// directly and reconstructs subtree reads that overlap a shred), so a hit here is always a row
// narrowed by a raw struct cast.
inline void ThrowIfManifestCoversPath(const JsonoView &view, const vector<PathStep> &read_steps, bool found_container,
                                      std::vector<ShredManifestEntry> &manifest_scratch,
                                      vector<PathStep> &steps_scratch) {
	view.ReadShredManifest(manifest_scratch);
	for (auto &entry : manifest_scratch) {
		// A manifest path is the lane's logical path in the project's one text form (`$.`-always), so
		// it parses back to steps with the shared grammar — keeping the manifest logical is what buys
		// this path its freedom from lane-name decoding.
		auto path = string(entry.path);
		steps_scratch = ParseJsonoPath(path, "jsono shred manifest");
		// A manifest path is an object-key chain by the shred writer's invariant. It covers the
		// read when it equals the read path (the value itself was stripped) or extends it (a
		// stripped leaf inside the read subtree). For a found scalar neither can hold; for a
		// miss both can, for a found container only the strict extension can.
		if (steps_scratch.size() < read_steps.size()) {
			continue;
		}
		if (found_container && steps_scratch.size() == read_steps.size()) {
			continue;
		}
		bool covers = true;
		for (idx_t i = 0; i < read_steps.size(); i++) {
			auto &m = steps_scratch[i];
			auto &e = read_steps[i];
			if (m.kind != e.kind || m.key != e.key || m.index != e.index) {
				covers = false;
				break;
			}
		}
		if (covers) {
			throw InvalidInputException(
			    "JSONO: path '%s' was shredded into a shred this value no longer carries (the row was narrowed "
			    "by a raw struct cast) and cannot be read losslessly",
			    path.c_str());
		}
	}
}

inline void ThrowManifestCoversPath(nonstd::string_view manifest_path) {
	throw InvalidInputException(
	    "JSONO: path '%s' was shredded into a shred this value no longer carries (the row was narrowed "
	    "by a raw struct cast) and cannot be read losslessly",
	    std::string(manifest_path).c_str());
}

inline void ThrowIfManifestCoversPathText(const JsonoView &view, const std::string &read_text, bool found_container) {
	struct CoverSink {
		const std::string &read_text;
		bool found_container;
		void OnEntry(const ShredManifestEntry &entry) {
			if (entry.path.size() < read_text.size()) {
				return;
			}
			if (entry.path.size() == read_text.size()) {
				if (!found_container && entry.path == nonstd::string_view(read_text.data(), read_text.size())) {
					ThrowManifestCoversPath(entry.path);
				}
				return;
			}
			if (std::memcmp(entry.path.data(), read_text.data(), read_text.size()) == 0 &&
			    entry.path[read_text.size()] == '.') {
				ThrowManifestCoversPath(entry.path);
			}
		}
	};
	auto tail = view.ManifestTail();
	CoverSink sink {read_text, found_container};
	WalkShredManifestBytes(tail.data(), tail.size(), sink);
}

// How a reader treats a row's shred manifest. One value per meaningful state, and every policy
// consumer switches over this enum, so a fourth policy fails compilation (-Werror=switch) at each
// place that must decide for it instead of inheriting a leftover default.
enum class ReadPolicy : uint8_t {
	// Verify every manifested row whole-document (the default): each manifest entry must name a
	// shred the input's type carries.
	WholeDocument,
	// No whole-document verify — the read checks only the two outcomes where a stripped value would
	// silently change ITS result (CheckPathMiss / CheckContainerRead).
	PointRead,
	// No verification at all (jsono_overlay's fold: its residual was verified upstream).
	Trusted,
};

// The row-read layer: every operator that decodes JSONO values reads rows through this object,
// which bundles the strict row read, the header parse and the shred-manifest check — the
// "every reader verifies the manifest" invariant (docs/jsono_format.md) holds by construction
// instead of per call site. Operators that only move blobs verbatim (identity copies, storage
// introspection) stay on ReadJsonoRowStrict: the manifest travels with the bytes, so a move is
// not a lossy read.
class JsonoRowReader {
public:
	// ReadPolicy::WholeDocument (a plain type carries no shreds, so any manifest fails loud as a
	// narrowed row).
	void Init(Vector &input, idx_t count) {
		Reset(ReadPolicy::WholeDocument);
		InitJsonoVectorData(input, count, data_);
		verifier_.InitFromType(input.GetType());
	}

	// Same policy, but with the signatures kept in an operator-lifetime cache so the per-lane decode
	// happens once instead of on every chunk.
	void Init(Vector &input, idx_t count, JsonoShredSignatures &cache) {
		Reset(ReadPolicy::WholeDocument);
		InitJsonoVectorData(input, count, data_);
		verifier_.InitSignaturesRef(cache.For(input.GetType()));
	}

	void Init(Vector &input, idx_t count, shared_ptr<const std::vector<JsonoShredSignature>> signatures) {
		Reset(ReadPolicy::WholeDocument);
		InitJsonoVectorData(input, count, data_);
		verifier_.InitSignaturesRef(std::move(signatures));
	}

	// Whole-document policy with caller-supplied signatures
	// (__jsono_internal_checked_residual receives them as plan constants).
	void Init(Vector &input, idx_t count, std::vector<JsonoShredSignature> signatures) {
		Reset(ReadPolicy::WholeDocument);
		InitJsonoVectorData(input, count, data_);
		verifier_.InitSignatures(std::move(signatures));
	}

	// ReadPolicy::PointRead (extract / introspect / match / project): these legitimately read a
	// manifested residual the optimizer handed over — the rewrite routes only non-covered paths
	// here.
	void InitPointRead(Vector &input, idx_t count) {
		Reset(ReadPolicy::PointRead);
		InitJsonoVectorData(input, count, data_);
	}

	// ReadPolicy::Trusted. The single sanctioned caller is jsono_overlay's fold: its residual
	// input legitimately carries a manifest that __jsono_internal_checked_residual already
	// verified upstream, and the overlay's job is precisely to refill those stripped paths from
	// the shreds.
	void InitTrusted(Vector &input, idx_t count) {
		Reset(ReadPolicy::Trusted);
		InitJsonoVectorData(input, count, data_);
	}

	// Always-inline: this wraps the per-row hot path of every reader (extract, match,
	// project, transform, ...); left out-of-line the row loop pays a call and the
	// blob/view out-params stop scalarizing.
	JSONO_ALWAYS_INLINE JsonoRowState Read(idx_t row, JsonoBlobRow &blob, JsonoView &view) {
		if (!ReadJsonoRowStrict(data_, row, blob)) {
			return JsonoRowState::Null;
		}
		return ParseAndVerify(blob, view);
	}

	// Permissive read for optimizer-injected expressions that can see NULL body fields on rows
	// the surrounding CASE discards (see ReadJsonoRow).
	JsonoRowState ReadPermissive(idx_t row, JsonoBlobRow &blob, JsonoView &view) {
		if (!ReadJsonoRow(data_, row, blob)) {
			return JsonoRowState::Null;
		}
		return ParseAndVerify(blob, view);
	}

	// Point-read checks, memoized by the manifest tail. `steps` must be a stable object across
	// calls on one reader (it keys the memo); both are no-ops for rows without a manifest.
	void CheckPathMiss(const JsonoView &view, const vector<PathStep> &steps) {
		CheckCover(view, steps, false, miss_memo_);
	}

	void CheckContainerRead(const JsonoView &view, const vector<PathStep> &steps) {
		CheckCover(view, steps, true, container_memo_);
	}

	// The verified manifest entries of the row just read (the reshred writer carries the
	// stripped state of kept shreds over to its output manifest). Only the whole-document policy
	// verifies and collects entries; under any other policy the verifier's answer for a manifested
	// row would be the EMPTY list, which reads as "nothing was stripped" — a confident wrong
	// answer — so asking is a caller bug. The switch has no default on purpose: a fourth policy
	// must decide its answer here before it compiles.
	const std::vector<ShredManifestEntry> &RowManifest(const JsonoView &view) {
		switch (policy_) {
		case ReadPolicy::PointRead:
		case ReadPolicy::Trusted:
			throw InternalException("jsono: RowManifest requires the whole-document read policy");
		case ReadPolicy::WholeDocument:
			break;
		}
		return verifier_.VerifiedEntries(view);
	}

	// The input string_heap child vector (post-flatten). A VARCHAR reader references its heap
	// into the result so a String/NumberText value can be emitted as a string_t pointing
	// straight into these bytes — see ZeroCopyHeapText / StringVector::AddHeapReference.
	Vector &StringHeapVector() {
		return *data_.string_heap_vec;
	}

private:
	struct CoverMemo {
		const vector<PathStep> *steps = nullptr;
		std::string read_text;
		bool renderable = false;
		std::string tail;
		bool ok = false;
	};

	// Every Init begins here: a re-init is a new input, and both the policy and the cover memos
	// (keyed by manifest tail plus a `steps` ADDRESS, which a new call site can reuse) describe the
	// old one. Carrying either over would answer a question about the new input with the old
	// input's proof.
	void Reset(ReadPolicy policy) {
		policy_ = policy;
		miss_memo_ = CoverMemo();
		container_memo_ = CoverMemo();
	}

	JSONO_ALWAYS_INLINE JsonoRowState ParseAndVerify(const JsonoBlobRow &blob, JsonoView &view) {
		view = MakeJsonoView(blob);
		if (!view.ParseHeader() || view.Slots() == 0) {
			return JsonoRowState::Empty;
		}
		if (policy_ == ReadPolicy::WholeDocument) {
			verifier_.Verify(view);
		}
		return JsonoRowState::Value;
	}

	void CheckCover(const JsonoView &view, const vector<PathStep> &steps, bool found_container, CoverMemo &memo) {
		if (!view.HasShredManifest()) {
			return;
		}
		auto tail = view.ManifestTail();
		if (memo.ok && memo.steps == &steps && tail.size() == memo.tail.size() &&
		    std::memcmp(tail.data(), memo.tail.data(), tail.size()) == 0) {
			return;
		}
		memo.ok = false;
		if (memo.steps != &steps) {
			memo.steps = &steps;
			memo.renderable = TryStepsToJsonPath(steps, memo.read_text);
		}
		if (memo.renderable) {
			ThrowIfManifestCoversPathText(view, memo.read_text, found_container);
		} else {
			ThrowIfManifestCoversPath(view, steps, found_container, manifest_scratch_, steps_scratch_);
		}
		memo.tail.assign(tail.data(), tail.size());
		memo.ok = true;
	}

	JsonoVectorData data_;
	ShredManifestVerifier verifier_;
	ReadPolicy policy_ = ReadPolicy::WholeDocument;
	CoverMemo miss_memo_;
	CoverMemo container_memo_;
	std::vector<ShredManifestEntry> manifest_scratch_;
	vector<PathStep> steps_scratch_;
};

// Locate-then-render: classify the value at `located_cursor` (container / inline scalar / rendered
// scalar / JSON null) and drive a SINK with the resulting `->>` text. The manifest guard
// (CheckContainerRead) fires for a container exactly as a direct read would. The readers that share
// this — the optimizer's filter matcher and projection writer, and jsono_extract_string — differ only
// in the terminal action, captured by the sink: OnInlineText sees heap-backed text (a String/
// NumberText value, zero-copyable straight out of string_heap), OnRenderedText sees transient
// `scratch` bytes (a serialized container or a rendered numeric/bool, must be copied), OnNull is the
// SQL-NULL outcome (JSON null, or — for the matcher — a non-match).
struct JsonoScalarText {
	nonstd::string_view value;
	bool inline_text = false;
};

inline bool TryGetScalarExtractText(const JsonoScalar &scalar, std::string &scratch, JsonoScalarText &text) {
	if (scalar.kind == JsonoScalarKind::Null) {
		return false;
	}
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		text.value = scalar.text;
		text.inline_text = true;
		return true;
	}
	scratch.clear();
	if (RenderExtractText(scalar, scratch)) {
		return false;
	}
	text.value = nonstd::string_view(scratch.data(), scratch.size());
	text.inline_text = false;
	return true;
}

template <class SINK>
JSONO_ALWAYS_INLINE void EmitLocatedText(JsonoRowReader &reader, const JsonoView &view, const vector<PathStep> &steps,
                                         const JsonoCursor &located_cursor, std::string &scratch, SINK &sink) {
	auto slot_tag = SlotTag(view.SlotAt(located_cursor.pos));
	if (slot_tag == tag::OBJ_START || slot_tag == tag::ARR_START) {
		// Serializing the whole container would silently include/drop a manifest leaf inside it.
		reader.CheckContainerRead(view, steps);
		scratch.clear();
		auto cursor = located_cursor;
		AppendJsonValueText(view, cursor, scratch, 0);
		sink.OnRenderedText(nonstd::string_view(scratch.data(), scratch.size()));
		return;
	}
	auto cursor = located_cursor;
	auto scalar = DecodeScalarAt(view, cursor);
	JsonoScalarText text;
	if (!TryGetScalarExtractText(scalar, scratch, text)) {
		sink.OnNull();
		return;
	}
	if (text.inline_text) {
		sink.OnInlineText(text.value);
		return;
	}
	sink.OnRenderedText(text.value);
}

// EmitLocatedText sink that writes `->>` text into a FLAT string result vector (the optimizer's
// projection writer and jsono_extract_string share it). A valid row references zero-copy heap text or
// copies the rendered bytes; a null outcome sets the row NULL. The caller must AddHeapReference the
// input string_heap so the zero-copy values stay alive.
struct JsonoExtractStringSink {
	Vector &result;
	string_t *result_data;
	idx_t row;

	void OnInlineText(nonstd::string_view text) {
		FlatVector::Validity(result).SetValid(row);
		result_data[row] = ZeroCopyHeapText(text);
	}
	void OnRenderedText(nonstd::string_view text) {
		FlatVector::Validity(result).SetValid(row);
		result_data[row] = StringVector::AddString(result, text.data(), text.size());
	}
	void OnNull() {
		FlatVector::SetNull(result, row, true);
	}
};

template <class POLICY>
void JsonoPointReadRows(JsonoRowReader &reader, idx_t count, const vector<PathStep> *steps,
                        JsonoPathLocateState &locate_state, POLICY &policy) {
	JsonoView view;
	for (idx_t row = 0; row < count; row++) {
		locate_state.NextRow();
		JsonoBlobRow blob;
		if (reader.Read(row, blob, view) != JsonoRowState::Value) {
			policy.Null(row);
			continue;
		}
		JsonoCursor cursor;
		if (steps) {
			if (!LocatePathSteps(locate_state, 0, *steps, view, cursor)) {
				reader.CheckPathMiss(view, *steps);
				policy.Null(row);
				continue;
			}
		}
		policy.Found(reader, view, steps, cursor, row);
	}
}

} // namespace jsono
} // namespace duckdb
