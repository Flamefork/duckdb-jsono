#pragma once

#include "jsono.hpp"
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

// Memoized shred-manifest verification (the check itself is VerifyShredManifestEntries).
// Vectors are overwhelmingly manifest-homogeneous — the rows of one column chunk were shredded
// with the same spec — so after the first row a verification is one byte-compare of the row's
// manifest tail against the last verified tail.
class ShredManifestVerifier {
public:
	// Signatures = the shreds the reading type carries, as (path, type-string) pairs.
	void InitFromType(const LogicalType &type) {
		JsonoLayoutType layout;
		if (TryParseJsonoLayoutType(type, layout)) {
			signatures_.reserve(layout.shreds.size());
			for (auto &shred : layout.shreds) {
				signatures_.emplace_back(shred.first, shred.second.ToString());
			}
		}
	}

	void InitSignatures(std::vector<std::pair<std::string, std::string>> signatures) {
		signatures_ = std::move(signatures);
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
			no_entries_.clear();
			return no_entries_;
		}
		return entries_;
	}

private:
	void VerifyManifested(nonstd::string_view tail) {
		verified_ = false;
		tail_.assign(tail.data(), tail.size());
		ParseShredManifestBytes(tail_.data(), tail_.size(), entries_, &signatures_);
		VerifyShredManifestEntries(entries_, signatures_);
		verified_ = true;
	}

	bool TailMatches(nonstd::string_view tail) const {
		return tail.size() == tail_.size() && std::memcmp(tail.data(), tail_.data(), tail.size()) == 0;
	}

	std::vector<std::pair<std::string, std::string>> signatures_;
	std::string tail_;
	std::vector<ShredManifestEntry> entries_;
	std::vector<ShredManifestEntry> no_entries_;
	bool verified_ = false;
};

// Manifest check for a point read over a residual that carries a shred manifest. `found_container`
// = false: the read path was not found — an exact manifest match (or a manifest leaf inside the
// missing subtree) means the value was stripped into a shred this reading context does not have,
// so fail loud instead of returning a silent NULL. `found_container` = true: the read returns a
// whole found container — a manifest leaf strictly inside it means the result would be missing
// that leaf. An honest shredded read never reaches either case (the optimizer reads shreds
// directly and reconstructs subtree reads that overlap a shred), so a hit here is always a row
// narrowed by a raw struct cast. Indexed/bitset manifests require the original shred layout to
// decode their paths; without it, the parser fails loud before this filter can prove non-coverage.
inline void ThrowIfManifestCoversPath(const JsonoView &view, const vector<PathStep> &read_steps, bool found_container,
                                      std::vector<ShredManifestEntry> &manifest_scratch,
                                      vector<PathStep> &steps_scratch,
                                      const std::vector<std::pair<std::string, std::string>> *signatures = nullptr) {
	view.ReadShredManifest(manifest_scratch, signatures);
	for (auto &entry : manifest_scratch) {
		auto path = string(entry.path);
		steps_scratch.clear();
		if (!path.empty() && path[0] == '$') {
			steps_scratch = ParseJsonoPath(path, "jsono shred manifest");
		} else {
			steps_scratch.push_back(PathStep {PathStepKind::Key, path, 0});
		}
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

// The row-read layer: every operator that decodes JSONO values reads rows through this object,
// which bundles the strict row read, the header parse and the shred-manifest check — the
// "every reader verifies the manifest" invariant (docs/jsono_format.md) holds by construction
// instead of per call site. Operators that only move blobs verbatim (identity copies, storage
// introspection) stay on ReadJsonoRowStrict: the manifest travels with the bytes, so a move is
// not a lossy read.
class JsonoRowReader {
public:
	// Whole-document policy (the default): every manifest entry must name a shred `input`'s
	// type carries (a plain type carries none, so any manifest fails loud as a narrowed row).
	void Init(Vector &input, idx_t count) {
		InitJsonoVectorData(input, count, data_);
		verifier_.InitFromType(input.GetType());
		verify_on_read_ = true;
	}

	// Whole-document policy with caller-supplied signatures
	// (__jsono_internal_checked_residual receives them as plan constants).
	void Init(Vector &input, idx_t count, std::vector<std::pair<std::string, std::string>> signatures) {
		InitJsonoVectorData(input, count, data_);
		verifier_.InitSignatures(std::move(signatures));
		verify_on_read_ = true;
	}

	// Point-read policy (extract / introspect / match / project): these legitimately read a
	// manifested residual the optimizer handed over (the rewrite routes only non-covered paths
	// here), so Read() does not verify whole-document. The reader instead checks the two
	// outcomes where a stripped value would silently change the result — a path miss and a
	// whole-container read — via CheckPathMiss / CheckContainerRead. Point reads also prefetch
	// the row's streams (bulk walkers touch them densely anyway).
	void InitPointRead(Vector &input, idx_t count) {
		InitJsonoVectorData(input, count, data_);
		InitSignaturesFromType(input.GetType(), point_read_signatures_);
		verify_on_read_ = false;
		prefetch_ = true;
	}

	// No verification at all. The single sanctioned caller is jsono_overlay's fold: its
	// residual input legitimately carries a manifest that __jsono_internal_checked_residual
	// already verified upstream, and the overlay's job is precisely to refill those stripped
	// paths from the shreds.
	void InitTrusted(Vector &input, idx_t count) {
		InitJsonoVectorData(input, count, data_);
		verify_on_read_ = false;
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
	// stripped state of kept shreds over to its output manifest).
	const std::vector<ShredManifestEntry> &RowManifest(const JsonoView &view) {
		return verifier_.VerifiedEntries(view);
	}

	// The input string_heap child vector (post-flatten). A VARCHAR reader references its heap
	// into the result so a String/NumberText value can be emitted as a string_t pointing
	// straight into these bytes — see ZeroCopyHeapText / StringVector::AddHeapReference.
	Vector &StringHeapVector() {
		return *data_.string_heap_vec;
	}

private:
	static void InitSignaturesFromType(const LogicalType &type,
	                                   std::vector<std::pair<std::string, std::string>> &signatures) {
		signatures.clear();
		JsonoLayoutType layout;
		if (TryParseJsonoLayoutType(type, layout)) {
			signatures.reserve(layout.shreds.size());
			for (auto &shred : layout.shreds) {
				signatures.emplace_back(shred.first, shred.second.ToString());
			}
		}
	}

	struct CoverMemo {
		const vector<PathStep> *steps = nullptr;
		std::string tail;
		bool ok = false;
	};

	JSONO_ALWAYS_INLINE JsonoRowState ParseAndVerify(const JsonoBlobRow &blob, JsonoView &view) {
		view = MakeJsonoView(blob);
		if (prefetch_) {
			PrefetchJsonoRowStreams(blob);
		}
		if (!view.ParseHeader() || view.Slots() == 0) {
			return JsonoRowState::Empty;
		}
		if (verify_on_read_) {
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
		ThrowIfManifestCoversPath(view, steps, found_container, manifest_scratch_, steps_scratch_,
		                          &point_read_signatures_);
		memo.steps = &steps;
		memo.tail.assign(tail.data(), tail.size());
		memo.ok = true;
	}

	JsonoVectorData data_;
	ShredManifestVerifier verifier_;
	bool verify_on_read_ = true;
	bool prefetch_ = false;
	CoverMemo miss_memo_;
	CoverMemo container_memo_;
	std::vector<ShredManifestEntry> manifest_scratch_;
	std::vector<std::pair<std::string, std::string>> point_read_signatures_;
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
	if (scalar.kind == JsonoScalarKind::Null) {
		sink.OnNull();
		return;
	}
	if (scalar.kind == JsonoScalarKind::String || scalar.kind == JsonoScalarKind::NumberText) {
		sink.OnInlineText(scalar.text);
		return;
	}
	scratch.clear();
	if (RenderExtractText(scalar, scratch)) {
		sink.OnNull();
		return;
	}
	sink.OnRenderedText(nonstd::string_view(scratch.data(), scratch.size()));
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

} // namespace jsono
} // namespace duckdb
