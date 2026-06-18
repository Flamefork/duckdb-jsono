#include "jsono_number.hpp"

#include "jsono.hpp"

#include "yyjson.hpp"
#include "string_view.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace duckdb {
namespace jsono {

namespace {

// Largest unsigned value whose 10x + 9 still fits uint64 without overflow; used
// to stop digit accumulation the moment a value leaves the uint64 range.
constexpr uint64_t UINT64_DIV10 = 1844674407370955161ULL; // floor((2^64-1)/10)
constexpr uint64_t UINT64_MOD10 = 5;                      // (2^64-1) % 10

// Accumulate `digits` into `acc` as an unsigned integer. Returns false the
// moment it would exceed uint64 — the caller then routes to NUMBER.
bool AccumulateUInt(nonstd::string_view digits, uint64_t &acc) {
	acc = 0;
	for (char c : digits) {
		uint64_t d = uint64_t(c - '0');
		if (acc > UINT64_DIV10 || (acc == UINT64_DIV10 && d > UINT64_MOD10)) {
			return false;
		}
		acc = acc * 10 + d;
	}
	return true;
}

// Accumulate int-part then frac-part digits into one uint64 mantissa without
// allocating a concatenated digit string. Returns false the moment it would
// exceed uint64 (→ NUMBER).
bool AccumulateUIntPair(nonstd::string_view int_digits, nonstd::string_view frac_digits, uint64_t &acc) {
	acc = 0;
	for (nonstd::string_view part : {int_digits, frac_digits}) {
		for (char c : part) {
			uint64_t d = uint64_t(c - '0');
			if (acc > UINT64_DIV10 || (acc == UINT64_DIV10 && d > UINT64_MOD10)) {
				return false;
			}
			acc = acc * 10 + d;
		}
	}
	return true;
}

// Write the canonical DEC60 decimal text (mantissa / 10^scale) into `out` and
// return its length, without heap allocation. Single source of truth for the
// decimal rendering (AppendDec60Text wraps this). Bounds: mantissa < 2^53
// (≤16 digits), scale ≤ 22 → padded ≤ 23, total ≤ 25; out must hold ≥ 26 bytes.
size_t BuildDec60Canonical(char *out, bool negative, uint64_t abs_mantissa, uint64_t scale) {
	char digits[24];
	size_t d = 0;
	if (abs_mantissa == 0) {
		digits[d++] = '0';
	} else {
		char rev[24];
		size_t t = 0;
		uint64_t m = abs_mantissa;
		while (m) {
			rev[t++] = char('0' + m % 10);
			m /= 10;
		}
		for (size_t k = 0; k < t; k++) {
			digits[k] = rev[t - 1 - k];
		}
		d = t;
	}
	// Zero-pad the front so the total length is at least scale + 1 (one integer digit).
	char padded[48];
	size_t plen;
	if (d < scale + 1) {
		size_t pad = scale + 1 - d;
		for (size_t k = 0; k < pad; k++) {
			padded[k] = '0';
		}
		for (size_t k = 0; k < d; k++) {
			padded[pad + k] = digits[k];
		}
		plen = pad + d;
	} else {
		for (size_t k = 0; k < d; k++) {
			padded[k] = digits[k];
		}
		plen = d;
	}
	size_t split = plen - scale;
	size_t pos = 0;
	if (negative) {
		out[pos++] = '-';
	}
	for (size_t k = 0; k < split; k++) {
		out[pos++] = padded[k];
	}
	out[pos++] = '.';
	for (size_t k = split; k < plen; k++) {
		out[pos++] = padded[k];
	}
	return pos;
}

ClassifiedNumber ClassifiedNumberText() {
	return ClassifiedNumber {MakeExtSlot(ext_subtype::NUMBER), 0, false};
}

ClassifiedNumber ClassifiedInt(int64_t v) {
	return ClassifiedNumber {FitsInt60(v) ? MakeSlot(tag::VAL_INT60, 0) : MakeExtSlot(ext_subtype::INT64), uint64_t(v),
	                         true};
}

ClassifiedNumber ClassifyIntegerFromDigits(bool negative, nonstd::string_view digits) {
	uint64_t magnitude;
	if (!AccumulateUInt(digits, magnitude)) {
		return ClassifiedNumberText(); // beyond uint64 → bignum NUMBER
	}
	if (negative) {
		// |v| <= 2^63 fits int64 (INT64_MIN = -2^63); anything larger is bignum.
		if (magnitude <= uint64_t(1) << 63) {
			return ClassifiedInt(magnitude == (uint64_t(1) << 63) ? std::numeric_limits<int64_t>::min()
			                                                      : -int64_t(magnitude));
		}
		return ClassifiedNumberText();
	}
	if (magnitude <= uint64_t(std::numeric_limits<int64_t>::max())) {
		return ClassifiedInt(int64_t(magnitude)); // INT60 or VAL_EXT/INT64
	}
	return ClassifiedNumber {MakeExtSlot(ext_subtype::UINT64), magnitude, true}; // VAL_EXT/UINT64
}

} // namespace

void EmitDouble(double v, std::string &out) {
	// Non-finite doubles have no JSON form; reject rather than emit invalid output.
	// Construction already blocks these, so this guards externally-crafted blobs.
	if (!std::isfinite(v)) {
		throw InvalidInputException("JSONO cannot serialize non-finite double value (NaN/Infinity)");
	}
	// yyjson's writer produces the shortest round-trippable decimal (Dragonbox);
	// libc++ on macOS doesn't ship std::to_chars for double, and yyjson is our
	// single JSON dependency, so we route a one-value mutable doc through it.
	duckdb_yyjson::yyjson_mut_doc *doc = duckdb_yyjson::yyjson_mut_doc_new(nullptr);
	duckdb_yyjson::yyjson_mut_val *val = duckdb_yyjson::yyjson_mut_real(doc, v);
	size_t len = 0;
	char *text = duckdb_yyjson::yyjson_mut_val_write(val, 0, &len);
	if (text) {
		out.append(text, len);
		free(text);
	}
	duckdb_yyjson::yyjson_mut_doc_free(doc);
}

void AppendDec60Text(std::string &out, bool negative, uint64_t abs_mantissa, uint64_t scale) {
	// Single source of truth for the mantissa/scale → decimal text: build the
	// canonical bytes into a stack buffer, then append in one shot.
	char buf[26];
	size_t len = BuildDec60Canonical(buf, negative, abs_mantissa, scale);
	out.append(buf, len);
}

ClassifiedNumber ClassifyNumberFromText(nonstd::string_view text) {
	bool negative = !text.empty() && text.front() == '-';
	size_t i = negative ? 1 : 0;

	// Scan the integer-part digits.
	size_t int_start = i;
	while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
		i++;
	}
	nonstd::string_view int_digits = text.substr(int_start, i - int_start);

	if (i == text.size()) {
		return ClassifyIntegerFromDigits(negative, int_digits);
	}

	// A fraction or exponent follows. Exponent forms are not reconstructible
	// from mantissa/scale, so they go straight to NUMBER.
	if (text[i] != '.') {
		return ClassifiedNumberText(); // 'e'/'E' exponent → NUMBER
	}

	size_t frac_start = i + 1;
	size_t j = frac_start;
	while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
		j++;
	}
	if (j != text.size()) {
		return ClassifiedNumberText(); // exponent after fraction (e.g. 1.5e-3) → NUMBER
	}
	nonstd::string_view frac_digits = text.substr(frac_start, j - frac_start);
	uint64_t scale = frac_digits.size();

	// Mantissa is the integer + fraction digits. Reject (→NUMBER) if it overflows
	// uint64; the tighter |mantissa|<2^53 bound is checked below. No concat alloc.
	uint64_t mantissa;
	if (!AccumulateUIntPair(int_digits, frac_digits, mantissa) || !FitsDec60(mantissa, scale)) {
		return ClassifiedNumberText();
	}

	// DEC60 is valid only when its canonical reconstruction is byte-identical to
	// the input — this is the contract that makes the round-trip byte-exact and
	// rejects forms like leading zeros that the canonical form would drop. Built
	// in a stack buffer (no heap alloc on the hot fractional path).
	char canonical[40];
	size_t canonical_len = BuildDec60Canonical(canonical, negative, mantissa, scale);
	if (nonstd::string_view(canonical, canonical_len) == text) {
		return ClassifiedNumber {MakeSlot(tag::VAL_DEC60, 0), MakeDec60Payload(negative, mantissa, scale), true};
	}
	return ClassifiedNumberText();
}

void EmitNumberFromText(nonstd::string_view text, JsonoBuilder &builder) {
	auto classified = ClassifyNumberFromText(text);
	if (!classified.consumes_num) {
		builder.EmitNumberText(text);
		return;
	}
	builder.slots.push_back(classified.slot);
	builder.nums.push_back(classified.num);
}

} // namespace jsono
} // namespace duckdb
