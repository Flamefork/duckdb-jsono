#pragma once

#include "jsono.hpp"
#include "jsono_writer.hpp"

#include "string_view.hpp"

#include <string>

namespace duckdb {
namespace jsono {

// Classify a raw JSON-number token (as delivered by yyjson under
// YYJSON_READ_NUMBER_AS_RAW — already validated to be a well-formed RFC 8259
// number) and emit it into `builder` along the spec ladder:
//
//   integer  |v|<2^59          -> VAL_INT60
//            2^59..2^63         -> VAL_EXT/INT64
//            2^63..2^64-1       -> VAL_EXT/UINT64
//            else (bignum)      -> VAL_EXT/NUMBER (raw text)
//   fraction fits-decimal       -> VAL_DEC60
//            else               -> VAL_EXT/NUMBER (raw text)
//
// The classifier never does float arithmetic for INT60/INT64/UINT64/DEC60; it
// reads digits out of the text. DEC60 is chosen only when the value's canonical
// mantissa/scale text reconstruction is byte-identical to the input — so any
// form not reconstructible that way (exponent, scale>22, mantissa>=2^53) falls
// through to NUMBER and is preserved byte-exact.
void EmitNumberFromText(nonstd::string_view text, JsonoBuilder &builder);

// Canonical mantissa/scale text used both to verify DEC60 eligibility at encode
// time and to materialise a stored DEC60 slot back to JSON text. scale must be
// >= 1 (pure integers never take the DEC60 path).
void AppendDec60Text(std::string &out, bool negative, uint64_t abs_mantissa, uint64_t scale);

// Serialize a binary double to its shortest round-trippable decimal text via
// yyjson (Dragonbox). Only VAL_EXT/DOUBLE values (struct-constructed, no source
// text) take this path; text-sourced numbers keep their bytes via DEC60/NUMBER.
void EmitDouble(double v, std::string &out);

} // namespace jsono
} // namespace duckdb
