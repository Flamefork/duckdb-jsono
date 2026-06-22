#pragma once

#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

// One shred column's reading lanes for a chunk. A scalar shred uses `fmt` as the value lane and
// `scalar_kind`; an object array (kind == Array) uses `fmt` as the list_entry lane plus
// `sub_fmt`/`sub_kind` for the element-struct subfield value lanes; a scalar array (kind ==
// ScalarArray) uses `fmt` as the list_entry lane plus a single element value lane stored in
// `sub_fmt[0]`/`sub_kind[0]` (the list child vector), so storage_size's per-element byte sum reads
// both array kinds the same way. Both native shred-lane readers (jsono_storage_size, jsono_entries)
// build these via InitShredLanes, so the list walk and shred-type classification live in one place.
struct ShredLane {
	ShredKind kind = ShredKind::Scalar;
	jsono::JsonoScalarPrimitive scalar_kind = jsono::JsonoScalarPrimitive::Varchar; // valid when kind == Scalar
	UnifiedVectorFormat fmt;
	vector<jsono::JsonoScalarPrimitive> sub_kind; // element value lanes; valid when kind == Array / ScalarArray
	vector<UnifiedVectorFormat> sub_fmt;          // element value lanes; valid when kind == Array / ScalarArray
};

// Build the reading lanes for every shred of a shredded jsono `input` vector, in shred order (shred
// f is JsonoShredVector(input, f)). `shreds` is the input type's shred child list from
// TryParseJsonoLayoutType. Categories and scalar kinds are resolved through the shared shred
// classifiers, so a reader never re-lists the shred type ids.
inline void InitShredLanes(Vector &input, idx_t count, const child_list_t<LogicalType> &shreds,
                           vector<ShredLane> &lanes) {
	lanes.resize(shreds.size());
	for (idx_t f = 0; f < shreds.size(); f++) {
		auto &lane = lanes[f];
		auto &shred_vec = jsono::JsonoShredVector(input, f);
		shred_vec.ToUnifiedFormat(count, lane.fmt);
		lane.kind = ClassifyShredKind(shreds[f].second);
		switch (lane.kind) {
		case ShredKind::Scalar:
			lane.scalar_kind = jsono::JsonoScalarPrimitiveFromType(shreds[f].second, "jsono shred lane reader");
			break;
		case ShredKind::Array: {
			auto &element_struct = ListVector::GetEntry(shred_vec);
			auto element_count = ListVector::GetListSize(shred_vec);
			auto &subfield_vecs = StructVector::GetEntries(element_struct);
			auto &element_fields = StructType::GetChildTypes(ListType::GetChildType(shreds[f].second));
			lane.sub_fmt.resize(subfield_vecs.size());
			lane.sub_kind.resize(subfield_vecs.size());
			for (idx_t j = 0; j < subfield_vecs.size(); j++) {
				subfield_vecs[j]->ToUnifiedFormat(element_count, lane.sub_fmt[j]);
				lane.sub_kind[j] =
				    jsono::JsonoScalarPrimitiveFromType(element_fields[j].second, "jsono shred lane reader");
			}
			break;
		}
		case ShredKind::ScalarArray: {
			// One element value lane (the list child vector) stored at index 0, so the per-element
			// byte sum and the entries expansion treat it as a single-column array.
			auto element_count = ListVector::GetListSize(shred_vec);
			lane.sub_fmt.resize(1);
			lane.sub_kind.resize(1);
			ListVector::GetEntry(shred_vec).ToUnifiedFormat(element_count, lane.sub_fmt[0]);
			lane.sub_kind[0] = jsono::JsonoScalarPrimitiveFromType(ListType::GetChildType(shreds[f].second),
			                                                       "jsono shred lane reader");
			break;
		}
		}
	}
}

} // namespace duckdb
