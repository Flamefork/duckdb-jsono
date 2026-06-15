#pragma once

#include "jsono_shred.hpp"
#include "jsono_writer.hpp"

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

// One shred column's reading lanes for a chunk. A scalar shred uses `fmt` as the value lane and
// `scalar_kind`; an array shred uses `fmt` as the list_entry lane plus `sub_fmt`/`sub_kind` for the
// element-struct subfield value lanes (always the scalar shred set). Both native shred-lane readers
// (jsono_storage_size, jsono_entries) build these via InitShredLanes, so the LIST<STRUCT> walk and
// the shred-type classification live in one place rather than being re-derived per reader.
struct ShredLane {
	ShredKind kind = ShredKind::Scalar;
	ShredPrimitive scalar_kind = ShredPrimitive::Varchar; // valid when kind == Scalar
	UnifiedVectorFormat fmt;
	vector<ShredPrimitive> sub_kind;     // element-struct order; valid when kind == Array
	vector<UnifiedVectorFormat> sub_fmt; // element-struct order; valid when kind == Array
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
			TypeToShredPrimitive(shreds[f].second, lane.scalar_kind);
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
				TypeToShredPrimitive(element_fields[j].second, lane.sub_kind[j]);
			}
			break;
		}
		}
	}
}

} // namespace duckdb
