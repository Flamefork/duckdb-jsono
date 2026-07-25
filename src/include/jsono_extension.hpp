//===----------------------------------------------------------------------===//
//                         DuckDB
//
// jsono_extension.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

// Narrow includes on purpose (not the duckdb.hpp umbrella): every file defining a hook below
// includes this header, so the compiler — not the linker — checks the signatures.
#include "duckdb/main/extension.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

class JsonoExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

// Per-feature registration hooks; LoadInternal calls every one, so this list is the extension's
// feature roster — a new feature file adds its hook here. Order is free: JsonoType() is a plain
// accessor over the physical struct (there is no registered type to create first), and each hook
// registers into the catalog independently.
void RegisterJsonoType(ExtensionLoader &loader);              // jsono.cpp
void RegisterJsonoStructConstructor(ExtensionLoader &loader); // jsono_struct_constructor.cpp
void RegisterJsonoParse(ExtensionLoader &loader);             // jsono_parse.cpp
void RegisterJsonoToJson(ExtensionLoader &loader);            // jsono_to_json.cpp
void RegisterJsonoPathOps(ExtensionLoader &loader);           // jsono_path_ops.cpp
void RegisterJsonoEntries(ExtensionLoader &loader);           // jsono_entries.cpp
void RegisterJsonoArrayElements(ExtensionLoader &loader);     // jsono_elements.cpp
void RegisterJsonoValidate(ExtensionLoader &loader);          // jsono_validate.cpp
void RegisterJsonoStorageSize(ExtensionLoader &loader);       // jsono_storage_size.cpp
void RegisterJsonoMerge(ExtensionLoader &loader);             // jsono_merge.cpp
void RegisterJsonoReconstruct(ExtensionLoader &loader);       // jsono_reconstruct.cpp
void RegisterJsonoGroupMerge(ExtensionLoader &loader);        // jsono_group_merge.cpp
void RegisterJsonoGroupMergeKeyed(ExtensionLoader &loader);   // jsono_group_merge_keyed.cpp
void RegisterJsonoDiff(ExtensionLoader &loader);              // jsono_diff.cpp
void RegisterJsonoAdvisor(ExtensionLoader &loader);           // jsono_advisor.cpp
void RegisterJsonoCollect(ExtensionLoader &loader);           // jsono_collect.cpp
void RegisterJsonoTransform(ExtensionLoader &loader);         // jsono_transform.cpp
void RegisterJsonoShred(ExtensionLoader &loader);             // jsono_shred.cpp
void RegisterJsonoExtract(ExtensionLoader &loader);           // jsono_extract.cpp
void RegisterJsonoOptimizer(ExtensionLoader &loader);         // jsono_optimizer.cpp

} // namespace duckdb
