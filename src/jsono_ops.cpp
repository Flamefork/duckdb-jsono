#include "jsono_ops.hpp"

namespace duckdb {

void RegisterJsonoPathOps(ExtensionLoader &loader);
void RegisterJsonoEntries(ExtensionLoader &loader);
void RegisterJsonoArrayElements(ExtensionLoader &loader);
void RegisterJsonoValidate(ExtensionLoader &loader);
void RegisterJsonoStorageSize(ExtensionLoader &loader);
void RegisterJsonoMerge(ExtensionLoader &loader);
void RegisterJsonoDiff(ExtensionLoader &loader);
void RegisterJsonoAdvisor(ExtensionLoader &loader);
void RegisterJsonoCollect(ExtensionLoader &loader);

void RegisterJsonoOps(ExtensionLoader &loader) {
	RegisterJsonoPathOps(loader);
	RegisterJsonoEntries(loader);
	RegisterJsonoArrayElements(loader);
	RegisterJsonoValidate(loader);
	RegisterJsonoStorageSize(loader);
	RegisterJsonoMerge(loader);
	RegisterJsonoDiff(loader);
	RegisterJsonoAdvisor(loader);
	RegisterJsonoCollect(loader);
}

} // namespace duckdb
