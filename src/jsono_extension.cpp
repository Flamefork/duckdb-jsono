#define DUCKDB_EXTENSION_MAIN

#include "jsono_extension.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	RegisterJsonoType(loader);
	RegisterJsonoStructConstructor(loader);
	RegisterJsonoParse(loader);
	RegisterJsonoToJson(loader);
	RegisterJsonoPathOps(loader);
	RegisterJsonoEntries(loader);
	RegisterJsonoArrayElements(loader);
	RegisterJsonoValidate(loader);
	RegisterJsonoStorageSize(loader);
	RegisterJsonoMerge(loader);
	RegisterJsonoReconstruct(loader);
	RegisterJsonoGroupMerge(loader);
	RegisterJsonoGroupMergeKeyed(loader);
	RegisterJsonoDiff(loader);
	RegisterJsonoAdvisor(loader);
	RegisterJsonoCollect(loader);
	RegisterJsonoTransform(loader);
	RegisterJsonoShred(loader);
	RegisterJsonoExtract(loader);
	RegisterJsonoOptimizer(loader);
}

void JsonoExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string JsonoExtension::Name() {
	return "jsono";
}

std::string JsonoExtension::Version() const {
#ifdef EXT_VERSION_JSONO
	return EXT_VERSION_JSONO;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(jsono, loader) {
	duckdb::LoadInternal(loader);
}
}
