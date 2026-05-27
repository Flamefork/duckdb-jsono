#define DUCKDB_EXTENSION_MAIN

#include "jsono_extension.hpp"
#include "jsono_extract.hpp"
#include "jsono.hpp"
#include "jsono_ops.hpp"
#include "jsono_transform.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	RegisterJsonoType(loader);
	RegisterJsonoOps(loader);
	RegisterJsonoTransform(loader);
	RegisterJsonoExtract(loader);
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
