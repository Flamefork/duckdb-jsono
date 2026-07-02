cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME jsono)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# C++ standard is inherited from DuckDB's root CMake (CMAKE_CXX_STANDARD=11),
# which is the single source of truth. std::string_view (C++17) is replaced by
# vendored nonstd::string_view (string-view-lite) so the code builds under C++11.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include third_party/string-view-lite duckdb/third_party/yyjson/include)

# string-view-lite's _sv user-defined literal operators are unused here and emit
# -Wdeprecated-literal-operator on modern clang; disable them at the source.
add_compile_definitions(nssv_CONFIG_USR_SV_OPERATOR=0)

set(EXTENSION_SOURCES
    src/jsono_extension.cpp
    src/jsono_ops.cpp
    src/jsono_path_ops.cpp
    src/jsono_entries.cpp
    src/jsono_elements.cpp
    src/jsono_storage_size.cpp
    src/jsono_validate.cpp
    src/jsono_merge.cpp
    src/jsono_diff.cpp
    src/jsono_advisor.cpp
    src/jsono_transform.cpp
    src/jsono_optimizer.cpp
    src/jsono_number.cpp
    src/jsono_dom.cpp
    src/jsono_parse.cpp
    src/jsono_struct_constructor.cpp
    src/jsono_to_json.cpp
    src/jsono_shred.cpp
    src/jsono.cpp
    src/jsono_extract.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Treat a switch over an enum that omits a case (with no `default`) as an error in our own sources.
# The shred-type dispatch (ShredKind / ShredPrimitive) relies on this: adding a shred category or
# scalar type then fails to compile in every native reader rather than reaching a silently-swallowing
# runtime `default`. Scoped to the jsono targets so DuckDB's own headers are unaffected.
if(NOT MSVC)
  foreach(_jsono_target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_compile_options(${_jsono_target} PRIVATE -Werror=switch)
  endforeach()
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
