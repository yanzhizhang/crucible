cmake_minimum_required(VERSION 3.24)
# C is enabled only because CapnProtoConfig's libatomic probe (check_library_exists) compiles C.
project(crucible_kernels LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CK_NATIVE "Compile for the build machine's CPU (-march=native)" ON)
option(CK_LTO "Link-time optimisation" ON)

set(CK_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/feitu"
    CACHE PATH "Directory holding the vendor feitu *.capnp schemas")

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(CapnProto CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)

# Generate C++ from the vendor schemas into the build tree; the schemas themselves stay the
# single source of truth in ../schemas/feitu.
set(CAPNPC_SRC_PREFIX "${CK_SCHEMA_DIR}")
set(CAPNPC_IMPORT_DIRS "${CK_SCHEMA_DIR}")
set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/capnp_gen")
file(MAKE_DIRECTORY "${CAPNPC_OUTPUT_DIR}")
capnp_generate_cpp(CK_CAPNP_SRCS CK_CAPNP_HDRS
  "${CK_SCHEMA_DIR}/base.capnp"
  "${CK_SCHEMA_DIR}/export.capnp"
  "${CK_SCHEMA_DIR}/order_v3.capnp"
  "${CK_SCHEMA_DIR}/transaction_v3.capnp"
  "${CK_SCHEMA_DIR}/quotation_v3.capnp"
  "${CK_SCHEMA_DIR}/index_v3.capnp")

nanobind_add_module(_feitu NB_STATIC src/bindings.cpp ${CK_CAPNP_SRCS})
target_include_directories(_feitu PRIVATE include "${CAPNPC_OUTPUT_DIR}")
target_link_libraries(_feitu PRIVATE CapnProto::capnp CapnProto::kj PkgConfig::ZSTD)
target_compile_options(_feitu PRIVATE -O3 $<$<BOOL:${CK_NATIVE}>:-march=native>)
if(CK_LTO)
  set_property(TARGET _feitu PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
endif()

install(TARGETS _feitu LIBRARY DESTINATION crucible_kernels)
