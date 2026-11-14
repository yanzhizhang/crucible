# The feature engine's C++ programs. Two ways to build:
#   plain   featgen / featvec / bookgen / lgbpred need only a C++20 compiler (the Python drivers also build them with one
#           g++ command when cmake is absent)
#   conda   inside the repo's conda env (environment.yml: libarrow, libparquet, libduckdb-devel, eigen) the programs that
#           read and write parquet / query the tick catalog themselves are built too
#       cmake -S research/featgen/cpp -B research/featgen/cpp/build -G Ninja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX && cmake --build research/featgen/cpp/build
cmake_minimum_required(VERSION 3.24)
project(crucible_featgen CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -O2 and no fast-math / FMA contraction: the programs must give the same doubles as the Python definitions
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -ffp-contract=off")
find_package(Threads REQUIRED)
foreach(prog featgen featvec bookgen lgbpred)
  add_executable(${prog} ${prog}.cpp)
  target_link_libraries(${prog} PRIVATE Threads::Threads)
endforeach()

# each library on its own: a machine with only Arrow / Parquet still gets the parquet-reading programs
find_package(Arrow CONFIG QUIET)
find_package(Parquet CONFIG QUIET)
# conda-forge's DuckDB CMake config names a static library the package does not ship: look the shared one up directly
find_library(DUCKDB_LIBRARY duckdb)
find_path(DUCKDB_INCLUDE_DIR duckdb.h)
find_package(Eigen3 CONFIG QUIET)
if(Arrow_FOUND AND Parquet_FOUND)
  message(STATUS "featgen: Arrow ${Arrow_VERSION} and Parquet: building lgbpred_pq (parquet in and out)")
  add_executable(lgbpred_pq lgbpred.cpp)
  target_compile_definitions(lgbpred_pq PRIVATE CRUCIBLE_ARROW)
  target_link_libraries(lgbpred_pq PRIVATE Arrow::arrow_shared Parquet::parquet_shared Threads::Threads)
  set_target_properties(lgbpred_pq PROPERTIES BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib")
else()
  message(STATUS "featgen: Arrow / Parquet not found: no lgbpred_pq")
endif()
if(Arrow_FOUND AND Parquet_FOUND AND DUCKDB_LIBRARY AND DUCKDB_INCLUDE_DIR AND Eigen3_FOUND)
  message(STATUS "featgen: DuckDB ${DUCKDB_LIBRARY}, Eigen ${Eigen3_VERSION}: building libcheck")
  add_executable(libcheck libcheck.cpp)
  target_include_directories(libcheck PRIVATE ${DUCKDB_INCLUDE_DIR})
  target_link_libraries(libcheck PRIVATE Arrow::arrow_shared Parquet::parquet_shared ${DUCKDB_LIBRARY} Eigen3::Eigen Threads::Threads)
  set_target_properties(libcheck PROPERTIES BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib" INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")
else()
  message(STATUS "featgen: DuckDB / Eigen not both found: no libcheck")
endif()
