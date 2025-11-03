cmake_minimum_required(VERSION 3.16)
project(lacam-project CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LACAM3_DIR "../../../../cpp_planners/lacam3")
add_subdirectory(${LACAM3_DIR}/lacam3 lacam3)

add_library(lacam SHARED main.cpp)
target_compile_features(lacam PUBLIC cxx_std_17)
target_include_directories(lacam PRIVATE ${LACAM3_DIR}/lacam3/include)
target_link_libraries(lacam lacam3)
