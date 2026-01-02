cmake_minimum_required(VERSION 3.16)
project(lacam-project CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LAGAT_DIR "../../../../cpp_planners/lagat")
add_subdirectory(${LAGAT_DIR}/lacam planner)

add_library(planner SHARED main.cpp)
target_compile_features(planner PUBLIC cxx_std_17)
target_include_directories(planner PRIVATE ${LAGAT_DIR}/lacam/include)
target_link_libraries(planner lacam)
