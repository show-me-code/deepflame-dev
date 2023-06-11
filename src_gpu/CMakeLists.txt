#
# dfMatrix CMake configuration
#
cmake_minimum_required(VERSION 3.5)

project(dfMatrix LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(LIBAMGXSH amgxsh PATHS $ENV{AMGX_DIR}/build)

include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    $ENV{AMGX_DIR}/include
)

add_library(${PROJECT_NAME} 
    SHARED 
        dfUEqn.cu 
        dfRhoEqn.cu 
        dfYEqn.cu
        dfEEqn.cu
        AmgXSolver.cu
        dfMatrixDataBase.cpp)

target_link_libraries(${PROJECT_NAME}
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${LIBAMGXSH}
)
target_compile_options(dfMatrix PUBLIC -g)
option(DFMATRIX_ENABLE_DETAILED_DEBUG "Enable detailed debug build" OFF)
if (DFMATRIX_ENABLE_DETAILED_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
endif()
