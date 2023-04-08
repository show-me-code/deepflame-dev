cmake_minimum_required(VERSION 3.5)
project(dfMatrix)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
enable_language(CUDA)
set(AMGX_DIR $ENV{AMGX_DIR})

include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${AMGX_DIR}/include
)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++17)

add_library(${PROJECT_NAME} 
    SHARED 
        dfMatrix.cu
        AmgXSolver.cu)

target_link_libraries(${PROJECT_NAME}
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${AMGX_DIR}/build/libamgxsh.so
)
