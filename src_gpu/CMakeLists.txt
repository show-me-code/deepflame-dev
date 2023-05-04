cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(dfMatrix)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
set(AMGX_DIR $ENV{AMGX_DIR})

include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${AMGX_DIR}/include
    # /usr/local/cuda-11.6/nsight-systems-2021.5.2/target-linux-x64/nvtx/include/
)

set(CMAKE_C_COMPILER g++)

add_library(${PROJECT_NAME} 
    SHARED 
        dfMatrix.cu
        AmgXSolver.cu)

target_link_libraries(${PROJECT_NAME}
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${AMGX_DIR}/build/libamgxsh.so
    # CUDA::nvToolsExt
    # -ldl
)
