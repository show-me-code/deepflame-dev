cmake_minimum_required(VERSION 3.5)
project(dfMatrix)
FIND_PACKAGE(CUDA REQUIRED)
enable_language(CUDA)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++17)

add_library(${PROJECT_NAME} SHARED dfMatrix.cu)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
