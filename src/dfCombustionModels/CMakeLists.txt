project(dfCombustionModels LANGUAGES CXX)
FIND_PACKAGE(MPI REQUIRED)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

include_directories(

    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude 
    ${DF_SRC}/dfCanteraMixture
    ${PROJECT_SOURCE_DIR}
    ${OpenFOAM_SRC}/Pstream/mpi
    ${dfcombustion_inc}
    ${dfchemistry_inc}
    ${MPI_INCLUDE_PATH}
    $ENV{CANTERA_ROOT}/include
    #$ENV{WM_DIR}/rules/mplibType 
)


if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  find_package (Python REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11)
  include_directories(
  ${Python_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}/pybind11
  )
endif()

SET(workDir ${DF_SRC}/${PROJECT_NAME})
SET(SOURCES 
${workDir}/combustionModel/combustionModel.C
${workDir}/CombustionModel/CombustionModel/CombustionModels.C
${workDir}/PaSR/PaSRs.C
${workDir}/EDC/EDCs.C
${workDir}/laminar/laminars.C
${workDir}/noCombustion/noCombustions.C
)

add_library(dfCombustionModels SHARED ${SOURCES})


# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libcompressibleTransportModels.so libturbulenceModels.so
libfiniteVolume.so libmeshTools.so )

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfCanteraMixture
dfChemistryModel
${MPI_LIBRARIES}
${OpenFOAM_LIB_DIR}/$ENV{FOAM_MPI}/libPstream.so
)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
