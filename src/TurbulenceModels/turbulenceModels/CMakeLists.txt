project(dfTurbulenceModels LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

include_directories(
    ${dfturbulence_inc}
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude 
)

SET(SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/makeTurbulentFluidThermoModels.C
)
add_library(dfTurbulenceModels SHARED ${SOURCES})


# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libcompressibleTransportModels.so 
libturbulenceModels.so
libfiniteVolume.so 
libmeshTools.so )

target_link_libraries(${PROJECT_NAME} 
dfCompressibleTurbulenceModels
dfFluidThermophysicalModels
)




install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
