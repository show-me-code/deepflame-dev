project(dfCompressibleTurbulenceModels LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

include_directories(
                    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
                    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
                    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
                    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
                    ${OpenFOAM_SRC}/meshTools/lnInclude
                    ${CANTERA_ROOT}/include
                    )



SET(WORKINGDIR ${SRC_ORIG}/TurbulenceModels/compressible/)
SET(SOURCES 
${WORKINGDIR}/compressibleTurbulenceModel.C
${WORKINGDIR}/turbulentFluidThermoModels/turbulentFluidThermoModels.C
${WORKINGDIR}/turbulentFluidThermoModels/derivedFvPatchFields/wallFunctions/alphatWallFunctions/alphatWallFunction/alphatWallFunctionFvPatchScalarField.C
)
add_library(dfCompressibleTurbulenceModels SHARED ${SOURCES})


# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libcompressibleTransportModels.so 
libturbulenceModels.so 
libfiniteVolume.so 
libmeshTools.so
)





install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
