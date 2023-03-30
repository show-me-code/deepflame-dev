project(dfFluidThermophysicalModels  LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

# add library
SET(workDir
    $ENV{SRC_ORIG}/thermophysicalModels/basic/)

SET(SOURCES
    ${workDir}/basicThermo/basicThermo.C
    ${workDir}/fluidThermo/fluidThermo.C

    ${workDir}/psiThermo/psiThermo.C

    ${workDir}/rhoThermo/rhoThermo.C

    ${workDir}/derivedFvPatchFields/fixedEnergy/fixedEnergyFvPatchScalarField.C
    ${workDir}/derivedFvPatchFields/gradientEnergy/gradientEnergyFvPatchScalarField.C
    ${workDir}/derivedFvPatchFields/mixedEnergy/mixedEnergyFvPatchScalarField.C

    ${workDir}/derivedFvPatchFields/energyJump/energyJump/energyJumpFvPatchScalarField.C
    ${workDir}/derivedFvPatchFields/energyJump/energyJumpAMI/energyJumpAMIFvPatchScalarField.C
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

# 
include_directories(
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${DF_SRC}/thermophysicalModels/thermophysicalProperties/solidProperties/solidMixtureProperties
    ${DF_SRC}/thermophysicalModels/thermophysicalProperties/solidProperties/solidProperties
)

target_link_libraries(${PROJECT_NAME} 
    ${OpenFOAM_LIB_DIR}/libfiniteVolume.so libmeshTools.so libcompressibleTransportModels.so
    ${CANTERA_ROOT}/lib/libcantera.so.2
    dfThermophysicalProperties
)


# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
