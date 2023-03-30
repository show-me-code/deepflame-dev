project(dfLagrangianIntermediate LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

include_directories(
    ${OpenFOAM_SRC}/lagrangian/basic/lnInclude 
    ${OpenFOAM_SRC}/lagrangian/distributionModels/lnInclude 
    ${dflagrangianinter_inc} 
    ${OpenFOAM_SRC}/lagrangian/intermediate/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc} 
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${dfthermophysicalslg_inc} 
    ${CANTERA_ROOT}/include
    ${dfcantera_inc} 
    ${OpenFOAM_SRC}/transportModels 
    ${OpenFOAM_SRC}/regionModels/regionModel/lnInclude 
    ${OpenFOAM_SRC}/regionModels/surfaceFilmModels/lnInclude 
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude 
    ${OpenFOAM_SRC}/sampling/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude

)

SET(workDir ${SRC_ORIG}/lagrangian/intermediate)
SET(PARCELS ${CMAKE_CURRENT_SOURCE_DIR}/parcels)
SET(BASEPARCELS ${PARCELS}/baseClasses)
SET(DERIVEDPARCELS ${PARCELS}/derived)

SET(CLOUDS ${CMAKE_CURRENT_SOURCE_DIR}/clouds)
SET(BASECLOUDS ${CLOUDS}/baseClasses)
SET(DERIVEDCLOUDS ${CLOUDS}/derived)
SET(KINEMATICPARCEL ${DERIVEDPARCELS}/basicKinematicParcel)
SET(KINEMATICCOLLIDINGPARCEL  ${DERIVEDPARCELS}/basicKinematicCollidingParcel)
SET(THERMOPARCEL  ${DERIVEDPARCELS}/basicThermoParcel)
SET(REACTINGPARCEL  ${DERIVEDPARCELS}/basicReactingParcel)
SET(REACTINGMPPARCEL  ${DERIVEDPARCELS}/basicReactingMultiphaseParcel)
SET(KINEMATICMPPICPARCEL  ${DERIVEDPARCELS}/basicKinematicMPPICParcel)
SET(KINEMATICINJECTION  ${workDir}/submodels/Kinematic/InjectionModel)
SET(THERMOINJECTION  ${workDir}/submodels/Thermodynamic/InjectionModel)
SET(REACTINGMPINJECTION  ${workDir}/submodels/ReactingMultiphase/InjectionModel)
SET(REACTINGINJECTION  ${workDir}/submodels/Reacting/InjectionModel)
SET(MPPICCORRECTIONLIMITING  ${workDir}/submodels/MPPIC/CorrectionLimitingMethods)
SET(MPPICPARTICLESTRESS  ${workDir}/submodels/MPPIC/ParticleStressModels)
SET(MPPICCORRECTIONLIMITING  ${workDir}/submodels/MPPIC/CorrectionLimitingMethods)
SET(MPPICTIMESCALE  ${workDir}/submodels/MPPIC/TimeScaleModels)

SET(SOURCES 
${BASECLOUDS}/kinematicCloud/kinematicCloud.C
${BASECLOUDS}/thermoCloud/thermoCloud.C
${BASECLOUDS}/reactingCloud/reactingCloud.C
${BASECLOUDS}/reactingMultiphaseCloud/reactingMultiphaseCloud.C

${KINEMATICPARCEL}/defineBasicKinematicParcel.C
${KINEMATICPARCEL}/makeBasicKinematicParcelSubmodels.C

${KINEMATICCOLLIDINGPARCEL}/defineBasicKinematicCollidingParcel.C
${KINEMATICCOLLIDINGPARCEL}/makeBasicKinematicCollidingParcelSubmodels.C



${THERMOPARCEL}/defineBasicThermoParcel.C
${THERMOPARCEL}/makeBasicThermoParcelSubmodels.C



${REACTINGPARCEL}/defineBasicReactingParcel.C
${REACTINGPARCEL}/makeBasicReactingParcelSubmodels.C



${REACTINGMPPARCEL}/defineBasicReactingMultiphaseParcel.C
${REACTINGMPPARCEL}/makeBasicReactingMultiphaseParcelSubmodels.C



${KINEMATICMPPICPARCEL}/defineBasicKinematicMPPICParcel.C
${KINEMATICMPPICPARCEL}/makeBasicKinematicMPPICParcelSubmodels.C



${workDir}/submodels/Kinematic/PatchInteractionModel/LocalInteraction/patchInteractionData.C
${workDir}/submodels/Kinematic/PatchInteractionModel/LocalInteraction/patchInteractionDataList.C


${KINEMATICINJECTION}/KinematicLookupTableInjection/kinematicParcelInjectionData.C
${KINEMATICINJECTION}/KinematicLookupTableInjection/kinematicParcelInjectionDataIO.C
${KINEMATICINJECTION}/KinematicLookupTableInjection/kinematicParcelInjectionDataIOList.C
${KINEMATICINJECTION}/PatchInjection/patchInjectionBase.C


${THERMOINJECTION}/ThermoLookupTableInjection/thermoParcelInjectionData.C
${THERMOINJECTION}/ThermoLookupTableInjection/thermoParcelInjectionDataIO.C
${THERMOINJECTION}/ThermoLookupTableInjection/thermoParcelInjectionDataIOList.C


${REACTINGINJECTION}/ReactingLookupTableInjection/reactingParcelInjectionData.C
${REACTINGINJECTION}/ReactingLookupTableInjection/reactingParcelInjectionDataIO.C
${REACTINGINJECTION}/ReactingLookupTableInjection/reactingParcelInjectionDataIOList.C


${REACTINGMPINJECTION}/ReactingMultiphaseLookupTableInjection/reactingMultiphaseParcelInjectionData.C
${REACTINGMPINJECTION}/ReactingMultiphaseLookupTableInjection/reactingMultiphaseParcelInjectionDataIO.C
${REACTINGMPINJECTION}/ReactingMultiphaseLookupTableInjection/reactingMultiphaseParcelInjectionDataIOList.C


${MPPICPARTICLESTRESS}/ParticleStressModel/ParticleStressModel.C
${MPPICPARTICLESTRESS}/HarrisCrighton/HarrisCrighton.C
${MPPICPARTICLESTRESS}/Lun/Lun.C
${MPPICPARTICLESTRESS}/exponential/exponential.C


${MPPICCORRECTIONLIMITING}/CorrectionLimitingMethod/CorrectionLimitingMethod.C
${MPPICCORRECTIONLIMITING}/noCorrectionLimiting/noCorrectionLimiting.C
${MPPICCORRECTIONLIMITING}/absolute/absolute.C
${MPPICCORRECTIONLIMITING}/relative/relative.C


${MPPICTIMESCALE}/TimeScaleModel/TimeScaleModel.C
${MPPICTIMESCALE}/equilibrium/equilibrium.C
${MPPICTIMESCALE}/nonEquilibrium/nonEquilibrium.C
${MPPICTIMESCALE}/isotropic/isotropic.C


${workDir}/integrationScheme/integrationScheme/integrationScheme.C
${workDir}/integrationScheme/integrationScheme/integrationSchemeNew.C
${workDir}/integrationScheme/Euler/Euler.C
${workDir}/integrationScheme/analytical/analytical.C


${workDir}/phaseProperties/phaseProperties/phaseProperties.C
${workDir}/phaseProperties/phaseProperties/phasePropertiesIO.C
${workDir}/phaseProperties/phasePropertiesList/phasePropertiesList.C


${CMAKE_CURRENT_SOURCE_DIR}/clouds/Templates/KinematicCloud/cloudSolution/cloudSolution.C



${workDir}/submodels/MPPIC/AveragingMethods/makeAveragingMethods.C
)

add_library(dfLagrangianIntermediate SHARED ${SOURCES})


# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libcompressibleTransportModels.so 
libregionModels.so
libfiniteVolume.so 
libmeshTools.so 
liblagrangian.so 
libdistributionModels.so 
libincompressibleTransportModels.so 
libdynamicFvMesh.so 
libsampling.so
libmeshTools.so
)

target_link_libraries(${PROJECT_NAME} 
dfCompressibleTurbulenceModels
dfCanteraMixture
dfChemistryModel
dfFluidThermophysicalModels
dfThermophysicalProperties
dfSLGThermo
dfSurfaceFilmModels
)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
