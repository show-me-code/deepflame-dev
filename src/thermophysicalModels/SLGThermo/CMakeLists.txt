project(dfSLGThermo  LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

# add library
SET(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SLGThermo/SLGThermo.C)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc}
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${CANTERA_ROOT}/include
    ${dfcantera_inc}
    ${OpenFOAM_SRC}/finiteVolume/lnInclude
)



target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libfiniteVolume.so 
libmeshTools.so 
libcompressibleTransportModels.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera.so.2)

target_link_libraries(${PROJECT_NAME} 
dfThermophysicalProperties
dfCanteraMixture
)

# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
