project(dfCanteraMixture LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

include_directories(
                    ./
                    ${OpenFOAM_SRC}/meshTools/lnInclude
                    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
                    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
                    ${CANTERA_ROOT}/include
                    )


SET(SOURCES 
    CanteraMixture.C
    makeThermos.C
)

add_library(dfCanteraMixture SHARED ${SOURCES})


# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libfiniteVolume.so libmeshTools.so)
target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera.so.2)




install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
