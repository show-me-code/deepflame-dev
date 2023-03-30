cmake_minimum_required(VERSION 3.5)
project(deepflame LANGUAGES CXX)

# find packages
FIND_PACKAGE(MPI REQUIRED)

# Check valid thirdparty
if(DEFINED ENV{WM_PROJECT_DIR})
	MESSAGE(STATUS "OpenFOAM: " $ENV{WM_PROJECT_DIR})
else()
	message(FATAL_ERROR "OpenFOAM is not sourced")
endif(DEFINED ENV{WM_PROJECT_DIR})

if(DEFINED ENV{CANTERA_ROOT})
	MESSAGE(STATUS "libcantera: " $ENV{CANTERA_ROOT})
  SET(CANTERA_ROOT $ENV{CANTERA_ROOT})
else()
	message(FATAL_ERROR "libcantera directory is not specified")
endif(DEFINED ENV{CANTERA_ROOT}) 

# define variables
SET(OpenFOAM_LIB_DIR $ENV{FOAM_LIBBIN})
SET(OpenFOAM_SRC $ENV{FOAM_SRC})

SET(SRC_ORIG $ENV{SRC_ORIG})
SET(DF_ROOT $ENV{DF_ROOT})
SET(DF_SRC $ENV{DF_SRC})

# set compilation options
SET(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=bfd -shared -Xlinker --add-needed -Xlinker --no-as-needed")
SET(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=bfd -Xlinker --add-needed -Xlinker --no-as-needed")

SET(CMAKE_C_COMPILER g++)
SET(PATH_LIB_OPENMPI "openmpi-system")  # Foundation version
SET(EXE_COMPILE_OPTION "-std=c++11 -m64 -Dlinux64 -DWM_ARCH_OPTION=64 
-DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor 
-Wno-unused-parameter -Wno-invalid-offsetof -Wno-attributes -O3  
-DNoRepository -ftemplate-depth-100 -std=c++14 
-Wno-unused-variable -Wno-unused-but-set-variable -Wno-old-style-cast -DOMPI_SKIP_MPICXX  
-pthread -fPIC")
SET(SHARED_COMPILE_OPTION "-std=c++11 -m64 -Dlinux64 -DWM_ARCH_OPTION=64 
-DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor 
-Wno-unused-parameter -Wno-invalid-offsetof -Wno-attributes -O3  
-DNoRepository -ftemplate-depth-100 -std=c++14 
-Wno-unused-variable -Wno-unused-but-set-variable -Wno-old-style-cast -DOMPI_SKIP_MPICXX  
-pthread -fPIC -c")

# find header files
FUNCTION(R_SEARCH search_path return_list)
    FILE(GLOB_RECURSE new_list ${search_path}/*.H)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list} PARENT_SCOPE)
ENDFUNCTION(R_SEARCH)

R_SEARCH(${DF_SRC}/dfCombustionModels dfcombustion_inc)
R_SEARCH(${DF_SRC}/dfCanteraMixture dfcantera_inc)
R_SEARCH(${DF_SRC}/lagrangian/intermediate dflagrangianinter_inc)
R_SEARCH(${DF_SRC}/lagrangian/spray dflagrangianspray_inc)
R_SEARCH(${DF_SRC}/lagrangian/turbulence dflagrangianturb_inc)
R_SEARCH(${DF_SRC}/dfChemistryModel dfchemistry_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/thermophysicalProperties dfthermophysicalprop_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/thermophysicalProperties dfthermophysicalprop_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/basic dfthermophysicalbasic_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/SLGThermo dfthermophysicalslg_inc)
R_SEARCH(${DF_SRC}/TurbulenceModels dfturbulence_inc)
R_SEARCH(${DF_SRC}/dynamicMesh dfnewdynamic_inc)
R_SEARCH(${DF_SRC}/dynamicFvMesh dffvdynamic_inc)

include_directories(
    ${OpenFOAM_SRC}/finiteVolume/lnInclude
    
    ${OpenFOAM_SRC}/OSspecific/POSIX/lnInclude
    ${OpenFOAM_SRC}/OpenFOAM/lnInclude
)

# add subdirectory
SET(CMAKE_INSTALL_PREFIX ${DF_ROOT})
add_subdirectory(${DF_SRC}/thermophysicalModels/thermophysicalProperties)
add_subdirectory(${DF_SRC}/thermophysicalModels/basic)
add_subdirectory(${DF_SRC}/functionObjects/field)
add_subdirectory(${DF_SRC}/dfCanteraMixture)
add_subdirectory(${DF_SRC}/thermophysicalModels/SLGThermo)
add_subdirectory(${DF_SRC}/dfChemistryModel)
add_subdirectory(${DF_SRC}/TurbulenceModels/compressible)
add_subdirectory(${DF_SRC}/TurbulenceModels/turbulenceModels)
add_subdirectory(${DF_SRC}/regionModels/surfaceFilmModels)
add_subdirectory(${DF_SRC}/lagrangian/turbulence)
add_subdirectory(${DF_SRC}/lagrangian/spray)
add_subdirectory(${DF_SRC}/lagrangian/intermediate)
add_subdirectory(${DF_SRC}/dfCombustionModels)
add_subdirectory(${DF_SRC}/dynamicMesh)
add_subdirectory(${DF_SRC}/dynamicFvMesh)
