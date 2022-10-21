#!/bin/sh

# if [ -z "$CONDA_PREFIX" ]; then
#     echo "You should run this script only when the conda enviorment including libcantera-devel activated."
#     return
# fi

print_usage() {
    #printf "Usage: ...\n"
    echo "Usage: . install.sh --libtorch_dir _path_to_libtorch | --libtorch_autodownload | --libtorch_no (default)"
}

# default
LIBTORCH_NO=true
LIBTORCH_DIR=''
LIBTORCH_AUTO=false

while test $# -gt 0; do
    case "$1" in
        -h|--help)
            print_usage
            return
            ;;
        --libtorch_dir)
            shift
            if test $# -gt 0; then
                LIBTORCH_DIR=$1
                LIBTORCH_NO=false
                echo LIBTORCH_DIR = $LIBTORCH_DIR
            else
                print_usage
            return
            fi
            shift
            ;;
        --libtorch_autodownload)
            LIBTORCH_AUTO=true
            LIBTORCH_DIR="$PWD/thirdParty/libtorch"
            LIBTORCH_NO=false
            shift
            ;;
        --libtorch_no)
            shift
            ;;
        --libcantera_dir)
            shift
            if test $# -gt 0; then
                LIBCANTERA_DIR=$1
                echo LIBCANTERA_DIR = $LIBCANTERA_DIR
            else
                print_usage
            return
            fi
            shift
            ;;            
        *)
            echo "$1 is not a recognized flag!"
            print_usage
            return
            ;;
    esac
done



echo LIBTORCH_NO=$LIBTORCH_NO
echo LIBTORCH_DIR=$LIBTORCH_DIR
echo LIBTORCH_AUTO=$LIBTORCH_AUTO

if [ $LIBTORCH_AUTO = true ]; then
    if [ -d "thirdParty/libtorch" ]; then
        echo "libtorch already exist."
    else
        if [ -e libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip ]
        then
            echo "libtorch.zip exist."
        else
            wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
        fi
        unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty
    fi
fi


cp bashrc.in bashrc
sed -i s#pwd#$PWD#g ./bashrc
#echo "LIBCANTERA_DIR is set to $CONDA_PREFIX"
sed -i s#CONDA_PREFIX#$LIBCANTERA_DIR#g ./bashrc
sed -i s#LIBTORCH_DIR#$LIBTORCH_DIR#g ./bashrc





if [ -d "src_orig" ]; then
    echo "src_orig exist."
else
    mkdir -p src_orig/TurbulenceModels
    mkdir -p src_orig/thermophysicalModels
    mkdir -p src_orig/lagrangian
    mkdir -p src_orig/regionModels
    mkdir -p src_orig/functionObjects
    cp -r $FOAM_SRC/TurbulenceModels/compressible src_orig/TurbulenceModels
    cp -r $FOAM_SRC/thermophysicalModels/basic src_orig/thermophysicalModels
    cp -r $FOAM_SRC/thermophysicalModels/thermophysicalProperties src_orig/thermophysicalModels
    cp -r $FOAM_SRC/lagrangian/intermediate src_orig/lagrangian
    cp -r $FOAM_SRC/lagrangian/turbulence src_orig/lagrangian
    cp -r $FOAM_SRC/regionModels/surfaceFilmModels src_orig/regionModels
    cp -r $FOAM_SRC/functionObjects/field src_orig/functionObjects
fi


source ./bashrc
./Allwmake -j