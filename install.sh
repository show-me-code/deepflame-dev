#!/bin/sh

print_finish() {
    if [ $USE_LIBTORCH = true ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and libtorch) compiled successfully! Enjoy!! |"
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        return
    fi
    if [ $USE_PYTORCH = true ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and pytorch) compiled successfully! Enjoy!! | "
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
	return
    fi
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
    echo "| deepflame (linked with libcantera) compiled successfully! Enjoy!! |"
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
}

./Allwmake -j && print_finish
