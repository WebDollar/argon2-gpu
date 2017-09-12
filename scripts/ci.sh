case $COMPILER in
    gcc)
        export CC=gcc CXX=g++
        ;;
    clang)
        export CC=clang CXX=clang++
        ;;
    *)
        echo "ERROR: Invalid compiler: $COMPILER" 1>&2
        exit 1
esac

case $CUDA in
    cuda)
        NO_CUDA=TRUE
        ;;
    nocuda)
        NO_CUDA=FALSE
        ;;
    *)
        echo "ERROR: Invalid CUDA mode: $COMPILER" 1>&2
        exit 1
esac

mkdir -p build/$COMPILER-$CUDA || exit 1
cd build/$COMPILER-$CUDA || exit 1
cmake -DNO_CUDA=$NO_CUDA ../.. || exit 1
make || exit 1
#CTEST_OUTPUT_ON_FAILURE=1 make test || exit 1
