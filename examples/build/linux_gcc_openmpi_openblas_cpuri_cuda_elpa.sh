#!/bin/bash
#SBATCH -p 48cp3
#SBATCH -J install
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../log_cuda
#SBATCH --error=../err_cuda

ulimit -s unlimited
ulimit -c unlimited

unset CPATH
module purge

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3

source /data/home/renxg/app/nvhpc/setup_nvhpc
TOOL=~/app/toolchain/260328/toolchain
INSTALL_DIR=$TOOL/install
SETUP_DIR=$TOOL/build

source $SETUP_DIR/setup_openblas_extern
source $SETUP_DIR/setup_scalapack_extern
source $SETUP_DIR/setup_cereal_extern
source $SETUP_DIR/setup_elpa_extern

LibDDLA_PATH=~/app/libddla/260522/LibDDLA-1_install
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH

PREFIX=./
LAPACK=$INSTALL_DIR/openblas-0.3.29/lib
SCALAPACK=$INSTALL_DIR/scalapack-2.2.2/lib
CEREAL=$INSTALL_DIR/cereal-master/include
LIBRI=~/app/libri/260415/LibRI-master
LIBCOMM=~/app/libcomm/260521/LibComm-fix_status
ELPA_DIR=~/app/toolchain/260328/toolchain/install/elpa-2025.01.001/nvidia

echo "========================="
echo 'LD_LIBRARY_PATH:' $LD_LIBRARY_PATH
echo "========================="
echo 'PATH:' $PATH
echo "========================="
echo 'CPATH:' $CPATH
echo "========================="
echo 'C_INCLUDE_PATH:' $C_INCLUDE_PATH
echo "========================="
echo 'LIBRARY_PATH:' $LIBRARY_PATH
echo "========================="
echo 'CPLUS_INCLUDE_PATH:' $CPLUS_INCLUDE_PATH
echo "========================="

export FCFLAGS="-fPIC -g -fallow-argument-mismatch -ffree-line-length-none"

BUILD_DIR=../build_cuda
INSTALL_DIR=../librpa_cuda
echo Start Time: `date`
# rm -rf $BUILD_DIR
rm -rf $INSTALL_DIR
cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DCMAKE_CXX_COMPILER=g++ \
        -DMPI_CXX_COMPILER=mpicxx \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -DSCALAPACK_DIR=$SCALAPACK \
        -DCEREAL_INCLUDE_DIR=$CEREAL \
        -DLIBRPA_USE_LIBRI=ON \
        -DLIBRI_INCLUDE_DIR=$LIBRI/include \
        -DLIBCOMM_INCLUDE_DIR=$LIBCOMM/include \
        -DCMAKE_CXX_FLAGS="-g -O2 -fopenmp -Wunused-result -Wterminate" \
        -DUSE_GREENX_API=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBRPA_VERBOSE_OUTPUT=ON \
        -DLIBRPA_USE_CUDA=ON \
        -DLIBRPA_USE_EXTERNAL_ELPA=ON \
        -DEXTERNAL_ELPA_DIR=${ELPA_DIR} \
        -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON \
        -DCMAKE_Fortran_FLAGS="$FCFLAGS"


cmake --build $BUILD_DIR -j 8

cmake --install $BUILD_DIR --prefix $INSTALL_DIR
echo End Time: `date`
