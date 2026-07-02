#!/bin/bash
#SBATCH -p normal
##SBATCH --nodelist gpu007
#SBATCH -J install
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../log_hip
#SBATCH --error=../err_hip

ulimit -s unlimited
ulimit -c unlimited

unset CPATH
module purge

module load compiler/rocm/dtk/25.04.3
export CPATH=$ROCM_PATH/include/rocrand:$CPATH
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh
module load compiler/cmake/3.24.1

TOOL=/public/home/hbchen/app/LibRPA/260212/toolchain
INSTALL_DIR=$TOOL/install
SETUP_DIR=$TOOL/build
source $SETUP_DIR/setup_openblas_extern
source $SETUP_DIR/setup_scalapack_extern
source $SETUP_DIR/setup_cereal_extern


source /public/home/hbchen/app/elpa/260611/setup_elpa

LibDDLA_PATH=/public/home/hbchen/app/LibDDLA/260603/LibDDLA_install
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH



PREFIX=./
LAPACK=$INSTALL_DIR/openblas-0.3.29/lib
SCALAPACK=$INSTALL_DIR/scalapack-2.2.2/lib
CEREAL=$INSTALL_DIR/cereal-master/include
LIBRI=/public/home/hbchen/app/libri/260214/LibRI-master
# LIBCOMM=/public/home/hbchen/app/libcomm/260516/LibComm-fix_status
ELPA_DIR=/public/home/hbchen/app/elpa/260611/elpa-2026.02.001_install_hip_rccl

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

BUILD_DIR=../build_hip_cpuri
INSTALL_DIR=../librpa_hip_cpuri
echo Start Time: `date`
rm -rf $BUILD_DIR

rm -rf $INSTALL_DIR
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DCMAKE_CXX_COMPILER=g++ \
        -DMPI_CXX_COMPILER=mpicxx \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -DSCALAPACK_DIR=$SCALAPACK \
        -DCEREAL_INCLUDE_DIR=$CEREAL \
        -DLIBRPA_USE_LIBRI=ON \
        -DLIBRI_INCLUDE_DIR=$LIBRI/include \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBRPA_VERBOSE_OUTPUT=ON\
        -DCMAKE_CXX_FLAGS="-g -O2"\
        -DLIBRPA_USE_HIP=ON \
        -DLIBRPA_USE_EXTERNAL_ELPA=ON \
        -DEXTERNAL_ELPA_DIR=${ELPA_DIR} \
        -DUSE_GREENX_API=ON \
        -DLIBDDLA_PATH=${LibDDLA_PATH}


        # -DCMAKE_HIP_FLAGS="-g -O2 -fopenmp -fgpu-rdc -Wno-return-type"
        # -DLIBCOMM_INCLUDE_DIR=$LIBCOMM/include \



        # -DCMAKE_HIP_SEPARABLE_COMPILATION=ON \
        # -DBLAS_DIR=$LAPACK \
        # -DLAPACK_DIR=$LAPACK \
cmake --build $BUILD_DIR -j 8
cmake --install $BUILD_DIR --prefix $INSTALL_DIR
echo End Time: `date`
