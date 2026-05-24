#!/bin/bash
#SBATCH -p normal
##SBATCH --nodelist gpu007
#SBATCH -J install
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../../../log_install
#SBATCH --error=../../../err_install

ulimit -s unlimited
ulimit -c unlimited

unset CPATH
module purge

module load compiler/rocm/dtk/25.04
export CPATH=$ROCM_PATH/include/rocrand:$CPATH
# export LIBRARY_PATH=$ROCM_PATH/lib:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LIBRARY_PATH
# export PATH=$ROCM_PATH/llvm/bin:$PATH
# export LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh
module load compiler/cmake/3.24.1

TOOL=/public/home/hbchen/app/LibRPA/260212/toolchain
INSTALL_DIR=$TOOL/install
SETUP_DIR=$TOOL/build
source $SETUP_DIR/setup_openblas_extern
source $SETUP_DIR/setup_scalapack_extern
source $SETUP_DIR/setup_cereal_extern

LibDDLA_PATH=/public/home/hbchen/app/LibDDLA/260524/LibDDLA-develop_install
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH

source /public/home/hbchen/app/elpa/260226/setup_elpa



PREFIX=./
LAPACK=$INSTALL_DIR/openblas-0.3.29/lib
SCALAPACK=$INSTALL_DIR/scalapack-2.2.2/lib
CEREAL=$INSTALL_DIR/cereal-master/include
LIBRI=/public/home/hbchen/app/libri/260214/LibRI-master
LIBCOMM=/public/home/hbchen/app/libcomm/260516/LibComm-fix_status

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

cd ../../
BUILD_DIR=../build_hip
INSTALL_DIR=../librpa_hip
echo Start Time: `date`
# rm -rf $BUILD_DIR

rm -rf $INSTALL_DIR
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DMPI_CXX_COMPILER=mpicxx \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -DSCALAPACK_DIR=$SCALAPACK \
        -DCEREAL_INCLUDE_DIR=$CEREAL \
        -DUSE_LIBRI=ON \
        -DLIBRI_INCLUDE_DIR=$LIBRI/include \
        -DLIBCOMM_INCLUDE_DIR=$LIBCOMM/include \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBRPA_VERBOSE_OUTPUT=ON\
        -DCMAKE_CXX_FLAGS="-dwarf-4 -g -O2 -fopenmp -fgpu-rdc -Wno-implicit-interface -Wno-return-type -Wno-return-stack-address -Wno-format -Wno-unused-command-line-argument -Wno-format-security -Wno-exceptions" \
        -DLIBRPA_ENABLE_HIP=ON \
        -DUSE_GREENX_API=ON \
        -DLIBRPA_ENABLE_ELPA=ON \
        -DCMAKE_HIP_FLAGS="-g -O2 -fopenmp -fgpu-rdc -Wno-return-type"



        # -DCMAKE_HIP_SEPARABLE_COMPILATION=ON \
        
        # -DENABLE_ELPA=ON \
        # -DCMAKE_CUDA_ARCHITECTURES=80 \

        # -DBLAS_DIR=$LAPACK \
        # -DLAPACK_DIR=$LAPACK \
        # -DCUDA_NVCC_FLAGS="-x cu" \
        # -fpermissive
        # -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON \
cmake --build $BUILD_DIR -j `nproc`
cmake --install $BUILD_DIR --prefix $INSTALL_DIR
echo End Time: `date`
