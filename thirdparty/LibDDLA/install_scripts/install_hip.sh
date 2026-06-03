#!/bin/bash
#SBATCH -p normal
##SBATCH --nodelist gpu007
#SBATCH -J install
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --output=../log_hip
#SBATCH --error=../err_hip

ulimit -s unlimited
ulimit -c unlimited

unset CPATH
module purge


module load compiler/rocm/dtk/25.04.3
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LIBRARY_PATH
module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh

module load compiler/cmake/3.24.1

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

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

export OMPI_CXX=$CXX
export OMPI_CC=$CC
export OMPI_FC=$FC
# echo 'ROCM_PATH：' $ROCM_PATH

echo Begin Time: `date`
### * * * Running the tasks * * * ###
BUILD_DIR=../build_hip
INSTALL_DIR="${PWD}_install"
# cd install_scripts
echo 'Build Dir:' $BUILD_DIR
echo 'Install Dir:' $INSTALL_DIR
echo "任务运行节点列表: ${SLURM_NODELIST}"
# rm -rf ${BUILD_DIR}
rm -rf ${INSTALL_DIR}
cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DROCM_PATH=$ROCM_PATH \
        -DDDLA_USE_HIP=ON \
        -DCMAKE_PREFIX_PATH=$ROCM_PATH \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_FLAGS="-g -O2 -fopenmp -Wno-return-type" \
        -DDDLA_USE_CCL=ON \
        -DDDLA_USE_GPU_CPU_TUNNEL=ON

        # -DDDLA_USE_DEBUG=ON \
        # -DMPI_CXX_COMPILER=mpicxx \
        # -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
        # -DCMAKE_HIP_COMPILER=hipcc \
        # -DCMAKE_Fortran_COMPILER=gfortran \
        
        # -DBUILD_TESTS=ON \

cmake --build $BUILD_DIR -j `nproc` 

cmake --install $BUILD_DIR --prefix $INSTALL_DIR

# cd ${BUILD_DIR}
# make test
echo End Time: `date`
