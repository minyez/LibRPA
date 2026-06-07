#!/bin/bash
#SBATCH -p 48cp3
##SBATCH --nodelist gpu005
#SBATCH -J test
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --output=../../log_install
#SBATCH --error=../../err_install

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3

source /data/home/renxg/app/nvhpc/setup_nvhpc


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


# export OMPI_CXX=$CXX
# export OMPI_CC=$CC
# export OMPI_FC=$FC


echo Begin Time: `date`
### * * * Running the tasks * * * ###
cd ..
BUILD_DIR=../build
INSTALL_DIR="${PWD}_install"

rm -rf ${BUILD_DIR}
rm -rf ${INSTALL_DIR}
# mkdir ${INSTALL_DIR}
cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DCMAKE_CXX_COMPILER=g++ \
        -DDDLA_USE_CUDA=ON \
        -DDDLA_USE_CCL=ON \


        # -DBUILD_TESTS=ON \
        # -DCMAKE_Fortran_COMPILER=gfortran \
        # -DMPI_CXX_COMPILER=mpicxx \

cmake --build $BUILD_DIR -j 8 

cmake --install $BUILD_DIR --prefix $INSTALL_DIR

# cd ${BUILD_DIR}
# make test
echo End Time: `date`
