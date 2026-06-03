#!/bin/bash
#SBATCH -p normal
##SBATCH --nodelist j14r3n06
#SBATCH -J test
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --output=./log_test
#SBATCH --error=./err_test

ulimit -s unlimited
ulimit -c unlimited


unset CPATH

module purge

module load compiler/rocm/dtk/25.04.3
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LIBRARY_PATH
module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh
cd ..
LibDDLA_PATH="${PWD}_install"
cd tests
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH

export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LIBRARY_PATH
export CPATH=$ROCM_PATH/include/rocrand:$CPATH
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

echo "任务运行节点列表: ${SLURM_NODELIST}"

echo Begin Time: `date`
### * * * Running the tasks * * * ###

FILENAME=test_pzgemm



mpicxx -gdwarf-4 -g -O2 -lamdhip64 -lgalaxyhip -lddla -fopenmp -lrccl -lhipblas -lhipsolver -lhiprand  ${FILENAME}.cpp -o ${FILENAME} -std=c++11 -DDDLA_USE_HIP -D__HIP_PLATFORM_AMD__ -DDDLA_USE_CCL
np=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
echo "np: $np"
# ldd ./${FILENAME}
mpirun -n $np ./${FILENAME} --mca btl ^openib
# mpirun -n $np ./${FILENAME}
echo End Time: `date`
