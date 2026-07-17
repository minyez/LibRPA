#!/bin/bash
#SBATCH -p v100g32fat
#SBATCH -J test_pgesv_nopiv
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --output=../../log_pgesv_nopiv
#SBATCH --error=../../err_pgesv_nopiv

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3

source /data/home/renxg/app/nvhpc/setup_nvhpc

cd /data/home/renxg/app/github/LibDDLA
export LD_LIBRARY_PATH=/data/app/gcc/11.3.0/lib64:/data/home/renxg/app/github/LibDDLA_install/lib:/data/group_home/renxg/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/11.8/nccl/lib:

echo 任务运行节点列表: 
echo Job id : 
echo Begin Time: Fri Jun 19 16:03:04 CST 2026

echo Running test_pgesv_nopiv with np=4

export OMPI_MCA_btl_openib_allow_ib=1
mpirun -n 4 --mca btl_tcp_if_include ib0,ib1 ./build/tests/test_pgesv_nopiv

echo End Time: Fri Jun 19 16:03:04 CST 2026
