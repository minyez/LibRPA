#!/bin/bash
#SBATCH -p v100g32
#SBATCH --nodelist=c2v100m2-9
#SBATCH -J cusolvermp_pzgetrf
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/renxg/app/log_cuda
#SBATCH --error=/data/home/renxg/app/err_cuda

set -eo pipefail

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3
source /data/home/renxg/app/nvhpc/setup_nvhpc

SOURCE_DIR=/data/home/renxg/app/github/LibDDLA/tests
BUILD_DIR=/data/home/renxg/app/cusolvermp_pzgetrf_benchmark
SDK_ROOT=/data/group_home/renxg/nvidia/hpc_sdk/Linux_x86_64/25.5
MATH_ROOT=${SDK_ROOT}/math_libs/11.8/targets/x86_64-linux
CUDA_ROOT=${SDK_ROOT}/cuda/11.8/targets/x86_64-linux

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Node: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Begin: $(date)"

mpicc -O3 -std=c11 -Wall -Wextra -Wpedantic \
    -I"${MATH_ROOT}/include" \
    -I"${CUDA_ROOT}/include" \
    "${SOURCE_DIR}/benchmark_pzgetrf_legacy.c" \
    -o benchmark_pzgetrf_legacy \
    -L"${MATH_ROOT}/lib" \
    -L"${CUDA_ROOT}/lib" \
    -L"${CUDA_ROOT}/lib/stubs" \
    -Wl,-rpath,"${MATH_ROOT}/lib" \
    -Wl,-rpath,"${CUDA_ROOT}/lib" \
    -lcusolverMp \
    -lcal \
    -lcurand \
    -lcusolver \
    -lcublas \
    -lcublasLt \
    -lcudart \
    -lcuda \
    -lnvidia-ml \
    -lm

export OMPI_MCA_mpi_warn_on_fork=0
mpirun -n 4 --bind-to none ./benchmark_pzgetrf_legacy 500 5000 10000 15000

echo "End: $(date)"
