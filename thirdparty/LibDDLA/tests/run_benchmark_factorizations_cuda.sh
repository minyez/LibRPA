#!/bin/bash
#SBATCH -p v100g32
#SBATCH -J factor4_hpd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/data/home/renxg/app/log_factor4_hpd_%j.out
#SBATCH --error=/data/home/renxg/app/log_factor4_hpd_%j.err

set -euo pipefail

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3
source /data/home/renxg/app/nvhpc/setup_nvhpc

REPO=/data/home/renxg/app/github/LibDDLA
BUILD=/data/home/renxg/app/codex_factorizations_cuda_build
SDK=/data/group_home/renxg/nvidia/hpc_sdk/Linux_x86_64/25.5
MATH=${SDK}/math_libs/11.8/targets/x86_64-linux
CUDA=${SDK}/cuda/11.8/targets/x86_64-linux
NCCL=${SDK}/comm_libs/11.8/nccl
EXE=${BUILD}/benchmark_factorizations_cuda

echo "Node: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Begin: $(date --iso-8601=seconds)"
echo "Repository: ${REPO}"
head_line=$(<"${REPO}/.git/HEAD")
if [[ ${head_line} == "ref: "* ]]; then
    head_ref=${head_line#ref: }
    if [[ -r "${REPO}/.git/${head_ref}" ]]; then
        commit=$(<"${REPO}/.git/${head_ref}")
    else
        commit=$(awk -v ref="${head_ref}" '$2 == ref { print $1; exit }' \
            "${REPO}/.git/packed-refs")
    fi
else
    commit=${head_line}
fi
echo "Commit: ${commit}"
echo "Compiler: $(mpicxx --version | head -n 1)"
echo "CUDA: $(nvcc --version | tail -n 1)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader

cmake -S "${REPO}" -B "${BUILD}" \
    -DDDLA_USE_CUDA=ON \
    -DDDLA_USE_CCL=ON \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build "${BUILD}" --target ddla_lib --parallel 12

mpicxx -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
    -DDDLA_USE_CUDA -DDDLA_USE_CCL \
    -I"${REPO}/include" \
    -I"${MATH}/include" \
    -I"${CUDA}/include" \
    "${REPO}/tests/benchmark_factorizations_cuda.cpp" \
    -o "${EXE}" \
    -L"${BUILD}/src" \
    -L"${MATH}/lib" \
    -L"${CUDA}/lib" \
    -L"${CUDA}/lib/stubs" \
    -L"${NCCL}/lib" \
    -Wl,-rpath,"${BUILD}/src" \
    -Wl,-rpath,"${MATH}/lib" \
    -Wl,-rpath,"${CUDA}/lib" \
    -Wl,-rpath,"${NCCL}/lib" \
    -lddla \
    -lcusolverMp \
    -lcal \
    -lcurand \
    -lcusolver \
    -lcublas \
    -lcublasLt \
    -lcudart \
    -lcuda \
    -lnvidia-ml \
    -lnccl \
    -lrt \
    -ldl \
    -lpthread \
    -lm

echo "Executable dependencies:"
ldd "${EXE}"
if ldd "${EXE}" | grep -q "not found"; then
    echo "ERROR: unresolved executable dependency"
    exit 1
fi

export OMPI_MCA_mpi_warn_on_fork=0
export OMP_NUM_THREADS=1

echo "Benchmark command: --warmup 500 --repeats 1 5000 10000 15000"
mpirun -n 4 --bind-to none --mca btl ^openib \
    "${EXE}" --warmup 500 --repeats 1 5000 10000 15000

echo "End: $(date --iso-8601=seconds)"
