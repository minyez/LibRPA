#!/bin/bash
#SBATCH -p v100g32fat
##SBATCH --nodelist gpu005
#SBATCH -J test
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=3
#SBATCH --output=../../log_cuda
#SBATCH --error=../../err_cuda

set -e


module load gcc/11.3.0

module load openmpi/4.1.8-cuda
module load cmake/3.25.3

source /data/home/renxg/app/nvhpc/setup_nvhpc

cd ..
LibDDLA_SRC_PATH="${PWD}"
LibDDLA_PATH="${PWD}_install"
BUILD_DIR=${BUILD_DIR:-"${LibDDLA_SRC_PATH}/build_codex_grid_sweep"}
CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:-"70;80"}
BUILD_JOBS=${BUILD_JOBS:-2}
cd tests
export CPATH=$LibDDLA_SRC_PATH/include:$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$BUILD_DIR/src:$LibDDLA_PATH/lib:$LD_LIBRARY_PATH


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

echo "任务运行节点列表: ${SLURM_NODELIST}"
echo Job id : ${SLURM_JOB_ID}
echo "CMake build dir: ${BUILD_DIR}"


echo Begin Time: `date`
### * * * Running the tasks * * * ###

legacy_np=${LEGACY_NP:-4}
api_grid_default_np=${API_GRID_DEFAULT_NP:-4}
api_grid_external_specs=("2:1x2" "4:2x2" "6:2x3")
api_subcomm_np=${API_SUBCOMM_NP:-4}
api_subcomm_timeout=${API_SUBCOMM_TIMEOUT:-120s}
export OMPI_MCA_mpi_warn_on_fork=${OMPI_MCA_mpi_warn_on_fork:-0}

mpi_extra_args=()
if [ -n "${MPI_EXTRA_ARGS:-}" ]; then
    mpi_extra_args=(${MPI_EXTRA_ARGS})
fi

files=(
    # "test_pgemm_min"
    # "test_sv_gemm"
    "test_sv_gemm_hermitian_positive"
    # "test_aware"
    # "test_pgeadd"
    # "test_potrf_solvermp"
    # "test_potrf_potrs"
    # test_pgetrf_bpiv
    # test_pzgemm
    # test_getrf_nopiv
)

api_grid_files=(
    "test_api_grid_device_memory"
    "test_api_grid_pgemm"
    "test_api_grid_pgeadd"
    "test_api_grid_ptran"
    "test_api_grid_transport_block"
    "test_api_grid_pdam"
    "test_api_grid_pswap"
    "test_api_grid_plapiv"
    "test_api_grid_ptrtrs"
    "test_api_grid_pgetrf"
    "test_api_grid_pgetrs"
    "test_api_grid_pgesv"
    "test_api_grid_pgetrf_nopiv"
    "test_api_grid_pgetrs_nopiv"
    "test_api_grid_pgesv_nopiv"
    "test_api_grid_pgetrf_bpiv"
    "test_api_grid_pgetf2"
    "test_api_grid_pgetf2_panel"
    "test_api_grid_getrf_nopiv"
    "test_api_grid_ppotrf"
    "test_api_grid_ppotrs"
    "test_api_grid_pposv"
)

api_subcomm_files=(
    "test_api_grid_subcomm_lu"
)

run_cuda_test()
{
    local np=$1
    local FILENAME=$2
    shift 2
    if [ ! -x "./${FILENAME}" ]; then
        echo "ERROR: missing executable ./${FILENAME}"
        exit 1
    fi
    echo "📊 NP: $np"
    echo "▶️ Running..."
    export OMPI_MCA_btl_openib_allow_ib=1
    mpirun -n "$np" "${mpi_extra_args[@]}" "./${FILENAME}" "$@"
}

echo "================================================="
echo "Configuring and building CUDA tests..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${LibDDLA_SRC_PATH}" -DDDLA_USE_CUDA=ON -DDDLA_USE_CCL=ON -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
make -j"${BUILD_JOBS}" "${files[@]}" "${api_grid_files[@]}" "${api_subcomm_files[@]}"
cd "${BUILD_DIR}/tests"

# 遍历数组中的每一个文件
for FILENAME in "${files[@]}"; do
    echo "================================================="
    echo "🚀 Running: ${FILENAME}"
    run_cuda_test "${legacy_np}" "${FILENAME}"

    echo "✅ Finished: ${FILENAME}"
    echo "" # 空一行，方便看日志
done

echo "================================================="
echo "Running API subcommunicator tests with ${api_subcomm_np} ranks"

for FILENAME in "${api_subcomm_files[@]}"; do
    echo "================================================="
    echo "🚀 Running: ${FILENAME}"
    if [ ! -x "./${FILENAME}" ]; then
        echo "ERROR: missing executable ./${FILENAME}"
        exit 1
    fi
    echo "📊 NP: ${api_subcomm_np}"
    echo "⏱️ Timeout: ${api_subcomm_timeout}"
    export OMPI_MCA_btl_openib_allow_ib=1
    timeout --signal=TERM --kill-after=30s "${api_subcomm_timeout}" \
        mpirun -n "${api_subcomm_np}" "${mpi_extra_args[@]}" "./${FILENAME}"
    echo "✅ Finished: ${FILENAME}"
    echo ""
done

echo "================================================="
echo "Running API grid tests with default grid sweep and explicit grids: ${api_grid_external_specs[*]}"

for FILENAME in "${api_grid_files[@]}"; do
    echo "================================================="
    echo "🚀 Running: ${FILENAME}"

    echo "▶️ Default grid sweep: ${FILENAME}"
    run_cuda_test "${api_grid_default_np}" "${FILENAME}"

    for grid_spec in "${api_grid_external_specs[@]}"; do
        api_grid_external_np="${grid_spec%%:*}"
        api_grid_external_grid="${grid_spec#*:}"
        echo "▶️ External grid: ${FILENAME} --grid ${api_grid_external_grid}"
        run_cuda_test "${api_grid_external_np}" "${FILENAME}" --grid "${api_grid_external_grid}"
    done

    echo "✅ Finished: ${FILENAME}"
    echo "" # 空一行，方便看日志
done

echo "================================================="
echo "All tests finished."

echo End Time: `date`
