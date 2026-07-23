#!/bin/bash
#SBATCH -p normal
#SBATCH -J build_cpu_belpa
#SBATCH --nodes=1
#SBATCH --gres=dcu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/public/home/hbchen/app/LibRPA/260531/log_openmpi_bundle_elpa
#SBATCH --error=/public/home/hbchen/app/LibRPA/260531/err_openmpi_bundle_elpa
#SBATCH --open-mode=truncate

set -eEo pipefail

export LC_ALL=C
export LANG=C
export LANGUAGE=C

ulimit -s unlimited
ulimit -c unlimited

unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset LIBRARY_PATH LD_LIBRARY_PATH LD_RUN_PATH
unset PKG_CONFIG_PATH CMAKE_PREFIX_PATH
unset ELPA_PATH ELPA_ROOT EXTERNAL_ELPA_DIR ELPA_DIR
unset ROCM_PATH HIP_PATH CUDA_HOME CUDA_PATH
module purge

module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh
module load compiler/cmake/3.24.1

tool_root=/public/home/hbchen/app/LibRPA/260212/toolchain
tool_install="${tool_root}/install"
tool_build="${tool_root}/build"
source "${tool_build}/setup_openblas_extern"
source "${tool_build}/setup_scalapack_extern"
source "${tool_build}/setup_cereal_extern"

repo_root=/public/home/hbchen/app/LibRPA/260531/LibRPA
work_root=/public/home/hbchen/app/LibRPA/260531
job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
build_dir="${work_root}/build_openmpi_elpa"
install_dir="${work_root}/librpa_openmpi_elpa"
stage_dir="${work_root}/.librpa_openmpi_bundle_elpa.stage.${job_id}"
expected_head=a6f1ded9ddf7175e46c80645457c3c4d37e5cc5b
expected_staged_sha=f7723491ab4926467ccdc35e733e09325fbeba2eae2e39da0af04437ad018a4d
empty_diff_sha=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
stage_promoted=0

cleanup_stage()
{
    status=$?
    if [[ "${status}" -ne 0 && "${stage_promoted}" -eq 0 && -d "${stage_dir}" ]]; then
        find "${stage_dir}" -depth -delete
    fi
    exit "${status}"
}
trap cleanup_stage EXIT

cd "${repo_root}"
current_head=$(git rev-parse HEAD)
staged_sha=$(git diff --cached --binary | sha256sum | awk '{print $1}')
unstaged_sha=$(git diff --binary | sha256sum | awk '{print $1}')
test "${current_head}" = "${expected_head}"
test "${staged_sha}" = "${expected_staged_sha}"
test "${unstaged_sha}" = "${empty_diff_sha}"
git diff --cached --check
git diff --check

for target in "${build_dir}" "${install_dir}" "${stage_dir}"; do
    if [[ -e "${target}" ]]; then
        echo "Refusing to overwrite existing path: ${target}" >&2
        exit 1
    fi
done

scalapack_dir="${tool_install}/scalapack-2.2.2/lib"
cereal_include="${tool_install}/cereal-master/include"
elpa_configure_args="--enable-option-checking=fatal --enable-single-precision --disable-cpp-tests"
install_rpath='$ORIGIN/../lib64;$ORIGIN/../lib'

echo "SLURM_JOB_ID=${job_id}"
echo "REPO_ROOT=${repo_root}"
echo "SOURCE_HEAD=${current_head}"
echo "STAGED_DIFF_SHA256=${staged_sha}"
echo "BUILD_DIR=${build_dir}"
echo "STAGE_DIR=${stage_dir}"
echo "INSTALL_DIR=${install_dir}"
echo "START_TIME=$(date --iso-8601=seconds)"

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_INSTALL_PREFIX="${stage_dir}" \
    -DCMAKE_INSTALL_RPATH="${install_rpath}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DMPI_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DMPI_Fortran_COMPILER=mpifort \
    -DSCALAPACK_DIR="${scalapack_dir}" \
    -DCEREAL_INCLUDE_DIR="${cereal_include}" \
    -DLIBRPA_USE_LIBRI=ON \
    -DLIBRPA_USE_LIBRI_GPU=OFF \
    -DLIBRPA_USE_CUDA=OFF \
    -DLIBRPA_USE_HIP=OFF \
    -DLIBRPA_USE_EXTERNAL_ELPA=OFF \
    -DLIBRPA_USE_BUNDLED_ELPA=ON \
    -DLIBRPA_BUNDLED_ELPA_VERSION=2026.02.001 \
    -DLIBRPA_BUNDLED_ELPA_OPENMP=ON \
    -DLIBRPA_BUNDLED_ELPA_KERNEL:STRING="" \
    -DLIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS:STRING="${elpa_configure_args}" \
    -DBUILD_SHARED_LIBS=ON \
    -DLIBRPA_VERBOSE_OUTPUT=ON \
    -DCMAKE_C_FLAGS="-g -O3 -fopenmp" \
    -DCMAKE_CXX_FLAGS="-g -O2 -fopenmp" \
    -DCMAKE_Fortran_FLAGS="-g -O2 -fopenmp -Wno-argument-mismatch -ffree-line-length-none"

cmake --build "${build_dir}" -j "${SLURM_CPUS_PER_TASK}"

grep -q '^LIBRPA_USE_EXTERNAL_ELPA:BOOL=OFF$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_USE_BUNDLED_ELPA:BOOL=ON$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_BUNDLED_ELPA_VERSION:STRING=2026.02.001$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_BUNDLED_ELPA_OPENMP:BOOL=ON$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_BUNDLED_ELPA_KERNEL:STRING=$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_USE_LIBRI_GPU:BOOL=OFF$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_USE_CUDA:BOOL=OFF$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_USE_HIP:BOOL=OFF$' "${build_dir}/CMakeCache.txt"

grep -q -- '-DLIBRPA_USE_ELPA' "${build_dir}/compile_commands.json"
grep -q -- '-DLIBRPA_MPI_THREAD_LEVEL=MPI_THREAD_MULTIPLE' "${build_dir}/compile_commands.json"
if grep -Eq -- '-D(__DDLA_RI|__HIP_RI|__CUDA_RI)' "${build_dir}/compile_commands.json"; then
    echo "Unexpected GPU LibRI compile definition in CPU build" >&2
    exit 1
fi

elpa_config=$(find "${build_dir}/thirdparty/ELPA" -name config.h -print -quit)
test -n "${elpa_config}"
for macro in WITH_MPI WITH_OPENMP_TRADITIONAL WANT_SINGLE_PRECISION_REAL WANT_SINGLE_PRECISION_COMPLEX; do
    grep -q "^#define ${macro} 1$" "${elpa_config}"
done
if grep -Eq '^#define WITH_(AMD|NVIDIA|SYCL|OPENMP_OFFLOAD).* 1$' "${elpa_config}"; then
    echo "Bundled CPU ELPA unexpectedly enabled a GPU backend" >&2
    grep -E '^#define WITH_(AMD|NVIDIA|SYCL|OPENMP_OFFLOAD).* 1$' "${elpa_config}" >&2
    exit 1
fi

elpa_archive=$(find "${build_dir}/thirdparty/ELPA/install" -name libelpa_openmp.a -print -quit)
test -n "${elpa_archive}"
test -s "${elpa_archive}"

cmake --install "${build_dir}"

test -x "${stage_dir}/bin/chi0_main.exe"
rpa_library=""
for library_dir in "${stage_dir}/lib" "${stage_dir}/lib64"; do
    if [[ -e "${library_dir}/librpa.so" ]]; then
        rpa_library="${library_dir}/librpa.so"
        break
    fi
done
test -n "${rpa_library}"

if find "${stage_dir}" \( -name 'libelpa*.so*' -o -name 'libelpa*.a' \) -print -quit | grep -q .; then
    echo "Private bundled ELPA unexpectedly appeared in the LibRPA install prefix" >&2
    find "${stage_dir}" \( -name 'libelpa*.so*' -o -name 'libelpa*.a' \) >&2
    exit 1
fi

export LD_LIBRARY_PATH="${stage_dir}/lib:${stage_dir}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for binary in "${stage_dir}/bin/chi0_main.exe" "${rpa_library}"; do
    ldd_output=$(ldd "${binary}")
    if grep -q 'not found' <<< "${ldd_output}"; then
        echo "Unresolved runtime dependency: ${binary}" >&2
        echo "${ldd_output}" >&2
        exit 1
    fi
    if grep -Eqi 'lib(elpa|amdhip64|rocblas|rocsolver|hipblas|hipsolver|cuda|cudart)' <<< "${ldd_output}"; then
        echo "Unexpected dynamic ELPA or GPU dependency: ${binary}" >&2
        echo "${ldd_output}" >&2
        exit 1
    fi
done

test ! -e "${install_dir}"
mv -T "${stage_dir}" "${install_dir}"
stage_promoted=1
export LD_LIBRARY_PATH="${install_dir}/lib:${install_dir}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
test -x "${install_dir}/bin/chi0_main.exe"
promoted_ldd=$(ldd "${install_dir}/bin/chi0_main.exe")
if grep -q 'not found' <<< "${promoted_ldd}"; then
    echo "Unresolved dependency after promotion" >&2
    echo "${promoted_ldd}" >&2
    exit 1
fi

cd "${repo_root}"
test "$(git diff --cached --binary | sha256sum | awk '{print $1}')" = "${expected_staged_sha}"
test "$(git diff --binary | sha256sum | awk '{print $1}')" = "${empty_diff_sha}"

trap - EXIT
echo "BUNDLED_ELPA_ARCHIVE=${elpa_archive}"
echo "DRIVER_MPI_THREAD_LEVEL=MPI_THREAD_MULTIPLE"
echo "BUNDLE_CPU_ELPA_BUILD_PASS"
echo "END_TIME=$(date --iso-8601=seconds)"
