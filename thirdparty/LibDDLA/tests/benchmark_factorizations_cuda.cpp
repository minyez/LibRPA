/*
 * Unified 4-GPU benchmark for LibDDLA and legacy cuSOLVERMp factorization
 * routines.  All paths factor the same deterministic Hermitian positive-
 * definite matrix, represented in the native local layout of each library.
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <cal.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverMp.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_desc.h>
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_stream.h>

namespace {

using Complex = std::complex<double>;

constexpr int kProcessRows = 2;
constexpr int kProcessCols = 2;
constexpr int kRanks = kProcessRows * kProcessCols;
constexpr int kBlockSize = 128;
constexpr int kSourceRow = 0;
constexpr int kSourceCol = 0;
constexpr double kLogDetTolerance = 1.0e-5;

static_assert(sizeof(Complex) == 2 * sizeof(double),
              "complex<double> must contain two doubles");
static_assert(sizeof(Complex) == sizeof(cuDoubleComplex),
              "complex<double> must match CUDA complex-double storage");

struct Options {
    int warmup_size = 500;
    int repetitions = 1;
    std::vector<int> sizes = {5000, 10000, 15000};
};

enum class Algorithm {
    libddla_pgetrf_nopiv = 0,
    libddla_ppotrf_lower = 1,
    libddla_pgetrf = 2,
    cusolvermp_pzgetrf = 3,
};

constexpr int kAlgorithmCount = 4;

const char* algorithm_name(Algorithm algorithm)
{
    switch(algorithm){
    case Algorithm::libddla_pgetrf_nopiv:
        return "libddla_pgetrf_nopiv";
    case Algorithm::libddla_ppotrf_lower:
        return "libddla_ppotrf_L";
    case Algorithm::libddla_pgetrf:
        return "libddla_pgetrf";
    case Algorithm::cusolvermp_pzgetrf:
        return "cusolvermp_pzgetrf";
    }
    return "unknown";
}

int algorithm_index(Algorithm algorithm)
{
    return static_cast<int>(algorithm);
}

[[noreturn]] void abort_with_status(const char* api, int status,
                                    const char* file, int line)
{
    int initialized = 0;
    int finalized = 0;
    int rank = -1;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if(initialized && !finalized){
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    std::cerr << "[rank " << rank << "] " << file << ':' << line << ": "
              << api << " failed with status " << status << std::endl;
    if(initialized && !finalized){
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::exit(EXIT_FAILURE);
}

#define BENCH_MPI_CHECK(call)                                                   \
    do {                                                                        \
        const int status_ = (call);                                             \
        if(status_ != MPI_SUCCESS){                                             \
            abort_with_status(#call, status_, __FILE__, __LINE__);              \
        }                                                                       \
    } while(0)

#define BENCH_CUDA_CHECK(call)                                                  \
    do {                                                                        \
        const cudaError_t status_ = (call);                                     \
        if(status_ != cudaSuccess){                                             \
            abort_with_status(#call, static_cast<int>(status_),                 \
                              __FILE__, __LINE__);                              \
        }                                                                       \
    } while(0)

#define BENCH_CAL_CHECK(call)                                                   \
    do {                                                                        \
        const calError_t status_ = (call);                                      \
        if(status_ != CAL_OK){                                                  \
            abort_with_status(#call, static_cast<int>(status_),                 \
                              __FILE__, __LINE__);                              \
        }                                                                       \
    } while(0)

#define BENCH_CUSOLVERMP_CHECK(call)                                            \
    do {                                                                        \
        const cusolverStatus_t status_ = (call);                                \
        if(status_ != CUSOLVER_STATUS_SUCCESS){                                 \
            abort_with_status(#call, static_cast<int>(status_),                 \
                              __FILE__, __LINE__);                              \
        }                                                                       \
    } while(0)

[[noreturn]] void abort_with_message(const std::string& message)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0){
        std::cerr << message << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::exit(EXIT_FAILURE);
}

bool parse_positive_int(const std::string& text, int& value)
{
    if(text.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if(errno != 0 || end == text.c_str() || *end != '\0' || parsed <= 0
       || parsed > std::numeric_limits<int>::max()){
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options& options, std::string& error)
{
    bool sizes_set = false;
    bool warmup_set = false;
    bool repetitions_set = false;

    for(int i = 1; i < argc; ++i){
        const std::string arg(argv[i]);
        if(arg == "--help" || arg == "-h"){
            return false;
        }
        if(arg == "--warmup"){
            if(warmup_set || i + 1 >= argc){
                error = warmup_set ? "--warmup was provided more than once"
                                   : "--warmup requires a positive dimension";
                return false;
            }
            warmup_set = true;
            if(!parse_positive_int(argv[++i], options.warmup_size)){
                error = "invalid --warmup value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--warmup=", 0) == 0){
            if(warmup_set){
                error = "--warmup was provided more than once";
                return false;
            }
            warmup_set = true;
            if(!parse_positive_int(arg.substr(9), options.warmup_size)){
                error = "invalid --warmup value: " + arg.substr(9);
                return false;
            }
        }else if(arg == "--repeats"){
            if(repetitions_set || i + 1 >= argc){
                error = repetitions_set
                      ? "--repeats was provided more than once"
                      : "--repeats requires a positive integer";
                return false;
            }
            repetitions_set = true;
            if(!parse_positive_int(argv[++i], options.repetitions)){
                error = "invalid --repeats value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--repeats=", 0) == 0){
            if(repetitions_set){
                error = "--repeats was provided more than once";
                return false;
            }
            repetitions_set = true;
            if(!parse_positive_int(arg.substr(10), options.repetitions)){
                error = "invalid --repeats value: " + arg.substr(10);
                return false;
            }
        }else if(!arg.empty() && arg[0] == '-'){
            error = "unknown option: " + arg;
            return false;
        }else{
            int size = 0;
            if(!parse_positive_int(arg, size)){
                error = "invalid matrix dimension: " + arg;
                return false;
            }
            if(!sizes_set){
                options.sizes.clear();
                sizes_set = true;
            }
            options.sizes.push_back(size);
        }
    }
    return true;
}

std::string usage(const char* program)
{
    std::ostringstream os;
    os << "Usage: " << program
       << " [--warmup N] [--repeats N] [5000 10000 15000]";
    return os.str();
}

calError_t mpi_allgather(void* send_buffer, void* receive_buffer,
                         size_t size, void* data, void** request)
{
    if(size > static_cast<size_t>(std::numeric_limits<int>::max())){
        return CAL_ERROR;
    }
    auto* mpi_request = static_cast<MPI_Request*>(std::malloc(sizeof(MPI_Request)));
    if(mpi_request == nullptr){
        return CAL_ERROR;
    }
    const MPI_Comm comm = *static_cast<MPI_Comm*>(data);
    const int status = MPI_Iallgather(send_buffer, static_cast<int>(size), MPI_BYTE,
                                      receive_buffer, static_cast<int>(size), MPI_BYTE,
                                      comm, mpi_request);
    if(status != MPI_SUCCESS){
        std::free(mpi_request);
        return CAL_ERROR;
    }
    *request = mpi_request;
    return CAL_OK;
}

calError_t mpi_request_test(void* request)
{
    int completed = 0;
    const int status = MPI_Test(static_cast<MPI_Request*>(request), &completed,
                                MPI_STATUS_IGNORE);
    if(status != MPI_SUCCESS){
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

calError_t mpi_request_free(void* request)
{
    std::free(request);
    return CAL_OK;
}

std::int64_t round_up(std::int64_t value, std::int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

std::vector<Complex> make_phase(int n)
{
    const double two_pi = 2.0 * std::acos(-1.0);
    std::vector<Complex> phase(static_cast<size_t>(n));
    for(int i = 0; i < n; ++i){
        const double x = static_cast<double>(i) / static_cast<double>(n);
        const double angle = two_pi * (x + 0.125 * x * x);
        phase[static_cast<size_t>(i)] = Complex(std::cos(angle), std::sin(angle));
    }
    return phase;
}

std::vector<Complex> make_compact_hpd(const ddla::DdlaDesc& desc,
                                      const std::vector<Complex>& phase)
{
    const int n = desc.n();
    const double scale = 1.0 / static_cast<double>(n);
    const size_t count = static_cast<size_t>(desc.lld())
                       * static_cast<size_t>(desc.n_loc());
    std::vector<Complex> local(count, Complex(0.0, 0.0));
    std::vector<int> global_rows(static_cast<size_t>(desc.m_loc()));
    std::vector<int> global_cols(static_cast<size_t>(desc.n_loc()));

    for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
        global_rows[static_cast<size_t>(iloc)] = desc.indx_l2g_r(iloc);
    }
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        global_cols[static_cast<size_t>(jloc)] = desc.indx_l2g_c(jloc);
    }

    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = global_cols[static_cast<size_t>(jloc)];
        if(j >= n) continue;
        const Complex phase_j = std::conj(phase[static_cast<size_t>(j)]);
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = global_rows[static_cast<size_t>(iloc)];
            if(i >= n) continue;
            Complex value = scale * phase[static_cast<size_t>(i)] * phase_j;
            if(i == j) value += Complex(1.0, 0.0);
            local[static_cast<size_t>(iloc)
                  + static_cast<size_t>(jloc) * desc.lld()] = value;
        }
    }
    return local;
}

template <typename T>
T* allocate_device(size_t count)
{
    T* pointer = nullptr;
    const size_t allocation_count = std::max<size_t>(count, 1);
    BENCH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                                allocation_count * sizeof(T)));
    return pointer;
}

struct VendorContext {
    cal_comm_t cal_comm = nullptr;
    cusolverMpHandle_t handle = nullptr;
    cusolverMpGrid_t grid = nullptr;
};

struct VendorProblem {
    int n = 0;
    int process_row = 0;
    int process_col = 0;
    std::int64_t local_rows = 0;
    std::int64_t local_cols = 0;
    std::int64_t lld = 0;
    std::int64_t allocated_cols = 0;
    size_t count = 0;
    Complex* d_reference = nullptr;
    Complex* d_matrix = nullptr;
    std::int64_t* d_pivots = nullptr;
    int* d_info = nullptr;
    void* d_workspace = nullptr;
    void* h_workspace = nullptr;
    size_t device_workspace_bytes = 0;
    size_t host_workspace_bytes = 0;
    cusolverMpMatrixDescriptor_t descriptor = nullptr;
};

VendorProblem create_vendor_problem(int n, int process_row, int process_col,
                                    const ddla::DdlaDesc& compact_desc,
                                    const Complex* d_compact_reference,
                                    const VendorContext& context,
                                    cudaStream_t stream)
{
    VendorProblem problem;
    problem.n = n;
    problem.process_row = process_row;
    problem.process_col = process_col;
    problem.local_rows = cusolverMpNUMROC(
        n, kBlockSize, static_cast<uint32_t>(process_row), kSourceRow,
        kProcessRows);
    problem.local_cols = cusolverMpNUMROC(
        n, kBlockSize, static_cast<uint32_t>(process_col), kSourceCol,
        kProcessCols);
    problem.lld = round_up(std::max<std::int64_t>(problem.local_rows, 1),
                           kBlockSize);
    problem.allocated_cols = round_up(
        std::max<std::int64_t>(problem.local_cols, 1), kBlockSize);
    problem.count = static_cast<size_t>(problem.lld)
                  * static_cast<size_t>(problem.allocated_cols);

    if(problem.local_rows != compact_desc.m_loc()
       || problem.local_cols != compact_desc.n_loc()){
        abort_with_message(
            "LibDDLA and cuSOLVERMp disagree on the logical local shape");
    }

    problem.d_reference = allocate_device<Complex>(problem.count);
    problem.d_matrix = allocate_device<Complex>(problem.count);
    problem.d_pivots = allocate_device<std::int64_t>(
        static_cast<size_t>(problem.local_rows + kBlockSize));
    problem.d_info = allocate_device<int>(1);

    BENCH_CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(
        &problem.descriptor, context.grid, CUDA_C_64F, n, n,
        kBlockSize, kBlockSize, kSourceRow, kSourceCol, problem.lld));

    BENCH_CUSOLVERMP_CHECK(cusolverMpGetrf_bufferSize(
        context.handle, n, n, problem.d_matrix, 1, 1, problem.descriptor,
        problem.d_pivots, CUDA_C_64F,
        &problem.device_workspace_bytes, &problem.host_workspace_bytes));

    if(problem.device_workspace_bytes > 0){
        BENCH_CUDA_CHECK(cudaMalloc(&problem.d_workspace,
                                    problem.device_workspace_bytes));
    }
    if(problem.host_workspace_bytes > 0){
        problem.h_workspace = std::malloc(problem.host_workspace_bytes);
        if(problem.h_workspace == nullptr){
            abort_with_status("malloc(host_workspace)", EXIT_FAILURE,
                              __FILE__, __LINE__);
        }
    }

    BENCH_CUDA_CHECK(cudaMemsetAsync(
        problem.d_reference, 0, problem.count * sizeof(Complex), stream));
    if(problem.local_rows > 0 && problem.local_cols > 0){
        // The active elements are copied bit-for-bit from the LibDDLA
        // reference.  Only the physical leading dimension and padding differ.
        BENCH_CUDA_CHECK(cudaMemcpy2DAsync(
            problem.d_reference,
            static_cast<size_t>(problem.lld) * sizeof(Complex),
            d_compact_reference,
            static_cast<size_t>(compact_desc.lld()) * sizeof(Complex),
            static_cast<size_t>(problem.local_rows) * sizeof(Complex),
            static_cast<size_t>(problem.local_cols),
            cudaMemcpyDeviceToDevice, stream));
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    return problem;
}

void destroy_vendor_problem(VendorProblem& problem, cudaStream_t stream)
{
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    if(problem.descriptor != nullptr){
        BENCH_CUSOLVERMP_CHECK(cusolverMpDestroyMatrixDesc(problem.descriptor));
        problem.descriptor = nullptr;
    }
    if(problem.h_workspace != nullptr){
        std::free(problem.h_workspace);
        problem.h_workspace = nullptr;
    }
    if(problem.d_workspace != nullptr){
        BENCH_CUDA_CHECK(cudaFree(problem.d_workspace));
        problem.d_workspace = nullptr;
    }
    if(problem.d_info != nullptr){
        BENCH_CUDA_CHECK(cudaFree(problem.d_info));
        problem.d_info = nullptr;
    }
    if(problem.d_pivots != nullptr){
        BENCH_CUDA_CHECK(cudaFree(problem.d_pivots));
        problem.d_pivots = nullptr;
    }
    if(problem.d_matrix != nullptr){
        BENCH_CUDA_CHECK(cudaFree(problem.d_matrix));
        problem.d_matrix = nullptr;
    }
    if(problem.d_reference != nullptr){
        BENCH_CUDA_CHECK(cudaFree(problem.d_reference));
        problem.d_reference = nullptr;
    }
}

void check_factorization_status(const char* label, int info,
                                bool sign_correction, MPI_Comm comm, int rank)
{
    const int local_failure = (info != 0 || sign_correction) ? 1 : 0;
    if(local_failure){
        std::cerr << "[rank " << rank << "] " << label
                  << " failed: info=" << info
                  << " sign_correction=" << sign_correction << std::endl;
    }
    int global_failure = 0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT,
                                  MPI_MAX, comm));
    if(global_failure){
        abort_with_message(std::string(label) + " failed the info check");
    }
}

double finalize_logdet(double local_sum, int local_valid,
                       int local_diagonal_count, int expected_diagonal_count,
                       const char* label, MPI_Comm comm, int rank)
{
    int global_valid = 0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT,
                                  MPI_MIN, comm));
    double global_sum = 0.0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
                                  MPI_SUM, comm));
    int global_diagonal_count = 0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_diagonal_count,
                                  &global_diagonal_count, 1, MPI_INT,
                                  MPI_SUM, comm));
    const double error = std::abs(global_sum - std::log(2.0));
    const int local_failure = (!global_valid || !std::isfinite(global_sum)
                               || global_diagonal_count != expected_diagonal_count
                               || error > kLogDetTolerance) ? 1 : 0;
    int global_failure = 0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT,
                                  MPI_MAX, comm));
    if(global_failure){
        if(rank == 0){
            std::cerr << label << " failed logdet check: value="
                      << std::setprecision(17) << global_sum
                      << " expected=" << std::log(2.0)
                      << " error=" << error
                      << " diagonal_count=" << global_diagonal_count
                      << " expected_count=" << expected_diagonal_count
                      << " tolerance=" << kLogDetTolerance << std::endl;
        }
        MPI_Abort(comm, EXIT_FAILURE);
        std::exit(EXIT_FAILURE);
    }
    return error;
}

double compact_logdet_error(const Complex* d_matrix,
                            const ddla::DdlaDesc& desc,
                            bool cholesky, cudaStream_t stream,
                            MPI_Comm comm, int rank, const char* label)
{
    double local_sum = 0.0;
    int local_valid = 1;
    int local_diagonal_count = 0;
    std::vector<Complex> block(static_cast<size_t>(kBlockSize) * kBlockSize);

    for(int start = 0; start < desc.n(); start += kBlockSize){
        const int iloc = desc.indx_g2l_r(start);
        const int jloc = desc.indx_g2l_c(start);
        if(iloc < 0 || jloc < 0) continue;
        const int width = std::min(kBlockSize, desc.n() - start);
        BENCH_CUDA_CHECK(cudaMemcpy2D(
            block.data(), static_cast<size_t>(width) * sizeof(Complex),
            d_matrix + iloc + static_cast<size_t>(jloc) * desc.lld(),
            static_cast<size_t>(desc.lld()) * sizeof(Complex),
            static_cast<size_t>(width) * sizeof(Complex), width,
            cudaMemcpyDeviceToHost));
        for(int k = 0; k < width; ++k){
            ++local_diagonal_count;
            const Complex diagonal = block[static_cast<size_t>(k)
                                          + static_cast<size_t>(k) * width];
            const double magnitude = std::abs(diagonal);
            if(!std::isfinite(magnitude) || magnitude <= 0.0){
                local_valid = 0;
                continue;
            }
            if(cholesky){
                const double imag_tolerance = 1.0e-10
                    * std::max(1.0, std::abs(diagonal.real()));
                if(diagonal.real() <= 0.0
                   || std::abs(diagonal.imag()) > imag_tolerance){
                    local_valid = 0;
                }
                local_sum += 2.0 * std::log(magnitude);
            }else{
                local_sum += std::log(magnitude);
            }
        }
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    return finalize_logdet(local_sum, local_valid, local_diagonal_count,
                           desc.n(), label, comm, rank);
}

double padded_logdet_error(const VendorProblem& problem,
                           cudaStream_t stream, MPI_Comm comm,
                           int rank, const char* label)
{
    double local_sum = 0.0;
    int local_valid = 1;
    int local_diagonal_count = 0;
    std::vector<Complex> block(static_cast<size_t>(kBlockSize) * kBlockSize);
    const int block_count = (problem.n + kBlockSize - 1) / kBlockSize;

    for(int block_index = 0; block_index < block_count; ++block_index){
        const int owner = block_index % kProcessRows;
        if(problem.process_row != owner || problem.process_col != owner) continue;
        const std::int64_t local_start =
            static_cast<std::int64_t>(block_index / kProcessRows) * kBlockSize;
        const int global_start = block_index * kBlockSize;
        const int width = std::min(kBlockSize, problem.n - global_start);
        BENCH_CUDA_CHECK(cudaMemcpy2D(
            block.data(), static_cast<size_t>(width) * sizeof(Complex),
            problem.d_matrix + local_start
                + static_cast<size_t>(local_start)
                    * static_cast<size_t>(problem.lld),
            static_cast<size_t>(problem.lld) * sizeof(Complex),
            static_cast<size_t>(width) * sizeof(Complex), width,
            cudaMemcpyDeviceToHost));
        for(int k = 0; k < width; ++k){
            ++local_diagonal_count;
            const Complex diagonal = block[static_cast<size_t>(k)
                                          + static_cast<size_t>(k) * width];
            const double magnitude = std::abs(diagonal);
            if(!std::isfinite(magnitude) || magnitude <= 0.0){
                local_valid = 0;
                continue;
            }
            local_sum += std::log(magnitude);
        }
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    return finalize_logdet(local_sum, local_valid, local_diagonal_count,
                           problem.n, label, comm, rank);
}

struct RunResult {
    double time_seconds = 0.0;
    double logdet_error = 0.0;
};

RunResult run_libddla(Algorithm algorithm, int n,
                      Complex* d_matrix, const Complex* d_reference,
                      size_t count, const ddla::DdlaDesc& desc,
                      std::vector<int>& pivots,
                      const ddla::DdlaHandle_t& handle)
{
    if(count > 0){
        BENCH_CUDA_CHECK(cudaMemcpyAsync(
            d_matrix, d_reference, count * sizeof(Complex),
            cudaMemcpyDeviceToDevice, handle->stream));
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(handle->stream));

    int info = 0;
    bool sign_correction = false;
    BENCH_MPI_CHECK(MPI_Barrier(handle->comm));
    const double start = MPI_Wtime();
    if(algorithm == Algorithm::libddla_pgetrf_nopiv){
        ddla::pgetrf_nopiv(n, n, d_matrix, desc, info);
    }else if(algorithm == Algorithm::libddla_ppotrf_lower){
        sign_correction = ddla::ppotrf(
            'L', n, d_matrix, 1, 1, desc, info);
    }else if(algorithm == Algorithm::libddla_pgetrf){
        ddla::pgetrf(n, n, d_matrix, desc, pivots.data(), info);
    }else{
        abort_with_message("internal error: vendor algorithm sent to LibDDLA");
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(handle->stream));
    const double local_elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE,
                                  MPI_MAX, handle->comm));
    check_factorization_status(algorithm_name(algorithm), info, sign_correction,
                               handle->comm, handle->myid);
    const bool cholesky = algorithm == Algorithm::libddla_ppotrf_lower;
    const double logdet_error = compact_logdet_error(
        d_matrix, desc, cholesky, handle->stream, handle->comm,
        handle->myid, algorithm_name(algorithm));
    return {max_elapsed, logdet_error};
}

RunResult run_cusolvermp(VendorProblem& problem,
                         const VendorContext& context,
                         const ddla::DdlaHandle_t& handle)
{
    BENCH_CUDA_CHECK(cudaMemcpyAsync(
        problem.d_matrix, problem.d_reference,
        problem.count * sizeof(Complex), cudaMemcpyDeviceToDevice,
        handle->stream));
    BENCH_CUDA_CHECK(cudaMemsetAsync(problem.d_info, 0, sizeof(int),
                                     handle->stream));
    BENCH_CAL_CHECK(cal_stream_sync(context.cal_comm, handle->stream));

    BENCH_MPI_CHECK(MPI_Barrier(handle->comm));
    const double start = MPI_Wtime();
    BENCH_CUSOLVERMP_CHECK(cusolverMpGetrf(
        context.handle, problem.n, problem.n, problem.d_matrix, 1, 1,
        problem.descriptor, problem.d_pivots, CUDA_C_64F,
        problem.d_workspace, problem.device_workspace_bytes,
        problem.h_workspace, problem.host_workspace_bytes, problem.d_info));
    BENCH_CAL_CHECK(cal_stream_sync(context.cal_comm, handle->stream));
    const double local_elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    BENCH_MPI_CHECK(MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE,
                                  MPI_MAX, handle->comm));
    int info = -1;
    BENCH_CUDA_CHECK(cudaMemcpyAsync(&info, problem.d_info, sizeof(info),
                                     cudaMemcpyDeviceToHost, handle->stream));
    BENCH_CUDA_CHECK(cudaStreamSynchronize(handle->stream));
    check_factorization_status(algorithm_name(Algorithm::cusolvermp_pzgetrf),
                               info, false, handle->comm, handle->myid);
    const double logdet_error = padded_logdet_error(
        problem, handle->stream, handle->comm, handle->myid,
        algorithm_name(Algorithm::cusolvermp_pzgetrf));
    return {max_elapsed, logdet_error};
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if(values.size() % 2 != 0) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

void print_run(const char* kind, Algorithm algorithm, int iteration,
               int n, const RunResult& result)
{
    std::cout << kind
              << " algorithm=" << algorithm_name(algorithm)
              << " iteration=" << iteration + 1
              << " n=" << n
              << " nb=" << kBlockSize
              << " grid=2x2 ranks=4 source=0,0"
              << " time_s=" << std::fixed << std::setprecision(6)
              << result.time_seconds
              << " logdet_error=" << std::scientific << std::setprecision(3)
              << result.logdet_error
              << std::defaultfloat << std::endl;
}

void benchmark_dimension(int n, int repetitions, int warmup_size, bool warmup,
                         const ddla::DdlaHandle_t& handle,
                         const VendorContext& vendor_context)
{
    const std::vector<Complex> phase = make_phase(n);

    ddla::DdlaDesc desc(handle);
    desc.init(n, n, kBlockSize, kBlockSize, kSourceRow, kSourceCol);
    const size_t compact_count = static_cast<size_t>(desc.lld())
                               * static_cast<size_t>(desc.n_loc());
    Complex* d_compact_reference = allocate_device<Complex>(compact_count);
    Complex* d_compact_matrix = allocate_device<Complex>(compact_count);
    {
        const std::vector<Complex> host = make_compact_hpd(desc, phase);
        if(compact_count > 0){
            BENCH_CUDA_CHECK(cudaMemcpyAsync(
                d_compact_reference, host.data(),
                compact_count * sizeof(Complex), cudaMemcpyHostToDevice,
                handle->stream));
        }
        BENCH_CUDA_CHECK(cudaStreamSynchronize(handle->stream));
    }
    std::vector<int> pivots(
        static_cast<size_t>(std::max(desc.m_loc(), 1)), 0);

    VendorProblem vendor_problem = create_vendor_problem(
        n, handle->myprow_, handle->mypcol_, desc, d_compact_reference,
        vendor_context, handle->stream);

    const std::vector<Algorithm> algorithms = {
        Algorithm::libddla_pgetrf_nopiv,
        Algorithm::libddla_ppotrf_lower,
        Algorithm::libddla_pgetrf,
        Algorithm::cusolvermp_pzgetrf,
    };
    std::vector<double> times[kAlgorithmCount];
    std::vector<double> errors[kAlgorithmCount];

    for(int iteration = 0; iteration < repetitions; ++iteration){
        for(int order = 0; order < kAlgorithmCount; ++order){
            const int rotated = warmup ? order : (order + iteration) % kAlgorithmCount;
            const Algorithm algorithm = algorithms[static_cast<size_t>(rotated)];
            RunResult result;
            if(algorithm == Algorithm::cusolvermp_pzgetrf){
                result = run_cusolvermp(vendor_problem, vendor_context, handle);
            }else{
                result = run_libddla(
                    algorithm, n, d_compact_matrix, d_compact_reference,
                    compact_count, desc, pivots, handle);
            }
            times[algorithm_index(algorithm)].push_back(result.time_seconds);
            errors[algorithm_index(algorithm)].push_back(result.logdet_error);
            if(handle->myid == 0){
                print_run(warmup ? "WARMUP" : "RUN",
                          algorithm, iteration, n, result);
            }
        }
    }

    if(!warmup && handle->myid == 0){
        double medians[kAlgorithmCount] = {};
        for(Algorithm algorithm : algorithms){
            const int index = algorithm_index(algorithm);
            medians[index] = median(times[index]);
            const double max_error = *std::max_element(
                errors[index].begin(), errors[index].end());
            std::cout << "RESULT"
                      << " algorithm=" << algorithm_name(algorithm)
                      << " n=" << n
                      << " type=complex<double> nb=" << kBlockSize
                      << " grid=2x2 ranks=4 source=0,0"
                      << " warmup_n=" << warmup_size
                      << " reps=" << repetitions
                      << " median_s=" << std::fixed << std::setprecision(6)
                      << medians[index]
                      << " max_logdet_error=" << std::scientific
                      << std::setprecision(3) << max_error
                      << std::defaultfloat << std::endl;
        }
        std::cout << "TABLE"
                  << " n=" << n
                  << " pgetrf_nopiv_s=" << std::fixed << std::setprecision(6)
                  << medians[algorithm_index(Algorithm::libddla_pgetrf_nopiv)]
                  << " ppotrf_L_s="
                  << medians[algorithm_index(Algorithm::libddla_ppotrf_lower)]
                  << " pgetrf_s="
                  << medians[algorithm_index(Algorithm::libddla_pgetrf)]
                  << " cusolvermp_pzgetrf_s="
                  << medians[algorithm_index(Algorithm::cusolvermp_pzgetrf)]
                  << std::defaultfloat << std::endl;
    }

    destroy_vendor_problem(vendor_problem, handle->stream);
    BENCH_CUDA_CHECK(cudaFree(d_compact_matrix));
    BENCH_CUDA_CHECK(cudaFree(d_compact_reference));
    BENCH_CUDA_CHECK(cudaStreamSynchronize(handle->stream));
}

} // namespace

int main(int argc, char** argv)
{
    const int mpi_status = MPI_Init(&argc, &argv);
    if(mpi_status != MPI_SUCCESS){
        std::cerr << "MPI_Init failed with status " << mpi_status << std::endl;
        return EXIT_FAILURE;
    }

    int rank = -1;
    int nranks = 0;
    BENCH_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    BENCH_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    Options options;
    std::string parse_error;
    if(!parse_options(argc, argv, options, parse_error)){
        if(rank == 0){
            if(!parse_error.empty()) std::cerr << parse_error << std::endl;
            std::cerr << usage(argv[0]) << std::endl;
        }
        BENCH_MPI_CHECK(MPI_Finalize());
        return parse_error.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if(nranks != kRanks){
        abort_with_message("benchmark requires exactly 4 MPI ranks for a 2x2 grid");
    }

    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddla_init(handle);
    ddla::ddla_set(handle, MPI_COMM_WORLD, kProcessRows, kProcessCols, 'R');

    MPI_Comm cal_mpi_comm = MPI_COMM_WORLD;
    cal_comm_create_params_t cal_params{};
    cal_params.allgather = mpi_allgather;
    cal_params.req_test = mpi_request_test;
    cal_params.req_free = mpi_request_free;
    cal_params.data = &cal_mpi_comm;
    cal_params.nranks = nranks;
    cal_params.rank = rank;
    cal_params.local_device = ddla::DdlaStream::local_device;

    VendorContext vendor_context;
    BENCH_CAL_CHECK(cal_comm_create(cal_params, &vendor_context.cal_comm));
    BENCH_CUSOLVERMP_CHECK(cusolverMpCreate(
        &vendor_context.handle, ddla::DdlaStream::local_device, handle->stream));
    BENCH_CUSOLVERMP_CHECK(cusolverMpCreateDeviceGrid(
        vendor_context.handle, &vendor_context.grid, vendor_context.cal_comm,
        kProcessRows, kProcessCols, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));

    int runtime_version = 0;
    BENCH_CUSOLVERMP_CHECK(
        cusolverMpGetVersion(vendor_context.handle, &runtime_version));
    if(rank == 0){
        std::cout << "CONFIG type=complex<double> grid=2x2 ranks=4 nb=128"
                  << " source=0,0 matrix=I+v*v^H/n"
                  << " warmup_n=" << options.warmup_size
                  << " reps=" << options.repetitions
                  << " cusolvermp_compile_version=" << CUSOLVERMP_VERSION
                  << " cusolvermp_runtime_version=" << runtime_version
                  << std::endl;
    }

    benchmark_dimension(options.warmup_size, 1, options.warmup_size, true,
                        handle, vendor_context);
    for(int n : options.sizes){
        benchmark_dimension(n, options.repetitions, options.warmup_size, false,
                            handle, vendor_context);
    }

    BENCH_CAL_CHECK(cal_comm_barrier(vendor_context.cal_comm, handle->stream));
    BENCH_CAL_CHECK(cal_stream_sync(vendor_context.cal_comm, handle->stream));
    BENCH_CUSOLVERMP_CHECK(cusolverMpDestroyGrid(vendor_context.grid));
    BENCH_CUSOLVERMP_CHECK(cusolverMpDestroy(vendor_context.handle));
    BENCH_CAL_CHECK(cal_comm_destroy(vendor_context.cal_comm));

    ddla::ddla_destroy(handle);
    BENCH_MPI_CHECK(MPI_Finalize());
    return EXIT_SUCCESS;
}
