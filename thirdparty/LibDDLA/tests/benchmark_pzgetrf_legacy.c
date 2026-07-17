/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from NVIDIA's cuSOLVERMp mp_getrf_getrs sample for the legacy
 * cuSOLVERMp 0.5.x CAL communicator API.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <cal.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverMp.h>

enum
{
    NPROW = 2,
    NPCOL = 2,
    NB = 128
};

static const unsigned long long RANDOM_SEED = 20260710ULL;

static void abort_with_status(const char* api, int status, const char* file, int line)
{
    int initialized = 0;
    int finalized = 0;
    int rank = -1;

    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    fprintf(stderr, "[rank %d] %s:%d: %s failed with status %d\n", rank, file, line, api, status);
    fflush(stderr);

    if (initialized && !finalized)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    exit(EXIT_FAILURE);
}

#define MPI_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        const int status_ = (call);                                                                                    \
        if (status_ != MPI_SUCCESS)                                                                                    \
        {                                                                                                              \
            abort_with_status(#call, status_, __FILE__, __LINE__);                                                     \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t status_ = (call);                                                                            \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            abort_with_status(#call, (int)status_, __FILE__, __LINE__);                                                \
        }                                                                                                              \
    } while (0)

#define CURAND_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        const curandStatus_t status_ = (call);                                                                         \
        if (status_ != CURAND_STATUS_SUCCESS)                                                                          \
        {                                                                                                              \
            abort_with_status(#call, (int)status_, __FILE__, __LINE__);                                                \
        }                                                                                                              \
    } while (0)

#define CUSOLVER_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        const cusolverStatus_t status_ = (call);                                                                       \
        if (status_ != CUSOLVER_STATUS_SUCCESS)                                                                        \
        {                                                                                                              \
            abort_with_status(#call, (int)status_, __FILE__, __LINE__);                                                \
        }                                                                                                              \
    } while (0)

#define CAL_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        const calError_t status_ = (call);                                                                             \
        if (status_ != CAL_OK)                                                                                         \
        {                                                                                                              \
            abort_with_status(#call, (int)status_, __FILE__, __LINE__);                                                \
        }                                                                                                              \
    } while (0)

static calError_t mpi_allgather(void* send_buffer,
                                void* receive_buffer,
                                size_t size,
                                void* data,
                                void** request)
{
    MPI_Request* mpi_request = (MPI_Request*)malloc(sizeof(*mpi_request));
    if (mpi_request == NULL)
    {
        return CAL_ERROR;
    }

    const MPI_Comm comm = *(MPI_Comm*)data;
    const int status = MPI_Iallgather(
            send_buffer, (int)size, MPI_BYTE, receive_buffer, (int)size, MPI_BYTE, comm, mpi_request);
    if (status != MPI_SUCCESS)
    {
        free(mpi_request);
        return CAL_ERROR;
    }

    *request = mpi_request;
    return CAL_OK;
}

static calError_t mpi_request_test(void* request)
{
    int completed = 0;
    const int status = MPI_Test((MPI_Request*)request, &completed, MPI_STATUS_IGNORE);
    if (status != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

static calError_t mpi_request_free(void* request)
{
    free(request);
    return CAL_OK;
}

static int get_local_rank(void)
{
    MPI_Comm local_comm = MPI_COMM_NULL;
    int local_rank = -1;

    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    return local_rank;
}

static int64_t round_up(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static int64_t parse_size(const char* text)
{
    char* end = NULL;
    errno = 0;
    const long long value = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0)
    {
        fprintf(stderr, "invalid matrix dimension: %s\n", text);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    return (int64_t)value;
}

static double benchmark_one(int64_t n,
                            int rank,
                            int process_row,
                            int process_col,
                            cusolverMpHandle_t handle,
                            cusolverMpGrid_t grid,
                            cal_comm_t cal_comm,
                            cudaStream_t stream,
                            curandGenerator_t random_generator)
{
    const int64_t local_rows = cusolverMpNUMROC(n, NB, (uint32_t)process_row, 0, NPROW);
    const int64_t local_cols = cusolverMpNUMROC(n, NB, (uint32_t)process_col, 0, NPCOL);

    /* cuSOLVERMp 0.5.x requires full-block local allocation dimensions. */
    const int64_t lld = round_up(local_rows > 0 ? local_rows : 1, NB);
    const int64_t allocated_cols = round_up(local_cols > 0 ? local_cols : 1, NB);
    const size_t local_elements = (size_t)lld * (size_t)allocated_cols;

    cuDoubleComplex* matrix = NULL;
    int64_t* pivots = NULL;
    int* device_info = NULL;
    void* device_workspace = NULL;
    void* host_workspace = NULL;
    size_t device_workspace_bytes = 0;
    size_t host_workspace_bytes = 0;
    cusolverMpMatrixDescriptor_t descriptor = NULL;

    CUDA_CHECK(cudaMalloc((void**)&matrix, local_elements * sizeof(*matrix)));
    CUDA_CHECK(cudaMalloc((void**)&pivots, (size_t)allocated_cols * sizeof(*pivots)));
    CUDA_CHECK(cudaMalloc((void**)&device_info, sizeof(*device_info)));

    CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(
            &descriptor, grid, CUDA_C_64F, n, n, NB, NB, 0, 0, lld));

    CUSOLVER_CHECK(cusolverMpGetrf_bufferSize(handle,
                                               n,
                                               n,
                                               matrix,
                                               1,
                                               1,
                                               descriptor,
                                               pivots,
                                               CUDA_C_64F,
                                               &device_workspace_bytes,
                                               &host_workspace_bytes));

    if (device_workspace_bytes > 0)
    {
        CUDA_CHECK(cudaMalloc(&device_workspace, device_workspace_bytes));
    }
    if (host_workspace_bytes > 0)
    {
        host_workspace = malloc(host_workspace_bytes);
        if (host_workspace == NULL)
        {
            abort_with_status("malloc(host_workspace)", EXIT_FAILURE, __FILE__, __LINE__);
        }
    }

    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED + (unsigned long long)rank));
    CURAND_CHECK(curandSetGeneratorOffset(random_generator, 0ULL));
    CURAND_CHECK(curandGenerateUniformDouble(
            random_generator, (double*)matrix, 2 * local_elements));
    CUDA_CHECK(cudaMemsetAsync(device_info, 0, sizeof(*device_info), stream));
    CAL_CHECK(cal_stream_sync(cal_comm, stream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    CUSOLVER_CHECK(cusolverMpGetrf(handle,
                                    n,
                                    n,
                                    matrix,
                                    1,
                                    1,
                                    descriptor,
                                    pivots,
                                    CUDA_C_64F,
                                    device_workspace,
                                    device_workspace_bytes,
                                    host_workspace,
                                    host_workspace_bytes,
                                    device_info));
    CAL_CHECK(cal_stream_sync(cal_comm, stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    int info = -1;
    CUDA_CHECK(cudaMemcpyAsync(&info, device_info, sizeof(info), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int local_failure = (info != 0);
    int global_failure = 0;
    MPI_CHECK(MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (global_failure)
    {
        if (info != 0)
        {
            fprintf(stderr, "[rank %d] cusolverMpGetrf returned info=%d for n=%" PRId64 "\n", rank, info, n);
            fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    if (rank == 0)
    {
        printf("RESULT n=%" PRId64
               " type=complex<double> grid=2x2 ranks=4 nb=128"
               " matrix=random_uniform time_s=%.6f info=0\n",
               n,
               max_elapsed);
        fflush(stdout);
    }

    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descriptor));
    if (host_workspace != NULL)
    {
        free(host_workspace);
    }
    if (device_workspace != NULL)
    {
        CUDA_CHECK(cudaFree(device_workspace));
    }
    CUDA_CHECK(cudaFree(device_info));
    CUDA_CHECK(cudaFree(pivots));
    CUDA_CHECK(cudaFree(matrix));

    return max_elapsed;
}

int main(int argc, char** argv)
{
    const int mpi_init_status = MPI_Init(&argc, &argv);
    if (mpi_init_status != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init failed with status %d\n", mpi_init_status);
        return EXIT_FAILURE;
    }

    int rank = -1;
    int nranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    if (nranks != NPROW * NPCOL)
    {
        if (rank == 0)
        {
            fprintf(stderr, "benchmark requires exactly 4 MPI ranks for a 2x2 grid\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    const int local_device = get_local_rank();
    CUDA_CHECK(cudaSetDevice(local_device));
    CUDA_CHECK(cudaFree(0));

    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_Comm cal_mpi_comm = MPI_COMM_WORLD;
    cal_comm_create_params_t cal_params;
    cal_params.allgather = mpi_allgather;
    cal_params.req_test = mpi_request_test;
    cal_params.req_free = mpi_request_free;
    cal_params.data = &cal_mpi_comm;
    cal_params.nranks = nranks;
    cal_params.rank = rank;
    cal_params.local_device = local_device;

    cal_comm_t cal_comm = NULL;
    CAL_CHECK(cal_comm_create(cal_params, &cal_comm));

    cusolverMpHandle_t handle = NULL;
    cusolverMpGrid_t grid = NULL;
    CUSOLVER_CHECK(cusolverMpCreate(&handle, local_device, stream));
    CUSOLVER_CHECK(cusolverMpCreateDeviceGrid(
            handle, &grid, cal_comm, NPROW, NPCOL, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));

    curandGenerator_t random_generator = NULL;
    CURAND_CHECK(curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(random_generator, stream));

    int runtime_version = 0;
    CUSOLVER_CHECK(cusolverMpGetVersion(handle, &runtime_version));
    if (rank == 0)
    {
        printf("=== cuSOLVERMp pzgetrf benchmark: complex<double>, 4 MPI ranks, 2x2 grid ===\n");
        printf("cuSOLVERMp compile_version=%d runtime_version=%d seed=%llu\n",
               CUSOLVERMP_VERSION,
               runtime_version,
               RANDOM_SEED);
        fflush(stdout);
    }

    const int default_count = 4;
    const int64_t default_sizes[] = {500, 5000, 10000, 15000};
    const int size_count = argc > 1 ? argc - 1 : default_count;
    int64_t* sizes = (int64_t*)malloc((size_t)size_count * sizeof(*sizes));
    if (sizes == NULL)
    {
        abort_with_status("malloc(sizes)", EXIT_FAILURE, __FILE__, __LINE__);
    }

    for (int i = 0; i < size_count; ++i)
    {
        sizes[i] = argc > 1 ? parse_size(argv[i + 1]) : default_sizes[i];
    }

    const int process_row = rank / NPCOL;
    const int process_col = rank % NPCOL;
    for (int i = 0; i < size_count; ++i)
    {
        benchmark_one(sizes[i],
                      rank,
                      process_row,
                      process_col,
                      handle,
                      grid,
                      cal_comm,
                      stream,
                      random_generator);
    }
    free(sizes);

    CURAND_CHECK(curandDestroyGenerator(random_generator));
    CAL_CHECK(cal_comm_barrier(cal_comm, stream));
    CAL_CHECK(cal_stream_sync(cal_comm, stream));
    CUSOLVER_CHECK(cusolverMpDestroyGrid(grid));
    CUSOLVER_CHECK(cusolverMpDestroy(handle));
    CAL_CHECK(cal_comm_destroy(cal_comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_CHECK(MPI_Finalize());
    return EXIT_SUCCESS;
}
