#ifndef DDLA_CONNECTOR_H
#define DDLA_CONNECTOR_H

#include <mpi.h>
#include <iostream>
#ifdef DDLA_USE_CUDA
#ifdef DDLA_USE_CCL
#include <nccl.h>
#endif
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#ifdef DDLA_USE_HIP
#ifdef DDLA_USE_CCL
#include <rccl/rccl.h>
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hiprand/hiprand.h>
#endif

#include <cmath>
#include <complex>

namespace ddla{

#ifdef DDLA_USE_CCL
using cclOp=ncclRedOp_t;
const auto cclSum=ncclRedOp_t::ncclSum;
#else
using cclOp=MPI_Op;
const auto cclSum=MPI_SUM;
#endif
#ifdef DDLA_USE_CUDA
using deviceStream_t = cudaStream_t;
using deviceError_t = cudaError_t;
constexpr auto deviceSuccess = deviceError_t::cudaSuccess;
#define deviceGetErrorString cudaGetErrorString
using deblasStatus_t = cublasStatus_t;
constexpr auto DEBLAS_STATUS_SUCCESS = deblasStatus_t::CUBLAS_STATUS_SUCCESS;
using deblasHandle_t = cublasHandle_t;
using desolverHandle_t = cusolverDnHandle_t;
using desolverStatus_t = cusolverStatus_t;
constexpr auto DESOLVER_STATUS_SUCCESS = desolverStatus_t::CUSOLVER_STATUS_SUCCESS;
#define desolverGetStream cusolverDnGetStream
#define deviceMemcpyAsync cudaMemcpyAsync
#define deviceMemcpy cudaMemcpy
#define deviceMemcpy2DAsync cudaMemcpy2DAsync
#define deviceMallocAsync cudaMallocAsync
#define deviceMalloc cudaMalloc
#define deviceMemsetAsync cudaMemsetAsync
#define deviceFreeAsync cudaFreeAsync
#define deviceFree cudaFree
using deviceDataType_t = cudaDataType_t;
constexpr auto DEVICE_R_64F = deviceDataType_t::CUDA_R_64F;
constexpr auto DEVICE_C_64F = deviceDataType_t::CUDA_C_64F;
constexpr auto DEVICE_R_32F = deviceDataType_t::CUDA_R_32F;
constexpr auto DEVICE_C_32F = deviceDataType_t::CUDA_C_32F;
using derandGenerator_t = curandGenerator_t;
using derandStatus_t = curandStatus_t;
constexpr auto DERAND_STATUS_SUCCESS = derandStatus_t::CURAND_STATUS_SUCCESS;
#define derandCreateGenerator curandCreateGenerator
#define derandSetPseudoRandomGeneratorSeed curandSetPseudoRandomGeneratorSeed
#define derandGenerateUniform curandGenerateUniform
#define derandGenerateUniformDouble curandGenerateUniformDouble
#define derandDestroyGenerator curandDestroyGenerator
using derandRngType = curandRngType;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = derandRngType::CURAND_RNG_PSEUDO_DEFAULT;
#define deviceMemGetInfo cudaMemGetInfo
using deblasSideMode_t = cublasSideMode_t;
constexpr auto DEBLAS_SIDE_LEFT = deblasSideMode_t::CUBLAS_SIDE_LEFT;
constexpr auto DEBLAS_SIDE_RIGHT = deblasSideMode_t::CUBLAS_SIDE_RIGHT;
using deblasFillMode_t = cublasFillMode_t;
constexpr auto DEBLAS_FILL_MODE_LOWER = deblasFillMode_t::CUBLAS_FILL_MODE_LOWER;
constexpr auto DEBLAS_FILL_MODE_UPPER = deblasFillMode_t::CUBLAS_FILL_MODE_UPPER;
using deblasDiagType_t = cublasDiagType_t;
constexpr auto DEBLAS_DIAG_UNIT = deblasDiagType_t::CUBLAS_DIAG_UNIT;
constexpr auto DEBLAS_DIAG_NON_UNIT = deblasDiagType_t::CUBLAS_DIAG_NON_UNIT;
using deblasOperation_t = cublasOperation_t;
constexpr auto DEBLAS_OP_N = deblasOperation_t::CUBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deblasOperation_t::CUBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deblasOperation_t::CUBLAS_OP_C;

#endif
#ifdef DDLA_USE_HIP
using deviceStream_t = hipStream_t;
using deviceError_t = hipError_t;
constexpr auto deviceSuccess = hipError_t::hipSuccess;
#define deviceGetErrorString hipGetErrorString
using deblasStatus_t = hipblasStatus_t;
constexpr auto DEBLAS_STATUS_SUCCESS = deblasStatus_t::HIPBLAS_STATUS_SUCCESS;
using deblasHandle_t = hipblasHandle_t;
using desolverHandle_t = hipsolverHandle_t;
using desolverStatus_t = hipsolverStatus_t;
constexpr auto DESOLVER_STATUS_SUCCESS = desolverStatus_t::HIPSOLVER_STATUS_SUCCESS;
#define desolverGetStream hipsolverGetStream
#define deviceMemcpyAsync hipMemcpyAsync
#define deviceMemcpy hipMemcpy
#define deviceMemcpy2DAsync hipMemcpy2DAsync
#define deviceMallocAsync hipMallocAsync
#define deviceMalloc hipMalloc
#define deviceMemsetAsync hipMemsetAsync
#define deviceFreeAsync hipFreeAsync
#define deviceFree hipFree
using deviceDataType_t = hipDataType;
constexpr auto DEVICE_R_64F = deviceDataType_t::HIP_R_64F;
constexpr auto DEVICE_C_64F = deviceDataType_t::HIP_C_64F;
constexpr auto DEVICE_R_32F = deviceDataType_t::HIP_R_32F;
constexpr auto DEVICE_C_32F = deviceDataType_t::HIP_C_32F;
using derandGenerator_t = hiprandGenerator_t;
using derandStatus_t = hiprandStatus_t;
constexpr auto DERAND_STATUS_SUCCESS = derandStatus_t::HIPRAND_STATUS_SUCCESS;
#define derandCreateGenerator hiprandCreateGenerator
#define derandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define derandGenerateUniform hiprandGenerateUniform
#define derandGenerateUniformDouble hiprandGenerateUniformDouble
#define derandDestroyGenerator hiprandDestroyGenerator
using derandRngType = hiprandRngType;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = derandRngType::HIPRAND_RNG_PSEUDO_DEFAULT;
#define deviceMemGetInfo hipMemGetInfo
using deblasSideMode_t = hipblasSideMode_t;
constexpr auto DEBLAS_SIDE_LEFT = deblasSideMode_t::HIPBLAS_SIDE_LEFT;
constexpr auto DEBLAS_SIDE_RIGHT = deblasSideMode_t::HIPBLAS_SIDE_RIGHT;
using deblasFillMode_t = hipblasFillMode_t;
constexpr auto DEBLAS_FILL_MODE_LOWER = deblasFillMode_t::HIPBLAS_FILL_MODE_LOWER;
constexpr auto DEBLAS_FILL_MODE_UPPER = deblasFillMode_t::HIPBLAS_FILL_MODE_UPPER;
using deblasDiagType_t = hipblasDiagType_t;
constexpr auto DEBLAS_DIAG_UNIT = deblasDiagType_t::HIPBLAS_DIAG_UNIT;
constexpr auto DEBLAS_DIAG_NON_UNIT = deblasDiagType_t::HIPBLAS_DIAG_NON_UNIT;
using deblasOperation_t = hipblasOperation_t;
constexpr auto DEBLAS_OP_N = deblasOperation_t::HIPBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deblasOperation_t::HIPBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deblasOperation_t::HIPBLAS_OP_C;
#endif

#ifdef DDLA_USE_CUDA
using deviceMemcpyKind=cudaMemcpyKind;
constexpr auto deviceMemcpyHostToDevice = deviceMemcpyKind::cudaMemcpyHostToDevice;
constexpr auto deviceMemcpyDeviceToHost = deviceMemcpyKind::cudaMemcpyDeviceToHost;
constexpr auto deviceMemcpyDeviceToDevice = deviceMemcpyKind::cudaMemcpyDeviceToDevice;
#endif
#ifdef DDLA_USE_HIP
using deviceMemcpyKind=hipMemcpyKind;
constexpr auto deviceMemcpyHostToDevice = deviceMemcpyKind::hipMemcpyHostToDevice;
constexpr auto deviceMemcpyDeviceToHost = deviceMemcpyKind::hipMemcpyDeviceToHost;
constexpr auto deviceMemcpyDeviceToDevice = deviceMemcpyKind::hipMemcpyDeviceToDevice;
#endif



inline deviceError_t deviceStreamSynchronize(deviceStream_t stream) {
#ifdef DDLA_USE_CUDA
    return cudaStreamSynchronize(stream);
#else
    return hipStreamSynchronize(stream);
#endif
}


inline deviceError_t deviceDeviceSynchronize(){
#ifdef DDLA_USE_CUDA 
    return cudaDeviceSynchronize();
#else
    return hipDeviceSynchronize();
#endif
}

inline deviceError_t deviceGetDeviceCount(int* count){
#ifdef DDLA_USE_CUDA
    return cudaGetDeviceCount(count);
#else
    return hipGetDeviceCount(count);
#endif
}


static inline void MPI_CHECK(int status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != MPI_SUCCESS)
    {
        fprintf(stderr, "mpi error at %s:%d : %d\n", file, line, status);
        exit(EXIT_FAILURE);
    }
}

static inline void DEVICE_CHECK(deviceError_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != deviceSuccess)
    {
        fprintf(stderr, "device error at %s:%d : %s\n", file, line, deviceGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static inline void BLAS_CHECK(deblasStatus_t err_, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (err_ != DEBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "deblas error %d at %s:%d\n", err_, file, line);
        exit(EXIT_FAILURE);
    }
}

static inline void SOLVER_CHECK(desolverStatus_t err_, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (err_ != DESOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "cusolver error %d at %s:%d\n", err_, file, line);
        exit(EXIT_FAILURE);
    }
}

#ifdef DDLA_USE_CCL
static inline void CCL_CHECK(ncclResult_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != ncclSuccess)
    {
        fprintf(stderr, "nccl error at %s:%d : %d\n", file, line, status);
        exit(EXIT_FAILURE);
    }
}
#else
static inline void CCL_CHECK(int status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    MPI_CHECK(status, file, line);
}
#endif

static inline void DERAND_CHECK(derandStatus_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != DERAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "derand error at %s:%d : %d\n", file, line, status);
        exit(EXIT_FAILURE);
    }
}

void random_generator(void* c_data, const int64_t& lengthOfData, const deviceDataType_t& compute_type);
// col major
void write_matrix(std::complex<double>* A, const int& m,const int& n, const char* filename);

} // namespace DDLA

#endif // DDLA_CONNECTOR_H