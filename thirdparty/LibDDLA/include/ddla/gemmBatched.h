#ifndef GEMM_BATCHED_H
#define GEMM_BATCHED_H


#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const std::complex<double>& alpha,
    const std::complex<double> *const Aarray[], int lda,
    const std::complex<double> *const Barray[], int ldb,
    const std::complex<double>& beta,
    std::complex<double> *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasZgemmBatched(
        handle, transa, transb, m, n, k,
        (hipblasDoubleComplex*)(&alpha),
        (hipblasDoubleComplex**)(Aarray), lda,
        (hipblasDoubleComplex**)(Barray), ldb,
        (hipblasDoubleComplex*)(&beta),
        (hipblasDoubleComplex**)(Carray), ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasZgemmBatched(
        handle, transa, transb, m, n, k,
        (cuDoubleComplex*)(&alpha),
        (cuDoubleComplex**)(Aarray), lda,
        (cuDoubleComplex**)(Barray), ldb,
        (cuDoubleComplex*)(&beta),
        (cuDoubleComplex**)(Carray), ldc,
        batchCount
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

}

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const std::complex<float>& alpha,
    const std::complex<float> *const Aarray[], int lda,
    const std::complex<float> *const Barray[], int ldb,
    const std::complex<float>& beta,
    std::complex<float> *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasCgemmBatched(
        handle, transa, transb, m, n, k,
        (hipblasComplex*)(&alpha),
        (hipblasComplex**)(Aarray), lda,
        (hipblasComplex**)(Barray), ldb,
        (hipblasComplex*)(&beta),
        (hipblasComplex**)(Carray), ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasCgemmBatched(
        handle, transa, transb, m, n, k,
        (cuFloatComplex*)(&alpha),
        (cuFloatComplex**)(Aarray), lda,
        (cuFloatComplex**)(Barray), ldb,
        (cuFloatComplex*)(&beta),
        (cuFloatComplex**)(Carray), ldc,
        batchCount
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

}

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const float& alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float& beta,
    float *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasSgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasSgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

}

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const double& alpha,
    const double *const Aarray[], int lda,
    const double *const Barray[], int ldb,
    const double& beta,
    double *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasDgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

}
}

#endif // GEMM_BATCHED_H