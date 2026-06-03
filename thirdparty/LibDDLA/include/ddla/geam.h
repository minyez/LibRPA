#ifndef GEAM_H
#define GEAM_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    const float& beta,
    const float* B, int ldb,
    float* C, int ldc
    )
{
#if defined(DDLA_USE_CUDA)
    return cublasSgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#elif defined(DDLA_USE_HIP)
    return hipblasSgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    const double& beta,
    const double* B, int ldb,
    double* C, int ldc
    )
{
#if defined(DDLA_USE_CUDA)
    return cublasDgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#elif defined(DDLA_USE_HIP)
    return hipblasDgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>& beta,
    const std::complex<float>* B, int ldb,
    std::complex<float>* C, int ldc
    )
{
    #if defined(DDLA_USE_CUDA)
    return cublasCgeam(
        handle, transA, transB,
        m, n,
        (cuFloatComplex*)&alpha,
        (cuFloatComplex*)A, lda,
        (cuFloatComplex*)&beta,
        (cuFloatComplex*)B, ldb,
        (cuFloatComplex*)C, ldc
    );
    #elif defined(DDLA_USE_HIP)
    return hipblasCgeam(
        handle, transA, transB,
        m, n,
        (hipblasComplex*)&alpha,
        (hipblasComplex*)A, lda,
        (hipblasComplex*)&beta,
        (hipblasComplex*)B, ldb,
        (hipblasComplex*)C, ldc
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>& beta,
    const std::complex<double>* B, int ldb,
    std::complex<double>* C, int ldc
    )
{
#if defined(DDLA_USE_CUDA)
    return cublasZgeam(
        handle, transA, transB,
        m, n,
        (cuDoubleComplex*)&alpha,
        (cuDoubleComplex*)A, lda,
        (cuDoubleComplex*)&beta,
        (cuDoubleComplex*)B, ldb,
        (cuDoubleComplex*)C, ldc
    );
#elif defined(DDLA_USE_HIP)
    return hipblasZgeam(
        handle, transA, transB,
        m, n,
        (hipblasDoubleComplex*)&alpha,
        (hipblasDoubleComplex*)A, lda,
        (hipblasDoubleComplex*)&beta,
        (hipblasDoubleComplex*)B, ldb,
        (hipblasDoubleComplex*)C, ldc
    );
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

}
#endif // GEAM_H