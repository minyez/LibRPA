#ifndef TRMM_H
#define TRMM_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasTrmm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    float* B, int ldb,
    float* C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasStrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasStrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasTrmm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    double* B, int ldb,
    double* C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasDtrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasDtrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasTrmm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    std::complex<float>* B, int ldb,
    std::complex<float>* C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasCtrmm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const cuFloatComplex*>(&alpha),
                       reinterpret_cast<const cuFloatComplex*>(A), lda,
                       reinterpret_cast<cuFloatComplex*>(B), ldb,
                       reinterpret_cast<cuFloatComplex*>(C), ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasCtrmm(handle, side, uplo, trans, diag, m, n,
                        reinterpret_cast<const hipblasComplex*>(&alpha),
                        reinterpret_cast<const hipblasComplex*>(A), lda,
                        reinterpret_cast<hipblasComplex*>(B), ldb,
                        reinterpret_cast<hipblasComplex*>(C), ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasTrmm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    std::complex<double>* B, int ldb,
    std::complex<double>* C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasZtrmm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const cuDoubleComplex*>(&alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<cuDoubleComplex*>(B), ldb,
                       reinterpret_cast<cuDoubleComplex*>(C), ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasZtrmm(handle, side, uplo, trans, diag, m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(&alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                        reinterpret_cast<hipblasDoubleComplex*>(B), ldb,
                        reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

}

#endif // TRMM_H
