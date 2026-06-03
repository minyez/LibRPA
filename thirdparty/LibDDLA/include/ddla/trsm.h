#ifndef TRSM_H
#define TRSM_H

#include "ddla_connector.h"

namespace ddla{


inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag, 
    int m, int n, 
    const std::complex<double>& alpha,
    std::complex<double> *A, int lda,
    std::complex<double> *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasZtrsm(handle, side, uplo, trans, diag, m, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasZtrsm(handle, side, uplo, trans, diag, m, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag, 
    int m, int n, 
    const std::complex<float>& alpha,
    std::complex<float> *A, int lda,
    std::complex<float> *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasCtrsm(handle, side, uplo, trans, diag, m, n, (hipblasComplex*)&alpha, (hipblasComplex*)A, lda, (hipblasComplex*)B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasCtrsm(handle, side, uplo, trans, diag, m, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda, (cuFloatComplex*)B, ldb);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag, 
    int m, int n, 
    const float& alpha,
    float *A, int lda,
    float *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}


inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag, 
    int m, int n, 
    const double& alpha,
    double *A, int lda,
    double *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

}


#endif // TRSM_H