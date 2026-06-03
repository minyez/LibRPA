#ifndef GEMM_H
#define GEMM_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const float& alpha, 
    const float *A, int lda, 
    const float *B, int ldb,
    const float& beta,
    float *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const double& alpha, 
    const double *A, int lda, 
    const double *B, int ldb,
    const double& beta,
    double *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const std::complex<float>& alpha, 
    const std::complex<float> *A, int lda, 
    const std::complex<float> *B, int ldb,
    const std::complex<float>& beta,
    std::complex<float> *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasCgemm(handle, transa, transb, m, n, k, (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda, (cuFloatComplex*)B, ldb, (cuFloatComplex*)&beta, (cuFloatComplex*)C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasCgemm(handle, transa, transb, m, n, k, (hipblasComplex*)&alpha, (hipblasComplex*)A, lda, (hipblasComplex*)B, ldb, (hipblasComplex*)&beta, (hipblasComplex*)C, ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif

}

inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const std::complex<double>& alpha, 
    const std::complex<double> *A, int lda, 
    const std::complex<double> *B, int ldb,
    const std::complex<double>& beta,
    std::complex<double> *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasZgemm(handle, transa, transb, m, n, k, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasZgemm(handle, transa, transb, m, n, k, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb, (hipblasDoubleComplex*)&beta, (hipblasDoubleComplex*)C, ldc);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


} // namespace ddla

#endif // GEMM_H