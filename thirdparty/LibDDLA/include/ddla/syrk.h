#ifndef DDLA_SYRK_H
#define DDLA_SYRK_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasSyrk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const float& alpha,
    const float* A, int lda,
    const float& beta,
    float* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasSsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasSsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasSyrk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const double& alpha,
    const double* A, int lda,
    const double& beta,
    double* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasDsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

} // namespace ddla

#endif // DDLA_SYRK_H
