#ifndef SCAL_H
#define SCAL_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const float& alpha, float *x, int64_t incx)
{
    #if defined(DDLA_USE_CUDA)
    return cublasSscal(handle, n, &alpha, x, incx);
    #elif defined(DDLA_USE_HIP)
    return hipblasSscal(handle, n, &alpha, x, incx);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double& alpha, double *x, int64_t incx)
{
#if defined(DDLA_USE_CUDA)
    return cublasDscal(handle, n, &alpha, x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasDscal(handle, n, &alpha, x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const float& alpha, std::complex<float> *x, int64_t incx)
{
#if defined(DDLA_USE_CUDA)
    return cublasCsscal(handle, n, &alpha, (cuFloatComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasCsscal(handle, n, &alpha, (hipblasComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double& alpha, std::complex<double> *x, int64_t incx)
{
#if defined(DDLA_USE_CUDA)
    return cublasZdscal(handle, n, &alpha, (cuDoubleComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasZdscal(handle, n, &alpha, (hipblasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<float>& alpha, std::complex<float> *x, int64_t incx) {
#if defined(DDLA_USE_CUDA)
    return cublasCscal(handle, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasCscal(handle, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<double>& alpha, std::complex<double> *x, int64_t incx) {
#if defined(DDLA_USE_CUDA)
    return cublasZscal(handle, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasZscal(handle, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

} // namespace ddla

#endif // SCAL_H
