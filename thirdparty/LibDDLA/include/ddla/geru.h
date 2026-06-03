#ifndef GERU_H
#define GERU_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const float& alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const double& alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy, std::complex<float> *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasCgeru(handle, m, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx, (cuFloatComplex*)y, incy, (cuFloatComplex*)A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasCgeru(handle, m, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx, (hipblasComplex*)y, incy, (hipblasComplex*)A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy, std::complex<double> *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasZgeru(handle, m, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy, (cuDoubleComplex*)A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasZgeru(handle, m, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy, (hipblasDoubleComplex*)A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

} // namespace ddla

#endif // GERU_H