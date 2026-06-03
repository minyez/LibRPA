#ifndef AXPY_H
#define AXPY_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const float& alpha, const float *x, int incx, float *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasSaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const double& alpha, const double *x, int incx, double *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasDaxpy(handle, n, &alpha, x, incx, y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasDaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, std::complex<float> *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasCaxpy(handle, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx, (cuFloatComplex*)y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasCaxpy(handle, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx, (hipblasComplex*)y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasZaxpy(handle, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasZaxpy(handle, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


} // namespace ddla

#endif // AXPY_H
