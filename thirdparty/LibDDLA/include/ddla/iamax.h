#ifndef IAMAX_H
#define IAMAX_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const float *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIsamax(handle, n, x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIsamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const double *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIdamax(handle, n, x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIdamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const std::complex<float> *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIcamax(handle, n, (cuFloatComplex*)x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIcamax(handle, n, (hipblasComplex*)x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const std::complex<double> *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIzamax(handle, n, (cuDoubleComplex*)x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIzamax(handle, n, (hipblasDoubleComplex*)x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

}


#endif // IAMAX_H