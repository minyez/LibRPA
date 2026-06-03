#ifndef SWAP_H
#define SWAP_H

#include "ddla_connector.h"

namespace ddla{
    inline deblasStatus_t deblasSwap(deblasHandle_t handle, int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy)
    {
        #if defined(DDLA_USE_HIP)
        return hipblasZswap(handle, n, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy);
        #elif defined(DDLA_USE_CUDA)
        return cublasZswap(handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }

    inline deblasStatus_t deblasSwap(deblasHandle_t handle, int n, std::complex<float> *x, int incx, std::complex<float> *y, int incy)
    {
        #if defined(DDLA_USE_HIP)
        return hipblasCswap(handle, n, (hipblasComplex*)x, incx, (hipblasComplex*)y, incy);
        #elif defined(DDLA_USE_CUDA)
        return cublasCswap(handle, n, (cuFloatComplex*)x, incx, (cuFloatComplex*)y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }

    inline deblasStatus_t deblasSwap(deblasHandle_t handle, int n, float *x, int incx, float *y, int incy)
    {
        #if defined(DDLA_USE_HIP)
        return hipblasSswap(handle, n, x, incx, y, incy);
        #elif defined(DDLA_USE_CUDA)
        return cublasSswap(handle, n, x, incx, y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }

    inline deblasStatus_t deblasSwap(deblasHandle_t handle, int n, double *x, int incx, double *y, int incy)
    {
        #if defined(DDLA_USE_HIP)
        return hipblasDswap(handle, n, x, incx, y, incy);
        #elif defined(DDLA_USE_CUDA)
        return cublasDswap(handle, n, x, incx, y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }
}// DDLA


#endif // SWAP_H