/*
 * Compatibility definitions for the MAGMA-derived variable-batched GEMM
 * kernels in this directory.
 *
 * The kernels retain their original MAGMA BSD license and attribution. This
 * header deliberately provides only the small type and arithmetic surface
 * needed to compile those kernels against CUDA or HIP and LibDDLA; it does
 * not include or link MAGMA.
 */
#pragma once

#if defined(DDLA_USE_CUDA) && defined(DDLA_USE_HIP)
#error "DDLA_USE_CUDA and DDLA_USE_HIP are mutually exclusive"
#endif
#if !defined(DDLA_USE_CUDA) && !defined(DDLA_USE_HIP)
#error "Exactly one of DDLA_USE_CUDA or DDLA_USE_HIP must be defined"
#endif

#include <ddla/ddla_connector.h>

#include <thrust/complex.h>

// ---------------------------------------------------------------------------
// Kernel launch compatibility layer
// ---------------------------------------------------------------------------
#define DDLA_UNPAREN_KERNEL(...) __VA_ARGS__

#ifdef DDLA_USE_HIP

#define DDLA_LAUNCH_KERNEL(kernel_tuple, grid, block, shmem, stream, ...)       \
    do {                                                                        \
        hipLaunchKernelGGL(                                                     \
            HIP_KERNEL_NAME(DDLA_UNPAREN_KERNEL kernel_tuple),                 \
            grid, block, shmem, stream, __VA_ARGS__);                          \
        ddla::DEVICE_CHECK(deviceGetLastError());                              \
    } while (false)

#else // DDLA_USE_CUDA

#define DDLA_LAUNCH_KERNEL(kernel_tuple, grid, block, shmem, stream, ...)       \
    do {                                                                        \
        DDLA_UNPAREN_KERNEL kernel_tuple<<<grid, block, shmem, stream>>>(       \
            __VA_ARGS__);                                                       \
        ddla::DEVICE_CHECK(deviceGetLastError());                              \
    } while (false)

#endif

using DdlaFloatComplex = thrust::complex<float>;
using DdlaDoubleComplex = thrust::complex<double>;

constexpr int ddlaVbatchedMaxBatchCount = 65535;

__host__ __device__ constexpr int ddlaVbatchedCeilDiv(
    int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <typename Real>
__host__ __device__ inline thrust::complex<Real> ddlaComplexMake(
    Real real, Real imag)
{
    return thrust::complex<Real>(real, imag);
}

template <typename Real>
__host__ __device__ inline thrust::complex<Real> ddlaComplexAdd(
    thrust::complex<Real> lhs, thrust::complex<Real> rhs)
{
    return ddlaComplexMake<Real>(
        lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

template <typename Real>
__host__ __device__ inline thrust::complex<Real> ddlaComplexMul(
    thrust::complex<Real> lhs, thrust::complex<Real> rhs)
{
    return ddlaComplexMake<Real>(
        lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
        lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

template <typename Real>
__host__ __device__ inline thrust::complex<Real> ddlaComplexDiv(
    thrust::complex<Real> lhs, thrust::complex<Real> rhs)
{
    const Real denominator =
        rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
    return ddlaComplexMake<Real>(
        (lhs.real() * rhs.real() + lhs.imag() * rhs.imag()) / denominator,
        (lhs.imag() * rhs.real() - lhs.real() * rhs.imag()) / denominator);
}

template <typename Real>
__host__ __device__ inline thrust::complex<Real> ddlaComplexConj(
    thrust::complex<Real> value)
{
    return thrust::conj(value);
}

__host__ __device__ inline float ddlaScalarFma(float a, float b, float c)
{
    return fmaf(a, b, c);
}

__host__ __device__ inline double ddlaScalarFma(double a, double b, double c)
{
    return fma(a, b, c);
}

template <typename Real>
__host__ __device__ inline thrust::complex<Real> ddlaComplexFma(
    thrust::complex<Real> lhs, thrust::complex<Real> rhs,
    thrust::complex<Real> addend)
{
    return ddlaComplexMake<Real>(
        ddlaScalarFma(-lhs.imag(), rhs.imag(),
                      ddlaScalarFma(lhs.real(), rhs.real(), addend.real())),
        ddlaScalarFma(lhs.imag(), rhs.real(),
                      ddlaScalarFma(lhs.real(), rhs.imag(), addend.imag())));
}

#define DDLA_C_MAKE(real, imag) ddlaComplexMake<float>((real), (imag))
#define DDLA_C_ADD(lhs, rhs) ddlaComplexAdd((lhs), (rhs))
#define DDLA_C_MUL(lhs, rhs) ddlaComplexMul((lhs), (rhs))
#define DDLA_C_DIV(lhs, rhs) ddlaComplexDiv((lhs), (rhs))
#define DDLA_C_CONJ(value) ddlaComplexConj((value))

#define DDLA_Z_MAKE(real, imag) ddlaComplexMake<double>((real), (imag))
#define DDLA_Z_ADD(lhs, rhs) ddlaComplexAdd((lhs), (rhs))
#define DDLA_Z_MUL(lhs, rhs) ddlaComplexMul((lhs), (rhs))
#define DDLA_Z_DIV(lhs, rhs) ddlaComplexDiv((lhs), (rhs))
#define DDLA_Z_CONJ(value) ddlaComplexConj((value))

// ---------------------------------------------------------------------------
// Function-pointer kernel launch (used by gemm_template_vbatched wrappers)
// ---------------------------------------------------------------------------
template <typename T>
using DdlaVbatchedKernelPtr = void (*)(
    int*, int*, int*,
    T const* const*, int, int, int*,
    T const* const*, int, int, int*,
    T**, int, int, int*,
    T, T,
    int, int, int);

template <typename T>
inline void ddlaLaunchKernelByPtr(
    DdlaVbatchedKernelPtr<T> func,
    dim3 grid, dim3 block, size_t shmem, ddla::deviceStream_t stream,
    int* m, int* n, int* k,
    T const* const* A, int Ai, int Aj, int* lda,
    T const* const* B, int Bi, int Bj, int* ldb,
    T** C, int Ci, int Cj, int* ldc,
    T alpha, T beta,
    int max_M, int max_N, int max_K)
{
#ifdef DDLA_USE_HIP
    hipLaunchKernelGGL(func, grid, block, shmem, stream,
                       m, n, k, A, Ai, Aj, lda, B, Bi, Bj, ldb,
                       C, Ci, Cj, ldc, alpha, beta, max_M, max_N, max_K);
#else
    void* args[] = {
        &m, &n, &k,
        (void*)&A, &Ai, &Aj, &lda,
        (void*)&B, &Bi, &Bj, &ldb,
        (void*)&C, &Ci, &Cj, &ldc,
        &alpha, &beta,
        &max_M, &max_N, &max_K};
    ddla::DEVICE_CHECK(cudaLaunchKernel(
        reinterpret_cast<const void*>(func),
        grid, block, args, shmem, stream));
#endif
    ddla::DEVICE_CHECK(deviceGetLastError());
}
