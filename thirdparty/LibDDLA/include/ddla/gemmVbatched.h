#ifndef DDLA_GEMM_VBATCHED_H
#define DDLA_GEMM_VBATCHED_H

#include "ddla_connector.h"
#include "ddla_handle_t.h"

#include <complex>

namespace ddla
{

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

/**
 * Execute a batch of GEMMs whose dimensions and leading dimensions may differ.
 *
 * Dimension, leading-dimension, and matrix-pointer arrays are device resident.
 * The operation is enqueued on handle->stream. A zero batch count is a no-op.
 */
template <typename T>
void gemmVbatched(
    deblasOperation_t transA, deblasOperation_t transB,
    int* d_m, int* d_n, int* d_k,
    T alpha,
    const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta,
    T** d_C_array, int* d_ldc,
    int batch_count,
    const DdlaHandle_t& handle);

/**
 * Execute two dependent variable-size GEMM stages using a reusable temporary.
 *
 * Stage zero writes through d_C_array_0. For each host-resident segment in
 * segment_sizes that temporary pointer array is reused by stage one. Stage-zero
 * beta must be zero, every segment must be positive, and the segment total must
 * equal batch_count.
 */
template <typename T>
void gemmVbatched2s(
    deblasOperation_t transA_0, deblasOperation_t transB_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    T alpha_0,
    const T* const* d_A_array_0, int* d_lda_0,
    const T* const* d_B_array_0, int* d_ldb_0,
    T beta_0,
    T** d_C_array_0, int* d_ldc_0,
    deblasOperation_t transA_1, deblasOperation_t transB_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    T alpha_1,
    const T* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    T beta_1,
    T** d_C_array_1, int* d_ldc_1,
    bool C0_left,
    int batch_count,
    const int* segment_sizes,
    int segment_count,
    const DdlaHandle_t& handle);

extern template void gemmVbatched<float>(
    deblasOperation_t, deblasOperation_t, int*, int*, int*, float,
    const float* const*, int*, const float* const*, int*, float, float**,
    int*, int, const DdlaHandle_t&);
extern template void gemmVbatched<double>(
    deblasOperation_t, deblasOperation_t, int*, int*, int*, double,
    const double* const*, int*, const double* const*, int*, double, double**,
    int*, int, const DdlaHandle_t&);
extern template void gemmVbatched<std::complex<float>>(
    deblasOperation_t, deblasOperation_t, int*, int*, int*,
    std::complex<float>, const std::complex<float>* const*, int*,
    const std::complex<float>* const*, int*, std::complex<float>,
    std::complex<float>**, int*, int, const DdlaHandle_t&);
extern template void gemmVbatched<std::complex<double>>(
    deblasOperation_t, deblasOperation_t, int*, int*, int*,
    std::complex<double>, const std::complex<double>* const*, int*,
    const std::complex<double>* const*, int*, std::complex<double>,
    std::complex<double>**, int*, int, const DdlaHandle_t&);

#endif

} // namespace ddla

#endif
