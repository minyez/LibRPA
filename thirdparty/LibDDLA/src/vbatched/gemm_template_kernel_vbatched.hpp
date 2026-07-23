/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/
#ifndef GEMM_TEMPLATE_KERNEL_VBATCHED_CUH
#define GEMM_TEMPLATE_KERNEL_VBATCHED_CUH

#include "gemm_vbatched_compat.h"
#include "gemm_template_device_defs.hpp"
#include "gemm_template_device.hpp"

template<typename T>
using Func_gemm_template_vbatched_kernel = void(*)(
    int*, int*, int*,
    T const * const *, int, int, int*,
    T const * const *, int, int, int*,
    T              **, int, int, int*,
    T, T,
    int, int, int);


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_vbatched_nn_kernel(
    int* M, int* N, int* K,
    T const * const * Aarray, int Ai, int Aj, int* LDA,
    T const * const * Barray, int Bi, int Bj, int* LDB,
    T               **Carray, int Ci, int Cj, int* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K)
{
    extern __shared__ __align__(16) unsigned char sdata_nn[];

    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < Ai || my_K < Aj ) return;
    if( my_K < Bi || my_N < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Ai, Ci );
    my_N -= max( Bj, Cj );
    my_K -= max( Aj, Bi );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K < 0) return;

    // now either my_M or my_N is +ve, but my_K >= 0
    // check for my_K == 0 && beta == 1, for which C is unchanged
    if(my_K == 0 && beta == make_FloatingPoint(1.,0.)) return;

    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= ddlaVbatchedCeilDiv( my_M, BLK_M ) ) return;
    if( blockIdx.y >= ddlaVbatchedCeilDiv( my_N, BLK_N ) ) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = reinterpret_cast<T*>(sdata_nn); // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_nn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( my_M, my_N, my_K,
      Aarray[batchid] + (int)LDA[batchid] * Aj + Ai, (int)LDA[batchid],
      Barray[batchid] + (int)LDB[batchid] * Bj + Bi, (int)LDB[batchid],
      Carray[batchid] + (int)LDC[batchid] * Cj + Ci, (int)LDC[batchid],
      alpha, beta,
      sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_vbatched_nt_kernel(
    int* M, int* N, int* K,
    T const * const * Aarray, int Ai, int Aj, int* LDA,
    T const * const * Barray, int Bi, int Bj, int* LDB,
    T              ** Carray, int Ci, int Cj, int* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K)
{
    extern __shared__ __align__(16) unsigned char sdata_nt[];

    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < Ai || my_K < Aj ) return;
    if( my_N < Bi || my_K < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Ai, Ci );
    my_N -= max( Bi, Cj );
    my_K -= max( Aj, Bj );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K < 0) return;

    // now either my_M or my_N is +ve, but my_K >= 0
    // check for my_K == 0 && beta == 1, for which C is unchanged
    if(my_K == 0 && beta == make_FloatingPoint(1.,0.)) return;

    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= (my_M+BLK_M-1)/BLK_M ) return;
    if( blockIdx.y >= (my_N+BLK_N-1)/BLK_N ) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = reinterpret_cast<T*>(sdata_nt); // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_nt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( my_M, my_N, my_K,
      Aarray[batchid] + (int)LDA[batchid] * Aj + Ai, (int)LDA[batchid],
      Barray[batchid] + (int)LDB[batchid] * Bj + Bi, (int)LDB[batchid],
      Carray[batchid] + (int)LDC[batchid] * Cj + Ci, (int)LDC[batchid],
      alpha, beta,
      sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_vbatched_tn_kernel(
    int* M, int* N, int* K,
    T const * const * Aarray, int Ai, int Aj, int* LDA,
    T const * const * Barray, int Bi, int Bj, int* LDB,
    T              ** Carray, int Ci, int Cj, int* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K)
{
    extern __shared__ __align__(16) unsigned char sdata_tn[];

    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_K < Ai || my_M < Aj ) return;
    if( my_K < Bi || my_N < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Aj, Ci );
    my_N -= max( Bj, Cj );
    my_K -= max( Ai, Bi );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K < 0) return;

    // now either my_M or my_N is +ve, but my_K >= 0
    // check for my_K == 0 && beta == 1, for which C is unchanged
    if(my_K == 0 && beta == make_FloatingPoint(1.,0.)) return;

    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= (my_M+BLK_M-1)/BLK_M ) return;
    if( blockIdx.y >= (my_N+BLK_N-1)/BLK_N ) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = reinterpret_cast<T*>(sdata_tn); // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_tn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( my_M, my_N, my_K,
      Aarray[batchid] + (int)LDA[batchid] * Aj + Ai, (int)LDA[batchid],
      Barray[batchid] + (int)LDB[batchid] * Bj + Bi, (int)LDB[batchid],
      Carray[batchid] + (int)LDC[batchid] * Cj + Ci, (int)LDC[batchid],
      alpha, beta,
      sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_vbatched_tt_kernel(
    int* M, int* N, int* K,
    T const * const * Aarray, int Ai, int Aj, int* LDA,
    T const * const * Barray, int Bi, int Bj, int* LDB,
    T              ** Carray, int Ci, int Cj, int* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K)
{
    extern __shared__ __align__(16) unsigned char sdata_tt[];

    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_K < Ai || my_M < Aj ) return;
    if( my_N < Bi || my_K < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Aj, Ci );
    my_N -= max( Bi, Cj );
    my_K -= max( Ai, Bj );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K < 0) return;

    // now either my_M or my_N is +ve, but my_K >= 0
    // check for my_K == 0 && beta == 1, for which C is unchanged
    if(my_K == 0 && beta == make_FloatingPoint(1.,0.)) return;

    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= (my_M+BLK_M-1)/BLK_M ) return;
    if( blockIdx.y >= (my_N+BLK_N-1)/BLK_N ) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = reinterpret_cast<T*>(sdata_tt); // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_tt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( my_M, my_N, my_K,
      Aarray[batchid] + (int)LDA[batchid] * Aj + Ai, (int)LDA[batchid],
      Barray[batchid] + (int)LDB[batchid] * Bj + Bi, (int)LDB[batchid],
      Carray[batchid] + (int)LDC[batchid] * Cj + Ci, (int)LDC[batchid],
      alpha, beta,
      sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
// kernel wrappers
// NN
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_nn(
    int max_m, int max_n, int max_k,
    int* m, int* n, int* k,
    T alpha, T const * const * dA_array, int Ai, int Aj, int* ldda,
             T const * const * dB_array, int Bi, int Bj, int* lddb,
    T beta,  T              ** dC_array, int Ci, int Cj, int* lddc,
    int batchCount, ddla::deviceStream_t stream)
{
    size_t shmem = 0;
    int max_batchCount = ddlaVbatchedMaxBatchCount;
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(int i = 0; i < batchCount; i += max_batchCount) {
        int ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( ddlaVbatchedCeilDiv( max_m, BLK_M ), ddlaVbatchedCeilDiv( max_n, BLK_N ), ibatch );

        DDLA_LAUNCH_KERNEL((gemm_template_vbatched_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, dim_vec, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, stream, m+i, n+i, k+i, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, dC_array+i, Ci, Cj, lddc+i, alpha, beta, max_m, max_n, max_k);
    }
}


/******************************************************************************/
// NT, NC
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_nt(
    int max_m, int max_n, int max_k,
    int* m, int* n, int* k,
    T alpha, T const * const * dA_array, int Ai, int Aj, int* ldda,
             T const * const * dB_array, int Bi, int Bj, int* lddb,
    T beta,  T              ** dC_array, int Ci, int Cj, int* lddc,
    int batchCount, ddla::deviceStream_t stream)
{
    size_t shmem = 0;
    int max_batchCount = ddlaVbatchedMaxBatchCount;
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(int i = 0; i < batchCount; i += max_batchCount) {
        int ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( ddlaVbatchedCeilDiv( max_m, BLK_M ), ddlaVbatchedCeilDiv( max_n, BLK_N ), ibatch );

        DDLA_LAUNCH_KERNEL((gemm_template_vbatched_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, dim_vec, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, stream, m+i, n+i, k+i, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, dC_array+i, Ci, Cj, lddc+i, alpha, beta, max_m, max_n, max_k);
    }
}


/******************************************************************************/
// TN, CN
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_tn(
    int max_m, int max_n, int max_k,
    int* m, int* n, int* k,
    T alpha, T const * const * dA_array, int Ai, int Aj, int* ldda,
             T const * const * dB_array, int Bi, int Bj, int* lddb,
    T beta,  T              ** dC_array, int Ci, int Cj, int* lddc,
    int batchCount, ddla::deviceStream_t stream)
{
    size_t shmem = 0;
    int max_batchCount = ddlaVbatchedMaxBatchCount;
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(int i = 0; i < batchCount; i += max_batchCount) {
        int ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( ddlaVbatchedCeilDiv( max_m, BLK_M ), ddlaVbatchedCeilDiv( max_n, BLK_N ), ibatch );

        DDLA_LAUNCH_KERNEL((gemm_template_vbatched_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, dim_vec, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, stream, m+i, n+i, k+i, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, dC_array+i, Ci, Cj, lddc+i, alpha, beta, max_m, max_n, max_k);
    }

}


/******************************************************************************/
// TT, TC, CT, CC
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_tt(
    int max_m, int max_n, int max_k,
    int* m, int* n, int* k,
    T alpha, T const * const * dA_array, int Ai, int Aj, int* ldda,
             T const * const * dB_array, int Bi, int Bj, int* lddb,
    T beta,  T              ** dC_array, int Ci, int Cj, int* lddc,
    int batchCount, ddla::deviceStream_t stream)
{
    size_t shmem = 0;
    int max_batchCount = ddlaVbatchedMaxBatchCount;
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(int i = 0; i < batchCount; i += max_batchCount) {
        int ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( ddlaVbatchedCeilDiv( max_m, BLK_M ), ddlaVbatchedCeilDiv( max_n, BLK_N ), ibatch );

        DDLA_LAUNCH_KERNEL((gemm_template_vbatched_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, dim_vec, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, stream, m+i, n+i, k+i, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, dC_array+i, Ci, Cj, lddc+i, alpha, beta, max_m, max_n, max_k);
    }
}


/******************************************************************************/
// kernel wrappers
template <typename T>
void gemm_template_vbatched(
    Func_gemm_template_vbatched_kernel<T> func_gemm_template_vbatched_kernel,
    int max_m, int max_n, int max_k,
    int* m, int* n, int* k,
    T alpha, T const * const * dA_array, int Ai, int Aj, int* ldda,
             T const * const * dB_array, int Bi, int Bj, int* lddb,
    T beta,  T              ** dC_array, int Ci, int Cj, int* lddc,
    const int DIM_X, const int DIM_Y,
    const int BLK_M, const int BLK_N, const int BLK_K,
    int batchCount, ddla::deviceStream_t stream)
{
    int max_batchCount = ddlaVbatchedMaxBatchCount;
    size_t shmem = 0;
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(int i = 0; i < batchCount; i += max_batchCount) {
        int ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( ddlaVbatchedCeilDiv( max_m, BLK_M ), ddlaVbatchedCeilDiv( max_n, BLK_N ), ibatch );

        ddlaLaunchKernelByPtr<T>(func_gemm_template_vbatched_kernel, dim3(dimGrid), dim3(dimBlock), shmem, stream, m+i, n+i, k+i, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, dC_array+i, Ci, Cj, lddc+i, alpha, beta, max_m, max_n, max_k);
    }
}


/******************************************************************************/
// kernel wrappers
/******************************************************************************/
// kernel wrappers
template <typename T>
void gemm_template_vbatched_2s(
    Func_gemm_template_vbatched_kernel<T> func_gemm_template_vbatched_kernel_0,
    int max_m_0, int max_n_0, int max_k_0,
    int* m_0, int* n_0, int* k_0,
    T alpha_0, T const * const * dA_array_0, int Ai_0, int Aj_0, int* ldda_0,
               T const * const * dB_array_0, int Bi_0, int Bj_0, int* lddb_0,
    T beta_0,  T              ** dC_array_0, int Ci_0, int Cj_0, int* lddc_0,
    const int DIM_X_0, const int DIM_Y_0,
    const int BLK_M_0, const int BLK_N_0, const int BLK_K_0,
    Func_gemm_template_vbatched_kernel<T> func_gemm_template_vbatched_kernel_1,
    int max_m_1, int max_n_1, int max_k_1,
    int* m_1, int* n_1, int* k_1,
    T alpha_1, T const * const * dAB_array_1, int Ai_1, int Aj_1, int* ldda_1,
                                              int Bi_1, int Bj_1, int* lddb_1,
    T beta_1,  T              ** dC_array_1,  int Ci_1, int Cj_1, int* lddc_1,
    const int DIM_X_1, const int DIM_Y_1,
    const int BLK_M_1, const int BLK_N_1, const int BLK_K_1,
    bool C0_left,
    int batchCount, const int* segment_sizes, ddla::deviceStream_t stream)
{
    size_t shmem_0 = 0;
    shmem_0 += (BLK_M_0+1) * BLK_K_0 * sizeof(T);  // sA
    shmem_0 += (BLK_K_0+1) * BLK_N_0 * sizeof(T);  // sB
    size_t shmem_1 = 0;
    shmem_1 += (BLK_M_1+1) * BLK_K_1 * sizeof(T);  // sA
    shmem_1 += (BLK_K_1+1) * BLK_N_1 * sizeof(T);  // sB

    dim3 dimBlock_0(DIM_X_0, DIM_Y_0);
    dim3 dimBlock_1(DIM_X_1, DIM_Y_1);
    for(int i=0, segment=0; i<batchCount; ++segment) {
        int ibatch = min(segment_sizes[segment], batchCount-i);
        dim3 dimGrid_0( ddlaVbatchedCeilDiv( max_m_0, BLK_M_0 ), ddlaVbatchedCeilDiv( max_n_0, BLK_N_0 ), ibatch );
        dim3 dimGrid_1( ddlaVbatchedCeilDiv( max_m_1, BLK_M_1 ), ddlaVbatchedCeilDiv( max_n_1, BLK_N_1 ), ibatch );

        ddlaLaunchKernelByPtr<T>(func_gemm_template_vbatched_kernel_0, dim3(dimGrid_0), dim3(dimBlock_0), shmem_0, stream, m_0+i, n_0+i, k_0+i, dA_array_0+i, Ai_0, Aj_0, ldda_0+i, dB_array_0+i, Bi_0, Bj_0, lddb_0+i, dC_array_0, Ci_0, Cj_0, lddc_0+i, alpha_0, beta_0, max_m_0, max_n_0, max_k_0);

        if(C0_left)
            ddlaLaunchKernelByPtr<T>(func_gemm_template_vbatched_kernel_1, dim3(dimGrid_1), dim3(dimBlock_1), shmem_1, stream, m_1+i, n_1+i, k_1+i, dC_array_0, Ai_1, Aj_1, ldda_1+i, dAB_array_1+i, Bi_1, Bj_1, lddb_1+i, dC_array_1+i, Ci_1, Cj_1, lddc_1+i, alpha_1, beta_1, max_m_1, max_n_1, max_k_1);
        else
            ddlaLaunchKernelByPtr<T>(func_gemm_template_vbatched_kernel_1, dim3(dimGrid_1), dim3(dimBlock_1), shmem_1, stream, m_1+i, n_1+i, k_1+i, dAB_array_1+i, Ai_1, Aj_1, ldda_1+i, dC_array_0, Bi_1, Bj_1, lddb_1+i, dC_array_1+i, Ci_1, Cj_1, lddc_1+i, alpha_1, beta_1, max_m_1, max_n_1, max_k_1);

		i += segment_sizes[segment];
	}
}
#endif //GEMM_TEMPLATE_KERNEL_VBATCHED_CUH
