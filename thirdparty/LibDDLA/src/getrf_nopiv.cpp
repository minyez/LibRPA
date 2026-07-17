#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/gemm.h>
#include <ddla/trsm.h>
#include <thrust/complex.h>
#include <algorithm>
#include <complex>
#include <type_traits>

// Maximum panel width for the register-based getf2_nopiv kernel.
// Each thread stores one row of the panel in registers, so this
// must be small enough to avoid register spilling.
// MAGMA uses n <= 32 for their register-cached nopiv kernel.
#define GETF2_MAX_N 32

namespace ddla {

// --------------------------------------------------------------------------
// Zero-pivot check helpers (device)
// --------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ bool is_zero_nopiv(T val) {
    return val == T(0);
}

template <typename T>
__device__ __forceinline__ bool is_zero_nopiv(thrust::complex<T> val) {
    return thrust::abs(val) == T(0);
}

// --------------------------------------------------------------------------
// Device helper: unblocked LU factorization without pivoting
//
// Follows MAGMA's zgetf2_nopiv_device pattern:
//   - Each thread processes one row (coalesced load/store)
//   - Matrix data stays in registers during factorization
//   - Shared memory used only for broadcasting pivot column
//   - Zero-pivot protection: divisor = 1 when pivot == 0
//
// This is called from the kernel with n <= GETF2_MAX_N.
// --------------------------------------------------------------------------

template <typename T, int MAX_N>
__device__ __forceinline__ void
getf2_nopiv_device(int m, int n, T* dA, int ldda, int* info,
                   int tx, T* sx, int gbstep)
{
    // Register array for one row.  n is a runtime value <= MAX_N,
    // but the compiler can partially unroll when n is known at launch.
    T rA[MAX_N];
    // Preserve the first non-zero info reported by earlier panels.
    if (gbstep > 0 && *info != 0) {
        return;
    }


    // Load one row from global memory (coalesced: adjacent threads
    // access adjacent addresses in the same column)
#ifdef DDLA_USE_CUDA
    #pragma unroll
#endif
    for (int i = 0; i < MAX_N; ++i) {
        if (i < n)
            rA[i] = dA[i * ldda + tx];
    }

    int linfo = 0;

    for (int col = 0; col < n; ++col) {
        // Broadcast pivot column via shared memory:
        // thread col copies its row [col..n-1] to shared mem.
        if (tx == col) {
#ifdef DDLA_USE_CUDA
            #pragma unroll
#endif
            for (int j = col; j < MAX_N; ++j) {
                if (j < n)
                    sx[j - col] = rA[j];
            }
        }
        __syncthreads();

        // Check for zero pivot (info is 1-based, global index)
        if (is_zero_nopiv(sx[0]) && linfo == 0)
            linfo = gbstep + col + 1;

        // Zero-pivot protection: use 1 instead of 0 to avoid NaN/Inf.
        // When pivot is zero, the arithmetic has no effect (multiplying by 1).
        T divisor = is_zero_nopiv(sx[0]) ? T(1) : sx[0];
        T reg_inv = T(1) / divisor;

        // Rank-1 update: for rows strictly below the pivot row
        if (tx > col && tx < m) {
            // Scale column col: store L factor (below diagonal)
            rA[col] *= reg_inv;
            T factor = rA[col];

            // GER: update trailing part of this row
            //      rA[j] -= factor * sx[j-col]  for j = col+1 .. n-1
#ifdef DDLA_USE_CUDA
            #pragma unroll
#endif
            for (int j = col + 1; j < MAX_N; ++j) {
                if (j < n)
                    rA[j] -= factor * sx[j - col];
            }
        }
        __syncthreads();
    }

    // Store result back to global memory (coalesced)
#ifdef DDLA_USE_CUDA
    #pragma unroll
#endif
    for (int i = 0; i < MAX_N; ++i) {
        if (i < n)
            dA[i * ldda + tx] = rA[i];
    }

    // Write info back (only thread 0)
    if (tx == 0) {
        (*info) = linfo;
    }
}

// --------------------------------------------------------------------------
// Unblocked panel kernel (launched with 1 block)
//
// Each thread handles one row.  Shared memory size = n * sizeof(T)
// is allocated dynamically at kernel launch.
// --------------------------------------------------------------------------

template <typename T>
__global__ void
getf2_nopiv_kernel(int m, int n, T* dA, int ldda, int* info, int gbstep)
{
    extern __shared__ char sdata[];
    T* sx = reinterpret_cast<T*>(sdata);

    const int tx = threadIdx.x;

    getf2_nopiv_device<T, GETF2_MAX_N>(m, n, dA, ldda, info, tx, sx, gbstep);
}

// --------------------------------------------------------------------------
// Blocked getrf_nopiv (pure GPU, right-looking Level-3 BLAS)
//
// Algorithm (matches MAGMA magma_zgetrf_nopiv):
//   for j = 0 to min(m,n) step nb:
//      jb = min(nb, min(m,n)-j)
//
//      1) Factor diagonal block: A(j:j+jb, j:j+jb) = L*U  (getf2)
//      2) If m > j+jb:
//         Solve L panel: A(j+jb:m, j:j+jb) = A(j+jb:m, j:j+jb) * U^{-1}  (trsm)
//      3) If n > j+jb:
//         Solve U panel: A(j:j+jb, j+jb:n) = L^{-1} * A(j:j+jb, j+jb:n)  (trsm)
//      4) If m > j+jb and n > j+jb:
//         Update trailing: A(j+jb:m, j+jb:n) -= L_panel * U_panel  (gemm)
//
// The panel factorization (step 1) is done by the getf2_nopiv kernel,
// which keeps data in registers and uses shared memory for pivot broadcast.
// The trailing update (step 4) is done by the highly optimized cuBLAS/hipBLAS
// gemm, which dominates the runtime for large matrices.
// --------------------------------------------------------------------------

template <typename T>
void getrf_nopiv(int m, int n, T* d_A, int lda, int* d_info, const DdlaHandle_t& ddla_handle)
{
    deviceStream_t stream = ddla_handle->stream;
    deblasHandle_t blas_handle = ddla_handle->blasH;

    // Block size for the blocked algorithm.  Must be <= GETF2_MAX_N
    // because the panel kernel stores one row in registers.
    // GETF2_MAX_N = 32 is chosen to balance register pressure
    // (32 * sizeof(T) per thread) with arithmetic intensity.
    const int nb = GETF2_MAX_N;

    // Initialize d_info to 0 on device (caller owns d_info memory)
    DEVICE_CHECK(deviceMemsetAsync(d_info, 0, sizeof(int), stream));

    for (int j = 0; j < std::min(m, n); j += nb)
    {
        int jb = std::min(nb, std::min(m, n) - j);
        int mm = m - j;          // rows remaining in the panel
        int nn = n - j - jb;     // cols in the trailing submatrix

        // ---------------------------------------------------------------
        // 1) Factor the diagonal block A(j:j+jb, j:j+jb)
        //
        // Launch: 1 block, jb threads, jb * sizeof(T) shared memory.
        // Each thread handles one row of the jb x jb block.
        // ---------------------------------------------------------------
        int threads = std::min(jb, 512);
        int shmem   = jb * sizeof(T);

        if constexpr (std::is_same_v<T, std::complex<float>>)
        {
            getf2_nopiv_kernel<thrust::complex<float>>
                <<<1, threads, shmem, stream>>>(
                    jb, jb,
                    reinterpret_cast<thrust::complex<float>*>(d_A + j * lda + j),
                    lda, d_info, j);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
            getf2_nopiv_kernel<thrust::complex<double>>
                <<<1, threads, shmem, stream>>>(
                    jb, jb,
                    reinterpret_cast<thrust::complex<double>*>(d_A + j * lda + j),
                    lda, d_info, j);
        }
        else
        {
            getf2_nopiv_kernel<T>
                <<<1, threads, shmem, stream>>>(
                    jb, jb, d_A + j * lda + j, lda, d_info, j);
        }

        // ---------------------------------------------------------------
        // 2) Solve L panel: A(j+jb:m-1, j:j+jb) = A(j+jb:m-1, j:j+jb) * U^{-1}
        //
        // The subdiagonal block A(j+jb:m, j:j+jb) now contains the
        // raw matrix values.  We need to compute L21 = A21 * U^{-1}
        // where U is the upper triangular factor from step 1.
        //
        // trsm(Right, Upper, NoTrans, NonUnit, m-jb, jb, 1, U, A21)
        // ---------------------------------------------------------------
        if (j + jb < m)
        {
            T one = T(1);
            BLAS_CHECK(deblasTrsm(
                blas_handle,
                DEBLAS_SIDE_RIGHT,          // op(A)*X = B  =>  X = B*inv(A)
                DEBLAS_FILL_MODE_UPPER,     // U is upper triangular
                DEBLAS_OP_N,                // no transpose
                DEBLAS_DIAG_NON_UNIT,       // U has non-unit diagonal
                mm - jb, jb,                // (m-jb) rows, jb cols
                one,
                d_A + j * lda + j,          // U factor (jb x jb)
                lda,
                d_A + j * lda + (j + jb),   // A21 input, L21 output
                lda
            ));
        }

        // ---------------------------------------------------------------
        // 3) Solve U panel: A(j:j+jb, j+jb:n-1) = L^{-1} * A(j:j+jb, j+jb:n-1)
        //
        // The block row A(j:j+jb, j+jb:n) now contains the raw values.
        // We need U12 = inv(L) * A12 where L is the lower triangular
        // factor (with unit diagonal) from step 1.
        //
        // trsm(Left, Lower, NoTrans, Unit, jb, n-jb, 1, L, A12)
        // ---------------------------------------------------------------
        if (j + jb < n)
        {
            T one = T(1);
            BLAS_CHECK(deblasTrsm(
                blas_handle,
                DEBLAS_SIDE_LEFT,           // op(A)*X = B  =>  X = inv(A)*B
                DEBLAS_FILL_MODE_LOWER,     // L is lower triangular
                DEBLAS_OP_N,                // no transpose
                DEBLAS_DIAG_UNIT,           // L has unit diagonal
                jb, nn,                     // jb rows, (n-jb) cols
                one,
                d_A + j * lda + j,          // L factor (jb x jb, implicit unit diag)
                lda,
                d_A + (j + jb) * lda + j,   // A12 input, U12 output
                lda
            ));

            // -----------------------------------------------------------
            // 4) Update trailing submatrix
            //
            // A(j+jb:m, j+jb:n) -= L_panel(j+jb:m, j:j+jb) * U_panel(j:j+jb, j+jb:n)
            //
            // gemm(NoTrans, NoTrans, m-jb, n-jb, jb, -1, L_panel, U_panel, 1, A22)
            // -----------------------------------------------------------
            if (j + jb < m)
            {
                T neg_one = T(-1);
                BLAS_CHECK(deblasGemm(
                    blas_handle,
                    DEBLAS_OP_N, DEBLAS_OP_N,
                    mm - jb, nn, jb,
                    neg_one,
                    d_A + j * lda + (j + jb),      // L_panel (m-jb x jb)
                    lda,
                    d_A + (j + jb) * lda + j,      // U_panel (jb x n-jb)
                    lda,
                    one,
                    d_A + (j + jb) * lda + (j + jb), // A22 trailing (m-jb x n-jb)
                    lda
                ));
            }
        }
    }
    // d_info stays on device; caller owns the memory and is responsible
    // for reading it back after stream synchronization.
}

// --------------------------------------------------------------------------
// Explicit instantiations
// --------------------------------------------------------------------------

template void getrf_nopiv<float>(int, int, float*, int, int*, const DdlaHandle_t&);
template void getrf_nopiv<double>(int, int, double*, int, int*, const DdlaHandle_t&);
template void getrf_nopiv<std::complex<float>>(int, int, std::complex<float>*, int, int*, const DdlaHandle_t&);
template void getrf_nopiv<std::complex<double>>(int, int, std::complex<double>*, int, int*, const DdlaHandle_t&);

} // namespace ddla
