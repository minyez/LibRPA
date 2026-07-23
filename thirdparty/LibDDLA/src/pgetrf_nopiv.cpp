#include <ddla/ddla.h>
#include <cassert>
#include <algorithm>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/gemm.h>
#include <ddla/trsm.h>
#include <ddla/ddla_comm.h>

namespace ddla {

/**
 * @brief Distributed LU factorization without pivoting.
 *
 * Right-looking block algorithm: at each step the nb*nb diagonal block is
 * factored with the local getrf_nopiv (single-GPU, no pivoting).  The U and
 * L panels are then computed via trsm and the trailing submatrix is updated
 * via gemm, exactly as in pgetrf_bpiv but without any row swaps.
 *
 * For each panel step k (global column n_s):
 *   1) Factor diagonal block: A11 = L1 * U1          (getrf_nopiv)
 *   2) Extract/broadcast the factored diagonal block (L1+U1)
 *   3) Compute U12:  U12 = L1^{-1} * B             (trsm LEFT/LOWER/UNIT)
 *   4) Compute L21:  L21 = C * U1^{-1}             (trsm RIGHT/UPPER/NON-UNIT)
 *   5) Broadcast U panel down column, L panel across row
 *   6) Update trailing: D <- D - L21 * U12         (gemm)
 */
template<typename T>
void pgetrf_nopiv(
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    int& info  // host
)
{
    assert(m <= array_descA.m() && n <= array_descA.n());
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    #ifdef DDLA_USE_CCL
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    int myprow = array_descA.myprow();
    int mypcol = array_descA.mypcol();

    int nb = array_descA.mb();
    assert(array_descA.mb() == array_descA.nb());
    int lld = array_descA.lld();

    int m_loc = num_loc(m, array_descA.mb(), myprow, array_descA.irsrc(), nprows);
    int n_loc = num_loc(n, array_descA.nb(), mypcol, array_descA.icsrc(), npcols);

    deviceStream_t stream = ddla_handle->stream;
    deblasHandle_t blasH = ddla_handle->blasH;

    int nb_real;
    int mm_row_start = 0;  // local row start of trailing submatrix
    int mm_col_start = 0;  // local col start of trailing submatrix

    // Temp buffers
    T* d_temp_block;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_block, sizeof(T) * nb * nb, stream));
    T* d_temp_L;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_L, sizeof(T) * m_loc * nb, stream));
    T* d_temp_U;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_U, sizeof(T) * nb * n_loc, stream));

    int* d_info = nullptr;
    DEVICE_CHECK(deviceMallocAsync(&d_info, sizeof(int), stream));

    info = 0;

    for (int n_s = 0; n_s < std::min(m, n); n_s += nb) {
        nb_real = std::min(nb, std::min(m, n) - n_s);

        int i_loc = array_descA.indx_g2l_r(n_s);
        int j_loc = array_descA.indx_g2l_c(n_s);

        int owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        int owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

        // ================================================================
        // Step 1: Panel LU -- factor the nb*nb diagonal block in place
        // ================================================================
        // The diagonal block A11 lives entirely on process (owner_row, owner_col).
        // Call the local getrf_nopiv directly on the local device buffer.
        int local_info = 0;
        if (myprow == owner_row && mypcol == owner_col) {
            getrf_nopiv(nb_real, nb_real, d_A + i_loc + j_loc * lld, lld, d_info, ddla_handle);
            DEVICE_CHECK(deviceMemcpyAsync(&local_info, d_info, sizeof(int), deviceMemcpyDeviceToHost, stream));
            DEVICE_CHECK(deviceStreamSynchronize(stream));
        }

        MPI_CHECK(MPI_Bcast(&local_info, 1, MPI_INT, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        if (local_info != 0) {
            // getrf_nopiv returns a 1-based index within the diagonal block;
            // shift it to the global position.
            info = local_info + n_s;
            break;
        }
        if (n_s + nb_real == n)
            break;

        // ================================================================
        // Step 2: Extract/broadcast the factored diagonal block (L1+U1)
        // ================================================================
        if (myprow == owner_row && mypcol == owner_col) {
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_temp_block, nb_real * sizeof(T),
                d_A + i_loc + j_loc * lld, lld * sizeof(T),
                nb_real * sizeof(T), nb_real,
                deviceMemcpyDeviceToDevice, stream
            ));
        }
        if (myprow == owner_row) {
            CCL_CHECK(cclBcast(d_temp_block, nb_real * nb_real, owner_col, row_nccl_comm, stream));
        }
        if (mypcol == owner_col) {
            CCL_CHECK(cclBcast(d_temp_block, nb_real * nb_real, owner_row, col_nccl_comm, stream));
        }

        // ================================================================
        // Step 3: Compute U12 = L1^{-1} * B  (trsm LEFT/LOWER/UNIT)
        // ================================================================
        int right_panel_col_start = (j_loc >= 0) ? (j_loc + nb_real) : mm_col_start;
        if (myprow == owner_row) {
            if (n_loc > right_panel_col_start) {
                T* d_right_panel = d_A + right_panel_col_start * lld + mm_row_start;
                BLAS_CHECK(deblasTrsm(
                    blasH,
                    DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER,
                    DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    nb_real, n_loc - right_panel_col_start, T(1.0),
                    d_temp_block, nb_real,
                    d_right_panel, lld
                ));
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_U, nb_real * sizeof(T),
                    d_right_panel, lld * sizeof(T),
                    nb_real * sizeof(T), n_loc - right_panel_col_start,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
        }

        // ================================================================
        // Step 4: Compute L21 = C * U1^{-1}  (trsm RIGHT/UPPER/NON-UNIT)
        // ================================================================
        int left_panel_row_start = (i_loc >= 0) ? (i_loc + nb_real) : mm_row_start;
        if (mypcol == owner_col) {
            if (m_loc > left_panel_row_start) {
                T* d_left_panel = d_A + mm_col_start * lld + left_panel_row_start;
                BLAS_CHECK(deblasTrsm(
                    blasH,
                    DEBLAS_SIDE_RIGHT, DEBLAS_FILL_MODE_UPPER,
                    DEBLAS_OP_N, DEBLAS_DIAG_NON_UNIT,
                    m_loc - left_panel_row_start, nb_real, T(1.0),
                    d_temp_block, nb_real,
                    d_left_panel, lld
                ));
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_L, (m_loc - left_panel_row_start) * sizeof(T),
                    d_left_panel, lld * sizeof(T),
                    (m_loc - left_panel_row_start) * sizeof(T), nb_real,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
        }

        // ================================================================
        // Step 5: Broadcast U panel down column, L panel across row
        // ================================================================
        if (n_loc > right_panel_col_start) {
            CCL_CHECK(cclBcast(d_temp_U, nb_real * (n_loc - right_panel_col_start), owner_row, col_nccl_comm, stream));
        }
        if (m_loc > left_panel_row_start) {
            CCL_CHECK(cclBcast(d_temp_L, (m_loc - left_panel_row_start) * nb_real, owner_col, row_nccl_comm, stream));
        }

        // ================================================================
        // Step 6: Schur-complement update  D <- D - L21 * U12
        // ================================================================
        int trailing_m = m_loc - left_panel_row_start;
        int trailing_n = n_loc - right_panel_col_start;

        if (trailing_m > 0 && trailing_n > 0) {
            BLAS_CHECK(deblasGemm(
                blasH, DEBLAS_OP_N, DEBLAS_OP_N,
                trailing_m, trailing_n, nb_real,
                T(-1.0),
                d_temp_L, trailing_m,
                d_temp_U, nb_real,
                T(1.0),
                d_A + left_panel_row_start + right_panel_col_start * lld, lld
            ));
        }

        DEVICE_CHECK(deviceStreamSynchronize(stream));

        // Advance local pointers for next panel
        if (i_loc >= 0)
            mm_row_start += nb;
        if (j_loc >= 0)
            mm_col_start += nb;
    }

    DEVICE_CHECK(deviceFreeAsync(d_temp_block, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_L, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_U, stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
}

// Explicit instantiations
template void pgetrf_nopiv<float>(
    const int& m, const int& n,
    float* d_A, const DdlaDesc& array_descA,
    int& info
);
template void pgetrf_nopiv<double>(
    const int& m, const int& n,
    double* d_A, const DdlaDesc& array_descA,
    int& info
);
template void pgetrf_nopiv<std::complex<float>>(
    const int& m, const int& n,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    int& info
);
template void pgetrf_nopiv<std::complex<double>>(
    const int& m, const int& n,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    int& info
);

} // namespace ddla
