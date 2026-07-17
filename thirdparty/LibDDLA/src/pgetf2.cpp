#include <ddla/ddla.h>
#include <complex>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/ddla_comm.h>
#include <ddla/swap.h>

#include "pgetf2_kernels.h"

namespace ddla{

namespace {

struct MaxLoc {
    double value;
    int index;
};

template <typename T>
struct PivotBroadcast {
    int max_row = 0;
    int max_prow = 0;
    T max_value = T{};
};

} // namespace

// now implement only support m=matrix_m, n = nb_real(<=nb), the the column block must belong to one process in a row
template <typename T>
void pgetf2(
    const int& m, const int& nb_real,
    T* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    (void)m;
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    MPI_Comm row_comm = ddla_handle->row_comm;
    MPI_Comm col_comm = ddla_handle->col_comm;

    #ifdef DDLA_USE_CCL
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif
    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    int myprow = array_descA.myprow();
    int mypcol = array_descA.mypcol();

    int m_loc = array_descA.m_loc();
    int n_loc = array_descA.n_loc();
    int lld = array_descA.lld();
    int nb = array_descA.nb();

    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;
    
    int max_row;
    int max_prow;
    T max_value{};

    T *d_temp;
    const size_t row_buffer_elems = static_cast<size_t>(n_loc > 0 ? n_loc : 1);
    DEVICE_CHECK(deviceMallocAsync(&d_temp, sizeof(T)*row_buffer_elems, stream));

    T *d_temp_peer = nullptr;
    #ifdef DDLA_USE_CCL
    DEVICE_CHECK(deviceMallocAsync(&d_temp_peer, sizeof(T)*row_buffer_elems, stream));
    #endif

    void* d_pivot_workspace = nullptr;
    DEVICE_CHECK(deviceMallocAsync(
        &d_pivot_workspace, detail::pgetf2_pivot_workspace_size<T>(), stream));
    

    int i_loc = array_descA.indx_g2l_r(n_s);
    int j_loc = array_descA.indx_g2l_c(n_s);

    int owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
    int owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

    int mm_row_start = num_loc(n_s, nb, myprow, array_descA.irsrc(), nprows);
    int mm_col_start = num_loc(n_s, nb, mypcol, array_descA.icsrc(), npcols);

    info = 0;
    for(int i_tf2 = 0; i_tf2 < nb_real; i_tf2++){
        // find max_rows and value
        int i_panel, j_panel;
        if(i_loc >= 0)
            i_panel = i_loc + i_tf2;
        else
            i_panel = mm_row_start;
        if(j_loc >= 0)
            j_panel = j_loc + i_tf2;
        else
            j_panel = mm_col_start;
        if(j_loc >= 0){
            MaxLoc local_max{-1.0, -1};
            MaxLoc global_max{-1.0, -1};
            T local_max_value{};
            if(i_panel<m_loc){
                int local_max_index = 0;
                detail::pgetf2_find_local_pivot(
                    d_A + j_panel * lld + i_panel, m_loc - i_panel,
                    d_pivot_workspace, stream,
                    local_max.value, local_max_index, local_max_value);
                const int local_row = i_panel + local_max_index;
                local_max.index = array_descA.indx_l2g_r(local_row);
            }
            MPI_CHECK(MPI_Allreduce(&local_max, &global_max, 1,
                                    MPI_DOUBLE_INT, MPI_MAXLOC, col_comm));
            max_row = global_max.index;
            max_prow = indxg2p(max_row, nb, array_descA.irsrc(), nprows);
            if(myprow == max_prow){
                max_value = local_max_value;
            }
            MPI_CHECK(MPI_Bcast(&max_value, static_cast<int>(sizeof(T)), MPI_BYTE,
                                max_prow, col_comm));
        }

        PivotBroadcast<T> pivot;
        if(mypcol == owner_col){
            pivot.max_row = max_row;
            pivot.max_prow = max_prow;
            pivot.max_value = max_value;
        }
        MPI_CHECK(MPI_Bcast(&pivot, static_cast<int>(sizeof(pivot)), MPI_BYTE, owner_col, row_comm));
        max_row = pivot.max_row;
        max_prow = pivot.max_prow;
        max_value = pivot.max_value;

        int max_loc_row = array_descA.indx_g2l_r(max_row);
        if(myprow == owner_row){
            ipiv[i_panel] = max_row + 1; // 1-based index like fortran
        }
        // exchange rows
        if(owner_row == max_prow){
            if(myprow == owner_row && max_loc_row != i_panel)
                BLAS_CHECK(deblasSwap(
                    blasH, n_loc,
                    d_A + i_panel, lld,
                    d_A + max_loc_row, lld
                ));
        }else{
            if(myprow == owner_row){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp, sizeof(T),
                    d_A + i_panel, lld * sizeof(T),
                    sizeof(T), n_loc,
                    deviceMemcpyDeviceToDevice, stream
                ));
                #ifdef DDLA_USE_CCL
                CCL_CHECK(ncclGroupStart());
                CCL_CHECK(
                    cclSend(d_temp, n_loc, max_prow, col_nccl_comm, stream)
                );
                CCL_CHECK(
                    cclRecv(d_temp_peer, n_loc, max_prow, col_nccl_comm, stream)
                );
                CCL_CHECK(ncclGroupEnd());
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_A + i_panel, lld * sizeof(T),
                    d_temp_peer, sizeof(T),
                    sizeof(T), n_loc,
                    deviceMemcpyDeviceToDevice, stream
                ));
                #else
                CCL_CHECK(
                    cclSend(d_temp, n_loc, max_prow, col_nccl_comm, stream)
                );
                CCL_CHECK(
                    cclRecv(d_temp, n_loc, max_prow, col_nccl_comm, stream)
                );
                BLAS_CHECK(deblasSwap(
                    blasH, n_loc,
                    d_A + i_panel, lld,
                    d_temp, 1
                ));
                #endif
                
            }else if(myprow == max_prow){
                #ifdef DDLA_USE_CCL
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp, sizeof(T),
                    d_A + max_loc_row, lld * sizeof(T),
                    sizeof(T), n_loc,
                    deviceMemcpyDeviceToDevice, stream
                ));
                CCL_CHECK(ncclGroupStart());
                CCL_CHECK(
                    cclSend(d_temp, n_loc, owner_row, col_nccl_comm, stream)
                );
                CCL_CHECK(
                    cclRecv(d_temp_peer, n_loc, owner_row, col_nccl_comm, stream)
                );
                CCL_CHECK(ncclGroupEnd());
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_A + max_loc_row, lld * sizeof(T),
                    d_temp_peer, sizeof(T),
                    sizeof(T), n_loc,
                    deviceMemcpyDeviceToDevice, stream
                ));
                #else
                CCL_CHECK(
                    cclRecv(d_temp, n_loc, owner_row, col_nccl_comm, stream)
                );
                BLAS_CHECK(deblasSwap(
                    blasH, n_loc,
                    d_A + max_loc_row, lld,
                    d_temp, 1
                ));
                CCL_CHECK(
                    cclSend(d_temp, n_loc, owner_row, col_nccl_comm, stream)
                );
                #endif
                
            }
        }
        if(std::abs(max_value) == 0.0){
            info = n_s+i_tf2+1;
            break;
        }
        if(j_loc>=0){
            const T inverse_pivot = T(1.0) / max_value;
            int64_t a_off;
            int length_row;
            
            if(i_loc>=0){
                a_off = (i_panel + 1) + j_panel * lld;
                length_row = m_loc - (i_panel + 1);
            }else{
                a_off = mm_row_start + j_panel * lld;
                length_row = m_loc - mm_row_start;
            }
            int length_col = nb_real - i_tf2 - 1;
            if(myprow == owner_row && length_col>0){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp, 1 * sizeof(T),
                    d_A + i_panel + (j_panel + 1) * lld, lld * sizeof(T),
                    1*sizeof(T), length_col,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
            if(length_col>0)
                CCL_CHECK(cclBcast(d_temp, length_col, owner_row, col_nccl_comm, stream));
            T* d_trailing = length_col > 0 ? d_A + a_off + lld : d_A + a_off;
            detail::pgetf2_scale_update(
                length_row, length_col, inverse_pivot,
                d_A + a_off, d_temp, d_trailing, lld, stream);
        }
    }
    DEVICE_CHECK(deviceFreeAsync(d_temp, stream));
    #ifdef DDLA_USE_CCL
    DEVICE_CHECK(deviceFreeAsync(d_temp_peer, stream));
    #endif
    DEVICE_CHECK(deviceFreeAsync(d_pivot_workspace, stream));
    if(info != 0){
        DEVICE_CHECK(deviceStreamSynchronize(stream));
    }
}

template void pgetf2<float>(
    const int& m, const int& nb_real,
    float* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2<double>(
    const int& m, const int& nb_real,
    double* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2<std::complex<float>>(
    const int& m, const int& nb_real,
    std::complex<float>* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2<std::complex<double>>(
    const int& m, const int& nb_real,
    std::complex<double>* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

} // namespace ddla
