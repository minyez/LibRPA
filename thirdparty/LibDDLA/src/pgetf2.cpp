#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/scal.h>
#include <ddla/geru.h>
#include <ddla/ddla_comm.h>
#include <ddla/iamax.h>
#include <ddla/swap.h>

namespace ddla{

// now implement only support m=matrix_m, n = nb_real(<=nb), the the column block must belong to one process in a row
template <typename T>
void pgetf2(
    const int& m, const int& nb_real,
    T* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
)
{
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
    T max_value;

    T *d_temp;
    DEVICE_CHECK(deviceMallocAsync(&d_temp, sizeof(T)*n_loc, stream));

    std::vector<int> h_id_max(nprows,0); // host

    T *d_max;
    DEVICE_CHECK(deviceMallocAsync(&d_max, sizeof(T)*nprows, stream));

    #ifdef DDLA_USE_DEBUG
    double time_for_max = 0.0;
    double time_for_swap = 0.0;
    double time_for_scal = 0.0;
    double time_for_geru = 0.0;
    double time_for_local_max = 0.0;
    double time_for_global_max = 0.0;
    double time_for_allreduce_device = 0.0;
    double time_for_allreduce_host = 0.0;
    double start_time;
    #endif
    

    int i_loc = array_descA.indx_g2l_r(n_s);
    int j_loc = array_descA.indx_g2l_c(n_s);

    int owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
    int owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

    int mm_row_start = num_loc(n_s, nb, myprow, array_descA.irsrc(), nprows);
    int mm_col_start = num_loc(n_s, nb, mypcol, array_descA.icsrc(), npcols);

    // start pgetf2
    #ifdef DDLA_USE_DEBUG
    start_time = MPI_Wtime();
    #endif
    // printf("start tf2, nprows:%d, npcols:%d\n",nprows, npcols);
    for(int i_tf2 = 0; i_tf2 < nb_real; i_tf2++){
        #ifdef DDLA_USE_DEBUG
        double start_time_tf2 = MPI_Wtime();
        #endif
        DEVICE_CHECK(deviceMemsetAsync(d_max,0,nprows*sizeof(T),stream));
        memset(h_id_max.data(),0,nprows*sizeof(int));
        
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
            #ifdef DDLA_USE_DEBUG
            double start_time_local_max = MPI_Wtime();
            #endif
            if(i_panel<m_loc){
                BLAS_CHECK(deblasIamax(
                    blasH, m_loc-i_panel,
                    d_A + j_panel * lld + i_panel,1,
                    h_id_max.data()+myprow
                ));
                DEVICE_CHECK(deviceMemcpyAsync(
                    d_max+myprow, d_A + (i_panel + (h_id_max[myprow]-1) + j_panel * lld), sizeof(T),
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
            #ifdef DDLA_USE_DEBUG
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            time_for_local_max += MPI_Wtime() - start_time_local_max;
            double start_time_allreduce = MPI_Wtime();
            #endif
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            // printf("before nccl all reduce\n");
            CCL_CHECK(cclAllReduce(d_max, d_max, nprows, cclSum, col_nccl_comm, stream));
            // printf("after nccl all reduce\n");
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            #ifdef DDLA_USE_DEBUG
            
            time_for_allreduce_device += MPI_Wtime() - start_time_allreduce;
            double start_time_global_max = MPI_Wtime();
            #endif
            // printf("before get max after synchronize\n");
            BLAS_CHECK(deblasIamax(blasH, nprows, d_max, 1, &max_prow));
            
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            // printf("after get max\n");
            #ifdef DDLA_USE_DEBUG
            
            time_for_global_max += MPI_Wtime() - start_time_global_max;
            double start_time_allreduce_host = MPI_Wtime();
            #endif
            max_prow--;
            h_id_max[myprow]=array_descA.indx_l2g_r(i_panel+h_id_max[myprow]-1);
            // printf("before mpi all reduce\n");
            MPI_Allreduce(MPI_IN_PLACE,h_id_max.data(),nprows,MPI_INT,MPI_SUM,col_comm);
            #ifdef DDLA_USE_DEBUG
            time_for_allreduce_host += MPI_Wtime() - start_time_allreduce_host;
            #endif
            max_row = h_id_max[max_prow];
            
            DEVICE_CHECK(deviceMemcpyAsync(
                &max_value, d_max+max_prow, sizeof(T),
                deviceMemcpyDeviceToHost, stream
            ));
        }
        #ifdef DDLA_USE_DEBUG
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_max += MPI_Wtime() - start_time_tf2;
        #endif
        // printf("before mpi bcast 1\n");
        MPI_Bcast(&max_row, 1, MPI_INT, owner_col, row_comm);
        int max_loc_row = array_descA.indx_g2l_r(max_row);
        if(myprow == owner_row){
            ipiv[i_panel] = max_row + 1; // 1-based index like fortran
        }
        // printf("before mpi bcast 2\n");
        MPI_Bcast(&max_prow, 1, MPI_INT, owner_col, row_comm);
        #ifdef DDLA_USE_DEBUG
        start_time_tf2 = MPI_Wtime();
        #endif
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
                // printf("before ccl owner_row send 1\n");
                CCL_CHECK(
                    cclSend(d_temp, n_loc, max_prow, col_nccl_comm, stream)
                );
                // printf("before ccl owner_row recv 1\n");
                CCL_CHECK(
                    cclRecv(d_temp, n_loc, max_prow, col_nccl_comm, stream)
                );
                BLAS_CHECK(deblasSwap(
                    blasH, n_loc,
                    d_A + i_panel, lld,
                    d_temp, 1
                ));
                
            }else if(myprow == max_prow){
                // printf("before ccl max_prow send 1\n");
                CCL_CHECK(
                    cclRecv(d_temp, n_loc, owner_row, col_nccl_comm, stream)
                );
                BLAS_CHECK(deblasSwap(
                    blasH, n_loc,
                    d_A + max_loc_row, lld,
                    d_temp, 1
                ));
                // printf("before ccl max_prow recv 1\n");
                CCL_CHECK(
                    cclSend(d_temp, n_loc, owner_row, col_nccl_comm, stream)
                );
                
            }
        }
        #ifdef DDLA_USE_DEBUG
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_swap += MPI_Wtime() - start_time_tf2;
        start_time_tf2 = MPI_Wtime();
        #endif
        // printf("before get max value\n");
        // finish exchange rows
        MPI_Bcast(&max_value, sizeof(T), MPI_BYTE, owner_col, row_comm);
        if(std::abs(max_value)<1e-10){
            info = n_s+i_tf2+1;
            return;
        }
        // start reduce columns
        if(j_loc>=0){
            max_value = (T)1.0 / max_value; // inverse
            int64_t a_off;
            int length_row;
            
            if(i_loc>=0){
                a_off = (i_panel + 1) + j_panel * lld;
                length_row = m_loc - (i_panel + 1);
            }else{
                a_off = mm_row_start + j_panel * lld;
                length_row = m_loc - mm_row_start;
            }
            if(length_row>0){
                BLAS_CHECK(deblasScal(
                    blasH, length_row,
                    max_value,
                    d_A + a_off, 1
                ));
            }
            #ifdef DDLA_USE_DEBUG
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            time_for_scal += MPI_Wtime() - start_time_tf2;
            start_time_tf2 = MPI_Wtime();
            #endif
            int length_col = nb_real - i_tf2 - 1;
            if(myprow == owner_row){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp, 1 * sizeof(T),
                    d_A + i_panel + (j_panel + 1) * lld, lld * sizeof(T),
                    1*sizeof(T), length_col,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
            if(length_col>0)
                cclBcast(d_temp, length_col, owner_row, col_nccl_comm, stream);
            // finish reduce columns
            // start update trailing matrix
            
            if(length_row>0&&length_row>0){
                BLAS_CHECK(deblasGeru(
                    blasH, length_row, length_col,
                    -1.0,
                    d_A + a_off, 1,
                    d_temp, 1,
                    d_A + a_off + lld, lld
                ));
            }
            #ifdef DDLA_USE_DEBUG
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            time_for_geru += MPI_Wtime() - start_time_tf2;
            #endif
        }

        DEVICE_CHECK(deviceStreamSynchronize(stream));
    }
    // #ifdef DDLA_USE_DEBUG
    // printf("myid:%d, max:%f, swap:%f, scal:%f, geru:%f, local max:%f,"
    //         "global max:%f, reduce device:%f, reduce host:%f\n",
    //         ddla_handle->myid, time_for_max, time_for_swap, time_for_geru, time_for_scal,
    //         time_for_local_max, time_for_global_max, time_for_allreduce_device, time_for_allreduce_host);
    // #endif
    // finish pgetf2
    DEVICE_CHECK(deviceFreeAsync(d_temp, stream));
    DEVICE_CHECK(deviceFreeAsync(d_max, stream));
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