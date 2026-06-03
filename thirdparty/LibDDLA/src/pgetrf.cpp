#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/gemm.h>
#include <ddla/trsm.h>
#include <ddla/ddla_comm.h>

namespace ddla{

template<typename T>
void pgetrf(
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
)
{
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
    int nb_real;
    assert(array_descA.mb()==array_descA.nb());
    int lld = array_descA.lld();

    int m_loc = array_descA.m_loc();
    int n_loc = array_descA.n_loc();

    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    int mm_row_start = 0;
    int mm_col_start = 0;
    int i_loc,j_loc;
    int owner_row,owner_col;

    T *d_temp_block;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_block, sizeof(T)*nb*nb, stream));

    T *d_temp_L;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_L, sizeof(T)*m_loc*nb, stream));

    T *d_temp_U;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_U, sizeof(T)*nb*n_loc, stream));

    DEVICE_CHECK(deviceStreamSynchronize(stream));
    
    MPI_Barrier(MPI_COMM_WORLD);

    double time_for_pgetf2 = 0.0;
    double time_for_other = 0.0;
    // double time_for_max = 0.0;
    // double time_for_swap = 0.0;
    // double time_for_scal = 0.0;
    // double time_for_geru = 0.0;
    // double time_for_local_max = 0.0;
    // double time_for_global_max = 0.0;
    // double time_for_allreduce_device = 0.0;
    // double time_for_allreduce_host = 0.0;

    double start_time;

    for(int n_s=0;n_s<std::min(m,n);n_s+=nb){
        nb_real = std::min(nb, std::min(m,n)-n_s);

        i_loc = array_descA.indx_g2l_r(n_s);
        j_loc = array_descA.indx_g2l_c(n_s);

        owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

        
        // start pgetf2

        start_time = MPI_Wtime();
        pgetf2(
            m, nb_real,
            d_A, n_s, array_descA,
            ipiv, info
        );
        // finish pgetf2
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_pgetf2 += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=nb; // update row start   
        if(j_loc>=0){
            mm_col_start+=nb;
            if(i_loc>=0){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_block, nb_real * sizeof(T),
                    d_A + i_loc + j_loc * lld, lld * sizeof(T),
                    nb_real * sizeof(T), nb_real,
                    deviceMemcpyDeviceToDevice, stream
                ));
                
            }
            if(mm_row_start<m_loc){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_L, (m_loc-mm_row_start) * sizeof(T),
                    d_A + mm_row_start + j_loc * lld, lld * sizeof(T),
                    (m_loc - mm_row_start) * sizeof(T), nb_real,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(mm_row_start<m_loc){
            CCL_CHECK(cclBcast(d_temp_L,(m_loc - mm_row_start) * nb_real,owner_col,row_nccl_comm,stream));
        }
        // broadcast block column
        if(i_loc>=0){
            CCL_CHECK(cclBcast(d_temp_block,nb_real * nb_real,owner_col,row_nccl_comm,stream));
            if(mm_col_start<n_loc){
                BLAS_CHECK(deblasTrsm(
                    blasH, DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER, DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    nb_real, n_loc - mm_col_start, 1.0,
                    d_temp_block, nb_real,
                    d_A + i_loc + mm_col_start * lld, lld)
                );
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_U, nb_real * sizeof(T),
                    d_A + i_loc + mm_col_start * lld, lld * sizeof(T),
                    nb_real * sizeof(T), n_loc - mm_col_start,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }   
        }
        if(mm_col_start<n_loc){
            CCL_CHECK(cclBcast(d_temp_U,nb_real * (n_loc - mm_col_start), owner_row, col_nccl_comm, stream));
        }
        // printf("myid:%d, n_s:%d, update trailing matrix mm_row_start:%d, mm_col_start:%d\n",mpi_comm_global_h.myid,n_s,mm_row_start,mm_col_start);
        if(mm_row_start<m_loc&&mm_col_start<n_loc){
            BLAS_CHECK(deblasGemm(
                blasH, DEBLAS_OP_N, DEBLAS_OP_N,
                m_loc - mm_row_start, n_loc - mm_col_start, nb_real,
                -1.0,
                d_temp_L, m_loc - mm_row_start,
                d_temp_U, nb_real,
                1.0,
                d_A + mm_row_start + mm_col_start * lld, lld
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_other += MPI_Wtime() - start_time;
    }
    info = 0;
    DEVICE_CHECK(deviceFreeAsync(d_temp_block, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_L, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_U, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    printf("myid:%d, pzgetrf time_for_pgetf2:%lf, time_for_other:%lf\n",ddla_handle->myid,time_for_pgetf2,time_for_other);

}

template void pgetrf<float>(
    const int& m, const int& n,
    float* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetrf<double>(
    const int& m, const int& n,
    double* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetrf<std::complex<float>>(
    const int& m, const int& n,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetrf<std::complex<double>>(
    const int& m, const int& n,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

}