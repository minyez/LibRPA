#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/gemm.h>
#include <ddla/trsm.h>
#include <ddla/ddla_comm.h>

namespace ddla{

template <typename T>

void pgetf2_panel(
    const int& m, const int& nb_real,
    T* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    #ifdef DDLA_USE_CCL
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    int myprow = array_descA.myprow();
    int mypcol = array_descA.mypcol();

    int nb = array_descA.mb();
    assert(array_descA.mb()==array_descA.nb());
    int lld = array_descA.lld();

    int m_loc = array_descA.m_loc();
    int n_loc = array_descA.n_loc();

    const int panel = std::min(32, nb/2>0?nb/2:1);
    int panel_real;

    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    int mm_row_start = num_loc(n_start, nb, myprow, array_descA.irsrc(), nprows);
    int mm_col_start = num_loc(n_start, nb, mypcol, array_descA.icsrc(), npcols);
    int i_loc,j_loc;
    int owner_row;

    T *d_temp_U;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_U, sizeof(T)*nb_real*panel, stream));

    DEVICE_CHECK(deviceStreamSynchronize(stream));

    double time_for_pgetf2 = 0.0;
    double time_for_other = 0.0;

    double start_time;

    int i_s = array_descA.indx_g2l_r(n_start);
    int j_s = array_descA.indx_g2l_c(n_start);
    

    for(int n_s=n_start;n_s<n_start+nb_real;n_s+=panel){
        panel_real = std::min(panel, nb_real+n_start-n_s);

        i_loc = array_descA.indx_g2l_r(n_s);
        j_loc = array_descA.indx_g2l_c(n_s);

        owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        // start pgetf2
        start_time = MPI_Wtime();
        pgetf2(
            m, panel_real,
            d_A, n_s, array_descA,
            ipiv, info
        );
        // finish pgetf2
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_pgetf2 += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=panel; // update row start   
        mm_col_start+=panel;

        // broadcast block column
        if(mm_col_start<j_s + nb_real && j_loc>=0){
            if(i_loc>=0){
                BLAS_CHECK(deblasTrsm(
                    blasH, DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER, DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    panel_real, j_s + nb_real - mm_col_start, 1.0,
                    d_A + i_loc + j_loc * lld, lld,
                    d_A + i_loc + mm_col_start * lld, lld)
                );
                // printf("before d_temp_U:%d, j_loc:%d, nb_real:%d, mm_col_start:%d\n", ddla_handle->myid, j_loc, nb_real, mm_col_start);
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_U, panel_real * sizeof(T),
                    d_A + i_loc + mm_col_start * lld, lld * sizeof(T),
                    panel_real * sizeof(T), j_s + nb_real - mm_col_start,
                    deviceMemcpyDeviceToDevice, stream
                ));
            } 
            CCL_CHECK(cclBcast(d_temp_U,panel_real * (j_s + nb_real - mm_col_start),owner_row,col_nccl_comm,stream));  
        }
        // printf("myid:%d, n_s:%d, update trailing matrix mm_row_start:%d, mm_col_start:%d\n",mpi_comm_global_h.myid,n_s,mm_row_start,mm_col_start);
        if(mm_row_start<m_loc && mm_col_start<j_s + nb_real && j_loc>=0){
            BLAS_CHECK(deblasGemm(
                blasH, DEBLAS_OP_N, DEBLAS_OP_N,
                m_loc - mm_row_start, j_s + nb_real - mm_col_start, panel_real,
                -1.0,
                d_A + mm_row_start + j_loc * lld, lld,
                d_temp_U, panel_real,
                1.0,
                d_A + mm_row_start + mm_col_start * lld, lld
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_other += MPI_Wtime() - start_time;
    }
    info = 0;
    DEVICE_CHECK(deviceFreeAsync(d_temp_U, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    // printf("myid:%d, pzgetrf time_for_pgetf2:%lf, time_for_other:%lf\n",ddla_handle->myid,time_for_pgetf2,time_for_other);

}

template void pgetf2_panel<float>(
    const int& m, const int& nb_real,
    float* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2_panel<double>(
    const int& m, const int& nb_real,
    double* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2_panel<std::complex<float>>(
    const int& m, const int& nb_real,
    std::complex<float>* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2_panel<std::complex<double>>(
    const int& m, const int& nb_real,
    std::complex<double>* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

}