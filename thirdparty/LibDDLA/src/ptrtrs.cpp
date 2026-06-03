#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/trsm.h>
#include <ddla/transport_block.h>
#include <ddla/ddla_comm.h>
#include <ddla/gemm.h>
#ifdef DDLA_USE_GPU_CPU_TUNNEL
#include <vector>
#endif
namespace ddla{


template<typename T>
void ptrtrs(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    assert(array_descA.m() == array_descA.n());
    assert(array_descA.mb()==array_descA.nb());
    assert(array_descA.mb()==array_descB.mb());
    assert(array_descA.m() == array_descB.m());
    assert(uplo=='L'||uplo=='U');
    assert(diag=='U'||diag=='N');
    assert(side=='L');
    assert(trans=='N'||trans=='T'||trans=='C');
    int nb = array_descA.mb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    // printf("nprows:%d, npcols:%d\n",nprows,npcols);

    // 初始化 NCCL  
    #ifdef DDLA_USE_CCL
    ncclComm_t row_comm=ddla_handle->nccl_row_comm;
    ncclComm_t col_comm=ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_comm=ddla_handle->row_comm;
    MPI_Comm col_comm=ddla_handle->col_comm;
    #endif
    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    deblasFillMode_t uplo_device = (uplo == 'U') ? DEBLAS_FILL_MODE_UPPER : DEBLAS_FILL_MODE_LOWER;
    deblasDiagType_t diag_device = (diag == 'U') ? DEBLAS_DIAG_UNIT : DEBLAS_DIAG_NON_UNIT;
    deblasOperation_t trans_device;
    if(trans == 'N'){
        trans_device = DEBLAS_OP_N;
    }else if(trans == 'T'){
        trans_device = DEBLAS_OP_T;
    }else{
        trans_device = DEBLAS_OP_C;
    }
    deblasSideMode_t side_device = (side == 'L') ? DEBLAS_SIDE_LEFT : DEBLAS_SIDE_RIGHT;
    
    // double start_time = MPI_Wtime();
    T* d_block_diag,*d_block_A,*d_block_B;
    DEVICE_CHECK(deviceMallocAsync(&d_block_diag, nb * nb * sizeof(T), stream));
    DEVICE_CHECK(deviceMallocAsync(&d_block_B, nb * array_descB.n_loc() * sizeof(T), stream));
    DEVICE_CHECK(deviceMallocAsync(&d_block_A, std::max(array_descA.m_loc(), array_descA.n_loc()) * nb * sizeof(T), stream));

    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(nb * std::max(array_descB.n_loc(), array_descA.m_loc()));
    #endif

    int mm_row_start, mm_col_start, mm_row_step, mm_col_step;
    int n_s_start,n_s_end,n_s_step;
    if((uplo == 'U' && trans == 'N') || (uplo == 'L' && trans != 'N')){
        int m_loc = array_descA.m_loc();
        n_s_start = m % nb == 0 ? m - nb : m - m % nb;
        n_s_end = -nb;
        n_s_step = -nb;
    }else{
        n_s_start = 0;
        n_s_end = m % nb == 0 ? m : m - m % nb + nb;
        n_s_step = nb;
    }
    int owner_row, owner_col;
    
    int64_t A_offset, B_offset;
    for(int n_s = n_s_start; n_s != n_s_end; n_s += n_s_step){
        int nb_real = std::min(nb, m - n_s);
        // printf("n_s=%d, nb_real=%d\n",n_s, nb_real);

        mm_row_start = num_loc(n_s, nb, array_descA.myprow(), array_descA.irsrc(), nprows);
        mm_col_start = num_loc(n_s, nb, array_descA.mypcol(), array_descA.icsrc(), npcols);

        owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

        if(array_descA.myprow() == owner_row)
            mm_row_step = nb_real;
        else
            mm_row_step = 0;
        if(array_descA.mypcol() == owner_col)
            mm_col_step = nb_real;
        else 
            mm_col_step = 0;
        // printf("owner_row:%d,owner_col:%d\n",owner_row,owner_col);

        if(array_descA.myprow() == owner_row && array_descA.mypcol() == owner_col){
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_block_diag, nb_real * sizeof(T),
                d_A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                nb_real * sizeof(T), nb_real,
                deviceMemcpyDeviceToDevice, stream
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        // 广播当前块行
        if(array_descA.myprow() == owner_row){
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_diag, nb_real * nb_real, owner_col, ddla_handle->row_comm, stream));
            #else
            CCL_CHECK(cclBcast(d_block_diag, nb_real * nb_real, owner_col, row_comm, stream));
            #endif
            BLAS_CHECK(deblasTrsm(
                blasH, side_device, uplo_device, trans_device, diag_device,
                nb_real, array_descB.n_loc(), 1.0,
                d_block_diag, nb_real,
                d_B + mm_row_start, lldB
            ));
            // DEVICE_CHECK(deviceMemcpy2DAsync(
            //     d_block_B, nb_real * sizeof(T),
            //     d_B + mm_row_start, lldB * sizeof(T),
            //     nb_real * sizeof(T), array_descB.n_loc(),
            //     deviceMemcpyDeviceToDevice, stream
            // ));
        }
        // #ifdef DDLA_USE_GPU_CPU_TUNNEL
        // MPI_CHECK(cclBcast(h_temp.data(), d_block_B, nb_real * array_descB.n_loc(), owner_row, ddla_handle->col_comm, stream));
        // #else
        // CCL_CHECK(cclBcast(d_block_B, nb_real * array_descB.n_loc(), owner_row, col_comm, stream));
        // #endif
        transport_block(
            'R', 'N', 
            nb_real, array_descB.n(),
            d_B, n_s, 0, array_descB,
            d_block_B
        );
        int length_block_A;
        int source_col;
        int g_m, g_n;
        int g_ia, g_ja;
        if(trans != 'N'){
            g_m = nb_real;
            g_ia = n_s;
            if(uplo == 'L'){
                A_offset = mm_row_start;
                length_block_A = mm_col_start;
                g_n = n_s;
                g_ja = 0;
            }else{
                A_offset = mm_row_start + (mm_col_start + mm_col_step) * array_descA.lld();
                length_block_A = array_descA.n_loc() - mm_col_start - mm_col_step;
                g_n = array_descA.n() - n_s - nb_real;
                g_ja = n_s + nb_real;
            }
            // if(length_block_A > 0){
            //     if(array_descA.myprow() == owner_row){
            //         DEVICE_CHECK(deviceMemcpy2DAsync(
            //             d_block_A, nb_real * sizeof(T),
            //             d_A + A_offset, array_descA.lld() * sizeof(T),
            //             nb_real * sizeof(T), length_block_A,
            //             deviceMemcpyDeviceToDevice, ddla_handle->stream
            //         ));
            //         if(array_descA.myprow() != array_descA.mypcol())
            //             CCL_CHECK(cclSend(d_block_A, nb_real * length_block_A, array_descA.mypcol(), col_comm, stream));
            //     }else if(array_descA.myprow() == array_descA.mypcol()){
            //         CCL_CHECK(cclRecv(d_block_A, nb_real * length_block_A, owner_row, col_comm, stream));
            //     }
            // }
            // source_col = array_descA.myprow();
        }else{
            g_ja = n_s;
            g_n = nb_real;
            if(uplo == 'L'){
                length_block_A = array_descA.m_loc() - mm_row_start - mm_row_step;
                A_offset = mm_row_start + mm_row_step + mm_col_start * array_descA.lld();
                g_m = array_descA.m() - n_s - nb_real;
                g_ia = n_s + nb_real;
            }else{
                length_block_A = mm_row_start;
                A_offset = mm_col_start * array_descA.lld();
                g_m = n_s;
                g_ia = 0;
            }
            // if(array_descA.mypcol() == owner_col && length_block_A > 0){
            //     DEVICE_CHECK(deviceMemcpy2DAsync(
            //         d_block_A, length_block_A * sizeof(T),
            //         d_A + A_offset, lldA * sizeof(T),
            //         length_block_A * sizeof(T), nb_real,
            //         deviceMemcpyDeviceToDevice, stream
            //     ));
            // }
            // source_col = owner_col;
        }
        transport_block(
            trans == 'N' ? 'C' : 'R', trans,
            g_m, g_n,
            d_A, g_ia, g_ja, array_descA,
            d_block_A
        );
        if((uplo == 'U' && trans == 'N') || (uplo == 'L' && trans != 'N')){
            length_block_A = mm_row_start;
            B_offset = 0;
        }else{
            length_block_A = array_descA.m_loc() - mm_row_start - mm_row_step;
            B_offset = mm_row_start + mm_row_step;
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        if(length_block_A > 0){
            // #ifdef DDLA_USE_GPU_CPU_TUNNEL
            // MPI_CHECK(cclBcast(h_temp.data(), d_block_A, length_block_A * nb_real, source_col, ddla_handle->row_comm, stream));
            // #else
            // CCL_CHECK(cclBcast(d_block_A, length_block_A * nb_real, source_col, row_comm, stream));
            // #endif
            BLAS_CHECK(deblasGemm(
                blasH, trans_device, DEBLAS_OP_N, 
                length_block_A, array_descB.n_loc(), nb_real,
                (T)-1.0,
                d_block_A, trans == 'N' ? length_block_A : nb_real,
                d_block_B, nb_real,
                (T)1.0,
                d_B + B_offset, lldB
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
    }
    DEVICE_CHECK(deviceFreeAsync(d_block_A,stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_B,stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_diag,stream));
}


template void ptrtrs<float>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);

template void ptrtrs<double>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);

template void ptrtrs<std::complex<float>>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);

template void ptrtrs<std::complex<double>>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);


}



