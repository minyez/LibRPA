#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <vector>
#include <ddla/transport_block.h>
#include <ddla/ddla_comm.h>
#include <ddla/scal.h>
#include <ddla/gemm.h>

namespace ddla{

template <typename T>
void pgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T* d_B, const DdlaDesc& array_descB,
    const T& beta,
    T* d_C, const DdlaDesc& array_descC
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    if(transa != 'N' || transb != 'N')
    {
        if(array_descA.nprows() != array_descA.npcols()){
            throw std::runtime_error("the trans multiplication is now implemented for mxn(m!=n) mpi grid");
        }
    }
    {
        int mbA,kbA,kbB,nbB,mbC,nbC;
        mbC = array_descC.mb();
        nbC = array_descC.nb();
        if(transa == 'N'){
            mbA = array_descA.mb();
            kbA = array_descA.nb();
        }else{
            mbA = array_descA.nb();
            kbA = array_descA.mb();
        }

        if(transb == 'N'){
            kbB = array_descB.mb();
            nbB = array_descB.nb();
        }else{
            kbB = array_descB.nb();
            nbB = array_descB.mb();
        }
        assert(mbA == mbC);
        assert(kbA == kbB);
        assert(nbB == nbC);
    }

    #ifdef DDLA_USE_CCL
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif
    int nprows = array_descC.nprows();
    int npcols = array_descC.npcols();
    int myprow = array_descC.myprow();
    int mypcol = array_descC.mypcol();

    const int m_loc_A = array_descA.m_loc();
    const int n_loc_A = array_descA.n_loc();
    const int m_loc_B = array_descB.m_loc();
    const int n_loc_B = array_descB.n_loc();
    const int m_loc_C = array_descC.m_loc();
    const int n_loc_C = array_descC.n_loc();

    int nb;
    if(transa=='N')
        nb = array_descA.nb();
    else
        nb = array_descA.mb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();

    deviceStream_t stream = ddla_handle->stream;
    deviceStream_t stream_data = ddla_handle->stream_data;
    deblasHandle_t blasH = ddla_handle->blasH;

    deblasOperation_t opA = (transa == 'N') ? DEBLAS_OP_N :
                            (transa == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;
    deblasOperation_t opB = (transb == 'N') ? DEBLAS_OP_N :
                            (transb == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;

    BLAS_CHECK(deblasScal(blasH, m_loc_C*n_loc_C, beta, d_C, 1));

    const int buffer_max = 2;
    T *d_A_temp[buffer_max],*d_B_temp[buffer_max];
    int count_a = (transa=='N'?m_loc_C:(std::max(n_loc_A, m_loc_C))) * nb;
    int count_b = nb * (transb=='N'?n_loc_C:(std::max(m_loc_B, n_loc_C)));
    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(std::max(count_a, count_b));
    #endif
    for(int i=0;i<buffer_max;i++){
        DEVICE_CHECK(deviceMallocAsync(&d_A_temp[i], sizeof(T) * count_a, stream_data));
        DEVICE_CHECK(deviceMallocAsync(&d_B_temp[i], sizeof(T) * count_b, stream_data));
    }

    int temp_buffer = 0;
    int k_s = 0 , kb;
    auto get_data = [&](int k_s) 
    {
        kb = std::min(nb, k - k_s);
        if(kb<=0) return;
        // int src_A;
        
        // if(transa != 'N'){
        //     int owner_row_A = indxg2p(k_s, nb, array_descA.irsrc(), array_descA.nprows());
        //     if(myprow == owner_row_A){
        //         DEVICE_CHECK(deviceMemcpy2DAsync(
        //             d_A_temp[temp_buffer], kb * sizeof(T),
        //             d_A + array_descA.indx_g2l_r(k_s), lldA * sizeof(T),
        //             kb * sizeof(T), n_loc_A,
        //             deviceMemcpyDeviceToDevice, stream_data
        //         ));
        //         if(myprow != mypcol){
        //             #ifdef DDLA_USE_GPU_CPU_TUNNEL
        //             MPI_CHECK(cclSend(h_temp.data(), d_A_temp[temp_buffer], kb * n_loc_A, mypcol, ddla_handle->col_comm, stream_data));
        //             #else
        //             CCL_CHECK(cclSend(d_A_temp[temp_buffer], kb * n_loc_A, mypcol, col_nccl_comm, stream_data));
        //             #endif
        //         }
        //     }else{
        //         if(myprow == mypcol){
        //             #ifdef DDLA_USE_GPU_CPU_TUNNEL
        //             MPI_CHECK(cclRecv(h_temp.data(), d_A_temp[temp_buffer], kb * n_loc_A, owner_row_A, ddla_handle->col_comm, stream_data));
        //             #else
        //             CCL_CHECK(cclRecv(d_A_temp[temp_buffer], kb * n_loc_A, owner_row_A, col_nccl_comm, stream_data));
        //             #endif
        //         }
        //     }
        //     src_A = myprow;
        // }else{
        //     src_A = indxg2p(k_s, nb, array_descA.icsrc(), array_descA.npcols());
        //     if(mypcol == src_A){
        //         DEVICE_CHECK(deviceMemcpy2DAsync(
        //             d_A_temp[temp_buffer], m_loc_A * sizeof(T),
        //             d_A + array_descA.indx_g2l_c(k_s) * lldA, lldA * sizeof(T),
        //             m_loc_A * sizeof(T), kb,
        //             deviceMemcpyDeviceToDevice, stream_data
        //         ));
        //     }
        // }
        char sData_a;
        int m_a, n_a, g_ia, g_ja;
        if(transa != 'N'){
            sData_a = 'R';
            m_a = kb;
            n_a = m;
            g_ia = k_s;
            g_ja = 0;
        }else{
            sData_a = 'C';
            m_a = m;
            n_a = kb;
            g_ia = 0;
            g_ja = k_s;
        }
        transport_block(
            sData_a, transa,
            m_a, n_a,
            d_A, g_ia, g_ja, array_descA,
            d_A_temp[temp_buffer] 
        );
        
        // broadcast A block
        // #ifdef DDLA_USE_GPU_CPU_TUNNEL
        // MPI_CHECK(cclBcast(h_temp.data(), d_A_temp[temp_buffer], m_loc_C * kb, src_A, ddla_handle->row_comm, stream_data));
        // #else        
        // CCL_CHECK(cclBcast(d_A_temp[temp_buffer], m_loc_C * kb, src_A, row_nccl_comm, stream_data));
        // #endif
        // // end communicate A

        // int src_B;
        // start communicate B
        char sData_b;
        int m_b, n_b, g_ib, g_jb;
        if(transb != 'N'){
            sData_b = 'C';
            m_b = n;
            n_b = kb;
            g_ib = 0;
            g_jb = k_s;
        }else{
            sData_b = 'R';
            m_b = kb;
            n_b = n;
            g_ib = k_s;
            g_jb = 0;
        }
        transport_block(
            sData_b, transb,
            m_b, n_b,
            d_B, g_ib, g_jb, array_descB,
            d_B_temp[temp_buffer]
        );
        // if(transb != 'N'){
        //     int owner_col_B = indxg2p(k_s, nb, array_descB.icsrc(), array_descB.npcols());
        //     if(mypcol == owner_col_B){
        //         DEVICE_CHECK(deviceMemcpy2DAsync(
        //             d_B_temp[temp_buffer], m_loc_B * sizeof(T),
        //             d_B + array_descB.indx_g2l_c(k_s) * lldB, lldB * sizeof(T),
        //             m_loc_B * sizeof(T), kb,
        //             deviceMemcpyDeviceToDevice, stream_data
        //         ));
        //         if(myprow != mypcol){
        //             #ifdef DDLA_USE_GPU_CPU_TUNNEL
        //             MPI_CHECK(cclSend(h_temp.data(), d_B_temp[temp_buffer], kb * m_loc_B, myprow, ddla_handle->row_comm, stream_data));
        //             #else
        //             CCL_CHECK(cclSend(d_B_temp[temp_buffer], kb * m_loc_B, myprow, row_nccl_comm, stream_data));
        //             #endif
        //         }
        //     }else{
        //         if(myprow == mypcol){
        //             #ifdef DDLA_USE_GPU_CPU_TUNNEL
        //             MPI_CHECK(cclRecv(h_temp.data(), d_B_temp[temp_buffer], kb * m_loc_B, owner_col_B, ddla_handle->row_comm, stream_data));
        //             #else
        //             CCL_CHECK(cclRecv(d_B_temp[temp_buffer], kb * m_loc_B, owner_col_B, row_nccl_comm, stream_data));
        //             #endif
        //         }
        //     }
        //     src_B = mypcol;
        // }
        // else{
        //     src_B = indxg2p(k_s, nb, array_descB.irsrc(), array_descB.nprows());
        //     if(myprow == src_B){
        //         DEVICE_CHECK(deviceMemcpy2DAsync(
        //             d_B_temp[temp_buffer], kb * sizeof(T),
        //             d_B + array_descB.indx_g2l_r(k_s), lldB * sizeof(T),
        //             kb * sizeof(T), n_loc_C,
        //             deviceMemcpyDeviceToDevice, stream_data
        //         ));
        //     }
        // }
        // // broadcast B block
        // #ifdef DDLA_USE_GPU_CPU_TUNNEL
        // MPI_CHECK(cclBcast(h_temp.data(), d_B_temp[temp_buffer], kb * n_loc_C, src_B, ddla_handle->col_comm, stream_data));
        // #else
        // CCL_CHECK(cclBcast(d_B_temp[temp_buffer], kb * n_loc_C, src_B, col_nccl_comm, stream_data));
        // #endif
    };
    get_data(k_s);
    for(k_s=0; k_s<k; k_s+=nb){
        DEVICE_CHECK(deviceStreamSynchronize(stream_data));
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        BLAS_CHECK(deblasGemm(
            blasH, opA, opB,
            m_loc_C, n_loc_C, kb,
            alpha,
            d_A_temp[temp_buffer], transa=='N'?m_loc_C:kb,
            d_B_temp[temp_buffer], transb=='N'?kb:n_loc_C,
            1.0,
            d_C, m_loc_C
        ));
        temp_buffer = (temp_buffer + 1) % buffer_max;
        get_data(k_s + nb);
    }
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream_data));
    for(int i=0;i<buffer_max;i++){
        DEVICE_CHECK(deviceFreeAsync(d_A_temp[i], stream_data));
        DEVICE_CHECK(deviceFreeAsync(d_B_temp[i], stream_data));
    }
    return;
}

template void pgemm<float>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const float& alpha,
    const float* d_A, const DdlaDesc& array_descA,
    const float* d_B, const DdlaDesc& array_descB,
    const float& beta,
    float* d_C, const DdlaDesc& array_descC
);

template void pgemm<double>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const double& alpha,
    const double* d_A, const DdlaDesc& array_descA,
    const double* d_B, const DdlaDesc& array_descB,
    const double& beta,
    double* d_C, const DdlaDesc& array_descC
);

template void pgemm<std::complex<float>>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const std::complex<float>& alpha,
    const std::complex<float>* d_A, const DdlaDesc& array_descA,
    const std::complex<float>* d_B, const DdlaDesc& array_descB,
    const std::complex<float>& beta,
    std::complex<float>* d_C, const DdlaDesc& array_descC
);

template void pgemm<std::complex<double>>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const std::complex<double>& alpha,
    const std::complex<double>* d_A, const DdlaDesc& array_descA,
    const std::complex<double>* d_B, const DdlaDesc& array_descB,
    const std::complex<double>& beta,
    std::complex<double>* d_C, const DdlaDesc& array_descC
);


}