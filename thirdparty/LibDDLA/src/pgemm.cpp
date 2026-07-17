#include <ddla/ddla.h>
#include <algorithm>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <vector>
#include <ddla/transport_block.h>
#include <ddla/ddla_comm.h>
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
    assert(transa == 'N' || transa == 'T' || transa == 'C');
    assert(transb == 'N' || transb == 'T' || transb == 'C');
    assert(m > 0 && n > 0 && k > 0);

    const int opA_m = (transa == 'N') ? array_descA.m() : array_descA.n();
    const int opA_n = (transa == 'N') ? array_descA.n() : array_descA.m();
    const int opB_m = (transb == 'N') ? array_descB.m() : array_descB.n();
    const int opB_n = (transb == 'N') ? array_descB.n() : array_descB.m();
    assert(m <= opA_m);
    assert(k <= opA_n);
    assert(k <= opB_m);
    assert(n <= opB_n);
    assert(m <= array_descC.m() && n <= array_descC.n());
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

    const int m_loc_A = num_loc((transa == 'N') ? m : k, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
    const int n_loc_A = num_loc((transa == 'N') ? k : m, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());
    const int m_loc_B = num_loc((transb == 'N') ? k : n, array_descB.mb(), array_descB.myprow(), array_descB.irsrc(), array_descB.nprows());
    const int n_loc_B = num_loc((transb == 'N') ? n : k, array_descB.nb(), array_descB.mypcol(), array_descB.icsrc(), array_descB.npcols());
    const int m_loc_C = num_loc(m, array_descC.mb(), array_descC.myprow(), array_descC.irsrc(), array_descC.nprows());
    const int n_loc_C = num_loc(n, array_descC.nb(), array_descC.mypcol(), array_descC.icsrc(), array_descC.npcols());

    int nb;
    if(transa=='N')
        nb = array_descA.nb();
    else
        nb = array_descA.mb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();
    int lldC = array_descC.lld();

    deviceStream_t stream = ddla_handle->stream;
    deviceStream_t stream_data = ddla_handle->stream_data;
    deblasHandle_t blasH = ddla_handle->blasH;

    deblasOperation_t opA = (transa == 'N') ? DEBLAS_OP_N :
                            (transa == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;

    const int buffer_max = 2;
    T *d_A_temp[buffer_max],*d_B_temp[buffer_max];
    int count_a = ((transa == 'N') ? std::max(m_loc_A, m_loc_C) : std::max(n_loc_A, m_loc_C)) * nb;
    int count_b = nb * ((transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C));
    count_a = std::max(1, count_a);
    count_b = std::max(1, count_b);
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
    };
    get_data(k_s);
    bool first_gemm = true;
    for(k_s=0; k_s<k; k_s+=nb){
        DEVICE_CHECK(deviceStreamSynchronize(stream_data));
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        if(m_loc_C > 0 && n_loc_C > 0){
            const T gemm_beta = first_gemm ? beta : (T)1.0;
            BLAS_CHECK(deblasGemm(
                blasH, opA, DEBLAS_OP_N,
                m_loc_C, n_loc_C, kb,
                alpha,
                d_A_temp[temp_buffer], transa=='N'?m_loc_C:kb,
                d_B_temp[temp_buffer], kb,
                gemm_beta,
                d_C, lldC
            ));
            first_gemm = false;
        }
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
