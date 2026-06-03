#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/trsm.h>
#include <ddla/ddla_comm.h>
#ifdef DDLA_USE_GPU_CPU_TUNNEL
#include <vector>
#endif

namespace ddla{

template <typename T>
void transport_block(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_block_A
)
{
    if(m==0 || n==0)
        return;
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    assert(sData == 'C' || sData == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    #ifdef DDLA_USE_CCL
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif

    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(array_descA.nb() * (std::max(array_descA.m_loc(), array_descA.n_loc())));
    #endif

    int i_loc = num_loc(ia, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
    int j_loc = num_loc(ja, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());

    int m_loc = num_loc(ia + m, array_descA.nb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
    int n_loc = num_loc(ja + n, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());

    int owner_row = indxg2p(ia, array_descA.nb(), array_descA.irsrc(), array_descA.nprows());
    int owner_col = indxg2p(ja, array_descA.nb(), array_descA.icsrc(), array_descA.npcols());

    if(trans == 'N'){
        if(sData == 'R' && n_loc > j_loc){
            if(array_descA.myprow() == owner_row){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_block_A, m * sizeof(T),
                    d_A + i_loc + j_loc * array_descA.lld(), array_descA.lld() * sizeof(T),
                    m * sizeof(T), n_loc - j_loc,
                    deviceMemcpyDeviceToDevice, ddla_handle->stream
                ));
            }
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_A, m * (n_loc - j_loc), owner_row, ddla_handle->col_comm, ddla_handle->stream));
            #else   
            CCL_CHECK(cclBcast(d_block_A, m * (n_loc - j_loc), owner_row, col_nccl_comm, ddla_handle->stream));
            #endif
        }else if(sData == 'C' && m_loc > i_loc){
            if(array_descA.mypcol() == owner_col){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_block_A, (m_loc - i_loc) * sizeof(T),
                    d_A + i_loc + j_loc * array_descA.lld(), array_descA.lld() * sizeof(T),
                    (m_loc - i_loc) * sizeof(T), n,
                    deviceMemcpyDeviceToDevice, ddla_handle->stream
                ));
            }
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_A, (m_loc - i_loc) * n, owner_col, ddla_handle->row_comm, ddla_handle->stream));
            #else
            CCL_CHECK(cclBcast(d_block_A, (m_loc - i_loc) * n, owner_col, row_nccl_comm, ddla_handle->stream));
            #endif
        }
    }else{
        int trans_j_loc = num_loc(ja, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
        int trans_n_loc = num_loc(ja + n, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
        
        int trans_i_loc = num_loc(ia ,array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());
        int trans_m_loc = num_loc(ia + m, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());
        // printf("myid:%d, owner_row:%d, trans_n_loc:%d, trans_j_loc:%d, n_loc:%d, j_loc:%d\n", ddla_handle->myid, owner_row, trans_n_loc, trans_j_loc, n_loc, j_loc);
        if(sData == 'R'){
            if(n_loc > j_loc){
                if(array_descA.myprow() == owner_row){
                    DEVICE_CHECK(deviceMemcpy2DAsync(
                        d_block_A, m * sizeof(T),
                        d_A + i_loc + j_loc * array_descA.lld(), array_descA.lld() * sizeof(T),
                        m * sizeof(T), n_loc - j_loc,
                        deviceMemcpyDeviceToDevice, ddla_handle->stream
                    ));
                    if(array_descA.myprow() != array_descA.mypcol()){
                        #ifdef DDLA_USE_GPU_CPU_TUNNEL
                        MPI_CHECK(cclSend(h_temp.data(), d_block_A, m * (n_loc - j_loc), array_descA.mypcol(), ddla_handle->col_comm, ddla_handle->stream));
                        #else
                        CCL_CHECK(cclSend(d_block_A, m * (n_loc - j_loc), array_descA.mypcol(), col_nccl_comm, ddla_handle->stream));
                        #endif
                    }
                }else{
                    if(array_descA.myprow() == array_descA.mypcol()){
                        #ifdef DDLA_USE_GPU_CPU_TUNNEL
                        MPI_CHECK(cclRecv(h_temp.data(), d_block_A, m * (n_loc - j_loc), owner_row, ddla_handle->col_comm, ddla_handle->stream));
                        #else
                        CCL_CHECK(cclRecv(d_block_A, m * (n_loc - j_loc), owner_row, col_nccl_comm, ddla_handle->stream));
                        #endif
                    }
                }
            }
            if(trans_n_loc > trans_j_loc){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_temp.data(), d_block_A, (trans_n_loc - trans_j_loc) * m, array_descA.myprow(), ddla_handle->row_comm, ddla_handle->stream));
                #else
                CCL_CHECK(cclBcast(d_block_A, (trans_n_loc - trans_j_loc) * m, array_descA.myprow(), row_nccl_comm, ddla_handle->stream));
                #endif
            }
        }else if(sData == 'C'){
            if(m_loc > i_loc){
                if(array_descA.mypcol() == owner_col){
                    DEVICE_CHECK(deviceMemcpy2DAsync(
                        d_block_A, (m_loc - i_loc) * sizeof(T),
                        d_A + i_loc + j_loc * array_descA.lld(), array_descA.lld() * sizeof(T),
                        (m_loc - i_loc) * sizeof(T), n,
                        deviceMemcpyDeviceToDevice, ddla_handle->stream
                    ));
                    if(array_descA.myprow() != array_descA.mypcol()){
                        #ifdef DDLA_USE_GPU_CPU_TUNNEL
                        MPI_CHECK(cclSend(h_temp.data(), d_block_A, (m_loc - i_loc) * n, array_descA.myprow(), ddla_handle->row_comm, ddla_handle->stream));
                        #else
                        CCL_CHECK(cclSend(d_block_A, (m_loc - i_loc) * n, array_descA.myprow(), row_nccl_comm, ddla_handle->stream));
                        #endif
                    }
                }else{
                    if(array_descA.myprow() == array_descA.mypcol()){
                        #ifdef DDLA_USE_GPU_CPU_TUNNEL
                        MPI_CHECK(cclRecv(h_temp.data(), d_block_A, (m_loc - i_loc) * n, owner_col, ddla_handle->row_comm, ddla_handle->stream));
                        #else
                        CCL_CHECK(cclRecv(d_block_A, (m_loc - i_loc) * n, owner_col, row_nccl_comm, ddla_handle->stream));
                        #endif
                    }
                }
            }
            if(trans_m_loc > trans_i_loc){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_temp.data(), d_block_A, (trans_m_loc - trans_i_loc) * n, array_descA.mypcol(), ddla_handle->col_comm, ddla_handle->stream));
                #else   
                CCL_CHECK(cclBcast(d_block_A, (trans_m_loc - trans_i_loc) * n, array_descA.mypcol(), col_nccl_comm, ddla_handle->stream));
                #endif
            }
        }
    }
    return;
}

template void transport_block<float>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const float* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    float* d_block_A
);

template void transport_block<double>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const double* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    double* d_block_A
);

template void transport_block<std::complex<float>>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const std::complex<float>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<float>* d_block_A
);

template void transport_block<std::complex<double>>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const std::complex<double>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<double>* d_block_A
);


} // namespace DDLA