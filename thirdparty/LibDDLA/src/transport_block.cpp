#include <ddla/ddla.h>
#include <algorithm>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/ddla_comm.h>
#include <ddla/geam.h>
#include <vector>

namespace ddla{

template <typename T>
void transport_block(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja,
    const DdlaDesc& array_descA, T* d_block_A
)
{
    if(m==0 || n==0)
        return;
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    assert(sData == 'C' || sData == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    #ifdef DDLA_USE_CCL
    ncclComm_t grid_nccl_comm = ddla_handle->nccl_comm;
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm grid_nccl_comm = ddla_handle->comm;
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif

    int i_loc = num_loc(ia, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
    int j_loc = num_loc(ja, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());

    int m_loc = num_loc(ia + m, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
    int n_loc = num_loc(ja + n, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());

    int owner_row = indxg2p(ia, array_descA.mb(), array_descA.irsrc(), array_descA.nprows());
    int owner_col = indxg2p(ja, array_descA.nb(), array_descA.icsrc(), array_descA.npcols());

    if(trans == 'N'){
        #ifdef DDLA_USE_GPU_CPU_TUNNEL
        const size_t host_count = sData == 'R'
                                ? static_cast<size_t>(m) * std::max(0, n_loc - j_loc)
                                : static_cast<size_t>(std::max(0, m_loc - i_loc)) * n;
        std::vector<T> h_temp(std::max<size_t>(1, host_count));
        #endif
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
    }else if(array_descA.nprows() == array_descA.npcols()){
        int trans_j_loc = num_loc(ja, array_descA.nb(), array_descA.myprow(), array_descA.icsrc(), array_descA.nprows());
        int trans_n_loc = num_loc(ja + n, array_descA.nb(), array_descA.myprow(), array_descA.icsrc(), array_descA.nprows());

        int trans_i_loc = num_loc(ia ,array_descA.mb(), array_descA.mypcol(), array_descA.irsrc(), array_descA.npcols());
        int trans_m_loc = num_loc(ia + m, array_descA.mb(), array_descA.mypcol(), array_descA.irsrc(), array_descA.npcols());
        #ifdef DDLA_USE_GPU_CPU_TUNNEL
        const size_t source_count = sData == 'R'
                                  ? static_cast<size_t>(m) * std::max(0, n_loc - j_loc)
                                  : static_cast<size_t>(std::max(0, m_loc - i_loc)) * n;
        const size_t target_count = sData == 'R'
                                  ? static_cast<size_t>(m) * std::max(0, trans_n_loc - trans_j_loc)
                                  : static_cast<size_t>(std::max(0, trans_m_loc - trans_i_loc)) * n;
        std::vector<T> h_temp(std::max<size_t>({1, source_count, target_count}));
        #endif
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
                    const deblasOperation_t pack_op = trans == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
                    const T one = (T)1.0;
                    const T zero = (T)0.0;
                    BLAS_CHECK(deblasGeam(
                        ddla_handle->blasH, pack_op, pack_op,
                        n, m_loc - i_loc,
                        one,
                        d_A + i_loc + j_loc * array_descA.lld(), array_descA.lld(),
                        zero,
                        d_A + i_loc + j_loc * array_descA.lld(), array_descA.lld(),
                        d_block_A, n
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
    }else{
        struct RectBlock{
            int src_rank;
            int dst_rank;
            int count;
            int send_offset;
            int dst_offset;
            int src_loc;
            int len;
        };

        const int myrank = ddla_handle->myid;
        auto block_end = [](const int g, const int block_size, const int end){
            return std::min(end, (g / block_size + 1) * block_size);
        };

        if(sData == 'R'){
            const int target_j_loc = num_loc(ja, array_descA.nb(), array_descA.myprow(), array_descA.icsrc(), array_descA.nprows());
            const int target_n_loc = num_loc(ja + n, array_descA.nb(), array_descA.myprow(), array_descA.icsrc(), array_descA.nprows());
            const int target_cols = target_n_loc - target_j_loc;
            std::vector<RectBlock> blocks;
            std::vector<RectBlock> send_blocks;
            std::vector<RectBlock> recv_blocks;
            int send_total = 0;

            for(int g = ja; g < ja + n; ){
                const int len = block_end(g, array_descA.nb(), ja + n) - g;
                const int src_col = indxg2p(g, array_descA.nb(), array_descA.icsrc(), array_descA.npcols());
                const int dst_row = indxg2p(g, array_descA.nb(), array_descA.icsrc(), array_descA.nprows());
                const int src_rank = ddla_handle->rc_to_rank(owner_row, src_col);
                const int dst_rank = ddla_handle->rc_to_rank(dst_row, 0);
                const int dst_col_loc = num_loc(g, array_descA.nb(), dst_row, array_descA.icsrc(), array_descA.nprows())
                                      - num_loc(ja, array_descA.nb(), dst_row, array_descA.icsrc(), array_descA.nprows());
                RectBlock block{src_rank, dst_rank, m * len, -1, dst_col_loc * m,
                                indxg2l(g, array_descA.nb(), array_descA.npcols()), len};

                if(src_rank == dst_rank){
                    if(myrank == src_rank){
                        DEVICE_CHECK(deviceMemcpy2DAsync(
                            d_block_A + block.dst_offset, m * sizeof(T),
                            d_A + i_loc + block.src_loc * array_descA.lld(), array_descA.lld() * sizeof(T),
                            m * sizeof(T), block.len,
                            deviceMemcpyDeviceToDevice, ddla_handle->stream
                        ));
                    }
                }else{
                    blocks.push_back(block);
                    if(myrank == src_rank){
                        block.send_offset = send_total;
                        send_total += block.count;
                        send_blocks.push_back(block);
                    }
                    if(myrank == dst_rank){
                        recv_blocks.push_back(block);
                    }
                }
                g += len;
            }

            T* d_sendbuf = nullptr;
            if(send_total > 0){
                DEVICE_CHECK(deviceMallocAsync(&d_sendbuf, sizeof(T) * send_total, ddla_handle->stream));
                for(const auto& block : send_blocks){
                    DEVICE_CHECK(deviceMemcpy2DAsync(
                        d_sendbuf + block.send_offset, m * sizeof(T),
                        d_A + i_loc + block.src_loc * array_descA.lld(), array_descA.lld() * sizeof(T),
                        m * sizeof(T), block.len,
                        deviceMemcpyDeviceToDevice, ddla_handle->stream
                    ));
                }
            }

            #if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
            if(!send_blocks.empty() || !recv_blocks.empty()){
                CCL_CHECK(ncclGroupStart());
                for(const auto& block : send_blocks){
                    CCL_CHECK(cclSend(d_sendbuf + block.send_offset, block.count, block.dst_rank, grid_nccl_comm, ddla_handle->stream));
                }
                for(const auto& block : recv_blocks){
                    CCL_CHECK(cclRecv(d_block_A + block.dst_offset, block.count, block.src_rank, grid_nccl_comm, ddla_handle->stream));
                }
                CCL_CHECK(ncclGroupEnd());
            }
            #else
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            std::vector<T> h_rect_temp(std::max(send_total, m * target_cols));
            #endif
            for(const auto& block : blocks){
                if(myrank == block.src_rank){
                    auto it = std::find_if(send_blocks.begin(), send_blocks.end(), [&](const RectBlock& item){
                        return item.dst_rank == block.dst_rank && item.dst_offset == block.dst_offset;
                    });
                    assert(it != send_blocks.end());
                    #ifdef DDLA_USE_GPU_CPU_TUNNEL
                    MPI_CHECK(cclSend(h_rect_temp.data(), d_sendbuf + it->send_offset, block.count, block.dst_rank, ddla_handle->comm, ddla_handle->stream));
                    #else
                    CCL_CHECK(cclSend(d_sendbuf + it->send_offset, block.count, block.dst_rank, grid_nccl_comm, ddla_handle->stream));
                    #endif
                }else if(myrank == block.dst_rank){
                    #ifdef DDLA_USE_GPU_CPU_TUNNEL
                    MPI_CHECK(cclRecv(h_rect_temp.data(), d_block_A + block.dst_offset, block.count, block.src_rank, ddla_handle->comm, ddla_handle->stream));
                    #else
                    CCL_CHECK(cclRecv(d_block_A + block.dst_offset, block.count, block.src_rank, grid_nccl_comm, ddla_handle->stream));
                    #endif
                }
            }
            #endif

            if(target_cols > 0){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_rect_temp.data(), d_block_A, m * target_cols, 0, ddla_handle->row_comm, ddla_handle->stream));
                #else
                CCL_CHECK(cclBcast(d_block_A, m * target_cols, 0, row_nccl_comm, ddla_handle->stream));
                #endif
            }
            if(d_sendbuf != nullptr){
                DEVICE_CHECK(deviceFreeAsync(d_sendbuf, ddla_handle->stream));
            }
        }else if(sData == 'C'){
            const int target_i_loc = num_loc(ia, array_descA.mb(), array_descA.mypcol(), array_descA.irsrc(), array_descA.npcols());
            const int target_m_loc = num_loc(ia + m, array_descA.mb(), array_descA.mypcol(), array_descA.irsrc(), array_descA.npcols());
            const int target_cols = target_m_loc - target_i_loc;
            const deblasOperation_t pack_op = trans == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
            const T one = (T)1.0;
            const T zero = (T)0.0;
            std::vector<RectBlock> blocks;
            std::vector<RectBlock> send_blocks;
            std::vector<RectBlock> recv_blocks;
            int send_total = 0;

            for(int g = ia; g < ia + m; ){
                const int len = block_end(g, array_descA.mb(), ia + m) - g;
                const int src_row = indxg2p(g, array_descA.mb(), array_descA.irsrc(), array_descA.nprows());
                const int dst_col = indxg2p(g, array_descA.mb(), array_descA.irsrc(), array_descA.npcols());
                const int src_rank = ddla_handle->rc_to_rank(src_row, owner_col);
                const int dst_rank = ddla_handle->rc_to_rank(0, dst_col);
                const int dst_col_loc = num_loc(g, array_descA.mb(), dst_col, array_descA.irsrc(), array_descA.npcols())
                                      - num_loc(ia, array_descA.mb(), dst_col, array_descA.irsrc(), array_descA.npcols());
                RectBlock block{src_rank, dst_rank, n * len, -1, dst_col_loc * n,
                                indxg2l(g, array_descA.mb(), array_descA.nprows()), len};

                if(src_rank == dst_rank){
                    if(myrank == src_rank){
                        BLAS_CHECK(deblasGeam(
                            ddla_handle->blasH, pack_op, pack_op,
                            n, block.len,
                            one,
                            d_A + block.src_loc + j_loc * array_descA.lld(), array_descA.lld(),
                            zero,
                            d_A + block.src_loc + j_loc * array_descA.lld(), array_descA.lld(),
                            d_block_A + block.dst_offset, n
                        ));
                    }
                }else{
                    blocks.push_back(block);
                    if(myrank == src_rank){
                        block.send_offset = send_total;
                        send_total += block.count;
                        send_blocks.push_back(block);
                    }
                    if(myrank == dst_rank){
                        recv_blocks.push_back(block);
                    }
                }
                g += len;
            }

            T* d_sendbuf = nullptr;
            if(send_total > 0){
                DEVICE_CHECK(deviceMallocAsync(&d_sendbuf, sizeof(T) * send_total, ddla_handle->stream));
                for(const auto& block : send_blocks){
                    BLAS_CHECK(deblasGeam(
                        ddla_handle->blasH, pack_op, pack_op,
                        n, block.len,
                        one,
                        d_A + block.src_loc + j_loc * array_descA.lld(), array_descA.lld(),
                        zero,
                        d_A + block.src_loc + j_loc * array_descA.lld(), array_descA.lld(),
                        d_sendbuf + block.send_offset, n
                    ));
                }
            }

            #if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
            if(!send_blocks.empty() || !recv_blocks.empty()){
                CCL_CHECK(ncclGroupStart());
                for(const auto& block : send_blocks){
                    CCL_CHECK(cclSend(d_sendbuf + block.send_offset, block.count, block.dst_rank, grid_nccl_comm, ddla_handle->stream));
                }
                for(const auto& block : recv_blocks){
                    CCL_CHECK(cclRecv(d_block_A + block.dst_offset, block.count, block.src_rank, grid_nccl_comm, ddla_handle->stream));
                }
                CCL_CHECK(ncclGroupEnd());
            }
            #else
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            std::vector<T> h_rect_temp(std::max(send_total, n * target_cols));
            #endif
            for(const auto& block : blocks){
                if(myrank == block.src_rank){
                    auto it = std::find_if(send_blocks.begin(), send_blocks.end(), [&](const RectBlock& item){
                        return item.dst_rank == block.dst_rank && item.dst_offset == block.dst_offset;
                    });
                    assert(it != send_blocks.end());
                    #ifdef DDLA_USE_GPU_CPU_TUNNEL
                    MPI_CHECK(cclSend(h_rect_temp.data(), d_sendbuf + it->send_offset, block.count, block.dst_rank, ddla_handle->comm, ddla_handle->stream));
                    #else
                    CCL_CHECK(cclSend(d_sendbuf + it->send_offset, block.count, block.dst_rank, grid_nccl_comm, ddla_handle->stream));
                    #endif
                }else if(myrank == block.dst_rank){
                    #ifdef DDLA_USE_GPU_CPU_TUNNEL
                    MPI_CHECK(cclRecv(h_rect_temp.data(), d_block_A + block.dst_offset, block.count, block.src_rank, ddla_handle->comm, ddla_handle->stream));
                    #else
                    CCL_CHECK(cclRecv(d_block_A + block.dst_offset, block.count, block.src_rank, grid_nccl_comm, ddla_handle->stream));
                    #endif
                }
            }
            #endif

            if(target_cols > 0){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_rect_temp.data(), d_block_A, n * target_cols, 0, ddla_handle->col_comm, ddla_handle->stream));
                #else
                CCL_CHECK(cclBcast(d_block_A, n * target_cols, 0, col_nccl_comm, ddla_handle->stream));
                #endif
            }
            if(d_sendbuf != nullptr){
                DEVICE_CHECK(deviceFreeAsync(d_sendbuf, ddla_handle->stream));
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
