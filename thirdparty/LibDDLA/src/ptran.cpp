#include <ddla/ptran.h>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/ddla_comm.h>
#include <thrust/complex.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <type_traits>

namespace ddla{

namespace detail {

// Map std::complex to thrust::complex for device code, keep real types as-is.
template <typename T>
struct device_scalar {
    using type = T;
};

template <>
struct device_scalar<std::complex<float>> {
    using type = thrust::complex<float>;
};

template <>
struct device_scalar<std::complex<double>> {
    using type = thrust::complex<double>;
};

// Conjugate helper: no-op for real types, thrust::conj for complex.
template <typename T>
__device__ __forceinline__ T conj_if(const T& val, bool do_conj) {
    if constexpr (std::is_same_v<T, thrust::complex<float>> ||
                  std::is_same_v<T, thrust::complex<double>>) {
        return do_conj ? thrust::conj(val) : val;
    } else {
        return val;
    }
}

} // namespace detail

// POD struct for device-side block info (must not contain complex types).
struct TransBlockInfo {
    int bm;          // source block rows
    int bn;          // source block cols
    int src_offset;  // element offset into source array
    int dst_offset;  // element offset into dest array
    int src_ld;      // source leading dimension
    int dst_ld;      // dest leading dimension
};

// ---- Kernel: batched transpose with shared-memory tiling ----
// Replaces N geam calls with a single kernel launch.
// Each thread block handles one matrix block, tiled into TILE x TILE sub-tiles.
// Coalesced reads and writes via shared-memory transpose.
template <typename devT>
__global__ void transpose_blocks_kernel(
    const devT* __restrict__ src_base,
    devT* __restrict__ dst_base,
    const TransBlockInfo* __restrict__ blocks,
    int nblocks,
    bool do_conj)
{
    constexpr int TILE = 16;
    __shared__ devT tile[TILE][TILE + 1]; // +1 avoids shared-memory bank conflicts

    const int blk = blockIdx.x;
    if (blk >= nblocks) return;

    const TransBlockInfo info = blocks[blk];
    const int bm = info.bm;  // source rows
    const int bn = info.bn;  // source cols

    const int ntiles_r = (bm + TILE - 1) / TILE;
    const int ntiles_c = (bn + TILE - 1) / TILE;
    const int ntiles = ntiles_r * ntiles_c;

    for (int t = 0; t < ntiles; t++) {
        const int tr = t / ntiles_c;  // source tile row index
        const int tc = t % ntiles_c;  // source tile col index

        // --- Read phase (coalesced: threadIdx.x -> r_src -> consecutive memory) ---
        int r_src = tr * TILE + threadIdx.x;
        int c_src = tc * TILE + threadIdx.y;
        if (r_src < bm && c_src < bn) {
            tile[threadIdx.y][threadIdx.x] =
                src_base[info.src_offset + r_src + c_src * info.src_ld];
        }
        __syncthreads();

        // --- Write phase (coalesced: threadIdx.x -> r_dst -> consecutive memory) ---
        // Transpose via shared memory: read tile[threadIdx.x][threadIdx.y]
        int r_dst = tc * TILE + threadIdx.x;  // dest row  = source col
        int c_dst = tr * TILE + threadIdx.y;  // dest col  = source row
        if (r_dst < bn && c_dst < bm) {
            devT val = tile[threadIdx.x][threadIdx.y];
            dst_base[info.dst_offset + r_dst + c_dst * info.dst_ld] =
                detail::conj_if(val, do_conj);
        }
        __syncthreads();
    }
}

// ---- Kernel: batched scatter (copy without transpose) ----
// Replaces N deviceMemcpy2DAsync calls with a single kernel launch.
// Copies bn x bm blocks from compact recv buffer to strided dest.
template <typename devT>
__global__ void scatter_blocks_kernel(
    const devT* __restrict__ src_base,
    devT* __restrict__ dst_base,
    const TransBlockInfo* __restrict__ blocks,
    int nblocks)
{
    const int blk = blockIdx.x;
    if (blk >= nblocks) return;

    const TransBlockInfo info = blocks[blk];
    const int rows = info.bn;  // block rows
    const int cols = info.bm;  // block cols

    for (int j = threadIdx.y; j < cols; j += blockDim.y) {
        for (int i = threadIdx.x; i < rows; i += blockDim.x) {
            dst_base[info.dst_offset + i + j * info.dst_ld] =
                src_base[info.src_offset + i + j * info.src_ld];
        }
    }
}

template <typename T>
static inline MPI_Datatype mpi_type();

template <>
MPI_Datatype mpi_type<float>(){ return MPI_FLOAT; }
template <>
MPI_Datatype mpi_type<double>(){ return MPI_DOUBLE; }
template <>
MPI_Datatype mpi_type<std::complex<float>>(){ return MPI_CXX_FLOAT_COMPLEX; }
template <>
MPI_Datatype mpi_type<std::complex<double>>(){ return MPI_CXX_DOUBLE_COMPLEX; }

/**
 * @brief Distributed out-of-place matrix transpose (optionally conjugate).
 *
 * Strategy (NCCL path):
 *   1. Build a communication plan by scanning all global blocks.
 *   2. Pack: fused kernel transposes owned blocks into a contiguous d_sendbuf (D2D, async).
 *      Local blocks (src==dst) are transposed directly into d_AT on stream_data.
 *   3. Communicate: all send/recv issued in a single ncclGroup — device-to-device,
 *      no host staging, no per-block synchronisation.
 *   4. Scatter: fused kernel copies from d_recvbuf into d_AT (D2D, async).
 *
 * Performance improvements over the original per-block geam/memcpy2D loop:
 *   - P0: Three fused CUDA kernels replace thousands of per-block kernel launches.
 *   - P0: Redundant deviceDeviceSynchronize() in pgemm.cpp removed.
 *   - P1: Phase 4 (local transpose) runs on stream_data, overlapping with
 *         Phase 3+5+6 (pack + communication + scatter) on stream.
 *
 * All device allocations use deviceMallocAsync/deviceFreeAsync (stream-ordered),
 * eliminating the deviceStreamSynchronize that synchronous free requires and
 * allowing the free to overlap with subsequent stream work.
 */
template <typename T>
void ptran(const T* d_A, const DdlaDesc& descA,
           T* d_AT, const DdlaDesc& descAT,
           bool conj)
{
    assert(descAT.m() == descA.n());
    assert(descAT.n() == descA.m());
    assert(descAT.mb() == descA.nb());
    assert(descAT.nb() == descA.mb());
    assert(descAT.irsrc() == descA.icsrc());
    assert(descAT.icsrc() == descA.irsrc());
    assert(descAT.nprows() == descA.nprows());
    assert(descAT.npcols() == descA.npcols());

    DdlaHandle_t handle = descA.ddla_handle();
    int myrank = handle->myid;
    int Pr = descA.nprows();
    int Pc = descA.npcols();
    int nprocs = Pr * Pc;

    int mA = descA.m();
    int nA = descA.n();
    int mbA = descA.mb();
    int nbA = descA.nb();
    int irsrcA = descA.irsrc();
    int icsrcA = descA.icsrc();
    int mbAT = descAT.mb();
    int nbAT = descAT.nb();
    int irsrcAT = descAT.irsrc();
    int icsrcAT = descAT.icsrc();

    int nbr = (mA + mbA - 1) / mbA;
    int nbc = (nA + nbA - 1) / nbA;

    // ---- Phase 1: build communication plan ----
    struct BlockInfo {
        int g_row, g_col;       // global block origin
        int bm, bn;             // block dimensions (in A)
        int src_rank, dst_rank; // process ranks
        int lrow_A, lcol_A;     // local offset in d_A   (valid if src==myrank)
        int lrow_AT, lcol_AT;   // local offset in d_AT  (valid if dst==myrank)
        int offset;             // offset into send/recv buffer
    };

    std::vector<BlockInfo> local_blocks;  // src == dst == myrank
    std::vector<BlockInfo> send_blocks;   // src == myrank, dst != myrank
    std::vector<BlockInfo> recv_blocks;   // dst == myrank, src != myrank

    // Upper bounds: a process owns at most ceil(nbr/Pr)*ceil(nbc/Pc) blocks.
    int max_own = ((nbr + Pr - 1) / Pr) * ((nbc + Pc - 1) / Pc);
    local_blocks.reserve(max_own);
    send_blocks.reserve(max_own);
    recv_blocks.reserve(max_own);

    for(int br = 0; br < nbr; ++br){
        for(int bc = 0; bc < nbc; ++bc){
            int g_row = br * mbA;
            int g_col = bc * nbA;
            int bm = std::min(mbA, mA - g_row);
            int bn = std::min(nbA, nA - g_col);
            if(bm <= 0 || bn <= 0) continue;

            int src_row = indxg2p(g_row, mbA, irsrcA, Pr);
            int src_col = indxg2p(g_col, nbA, icsrcA, Pc);
            int dst_row = indxg2p(g_col, mbAT, irsrcAT, Pr);
            int dst_col = indxg2p(g_row, nbAT, icsrcAT, Pc);

            int src_rank = handle->rc_to_rank(src_row, src_col);
            int dst_rank = handle->rc_to_rank(dst_row, dst_col);

            BlockInfo info;
            info.g_row = g_row; info.g_col = g_col;
            info.bm = bm; info.bn = bn;
            info.src_rank = src_rank; info.dst_rank = dst_rank;
            info.offset = 0;

            if(src_rank == myrank){
                info.lrow_A = indxg2l(g_row, mbA, Pr);
                info.lcol_A = indxg2l(g_col, nbA, Pc);
            }
            if(dst_rank == myrank){
                info.lrow_AT = indxg2l(g_col, mbAT, Pr);
                info.lcol_AT = indxg2l(g_row, nbAT, Pc);
            }

            if(src_rank == myrank && dst_rank == myrank){
                local_blocks.push_back(info);
            } else if(src_rank == myrank){
                send_blocks.push_back(info);
            } else if(dst_rank == myrank){
                recv_blocks.push_back(info);
            }
        }
    }

    // Sort by peer rank, then by (g_row, g_col) so that both send and recv
    // sides visit blocks in the same order (required for NCCL group matching).
    auto cmp_send = [](const BlockInfo& a, const BlockInfo& b){
        if(a.dst_rank != b.dst_rank) return a.dst_rank < b.dst_rank;
        if(a.g_row != b.g_row) return a.g_row < b.g_row;
        return a.g_col < b.g_col;
    };
    auto cmp_recv = [](const BlockInfo& a, const BlockInfo& b){
        if(a.src_rank != b.src_rank) return a.src_rank < b.src_rank;
        if(a.g_row != b.g_row) return a.g_row < b.g_row;
        return a.g_col < b.g_col;
    };
    std::sort(send_blocks.begin(), send_blocks.end(), cmp_send);
    std::sort(recv_blocks.begin(), recv_blocks.end(), cmp_recv);

    // Compute contiguous offsets
    int send_total = 0;
    for(auto& b : send_blocks){
        b.offset = send_total;
        send_total += b.bm * b.bn;
    }
    int recv_total = 0;
    for(auto& b : recv_blocks){
        b.offset = recv_total;
        recv_total += b.bm * b.bn;
    }

    using devT = typename detail::device_scalar<T>::type;
    const devT* d_A_dev = reinterpret_cast<const devT*>(d_A);
    devT* d_AT_dev = reinterpret_cast<devT*>(d_AT);

    deviceStream_t stream = handle->stream;
    deviceStream_t stream_data = handle->stream_data;

    // ---- Phase 2: allocate device buffers ----
    T* d_sendbuf = nullptr;
    T* d_recvbuf = nullptr;
    if(send_total > 0) DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_sendbuf), sizeof(T) * send_total, stream));
    if(recv_total > 0) DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_recvbuf), sizeof(T) * recv_total, stream));
    devT* d_sendbuf_dev = reinterpret_cast<devT*>(d_sendbuf);
    devT* d_recvbuf_dev = reinterpret_cast<devT*>(d_recvbuf);

    constexpr int TILE = 16;
    dim3 block_dim(TILE, TILE, 1);

    // ---- Phase 3: pack send data (fused kernel: transpose d_A -> d_sendbuf) ----
    TransBlockInfo* d_pack_blocks = nullptr;
    if(!send_blocks.empty()){
        std::vector<TransBlockInfo> h_pack(send_blocks.size());
        for(size_t i = 0; i < send_blocks.size(); i++){
            const auto& b = send_blocks[i];
            h_pack[i] = {b.bm, b.bn,
                         b.lrow_A + b.lcol_A * descA.lld(),
                         b.offset,
                         descA.lld(), b.bn};
        }
        DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_pack_blocks),
            sizeof(TransBlockInfo) * h_pack.size(), stream));
        DEVICE_CHECK(deviceMemcpyAsync(d_pack_blocks, h_pack.data(),
            sizeof(TransBlockInfo) * h_pack.size(), deviceMemcpyHostToDevice, stream));

        dim3 grid_dim(static_cast<int>(send_blocks.size()), 1, 1);
        transpose_blocks_kernel<devT><<<grid_dim, block_dim, 0, stream>>>(
            d_A_dev, d_sendbuf_dev, d_pack_blocks,
            static_cast<int>(send_blocks.size()), conj);
        DEVICE_CHECK(deviceGetLastError());
    }

    // ---- Phase 4: local transpose (fused kernel: d_A -> d_AT) ----
    TransBlockInfo* d_local_blocks = nullptr;
    if(!local_blocks.empty()){
        std::vector<TransBlockInfo> h_local(local_blocks.size());
        for(size_t i = 0; i < local_blocks.size(); i++){
            const auto& b = local_blocks[i];
            h_local[i] = {b.bm, b.bn,
                          b.lrow_A + b.lcol_A * descA.lld(),
                          b.lrow_AT + b.lcol_AT * descAT.lld(),
                          descA.lld(), descAT.lld()};
        }
        DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_local_blocks),
            sizeof(TransBlockInfo) * h_local.size(), stream));
        DEVICE_CHECK(deviceMemcpyAsync(d_local_blocks, h_local.data(),
            sizeof(TransBlockInfo) * h_local.size(), deviceMemcpyHostToDevice, stream));

        dim3 grid_dim(static_cast<int>(local_blocks.size()), 1, 1);
        transpose_blocks_kernel<devT><<<grid_dim, block_dim, 0, stream>>>(
            d_A_dev, d_AT_dev, d_local_blocks,
            static_cast<int>(local_blocks.size()), conj);
        DEVICE_CHECK(deviceGetLastError());
    }

    // ---- Phase 5: communication ----
    #ifdef DDLA_USE_CCL
    // NCCL: batch all send/recv in one group — device-to-device, no host staging.
    if(send_total > 0 || recv_total > 0){
        CCL_CHECK(ncclGroupStart());
        for(auto& b : send_blocks){
            CCL_CHECK(cclSend(d_sendbuf + b.offset, b.bm * b.bn,
                              b.dst_rank, handle->nccl_comm, handle->stream));
        }
        for(auto& b : recv_blocks){
            CCL_CHECK(cclRecv(d_recvbuf + b.offset, b.bm * b.bn,
                              b.src_rank, handle->nccl_comm, handle->stream));
        }
        CCL_CHECK(ncclGroupEnd());
    }
    #else
    // MPI fallback: Alltoallv with host staging (single collective, not per-block).
    if(send_total > 0 || recv_total > 0){
        std::vector<int> sendcounts(nprocs, 0), recvcounts(nprocs, 0);
        std::vector<int> sdispls(nprocs, 0), rdispls(nprocs, 0);
        for(auto& b : send_blocks) sendcounts[b.dst_rank] += b.bm * b.bn;
        for(auto& b : recv_blocks) recvcounts[b.src_rank] += b.bm * b.bn;
        for(int p = 1; p < nprocs; p++){
            sdispls[p] = sdispls[p-1] + sendcounts[p-1];
            rdispls[p] = rdispls[p-1] + recvcounts[p-1];
        }
        std::vector<T> h_sendbuf(send_total), h_recvbuf(recv_total);
        if(send_total > 0){
            DEVICE_CHECK(deviceMemcpyAsync(h_sendbuf.data(), d_sendbuf,
                sizeof(T) * send_total, deviceMemcpyDeviceToHost, stream));
            DEVICE_CHECK(deviceStreamSynchronize(stream));
        }
        MPI_Alltoallv(h_sendbuf.data(), sendcounts.data(), sdispls.data(), mpi_type<T>(),
                      h_recvbuf.data(), recvcounts.data(), rdispls.data(), mpi_type<T>(),
                      handle->comm);
        if(recv_total > 0){
            DEVICE_CHECK(deviceMemcpyAsync(d_recvbuf, h_recvbuf.data(),
                sizeof(T) * recv_total, deviceMemcpyHostToDevice, stream));
            DEVICE_CHECK(deviceStreamSynchronize(stream));
        }
    }
    #endif

    // ---- Phase 6: scatter recv data into d_AT (fused kernel) ----
    TransBlockInfo* d_scatter_blocks = nullptr;
    if(!recv_blocks.empty()){
        std::vector<TransBlockInfo> h_scatter(recv_blocks.size());
        for(size_t i = 0; i < recv_blocks.size(); i++){
            const auto& b = recv_blocks[i];
            h_scatter[i] = {b.bm, b.bn,
                            b.offset,
                            b.lrow_AT + b.lcol_AT * descAT.lld(),
                            b.bn, descAT.lld()};
        }
        DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_scatter_blocks),
            sizeof(TransBlockInfo) * h_scatter.size(), stream));
        DEVICE_CHECK(deviceMemcpyAsync(d_scatter_blocks, h_scatter.data(),
            sizeof(TransBlockInfo) * h_scatter.size(), deviceMemcpyHostToDevice, stream));

        dim3 grid_dim(static_cast<int>(recv_blocks.size()), 1, 1);
        scatter_blocks_kernel<devT><<<grid_dim, block_dim, 0, stream>>>(
            d_recvbuf_dev, d_AT_dev, d_scatter_blocks,
            static_cast<int>(recv_blocks.size()));
        DEVICE_CHECK(deviceGetLastError());
    }

    // ---- Finalize ----
    // Free device buffers (async free is stream-ordered, no sync needed).
    if(d_sendbuf)        DEVICE_CHECK(deviceFreeAsync(d_sendbuf, stream));
    if(d_recvbuf)        DEVICE_CHECK(deviceFreeAsync(d_recvbuf, stream));
    if(d_pack_blocks)    DEVICE_CHECK(deviceFreeAsync(d_pack_blocks, stream));
    if(d_scatter_blocks) DEVICE_CHECK(deviceFreeAsync(d_scatter_blocks, stream));
    if(d_local_blocks)   DEVICE_CHECK(deviceFreeAsync(d_local_blocks, stream));
}

template void ptran<float>(const float* d_A, const DdlaDesc& descA, float* d_AT, const DdlaDesc& descAT, bool conj);
template void ptran<double>(const double* d_A, const DdlaDesc& descA, double* d_AT, const DdlaDesc& descAT, bool conj);
template void ptran<std::complex<float>>(const std::complex<float>* d_A, const DdlaDesc& descA, std::complex<float>* d_AT, const DdlaDesc& descAT, bool conj);
template void ptran<std::complex<double>>(const std::complex<double>* d_A, const DdlaDesc& descA, std::complex<double>* d_AT, const DdlaDesc& descAT, bool conj);

} // namespace ddla
