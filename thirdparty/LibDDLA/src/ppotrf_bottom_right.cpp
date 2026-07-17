#include "potrf_bottom_right_internal.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <type_traits>
#include <vector>

#include <ddla/ddla.h>
#include <ddla/ddla_comm.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/gemmBatched.h>
#include <ddla/herk.h>
#include <ddla/syrk.h>
#include <ddla/trsm.h>

namespace ddla{

namespace {

template <typename T>
void update_diagonal_tile(
    const deblasHandle_t blas_handle,
    const char uplo,
    const int nb, const int panel_width,
    const T* d_panel, const int ld_panel,
    T* d_tile, const int ldd
)
{
    const bool upper = uplo == 'U';
    const deblasFillMode_t fill_mode = upper
        ? DEBLAS_FILL_MODE_UPPER
        : DEBLAS_FILL_MODE_LOWER;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
        const T minus_one = T(-1);
        const T one = T(1);
        BLAS_CHECK(deblasSyrk(
            blas_handle,
            fill_mode, upper ? DEBLAS_OP_N : DEBLAS_OP_T,
            nb, panel_width,
            minus_one, d_panel, ld_panel,
            one, d_tile, ldd
        ));
    }else{
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        BLAS_CHECK(deblasHerk(
            blas_handle,
            fill_mode, upper ? DEBLAS_OP_N : DEBLAS_OP_C,
            nb, panel_width,
            minus_one, d_panel, ld_panel,
            one, d_tile, ldd
        ));
    }
}

} // namespace

template <typename T>
void ppotrf_bottom_right(
    const char& uplo, const int& n,
    T* d_A, const DdlaDesc& descA, int& info
)
{
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(descA.is_initialized());
    assert(n == descA.m() && n == descA.n());
    assert(descA.mb() == descA.nb());
    assert(descA.nprows() == descA.npcols());

    const DdlaHandle_t handle = descA.ddla_handle();
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    const int nb = descA.nb();
    assert(nb > 0);

    const int nprocs_dim = descA.nprows();
    const int myprow = descA.myprow();
    const int mypcol = descA.mypcol();
    const int lld = descA.lld();

    const deviceStream_t stream = handle->stream;
    const deblasHandle_t blas_handle = handle->blasH;

#ifdef DDLA_USE_CCL
    const ncclComm_t row_comm = handle->nccl_row_comm;
    const ncclComm_t col_comm = handle->nccl_col_comm;
#else
    const MPI_Comm row_comm = handle->row_comm;
    const MPI_Comm col_comm = handle->col_comm;
#endif

    const int max_local_rows = num_loc(
        n, nb, myprow, descA.irsrc(), nprocs_dim);
    const int max_local_cols = num_loc(
        n, nb, mypcol, descA.icsrc(), nprocs_dim);
    assert(d_A != nullptr || max_local_rows == 0 || max_local_cols == 0);

    T* d_diag = nullptr;
    T* d_row_panel = nullptr;
    T* d_col_panel = nullptr;
    int* d_info = nullptr;
    T** d_left_array = nullptr;
    T** d_right_array = nullptr;
    T** d_target_array = nullptr;

    const int max_row_blocks = max_local_rows / nb;
    const int max_col_blocks = max_local_cols / nb;
    const std::size_t max_batch_count =
        static_cast<std::size_t>(max_row_blocks) * max_col_blocks;

    auto device_malloc_if_nonzero = [&](void** ptr, const std::size_t bytes)
    {
        if(bytes == 0){
            *ptr = nullptr;
            return;
        }
        DEVICE_CHECK(deviceMallocAsync(ptr, bytes, stream));
    };
    auto device_free_if_nonnull = [&](void* ptr)
    {
        if(ptr != nullptr){
            DEVICE_CHECK(deviceFreeAsync(ptr, stream));
        }
    };

    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_diag),
        static_cast<std::size_t>(nb) * nb * sizeof(T));
    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_row_panel),
        static_cast<std::size_t>(max_local_rows) * nb * sizeof(T));
    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_col_panel),
        static_cast<std::size_t>(max_local_cols) * nb * sizeof(T));
    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_info), sizeof(int));

    const std::size_t pointer_buffer_bytes =
        max_batch_count * sizeof(T*);
    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_left_array), pointer_buffer_bytes);
    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_right_array), pointer_buffer_bytes);
    device_malloc_if_nonzero(
        reinterpret_cast<void**>(&d_target_array), pointer_buffer_bytes);

    std::vector<T*> h_left_array(max_batch_count);
    std::vector<T*> h_right_array(max_batch_count);
    std::vector<T*> h_target_array(max_batch_count);

#ifdef DDLA_USE_GPU_CPU_TUNNEL
    const std::size_t host_buffer_count = std::max({
        static_cast<std::size_t>(nb) * nb,
        static_cast<std::size_t>(max_local_rows) * nb,
        static_cast<std::size_t>(max_local_cols) * nb
    });
    std::vector<T> h_communication_buffer(host_buffer_count);
#endif

    auto cleanup = [&]()
    {
        device_free_if_nonnull(d_left_array);
        device_free_if_nonnull(d_right_array);
        device_free_if_nonnull(d_target_array);
        device_free_if_nonnull(d_diag);
        device_free_if_nonnull(d_row_panel);
        device_free_if_nonnull(d_col_panel);
        device_free_if_nonnull(d_info);
        DEVICE_CHECK(deviceStreamSynchronize(stream));
    };

    const int last_block_start = ((n - 1) / nb) * nb;
    // block_start is always a block boundary.  The first diagonal block may be
    // partial, but every leading tile updated below is therefore nb-by-nb.
    for(int block_start = last_block_start;
        block_start >= 0;
        block_start -= nb){
        const int block_width = std::min(nb, n - block_start);
        const int owner_row = indxg2p(
            block_start, nb, descA.irsrc(), nprocs_dim);
        const int owner_col = indxg2p(
            block_start, nb, descA.icsrc(), nprocs_dim);

        const int local_row_prefix = num_loc(
            block_start, nb, myprow, descA.irsrc(), nprocs_dim);
        const int local_col_prefix = num_loc(
            block_start, nb, mypcol, descA.icsrc(), nprocs_dim);
        assert(local_row_prefix % nb == 0);
        assert(local_col_prefix % nb == 0);

        int block_info = 0;
        if(myprow == owner_row && mypcol == owner_col){
            T* const d_local_diag = d_A + local_row_prefix
                                  + static_cast<std::size_t>(local_col_prefix) * lld;
            detail::potrf_bottom_right_block(
                uplo, block_width, d_local_diag, lld,
                block_start, d_diag, d_info, block_info, handle
            );
        }

        MPI_CHECK(MPI_Bcast(
            &block_info, 1, MPI_INT,
            handle->rc_to_rank(owner_row, owner_col), handle->comm));
        info = block_info;
        if(info != 0){
            cleanup();
            return;
        }

        if(myprow == owner_row && mypcol == owner_col){
            const T* const d_local_diag = d_A + local_row_prefix
                                        + static_cast<std::size_t>(local_col_prefix) * lld;
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_diag, static_cast<std::size_t>(block_width) * sizeof(T),
                d_local_diag, static_cast<std::size_t>(lld) * sizeof(T),
                static_cast<std::size_t>(block_width) * sizeof(T), block_width,
                deviceMemcpyDeviceToDevice, stream
            ));
        }

        if(uplo == 'U' && mypcol == owner_col){
#ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(
                h_communication_buffer.data(), d_diag,
                static_cast<std::size_t>(block_width) * block_width,
                owner_row, handle->col_comm, stream));
#else
            CCL_CHECK(cclBcast(
                d_diag,
                static_cast<std::size_t>(block_width) * block_width,
                owner_row, col_comm, stream));
#endif
        }

        if(uplo == 'L' && myprow == owner_row){
#ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(
                h_communication_buffer.data(), d_diag,
                static_cast<std::size_t>(block_width) * block_width,
                owner_col, handle->row_comm, stream));
#else
            CCL_CHECK(cclBcast(
                d_diag,
                static_cast<std::size_t>(block_width) * block_width,
                owner_col, row_comm, stream));
#endif
        }

        if(uplo == 'U' && local_row_prefix > 0){
            if(mypcol == owner_col){
                const T one = T(1);
                T* const d_local_panel = d_A
                                       + static_cast<std::size_t>(local_col_prefix) * lld;
                BLAS_CHECK(deblasTrsm(
                    blas_handle,
                    DEBLAS_SIDE_RIGHT,
                    DEBLAS_FILL_MODE_UPPER,
                    DEBLAS_OP_C,
                    DEBLAS_DIAG_NON_UNIT,
                    local_row_prefix, block_width,
                    one,
                    d_diag, block_width,
                    d_local_panel, lld
                ));
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_row_panel,
                    static_cast<std::size_t>(local_row_prefix) * sizeof(T),
                    d_local_panel,
                    static_cast<std::size_t>(lld) * sizeof(T),
                    static_cast<std::size_t>(local_row_prefix) * sizeof(T),
                    block_width,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }

#ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(
                h_communication_buffer.data(), d_row_panel,
                static_cast<std::size_t>(local_row_prefix) * block_width,
                owner_col, handle->row_comm, stream));
#else
            CCL_CHECK(cclBcast(
                d_row_panel,
                static_cast<std::size_t>(local_row_prefix) * block_width,
                owner_col, row_comm, stream));
#endif
        }

        if(uplo == 'U'){
            // A row tile owned by relay_row under irsrc is the same global
            // tile that this process column owns under icsrc.
            const int relay_row =
                (mypcol + descA.irsrc() - descA.icsrc() + nprocs_dim)
                % nprocs_dim;
            if(local_col_prefix > 0){
                if(myprow == relay_row){
                    const int relay_row_count = num_loc(
                        block_start, nb, relay_row,
                        descA.irsrc(), nprocs_dim);
                    assert(relay_row_count == local_col_prefix);
                    DEVICE_CHECK(deviceMemcpyAsync(
                        d_col_panel, d_row_panel,
                        static_cast<std::size_t>(local_col_prefix)
                            * block_width * sizeof(T),
                        deviceMemcpyDeviceToDevice, stream
                    ));
                }

#ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(
                    h_communication_buffer.data(), d_col_panel,
                    static_cast<std::size_t>(local_col_prefix) * block_width,
                    relay_row, handle->col_comm, stream));
#else
                CCL_CHECK(cclBcast(
                    d_col_panel,
                    static_cast<std::size_t>(local_col_prefix) * block_width,
                    relay_row, col_comm, stream));
#endif
            }
        }else{
            if(local_col_prefix > 0){
                if(myprow == owner_row){
                    const T one = T(1);
                    T* const d_local_panel = d_A + local_row_prefix;
                    const deblasOperation_t trans =
                        std::is_same_v<T, float>
                            || std::is_same_v<T, double>
                        ? DEBLAS_OP_T
                        : DEBLAS_OP_C;
                    BLAS_CHECK(deblasTrsm(
                        blas_handle,
                        DEBLAS_SIDE_LEFT,
                        DEBLAS_FILL_MODE_LOWER,
                        trans,
                        DEBLAS_DIAG_NON_UNIT,
                        block_width, local_col_prefix,
                        one,
                        d_diag, block_width,
                        d_local_panel, lld
                    ));
                    DEVICE_CHECK(deviceMemcpy2DAsync(
                        d_col_panel,
                        static_cast<std::size_t>(block_width) * sizeof(T),
                        d_local_panel,
                        static_cast<std::size_t>(lld) * sizeof(T),
                        static_cast<std::size_t>(block_width) * sizeof(T),
                        local_col_prefix,
                        deviceMemcpyDeviceToDevice, stream
                    ));
                }

#ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(
                    h_communication_buffer.data(), d_col_panel,
                    static_cast<std::size_t>(local_col_prefix) * block_width,
                    owner_row, handle->col_comm, stream));
#else
                CCL_CHECK(cclBcast(
                    d_col_panel,
                    static_cast<std::size_t>(local_col_prefix) * block_width,
                    owner_row, col_comm, stream));
#endif
            }

            // A column tile owned by relay_col under icsrc is the same global
            // tile that this process row owns under irsrc.
            const int relay_col =
                (myprow + descA.icsrc() - descA.irsrc() + nprocs_dim)
                % nprocs_dim;
            if(local_row_prefix > 0){
                if(mypcol == relay_col){
                    const int relay_col_count = num_loc(
                        block_start, nb, relay_col,
                        descA.icsrc(), nprocs_dim);
                    assert(relay_col_count == local_row_prefix);
                    DEVICE_CHECK(deviceMemcpyAsync(
                        d_row_panel, d_col_panel,
                        static_cast<std::size_t>(local_row_prefix)
                            * block_width * sizeof(T),
                        deviceMemcpyDeviceToDevice, stream
                    ));
                }

#ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(
                    h_communication_buffer.data(), d_row_panel,
                    static_cast<std::size_t>(local_row_prefix) * block_width,
                    relay_col, handle->row_comm, stream));
#else
                CCL_CHECK(cclBcast(
                    d_row_panel,
                    static_cast<std::size_t>(local_row_prefix) * block_width,
                    relay_col, row_comm, stream));
#endif
            }
        }

        int batch_count = 0;
        for(int local_col = 0;
            local_col < local_col_prefix;
            local_col += nb){
            const int global_col = indxl2g(
                local_col, nb, mypcol, descA.icsrc(), nprocs_dim);
            for(int local_row = 0;
                local_row < local_row_prefix;
                local_row += nb){
                const int global_row = indxl2g(
                    local_row, nb, myprow, descA.irsrc(), nprocs_dim);
                if((uplo == 'U' && global_row > global_col)
                   || (uplo == 'L' && global_row < global_col)){
                    continue;
                }

                T* const d_target = d_A + local_row
                                  + static_cast<std::size_t>(local_col) * lld;
                T* const d_left = uplo == 'U'
                    ? d_row_panel + local_row
                    : d_row_panel
                        + static_cast<std::size_t>(local_row) * block_width;
                T* const d_right = uplo == 'U'
                    ? d_col_panel + local_col
                    : d_col_panel
                        + static_cast<std::size_t>(local_col) * block_width;
                const int ld_left = uplo == 'U'
                    ? local_row_prefix
                    : block_width;
                const int ld_right = uplo == 'U'
                    ? local_col_prefix
                    : block_width;
                if(global_row == global_col){
                    update_diagonal_tile(
                        blas_handle,
                        uplo,
                        nb, block_width,
                        d_left, ld_left,
                        d_target, lld
                    );
                    continue;
                }

                assert(static_cast<std::size_t>(batch_count)
                       < max_batch_count);
                h_left_array[batch_count] = d_left;
                h_right_array[batch_count] = d_right;
                h_target_array[batch_count] = d_target;
                ++batch_count;
            }
        }

        if(batch_count > 0){
            const std::size_t active_pointer_bytes =
                static_cast<std::size_t>(batch_count) * sizeof(T*);
            DEVICE_CHECK(deviceMemcpyAsync(
                d_left_array, h_left_array.data(), active_pointer_bytes,
                deviceMemcpyHostToDevice, stream));
            DEVICE_CHECK(deviceMemcpyAsync(
                d_right_array, h_right_array.data(), active_pointer_bytes,
                deviceMemcpyHostToDevice, stream));
            DEVICE_CHECK(deviceMemcpyAsync(
                d_target_array, h_target_array.data(), active_pointer_bytes,
                deviceMemcpyHostToDevice, stream));

            const T minus_one = T(-1);
            const T one = T(1);
            BLAS_CHECK(deblasGemmBatched(
                blas_handle,
                uplo == 'U' ? DEBLAS_OP_N : DEBLAS_OP_C,
                uplo == 'U' ? DEBLAS_OP_C : DEBLAS_OP_N,
                nb, nb, block_width,
                minus_one,
                d_left_array,
                uplo == 'U' ? local_row_prefix : block_width,
                d_right_array,
                uplo == 'U' ? local_col_prefix : block_width,
                one,
                d_target_array, lld,
                batch_count
            ));
        }
    }

    cleanup();
}

template void ppotrf_bottom_right<float>(
    const char&, const int&, float*, const DdlaDesc&, int&);
template void ppotrf_bottom_right<double>(
    const char&, const int&, double*, const DdlaDesc&, int&);
template void ppotrf_bottom_right<std::complex<float>>(
    const char&, const int&, std::complex<float>*, const DdlaDesc&, int&);
template void ppotrf_bottom_right<std::complex<double>>(
    const char&, const int&, std::complex<double>*, const DdlaDesc&, int&);

} // namespace ddla
