#include "potrf_bottom_right_internal.h"
#include "potrf_bottom_right_transform.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <type_traits>

#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/herk.h>
#include <ddla/potrf.h>
#include <ddla/syrk.h>
#include <ddla/trsm.h>

namespace ddla{

namespace detail {

namespace {

constexpr int kBlockSize = 128;

template <typename T>
void update_leading(
    deblasHandle_t blas_handle,
    const char uplo,
    int leading_n, int block_n,
    const T* d_panel, int lda,
    T* d_leading)
{
    const bool upper = uplo == 'U';
    const deblasFillMode_t fill_mode = upper
        ? DEBLAS_FILL_MODE_UPPER
        : DEBLAS_FILL_MODE_LOWER;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
        const T minus_one = T(-1);
        const T one = T(1);
        BLAS_CHECK(deblasSyrk(
            blas_handle, fill_mode,
            upper ? DEBLAS_OP_N : DEBLAS_OP_T,
            leading_n, block_n,
            minus_one, d_panel, lda,
            one, d_leading, lda));
    }else{
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        BLAS_CHECK(deblasHerk(
            blas_handle, fill_mode,
            upper ? DEBLAS_OP_N : DEBLAS_OP_C,
            leading_n, block_n,
            minus_one, d_panel, lda,
            one, d_leading, lda));
    }
}

} // namespace

template <typename T>
void potrf_bottom_right_block(
    const char& uplo, const int& n, T* d_A, const int& lda,
    const int& global_offset,
    T* d_work, int* d_info, int& info,
    const DdlaHandle_t& handle)
{
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(global_offset >= 0);
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    assert(d_A != nullptr);
    assert(d_work != nullptr);
    assert(d_info != nullptr);
    assert(lda >= n);

    const deviceStream_t stream = handle->stream;
    DEVICE_CHECK(deviceMemsetAsync(d_info, 0, sizeof(int), stream));
    const deblasFillMode_t solver_uplo = uplo == 'U'
        ? DEBLAS_FILL_MODE_LOWER
        : DEBLAS_FILL_MODE_UPPER;
    if(uplo == 'U'){
        reverse_upper_to_lower(n, d_A, lda, d_work, n, stream);
    }else{
        reverse_lower_to_upper(n, d_A, lda, d_work, n, stream);
    }
    SOLVER_CHECK(desolverPotrf(
        handle->solverH, solver_uplo,
        n, d_work, n, d_info));

    int solver_info = 0;
    DEVICE_CHECK(deviceMemcpyAsync(
        &solver_info, d_info, sizeof(int),
        deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    if(solver_info != 0){
        info = global_offset + n - solver_info + 1;
        return;
    }

    if(uplo == 'U'){
        reverse_lower_to_upper(n, d_work, n, d_A, lda, stream);
    }else{
        reverse_upper_to_lower(n, d_work, n, d_A, lda, stream);
    }
}

template void potrf_bottom_right_block<float>(
    const char&, const int&, float*, const int&, const int&, float*, int*, int&,
    const DdlaHandle_t&);
template void potrf_bottom_right_block<double>(
    const char&, const int&, double*, const int&, const int&, double*, int*, int&,
    const DdlaHandle_t&);
template void potrf_bottom_right_block<std::complex<float>>(
    const char&, const int&, std::complex<float>*, const int&, const int&,
    std::complex<float>*, int*, int&, const DdlaHandle_t&);
template void potrf_bottom_right_block<std::complex<double>>(
    const char&, const int&, std::complex<double>*, const int&, const int&,
    std::complex<double>*, int*, int&, const DdlaHandle_t&);

} // namespace detail

template <typename T>
void potrf_bottom_right(
    const char& uplo, const int& n, T* d_A, const int& lda,
    int& info, const DdlaHandle_t& handle)
{
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    assert(d_A != nullptr);
    assert(lda >= n);

    const deviceStream_t stream = handle->stream;
    const deblasHandle_t blas_handle = handle->blasH;
    T* d_work = nullptr;
    int* d_info = nullptr;
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_work),
        static_cast<std::size_t>(detail::kBlockSize)
            * detail::kBlockSize * sizeof(T),
        stream));
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_info), sizeof(int), stream));

    const int last_block_start =
        ((n - 1) / detail::kBlockSize) * detail::kBlockSize;
    for(int block_start = last_block_start;
        block_start >= 0;
        block_start -= detail::kBlockSize){
        const int block_n = std::min(detail::kBlockSize, n - block_start);
        T* const d_diag = d_A + block_start
                        + static_cast<std::size_t>(block_start) * lda;
        detail::potrf_bottom_right_block(
            uplo, block_n, d_diag, lda, block_start,
            d_work, d_info, info, handle);
        if(info != 0){
            break;
        }
        if(block_start == 0){
            continue;
        }

        T* const d_panel = uplo == 'U'
            ? d_A + static_cast<std::size_t>(block_start) * lda
            : d_A + block_start;
        const T one = T(1);
        if(uplo == 'U'){
            BLAS_CHECK(deblasTrsm(
                blas_handle,
                DEBLAS_SIDE_RIGHT,
                DEBLAS_FILL_MODE_UPPER,
                DEBLAS_OP_C,
                DEBLAS_DIAG_NON_UNIT,
                block_start, block_n,
                one,
                d_diag, lda,
                d_panel, lda));
        }else{
            const deblasOperation_t trans =
                std::is_same_v<T, float> || std::is_same_v<T, double>
                ? DEBLAS_OP_T
                : DEBLAS_OP_C;
            BLAS_CHECK(deblasTrsm(
                blas_handle,
                DEBLAS_SIDE_LEFT,
                DEBLAS_FILL_MODE_LOWER,
                trans,
                DEBLAS_DIAG_NON_UNIT,
                block_n, block_start,
                one,
                d_diag, lda,
                d_panel, lda));
        }
        detail::update_leading(
            blas_handle,
            uplo,
            block_start, block_n,
            d_panel, lda,
            d_A);
    }

    DEVICE_CHECK(deviceFreeAsync(d_work, stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
}

template void potrf_bottom_right<float>(
    const char&, const int&, float*, const int&, int&, const DdlaHandle_t&);
template void potrf_bottom_right<double>(
    const char&, const int&, double*, const int&, int&, const DdlaHandle_t&);
template void potrf_bottom_right<std::complex<float>>(
    const char&, const int&, std::complex<float>*, const int&, int&,
    const DdlaHandle_t&);
template void potrf_bottom_right<std::complex<double>>(
    const char&, const int&, std::complex<double>*, const int&, int&,
    const DdlaHandle_t&);

} // namespace ddla
