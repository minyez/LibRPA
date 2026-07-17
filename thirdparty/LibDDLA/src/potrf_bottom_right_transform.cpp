#include "potrf_bottom_right_transform.h"

#include <cassert>
#include <cstddef>
#include <complex>

#include <thrust/complex.h>

namespace ddla::detail {

namespace {

template <typename T>
struct DeviceScalar {
    using type = T;
};

template <>
struct DeviceScalar<std::complex<float>> {
    using type = thrust::complex<float>;
};

template <>
struct DeviceScalar<std::complex<double>> {
    using type = thrust::complex<double>;
};

template <typename T>
using DeviceScalarT = typename DeviceScalar<T>::type;

template <typename T>
__global__ void reverse_upper_to_lower_kernel(
    int n, const T* A, int lda, T* B, int ldb)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= n || column >= n || row < column){
        return;
    }

    const int source_row = n - 1 - row;
    const int source_column = n - 1 - column;
    B[row + static_cast<std::size_t>(column) * ldb] =
        A[source_row + static_cast<std::size_t>(source_column) * lda];
}

template <typename T>
__global__ void reverse_lower_to_upper_kernel(
    int n, const T* B, int ldb, T* A, int lda)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= n || column >= n || row > column){
        return;
    }

    const int source_row = n - 1 - row;
    const int source_column = n - 1 - column;
    A[row + static_cast<std::size_t>(column) * lda] =
        B[source_row + static_cast<std::size_t>(source_column) * ldb];
}

constexpr int kTileSize = 16;

} // namespace

template <typename T>
void reverse_upper_to_lower(
    int n, const T* d_A, int lda,
    T* d_B, int ldb, deviceStream_t stream)
{
    assert(n >= 0);
    assert(lda >= (n > 0 ? n : 1));
    assert(ldb >= (n > 0 ? n : 1));
    if(n == 0){
        return;
    }
    assert(d_A != nullptr);
    assert(d_B != nullptr);

    using DeviceT = DeviceScalarT<T>;
    static_assert(sizeof(DeviceT) == sizeof(T));
    const dim3 threads(kTileSize, kTileSize);
    const dim3 blocks(
        (n + kTileSize - 1) / kTileSize,
        (n + kTileSize - 1) / kTileSize);
    reverse_upper_to_lower_kernel<DeviceT><<<blocks, threads, 0, stream>>>(
        n, reinterpret_cast<const DeviceT*>(d_A), lda,
        reinterpret_cast<DeviceT*>(d_B), ldb);
    DEVICE_CHECK(deviceGetLastError());
}

template <typename T>
void reverse_lower_to_upper(
    int n, const T* d_B, int ldb,
    T* d_A, int lda, deviceStream_t stream)
{
    assert(n >= 0);
    assert(lda >= (n > 0 ? n : 1));
    assert(ldb >= (n > 0 ? n : 1));
    if(n == 0){
        return;
    }
    assert(d_A != nullptr);
    assert(d_B != nullptr);

    using DeviceT = DeviceScalarT<T>;
    static_assert(sizeof(DeviceT) == sizeof(T));
    const dim3 threads(kTileSize, kTileSize);
    const dim3 blocks(
        (n + kTileSize - 1) / kTileSize,
        (n + kTileSize - 1) / kTileSize);
    reverse_lower_to_upper_kernel<DeviceT><<<blocks, threads, 0, stream>>>(
        n, reinterpret_cast<const DeviceT*>(d_B), ldb,
        reinterpret_cast<DeviceT*>(d_A), lda);
    DEVICE_CHECK(deviceGetLastError());
}

template void reverse_upper_to_lower<float>(
    int, const float*, int, float*, int, deviceStream_t);
template void reverse_upper_to_lower<double>(
    int, const double*, int, double*, int, deviceStream_t);
template void reverse_upper_to_lower<std::complex<float>>(
    int, const std::complex<float>*, int,
    std::complex<float>*, int, deviceStream_t);
template void reverse_upper_to_lower<std::complex<double>>(
    int, const std::complex<double>*, int,
    std::complex<double>*, int, deviceStream_t);

template void reverse_lower_to_upper<float>(
    int, const float*, int, float*, int, deviceStream_t);
template void reverse_lower_to_upper<double>(
    int, const double*, int, double*, int, deviceStream_t);
template void reverse_lower_to_upper<std::complex<float>>(
    int, const std::complex<float>*, int,
    std::complex<float>*, int, deviceStream_t);
template void reverse_lower_to_upper<std::complex<double>>(
    int, const std::complex<double>*, int,
    std::complex<double>*, int, deviceStream_t);

} // namespace ddla::detail
