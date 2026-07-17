#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <thrust/complex.h>
#include <type_traits>
#include <cassert>

namespace ddla {

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

} // namespace detail

/**
 * @brief Device kernel: add alpha to each globally owned diagonal element.
 *
 * The global diagonal index i maps to a 2D block-cyclic local element via
 * the standard ScaLAPACK formulas.  Each thread checks ownership and updates
 * A[ilo + jlo*lld] in place.
 */
template <typename T1, typename T2>
__global__ void pdam_kernel(const T1* alpha, T2* A,
                            int n, int mb, int nb,
                            int irsrc, int icsrc,
                            int myprow, int mypcol,
                            int nprows, int npcols,
                            int lld)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    // ownership check for row i and column i
    const int row_owner = (irsrc + i / mb) % nprows;
    if (row_owner != myprow) return;

    const int col_owner = (icsrc + i / nb) % npcols;
    if (col_owner != mypcol) return;

    // local indices (ScaLAPACK block-cyclic mapping)
    const int ilo = mb * (i / (mb * nprows)) + i % mb;
    const int jlo = nb * (i / (nb * npcols)) + i % nb;

    A[ilo + jlo * lld] += *alpha;
}

template <typename T1, typename T2>
void pdam(const T1& alpha, T2* d_A, const DdlaDesc& array_descA)
{
    assert(array_descA.m() == array_descA.n());
    const int n = array_descA.m();
    if (n <= 0) return;

    DdlaHandle_t handle = array_descA.ddla_handle();
    deviceStream_t stream = handle->stream;

    using deviceT1 = typename detail::device_scalar<T1>::type;
    using deviceT2 = typename detail::device_scalar<T2>::type;

    deviceT1* d_alpha = nullptr;
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_alpha), sizeof(deviceT1), stream));
    DEVICE_CHECK(deviceMemcpyAsync(d_alpha, &alpha, sizeof(deviceT1), deviceMemcpyHostToDevice, stream));

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    deviceT2* d_A_dev = reinterpret_cast<deviceT2*>(d_A);
    pdam_kernel<deviceT1, deviceT2><<<gridSize, blockSize, 0, stream>>>(
        d_alpha, d_A_dev,
        n,
        array_descA.mb(), array_descA.nb(),
        array_descA.irsrc(), array_descA.icsrc(),
        array_descA.myprow(), array_descA.mypcol(),
        array_descA.nprows(), array_descA.npcols(),
        array_descA.lld());
    DEVICE_CHECK(deviceGetLastError());

    DEVICE_CHECK(deviceFreeAsync(d_alpha, stream));
}

// Supported type combinations match LibRPA's DeviceConnector::pdam.
template void pdam<float, float>(const float&, float*, const DdlaDesc&);
template void pdam<double, double>(const double&, double*, const DdlaDesc&);
template void pdam<float, std::complex<float>>(const float&, std::complex<float>*, const DdlaDesc&);
template void pdam<std::complex<float>, std::complex<float>>(const std::complex<float>&, std::complex<float>*, const DdlaDesc&);
template void pdam<double, std::complex<double>>(const double&, std::complex<double>*, const DdlaDesc&);
template void pdam<std::complex<double>, std::complex<double>>(const std::complex<double>&, std::complex<double>*, const DdlaDesc&);

} // namespace ddla
