#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/getrf.h>
#include <ddla/scal.h>

#include <iomanip>
#include <magma_v2.h>

using namespace ddla;

template <typename T>
double log_abs_det(const std::vector<T>& a, int n, int lda)
{
    double logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += std::log(std::abs(a[i + i * lda]));
    }
    return logdet;
}

void benchmark_getrf_nopiv(int n, const DdlaHandle_t& ddla_handle)
{
    const int lda = n;
    const size_t nelem = static_cast<size_t>(n) * n;
    const size_t bytes = nelem * sizeof(std::complex<double>);

    std::complex<double> *d_A_lib, *d_A_magma, *d_A_ref;
    int *d_info_lib, *d_info_ref, *d_ipiv;

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_lib,   bytes, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_magma, bytes, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_ref,   bytes, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_info_lib,   sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_info_ref,   sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_ipiv,       n * sizeof(int), ddla_handle->stream));

    // Generate a non-singular random matrix.
    random_generator(d_A_lib, nelem, DEVICE_C_64F);
    std::complex<double> scale(0.01, 0.0);
    BLAS_CHECK(deblasScal(ddla_handle->blasH, n * n, scale, d_A_lib, 1));
    std::complex<double> diag_shift(2.0, 0.0);
    for (int i = 0; i < n; ++i) {
        DEVICE_CHECK(deviceMemcpyAsync(d_A_lib + i + i * n, &diag_shift,
                                       sizeof(std::complex<double>),
                                       deviceMemcpyHostToDevice,
                                       ddla_handle->stream));
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));

    // Copy to MAGMA and reference buffers.
    DEVICE_CHECK(deviceMemcpyAsync(d_A_magma, d_A_lib, bytes,
                                   deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(d_A_ref, d_A_lib, bytes,
                                   deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceMemsetAsync(d_info_lib,   0, sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceMemsetAsync(d_info_ref,   0, sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));

    // 1) LibDDLA getrf_nopiv
    double t_lib = MPI_Wtime();
    getrf_nopiv(n, n, d_A_lib, lda, d_info_lib, ddla_handle);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    t_lib = MPI_Wtime() - t_lib;

    // 2) MAGMA zgetrf_nopiv_gpu
    double t_magma = MPI_Wtime();
    int info_magma = 0;
    magma_zgetrf_nopiv_gpu(n, n,
                           reinterpret_cast<magmaDoubleComplex_ptr>(d_A_magma),
                           lda, &info_magma);
    DEVICE_CHECK(deviceDeviceSynchronize());
    t_magma = MPI_Wtime() - t_magma;

    // 3) cuSOLVER/hipSOLVER LU with partial pivoting (reference)
    double t_ref = MPI_Wtime();
    SOLVER_CHECK(desolverGetrf(ddla_handle->solverH, n, n, d_A_ref, lda,
                               d_ipiv, d_info_ref));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    t_ref = MPI_Wtime() - t_ref;

    // Copy results back to host.
    std::vector<std::complex<double>> h_A_lib(nelem);
    std::vector<std::complex<double>> h_A_magma(nelem);
    std::vector<std::complex<double>> h_A_ref(nelem);
    DEVICE_CHECK(deviceMemcpyAsync(h_A_lib.data(),   d_A_lib,   bytes,
                                   deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(h_A_magma.data(), d_A_magma, bytes,
                                   deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(h_A_ref.data(),   d_A_ref,   bytes,
                                   deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));

    int info_lib = 0, info_ref = 0;
    DEVICE_CHECK(deviceMemcpy(&info_lib,   d_info_lib,   sizeof(int), deviceMemcpyDeviceToHost));
    DEVICE_CHECK(deviceMemcpy(&info_ref,   d_info_ref,   sizeof(int), deviceMemcpyDeviceToHost));

    double logdet_lib   = log_abs_det(h_A_lib,   n, lda);
    double logdet_magma = log_abs_det(h_A_magma, n, lda);
    double logdet_ref   = log_abs_det(h_A_ref,   n, lda);

    double diff_lib_vs_ref   = std::abs(logdet_lib - logdet_ref);
    double diff_magma_vs_ref = std::abs(logdet_magma - logdet_ref);
    double diff_lib_vs_magma = std::abs(logdet_lib - logdet_magma);

    std::cout << "n=" << n
              << "  LibDDLA_time=" << std::setw(10) << t_lib
              << "  MAGMA_time=" << std::setw(10) << t_magma
              << "  RefLU_time=" << std::setw(10) << t_ref
              << "  info(lib/magma/ref)=" << info_lib << "/" << info_magma << "/" << info_ref
              << "  logdet(L/M/R)=" << logdet_lib << "/" << logdet_magma << "/" << logdet_ref
              << "  diff(L/R)=" << diff_lib_vs_ref
              << "  diff(M/R)=" << diff_magma_vs_ref
              << "  diff(L/M)=" << diff_lib_vs_magma
              << std::endl;

    if (info_lib != 0 || info_magma != 0 || info_ref != 0) {
        std::cerr << "Warning: non-zero info (lib=" << info_lib
                  << ", magma=" << info_magma
                  << ", ref=" << info_ref << ")" << std::endl;
    }

    DEVICE_CHECK(deviceFreeAsync(d_A_lib,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_magma, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_ref,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info_lib,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info_ref,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_ipiv,       ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle, MPI_COMM_WORLD);

    magma_init();

    std::cout << "Benchmark: LibDDLA getrf_nopiv vs MAGMA zgetrf_nopiv_gpu vs cuSOLVER/hipSOLVER LU"
              << std::endl;

    std::vector<int> sizes = {100, 500, 1000, 5000, 10000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    for (int n : sizes) {
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Benchmarking matrix size: " << n << std::endl;
        benchmark_getrf_nopiv(n, ddla_handle);
    }

    magma_finalize();
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
