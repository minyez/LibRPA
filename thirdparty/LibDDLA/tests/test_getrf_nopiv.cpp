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

using namespace ddla;

// Sum of log(|diag|) of the in-place LU factors.  For an LU decomposition
// (with or without pivoting) the absolute value of the determinant is the
// product of the absolute values of the diagonal entries of U.
template <typename T>
double log_abs_det(const std::vector<T>& a, int n, int lda)
{
    double logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += std::log(std::abs(a[i + i * lda]));
    }
    return logdet;
}

void check_getrf_nopiv(int n, const DdlaHandle_t& ddla_handle)
{
    const int lda = n;
    const size_t nelem = static_cast<size_t>(n) * n;
    const size_t bytes = nelem * sizeof(std::complex<double>);

    std::complex<double> *d_A, *d_A_ref;
    int *d_info, *d_info_ref, *d_ipiv;

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, bytes, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_ref, bytes, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_info, sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_info_ref, sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_ipiv, n * sizeof(int), ddla_handle->stream));

    // Fill with random values on device, then copy to the reference buffer.
    random_generator(d_A, nelem, DEVICE_C_64F);
    BLAS_CHECK(deblasScal(ddla_handle->blasH, n * n, 0.01, d_A, 1));
    std::complex<double> cons_i = 2.0;
    for(int i = 0; i < n; i++){
        DEVICE_CHECK(deviceMemcpyAsync(d_A + i + i * n, &cons_i, sizeof(std::complex<double>), deviceMemcpyHostToDevice, ddla_handle->stream));
    }
    DEVICE_CHECK(deviceMemcpyAsync(d_A_ref, d_A, bytes, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceMemsetAsync(d_info, 0, sizeof(int), ddla_handle->stream));
    DEVICE_CHECK(deviceMemsetAsync(d_info_ref, 0, sizeof(int), ddla_handle->stream));

    // Run the local nopiv LU factorization.
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    double t_nopiv = MPI_Wtime();
    getrf_nopiv(n, n, d_A, lda, d_info, ddla_handle);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    t_nopiv = MPI_Wtime() - t_nopiv;

    // Run cuSOLVER/hipSOLVER LU with partial pivoting as the reference.
    double t_ref = MPI_Wtime();
    SOLVER_CHECK(desolverGetrf(ddla_handle->solverH, n, n, d_A_ref, lda, d_ipiv, d_info_ref));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    t_ref = MPI_Wtime() - t_ref;

    // Copy results back to host.
    std::vector<std::complex<double>> h_A(nelem);
    std::vector<std::complex<double>> h_A_ref(nelem);
    DEVICE_CHECK(deviceMemcpyAsync(h_A.data(), d_A, bytes, deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(h_A_ref.data(), d_A_ref, bytes, deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));

    int info = 0, info_ref = 0;
    DEVICE_CHECK(deviceMemcpy(&info, d_info, sizeof(int), deviceMemcpyDeviceToHost));
    DEVICE_CHECK(deviceMemcpy(&info_ref, d_info_ref, sizeof(int), deviceMemcpyDeviceToHost));

    double logdet_nopiv = log_abs_det(h_A, n, lda);
    double logdet_ref   = log_abs_det(h_A_ref, n, lda);
    double diff = std::abs(logdet_nopiv - logdet_ref);

    std::cout << "n=" << n
              << "  getrf_nopiv time=" << t_nopiv
              << "  reference LU time=" << t_ref
              << "  info=" << info
              << "  info_ref=" << info_ref
              << "  log|det|_nopiv=" << logdet_nopiv
              << "  log|det|_ref=" << logdet_ref
              << "  diff=" << diff
              << std::endl;

    if (info != 0 || info_ref != 0) {
        std::cerr << "Error: non-zero info (nopiv=" << info << ", ref=" << info_ref << ")" << std::endl;
        std::exit(1);
    }
    if (diff > 1e-6) {
        std::cerr << "Error: log-determinant mismatch" << std::endl;
        std::exit(1);
    }

    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_ref, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info_ref, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_ipiv, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle, MPI_COMM_WORLD);
    check_getrf_nopiv(100, ddla_handle);
    std::cout << "Testing getrf_nopiv against cuSOLVER/hipSOLVER LU" << std::endl;

    for (int n = 5000; n <= 20000; n += 5000) {
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Testing matrix size: " << n << std::endl;
        check_getrf_nopiv(n, ddla_handle);
    }

    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
