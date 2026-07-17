#include <cassert>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/scal.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

void check_pgetrf_nopiv(int n, const DdlaHandle_t& ddla_handle)
{
    DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow() * matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n",
           myid, matrix_desc.m_loc(), matrix_desc.n_loc(),
           matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());

    std::complex<double> *d_A, *d_A_piv, *d_A_bpiv;
    int* d_ipiv_bpiv;

    const size_t nelem = static_cast<size_t>(matrix_desc.m_loc()) * matrix_desc.n_loc();
    const size_t size = nelem * sizeof(std::complex<double>);

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_piv, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_bpiv, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_ipiv_bpiv, matrix_desc.m_loc() * sizeof(int), ddla_handle->stream));

    random_generator(d_A, nelem, DEVICE_C_64F);
    BLAS_CHECK(deblasScal(ddla_handle->blasH, nelem, 0.01, d_A, 1));
    std::complex<double> cons_i = 2.0;
    for (int i = 0; i < matrix_desc.m(); i++) {
        int i_loc = matrix_desc.indx_g2l_r(i);
        if (i_loc < 0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if (j_loc < 0) continue;
        DEVICE_CHECK(deviceMemcpy(d_A + i_loc + j_loc * matrix_desc.lld(), &cons_i,
                                  sizeof(std::complex<double>), deviceMemcpyHostToDevice));
    }

    DEVICE_CHECK(deviceMemcpyAsync(d_A_piv, d_A, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(d_A_bpiv, d_A, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> ipiv(matrix_desc.m_loc());

    // 1) pgetrf (partial pivoting)
    int info_piv = -1;
    double t_piv_start = MPI_Wtime();
    pgetrf(n, n, d_A_piv, matrix_desc, ipiv.data(), info_piv);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_piv = MPI_Wtime() - t_piv_start;
    assert(info_piv == 0);

    // 2) pgetrf_bpiv (block partial pivoting)
    int info_bpiv = -1;
    double t_bpiv_start = MPI_Wtime();
    pgetrf_bpiv(n, n, d_A_bpiv, matrix_desc, d_ipiv_bpiv, info_bpiv);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_bpiv = MPI_Wtime() - t_bpiv_start;
    assert(info_bpiv == 0);

    // 3) pgetrf_nopiv (no pivoting)
    int info_nopiv = -1;
    double t_nopiv_start = MPI_Wtime();
    pgetrf_nopiv(n, n, d_A, matrix_desc, info_nopiv);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_nopiv = MPI_Wtime() - t_nopiv_start;
    assert(info_nopiv == 0);

    std::vector<std::complex<double>> a(nelem);
    std::vector<std::complex<double>> a_piv(nelem);
    std::vector<std::complex<double>> a_bpiv(nelem);
    DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, size, deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(a_piv.data(), d_A_piv, size, deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(a_bpiv.data(), d_A_bpiv, size, deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));

    std::complex<double> ln_det_loc[3] = {0.0, 0.0, 0.0};
    std::complex<double> ln_det_all[3] = {0.0, 0.0, 0.0};
    std::complex<double> tmp;
    for (int ig = 0; ig != n; ig++) {
        int locr = matrix_desc.indx_g2l_r(ig);
        int locc = matrix_desc.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0) {
            tmp = a[locr + locc * matrix_desc.lld()];
            ln_det_loc[0] += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
            tmp = a_piv[locr + locc * matrix_desc.lld()];
            ln_det_loc[1] += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
            tmp = a_bpiv[locr + locc * matrix_desc.lld()];
            ln_det_loc[2] += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
        }
    }
    MPI_Allreduce(&ln_det_loc[0], &ln_det_all[0], 3, MPI_DOUBLE_COMPLEX, MPI_SUM, ddla_handle->comm);
    MPI_Barrier(ddla_handle->comm);

    double diff_piv = std::abs(ln_det_all[0].real() - ln_det_all[1].real())
                    + std::abs(ln_det_all[0].imag() - ln_det_all[1].imag());
    double diff_bpiv = std::abs(ln_det_all[0].real() - ln_det_all[2].real())
                     + std::abs(ln_det_all[0].imag() - ln_det_all[2].imag());

    printf("myid:%d, n:%d, pgetrf:%lf, pgetrf_bpiv:%lf, pgetrf_nopiv:%lf\n",
           myid, n, t_piv, t_bpiv, t_nopiv);
    printf("myid:%d, n:%d, ln_det_nopiv:%lf+i%lf, ln_det_pgetrf:%lf+i%lf, ln_det_bpiv:%lf+i%lf\n",
           myid, n,
           ln_det_all[0].real(), ln_det_all[0].imag(),
           ln_det_all[1].real(), ln_det_all[1].imag(),
           ln_det_all[2].real(), ln_det_all[2].imag());
    printf("myid:%d, n:%d, diff(nopiv/pgetrf)=%.6e, diff(nopiv/bpiv)=%.6e\n",
           myid, n, diff_piv, diff_bpiv);

    if (diff_piv > 1e-5 || diff_bpiv > 1e-5) {
        std::cerr << "Error: log-determinant mismatch (n=" << n << ")" << std::endl;
        std::exit(1);
    }

    DEVICE_CHECK(deviceFreeAsync(d_ipiv_bpiv, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_piv, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_bpiv, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle);
    printf("after stream init\n");

    std::vector<int> sizes = {500, 1000, 5000, 10000, 15000, 20000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    for (int n : sizes) {
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n", n);
        check_pgetrf_nopiv(n, ddla_handle);
    }

    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
