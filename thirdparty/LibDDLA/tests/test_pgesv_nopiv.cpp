#include <cassert>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/scal.h>

using namespace ddla;

void check_pgesv_nopiv(int n, const DdlaHandle_t& ddla_handle)
{
    DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow() * matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n",
           myid, matrix_desc.m_loc(), matrix_desc.n_loc(),
           matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());

    std::complex<double>* d_A;
    std::complex<double>* d_A_copy;
    std::complex<double>* d_identity;

    const size_t nelem = static_cast<size_t>(matrix_desc.m_loc()) * matrix_desc.n_loc();
    const size_t size = nelem * sizeof(std::complex<double>);

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_copy, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_identity, size, ddla_handle->stream));

    // Build distributed identity matrix I on host.
    std::vector<std::complex<double>> h_identity(nelem, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < matrix_desc.m(); i++) {
        int i_loc = matrix_desc.indx_g2l_r(i);
        if (i_loc < 0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if (j_loc < 0) continue;
        h_identity[i_loc + j_loc * matrix_desc.lld()] = std::complex<double>(1.0, 0.0);
    }

    // Generate distributed random matrix A.
    random_generator(d_A, nelem, DEVICE_C_64F);
    BLAS_CHECK(deblasScal(ddla_handle->blasH, nelem, 0.01, d_A, 1));
    std::complex<double> diag_shift(2.0, 0.0);
    for (int i = 0; i < matrix_desc.m(); i++) {
        int i_loc = matrix_desc.indx_g2l_r(i);
        if (i_loc < 0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if (j_loc < 0) continue;
        DEVICE_CHECK(deviceMemcpy(d_A + i_loc + j_loc * matrix_desc.lld(), &diag_shift,
                                  sizeof(std::complex<double>), deviceMemcpyHostToDevice));
    }

    DEVICE_CHECK(deviceMemcpyAsync(d_A_copy, d_A, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(d_identity, h_identity.data(), size, deviceMemcpyHostToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Solve A * X = I (overwrites d_A with LU factors, d_identity with X).
    double start_time_sv = MPI_Wtime();
    pgesv_nopiv(n, n, d_A, matrix_desc, d_identity, matrix_desc);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_sv = MPI_Wtime() - start_time_sv;

    // Compute A * X and store in d_A.
    double start_time_gemm = MPI_Wtime();
    pgemm('N', 'N', n, n, n,
          std::complex<double>(1.0, 0.0),
          d_A_copy, matrix_desc,
          d_identity, matrix_desc,
          std::complex<double>(0.0, 0.0),
          d_A, matrix_desc);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_gemm = MPI_Wtime() - start_time_gemm;

    // Check locally on each rank: result should be close to identity.
    std::vector<std::complex<double>> h_result(nelem);
    DEVICE_CHECK(deviceMemcpyAsync(h_result.data(), d_A, size, deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));

    double local_max_err = 0.0;
    for (int i = 0; i < matrix_desc.m(); i++) {
        int i_loc = matrix_desc.indx_g2l_r(i);
        if (i_loc < 0) continue;
        for (int j = 0; j < matrix_desc.n(); j++) {
            int j_loc = matrix_desc.indx_g2l_c(j);
            if (j_loc < 0) continue;
            double expected = (i == j) ? 1.0 : 0.0;
            std::complex<double> val = h_result[i_loc + j_loc * matrix_desc.lld()];
            double err = std::abs(val - std::complex<double>(expected, 0.0));
            if (err > local_max_err) local_max_err = err;
        }
    }

    double global_max_err = 0.0;
    MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, ddla_handle->comm);

    printf("myid:%d, n:%d, pgesv_nopiv time:%lf, pgemm time:%lf, local max err:%.6e\n",
           myid, n, t_sv, t_gemm, local_max_err);

    if (myid == 0) {
        printf("n:%d, global max error:%.6e\n", n, global_max_err);
        if (global_max_err > 1e-8) {
            std::cerr << "Error: global max error too large: " << global_max_err << std::endl;
            std::exit(1);
        }
    }

    DEVICE_CHECK(deviceFreeAsync(d_identity, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_copy, ddla_handle->stream));
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

    check_pgesv_nopiv(500, ddla_handle);
    std::vector<int> sizes = {1000, 5000, 10000};
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
        check_pgesv_nopiv(n, ddla_handle);
    }

    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
