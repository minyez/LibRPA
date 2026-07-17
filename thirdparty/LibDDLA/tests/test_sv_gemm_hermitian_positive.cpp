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

/**
 * 构造 Hermitian 正定矩阵 A = B^H * B + lambda * I，
 * 用 pgesv 做 LU 分解解 A * X = I 得到 X = A^-1，
 * 再用 pgemm('C','N') 计算 A^\dagger * X 验证是否为单位矩阵。
 * 由于 A 是 Hermitian 的，A^\dagger = A，故 A^\dagger * A^-1 = I。
 */
void check_sv_gemm_hermitian_positive(int n, const DdlaHandle_t& ddla_handle)
{
    DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow() * matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n",
           myid, matrix_desc.m_loc(), matrix_desc.n_loc(),
           matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());

    const size_t nelem = static_cast<size_t>(matrix_desc.m_loc()) * matrix_desc.n_loc();
    const size_t size = nelem * sizeof(std::complex<double>);

    std::complex<double>* d_B   = nullptr; // 随机矩阵 B
    std::complex<double>* d_Bcp = nullptr; // B 的副本（构造 A = B^H * B 时作为第二个操作数）
    std::complex<double>* d_A   = nullptr; // Hermitian 正定矩阵 A
    std::complex<double>* d_Acp = nullptr; // A 的副本（pgesv 会覆盖 d_A）
    std::complex<double>* d_I   = nullptr; // 单位矩阵 I / 解 X = A^-1
    std::complex<double>* d_R   = nullptr; // 结果 A^\dagger * X

    DEVICE_CHECK(deviceMallocAsync((void**)&d_B,   size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_Bcp, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A,   size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_Acp, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_I,   size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_R,   size, ddla_handle->stream));

    // ---- 1. 生成随机矩阵 B，缩小幅度以控制 A 的条件数 ----
    random_generator(d_B, nelem, DEVICE_C_64F);
    BLAS_CHECK(deblasScal(ddla_handle->blasH, nelem, 0.01, d_B, 1));
    DEVICE_CHECK(deviceMemcpyAsync(d_Bcp, d_B, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));

    // ---- 2. 构造 Hermitian 正定矩阵 A = B^H * B（pgemm 'C','N'）----
    pgemm('C', 'N', n, n, n,
          std::complex<double>(1.0, 0.0),
          d_B, matrix_desc,
          d_Bcp, matrix_desc,
          std::complex<double>(0.0, 0.0),
          d_A, matrix_desc);

    // ---- 3. 加 lambda * I 改善条件数 ----
    std::complex<double> diag_shift(1.0, 0.0);
    for (int i = 0; i < matrix_desc.m(); i++) {
        int i_loc = matrix_desc.indx_g2l_r(i);
        if (i_loc < 0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if (j_loc < 0) continue;
        DEVICE_CHECK(deviceMemcpy(d_A + i_loc + j_loc * matrix_desc.lld(), &diag_shift,
                                  sizeof(std::complex<double>), deviceMemcpyHostToDevice));
    }

    // 保存 A 的副本（pgesv 会覆盖 d_A 为 LU 因子）
    DEVICE_CHECK(deviceMemcpyAsync(d_Acp, d_A, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));

    // ---- 4. 构造分布式单位矩阵 I ----
    std::vector<std::complex<double>> h_identity(nelem, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < matrix_desc.m(); i++) {
        int i_loc = matrix_desc.indx_g2l_r(i);
        if (i_loc < 0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if (j_loc < 0) continue;
        h_identity[i_loc + j_loc * matrix_desc.lld()] = std::complex<double>(1.0, 0.0);
    }
    DEVICE_CHECK(deviceMemcpyAsync(d_I, h_identity.data(), size, deviceMemcpyHostToDevice, ddla_handle->stream));

    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- 5. LU 分解解 A * X = I → d_I 变为 X = A^-1，d_A 变为 LU 因子 ----
    printf("myid:%d, start pgesv\n", myid);
    double start_time_sv = MPI_Wtime();
    pgesv(n, n, d_A, matrix_desc, d_I, matrix_desc);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_sv = MPI_Wtime() - start_time_sv;
    printf("myid:%d, pgesv time:%lf\n", myid, t_sv);

    // ---- 6. 计算 A^\dagger * X = A^H * A^-1（A Hermitian → A^H = A → 应为单位矩阵）----
    printf("myid:%d, start pgemm (A^dagger * X)\n", myid);
    double start_time_gemm = MPI_Wtime();
    pgemm('C', 'N', n, n, n,
          std::complex<double>(1.0, 0.0),
          d_Acp, matrix_desc,
          d_I, matrix_desc,
          std::complex<double>(0.0, 0.0),
          d_R, matrix_desc);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double t_gemm = MPI_Wtime() - start_time_gemm;
    printf("myid:%d, pgemm time:%lf\n", myid, t_gemm);

    // ---- 7. 检查结果是否为单位矩阵 ----
    std::vector<std::complex<double>> h_result(nelem);
    DEVICE_CHECK(deviceMemcpyAsync(h_result.data(), d_R, size, deviceMemcpyDeviceToHost, ddla_handle->stream));
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

    printf("myid:%d, n:%d, pgesv time:%lf, pgemm time:%lf, local max err:%.6e\n",
           myid, n, t_sv, t_gemm, local_max_err);

    if (myid == 0) {
        printf("n:%d, global max error:%.6e\n", n, global_max_err);
        if (global_max_err > 1e-8) {
            std::cerr << "FAIL: A^dagger * A^-1 != I, global max error too large: " << global_max_err << std::endl;
            std::exit(1);
        }
        printf("n:%d, PASS: A^dagger * A^-1 == I (Hermitian positive matrix)\n", n);
    }

    DEVICE_CHECK(deviceFreeAsync(d_R,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_I,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_Acp, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A,   ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_Bcp, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_B,   ddla_handle->stream));
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

    std::vector<int> sizes = {500, 1000, 5000};
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
        check_sv_gemm_hermitian_positive(n, ddla_handle);
    }

    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
