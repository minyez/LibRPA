#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pposv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "pposv")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [=](int i, int j){ return hpd_value(i, j, n); });
    auto h_B = build_rhs(descB, n, hpd_value, n);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    int info = -1;
    ddla::pposv('L', 'L', 'N', n, nrhs, d_A.ptr, 1, 1, descA,
                d_B.ptr, 1, 1, descB, info);
    if(info != 0) MPI_Abort(handle->comm, 1);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "pposv", 5e-9);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pposv", check_pposv);
}
