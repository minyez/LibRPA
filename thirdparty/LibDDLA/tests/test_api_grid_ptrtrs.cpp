#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_ptrtrs(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base, 5);
    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [](int i, int j){ return triangular_l_value(i, j); });
    auto h_B = make_local<Complex>(descB, [&](int i, int j){
        Complex sum(0.0, 0.0);
        for(int l = 0; l <= i; ++l){
            sum += triangular_l_value(i, l) * x_value(l, j);
        }
        return sum;
    });

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    ddla::ptrtrs('L', 'L', 'N', 'N', n, nrhs, d_A.ptr, descA, d_B.ptr, descB);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "ptrtrs(L,L,N,N)", 2e-10);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ptrtrs", check_ptrtrs);
}
