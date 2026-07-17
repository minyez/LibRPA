#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetrf_bpiv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [=](int i, int j){ return dominant_value(i, j, n); });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<int> d_ipiv(handle, std::max(1, descA.m_loc()));
    upload(handle, d_A.ptr, h_A);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    int info = -1;
    ddla::pgetrf_bpiv(n, n, d_A.ptr, descA, d_ipiv.ptr, info);
    if(info != 0) MPI_Abort(handle->comm, 1);
    require_close(handle, "pgetrf_bpiv(info)", 0.0, 0.0);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetrf_bpiv", check_pgetrf_bpiv);
}
