#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_ptran(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int m = round_up_for_grid(base.m, nb, handle->nprows_);
    const int n = round_up_for_grid(base.n, nb, handle->npcols_);
    ddla::DdlaDesc descA(handle), descAT(handle);
    descA.init(m, n, nb, nb, 0, 0);
    descAT.init(n, m, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [](int i, int j){ return general_value(i, j, 6); });
    std::vector<Complex> h_AT(local_size(descAT), Complex(0.0, 0.0));
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_AT(handle, h_AT.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_AT.ptr, h_AT);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    ddla::ptran(d_A.ptr, descA, d_AT.ptr, descAT, true);
    auto out = download(handle, d_AT.ptr, h_AT.size());
    const double err = local_max_error<Complex>(descAT, out, [&](int i, int j){
        return std::conj(general_value(j, i, 6));
    });
    require_close(handle, "ptran(conj)", err, 1e-12);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ptran", check_ptran);
}
