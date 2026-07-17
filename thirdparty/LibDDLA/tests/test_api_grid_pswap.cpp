#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pswap(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int m = round_up_for_grid(base.m, nb, handle->nprows_);
    const int n = round_up_for_grid(base.n, nb, handle->npcols_);
    ddla::DdlaDesc desc(handle);
    desc.init(m, n, nb, nb, 0, 0);
    auto base_value = [](int i, int j){ return Complex(10.0 * i + j, -0.5 * i + 0.25 * j); };

    auto h_A = make_local<Complex>(desc, base_value);
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla::pswap(n, d_A.ptr, 1, 1, desc, desc.m(), d_A.ptr, m, 1, desc, desc.m());
    auto row_out = download(handle, d_A.ptr, h_A.size());
    const double row_err = local_max_error<Complex>(desc, row_out, [&](int i, int j){
        int src_i = i;
        if(i == 0) src_i = m - 1;
        else if(i == m - 1) src_i = 0;
        return base_value(src_i, j);
    });
    require_close(handle, "pswap(row)", row_err, 1e-12);

    upload(handle, d_A.ptr, h_A);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla::pswap(m, d_A.ptr, 1, 1, desc, 1, d_A.ptr, 1, n, desc, 1);
    auto col_out = download(handle, d_A.ptr, h_A.size());
    const double col_err = local_max_error<Complex>(desc, col_out, [&](int i, int j){
        int src_j = j;
        if(j == 0) src_j = n - 1;
        else if(j == n - 1) src_j = 0;
        return base_value(i, src_j);
    });
    require_close(handle, "pswap(col)", col_err, 1e-12);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pswap", check_pswap);
}
