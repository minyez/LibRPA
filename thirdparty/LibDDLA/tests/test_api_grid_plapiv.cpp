#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_plapiv(const ddla::DdlaHandle_t& handle, const Shape& base)
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
    std::vector<int> ipiv(desc.m_loc());
    for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
        ipiv[iloc] = desc.indx_l2g_r(iloc) + 1;
    }
    const int row0_loc = desc.indx_g2l_r(0);
    if(row0_loc >= 0){
        ipiv[row0_loc] = m;
    }
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla::plapiv('F', 'R', 'C', m, n, d_A.ptr, desc, ipiv.data(), desc, nullptr);
    auto out = download(handle, d_A.ptr, h_A.size());
    const double err = local_max_error<Complex>(desc, out, [&](int i, int j){
        int src_i = i;
        if(i == 0) src_i = m - 1;
        else if(i == m - 1) src_i = 0;
        return base_value(src_i, j);
    });
    require_close(handle, "plapiv", err, 1e-12);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_plapiv", check_plapiv);
}
