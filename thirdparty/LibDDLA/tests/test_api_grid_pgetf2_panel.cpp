#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetf2_panel(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    {
        auto h_A = make_local<Complex>(descA, [](int i, int j){
            if(j == 0) return Complex(0.0, 0.0);
            return i == j ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
        });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2_panel(n, std::min(nb, n), d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2_panel(singular info)", std::abs(info - 1), 0.0);
    }

    {
        auto h_A = make_local<Complex>(descA, [=](int i, int j){ return dominant_value(i, j, n); });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2_panel(n, std::min(nb, n), d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2_panel(normal info)", std::abs(info), 0.0);
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetf2_panel", check_pgetf2_panel);
}
