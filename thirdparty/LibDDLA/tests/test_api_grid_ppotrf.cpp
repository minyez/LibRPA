#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_ppotrf(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrf")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    {
        ddla::DdlaDesc tiny_desc(handle);
        tiny_desc.init(1, 1, nb, nb, 0, 0);
        auto h_tiny = make_local<Complex>(tiny_desc, [](int, int){ return Complex(4.0, 0.0); });
        DeviceBuffer<Complex> d_tiny(handle, h_tiny.size());
        upload(handle, d_tiny.ptr, h_tiny);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        int tiny_info = -1;
        const bool tiny_is_nega = ddla::ppotrf('L', 1, d_tiny.ptr, 1, 1,
                                               tiny_desc, tiny_info);
        const double tiny_status_err = std::abs(tiny_info) + (tiny_is_nega ? 1.0 : 0.0);
        require_close(handle, "ppotrf(tiny status)", tiny_status_err, 0.0);
        auto tiny_out = download(handle, d_tiny.ptr, h_tiny.size());
        const double tiny_value_err = local_max_error<Complex>(
            tiny_desc, tiny_out, [](int, int){ return Complex(2.0, 0.0); });
        require_close(handle, "ppotrf(tiny factor)", tiny_value_err, 1e-12);
    }

    {
        auto h_not_pd = make_local<Complex>(descA, [](int i, int j){
            if(i != j) return Complex(0.0, 0.0);
            return Complex(i == 0 ? -1.0 : 2.0, 0.0);
        });
        DeviceBuffer<Complex> d_not_pd(handle, h_not_pd.size());
        upload(handle, d_not_pd.ptr, h_not_pd);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        int not_pd_info = -1;
        const bool not_pd_is_nega = ddla::ppotrf('L', n, d_not_pd.ptr, 1, 1,
                                                 descA, not_pd_info);
        const double not_pd_err = std::abs(not_pd_info - 1)
                                + (not_pd_is_nega ? 1.0 : 0.0);
        require_close(handle, "ppotrf(non-PD cleanup)", not_pd_err, 0.0);
    }

    auto h_A = make_local<Complex>(descA, [=](int i, int j){ return hpd_value(i, j, n); });
    auto h_B = build_rhs(descB, n, hpd_value, n);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    int info = -1;
    const bool is_nega = ddla::ppotrf('L', n, d_A.ptr, 1, 1, descA, info);
    if(info != 0 || is_nega) MPI_Abort(handle->comm, 1);
    ddla::ppotrs('L', 'L', 'N', n, nrhs, d_A.ptr, descA, d_B.ptr, descB, is_nega, -1);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "ppotrf", 5e-9);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ppotrf", check_ppotrf);
}
