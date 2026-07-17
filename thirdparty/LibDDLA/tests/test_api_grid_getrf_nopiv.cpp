#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_getrf_nopiv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int n = std::max(6, std::min(10, base.m));
    std::vector<Complex> h_A(static_cast<size_t>(n) * n);
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            h_A[i + j * n] = dominant_value(i, j, n);
        }
    }

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<int> d_info(handle, 1);
    upload(handle, d_A.ptr, h_A);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla::getrf_nopiv(n, n, d_A.ptr, n, d_info.ptr, handle);
    auto info = download(handle, d_info.ptr, 1);
    if(info[0] != 0) MPI_Abort(handle->comm, 1);
    require_close(handle, "getrf_nopiv(local info)", 0.0, 0.0);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_getrf_nopiv", check_getrf_nopiv);
}
