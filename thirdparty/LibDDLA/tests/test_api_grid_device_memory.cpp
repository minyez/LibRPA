#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_device_memory(const ddla::DdlaHandle_t& handle, const Shape&)
{
    double host_value = 0.0;

    double* typed_sync = &host_value;
    DEVICE_CHECK(deviceMalloc(&typed_sync, 0));
    require_close(handle, "deviceMalloc(typed, zero)",
                  typed_sync == nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFree(typed_sync));

    void* raw_sync = &host_value;
    DEVICE_CHECK(deviceMalloc(&raw_sync, 0));
    require_close(handle, "deviceMalloc(void, zero)",
                  raw_sync == nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFree(raw_sync));

    double* typed_async = &host_value;
    DEVICE_CHECK(deviceMallocAsync(&typed_async, 0, handle->stream));
    require_close(handle, "deviceMallocAsync(typed, zero)",
                  typed_async == nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFreeAsync(typed_async, handle->stream));

    void* raw_async = &host_value;
    DEVICE_CHECK(deviceMallocAsync(&raw_async, 0, handle->stream));
    require_close(handle, "deviceMallocAsync(void, zero)",
                  raw_async == nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFreeAsync(raw_async, handle->stream));

    DEVICE_CHECK(deviceFree(nullptr));
    DEVICE_CHECK(deviceFreeAsync(nullptr, handle->stream));

    const deviceError_t null_sync_status = deviceMalloc(nullptr, 0);
    require_close(handle, "deviceMalloc(null output)",
                  null_sync_status != deviceSuccess ? 0.0 : 1.0, 0.0);
    (void)deviceGetLastError();

    const deviceError_t null_async_status = deviceMallocAsync(nullptr, 0, handle->stream);
    require_close(handle, "deviceMallocAsync(null output)",
                  null_async_status != deviceSuccess ? 0.0 : 1.0, 0.0);
    (void)deviceGetLastError();

    double* d_sync = nullptr;
    DEVICE_CHECK(deviceMalloc(&d_sync, sizeof(double)));
    require_close(handle, "deviceMalloc(typed, nonzero)",
                  d_sync != nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFree(d_sync));

    void* d_raw_sync = nullptr;
    DEVICE_CHECK(deviceMalloc(&d_raw_sync, sizeof(double)));
    require_close(handle, "deviceMalloc(void, nonzero)",
                  d_raw_sync != nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFree(d_raw_sync));

    double* d_async = nullptr;
    DEVICE_CHECK(deviceMallocAsync(&d_async, sizeof(double), handle->stream));
    require_close(handle, "deviceMallocAsync(typed, nonzero)",
                  d_async != nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFreeAsync(d_async, handle->stream));

    void* d_raw_async = nullptr;
    DEVICE_CHECK(deviceMallocAsync(&d_raw_async, sizeof(double), handle->stream));
    require_close(handle, "deviceMallocAsync(void, nonzero)",
                  d_raw_async != nullptr ? 0.0 : 1.0, 0.0);
    DEVICE_CHECK(deviceFreeAsync(d_raw_async, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_device_memory", check_device_memory);
}
