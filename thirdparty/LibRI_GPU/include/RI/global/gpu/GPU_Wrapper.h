// ===================
//  Author: Laiyuan Yang
//  date: 2026.2.1
// ===================

#pragma once

#ifdef __CUDA_RI
// cuda
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h> // 查询 GPU 显存
#elif defined(__HIP_RI)
// hip
#include <hip/hip_runtime.h>
#endif

// magma
#include "magma_v2.h"
#include "magma_operators.h"

#define TESTING_CHECK(err)                                                                          \
    do                                                                                              \
    {                                                                                               \
        magma_int_t err_ = (err);                                                                   \
        if (err_ != 0)                                                                              \
        {                                                                                           \
            fprintf(stderr, "Error raised by TESTING_CHECK: %s\nfailed at %s:%d: error %lld: %s\n", \
                    #err, __FILE__, __LINE__,                                                       \
                    (long long)err_, magma_strerror(err_));                                         \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

namespace RI
{

namespace GPU_Wrapper
{

#ifdef __CUDA_RI
inline void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
}
#elif defined(__HIP_RI)
inline void initDevice(int devNum)
{
    int dev = devNum;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("Using device %d: %s\n", dev, deviceProp.name);
    hipSetDevice(dev);
}
#endif

// 查询 gpu 显存使用情况
#ifdef __CUDA_RI
inline double getGPUMemory(bool print_info = true)
{
    size_t free_byte;
    size_t total_byte;

    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status)
    {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;

    if (print_info)
        // std::cout << "Used GPU memory " <<std::precision(1)<< used_db_1 << " MB\n";
        printf("Used GPU memory: %.1f MiB / %.1f MiB\n", used_db_1, total_db / 1024.0 / 1024.0);

    return used_db_1;
}
#elif defined(__HIP_RI)
inline double getGPUMemory(bool print_info = true)
{
    size_t free_byte;
    size_t total_byte;

    hipError_t hip_status = hipMemGetInfo(&free_byte, &total_byte);

    if (hipSuccess != hip_status)
    {
        printf("Error: hipMemGetInfo fails, %s \n", hipGetErrorString(hip_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;

    if (print_info)
        // std::cout << "Used GPU memory " <<std::precision(1)<< used_db_1 << " MB\n";
        printf("Used GPU memory: %.1f MiB / %.1f MiB\n", used_db_1, total_db / 1024.0 / 1024.0);

    return used_db_1;
}
#endif

#ifdef __CUDA_RI
inline cudaError_t GPUMemset(
	void * devPtr,
	int value,
	size_t count)
{
	return cudaMemset(devPtr, value, count);
}
#elif defined(__HIP_RI)
inline hipError_t GPUMemset(
	void *devPtr,
	int value,
	size_t count)
{
	return hipMemset(devPtr, value, count);
}
#endif

}

}