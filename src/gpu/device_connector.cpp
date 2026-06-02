#include "device_connector.h"

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <thrust/complex.h>
#include <ddla/ddla_stream.h>
#include <type_traits>

#endif

#include <cassert>

namespace librpa_int{
namespace DeviceConnector{
bool check_device_ptr(void* ptr){
#if defined(ENABLE_CUDA)
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    
    if (err != cudaSuccess) {
        std::cerr << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    return (attr.type == cudaMemoryTypeDevice);
#elif defined(ENABLE_HIP)
    hipPointerAttribute_t attr;
    hipError_t status = hipPointerGetAttributes(&attr, ptr);
    
    if (status != hipSuccess) {
        std::cerr << "Error: " << hipGetErrorString(status) << std::endl;
        return false;
    }
    return (attr.memoryType == hipMemoryTypeDevice);
#else
    return false;
#endif
}

template<typename T1, typename T2>
__global__ void pdam_kernel(const T1* num, T2* d_A, const ArrayDesc& array_desc_device, const int* g2l_r, const int* g2l_c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=array_desc_device.m())
        return;
    int ilo = g2l_r[i];
    if(ilo==-1)
        return;
    int jlo = g2l_c[i];
    if(jlo==-1)
        return;
    if(ilo<array_desc_device.m_loc() && jlo<array_desc_device.n_loc())
    {
        d_A[ilo + jlo*array_desc_device.m_loc()] += *num;
    }
}

template <typename T1, typename T2>
void pdam(const T1& num, T2* d_A, const ArrayDesc& array_desc)
{
    ddla::DdlaHandle_t ddla_handle = array_desc.ddla_desc().ddla_handle();
    ArrayDesc* d_array_desc;
    T1* d_num;
    int * d_g2l_r, * d_g2l_c;
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_g2l_r, sizeof(int)*array_desc.m(), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_g2l_c, sizeof(int)*array_desc.n(), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_g2l_r, array_desc.g2l_r().data(), sizeof(int)*array_desc.m(), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_g2l_c, array_desc.g2l_c().data(), sizeof(int)*array_desc.n(), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_array_desc, sizeof(ArrayDesc), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_array_desc, &array_desc, sizeof(ArrayDesc), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    int blockSize = 256;
    int gridSize = (array_desc.m() + blockSize - 1) / blockSize;
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_num, sizeof(double), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_num, &num, sizeof(double), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    if constexpr (std::is_same_v<T1, double> || std::is_same_v<T1, float>){
        if constexpr (std::is_same_v<T2, double> || std::is_same_v<T2, float>){
            pdam_kernel<<<gridSize, blockSize, 0, ddla_handle->stream>>>(d_num, d_A, *d_array_desc, d_g2l_r, d_g2l_c);
        }else if constexpr (std::is_same_v<T2, std::complex<float>>){
            pdam_kernel<<<gridSize, blockSize, 0, ddla_handle->stream>>>(d_num, (thrust::complex<float>*)d_A, *d_array_desc, d_g2l_r, d_g2l_c);
        }else if constexpr (std::is_same_v<T2, std::complex<double>>){
            pdam_kernel<<<gridSize, blockSize, 0, ddla_handle->stream>>>(d_num, (thrust::complex<double>*)d_A, *d_array_desc, d_g2l_r, d_g2l_c);
        }else{
            throw std::runtime_error("Unsupported type combination in pdam");
        }
    }else if constexpr (std::is_same_v<T1, std::complex<float>>){
        if constexpr (std::is_same_v<T2, std::complex<float>>){
            pdam_kernel<<<gridSize, blockSize, 0, ddla_handle->stream>>>((thrust::complex<float>*)d_num, (thrust::complex<float>*)d_A, *d_array_desc, d_g2l_r, d_g2l_c);
        }else{
            throw std::runtime_error("Unsupported type combination in pdam");
        }
    }else if constexpr (std::is_same_v<T1, std::complex<double>>){
        if constexpr (std::is_same_v<T2, std::complex<double>>){
            pdam_kernel<<<gridSize, blockSize, 0, ddla_handle->stream>>>((thrust::complex<double>*)d_num, (thrust::complex<double>*)d_A, *d_array_desc, d_g2l_r, d_g2l_c);
        }else{
            throw std::runtime_error("Unsupported type combination in pdam");
        }
    }else{
        throw std::runtime_error("Unsupported type combination in pdam");
    }
    ddla::DEVICE_CHECK(deviceFreeAsync(d_g2l_r, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_g2l_c, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_num, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_array_desc, ddla_handle->stream));
    ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(ddla_handle->stream));
}

template void pdam<float, float>(const float& num, float* d_A, const ArrayDesc& array_desc);
template void pdam<double, double>(const double& num, double* d_A, const ArrayDesc& array_desc);
template void pdam<float, std::complex<float>>(const float& num, std::complex<float>* d_A, const ArrayDesc& array_desc);
template void pdam<std::complex<float>, std::complex<float>>(const std::complex<float>& num, std::complex<float>* d_A, const ArrayDesc& array_desc);
template void pdam<double, std::complex<double>>(const double& num, std::complex<double>* d_A, const ArrayDesc& array_desc);
template void pdam<std::complex<double>, std::complex<double>>(const std::complex<double>& num, std::complex<double>* d_A, const ArrayDesc& array_desc);

} // namespace DeviceConnector

}