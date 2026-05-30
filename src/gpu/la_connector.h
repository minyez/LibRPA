#ifndef LA_CONNECTOR_H
#define LA_CONNECTOR_H

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include "device_connector.h"
// #include "device_stream.h"
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/scal.h>
#include <ddla/axpy.h>
#endif
#include "../math/scalapack_connector.h"
#include "../mpi/base_blacs.h"

namespace librpa_int
{

namespace LaConnector
{

template <typename T>
inline void pgemm(
    const char& transa, const char& transb,
    const int & m, const int & n,const int & k,
    const T & alpha,
    const T* A, const int64_t& ia, const int64_t& ja, const ArrayDesc& array_descA,
    const T* B, const int64_t& ib, const int64_t& jb, const ArrayDesc& array_descB,
    const T & beta,
    T* C,const int64_t& ic,const int64_t& jc,const ArrayDesc& array_descC
)
{
    #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if(DeviceConnector::check_device_ptr((void*)A)){
        ddla::pgemm(
            transa, transb,
            m, n, k,
            alpha,
            A, array_descA.ddla_desc(),
            B, array_descB.ddla_desc(),
            beta,
            C, array_descC.ddla_desc()
        );
    }
    else
    #endif
    ScalapackConnector::pgemm_f(
        transa, transb,
        m, n, k,
        alpha,
        A, ia, ja, array_descA.desc,
        B, ib, jb, array_descB.desc,
        beta,
        C, ic, jc, array_descC.desc
    );

}

template <typename T1, typename T2>
inline void scal(
    const int& N,
    const T1& alpha,
    T2* X,
    const int& incX,
    const ArrayDesc &array_desc
){
    #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if(DeviceConnector::check_device_ptr((void*)X)){
        ddla::BLAS_CHECK(ddla::deblasScal(array_desc.ddla_desc().ddla_handle()->blasH, N, alpha, X, incX));
    }else
    #endif
    {
        LapackConnector::scal(N, alpha, X, incX);
    }
}

// template <typename T1, typename T2>
// inline void pscal(
//     const int& N,
//     const T1& alpha,
//     T2* X,
//     int ix,
//     int jx,
//     ArrayDesc array_desc,
//     const int incx
// ){
//     #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
//     if(DeviceConnector::check_device_ptr((void*)X)){
//         ix--;jx--;
//         BLAS_CHECK(ddla::deblasScal(array_desc.ddla_desc().ddla_handle()->blasH, N, alpha, X, incX));
//     }else
//     #endif
//     {
//         ScalapackConnector::scal(N, alpha, X, ix, jx,array_desc.desc, incX);
//     }
// }

template <typename T1, typename T2>
inline void pdam(const T1& num, T2* A, const ArrayDesc& array_desc)
{
    if(array_desc.m() != array_desc.n()){
        throw std::runtime_error("In LaConnector::pdam, only square matrix is supported!");
    }
    #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if(DeviceConnector::check_device_ptr((void*)A)){
        DeviceConnector::pdam(
            num, A, array_desc
        );
    }else
    #endif
    {
        #pragma omp parallel for
        for (int i = 0; i != array_desc.m(); i++)
        {
            const int ilo = array_desc.indx_g2l_r(i);
            if (ilo < 0) continue;
            const int jlo = array_desc.indx_g2l_c(i);
            if (jlo < 0) continue;
            A[ilo + array_desc.lld() * jlo] += num;
        }
    }
}

template <typename T>
inline void axpy(
    const int& N,
    const T& alpha,
    const T* X, const int& incX,
    T* Y, const int& incY,
    const BlacsCtxtHandler &blacs_h
){
    #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if(DeviceConnector::check_device_ptr((void*)X)){
        ddla::deblasAxpy(
            blacs_h.ddla_handle->blasH,
            N,
            alpha,
            X, incX,
            Y, incY
        );
    }else
    #endif
    {
        LapackConnector::axpy(
            N,
            alpha,
            X, incX,
            Y, incY
        );
    }
}

template <typename T>
inline void pgesv(
    const int& n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const ArrayDesc& array_descB,
    int& info
)
{
    #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if(ia!=1 || ja!=1 || ib!=1 || ib!=1){
        throw std::runtime_error("In LaConnector::pgesv, only support ia=ja=ib=jb=1 for device implementation!");
    }
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        ddla::pgesv(
            n, nrhs,
            d_A, array_descA.ddla_desc(),
            d_B, array_descB.ddla_desc()
        );
    }else
    #endif
    {
        std::vector<int> ipiv(array_descA.m_loc() + array_descA.mb());
        ScalapackConnector::pgesv_f(
            n, nrhs,
            d_A, ia, ja, array_descA.desc,
            ipiv.data(),
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("Error in ScalapackConnector::pgesv_f, info = %d\n", info);
            throw std::runtime_error("info !=0\n");
        }
    }
}

template <typename T>
inline void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const ArrayDesc& array_descB,
    int& info, // host
    bool is_head = false, int location = -1
)
{
    assert(side == 'L');
    #if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    if(ia!=1 || ja!=1 || ib!=1 || ib!=1){
        throw std::runtime_error("In LaConnector::pposv, only support ia=ja=ib=jb=1 for device implementation!");
    }
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        ddla::pposv(
            side, uplo, trans,
            n, nrhs,
            d_A, 1, 1, array_descA.ddla_desc(),
            d_B, 1, 1, array_descB.ddla_desc(),
            info,
            is_head, location
        );
    }else
    #endif
    {
        assert(trans == 'N' && size == 'L');
        ScalapackConnector::pposv_f(
            uplo, n, nrhs,
            d_A, ia, ja, array_descA.desc,
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("the matrix is not positive definite info:%d\n", info);
            throw std::runtime_error("info !=0\n");
        }
    }
}

}  // namespace LaConnector

}  // namespace librpa_int

#endif // LA_CONNECTOR_H
