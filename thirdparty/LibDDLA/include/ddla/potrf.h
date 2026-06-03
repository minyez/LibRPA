#ifndef POTRF_H
#define POTRF_H

#include "ddla_connector.h"

namespace ddla{

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    std::complex<double> *A,
    int lda,
    int *devInfo
)
{
    std::complex<double> *Workspace;
    int Lwork;

    deviceStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnZpotrf_bufferSize(handle, uplo, n, (cuDoubleComplex*)A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverZpotrf_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    DEVICE_CHECK(deviceMallocAsync(&Workspace, Lwork*sizeof(std::complex<double>), stream));

    #if defined(DDLA_USE_CUDA)
    return cusolverDnZpotrf(handle, uplo, n, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    return hipsolverZpotrf(handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)Workspace, Lwork, devInfo);
    #endif

}

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    std::complex<float> *A,
    int lda,
    int *devInfo
)
{
    std::complex<float> *Workspace;
    int Lwork;

    deviceStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnCpotrf_bufferSize(handle, uplo, n, (cuFloatComplex*)A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverCpotrf_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    DEVICE_CHECK(deviceMallocAsync(&Workspace, Lwork*sizeof(std::complex<float>), stream));

    #if defined(DDLA_USE_CUDA)
    return cusolverDnCpotrf(handle, uplo, n, (cuFloatComplex*)A, lda, (cuFloatComplex*)Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    return hipsolverCpotrf(handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)Workspace, Lwork, devInfo);
    #endif

}

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *devInfo
)
{
    float *Workspace;
    int Lwork;

    deviceStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    DEVICE_CHECK(deviceMallocAsync(&Workspace, Lwork*sizeof(float), stream));

    #if defined(DDLA_USE_CUDA)
    return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    return hipsolverSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
    #endif

}

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *devInfo
)
{
    double *Workspace;
    int Lwork;

    deviceStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    DEVICE_CHECK(deviceMallocAsync(&Workspace, Lwork*sizeof(double), stream));

    #if defined(DDLA_USE_CUDA)
    return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    return hipsolverDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
    #endif

}

}


#endif