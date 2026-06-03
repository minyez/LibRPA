#ifndef DDLA_COMM_H
#define DDLA_COMM_H

#include "ddla_connector.h"

namespace ddla{


inline int MPI_Allreduce_ddla(const float* sendbuff, float* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_FLOAT, op, comm);
}

inline int MPI_Allreduce_ddla(const std::complex<float>* sendbuff, std::complex<float>* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count * 2, MPI_FLOAT, op, comm);
}

inline int MPI_Allreduce_ddla(const double* sendbuff, double* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_DOUBLE, op, comm);
}

inline int MPI_Allreduce_ddla(const std::complex<double>* sendbuff, std::complex<double>* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_DOUBLE_COMPLEX, op, comm);
}


#ifdef DDLA_USE_CCL
template<typename T>
inline ncclResult_t cclSend(const T* sendbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclSend(sendbuff, count * sizeof(T), ncclInt8, peer, comm, stream);
}

template<typename T>
inline ncclResult_t cclRecv(T* recvbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclRecv(recvbuff, count * sizeof(T), ncclInt8, peer, comm, stream);
}

template<typename T>
inline ncclResult_t cclBroadcast(const T* sendbuff, T* recvbuff, size_t count, int root, ncclComm_t comm, deviceStream_t stream)
{
    return ncclBroadcast(sendbuff, recvbuff, count * sizeof(T), ncclInt8, root, comm, stream);
}

template<typename T>
inline ncclResult_t cclBcast(T* buff, size_t count, int root, ncclComm_t comm, deviceStream_t stream)
{
    return ncclBcast(buff, count * sizeof(T), ncclInt8, root, comm, stream);
}

inline ncclResult_t cclAllReduce(const float* sendbuff, float* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, op, comm, stream);
}

inline ncclResult_t cclAllReduce(const std::complex<float>* sendbuff, std::complex<float>* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count * 2, ncclFloat32, op, comm, stream);
}

inline ncclResult_t cclAllReduce(const double* sendbuff, double* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, ncclFloat64, op, comm, stream);
}

inline ncclResult_t cclAllReduce(const std::complex<double>* sendbuff, std::complex<double>* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count * 2, ncclFloat64, op, comm, stream);
}

#else

template<typename T>
inline int cclSend(const T* sendbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Send(sendbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm);
}

template<typename T>
inline int cclRecv(T* recvbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Recv(recvbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
}

template<typename T>
inline int cclBcast(T* buff, size_t count, int root, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Bcast(buff, count * sizeof(T), MPI_BYTE, root, comm);
}

template<typename T>
inline int cclAllReduce(const T* sendbuff, T* recvbuff, int count, cclOp op, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Allreduce_ddla(sendbuff, recvbuff, count, op, comm);
}

template<typename T>
inline int cclBroadcast(const T* sendbuff, T* recvbuff, int count, int root, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Broadcast(sendbuff, recvbuff, count * sizeof(T), MPI_BYTE, root, comm);
}
#endif

#ifdef DDLA_USE_GPU_CPU_TUNNEL
template<typename T>
inline int cclBcast(T* h_sendbuff, T* d_sendbuff, size_t count, int root, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceMemcpyAsync(h_sendbuff, d_sendbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    int value =  MPI_Bcast(h_sendbuff, count * sizeof(T), MPI_BYTE, root, comm);
    DEVICE_CHECK(deviceMemcpyAsync(d_sendbuff, h_sendbuff, count * sizeof(T), deviceMemcpyHostToDevice, stream));
    return value;
}

template<typename T>
inline int cclSend(T* h_sendbuff, const T* d_sendbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceMemcpyAsync(h_sendbuff, d_sendbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Send(h_sendbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm);
}

template<typename T>
inline int cclRecv(T* h_recvbuff, T* d_recvbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    int value = MPI_Recv(h_recvbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
    DEVICE_CHECK(deviceMemcpyAsync(d_recvbuff, h_recvbuff, count * sizeof(T), deviceMemcpyHostToDevice, stream));
    return value;
}

template<typename T>
inline int cclAllReduce(T* h_sendbuff, const T* d_sendbuff, T* h_recvbuff, T* d_recvbuff, int count, cclOp op, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceMemcpyAsync(h_sendbuff, d_sendbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceMemcpyAsync(h_recvbuff, d_recvbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    int value = MPI_Allreduce_ddla(h_sendbuff, h_recvbuff, count, op, comm);
    DEVICE_CHECK(deviceMemcpyAsync(d_recvbuff, h_recvbuff, count * sizeof(T), deviceMemcpyHostToDevice, stream));
    return value;
}
#endif


} // namespace ddla

#endif  // DDLA_COMM_H