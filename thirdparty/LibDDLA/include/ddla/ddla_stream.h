#ifndef DDLA_STREAM_H
#define DDLA_STREAM_H

#include "ddla_connector.h"
#include <mpi.h>
#include <iostream>
#include <cmath>

namespace ddla{


class DdlaStream{
private:
    static inline int getLocalDevice()
    {
        int localRank;
        MPI_Comm localComm;

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);

        int deviceCount = 0;
        DEVICE_CHECK(deviceGetDeviceCount(&deviceCount));

        return localRank % deviceCount;
    }
public:
    static int local_device;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm row_comm = MPI_COMM_NULL;
    MPI_Comm col_comm = MPI_COMM_NULL;
    int myid,nprocs;
    int myprow_,nprows_,mypcol_,npcols_;
    #ifdef DDLA_USE_CCL
    ncclComm_t nccl_comm = nullptr, nccl_row_comm = nullptr, nccl_col_comm = nullptr;
    #endif

    deviceStream_t stream = nullptr;
    deviceStream_t stream_data = nullptr; // for data transfer
    desolverHandle_t solverH = nullptr;
    deblasHandle_t blasH = nullptr;

    char major;
    #ifdef DDLA_USE_CCL
    static inline void nccl_comm_create(ncclComm_t &comm, const MPI_Comm &comm_group){
        int rank,size;
        MPI_Comm_rank(comm_group, &rank);
        MPI_Comm_size(comm_group, &size);
        ncclUniqueId id;  
        
        // Rank 0 生成唯一ID并广播给其他rank  
        if (rank == 0) {  
            ncclGetUniqueId(&id);  
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm_group);

        // 初始化NCCL通信器  
        CCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
        return;
    }
    #endif
    void set_local_device(int local_device){
        DdlaStream::local_device = local_device;
        // printf("myid:%d,local_device:%d\n",myid,local_device);
        #ifdef DDLA_USE_CUDA
        DEVICE_CHECK(cudaSetDevice(DdlaStream::local_device));
        #endif
        #ifdef DDLA_USE_HIP
        DEVICE_CHECK(hipSetDevice(DdlaStream::local_device));
        #endif
        return;
    }
    void init(MPI_Comm comm_group = MPI_COMM_WORLD, const char& major = 'R'){
        MPI_Comm_rank(comm_group, &myid);
        MPI_Comm_size(comm_group, &nprocs);
        nprows_ = std::ceil(std::sqrt(nprocs));
        while(nprocs % nprows_ != 0){
            nprows_--;
        }
        npcols_ = nprocs / nprows_;
        init(nprows_,npcols_,comm_group, major);
        return;
    }
    void init(int nprows, int npcols, MPI_Comm comm_group = MPI_COMM_WORLD, const char& major = 'R'){
        if(comm_group == MPI_COMM_WORLD){
            DEVICE_CHECK(deviceFree(NULL));
            set_local_device(DdlaStream::getLocalDevice());
        }
        this->major = major;
        this->comm = comm_group;
        MPI_Comm_rank(comm_group, &myid);
        MPI_Comm_size(comm_group, &nprocs);
        nprows_ = nprows;
        npcols_ = npcols;
        if(nprows_ * npcols_ != nprocs){
            std::cerr << "nprows * npcols != nprocs" << std::endl;
            exit(1);
        }
        if(major == 'R'){
            myprow_ = myid / npcols;
            mypcol_ = myid % npcols;
        }else if(major == 'C'){
            mypcol_ = myid / nprows;
            myprow_ = myid % nprows;
        }else{
            std::cerr << "major must be 'R' or 'C'" << std::endl;
            exit(1);
        }
        MPI_Comm_split(comm_group, myprow_, myid, &row_comm);
        MPI_Comm_split(comm_group, mypcol_, myid, &col_comm);
        #ifdef DDLA_USE_CCL
        DdlaStream::nccl_comm_create(nccl_comm, comm_group);
        DdlaStream::nccl_comm_create(nccl_row_comm, row_comm);
        DdlaStream::nccl_comm_create(nccl_col_comm, col_comm);
        #endif
        
        #ifdef DDLA_USE_CUDA
        DEVICE_CHECK(cudaStreamCreate(&stream));
        DEVICE_CHECK(cudaStreamCreate(&stream_data));
        BLAS_CHECK(cublasCreate(&blasH));
        BLAS_CHECK(cublasSetStream(blasH, stream));
        SOLVER_CHECK(cusolverDnCreate(&solverH));
        SOLVER_CHECK(cusolverDnSetStream(solverH, stream));
        #endif
        #ifdef DDLA_USE_HIP
        DEVICE_CHECK(hipStreamCreate(&stream));
        // printf("after stream create, myid = %d, nprocs = %d\n",myid,nprocs);
        DEVICE_CHECK(hipStreamCreate(&stream_data));
        // printf("after stream_data create, myid = %d, nprocs = %d\n",myid,nprocs);
        SOLVER_CHECK(hipsolverCreate(&solverH));
        // printf("after solver create, myid = %d, nprocs = %d\n",myid,nprocs);
        SOLVER_CHECK(hipsolverSetStream(solverH, stream));
        // printf("after set stream, myid = %d, nprocs = %d\n",myid,nprocs);

        BLAS_CHECK(hipblasCreate(&blasH));
        // printf("after blas create, myid = %d, nprocs = %d\n",myid,nprocs);
        BLAS_CHECK(hipblasSetStream(blasH, stream));
        // printf("after set stream, myid = %d, nprocs = %d\n",myid,nprocs);
        #endif
        return;
    }
    void clean(){
        if(stream!=nullptr){
            #ifdef DDLA_USE_CUDA
            DEVICE_CHECK(cudaStreamDestroy(stream));
            #endif
            #ifdef DDLA_USE_HIP
            DEVICE_CHECK(hipStreamDestroy(stream));
            #endif
            stream=nullptr;
        }
        if(stream_data!=nullptr){
            #ifdef DDLA_USE_CUDA
            DEVICE_CHECK(cudaStreamDestroy(stream_data));
            #endif
            #ifdef DDLA_USE_HIP
            DEVICE_CHECK(hipStreamDestroy(stream_data));
            #endif
            stream_data=nullptr;
        }
        if(solverH != nullptr){
            #ifdef DDLA_USE_CUDA
            SOLVER_CHECK(cusolverDnDestroy(solverH));
            #endif
            #ifdef DDLA_USE_HIP
            SOLVER_CHECK(hipsolverDestroy(solverH));
            #endif
            solverH = nullptr;
        }
        if(blasH != nullptr){
            #ifdef DDLA_USE_CUDA
            BLAS_CHECK(cublasDestroy(blasH));
            #endif
            #ifdef DDLA_USE_HIP
            BLAS_CHECK(hipblasDestroy(blasH));
            #endif
            blasH = nullptr;
        }
        #ifdef DDLA_USE_CCL
        if(nccl_comm != nullptr){
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }
        if(nccl_row_comm != nullptr){
            ncclCommDestroy(nccl_row_comm);
            nccl_row_comm = nullptr;
        }
        if(nccl_col_comm != nullptr){
            ncclCommDestroy(nccl_col_comm);
            nccl_col_comm = nullptr;
        }
        #endif
        if(row_comm != MPI_COMM_NULL){
            MPI_Comm_free(&row_comm);
            row_comm = MPI_COMM_NULL;
        }
        if(col_comm != MPI_COMM_NULL){
            MPI_Comm_free(&col_comm);
            col_comm = MPI_COMM_NULL;
        }
        return;
    }
    ~DdlaStream(){
        clean();
    }
    void check_memory(){
        size_t free_mem, total_mem;
        DEVICE_CHECK(deviceMemGetInfo(&free_mem, &total_mem));
        printf("myid:%d, local_device:%d, free_mem:%lf GB, total_mem:%lf GB\n", myid, local_device, free_mem/1024./1024/1024, total_mem/1024./1024/1024);
        return;
    }

    void rank_to_rc(const int& rank, int& row, int& col){
        if(major == 'R'){
            row = rank / this->npcols_;
            col = rank % this->npcols_;
        }else if(major == 'C'){
            row = rank % this->nprows_;
            col = rank / this->nprows_;
        }else{
            throw std::runtime_error("major should be 'R' or 'C'\n");
        }
    }

    int rc_to_rank(const int& row, const int& col){
        if(major == 'R'){
            return row * this->npcols_ + col;
        }else if(major == 'C'){
            return col * this->nprows_ + row;
        }else{
            throw std::runtime_error("major should be 'R' or 'C'\n");
        }
    }
};

// extern DdlaStream ddla_stream_global;
} // end of namespace DDLA


#endif // DDLA_STREAM_H