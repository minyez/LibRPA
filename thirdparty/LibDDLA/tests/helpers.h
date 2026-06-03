#pragma once

#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <cal.h>

static inline int getLocalRank()
{
    int      localRank;
    MPI_Comm localComm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    return localRank;
}

static calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request)
{
    MPI_Request req;
    int         err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, (MPI_Comm)(data), &req);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    *request = (void*)(req);
    return CAL_OK;
}

static calError_t request_test(void* request)
{
    MPI_Request req = (MPI_Request)(request);
    int         completed;
    int         err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

static calError_t request_free(void* request)
{
    return CAL_OK;
}
