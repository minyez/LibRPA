#ifndef DDLA_HANDLE_T_H
#define DDLA_HANDLE_T_H

#include <iostream>
#include <mpi.h>

namespace ddla{
class DdlaStream;

using DdlaHandle_t = DdlaStream*;

void ddla_init(DdlaHandle_t& ddla_handle);

void ddla_set(DdlaHandle_t ddla_handle, const MPI_Comm& comm = MPI_COMM_WORLD, const char& major = 'R');

void ddla_set(DdlaHandle_t ddla_handle, const MPI_Comm& comm, const int& nprows, const int& npcols, const char& major = 'R');

void ddla_destroy(DdlaHandle_t& ddla_handle);

}


#endif // DDLA_HANLE_T_H