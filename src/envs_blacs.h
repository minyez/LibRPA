#pragma once

#include "base_blacs.h"

namespace LIBRPA
{

namespace envs
{

//! Handler of the global BLACS context
extern LIBRPA::BLACS_CTXT_handler blacs_ctxt_global_h;

//! Initialize the MPI environment of LibRPA
/*!
 * @param  [in]  mpi_comm_global_in    Global MPI communicator
 */
void initialize_blacs(const MPI_Comm &mpi_comm_global_in);

bool is_blacs_initialized();

void finalize_blacs();

} /* end of namespace envs */

} /* end of namespace LIBRPA */
