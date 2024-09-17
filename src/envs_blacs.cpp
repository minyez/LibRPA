#include "envs_blacs.h"

namespace LIBRPA
{

namespace envs
{

static bool librpa_blacs_initialized = false;

LIBRPA::BLACS_CTXT_handler blacs_ctxt_global_h;

void initialize_blacs(const MPI_Comm &mpi_comm_global_in)
{
    blacs_ctxt_global_h.reset_comm(mpi_comm_global_in);
    blacs_ctxt_global_h.init();

    librpa_blacs_initialized = true;
}

bool is_blacs_initialized()
{
    return librpa_blacs_initialized;
}

void finalize_blacs()
{
    blacs_ctxt_global_h.exit();
    librpa_blacs_initialized = false;
}

} /* end of namespace envs */

} /* end of namespace LIBRPA */
