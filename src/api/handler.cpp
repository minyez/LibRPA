// Internal headers
#include "handler.h"
#include "instance_manager.h"

static LibrpaHandler* librpa_create_handler_common(MPI_Comm c_comm, const bool input_blacs_matloc_row_major)
{
    return librpa_int::api::push_back_dataset(c_comm);
}

// C APIs
LibrpaHandler* librpa_create_handler(MPI_Comm comm)
{
    return ::librpa_create_handler_common(comm, true);
}

void librpa_destroy_handler(LibrpaHandler *h)
{
    if (!h) return;
    librpa_int::api::destroy_dataset(h);
    delete h;
}

// Internal function for Fortran API
LibrpaHandler* librpa_create_handler_fortran(MPI_Fint *f_comm)
{
    MPI_Comm c_comm = MPI_Comm_f2c(*f_comm);
    return ::librpa_create_handler_common(c_comm, false);
}
