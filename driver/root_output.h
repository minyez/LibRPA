#pragma once

#include <stdexcept>
#include <string>

#include "../src/mpi/global_mpi.h"

namespace driver
{

template <typename Function>
void run_root_output_collective(Function &&function)
{
    using librpa_int::global::mpi_comm_global_h;

    std::string root_error;
    if (mpi_comm_global_h.is_root())
    {
        try
        {
            function();
        }
        catch (const std::exception &error)
        {
            root_error = error.what();
        }
        catch (...)
        {
            root_error = "unknown root output exception";
        }
    }

    const int local_failed = root_error.empty() ? 0 : 1;
    int any_failed = 0;
    mpi_comm_global_h.allreduce(&local_failed, &any_failed, 1, MPI_MAX);
    if (any_failed != 0)
    {
        throw std::runtime_error(root_error.empty() ? "root output failed" : root_error);
    }
}

}  // namespace driver
