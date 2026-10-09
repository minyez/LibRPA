#include <cassert>
#include <stdexcept>
#include <string>

#include "../reader_sternheimer.h"
#include "../root_output.h"
#include "../../src/mpi/global_mpi.h"

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);

    bool caught = false;
    try
    {
        driver::run_root_output_collective([] {
            if (librpa_int::global::mpi_comm_global_h.is_root())
            {
                driver::SternheimerChi0V1Matrix matrix;
                matrix.iq = 1;
                matrix.ifreq = 1;
                matrix.omega = 0.5;
                matrix.weight = 1.0;
                matrix.atom_naux = {1};
                matrix.matrix = librpa_int::ComplexMatrix(1, 1);
                driver::write_sternheimer_chi0_v1_matrix_file(
                    "/this/path/does/not/exist/sternheimer.dat", matrix);
            }
        });
    }
    catch (const std::runtime_error &error)
    {
        caught = true;
        if (librpa_int::global::mpi_comm_global_h.is_root())
        {
            assert(std::string(error.what()).find("Cannot open") != std::string::npos);
        }
    }
    assert(caught);

    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
