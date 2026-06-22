#include <librpa_enums.h>

#include "../../src/io/global_io.h"
#include "../task.h"

void driver::task_g0w0_band()
{
    using namespace librpa_int::global;

    lib_printf_root(LIBRPA_VERBOSE_WARN,
                    "Deprecation warning: g0w0_band task is now an alias of g0w0\n");
    mpi_comm_global_h.barrier();

    task_g0w0();
}
