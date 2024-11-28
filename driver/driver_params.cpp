#include "driver_params.h"

#include "utils_io.h"

void DriverParams::print()
{
    LIBRPA::utils::lib_printf("%s = %s\n", "input_dir", input_dir.c_str());
}

DriverParams driver_params;
