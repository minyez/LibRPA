#include "driver_utils.h"

#include <stdexcept>

std::vector<double> interpolate_dielec_func(int option, const std::vector<double> &frequencies_in,
                                            const std::vector<double> &df_in,
                                            const std::vector<double> &frequencies_target)
{
    if (option == 3 || option == 4)
    {
        throw std::logic_error(
            "analytic head/wing options are initialized through read_headwing_input and Dataset");
    }
    return librpa_int::interpolate_dielec_func(option, frequencies_in, df_in, frequencies_target);
}
