#include "dev_options.h"

namespace librpa_int
{

DevOptions::DevOptions() : use_chi0_q_uhap_split(false) {}

namespace global
{

DevOptions dev_opts;

}  // namespace global

}  // namespace librpa_int
