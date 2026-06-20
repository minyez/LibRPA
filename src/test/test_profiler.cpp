#include <cassert>
#include <string>

#include "../src/utils/profiler.h"

int main(int argc, char *argv[])
{
    using namespace librpa_int;

    Profiler profiler;
    profiler.start("info_timer", LIBRPA_VERBOSE_INFO);
    profiler.stop("info_timer");
    profiler.start("debug_timer", LIBRPA_VERBOSE_DEBUG);
    profiler.stop("debug_timer");

    const auto info_profile = profiler.get_profile_string(LIBRPA_VERBOSE_INFO);
    assert(info_profile.find("info_timer") != std::string::npos);
    assert(info_profile.find("debug_timer") == std::string::npos);

    const auto debug_profile = profiler.get_profile_string(LIBRPA_VERBOSE_DEBUG);
    assert(debug_profile.find("debug_timer") != std::string::npos);

    return 0;
}
