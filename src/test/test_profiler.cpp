#include <cassert>
#include <chrono>
#include <string>
#include <thread>

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

    profiler.start("outer_timer");
    profiler.start("inner_timer");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    profiler.terminate();
    assert(profiler.get_wall_time_last("inner_timer") > 0.0);
    assert(profiler.get_wall_time_last("outer_timer") > 0.0);

    return 0;
}
