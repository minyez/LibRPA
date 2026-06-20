/*!
 @file profiler.h
 @brief Utilities to profile the program
 */
#ifndef PROFILER_H
#define PROFILER_H
#include <cstddef>
#include <memory>
#include <vector>
#include <ctime>
#include <string>
#include <map>

#include "librpa_enums.h"

namespace librpa_int {

double cpu_time_from_clocks_diff(const std::clock_t& ct_start,
                                 const std::clock_t& ct_end);

std::string get_timestamp();

//! A simple profiler object to record timing of code snippet runs in the program.
class Profiler
{
private:
    //! Class to track timing of a particular part of code
    class Timer
    {
    public:
        //! the number of timer calls
        size_t ncalls;
        //! clock when the timer is started
        std::clock_t clock_start;
        //! wall time when the timer is started
        double wt_start;
        //! accumulated cpu time
        double cpu_time_accu;
        //! accumulated wall time, i.e. elapsed time
        double wall_time_accu;
        //! cpu time during last call
        double cpu_time_last;
        //! wall time during last call
        double wall_time_last;
        //! Timer name
        std::string name;
        //! Side note for the timer, not used as timer identification
        std::string note;
        //! Verbosity level needed to print this timer in profiling summaries
        LibrpaVerbose verbose_level;

        std::shared_ptr<Timer> parent;
        std::shared_ptr<Timer> prev;
        std::shared_ptr<Timer> next;
        // First child
        std::shared_ptr<Timer> child;

        Timer(const std::string &tname, const std::string &tnote,
              LibrpaVerbose verbose_level)
            : ncalls(0),
              clock_start(0),
              wt_start(0),
              cpu_time_accu(0),
              wall_time_accu(0),
              cpu_time_last(0),
              wall_time_last(0),
              name(tname),
              note(tnote),
              verbose_level(verbose_level),
              parent(nullptr),
              prev(nullptr),
              next(nullptr),
              child(nullptr)
        {
        }

        //! start the timer
        void start() noexcept;
        //! stop the timer and record the timing
        void stop() noexcept;
        bool is_on() const { return clock_start != 0; };
        size_t get_ncalls() const { return ncalls; };
        double get_cpu_time() const { return cpu_time_accu; };
        double get_wall_time() const { return wall_time_accu; };
        double get_cpu_time_last() const { return cpu_time_last; };
        double get_wall_time_last() const { return wall_time_last; };
    };

    std::shared_ptr<Timer> root;
    std::shared_ptr<Timer> current;

    // Find child timer with timer name
    std::shared_ptr<Timer> find_timer_in_hierarchy(const std::string &tname);
    std::shared_ptr<Timer> search_timer_in_hierarchy(std::shared_ptr<Timer> timer, const std::string& tname);

    std::string get_profile_string_of_timer(std::shared_ptr<Timer> timer, int level,
                                            LibrpaVerbose verbose_level);

    //! Add a timer as the last child of the current node
    std::shared_ptr<Timer> add(const std::string &tname, const std::string &tnote,
                               LibrpaVerbose verbose_level) noexcept;

public:
    Profiler(): root(nullptr), current(nullptr) {};
    //! Start a timer. If the timer is not added before, add it.
    void start(const std::string &tname, const std::string &tnote = "",
               LibrpaVerbose verbose = LIBRPA_VERBOSE_INFO) noexcept;
    void start(const std::string &tname, LibrpaVerbose verbose) noexcept
    {
        start(tname, "", verbose);
    }
    //! Stop a timer and record the timing
    void stop(const std::string &tname) noexcept;
    //! Get cpu time of last call of timer
    double get_cpu_time_last(const std::string &tname) noexcept;
    //! Get wall time of last call of timer
    double get_wall_time_last(const std::string &tname) noexcept;
    //! Display the current profiling result
    void display(int verbose = LIBRPA_VERBOSE_DEBUG) noexcept;
    std::string get_profile_string(int verbose = LIBRPA_VERBOSE_DEBUG) noexcept;
    //! Get the number of created timers
    int get_num_timers() noexcept;
};

// TODO: move it somewhere else, for example, to `global_utils.h`
namespace global
{

extern Profiler profiler;

}

}
#endif
