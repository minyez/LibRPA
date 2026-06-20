#pragma once
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

#include "../mpi/global_mpi.h"
#include "librpa_enums.h"

namespace librpa_int
{

namespace global
{

//! File output stream handler of each process
extern std::ofstream ofs_myid;

//! File stream used by fprintf when stdout is redirected
extern FILE *pfile_redirect;

//! Current stdout verbosity. Plain lib_printf messages are informational.
extern LibrpaVerbose output_level;

inline void set_output_level(const LibrpaVerbose level) noexcept { output_level = level; }

inline LibrpaVerbose get_output_level() noexcept { return output_level; }

inline bool should_output(const LibrpaVerbose verbose_level = LIBRPA_VERBOSE_INFO) noexcept
{
    return output_level >= verbose_level;
}

//! Initialize the IO environment of LibRPA
/*!
 * @param  [in]  redirect_stdout    Flag to control to redirect stdout to file, useful when separating LibRPA output from the main program
 * @pram   [in]  redirect_path      Path of file to redirect stdout, used only when redirect_stdout is true.
 * @pram   [in]  enable_task_output Flag to enable output by each MPI tasks.
 *                                  The output file name is hard-coded for each process.
 */
void init_global_io(bool redirect_stdout = false,
                    const char *redirect_path = "LibRPA_output.txt",
                    bool enable_task_output = true);

//! Check the IO environment of LibRPA is correctly initialized
bool is_io_initialized();

//! Finalize the IO environment of LibRPA
void finalize_global_io();

//! printf that handles the stdout redirect
template <typename... Args>
void lib_printf(const LibrpaVerbose verbose_level, const char* s, Args&&... args) noexcept
{
    if (!should_output(verbose_level)) return;
    if (pfile_redirect != nullptr)
    {
        std::fprintf(pfile_redirect, s, std::forward<Args>(args)...);
        std::fflush(pfile_redirect);
    }
    else
    {
        std::printf(s, std::forward<Args>(args)...);
    }
}

template <typename... Args>
void lib_printf(const char* s, Args&&... args) noexcept
{
    lib_printf(LIBRPA_VERBOSE_INFO, s, std::forward<Args>(args)...);
}

// raw string, no formatting at all
inline void lib_printf(const LibrpaVerbose verbose_level, const char* s) noexcept
{
    if (!should_output(verbose_level)) return;
    if (pfile_redirect)
    {
        std::fprintf(pfile_redirect, "%s", s);
        std::fflush(pfile_redirect);
    }
    else
    {
        std::printf("%s", s);
    }
}

inline void lib_printf(const char* s) noexcept
{
    lib_printf(LIBRPA_VERBOSE_INFO, s);
}

//! simlar to global::printf, but only proc 0 of global communicator will dump
template <typename... Args>
void lib_printf_root(const LibrpaVerbose verbose_level, const char* s, Args&&... args) noexcept
{
    if (myid_global == 0)
    {
        lib_printf(verbose_level, s, std::forward<Args>(args)...);
    }
}

template <typename... Args>
void lib_printf_root(const char* s, Args&&... args) noexcept
{
    lib_printf_root(LIBRPA_VERBOSE_INFO, s, std::forward<Args>(args)...);
}

namespace detail
{

void lib_printf_coll_msg_impl(const librpa_int::MpiCommHandler &comm_h,
                              const LibrpaVerbose verbose_level,
                              const char *local_msg,
                              const int count) noexcept;

template <typename... Args>
void lib_printf_coll_impl(const librpa_int::MpiCommHandler &comm_h,
                          const LibrpaVerbose verbose_level,
                          const char* s, Args&&... args) noexcept
{
    int count = 0;
    std::vector<char> local_msg;
    if (should_output(verbose_level))
    {
        const int msg_len = std::snprintf(nullptr, 0, s, args...);
        if (msg_len > 0)
        {
            count = msg_len;
            local_msg.resize(count + 1);
            std::snprintf(local_msg.data(), local_msg.size(), s, args...);
        }
    }
    lib_printf_coll_msg_impl(comm_h, verbose_level, count > 0 ? local_msg.data() : nullptr, count);
}

} /* end of namespace detail */

//! Similar to global::printf, but messages from all processes are printed in myid order.
template <typename... Args>
void lib_printf_coll(const LibrpaVerbose verbose_level, const char* s, Args&&... args) noexcept
{
    detail::lib_printf_coll_impl(mpi_comm_global_h, verbose_level, s, std::forward<Args>(args)...);
}

template <typename... Args>
void lib_printf_coll(const char* s, Args&&... args) noexcept
{
    lib_printf_coll(LIBRPA_VERBOSE_INFO, s, std::forward<Args>(args)...);
}

} /* end of namespace global */

inline bool verbalize(const LibrpaVerbose current_level, const LibrpaVerbose verbose_level) noexcept
{
    return current_level >= verbose_level;
}

//! Similar to lib_printf_root, but one can specify any communicator
template <typename... Args>
void printf_comm_root(const librpa_int::MpiCommHandler &comm_h, const char* s, Args&&... args) noexcept
{
    comm_h.check_initialized();
    if (0 == comm_h.myid)
    {
        global::lib_printf(s, std::forward<Args>(args)...);
    }
}

//! Similar to lib_printf_coll, but one can specify any communicator
template <typename... Args>
void printf_comm_coll(const librpa_int::MpiCommHandler &comm_h, const char* s, Args&&... args) noexcept
{
    comm_h.check_initialized();
    global::detail::lib_printf_coll_impl(comm_h, LIBRPA_VERBOSE_INFO, s, std::forward<Args>(args)...);
}

} /* end of namespace librpa_int */
