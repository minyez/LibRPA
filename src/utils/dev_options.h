/*!
 * @file      dev_options.h
 * @brief     Internal options for development and prototype in C++.
 * @date      2026-06-24
 */

#pragma once

namespace librpa_int {

struct DevOptions
{
    bool use_chi0_q_uhap_split;

    DevOptions();
};

namespace global {

extern DevOptions dev_opts;

}  // namespace global

}  // namespace librpa_int
