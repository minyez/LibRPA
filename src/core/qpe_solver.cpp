#include "qpe_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
// #include <iomanip>

#include "../io/global_io.h"

namespace librpa_int
{

namespace
{

struct QpeIteration
{
    int iter;
    double e_trial;
    double e_qp;
    cplxdb sigc;
    double diff;
    double damp;
};

constexpr int n_qpe_iters_to_dump = 10;
constexpr int n_qpe_max_backtrack = 12;
constexpr double qpe_damp_min = 1.0e-4;
constexpr double qpe_damp_max = 1.0;
constexpr double qpe_damp_shrink = 0.5;
constexpr double qpe_damp_grow = 1.25;

double qpe_damp_upper_bound(const double damp_fac)
{
    if (!std::isfinite(damp_fac) || damp_fac <= 0.0)
    {
        return 0.0;
    }
    return std::clamp(damp_fac, qpe_damp_min, qpe_damp_max);
}

} /* end of anonymous namespace */

int qpe_solver_pade_self_consistent(
        const AnalyContPade &pade,
        const double &e_mf,
        const double &e_fermi,
        const double &vxc,
        const double &sigma_x,
        double &e_qp,
        cplxdb &sigc,
        const double diff_init,
        const double thres,
        const int n_iter_max,
        const double damp_fac,
        const bool use_adaptive_damp)
{
    int info = 0;
    int n_iter = 0;
    bool converged = false;

    // initial guess of e_qp as input mean-field energy
    const double e0 = e_mf - vxc + sigma_x;
    double e_qp_last = e_mf;
    double e_trial_this;
    double e_qp_this = e_mf;
    cplxdb sigc_this, sigc_last;
    std::array<QpeIteration, n_qpe_iters_to_dump> last_iters;
    int n_iters_recorded = 0;

    if (n_iter_max <= 0)
    {
        e_qp = e_qp_last;
        sigc = cplxdb{0.0, 0.0};
        return 1;
    }

    double diff = diff_init;
    const double damp_max = use_adaptive_damp ? qpe_damp_upper_bound(damp_fac) : damp_fac;
    const double damp_min = use_adaptive_damp ? std::min(qpe_damp_min, damp_max) : damp_max;
    double damp = damp_max;
    double abs_diff_last = std::numeric_limits<double>::infinity();
    double diff_last = std::numeric_limits<double>::quiet_NaN();

    // std::cout << "QPE: " << e_mf << " " << e_fermi << " " << vxc << " " << sigma_x << "\n";
    while (n_iter++ < n_iter_max)
    {
        double damp_this = damp;
        double abs_diff_this = std::numeric_limits<double>::infinity();

        const int n_backtrack = use_adaptive_damp ? n_qpe_max_backtrack : 0;
        for (int i_backtrack = 0; i_backtrack <= n_backtrack; ++i_backtrack)
        {
            e_trial_this = e_qp_last + damp_this * diff;
            sigc_this = pade.get(static_cast<cplxdb>(e_trial_this - e_fermi));
            e_qp_this = e0 + sigc_this.real();
            abs_diff_this = std::abs(e_qp_this - e_trial_this);

            const bool residual_worse =
                use_adaptive_damp &&
                std::isfinite(abs_diff_last) &&
                (!std::isfinite(abs_diff_this) || abs_diff_this > abs_diff_last);
            if (!residual_worse || damp_this <= qpe_damp_min || i_backtrack == n_qpe_max_backtrack)
            {
                break;
            }
            damp_this = std::max(qpe_damp_min, qpe_damp_shrink * damp_this);
        }

        diff = e_qp_this - e_trial_this;
        damp = damp_this;
        e_qp_last = e_qp_this;
        sigc_last = sigc_this;
        last_iters[n_iters_recorded % n_qpe_iters_to_dump] =
                QpeIteration{n_iter, e_trial_this, e_qp_this, sigc_last, diff, damp};
        ++n_iters_recorded;
        if (abs_diff_this < thres)
        {
            converged = true;
            // std::cout << "Iteration " << n_iter << ": "
            //         << "e_qp = " << e_qp_this << ", "
            //         << "sigc = " << sigc_last << ", "
            //         << "diff = " << diff << std::endl;
            break;
        }
        if (!std::isfinite(abs_diff_this))
        {
            break;
        }
        const bool sign_changed = std::isfinite(diff_last) && diff * diff_last < 0.0;
        if (use_adaptive_damp && sign_changed)
        {
            damp = std::max(damp_min, qpe_damp_shrink * damp);
        }
        else if (use_adaptive_damp && abs_diff_this < abs_diff_last)
        {
            damp = std::min(damp_max, std::max(damp_min, qpe_damp_grow * damp));
        }
        diff_last = diff;
        abs_diff_last = abs_diff_this;
    }
    // std::cout << "Finished QPE solve: " << n_iter << " " << std::scientific << diff << " " << std::abs(diff) << "\n";
    sigc = sigc_last;
    e_qp = e_qp_this;
    // std::cout << "Compare QPE solver: " << std::setw(16) << std::setprecision(8) << e_mf - vxc + sigma_x + sigc.real() << " " << e_qp << std::endl;

    if (!converged)
    {
        info = 1;
        auto &ofs_myid = global::ofs_myid;
        ofs_myid << "QPE solver did not converge after " << n_iter_max
                 << " iterations. Last " << std::min(n_iters_recorded, n_qpe_iters_to_dump)
                 << " iterations:" << std::endl;
        ofs_myid << "iter e_trial e_qp sigc_re sigc_im diff abs_diff damp" << std::endl;
        const int i_start = n_iters_recorded > n_qpe_iters_to_dump
                                    ? n_iters_recorded - n_qpe_iters_to_dump
                                    : 0;
        for (int i = i_start; i < n_iters_recorded; ++i)
        {
            const auto &iter = last_iters[i % n_qpe_iters_to_dump];
            ofs_myid << iter.iter << " " << iter.e_trial << " " << iter.e_qp << " "
                     << iter.sigc.real() << " " << iter.sigc.imag() << " " << iter.diff << " "
                     << std::abs(iter.diff) << " " << iter.damp << std::endl;
        }
    }

    return info;
}

} /* end of namespace librpa_int */
