#include "qpe_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
// #include <iomanip>

#include "../io/global_io.h"

namespace librpa_int
{

namespace
{

struct QpeIteration
{
    int iter;
    double e_qp;
    cplxdb sigc;
    double diff;
};

constexpr int n_qpe_iters_to_dump = 10;

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
        const double damp_fac)
{
    int info = 0;
    int n_iter = 0;
    bool converged = false;

    // initial guess of e_qp as input mean-field energy
    double e_qp_last = e_mf, e_qp_this;
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

    // std::cout << "QPE: " << e_mf << " " << e_fermi << " " << vxc << " " << sigma_x << "\n";
    while (n_iter++ < n_iter_max)
    {
        e_qp_this = e_qp_last + damp_fac * diff;
        sigc_this = pade.get(static_cast<cplxdb>(e_qp_this - e_fermi));
        diff = e_mf - vxc + sigma_x + sigc_this.real() - e_qp_this;
        e_qp_last = e_mf - vxc + sigma_x + sigc_this.real();
        sigc_last = sigc_this;
        last_iters[n_iters_recorded % n_qpe_iters_to_dump] =
                QpeIteration{n_iter, e_qp_last, sigc_last, diff};
        ++n_iters_recorded;
        if (std::abs(diff) < thres)
        {
            converged = true;
            // std::cout << "Iteration " << n_iter << ": "
            //         << "e_qp = " << e_qp_last << ", "
            //         << "sigc = " << sigc_last << ", "
            //         << "diff = " << diff << std::endl;
            break;
        }
    }
    // std::cout << "Finished QPE solve: " << n_iter << " " << std::scientific << diff << " " << std::abs(diff) << "\n";
    sigc = sigc_last;
    e_qp = e_qp_last;
    // std::cout << "Compare QPE solver: " << std::setw(16) << std::setprecision(8) << e_mf - vxc + sigma_x + sigc.real() << " " << e_qp << std::endl;

    if (!converged)
    {
        info = 1;
        auto &ofs_myid = global::ofs_myid;
        ofs_myid << "QPE solver did not converge after " << n_iter_max
                 << " iterations. Last " << std::min(n_iters_recorded, n_qpe_iters_to_dump)
                 << " iterations:" << std::endl;
        ofs_myid << "iter e_qp sigc_re sigc_im diff abs_diff" << std::endl;
        const int i_start = n_iters_recorded > n_qpe_iters_to_dump
                                    ? n_iters_recorded - n_qpe_iters_to_dump
                                    : 0;
        for (int i = i_start; i < n_iters_recorded; ++i)
        {
            const auto &iter = last_iters[i % n_qpe_iters_to_dump];
            ofs_myid << iter.iter << " "
                     << iter.e_qp << " "
                     << iter.sigc.real() << " "
                     << iter.sigc.imag() << " "
                     << iter.diff << " "
                     << std::abs(iter.diff) << std::endl;
        }
    }

    return info;
}

} /* end of namespace librpa_int */
