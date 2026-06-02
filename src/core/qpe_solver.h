#pragma once

#include "analycont.h"

namespace librpa_int
{

/*!
 * \brief Solve quasi-particle equation self-consistently by computing self-energy on real frequency through Pade analytic continuation.
 *
 * \param [in]     pade        AnalyContPade object, constructed from correlation self-energy at imaginary frequency
 * \param [in]     e_mf        Energy of the state of mean-field calculation
 * \param [in]     e_fermi     Fermi energy
 * \param [in]     vxc         Exchange-correlation potential in the mean-field calculation
 * \param [in]     sigma_x     Exchange self-energy
 * \param [out]    e_qp        Quasi-particle energy as the solution of QPE
 * \param [out]    sigc        Correlation self-energy of the quasi-particle
 * \param [in]     diff_init   Initial residual used for the first damped QPE update
 * \param [in]     thres       Convergence threshold for the QPE residual, in Hartree
 * \param [in]     n_iter_max  Maximum number of self-consistent QPE iterations; must be positive
 * \param [in]     damp_fac    Damping factor for QPE updates; used as the initial and maximum factor when adaptive damping is enabled
 * \param [in]     use_adaptive_damp
 *                             Adapt the damping factor during the solve
 * \retval         info        0 if QPE is solved successfully, non-zero otherwise
 */
int qpe_solver_pade_self_consistent(
        const AnalyContPade &pade,
        const double &e_mf,
        const double &e_fermi,
        const double &vxc,
        const double &sigma_x,
        double &e_qp,
        cplxdb &sigc,
        const double diff_init = 1.0e-3,
        const double thres = 1.0e-5,
        const int n_iter_max = 200,
        const double damp_fac = 0.1,
        const bool use_adaptive_damp = false
        );

}
