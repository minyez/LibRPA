/*!
 * @author    Min-Ye Zhang
 * @date      2024-04-25
 */
#include "../utils/constants.h"
#include "analycont.h"

#include <cassert>
// #include <iostream>

#include "../math/complexmatrix.h"
// #include "../io/stl_io_helper.h"

namespace librpa_int
{

AnalyContPade::AnalyContPade(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : n_pars(n_pars_in)
{
    const int n_data = data.size();

    // Use all data if requested number of parameters is invalid
    if (n_pars < 0) n_pars = n_data;

    if (n_data <= n_pars)
    {
        // Use all data points
        n_pars = n_data;
        source_xs = xs;
        source_data = data;
    }
    else
    {
        // Select the data points evenly, when number of parameters are fewer than data points
        source_xs.resize(n_pars);
        source_data.resize(n_pars);
        int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            source_xs[ipar] = xs[ipar * step];
            source_data[ipar] = data[ipar * step];
        }
        source_xs[n_pars-1] = xs[n_data-1];
        source_data[n_pars-1] = data[n_data-1];
    }

    // Calculate the continuation coefficients, using Thiel's reciprocal difference method
    ComplexMatrix g(n_pars, n_pars);
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        g(i_par, 0) = source_data[i_par];
    }

    for (int i_par = 1; i_par < n_pars; i_par++)
    {
        for (int i = i_par; i < n_pars; i++)
        {
            g(i, i_par) = 
                (g(i_par-1, i_par-1) - g(i, i_par-1)) / ((source_xs[i] - source_xs[i_par-1]) * g(i, i_par-1));
        }
    }

    par_y.resize(n_pars);
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        par_y[i_par] = g(i_par, i_par);
    }
}

cplxdb
AnalyContPade::get(const cplxdb &x) const
{
    cplxdb tmp = {1.0, 0.0};

    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - source_xs[i_par-1]) / tmp;
    }
    return par_y[0] / tmp;
}

cplxdb
AnalyContPade::get_derivative(const cplxdb &x) const
{
    if (n_pars <= 1)
    {
        return {0.0, 0.0};
    }

    std::vector<cplxdb> g(n_pars);
    std::vector<cplxdb> dg(n_pars, {0.0, 0.0});

    g[n_pars-1] = par_y[n_pars-1];
    for (int i_par = n_pars - 2; i_par >= 0; i_par--)
    {
        const cplxdb denominator = 1.0 + (x - source_xs[i_par]) * g[i_par+1];
        g[i_par] = par_y[i_par] / denominator;
        dg[i_par] = -par_y[i_par]
                    * (g[i_par+1] + (x - source_xs[i_par]) * dg[i_par+1])
                    / (denominator * denominator);
    }
    return dg[0];
}

const std::vector<double> get_specfunc(const AnalyCont &ac, const std::vector<cplxdb> omegas,
                                       const double &ref, const double &e_ks, const double &v_xc,
                                       const double &v_exx,
                                       const double &sigc_omega_imag_shift,
                                       const double &gf_omega_imag_shift)
{
    const int n_freq = omegas.size();
    std::vector<double> sf(n_freq, 0.0);
    for (int i = 0; i < n_freq; i++)
    {
        const auto omega_gf = omegas[i] + cplxdb(0.0, gf_omega_imag_shift);
        const auto omega_sigc = omegas[i] + cplxdb(0.0, sigc_omega_imag_shift);
        cplxdb sf_c = omega_gf - e_ks + v_xc - v_exx - ac.get(omega_sigc - ref);
        sf[i] = - ((1.0 / PI) / sf_c).imag();
    }
    return sf;
}

}
