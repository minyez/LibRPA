#include "correlation_potential.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "../core/analycont.h"

namespace librpa_int
{
namespace qsgw
{
namespace
{

constexpr double sigc_trunc_in_scale = 1.0e8;
constexpr double ac_trunc_out_scale = 1.0e6;
constexpr double pade_threshold = 1.0e-6;

double fermi_dirac_step(const double energy, const double mu)
{
    return energy <= mu ? 1.0 : 0.0;
}

cplxdb round_complex(const cplxdb& val, const double scale)
{
    const double re = std::round(val.real() * scale) / scale;
    const double im = std::round(val.imag() * scale) / scale;
    return cplxdb(re, im);
}

cplxdb sanitize_complex(const cplxdb& val)
{
    if (!std::isfinite(val.real()) || !std::isfinite(val.imag()))
    {
        return cplxdb(0.0, 0.0);
    }
    return val;
}

} // namespace

std::vector<std::vector<cplxdb>> build_G0(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const int ispin,
    const int ikpt,
    const int n_bands)
{
    std::vector<double> eigenvals(n_bands);
    for (int n = 0; n < n_bands; ++n)
    {
        eigenvals[n] = meanfield.get_eigenvals()[ispin](ikpt, n);
    }

    std::vector<std::vector<cplxdb>> G0(
        n_bands, std::vector<cplxdb>(2 * freq_nodes.size()));

    for (int n = 0; n < n_bands; ++n)
    {
        for (size_t w = 0; w < freq_nodes.size(); ++w)
        {
            const cplxdb iw(0.0, freq_nodes[w]);
            G0[n][w] = 1.0 / (iw - eigenvals[n]);
            G0[n][w + freq_nodes.size()] = 1.0 / (-iw - eigenvals[n]);
        }
    }

    return G0;
}

Matz build_correlation_potential_spin_k_modeA(
    const SigmaRealAxisBlocks& sigc_spin_k,
    const int n_bands)
{
    Matz Vc_spin_k(n_bands, n_bands, MAJOR::COL);
    std::map<int, Matz> Re_sigma;
    std::map<int, Matz> sigma;
    const double threshold = 1e-6;

    for (int k = 0; k < n_bands; ++k)
    {
        Re_sigma[k] = Matz(n_bands, n_bands, MAJOR::COL);
        sigma[k] = Matz(n_bands, n_bands, MAJOR::COL);
        for (int i = 0; i < n_bands; ++i)
        {
            assert(static_cast<int>(sigc_spin_k.size()) > i);
            for (int j = 0; j < n_bands; ++j)
            {
                assert(static_cast<int>(sigc_spin_k[i].size()) > j);
                assert(static_cast<int>(sigc_spin_k[i][j].size()) > k);
                sigma[k](i, j) = sigc_spin_k[i][j][k];
            }
        }
        Re_sigma[k] = 0.5 * (sigma[k] + transpose(sigma[k], true));
    }

    for (int i = 0; i < n_bands; ++i)
    {
        for (int j = 0; j < n_bands; ++j)
        {
            cplxdb Vc_ij = 0.5 * (Re_sigma[i](i, j) + Re_sigma[j](i, j));
            Vc_ij = cplxdb(Vc_ij.real(), 0.0);
            if (std::abs(Vc_ij) < threshold)
            {
                Vc_ij = 0.0;
            }
            Vc_spin_k(i, j) = Vc_ij;
        }
    }

    return Vc_spin_k;
}

Matz build_correlation_potential_spin_k(
    const SigmaRealAxisBlocks& sigc_spin_k,
    const int n_bands)
{
    Matz Vc_spin_k(n_bands, n_bands, MAJOR::COL);
    std::map<int, Matz> Re_sigma;
    std::map<int, Matz> sigma;

    for (int k = 0; k < n_bands + 1; ++k)
    {
        Re_sigma[k] = Matz(n_bands, n_bands, MAJOR::COL);
        sigma[k] = Matz(n_bands, n_bands, MAJOR::COL);
        for (int i = 0; i < n_bands; ++i)
        {
            assert(static_cast<int>(sigc_spin_k.size()) > i);
            for (int j = 0; j < n_bands; ++j)
            {
                assert(static_cast<int>(sigc_spin_k[i].size()) > j);
                assert(static_cast<int>(sigc_spin_k[i][j].size()) > k);
                sigma[k](i, j) = sigc_spin_k[i][j][k];
            }
        }
        Re_sigma[k] = 0.5 * (sigma[k] + transpose(sigma[k], true));
    }

    for (int i = 0; i < n_bands; ++i)
    {
        const cplxdb Vc_ii = Re_sigma[i](i, i);
        Vc_spin_k(i, i) = Vc_ii;
        for (int j = 0; j < n_bands; ++j)
        {
            if (i != j)
            {
                const cplxdb Vc_ij = Re_sigma[n_bands](i, j);
                Vc_spin_k(i, j) = Vc_ij;
            }
        }
    }

    return Vc_spin_k;
}

SigmaRealAxisBlocks build_sigma_real_axis_blocks_qsgw(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::map<double, Matz>& sigc_spin_k,
    const int ispin,
    const int ikpt,
    const int n_bands,
    const int n_params_anacon)
{
    std::vector<cplxdb> imagfreqs;
    imagfreqs.reserve(freq_nodes.size());
    for (const auto& freq : freq_nodes)
    {
        imagfreqs.push_back(cplxdb(0.0, freq));
    }

    SigmaRealAxisBlocks sigcmat(
        n_bands, std::vector<std::vector<cplxdb>>(
                     n_bands, std::vector<cplxdb>(n_bands + 1)));
    const double efermi = meanfield.get_efermi();

    for (int i_state_row = 0; i_state_row < n_bands; ++i_state_row)
    {
        for (int i_state_col = 0; i_state_col < n_bands; ++i_state_col)
        {
            std::vector<cplxdb> sigc_mn;
            sigc_mn.reserve(freq_nodes.size());
            double max_magnitude = 0.0;
            for (const auto& freq : freq_nodes)
            {
                cplxdb val = sigc_spin_k.at(freq)(i_state_row, i_state_col);
                val = round_complex(val, sigc_trunc_in_scale);

                max_magnitude = std::max(max_magnitude, std::abs(val));
                sigc_mn.push_back(val);
            }

            double energy0 = meanfield.get_eigenvals()[ispin](ikpt, i_state_row);
            if (!std::isfinite(energy0))
            {
                energy0 = efermi;
            }

            cplxdb result(0.0, 0.0);
            cplxdb result_fermi(0.0, 0.0);
            if (max_magnitude > pade_threshold)
            {
                try
                {
                    const AnalyContPade pade(n_params_anacon, imagfreqs, sigc_mn);
                    result = pade.get(energy0 - efermi);
                    result_fermi = pade.get(0.0);
                }
                catch (...)
                {
                    if (!sigc_mn.empty())
                    {
                        result = sigc_mn[0];
                        result_fermi = sigc_mn[0];
                    }
                }
            }
            else if (!sigc_mn.empty())
            {
                result = sigc_mn[0];
                result_fermi = sigc_mn[0];
            }

            result = round_complex(sanitize_complex(result), ac_trunc_out_scale);
            result_fermi = round_complex(sanitize_complex(result_fermi), ac_trunc_out_scale);

            sigcmat[i_state_row][i_state_col][i_state_row] = result;
            sigcmat[i_state_row][i_state_col][n_bands] = result_fermi;
        }
    }

    return sigcmat;
}

Matz build_correlation_potential_spin_k_modeB_ac(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::map<double, Matz>& sigc_spin_k,
    const int ispin,
    const int ikpt,
    const int n_bands,
    const int n_params_anacon)
{
    return build_correlation_potential_spin_k(
        build_sigma_real_axis_blocks_qsgw(
            meanfield,
            freq_nodes,
            sigc_spin_k,
            ispin,
            ikpt,
            n_bands,
            n_params_anacon),
        n_bands);
}

Matz build_correlation_potential_spin_k_modeA_ac(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::map<double, Matz>& sigc_spin_k,
    const int ispin,
    const int ikpt,
    const int n_bands,
    const int n_params_anacon)
{
    return build_correlation_potential_spin_k_modeA(
        build_sigma_real_axis_blocks_qsgw(
            meanfield,
            freq_nodes,
            sigc_spin_k,
            ispin,
            ikpt,
            n_bands,
            n_params_anacon),
        n_bands);
}

Matz calculate_scRPA_exchange_correlation(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::vector<double>& freq_weights,
    const std::map<double, Matz>& sigc_spin_k,
    const SigmaRealAxisBlocks& sigc_sk_mat,
    const std::vector<std::vector<cplxdb>>& G0,
    const int ispin,
    const int ikpt,
    const int n_bands,
    const double temperature)
{
    Matz V_rpa_ks(n_bands, n_bands, MAJOR::COL);
    std::map<int, Matz> Re_sigma;
    std::map<int, Matz> sigma;
    const double mu = meanfield.get_efermi();

    for (int k = 0; k < n_bands; ++k)
    {
        Re_sigma[k] = Matz(n_bands, n_bands, MAJOR::COL);
        sigma[k] = Matz(n_bands, n_bands, MAJOR::COL);
        for (int i = 0; i < n_bands; ++i)
        {
            for (int j = 0; j < n_bands; ++j)
            {
                sigma[k](i, j) = sigc_sk_mat[i][j][k];
            }
        }
        Re_sigma[k] = 0.5 * (sigma[k] + transpose(sigma[k], true));
    }

    for (int n = 0; n < n_bands; ++n)
    {
        const double energy_n = meanfield.get_eigenvals()[ispin](ikpt, n);
        const double f_n =
            fermi_dirac_step(energy_n, mu) * 2.0 / meanfield.get_n_spins();

        for (int m = 0; m < n_bands; ++m)
        {
            cplxdb V_nm = 0.0;
            const double energy_m = meanfield.get_eigenvals()[ispin](ikpt, m);
            const double f_m =
                fermi_dirac_step(energy_m, mu) * 2.0 / meanfield.get_n_spins();
            const cplxdb Vc_nm = 0.5 * (Re_sigma[n](n, m) + Re_sigma[m](n, m));

            if ((energy_n - mu) * (energy_m - mu) < 0)
            {
                const double delta_nm = f_n - f_m;
                for (size_t w = 0; w < freq_weights.size(); ++w)
                {
                    const cplxdb sigc_nm_iw = sigc_spin_k.at(freq_nodes[w])(n, m);
                    const cplxdb sigc_nm_minus_iw = sigc_spin_k.at(-freq_nodes[w])(n, m);
                    V_nm += freq_weights[w] *
                            (sigc_nm_iw * (G0[n][w] - G0[m][w]) +
                             sigc_nm_minus_iw *
                                 (G0[n][w + freq_weights.size()] -
                                  G0[m][w + freq_weights.size()]));
                }
                V_rpa_ks(n, m) = V_nm / delta_nm;
            }
            else
            {
                V_rpa_ks(n, m) = Vc_nm;
            }
        }
    }

    return V_rpa_ks;
}

} // namespace qsgw
} // namespace librpa_int
