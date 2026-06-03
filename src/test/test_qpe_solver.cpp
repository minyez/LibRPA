#include "../core/qpe_solver.h"

#include <cassert>
#include <vector>

#include "testutils.h"

using namespace librpa_int;

static cplxdb single_pole_self_energy(const cplxdb &x, const cplxdb &x0,
                                      const cplxdb &strength = {-1.0, 0.0})
{
    return strength / (x - x0);
}

static cplxdb single_pole_self_energy_derivative(const cplxdb &x, const cplxdb &x0,
                                                 const cplxdb &strength = {-1.0, 0.0})
{
    return -strength / ((x - x0) * (x - x0));
}

static void initialize_single_pole_data(int n, const cplxdb &x0, std::vector<cplxdb> &xs,
                                        std::vector<cplxdb> &data,
                                        const cplxdb &strength = {-1.0, 0.0})
{
    xs.resize(n);
    data.resize(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, static_cast<double>(i+1)};
        data[i] = single_pole_self_energy(xs[i], x0, strength);
    }
}

static void initialize_linear_data(int n, const cplxdb &slope, const cplxdb &intercept,
                                   std::vector<cplxdb> &xs, std::vector<cplxdb> &data)
{
    xs.resize(n);
    data.resize(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, static_cast<double>(i+1)};
        data[i] = slope * xs[i] + intercept;
    }
}

void check_single_pole_self_energy(const bool use_adaptive_damp)
{
    constexpr int nfreq = 6;
    constexpr double e_mf = 0.5;
    constexpr double e_fermi = 0.2;
    constexpr double sigma_x = 0.3;
    constexpr double e_qp_ref = 0.75;
    const cplxdb x0 = {2.0, 2.0};
    const cplxdb strength = {-0.5, 0.0};
    const cplxdb sigc_ref = single_pole_self_energy(e_qp_ref - e_fermi, x0, strength);
    const double vxc = e_mf + sigma_x + sigc_ref.real() - e_qp_ref;

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_single_pole_data(nfreq, x0, xs, data, strength);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_self_consistent(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 1.0e-3, 1.0e-8, 200, 0.5,
        use_adaptive_damp);

    assert(info == 0);
    assert(fequal(e_qp, e_qp_ref, 1.0e-8));
    assert(std::abs(sigc - sigc_ref) < 1.0e-8);
}

void test_quasi_newton_uses_pade_derivative()
{
    constexpr int nfreq = 4;
    constexpr double e_mf = 0.5;
    constexpr double e_fermi = 0.2;
    constexpr double sigma_x = 0.1;
    constexpr double e_qp_ref = 0.8;
    const cplxdb slope = {2.0, 0.0};
    const cplxdb intercept = {0.05, 0.0};
    const cplxdb sigc_ref = slope * cplxdb{e_qp_ref - e_fermi, 0.0} + intercept;
    const double vxc = e_mf + sigma_x + sigc_ref.real() - e_qp_ref;

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_linear_data(nfreq, slope, intercept, xs, data);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_quasi_newton(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 0.0, 1.0e-10, 8, 1.0,
        false);

    assert(info == 0);
    assert(fequal(e_qp, e_qp_ref, 1.0e-10));
    assert(std::abs(sigc - sigc_ref) < 1.0e-10);
}

void test_perturbative_qp_weight()
{
    constexpr int nfreq = 6;
    constexpr double e_mf = 0.5;
    constexpr double e_fermi = 0.2;
    constexpr double vxc = 0.1;
    constexpr double sigma_x = 0.3;
    const cplxdb x0 = {2.0, 2.0};

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_single_pole_data(nfreq, x0, xs, data);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    cplxdb sigc_deriv;
    double qp_weight = 0.0;
    const int info = qpe_solver_pade_perturbative(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, sigc_deriv, qp_weight);

    const cplxdb omega = e_mf - e_fermi;
    const cplxdb sigc_ref = single_pole_self_energy(omega, x0);
    const cplxdb sigc_deriv_ref = single_pole_self_energy_derivative(omega, x0);
    const double qp_weight_ref = 1.0 / (1.0 - sigc_deriv_ref.real());
    const double e_qp_ref = e_mf + qp_weight_ref * (sigma_x + sigc_ref.real() - vxc);

    assert(info == 0);
    assert(std::abs(sigc - sigc_ref) < 1.0e-10);
    assert(std::abs(sigc_deriv - sigc_deriv_ref) < 1.0e-10);
    assert(fequal(qp_weight, qp_weight_ref, 1.0e-10));
    assert(fequal(e_qp, e_qp_ref, 1.0e-10));
}

int main(int argc, char *argv[])
{
    check_single_pole_self_energy(false);
    check_single_pole_self_energy(true);
    test_quasi_newton_uses_pade_derivative();
    test_perturbative_qp_weight();
}
