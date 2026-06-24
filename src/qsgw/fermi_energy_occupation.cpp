#include "fermi_energy_occupation.h"

#include <cmath>
#include <vector>
#include "../utils/constants.h"

namespace librpa_int {

// Fermi-Dirac distribution (zero-temperature step, faithful to legacy code)
double fermi_dirac(double energy, double mu, double temperature)
{
    // const double K_B = 3.16681e-6;  // Hartree/K
    // return 1.0 / (1.0 + exp((energy - mu) / (K_B * temperature)));
    if (energy <= mu) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// Total occupation at a given chemical potential
double calculate_total_occupation(const MeanField &mf, double mu, double temperature) {
    double total_occupation = 0.0;

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / (mf.get_n_kpoints() * mf.get_n_spins());
                total_occupation += occupation;
            }
        }
    }

    return total_occupation;
}
// local occupation for each ispin-ikpoint at zero temperature
static double calculate_local_occupation(const MeanField &mf, double mu, double temperature, int ispin, int ikpt) {
    double local_occupation = 0.0;
    for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
        double energy = mf.get_eigenvals()[ispin](ikpt, ib);
        double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / mf.get_n_spins();
        local_occupation += occupation;
    }
    return local_occupation;
}

// semiconductor gap
double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons) {
    double tolerance = 1e-5;
    double mu = 0.0;
    double gap = 0.0;
    double vbm = -10000.0;  // largest value below mu
    double cbm = 10000.0;  // smallest value above mu

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            double local_vbm = -10000.0;
            double local_cbm = 10000.0;
            double local_occupation = 0.0;
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double local_mu = mf.get_eigenvals()[ispin](ikpt, ib);
                local_occupation = calculate_local_occupation(mf, local_mu, temperature, ispin, ikpt);

                // update vbm,cbm;
                if (local_occupation <= total_electrons + tolerance ) {
                    local_vbm = local_mu;
                }
                if (local_occupation > total_electrons + tolerance ) {
                    if (local_mu < local_cbm) {
                        local_cbm = local_mu;
                    }
                }
            }
            if (local_vbm > vbm) {
                vbm = local_vbm;
            }
            if (local_cbm < cbm){
                cbm = local_cbm;
            }
        }
    }
    mu = (vbm + cbm) * 0.5;
    gap = cbm - vbm ;
    std::cout << "Final VBM: " << vbm * HA2EV<< ", CBM: " << cbm * HA2EV << ", Final Fermi level: " << mu * HA2EV << std::endl;
    std::cout << "Hamiltonian_gap: " << gap * HA2EV << " eV, "<< std::endl;
    return mu;
}


// local occupation for each ispin-ikpoint at zero temperature (eqp basis)
static double calculate_eqp_local_occupation(const MeanField &mf, std::map<int, std::map<int, std::map<int, double>>> e_qp_all, double mu, double temperature, int ispin, int ikpt) {
    double local_occupation = 0.0;
    for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
        double energy = e_qp_all[ispin][ikpt][ib];
        double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / mf.get_n_spins();
        local_occupation += occupation;
    }
    return local_occupation;
}

static double calculate_eqp_total_occupation(const MeanField &mf, std::map<int, std::map<int, std::map<int, double>>> e_qp_all, double mu, double temperature) {
    double total_occupation = 0.0;

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = e_qp_all[ispin][ikpt][ib];
                double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / (mf.get_n_kpoints() * mf.get_n_spins());
                total_occupation += occupation;
            }
        }
    }

    return total_occupation;
}

double calculate_eqp_fermi_energy(const MeanField &mf,
                                  std::map<int, std::map<int, std::map<int, double>>> e_qp_all,
                                  double temperature,
                                  double total_electrons) {
    double tolerance = 1e-5;
    double mu = 0.0;
    double gap = 0.0;
    double vbm = -10000.0;  // largest value below mu
    double cbm = 10000.0;  // smallest value above mu

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            double local_vbm = -10000.0;
            double local_cbm = 10000.0;
            double local_occupation = 0.0;
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double local_mu = e_qp_all[ispin][ikpt][ib];
                local_occupation = calculate_eqp_local_occupation(mf,e_qp_all ,local_mu, temperature, ispin, ikpt);
                std::cout << "local_occupation: " << local_occupation << std::endl;
                // update vbm,cbm;
                if (local_occupation <= total_electrons + tolerance ) {
                    local_vbm = local_mu;
                }
                if (local_occupation > total_electrons + tolerance) {
                    if (local_mu < local_cbm) {
                        local_cbm = local_mu;
                    }
                }
            }
            if (local_vbm > vbm) {
                vbm = local_vbm;
            }
            if (local_cbm < cbm){
                cbm = local_cbm;
            }
        }
    }

    // Fermi level taken as the midpoint of vbm and cbm
    mu = (vbm + cbm) * 0.5;
    gap = cbm - vbm ;
    std::cout << "Final eqp_VBM: " << vbm* HA2EV << ", eqp_CBM: " << cbm* HA2EV << ", Final eqp_Fermi level: " << mu* HA2EV<< std::endl;
    std::cout << "eqp_gap: " << gap * HA2EV << " eV, "<< std::endl;
    return gap;
}




void update_fermi_energy_and_occupations(MeanField &mf, const double temperature, const double efermi)
{
    double total_electrons1 = 0.0;
    // update occupations
    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt)
        {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib)
            {
                const double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                mf.get_weight()[ispin](ikpt, ib) = fermi_dirac(energy, efermi, temperature) * 2.0 / (mf.get_n_kpoints() * mf.get_n_spins());
                total_electrons1 += (mf.get_weight()[ispin](ikpt, ib)*mf.get_n_kpoints());  // accumulate total occupation
            }
        }
    }
    total_electrons1 = total_electrons1 / mf.get_n_kpoints();
    // print total electrons
    std::cout << "Total electrons: " << total_electrons1 << std::endl;
    std::cout << "efermi: " << efermi << std::endl;
    mf.get_efermi() = efermi;
}

// Replaces legacy MeanField::get_total_weight() (old meanfield.cpp:370):
// plain sum of all occupation weights over spins/kpoints/bands.
double calculate_total_weight(const MeanField &mf)
{
    double total_electrons = 0.0;
    for (int is = 0; is < mf.get_n_spins(); ++is)
    {
        for (int ik = 0; ik < mf.get_n_kpoints(); ++ik)
        {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib)
            {
                total_electrons += mf.get_weight()[is](ik, ib);
            }
        }
    }
    return total_electrons;
}

} // namespace librpa_int
