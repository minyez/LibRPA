#pragma once
#include "../core/meanfield.h"
#include <map>

namespace librpa_int {

// NOTE: faithful port of legacy qsgw/fermi_energy_occupation.h (7a7ff17f).
// The new MeanField no longer has get_total_weight() (LEADER_AUDIT hard-fact #4),
// so a calculate_total_weight() helper is provided for the driver to replace the
// old meanfield.get_total_weight() call (task_qsgw.cpp:944) — it is a plain
// internal accumulation over get_weight(), matching old meanfield.cpp:370.

double calculate_total_occupation(const MeanField &mf, double mu, double temperature);
double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons);
double calculate_eqp_fermi_energy(const MeanField &mf,
                                  std::map<int, std::map<int, std::map<int, double>>> e_qp_all,
                                  double temperature,
                                  double total_electrons);
double fermi_dirac(double energy, double mu, double temperature);
void update_fermi_energy_and_occupations(MeanField &meanfield, const double temperature, const double efermi);

//! Sum of all occupation weights (replaces legacy MeanField::get_total_weight).
double calculate_total_weight(const MeanField &mf);

} // namespace librpa_int
