#include "task_utils.h"

#include <fstream>
#include <iomanip>
#include <string>

void driver::write_energy_qp(const librpa_int::MeanField &mf,
                             const std::vector<librpa_int::Vector3_Order<double>> &kfrac_output,
                             const std::vector<int> &output_to_input_kpoint,
                             const std::vector<librpa_int::matrix> &vxc,
                             const std::vector<double> &vexx,
                             const std::vector<librpa_int::cplxdb> &sigc,
                             const int n_kpoints_data,
                             const int i_state_low,
                             const int n_states_calc,
                             const double occupation_scale)
{
    const std::string banner(124, '-');
    std::ofstream ofs("energy_qp");
    ofs << "  state     occ_num        e_gs(Ha)        e_qp(Ha)" << std::endl;
    ofs << banner << std::endl;
    for (int i_spin = 0; i_spin < mf.get_n_spins(); i_spin++)
    {
        for (int i_kpoint = 0; i_kpoint < static_cast<int>(kfrac_output.size()); i_kpoint++)
        {
            const int i_kpoint_input = output_to_input_kpoint.empty()
                ? i_kpoint
                : output_to_input_kpoint[static_cast<size_t>(i_kpoint)];
            const auto &k = kfrac_output[static_cast<size_t>(i_kpoint)];
            const size_t start_k = (i_spin * n_kpoints_data + i_kpoint_input) * n_states_calc;
            ofs << " K_point " << i_kpoint + 1 << " :" << std::setw(10)
                << std::setprecision(7) << k.x << std::setw(10) << k.y << std::setw(10)
                << k.z << std::setw(10) << " Spin " << i_spin + 1 << std::endl;
            ofs << banner << std::endl;
            for (int i = 0; i < n_states_calc; i++)
            {
                const int i_state = i + i_state_low;
                const auto occ_state =
                    mf.get_weight()[i_spin](i_kpoint_input, i_state) * occupation_scale;
                const auto eks_state = mf.get_eigenvals()[i_spin](i_kpoint_input, i_state);
                const auto eqp = eks_state - vxc[i_spin](i_kpoint_input, i_state) +
                                 vexx[start_k+i] + sigc[start_k+i].real();
                ofs << std::setw(7) << i_state + 1 << std::setw(10) << std::setprecision(5)
                    << occ_state << std::setw(18) << std::setprecision(10) << eks_state
                    << std::setw(18) << std::setprecision(10) << eqp << std::endl;
            }
            ofs << banner << std::endl;
            ofs << std::endl;
        }
    }
}
