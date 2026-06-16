#include "task_helper.h"

#include <fstream>
#include <iomanip>
#include <string>

void write_energy_qp(const librpa_int::MeanField &mf,
                     const std::vector<librpa_int::Vector3_Order<double>> &kfrac_output,
                     const std::vector<int> &output_to_input_kpoint,
                     const std::vector<librpa_int::matrix> &vxc, const std::vector<double> &vexx,
                     const std::vector<librpa_int::cplxdb> &sigc, const int n_kpoints_data,
                     const int i_state_low, const int n_states_calc, const double occupation_scale)
{
    const std::string sep =
        "-----------------------------------------"
        "----------------------------------------------------------";
    std::ofstream ofs("energy_qp");
    ofs << "  state     occ_num        e_gs(Ha)        e_qp(Ha)" << std::endl;
    ofs << sep << std::endl;
    for (int i_kpoint = 0; i_kpoint < static_cast<int>(kfrac_output.size()); i_kpoint++)
    {
        const int i_kpoint_input = output_to_input_kpoint.empty()
            ? i_kpoint
            : output_to_input_kpoint[static_cast<size_t>(i_kpoint)];
        const auto &k = kfrac_output[static_cast<size_t>(i_kpoint)];
        for (int i_spin = 0; i_spin < mf.get_n_spins(); i_spin++)
        {
            if (mf.get_n_spins() == 2)
            {
                ofs << std::setw(35) << "" << (i_spin == 0 ? "Spin Up" : "Spin Down")
                    << std::endl;
            }
            const size_t start_k = (i_spin * n_kpoints_data + i_kpoint_input) * n_states_calc;
            ofs << "  K_point " << std::setw(4) << i_kpoint + 1 << " : " << std::fixed
                << std::setprecision(4) << std::setw(16) << k.x << std::setw(16) << k.y
                << std::setw(16) << k.z << std::endl;
            ofs << sep << std::endl;
            for (int i = 0; i < n_states_calc; i++)
            {
                const int i_state = i + i_state_low;
                const auto occ_state =
                    mf.get_weight()[i_spin](i_kpoint_input, i_state) * occupation_scale;
                const auto eks_state = mf.get_eigenvals()[i_spin](i_kpoint_input, i_state);
                const auto eqp = eks_state - vxc[i_spin](i_kpoint_input, i_state) +
                                 vexx[start_k+i] + sigc[start_k+i].real();
                ofs << "  " << std::setw(6) << i_state + 1 << "  " << std::fixed
                    << std::setprecision(4) << std::setw(8) << occ_state << std::scientific
                    << std::uppercase << std::setprecision(10) << std::setw(20) << eks_state
                    << std::setw(20) << eqp << std::endl;
            }
            if (mf.get_n_spins() == 2 && i_spin == 0)
            {
                ofs << sep << std::endl;
            }
        }
        ofs << sep << std::endl;
        ofs << std::endl;
    }
}
