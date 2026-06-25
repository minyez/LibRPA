#include "hamiltonian_qsgw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../utils/constants.h"

namespace librpa_int
{
namespace qsgw
{
namespace
{

ComplexMatrix matz_to_complex_matrix(const Matz& mat)
{
    ComplexMatrix converted(mat.nr(), mat.nc());
    for (int ir = 0; ir < mat.nr(); ++ir)
    {
        for (int ic = 0; ic < mat.nc(); ++ic)
        {
            converted(ir, ic) = mat(ir, ic);
        }
    }
    return converted;
}

const WfcMap& select_wfc_source(const MeanField& meanfield, const WfcMap* fixed_wfc)
{
    return fixed_wfc == nullptr ? meanfield.get_eigenvectors() : *fixed_wfc;
}

} // namespace

double qsgw_hround_scale()
{
    const char* env_scale = std::getenv("QSGW_HROUND_SCALE");
    double scale = 1.0e10;
    if (env_scale != nullptr)
    {
        const double parsed = std::atof(env_scale);
        scale = parsed <= 0.0 ? 0.0 : parsed;
    }
    return scale;
}

void apply_qsgw_hround(Matz& mat, const int n_bands)
{
    const double scale = qsgw_hround_scale();
    if (scale <= 0.0)
    {
        return;
    }

    for (int i_row = 0; i_row < n_bands; ++i_row)
    {
        for (int i_col = 0; i_col < n_bands; ++i_col)
        {
            const cplxdb val = mat(i_row, i_col);
            const double re = std::round(val.real() * scale) / scale;
            const double im = std::round(val.imag() * scale) / scale;
            mat(i_row, i_col) = cplxdb(re, im);
        }
    }
}

SpinKMatrixMap construct_H0_GW(
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    const int n_spins,
    const int n_kpoints,
    const int n_bands)
{
    SpinKMatrixMap H0_GW_all;

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const Matz Hexx_ispin_ik = Hexx_all.at(ispin).at(ikpt);
            const Matz Vxc_construct_ispin_ik = Hexx_ispin_ik + Vc_all.at(ispin).at(ikpt);

            Matz H0_GW_spin_k =
                H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
                Vxc_construct_ispin_ik;

            apply_qsgw_hround(H0_GW_spin_k, n_bands);
            H0_GW_all[ispin][ikpt] = H0_GW_spin_k;
        }
    }

    return H0_GW_all;
}

SpinKMatrixMap construct_H0_GW_fermi_window(
    const MeanField& meanfield,
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    const int n_spins,
    const int n_kpoints,
    const int n_bands,
    const int bands_above_fermi,
    const double diag_shift_ev)
{
    SpinKMatrixMap H0_GW_all;

    const double efermi = meanfield.get_efermi();
    const double diag_shift_ha = diag_shift_ev / HA2EV;

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const Matz Hexx_ispin_ik = Hexx_all.at(ispin).at(ikpt);
            const Matz Vxc_construct_ispin_ik = Hexx_ispin_ik + Vc_all.at(ispin).at(ikpt);

            Matz H0_GW_spin_k =
                H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
                Vxc_construct_ispin_ik;
            const Matz H0_KS_spin_k = H_KS_all.at(ispin).at(ikpt);

            apply_qsgw_hround(H0_GW_spin_k, n_bands);

            int N0 = 0;
            for (int i = 0; i < n_bands; ++i)
            {
                const double ei = meanfield.get_eigenvals()[ispin](ikpt, i);
                if (ei < efermi)
                {
                    N0 += 1;
                }
            }
            const int cutoff = std::min(n_bands - 1, N0 + bands_above_fermi);

            for (int i = 0; i < n_bands; ++i)
            {
                for (int j = 0; j < n_bands; ++j)
                {
                    if ((i > cutoff) || (j > cutoff))
                    {
                        if (i == j)
                        {
                            H0_GW_spin_k(i, j) = H0_KS_spin_k(i, j) + diag_shift_ha;
                        }
                        else
                        {
                            H0_GW_spin_k(i, j) = 0.0;
                        }
                    }
                }
            }

            H0_GW_all[ispin][ikpt] = H0_GW_spin_k;
        }
    }

    return H0_GW_all;
}

SpinKMatrixMap construct_H0_GW_cut(
    const MeanField& meanfield,
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    const int n_spins,
    const int n_kpoints,
    const int n_bands)
{
    SpinKMatrixMap H0_GW_all;

    const double efermi = meanfield.get_efermi();
    const int band_alived = 8;

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const Matz Hexx_ispin_ik = Hexx_all.at(ispin).at(ikpt);
            const Matz Vxc_construct_ispin_ik = Hexx_ispin_ik + Vc_all.at(ispin).at(ikpt);
            Matz H0_GW_spin_k =
                H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
                Vxc_construct_ispin_ik;
            const Matz H0_KS_spin_k = H_KS_all.at(ispin).at(ikpt);

            int N0 = 0;
            for (int i = 0; i < n_bands; ++i)
            {
                const double energy_i = meanfield.get_eigenvals()[ispin](ikpt, i);
                if (energy_i < efermi)
                {
                    N0 += 1;
                }
            }
            for (int i = 0; i < n_bands; ++i)
            {
                for (int j = 0; j < n_bands; ++j)
                {
                    if ((i > N0 + band_alived) || (j > N0 + band_alived))
                    {
                        H0_GW_spin_k(i, j) = (i == j) ? H0_KS_spin_k(i, j) + 20.0 : 0.0;
                    }
                }
            }
            H0_GW_all[ispin][ikpt] = H0_GW_spin_k;
        }
    }

    return H0_GW_all;
}

SpinKMatrixMap construct_H0_GW_new_basis(
    const MeanField& meanfield,
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& H_DFT_nao,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    const int n_spins,
    const int n_kpoints,
    const int n_bands)
{
    SpinKMatrixMap H0_GW_all;

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const Matz Hexx_ispin_ik = Hexx_all.at(ispin).at(ikpt);
            const Matz Vxc_construct_ispin_ik = Hexx_ispin_ik + Vc_all.at(ispin).at(ikpt);
            const Matz wfc1 = collect_wfc_rows(meanfield, ispin, ikpt, nullptr);

            Matz H_DFT_spin_k = conj(wfc1) * H_DFT_nao.at(ispin).at(ikpt) * transpose(wfc1);
            Matz H0_GW_spin_k = H_DFT_spin_k + Vxc_construct_ispin_ik;

            H0_GW_spin_k = 0.5 * (H0_GW_spin_k + transpose(H0_GW_spin_k, true));
            H0_GW_all[ispin][ikpt] = H0_GW_spin_k;
        }
    }

    return H0_GW_all;
}

SpinKMatrixMap construct_H0_HF(
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const int n_spins,
    const int n_kpoints)
{
    SpinKMatrixMap H0_HF_all;
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const Matz Hexx_ispin_ik = Hexx_all.at(ispin).at(ikpt);
            const Matz H0_HF_spin_k =
                H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
                Hexx_ispin_ik;
            H0_HF_all[ispin][ikpt] = H0_HF_spin_k;
        }
    }

    return H0_HF_all;
}

Matz collect_wfc_rows(
    const MeanField& meanfield,
    const int ispin,
    const int ikpt,
    const WfcMap* fixed_wfc)
{
    const int dimension = meanfield.get_n_bands();
    const int n_spinor = meanfield.get_n_spinor();
    const int nao = meanfield.get_n_aos();
    Matz wfc_rows(dimension, nao * n_spinor, MAJOR::COL);
    const auto& source = select_wfc_source(meanfield, fixed_wfc);

    for (int ib = 0; ib < dimension; ++ib)
    {
        for (int ispinor = 0; ispinor < n_spinor; ++ispinor)
        {
            const auto& wfc_spinor = source.at(ispin).at(ispinor).at(ikpt);
            for (int iao = 0; iao < nao; ++iao)
            {
                const int col = iao * n_spinor + ispinor;
                wfc_rows(ib, col) = wfc_spinor(ib, iao);
            }
        }
    }

    return wfc_rows;
}

void store_rotated_wfc(
    MeanField& meanfield,
    const int ispin,
    const int ikpt,
    const Matz& eigvec_nao)
{
    const int dimension = meanfield.get_n_bands();
    const int n_spinor = meanfield.get_n_spinor();
    const int nao = meanfield.get_n_aos();

    for (int ib = 0; ib < dimension; ++ib)
    {
        for (int ispinor = 0; ispinor < n_spinor; ++ispinor)
        {
            auto& wfc_spinor = meanfield.get_eigenvectors()[ispin][ispinor][ikpt];
            for (int iao = 0; iao < nao; ++iao)
            {
                const int col = iao * n_spinor + ispinor;
                wfc_spinor(ib, iao) = eigvec_nao(ib, col);
            }
        }
    }
}

void rotate_velocity_to_qp_basis(
    const Matz& eigvec_ks,
    const VelocityMatrix& velocity_source,
    const int ispin,
    const int ikpt,
    VelocityMatrix& velocity_target)
{
    const ComplexMatrix rotation_right = matz_to_complex_matrix(eigvec_ks);
    const ComplexMatrix rotation_left = transpose(rotation_right, true);
    for (int ia = 0; ia != 3; ++ia)
    {
        velocity_target[ispin][ikpt][ia] =
            rotation_left * velocity_source[ispin][ikpt][ia] * rotation_right;
    }
}

void diagonalize_and_store(
    MeanField& meanfield,
    const SpinKMatrixMap& H0_GW_all,
    const int n_spins,
    const int n_kpoints,
    const int dimension,
    VelocityMatrix* velocity)
{
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const auto h = H0_GW_all.at(ispin).at(ikpt).copy();
            std::vector<double> w;
            Matz eigvec_KS;

            eigsh(h, w, eigvec_KS);

            for (int ib = 0; ib < dimension; ++ib)
            {
                meanfield.get_eigenvals()[ispin](ikpt, ib) = w[ib];
            }
            const Matz eigvec_NAO =
                transpose(eigvec_KS, true) * collect_wfc_rows(meanfield, ispin, ikpt, nullptr);
            store_rotated_wfc(meanfield, ispin, ikpt, eigvec_NAO);
            if (velocity != nullptr)
            {
                rotate_velocity_to_qp_basis(eigvec_KS, *velocity, ispin, ikpt, *velocity);
            }
        }
    }
}

#if LIBRPA_QSGW_HAS_STATE_HEADER
void diagonalize_and_store_fixed_basis(
    MeanField& meanfield,
    const SpinKMatrixMap& H0_GW_all,
    const QsgwState& state,
    const int n_spins,
    const int n_kpoints,
    const int dimension,
    VelocityMatrix* velocity)
{
    diagonalize_and_store_fixed_basis(
        meanfield,
        H0_GW_all,
        state.wfc0,
        &state.velocity0,
        n_spins,
        n_kpoints,
        dimension,
        velocity);
}
#endif

void diagonalize_and_store_fixed_basis(
    MeanField& meanfield,
    const SpinKMatrixMap& H0_GW_all,
    const WfcMap& wfc0,
    const VelocityMatrix* velocity0,
    const int n_spins,
    const int n_kpoints,
    const int dimension,
    VelocityMatrix* velocity)
{
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const auto h = H0_GW_all.at(ispin).at(ikpt).copy();
            std::vector<double> w;
            Matz eigvec_KS;

            eigsh(h, w, eigvec_KS);

            std::cout << "Eigenvalues (w):\n";
            for (int i = 0; i < dimension; ++i)
            {
                printf("%30.20e\n", w[i]);
            }
            std::cout << std::string(77, '-') << "\n";

            for (int ib = 0; ib < dimension; ++ib)
            {
                meanfield.get_eigenvals()[ispin](ikpt, ib) = w[ib];
            }
            const Matz eigvec_NAO =
                transpose(eigvec_KS, true) * collect_wfc_rows(meanfield, ispin, ikpt, &wfc0);
            store_rotated_wfc(meanfield, ispin, ikpt, eigvec_NAO);
            if (velocity != nullptr && velocity0 != nullptr)
            {
                rotate_velocity_to_qp_basis(eigvec_KS, *velocity0, ispin, ikpt, *velocity);
            }
        }
    }
}

} // namespace qsgw
} // namespace librpa_int
