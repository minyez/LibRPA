/*!
 * @file hartree.cpp
 * @brief Hartree (classical Coulomb) kernel for QSGW — the H3 carrier.
 *
 * See hartree.h for the migration rationale. This file reconstructs the legacy
 * LIBRPA::Hartree class (7a7ff17f:src/hartree.cpp, 618 lines) on the new
 * LibRPA architecture.
 *
 * Implementation status (line refs are to legacy hartree.cpp @ 7a7ff17f):
 *   - DONE, aligned to legacy: constructor, get_dmat_cplx_k_global (:88-100,
 *     n_soc -> n_spinor), extract_dmat_cplx_IJblock (:102-117, via the new
 *     AtomicBasis::get_global_indices), reset_rspace/reset_kspace, the
 *     build_KS_kgrid0/kgrid/band wavefunction wiring (QsgwState::wfc0 anchor
 *     for H2, MeanField::get_eigenvectors for the current wfc, kfrac from
 *     pbc.kfrac_list), and the build_KS() NAO->KS projection (mirror legacy
 *     :292-410 on the new ScaLAPACK API, no exchange minus sign).
 *   - PENDING leader decision (H3): build() real-space kernel. The legacy core
 *     hartree_libri.lri.cal_cvcd_k_hartree (:243) is a localized-RI multi-center
 *     Coulomb contraction; the bundled LibRI has dropped RI::Hartree. See
 *     h3_hartree_build_plan.md (paths C / A). Per red line #5, the kernel body
 *     is NOT written until the leader rules on C vs A.
 *   - PENDING: nearest_R (use the existing find_nearest_bvk_cell, only needed
 *     if the build() kernel path returns k-space and requires a manual BvK
 *     Fourier as the legacy code did).
 */
#include "hartree.h"

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <valarray>
#include <vector>

#include "../math/lapack_connector.h"          // ScalapackConnector
#include "../math/utils_matrix_m_mpi.h"        // init_local_mat, collect_block_from_IJ_storage_tensor_transform, get_local_mat
#include "../utils/constants.h"                // TWO_PI
#include "../core/utils_atomic_basis_blacs.h"  // get_necessary_IJ_from_block_2D
#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h> // comm_map2_first, comm_map
#include <RI/global/Blas_Interface-Tensor.h>           // Blas_Interface::gemv
#include <RI/global/Global_Func-2.h>                   // Global_Func::convert
#else
// ri.h (pulled in by hartree.h) includes libri_stub.h, which stubs comm_map2_first.
#endif

namespace
{
//! Element-wise complex conjugate of an RI::Tensor<complex>.
//! The current libRI Tensor has no conjugate() method (verified), so this
//! mirrors the legacy Tensor::conjugate used by the patched cal_cvcd_k_hartree.
inline RI::Tensor<std::complex<double>> conj_tensor(const RI::Tensor<std::complex<double>> &t)
{
    auto out = std::make_shared<std::valarray<std::complex<double>>>(t.shape.get_shape_all());
    for (std::size_t i = 0; i < t.shape.get_shape_all(); ++i)
        (*out)[i] = std::conj((*t.data)[i]);
    return RI::Tensor<std::complex<double>>(t.shape, out);
}

//! M[a] = sum_{n1,n2} C[a,n1,n2] * D[n1,n2]   (legacy Tensor_Multiply::gemv).
//! Implemented as reshape(C)->{a, n1*n2}, flatten(D)->{n1*n2}, Blas_Interface::gemv('N').
//! PENDING coremath confirmation of the contraction axis direction.
inline RI::Tensor<std::complex<double>> contract_3_2_to_1(
    const RI::Tensor<std::complex<double>> &C, const RI::Tensor<std::complex<double>> &D)
{
    assert(C.shape.size() == 3 && D.shape.size() == 2);
    assert(C.shape[1] == D.shape[0] && C.shape[2] == D.shape[1]);
    const std::size_t a = C.shape[0];
    const std::size_t n12 = C.shape[1] * C.shape[2];
    const RI::Tensor<std::complex<double>> C2({a, n12}, C.data);
    const RI::Tensor<std::complex<double>> D1({n12}, D.data);
#ifdef LIBRPA_USE_LIBRI
    return RI::Blas_Interface::gemv('N', std::complex<double>(1.0), C2, D1);
#else
    RI::Tensor<std::complex<double>> M({a});
    for (std::size_t ia = 0; ia < a; ++ia)
    {
        std::complex<double> s(0.0, 0.0);
        for (std::size_t j = 0; j < n12; ++j)
            s += (*C2.data)[ia * n12 + j] * (*D1.data)[j];
        (*M.data)[ia] = s;
    }
    return M;
#endif
}

//! H[n1,n2] = sum_a C[a,n1,n2] * N[a]   (legacy Tensor_Multiply::gemv_trans).
//! Implemented as reshape(C)->{a, n1*n2}, Blas_Interface::gemv('T'), reshape back.
//! PENDING coremath confirmation of the contraction axis direction.
inline RI::Tensor<std::complex<double>> contract_3_1_to_2(
    const RI::Tensor<std::complex<double>> &C, const RI::Tensor<std::complex<double>> &N)
{
    assert(C.shape.size() == 3 && N.shape.size() == 1);
    assert(C.shape[0] == N.shape[0]);
    const std::size_t a = C.shape[0];
    const std::size_t n1 = C.shape[1], n2 = C.shape[2], n12 = n1 * n2;
    const RI::Tensor<std::complex<double>> C2({a, n12}, C.data);
    RI::Tensor<std::complex<double>> H({n1, n2},
        std::make_shared<std::valarray<std::complex<double>>>(n12));
#ifdef LIBRPA_USE_LIBRI
    const RI::Tensor<std::complex<double>> H1 =
        RI::Blas_Interface::gemv('T', std::complex<double>(1.0), C2, N);
    H.data = H1.data;
#else
    for (std::size_t j = 0; j < n12; ++j)
    {
        std::complex<double> s(0.0, 0.0);
        for (std::size_t ia = 0; ia < a; ++ia)
            s += (*C2.data)[ia * n12 + j] * (*N.data)[ia];
        (*H.data)[j] = s;
    }
#endif
    return H;
}
} // namespace

namespace librpa_int
{
namespace qsgw
{

Hartree::Hartree(const MeanField &mf_in,
                 const AtomicBasis &atbasis_wfc_in,
                 const PeriodicBoundaryData &pbc_in,
                 const KPointBlacsParallelContext &kblacs_ctxt_in,
                 const ArrayDesc &desc_wfc_in)
    : mf(mf_in),
      desc_wfc(desc_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      pbc(pbc_in),
      symmetry_context(symmetry_context_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in)
{
}

ComplexMatrix Hartree::get_dmat_cplx_k_global(int ik) const
{
    // Mirrors legacy get_dmat_cplx_k_global (legacy hartree.cpp:88-100):
    // sum the density matrix over spin and spinor channels (n_soc -> n_spinor),
    // since the Hartree kernel couples only to the total charge density.
    // The per-channel 0.5*n_spins*n_spinor renormalization is built into
    // MeanField::get_dmat_cplx (meanfield.cpp:630), so summing the channels
    // yields the total charge density directly.
    const auto nspins = mf.get_n_spins();
    const auto nspinor = mf.get_n_spinor();
    ComplexMatrix dmat(mf.get_n_aos(), mf.get_n_aos());
    for (int ispin = 0; ispin < nspins; ++ispin)
    {
        for (int ispinor = 0; ispinor < nspinor; ++ispinor)
        {
            dmat += mf.get_dmat_cplx(ispin, ispinor, ispinor, ik);
        }
    }
    return dmat;
}

ComplexMatrix Hartree::extract_dmat_cplx_IJblock(const ComplexMatrix &dmat,
                                                 atom_t I, atom_t J) const
{
    // Mirrors legacy extract_dmat_cplx_IJblock (legacy hartree.cpp:102-117),
    // using the new AtomicBasis::get_global_indices in place of the legacy
    // atom_iw_loc2glo global-index helper.
    const auto I_idx = atbasis_wfc.get_global_indices(I);
    const auto J_idx = atbasis_wfc.get_global_indices(J);
    ComplexMatrix block(I_idx.size(), J_idx.size());
    for (std::size_t i = 0; i < I_idx.size(); ++i)
    {
        for (std::size_t j = 0; j < J_idx.size(); ++j)
        {
            block(i, j) = dmat(I_idx[i], J_idx[j]);
        }
    }
    return block;
}

std::array<int, 3> Hartree::nearest_R(atom_t I, atom_t J,
                                      const std::array<int, 3> &R) const
{
    // Mirrors legacy nearest_R (legacy hartree.cpp:120-150) via the existing
    // find_nearest_bvk_cell (pbc.h:120, impl pbc.cpp:581), which maps a direct
    // lattice vector R to the nearest BvK image for the atom pair (I,J). The
    // per-atom fractional coordinates come from SymmetryContext::input_coord_frac
    // (input_symmetry.h:132). NOTE: only needed if the build() kernel path
    // returns k-space and requires a manual k->R Fourier (as the legacy
    // cal_cvcd_k_hartree path did); the A1 cal_loop3 path returns real-space Hs
    // and may not call this.
    const auto &coord_frac = symmetry_context.input_coord_frac;
    const auto &cI_arr = coord_frac.at(I);
    const auto &cJ_arr = coord_frac.at(J);
    const Vector3<double> cI(cI_arr[0], cI_arr[1], cI_arr[2]);
    const Vector3<double> cJ(cJ_arr[0], cJ_arr[1], cJ_arr[2]);
    const Vector3_Order<int> bvk_direct(R[0], R[1], R[2]);
    const auto R_bvk = find_nearest_bvk_cell(cI, cJ, bvk_direct,
                                             pbc.period, pbc.latvec);
    return {R_bvk.x, R_bvk.y, R_bvk.z};
}

void Hartree::build(const AtomicBasis &atbasis_abf,
                    const Cs_LRI &Cs,
                    const atpair_R_mat_t &coul_mat)
{
    // Hand-written RI-Hartree (J) contraction, path A. Mirrors the patched
    // cal_cvcd_k_hartree (libri-overrides LRI-cal_hartree.hpp) + the legacy
    // k->R FT (legacy hartree.cpp:130-289), using current-libRI primitives:
    //   RI::Tensor transpose/operator, Blas_Interface::gemv (direct for the
    //   2D Vq*M step; via the file-local contract_* helpers for the 3D Cs
    //   steps), hand-written conjugate/FT (current libRI has no Tensor::conjugate
    //   / FT_Ds).
    // PENDING: coremath confirmation of contraction axis directions + atom-pair
    //   Hermitian fold; MPI atom-pair/k distribution + OpenMP (serial for the
    //   non-SOC spike); SOC complex density.  Spike vs legacy Hartree_is_ik_KS.
    (void)atbasis_abf;
    if (is_rspace_built_) return;

#ifdef LIBRPA_USE_LIBRI
    const int nk = mf.get_n_kpoints();
    const int natom = atbasis_wfc.n_atoms;
    const auto &kfrac_list = pbc.kfrac_list;

    // Initialize the libRI LRI container (patched Hartree.h set_parallel, path C).
    {
        std::map<int, std::array<double, 3>> atoms_pos;
        for (int i = 0; i < natom; ++i) atoms_pos[i] = {0.0, 0.0, 0.0};
        lri_.set_parallel(comm_h.comm, atoms_pos, pbc.latvec_array, pbc.period_array, {});
    }

    std::vector<int> list_I(natom), list_J(natom), list_k(nk);
    std::iota(list_I.begin(), list_I.end(), 0);
    std::iota(list_J.begin(), list_J.end(), 0);
    std::iota(list_k.begin(), list_k.end(), 0);
    std::set<int> set_IJ(list_I.begin(), list_I.end());
    std::vector<int> list_IJ(set_IJ.begin(), set_IJ.end());

    // --- 1. FT Cs_R -> Csk[I][J][k] = sum_R Cs[I][{J,R}] * exp(+2πi k·R) ---
    // (legacy LRI-cal_hartree.hpp L80-110). Cs shape {aux_I, nao_I, nao_J}.
    std::map<int, std::map<int, std::map<int, RI::Tensor<std::complex<double>>>>> Csk;
    for (const auto &I_JR_C : Cs.data_libri)
    {
        const int I = I_JR_C.first;
        for (const auto &JR_C : I_JR_C.second)
        {
            const int J = JR_C.first.first;
            const auto &Ra = JR_C.first.second;
            const auto C_cplx = RI::Global_Func::convert<std::complex<double>>(JR_C.second);
            for (const int k : list_k)
            {
                const auto &kf = kfrac_list[k];
                const double arg = TWO_PI * (kf.x * Ra[0] + kf.y * Ra[1] + kf.z * Ra[2]);
                const std::complex<double> phase(std::cos(arg), std::sin(arg));
                auto &tgt = Csk[I][J][k];
                const auto weighted = phase * C_cplx;
                if (tgt.empty()) tgt = weighted;
                else tgt = tgt + weighted;
            }
        }
    }

    // --- 2. FT V_R -> Vq[I][J] = sum_R V[I][J][R] (q=0)  (legacy L52-77) ---
    std::map<int, std::map<int, RI::Tensor<std::complex<double>>>> Vq;
    for (const auto &I_JRV : coul_mat)
    {
        const int I = I_JRV.first;
        for (const auto &J_RV : I_JRV.second)
        {
            const int J = J_RV.first;
            for (const auto &R_V : J_RV.second)
            {
                const auto &V = R_V.second; // shared_ptr<matrix> {aux_I, aux_J}, real
                const std::size_t n = static_cast<std::size_t>(V->nr) * V->nc;
                auto &tgt = Vq[I][J];
                if (tgt.empty())
                    tgt = RI::Tensor<std::complex<double>>(
                        {static_cast<std::size_t>(V->nr), static_cast<std::size_t>(V->nc)},
                        std::make_shared<std::valarray<std::complex<double>>>(n));
                for (std::size_t i = 0; i < n; ++i)
                    (*tgt.data)[i] += std::complex<double>((*V->c)[i], 0.0);
            }
        }
    }

    // --- 3. M_nu = sum_{u,v,k} (Csk[u][v][k]·D[v][u][k]^T + conj(Csk[v][u][k])·D[v][u][k])
    //     (legacy L124-182; atom-pair Hermitian fold). M indexed by the auxiliary atom.
    std::map<int, RI::Tensor<std::complex<double>>> M;
    auto add_M = [&](const int atom, const RI::Tensor<std::complex<double>> &m) {
        auto &t = M[atom];
        if (t.empty()) t = m; else t = t + m;
    };
    for (const int v : list_I)
    {
        for (const int u : list_J)
        {
            for (const int k : list_k)
            {
                const auto dmat_k = get_dmat_cplx_k_global(k);
                const auto D_vu = extract_dmat_cplx_IJblock(dmat_k, v, u); // {nao_v, nao_u}
                RI::Tensor<std::complex<double>> D_vu_k(
                    {static_cast<std::size_t>(D_vu.nr), static_cast<std::size_t>(D_vu.nc)},
                    std::make_shared<std::valarray<std::complex<double>>>(D_vu.c, D_vu.size));
                // M[u] += Csk[u][v][k] · D_vu_k^T   (Csk shape {aux_u, nao_u, nao_v})
                if (Csk.count(u) && Csk.at(u).count(v) && Csk.at(u).at(v).count(k))
                    add_M(u, contract_3_2_to_1(Csk.at(u).at(v).at(k), D_vu_k.transpose()));
                // M[v] += conj(Csk[v][u][k]) · D_vu_k  (Csk shape {aux_v, nao_v, nao_u})
                if (Csk.count(v) && Csk.at(v).count(u) && Csk.at(v).at(u).count(k))
                    add_M(v, contract_3_2_to_1(conj_tensor(Csk.at(v).at(u).at(k)), D_vu_k));
            }
        }
    }

    // --- 4. N_mu = (1/nk) sum_nu Vq[mu][nu] · M[nu]  (legacy L188-217; serial, no comm_map)
    //     Vq{aux_mu,aux_nu} (2D) · M{aux_nu} (1D) -> N{aux_mu} via Blas_Interface::gemv('N').
    std::map<int, RI::Tensor<std::complex<double>>> N;
    for (const int mu : list_I)
    {
        if (!Vq.count(mu)) continue;
        auto &Nmu = N[mu];
        for (const int nu : list_IJ)
        {
            if (!Vq.at(mu).count(nu)) continue;
            const auto &Vq_mn = Vq.at(mu).at(nu);
            auto it_m = M.find(nu);
            if (it_m == M.end() || it_m->second.empty()) continue;
            const auto n = RI::Blas_Interface::gemv('N', std::complex<double>(1.0), Vq_mn, it_m->second);
            if (Nmu.empty()) Nmu = n; else Nmu = Nmu + n;
        }
        if (!Nmu.empty()) Nmu = std::complex<double>(1.0 / static_cast<double>(nk)) * Nmu;
    }
    // N_mu MPI all-gather over set_IJ (patched LRI-cal_hartree.hpp L217, path-C framework).
    N = RI::Communicate_Tensors_Map_Judge::comm_map(comm_h.comm, std::move(N), set_IJ);

    // --- 5. H_st[k] = Csk[s][t][k]·N[s] + (conj(Csk[t][s][k])·N[t])^T
    //     (legacy L219-274, TWO Hermitian atom-pair-endpoint terms). Output H[s][t][k]
    //     {nao_s, nao_t}. Fixed per coremath: the legacy patch sums two terms with
    //     mu = s and mu = t (NOT a single sum over all mu); the t-term carries a
    //     conjugate + transpose (Hermitian cross term). A single-term version
    //     silently breaks the Hartree numerics. Csk is 3-level map[I][J][k].
    std::map<int, std::map<int, std::map<int, RI::Tensor<std::complex<double>>>>> Hk;
    for (const int s : list_I)
    {
        for (const int t : list_J)
        {
            for (const int k : list_k)
            {
                bool has = false;
                RI::Tensor<std::complex<double>> H_st;
                // term 1: Csk[s][t][k] · N[s]  -> {nao_s, nao_t}
                if (Csk.count(s) && Csk.at(s).count(t) && Csk.at(s).at(t).count(k) &&
                    N.count(s) && !N.at(s).empty())
                {
                    const auto h1 = contract_3_1_to_2(Csk.at(s).at(t).at(k), N.at(s));
                    if (!has) { H_st = h1; has = true; } else H_st = H_st + h1;
                }
                // term 2: conj(Csk[t][s][k]) · N[t] -> {nao_t, nao_s}, transpose -> {nao_s, nao_t}
                if (Csk.count(t) && Csk.at(t).count(s) && Csk.at(t).at(s).count(k) &&
                    N.count(t) && !N.at(t).empty())
                {
                    auto h2 = contract_3_1_to_2(conj_tensor(Csk.at(t).at(s).at(k)), N.at(t));
                    h2 = h2.transpose();
                    if (!has) { H_st = h2; has = true; } else H_st = H_st + h2;
                }
                if (has) Hk[s][t][k] = H_st;
            }
        }
    }

    // --- 6. H_st[k] -> HHartree_IJR[I][{J,R_bvk}] = sum_k H[I][J][k]·exp(-2πi k·R_bvk)/nk
    //     (legacy hartree.cpp:333-367, with nearest_R BvK fold).
    for (const int s : list_I)
    {
        if (!Hk.count(s)) continue;
        for (const int t : list_J)
        {
            if (!Hk.at(s).count(t)) continue;
            for (const auto &R : pbc.Rlist)
            {
                const auto R_bvk = nearest_R(s, t, {R.x, R.y, R.z});
                RI::Tensor<std::complex<double>> HIJR;
                bool has = false;
                for (const int k : list_k)
                {
                    if (!Hk.at(s).at(t).count(k)) continue;
                    const auto &kf = kfrac_list[k];
                    const double ang = -TWO_PI * (kf.x * R_bvk[0] + kf.y * R_bvk[1] + kf.z * R_bvk[2]);
                    const std::complex<double> phase(std::cos(ang), std::sin(ang));
                    const auto weighted = (phase / static_cast<double>(nk)) * Hk.at(s).at(t).at(k);
                    if (!has) { HIJR = weighted; has = true; } else HIJR = HIJR + weighted;
                }
                if (has) HHartree_IJR[s][{t, R_bvk}] = std::move(HIJR);
            }
        }
    }

    is_rspace_built_ = true;
#else
    (void)Cs;
    (void)coul_mat;
    throw std::runtime_error("Hartree::build: requires LIBRPA_USE_LIBRI");
#endif
}

void Hartree::build_KS(const WfcMap &wfc_target,
                       const std::vector<Vector3_Order<double>> &kfrac_target)
{
    // Mirrors legacy build_KS (legacy hartree.cpp:292-410) on the new ScaLAPACK
    // API. Simpler than Exx::build_KS_blacs because HHartree_IJR is spin-summed
    // (a single copy): for each k-point Fourier-transform HHartree_IJR to the
    // k-space NAO block, then for each (ispin, ispinor) rotate
    //   H_KS = conj(wfc) * HH_nao_nao(k) * wfc^T   (NO exchange minus sign)
    // and accumulate into Hartree_is_ik_KS / EHartree. SOC: legacy isoc loop
    // becomes the ispinor loop (n_soc -> n_spinor).
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
    using TAC = libri_types<int, int>::TAC; // std::pair<int, std::array<int,3>>

    assert(this->is_rspace_built_);
    if (this->is_kspace_built_)
    {
        reset_kspace();
    }

    const auto n_aos = mf.get_n_aos();
    const auto n_spins = mf.get_n_spins();
    const auto n_bands = mf.get_n_bands();
    const auto n_spinor = mf.get_n_spinor();
    const auto &blacs_ctxt_h = kblacs_ctxt.blacs_h;

    // ScaLAPACK array descriptors (legacy :298-311).
    ArrayDesc desc_nao_nao(blacs_ctxt_h);
    ArrayDesc desc_nband_nao(blacs_ctxt_h);
    ArrayDesc desc_nband_nband(blacs_ctxt_h);
    ArrayDesc desc_nband_nband_fb(blacs_ctxt_h);
    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);
    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);

    auto HH_nao_nao = init_local_mat<std::complex<double>>(desc_nao_nao, MAJOR::COL);
    auto temp_nband_nao = init_local_mat<std::complex<double>>(desc_nband_nao, MAJOR::COL);
    auto HH_nband_nband = init_local_mat<std::complex<double>>(desc_nband_nband, MAJOR::COL);
    auto HH_nband_nband_fb = init_local_mat<std::complex<double>>(desc_nband_nband_fb, MAJOR::COL);

    // Gather HHartree IJ blocks with complete R across processes (legacy :320-333),
    // so each rank can Fourier-transform independently.
    const auto set_IJ_naonao = get_necessary_IJ_from_block_2D(
        atbasis_wfc, atbasis_wfc, desc_nao_nao);
    const auto Iset_Jset = convert_IJset_to_Iset_Jset(set_IJ_naonao);
    auto HH_I_JR_local = comm_map2_first(comm_h.comm, HHartree_IJR,
                                         Iset_Jset.first, Iset_Jset.second);

    for (int ik = 0; ik < static_cast<int>(kfrac_target.size()); ++ik)
    {
        HH_nao_nao.zero_out();
        const auto &kfrac = kfrac_target[ik];
        // Fourier phase exp(i k.R) over the real-space atom-pair blocks (legacy :325-337).
        std::function<std::complex<double>(const int &, const TAC &)> fourier =
            [kfrac](const int & /*I*/, const TAC &JR) {
                const auto &Ra = JR.second;
                const Vector3_Order<int> R(Ra[0], Ra[1], Ra[2]);
                const auto ang = (kfrac * R) * TWO_PI;
                return std::complex<double>{std::cos(ang), std::sin(ang)};
            };
        collect_block_from_IJ_storage_tensor_transform(
            HH_nao_nao, desc_nao_nao, atbasis_wfc, atbasis_wfc, fourier, HH_I_JR_local);

        for (int isp = 0; isp < n_spins; ++isp)
        {
            // NAO-basis Hartree (legacy :379-386); same spin-summed copy per spin.
            if (Hartree_is_ik_nao.count(isp) == 0 || Hartree_is_ik_nao.at(isp).count(ik) == 0)
            {
                Hartree_is_ik_nao[isp][ik] =
                    init_local_mat<std::complex<double>>(desc_nao_nao, MAJOR::COL);
            }
            Hartree_is_ik_nao[isp][ik] += HH_nao_nao;

            for (int ispinor = 0; ispinor < n_spinor; ++ispinor)
            {
                const auto &wfc = wfc_target.at(isp).at(ispinor).at(ik);
                const auto wfc_block =
                    get_local_mat(wfc.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                // temp = wfc^dagger . HH_nao_nao   (legacy :393-397, coefficient +1, no minus).
                ScalapackConnector::pgemm_f('N', 'N', n_bands, n_aos, n_aos, 1.0,
                                             wfc_block.ptr(), 1, 1, desc_nband_nao.desc,
                                             HH_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                             0.0, temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                // HH_KS = temp . wfc   (legacy :398-402, coefficient +1, no minus).
                ScalapackConnector::pgemm_f('N', 'C', n_bands, n_bands, n_aos, 1.0,
                                             temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                             wfc_block.ptr(), 1, 1, desc_nband_nao.desc,
                                             0.0, HH_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                // Collect the full n_bands x n_bands block to the root grid (legacy :404-407).
                ScalapackConnector::pgemr2d_f(n_bands, n_bands,
                                               HH_nband_nband.ptr(), 1, 1, desc_nband_nband.desc,
                                               HH_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc,
                                               desc_nband_nband_fb.ictxt());

                if (Hartree_is_ik_KS.count(isp) == 0 || Hartree_is_ik_KS.at(isp).count(ik) == 0)
                {
                    Hartree_is_ik_KS[isp][ik] =
                        init_local_mat<std::complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                    if (comm_h.is_root())
                    {
                        for (int ib = 0; ib < n_bands; ++ib)
                        {
                            EHartree[isp][ik][ib] = 0.0;
                        }
                    }
                }
                Hartree_is_ik_KS[isp][ik] += HH_nband_nband_fb;
                if (comm_h.is_root())
                {
                    for (int ib = 0; ib < n_bands; ++ib)
                    {
                        EHartree[isp][ik][ib] += HH_nband_nband_fb(ib, ib).real();
                    }
                }
            }
        }
    }
    is_kspace_built_ = true;
}

void Hartree::build_KS_kgrid0(const QsgwState &state)
{
    // H2 fixed-basis anchor: project using the step-0 wavefunctions cached in
    // QsgwState::wfc0 (replaces the legacy meanfield.get_eigenvectors0()).
    build_KS(state.wfc0, pbc.kfrac_list);
}

void Hartree::build_KS_kgrid()
{
    // Current wavefunctions (same behavior as legacy build_KS_kgrid).
    build_KS(mf.get_eigenvectors(), pbc.kfrac_list);
}

void Hartree::build_KS_band(const WfcMap &wfc_band,
                            const std::vector<Vector3_Order<double>> &kfrac_band)
{
    // Band wavefunctions supplied by the caller (band path does not use wfc0).
    build_KS(wfc_band, kfrac_band);
}

void Hartree::reset_rspace()
{
    HHartree_IJR.clear();
    is_rspace_built_ = false;
}

void Hartree::reset_kspace()
{
    Hartree_is_ik_KS.clear();
    Hartree_is_ik_nao.clear();
    EHartree.clear();
    is_kspace_built_ = false;
}

} // namespace qsgw
} // namespace librpa_int
