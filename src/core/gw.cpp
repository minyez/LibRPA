#include "gw.h"

// Public API headers
#include "librpa_enums.h"

#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../io/fs.h"
#include "../io/global_io.h"
#include "../io/output_gw.h"
#include "../io/stl_io_helper.h"
// #include "../math/utils_matrix_m.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "../gpu/la_connector.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla_connector.h>
using namespace ddla;
#endif
#include "../utils/utils_mem.h"
#include "input_symmetry.h"
#include "atom.h"
#include "atomic_basis.h"
#include "chi0.h"
#include "epsilon.h"
#include "geometry.h"
#include "meanfield_mpi.h"
#include "pbc.h"
#include "ri.h"
#include "utils_atomic_basis_blacs.h"
#include "utils_complexmatrix.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/global/Tensor.h>
#include <RI/physics/GW.h>
#include <RI/physics/symmetry/Symmetry_Filter.h>
using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif

namespace librpa_int
{

using std::vector;

namespace
{

int infer_target_n_bands(
    const MpiCommHandler &comm_h,
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
    const int fallback)
{
    int n_bands_local = 0;
    for (const auto &spin_wfc : wfc_target)
    {
        for (const auto &spinor_wfc : spin_wfc.second)
        {
            for (const auto &k_wfc : spinor_wfc.second)
            {
                const auto &wfc = k_wfc.second;
                if (wfc.nr <= 0) continue;
                if (n_bands_local != 0 && n_bands_local != wfc.nr)
                    throw LIBRPA_RUNTIME_ERROR("inconsistent target wave-function band counts");
                n_bands_local = wfc.nr;
            }
        }
    }

    int n_bands_max = 0;
    comm_h.allreduce(&n_bands_local, &n_bands_max, 1, MPI_MAX);
    if (n_bands_max == 0) return fallback;

    const int n_bands_min_send = n_bands_local == 0 ? n_bands_max : n_bands_local;
    int n_bands_min = 0;
    comm_h.allreduce(&n_bands_min_send, &n_bands_min, 1, MPI_MIN);
    if (n_bands_min != n_bands_max)
        throw LIBRPA_RUNTIME_ERROR("inconsistent target wave-function band counts");

    return n_bands_max;
}

std::vector<int> collect_target_iks(
    const MpiCommHandler &comm_h,
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
    const int n_kpts)
{
    std::vector<int> has_local(n_kpts, 0);
    std::vector<int> has_global(n_kpts, 0);

    for (const auto &spin_wfc : wfc_target)
    {
        for (const auto &spinor_wfc : spin_wfc.second)
        {
            for (const auto &k_wfc : spinor_wfc.second)
            {
                const int ik = k_wfc.first;
                if (ik >= 0 && ik < n_kpts) has_local[ik] = 1;
            }
        }
    }

    if (n_kpts > 0)
        comm_h.allreduce(has_local.data(), has_global.data(), n_kpts, MPI_MAX);

    std::vector<int> iks;
    for (int ik = 0; ik != n_kpts; ++ik)
    {
        if (has_global[ik]) iks.emplace_back(ik);
    }
    return iks;
}

std::string make_sigc_ks_imagfreq_band_stem(const std::string &output_dir, const int index)
{
    std::ostringstream ss;
    ss << path_as_directory(output_dir) << "self_energy_omega_band_"
       << std::setfill('0') << std::setw(5) << index;
    return ss.str();
}

std::map<atom_t, size_t> build_atom_nw_map(const AtomicBasis& atbasis)
{
    std::map<atom_t, size_t> atom_nw;
    for (atom_t atom = 0; atom != as_atom(atbasis.n_atoms); ++atom)
    {
        atom_nw[atom] = atbasis.get_atom_nb(atom);
    }
    return atom_nw;
}

bool use_input_symmetry_ibz_root_projection(
    const SymmetryContext& ctx,
    const PeriodicBoundaryData& pbc,
    const int n_target_kpoints,
    const int n_meanfield_kpoints)
{
    return ctx.available && !ctx.kstars.empty()
        && pbc.klist.size() < static_cast<std::size_t>(pbc.get_n_cells_bvk())
        && ctx.kstars.size() == static_cast<std::size_t>(n_meanfield_kpoints)
        && n_target_kpoints == n_meanfield_kpoints;
}

std::map<std::pair<int, int>, std::set<std::array<int, 3>>>
convert_input_symmetry_irreducible_sector_to_libri_gw(
    const librpa_int::input_symmetry_irreducible_sector_t& irreducible_sector,
    const std::array<int, 3>& period)
{
    auto canonicalize_r = [&period](const std::array<int, 3>& r) {
        auto centered_mod = [](const int value, const int cell_period) {
            if (cell_period <= 0)
            {
                return value;
            }
            return (value % cell_period + 3 * cell_period / 2) % cell_period
                - cell_period / 2;
        };
        return std::array<int, 3>{centered_mod(r[0], period[0]),
                                  centered_mod(r[1], period[1]),
                                  centered_mod(r[2], period[2])};
    };

    std::map<std::pair<int, int>, std::set<std::array<int, 3>>> libri_sector;
    for (const auto& pair_Rs : irreducible_sector)
    {
        const std::pair<int, int> atom_pair{
            static_cast<int>(pair_Rs.first.first),
            static_cast<int>(pair_Rs.first.second)};
        for (const auto& r : pair_Rs.second)
        {
            libri_sector[atom_pair].insert(canonicalize_r(r));
        }
    }
    return libri_sector;
}

#ifdef LIBRPA_USE_LIBRI
template <typename TA, typename TC, typename Tdata>
class OutputOnlyFilter_GW_Symmetry : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
    using TAC = std::pair<TA, TC>;

    OutputOnlyFilter_GW_Symmetry(
        const TC& period,
        const std::map<std::pair<TA, TA>, std::set<TC>>& irreducible_sector)
        : symmetry_(period, irreducible_sector)
    {
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b1:
                return !this->symmetry_.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TA&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b2:
                return !this->symmetry_.in_irreducible_sector(A3, A1);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TA& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a2b1:
            case RI::Label::ab_ab::a0b0_a2b2:
                return !this->symmetry_.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

  private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry_;
};

template <typename Tdata>
ComplexMatrix convert_libri_tensor_to_complex_matrix_gw(
    const RI::Tensor<Tdata>& tensor,
    const int nrows,
    const int ncols)
{
    ComplexMatrix matrix(nrows, ncols);
    for (int row = 0; row != nrows; ++row)
    {
        for (int col = 0; col != ncols; ++col)
        {
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                matrix(row, col) = tensor(row, col);
            }
            else
            {
                matrix(row, col) = std::complex<double>(tensor(row, col), 0.0);
            }
        }
    }
    return matrix;
}

template <typename Tdata>
RI::Tensor<Tdata> convert_complex_matrix_to_libri_tensor_gw(
    const ComplexMatrix& matrix)
{
    RI::Tensor<Tdata> tensor(
        {static_cast<std::size_t>(matrix.nr), static_cast<std::size_t>(matrix.nc)});
    for (int row = 0; row != matrix.nr; ++row)
    {
        for (int col = 0; col != matrix.nc; ++col)
        {
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                tensor(row, col) = matrix(row, col);
            }
            else
            {
                tensor(row, col) = matrix(row, col).real();
            }
        }
    }
    return tensor;
}

template <typename Tdata>
std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>
restore_input_symmetry_ao_rspace_tensor_map_gw(
    const std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>& tensors_ir,
    const librpa_int::SymmetryContext& symmetry_ctx,
    const librpa_int::input_symmetry_rspace_sector_stars_t& sector_stars,
    const AtomicBasis& atbasis_wfc)
{
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> tensors_full;
    for (const auto& i_entry : tensors_ir)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto& jr_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(jr_entry.first.first);
            const Vector3_Order<int> ir_R{
                jr_entry.first.second[0], jr_entry.first.second[1], jr_entry.first.second[2]};
            const auto pair_iter = sector_stars.find({ir_I, ir_J});
            if (pair_iter == sector_stars.end() || pair_iter->second.count(ir_R) == 0)
            {
                std::ostringstream oss;
                oss << "Failed to match a symmetry-filtered GW self-energy block with the"
                    << " ABACUS irreducible-sector restore map for I=" << ir_I
                    << " J=" << ir_J << " R=(" << ir_R.x << "," << ir_R.y << ","
                    << ir_R.z << ")";
                throw std::runtime_error(oss.str());
            }

            const auto nao_I = atbasis_wfc.get_atom_nb(ir_I);
            const auto nao_J = atbasis_wfc.get_atom_nb(ir_J);
            const ComplexMatrix sigma_ir =
                convert_libri_tensor_to_complex_matrix_gw(jr_entry.second, nao_I, nao_J);
            for (const auto& restore_member : pair_iter->second.at(ir_R))
            {
                const ComplexMatrix sigma_full = librpa_int::rotate_input_symmetry_rspace_matrix(
                    symmetry_ctx, restore_member.isym, ir_I, ir_J, sigma_ir);
                auto& target = tensors_full[restore_member.full_atom_pair.first][{
                    static_cast<int>(restore_member.full_atom_pair.second),
                    {restore_member.full_R.x, restore_member.full_R.y, restore_member.full_R.z}}];
                if (!target.empty())
                {
                    throw std::runtime_error(
                        "Duplicate full-sector GW self-energy block appears during ABACUS symmetry restore");
                }
                target = convert_complex_matrix_to_libri_tensor_gw<Tdata>(sigma_full);
            }
        }
    }
    return tensors_full;
}
#endif

template <typename T>
bool is_effectively_zero_matrix(const matrix_m<T> &mat,
                                const typename matrix_m<T>::real_t threshold = 1e-15)
{
    const auto data = mat.sptr();
    if (!data) return true;
    for (size_t i = 0; i != mat.size(); ++i)
    {
        if (std::abs((*data)[i]) > threshold) return false;
    }
    return true;
}

void complete_hermitian_Wc_q_blocks(
    atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old &Wc_q)
{
    std::vector<atom_t> atoms_row;
    for (const auto &atom_i_pair : Wc_q) atoms_row.push_back(atom_i_pair.first);

    for (const auto atom_i : atoms_row)
    {
        std::vector<atom_t> atoms_col;
        for (const auto &atom_j_pair : Wc_q.at(atom_i))
            atoms_col.push_back(atom_j_pair.first);

        for (const auto atom_j : atoms_col)
        {
            for (const auto &q_block : Wc_q.at(atom_i).at(atom_j))
            {
                assert(q_block.second.major() == MAJOR::ROW);
                if (atom_i != atom_j)
                {
                    Wc_q[atom_j][atom_i][q_block.first] = q_block.second.get_transpose(true);
                }
                else
                {
                    auto hermitian_block = q_block.second;
                    hermitian_block =
                        (hermitian_block + hermitian_block.get_transpose(true)) * 0.5;
                    Wc_q[atom_i][atom_i][q_block.first] = hermitian_block;
                }
            }
        }
    }
}

int wc_rf_checked_ifreq_end(const int start, const int end, const int n_freq)
{
    if (start < 0)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_start must be non-negative");
    if (start >= n_freq)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_start is outside the Wc frequency grid");
    if (end >= 0 && end <= start)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_end must be negative or greater than ifreq_output_wc_start");
    const int checked_end = end < 0 ? n_freq : end;
    if (checked_end > n_freq)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_end is outside the Wc frequency grid");
    return checked_end;
}

void write_wc_rf_atom_blocks(
    const atom_mapping<std::map<Vector3_Order<int>, Matz>>::pair_t_old &Wc_R,
    const PeriodicBoundaryData &pbc, const std::string &output_dir, const int ifreq)
{
    for (const auto &[I, J_RWc] : Wc_R)
    {
        for (const auto &[J, R_Wc] : J_RWc)
        {
            for (const auto &[R, Wc] : R_Wc)
            {
                std::ostringstream ss;
                ss << path_as_directory(output_dir)
                   << "Wc_Mu_" << I << "_Nu_" << J
                   << "_iR_" << pbc.get_R_index(R)
                   << "_ifreq_" << ifreq << ".mtx";
                print_matrix_mm_file(Wc, ss.str(), "", 1e-10);
            }
        }
    }
}

}  // namespace

G0W0::G0W0(const MeanField &mf_in, const AtomicBasis &atbasis_wfc_in,
           const PeriodicBoundaryData &pbc_in,
           const SymmetryContext &symmetry_context_in,
           const TFGrids &tfg_in,
           const KPointBlacsParallelContext &kblacs_ctxt_in,
           const ArrayDesc &desc_wfc_in, bool is_mf_eigvec_k_distributed,
           const bool use_symmetry_context_in)
    : mf(mf_in),
      desc_wfc(desc_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      pbc(pbc_in),
      symmetry_context(symmetry_context_in),
      use_symmetry_context(use_symmetry_context_in),
      tfg(tfg_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in)
{
    comm_h.check_initialized();

    is_mf_eigvec_k_distributed_ = is_mf_eigvec_k_distributed;
    is_rspace_built_ = false;
    is_kspace_built_ = false;
    is_rspace_redist_for_KS_ = false;
    is_rspace_redist_blacs_ = false;
    output_sigc_ks_if_band_index_ = 0;

    // Public runtime options
    libri_threshold_C = 0.0;
    libri_threshold_Wc = 0.0;
    libri_threshold_G = 0.0;
    output_dir = "./";  // POSIX
    output_sigc_mat = false;
    output_sigc_ks_if = true;
    output_sigc_mat_rt = false;
    output_sigc_mat_rf = false;
    output_wc_rf = false;
    ifreq_output_wc_start = 0;
    ifreq_output_wc_end = -1;
}

void G0W0::reset_rspace()
{
    sigc_is_f_IJ_R.clear();
    is_rspace_built_ = false;
    is_rspace_redist_for_KS_ = false;
    is_rspace_redist_blacs_ = false;
}

void G0W0::reset_kspace()
{
    sigc_is_ik_f_KS.clear(); is_kspace_built_ = false;
}

#ifdef LIBRPA_USE_LIBRI
template <typename Tdata>
static void build_gf_libri_kpara(
    const MeanField &mf,
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<double> &taus,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    std::map<double, std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<Tdata>>>> &tau_gf_libri)
{
    global::profiler.start("g0w0_build_gf_libri_kpara");
    std::map<Vector3_Order<int>, std::vector<atpair_t>> map_R_IJs;
    for (const auto &[IJ, R]: IJRs)
    {
        map_R_IJs[R].emplace_back(IJ);
    }
    std::vector<Vector3_Order<int>> Rs_this;
    for (const auto &[R, _]: map_R_IJs)
        Rs_this.emplace_back(R);
    const int n_Rs_this = map_R_IJs.size();
    int n_Rs_max = n_Rs_this;
    comm_h.allreduce(MPI_IN_PLACE, &n_Rs_max, 1, MPI_MAX);
    auto gf_taus_Rs_cplx = get_gf_cplx_imagtimes_Rs_kpara(ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, taus, Rs_this, comm_h);
    // global::ofs_myid << "gf_taus_Rs_cplx " << gf_taus_Rs_cplx << std::endl;
    for (const auto &tau_gf_R_cplx: gf_taus_Rs_cplx)
    {
        const auto &tau = tau_gf_R_cplx.first;
        const auto &gf_R_cplx = tau_gf_R_cplx.second;
        for (const auto &R_gf_cplx: gf_R_cplx)
        {
            const auto &R = R_gf_cplx.first;
            const auto &gf_cplx = R_gf_cplx.second;
            std::array<int,3> Ra{R.x,R.y,R.z};
            const auto &map_IJs = map_R_IJs.at(R);
            omp_lock_t gf_lock;
            omp_init_lock(&gf_lock);
#pragma omp parallel for schedule(dynamic)
            for (const auto &IJ: map_IJs)
            {
                const auto &I = IJ.first;
                const auto &J = IJ.second;
                const auto gf_block = get_ap_block_from_global(gf_cplx, IJ, atbasis_wfc, atbasis_wfc);
                if constexpr (is_complex<Tdata>())
                {
                    auto p_gfmat = std::make_shared<std::valarray<Tdata>>(gf_block.c, gf_block.size);
                    omp_set_lock(&gf_lock);
                    tau_gf_libri[tau][I][{J, Ra}] = RI::Tensor<Tdata>({as_size(gf_block.nr), as_size(gf_block.nc)}, p_gfmat);
                    omp_unset_lock(&gf_lock);
                }
                else
                {
                    auto p_gfmat = std::make_shared<std::valarray<Tdata>>(gf_block.real().c, gf_block.size);
                    omp_set_lock(&gf_lock);
                    tau_gf_libri[tau][I][{J, Ra}] = RI::Tensor<Tdata>({as_size(gf_block.nr), as_size(gf_block.nc)}, p_gfmat);
                    omp_unset_lock(&gf_lock);
                }
            }
#pragma omp barrier
            omp_destroy_lock(&gf_lock);
        }
        // global::ofs_myid << R << std::endl;
        // print_complex_matrix("test", dmat_cplx, global::ofs_myid, true);
    }

    global::profiler.stop("g0w0_build_gf_libri_kpara");
}

template <typename Tdata>
static void build_gf_libri_kserial(
    const MeanField &mf,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const bool use_symmetry_context,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<double> &taus,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    std::map<double, std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<Tdata>>>> &tau_gf_libri)
{
    global::profiler.start("g0w0_build_gf_libri_kserial");
    std::set<Vector3_Order<int>> Rs_local;
    for (const auto &IJR: IJRs)
    {
        Rs_local.insert(IJR.second);
    }
    const std::vector<Vector3_Order<int>> Rs_vec{Rs_local.cbegin(), Rs_local.cend()};
    const auto atom_nw = build_atom_nw_map(atbasis_wfc);
    const bool restore_input_symmetry_kstars =
        use_symmetry_context
        && can_restore_input_symmetry_kstar_meanfield(
            symmetry_context, mf, kfrac_list, atom_nw, symmetry_context.input_coord_frac);
    const auto member_kfrac_targets = restore_input_symmetry_kstars
        ? build_input_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc)
        : input_symmetry_kstar_member_kfrac_targets_t{};
    auto gf = restore_input_symmetry_kstars
        ? get_input_symmetry_restored_gf_cplx_imagtimes_Rs(
              symmetry_context, mf, ispin, ispinor_bra, ispinor_ket, kfrac_list, taus, Rs_vec, atom_nw,
              symmetry_context.input_coord_frac, -1, &member_kfrac_targets)
        : mf.get_gf_cplx_imagtimes_Rs(ispin, ispinor_bra, ispinor_ket, kfrac_list, taus, Rs_vec);
    // global::ofs_myid << "gf " << gf << std::endl;
    tau_gf_libri.clear();
    // TODO: enable threading below
    for (auto t: taus)
    {
        tau_gf_libri[t] = {};
    }
    for (const auto &IJR: IJRs)
    {
        const auto &IJ = IJR.first;
        const auto &I = IJ.first;
        const auto &n_I = atbasis_wfc.get_atom_nb(I);
        const auto &J = IJ.second;
        const auto &n_J = atbasis_wfc.get_atom_nb(J);
        const auto &R = IJR.second;
        for (auto t: taus)
        {
            // skip when this process does not have any local GF data, i.e. Rs_local is empty
            if (gf.count(t) == 0 || gf.at(t).count(R) == 0) continue;
            const auto &gf_global = gf.at(t).at(R);
            std::shared_ptr<std::valarray<Tdata>> mat_ptr = std::make_shared<std::valarray<Tdata>>(n_I * n_J);
            if constexpr (is_complex<Tdata>())
            {
                for (size_t i = 0; i != n_I; i++)
                    for (size_t j = 0; j != n_J; j++)
                    {
                        (*mat_ptr)[i * n_J + j] = gf_global(atbasis_wfc.get_global_index(I, i),
                                                            atbasis_wfc.get_global_index(J, j));
                    }
            }
            else
            {
                for (size_t i = 0; i != n_I; i++)
                    for (size_t j = 0; j != n_J; j++)
                    {
                        (*mat_ptr)[i * n_J + j] = gf_global(atbasis_wfc.get_global_index(I, i),
                                                            atbasis_wfc.get_global_index(J, j)).real();
                    }
            }
            tau_gf_libri[t][as_int(I)][{as_int(J), {R.x, R.y, R.z}}] = RI::Tensor<Tdata>({n_I, n_J}, mat_ptr);
        }
    }
    gf.clear();
    global::profiler.stop("g0w0_build_gf_libri_kserial");
}

template <typename Tdata>
static void build_gf_libri_kblacs_para(
    const MeanField &mf,
    const KPointBlacsParallelContext &kblacs_ctxt,
    const ArrayDesc &desc_wfc, const ArrayDesc &desc_gf,
    const IndexScheduler &sched,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<double> &taus,
    const std::vector<Vector3_Order<int>> &Rs,
    std::map<double, std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<Tdata>>>> &tau_gf_libri)
{
    global::profiler.start("g0w0_build_gf_libri_kblacs_para");

    tau_gf_libri.clear();
    for (auto tau: taus)
        tau_gf_libri[tau] = {};

    auto gf_taus_Rs_cplx =
        get_gf_cplx_imagtimes_Rs_kblacs_para(ispin, ispinor_bra, ispinor_ket, mf,
                                             kfrac_list, taus, Rs, kblacs_ctxt,
                                             desc_wfc, desc_gf);

    for (auto &tau_gf_R_cplx: gf_taus_Rs_cplx)
    {
        const auto &tau = tau_gf_R_cplx.first;
        auto &gf_R_cplx = tau_gf_R_cplx.second;
        for (auto &R_gf_cplx: gf_R_cplx)
        {
            const auto &R = R_gf_cplx.first;
            auto &mat_blacs = R_gf_cplx.second;
            auto pair_mat =
                get_ap_map_from_blacs_dist_scheduler(mat_blacs, sched, atbasis_wfc,
                                                     atbasis_wfc, desc_gf);
            for (auto &[pair, mat_ap]: pair_mat)
            {
                const auto &I = as_int(pair.first);
                const auto &J = as_int(pair.second);
                const auto &n_I = atbasis_wfc.get_atom_nb(I);
                const auto &n_J = atbasis_wfc.get_atom_nb(J);
                mat_ap.swap_to_row_major();
                if constexpr (is_complex<Tdata>())
                {
                    tau_gf_libri[tau][I][{J, {R.x, R.y, R.z}}] =
                        RI::Tensor<Tdata>({n_I, n_J}, mat_ap.sptr());
                }
                else
                {
                    tau_gf_libri[tau][I][{J, {R.x, R.y, R.z}}] =
                        RI::Tensor<Tdata>({n_I, n_J}, mat_ap.get_real().sptr());
                }
            }
            mat_blacs.clear();
        }
    }

    global::profiler.stop("g0w0_build_gf_libri_kblacs_para");
}
#endif

void G0W0::build_spacetime(
    const LibrpaParallelRouting parallel_routing,
    const AtomicBasis& atbasis_abf,
    const Cs_LRI &LRI_Cs,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const ArrayDesc &ad_Wc,
    std::map<double, atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old>
        *Wc_freq_q_atom_pair,
    std::map<Vector3_Order<double>, ComplexMatrix> *sinvS,
    const AtomicBasis *basis_aux_compressed,
    const AtomicBasis *basis_aux_unfold,
    const BlacsCtxtHandler *blacs_ctxt_h,
    const ArrayDesc *desc_wfc_in)
{
    using global::profiler;
    using global::lib_printf_root;
    using global::ofs_myid;

    if (parallel_routing != LIBRPA_ROUTING_LIBRI)
    {
        comm_h.barrier();
        throw LIBRPA_RUNTIME_ERROR("not implemented");
    }

    lib_printf_root("Calculating correlation self-energy by space-time method\n");
    if (!tfg.has_time_grids())
    {
        lib_printf_root("Parsed time-frequency object do not have time grids, exiting\n");
        comm_h.barrier();
        throw LIBRPA_RUNTIME_ERROR("input TFGrids object has no time grids");
    }
    comm_h.barrier();

    const bool use_complex_tensor = mf.get_n_spinor() > 1;
    if (use_complex_tensor)
        lib_printf_root("Using complex tensor\n");
    else
        lib_printf_root("Using real tensor\n");

    assert(ad_Wc.initialized());
    // assert(blacs_sigc_h.initialized());
    // blacs_sigc_h_ = blacs_sigc_h;

    const int natom = this->atbasis_wfc.n_atoms;

    std::ofstream ofs_sigmac_r;

#ifndef LIBRPA_USE_LIBRI
    if (comm_h.myid == 0)
    {
        std::cout << "LIBRA::G0W0::build_spacetime is only implemented on top of LibRI" << std::endl;
        std::cout << "Please recompiler LibRPA with -DUSE_LIBRI and configure include path" << std::endl;
    }
    comm_h.barrier();
    throw LIBRPA_RUNTIME_ERROR("compilation");
#else
    const bool use_atom_pair_Wc = Wc_freq_q_atom_pair != nullptr;

    typedef std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> dtensor_map;
    typedef std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>> ztensor_map;
    std::map<double, dtensor_map> tau_Wc_libri;
    std::map<double, ztensor_map> tau_Wc_libri_cplx;

    if (use_atom_pair_Wc)
    {
        if (sinvS == nullptr || basis_aux_compressed == nullptr || basis_aux_unfold == nullptr ||
            blacs_ctxt_h == nullptr || desc_wfc_in == nullptr)
            throw LIBRPA_RUNTIME_ERROR(
                "shrink Wc build_spacetime requires sinvS, compressed/full ABF bases, BLACS context, and WFC descriptor");

        Chi0 unfold_helper(mf, atbasis_wfc, *basis_aux_compressed, pbc, symmetry_context, tfg, kblacs_ctxt,
                           *desc_wfc_in, is_mf_eigvec_k_distributed_, this->use_symmetry_context);

        if (output_wc_rf)
        {
            const int ifreq_end = wc_rf_checked_ifreq_end(
                ifreq_output_wc_start, ifreq_output_wc_end, tfg.get_n_grids());
            profiler.start("write_Wc_freq_R", "Export Wc(R,w) to file");
            for (int ifreq = ifreq_output_wc_start; ifreq != ifreq_end; ++ifreq)
            {
                const auto freq = tfg.get_freq_nodes()[ifreq];
                auto freq_iter = Wc_freq_q_atom_pair->find(freq);
                if (freq_iter == Wc_freq_q_atom_pair->end()) continue;
                auto Wc_q = freq_iter->second;

                profiler.start("unfold_Wc_abfs", "Do shrink transformation");
                unfold_helper.unfold_abfs_Wc_q(*sinvS, Wc_q, pbc.klist_coul,
                                               *basis_aux_unfold, *blacs_ctxt_h);
                profiler.stop("unfold_Wc_abfs");

                profiler.start("construct_Wc_freq_lower_half", "Construct Lower Half of Wc(q,w)");
                complete_hermitian_Wc_q_blocks(Wc_q);
                profiler.stop("construct_Wc_freq_lower_half");

                auto Wc_R = FT_Wc_q2R(comm_h, *basis_aux_unfold, symmetry_context, Wc_q, tfg,
                                      pbc, pbc.Rlist, true, output_dir, this->use_symmetry_context);
                write_wc_rf_atom_blocks(Wc_R, pbc, output_dir, ifreq);
            }
            profiler.stop("write_Wc_freq_R");
        }

        profiler.start("g0w0_build_spacetime_wt_ft_wc", "Tranform Wc (q,w) -> (q,t)");
        auto Wc_tau_q = CT_Wc_freq2time_q(
            comm_h, *basis_aux_compressed, *Wc_freq_q_atom_pair, tfg, pbc.get_n_cells_bvk(),
            pbc.Rlist, pbc.klist_coul);
        Wc_freq_q_atom_pair->clear();
        release_free_mem();
        profiler.stop("g0w0_build_spacetime_wt_ft_wc");

        lib_printf_root("Time for Fourier transform of Wc in GW (seconds, Wall/CPU): %f %f\n",
                        profiler.get_wall_time_last("g0w0_build_spacetime_wt_ft_wc"),
                        profiler.get_cpu_time_last("g0w0_build_spacetime_wt_ft_wc"));

        profiler.start("g0w0_build_spacetime_ct_ft_wc", "Tranform Wc (q,t) -> (R,t)");
        for (auto itau = 0; itau != tfg.get_n_grids(); ++itau)
        {
            const auto tau = tfg.get_time_nodes()[itau];
            auto &Wc_q = Wc_tau_q[tau];

            profiler.start("unfold_Wc_abfs", "Do shrink transformation");
            unfold_helper.unfold_abfs_Wc_q(*sinvS, Wc_q, pbc.klist_coul,
                                           *basis_aux_unfold, *blacs_ctxt_h);
            profiler.stop("unfold_Wc_abfs");

            profiler.start("construct_Wc_tau_lower_half", "Construct Lower Half of Wc(q,t)");
            complete_hermitian_Wc_q_blocks(Wc_q);
            profiler.stop("construct_Wc_tau_lower_half");

            profiler.start("g0w0_build_spacetime_Rq_ft_wc", "Tranform Wc (q,t) -> (R,t)");
            auto Wc_R = FT_Wc_q2R(comm_h, *basis_aux_unfold, symmetry_context, Wc_q, tfg,
                                  pbc, pbc.Rlist, false, output_dir, this->use_symmetry_context);
            profiler.stop("g0w0_build_spacetime_Rq_ft_wc");
            Wc_q.clear();

            profiler.start("g0w0_build_spacetime_prep_Wc_all", "Prepare LibRI Wc object");
            for (auto &[I, J_RWc] : Wc_R)
            {
                const auto n_I = atbasis_abf.get_atom_nb(I);
                for (auto &[J, R_Wc] : J_RWc)
                {
                    const auto n_J = atbasis_abf.get_atom_nb(J);
                    for (auto &[R, Wc_block] : R_Wc)
                    {
                        if (is_effectively_zero_matrix(Wc_block)) continue;
                        if (Wc_block.is_col_major()) Wc_block.swap_to_row_major();
                        if (use_complex_tensor)
                        {
                            tau_Wc_libri_cplx[tau][as_int(I)][{as_int(J), {R.x, R.y, R.z}}] =
                                RI::Tensor<cplxdb>({n_I, n_J}, Wc_block.sptr());
                        }
                        else
                        {
                            tau_Wc_libri[tau][as_int(I)][{as_int(J), {R.x, R.y, R.z}}] =
                                RI::Tensor<double>({n_I, n_J}, Wc_block.get_real().sptr());
                        }

                        if (I == J) continue;

                        const auto minus_R = (-R) % pbc.period;
                        const auto minus_R_iter = R_Wc.find(minus_R);
                        if (minus_R_iter == R_Wc.end()) continue;
                        if (is_effectively_zero_matrix(minus_R_iter->second)) continue;

                        if (use_complex_tensor)
                        {
                            const auto Wc_IJ_minus_R =
                                minus_R_iter->second.get_transpose(true);
                            tau_Wc_libri_cplx[tau][as_int(J)][{as_int(I), {R.x, R.y, R.z}}] =
                                RI::Tensor<cplxdb>({n_J, n_I}, Wc_IJ_minus_R.sptr());
                        }
                        else
                        {
                            const auto Wc_IJ_minus_R =
                                minus_R_iter->second.get_real().get_transpose();
                            tau_Wc_libri[tau][as_int(J)][{as_int(I), {R.x, R.y, R.z}}] =
                                RI::Tensor<double>({n_J, n_I}, Wc_IJ_minus_R.sptr());
                        }
                    }
                }
            }
            profiler.stop("g0w0_build_spacetime_prep_Wc_all");
        }
        Wc_tau_q.clear();
        profiler.stop("g0w0_build_spacetime_ct_ft_wc");
    }
    else
    {
        // Transform from frequency/reciprocal to time/real-space
        profiler.start("g0w0_build_spacetime_ct_ft_wc", "Tranform Wc (q,w) -> (R,t)");
        profiler.start("g0w0_build_spacetime_ct_ft_real_work", "Perform transformation");
        auto Wc_tau_R_blacs = CT_FT_Wc_freq_q(
            comm_h, Wc_freq_q, pbc, tfg, true, output_wc_rf, ifreq_output_wc_start,
            ifreq_output_wc_end, output_dir, &ad_Wc);
        release_free_mem();
        profiler.stop("g0w0_build_spacetime_ct_ft_real_work");

        // NOTE: for case of a few atoms, some process may have much more memory load than others
        profiler.start("g0w0_build_spacetime_get_ap", "Compute balance atom-pairs distribution");
        const auto map_atpairs_balanced = get_balanced_ap_distribution_for_consec_descriptor(
            atbasis_abf, atbasis_abf, ad_Wc);
        const auto it_ap_myid = map_atpairs_balanced.find(ad_Wc.myid());
        if (it_ap_myid != map_atpairs_balanced.cend())
            ofs_myid << it_ap_myid->second << std::endl;
        else
            ofs_myid << "No atom pairs for Wc on this process" << std::endl;
        profiler.stop("g0w0_build_spacetime_get_ap");

        int wc_major_mask = 0;
        for (const auto &[tau, map_R_mat]: Wc_tau_R_blacs)
        {
            for (const auto &[R, mat_blacs]: map_R_mat)
            {
                if (mat_blacs.major() == MAJOR::ROW)
                    wc_major_mask |= 1;
                else if (mat_blacs.major() == MAJOR::COL)
                    wc_major_mask |= 2;
                else
                    wc_major_mask |= 4;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &wc_major_mask, 1, mpi_datatype<int>::value, MPI_BOR, ad_Wc.comm());

        MAJOR wc_major = MAJOR::AUTO;
        if (wc_major_mask == 1)
            wc_major = MAJOR::ROW;
        else if (wc_major_mask == 2)
            wc_major = MAJOR::COL;
        else if (wc_major_mask == 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Wc(R,t) is empty");
        }
        else
        {
            throw LIBRPA_RUNTIME_ERROR("Inconsistent storage order among Wc(R,t) blocks");
        }

        ofs_myid << "wc_major == MAJOR::ROW ? " << std::boolalpha << (wc_major == MAJOR::ROW) << std::endl;

        IndexScheduler sched;
        // The scheduler indexes the existing BLACS buffers, so this must match Wc_tau_R_blacs.
        // LibRI's row-major requirement is handled below when each atom-pair block is wrapped.
        sched.init(map_atpairs_balanced, atbasis_abf, atbasis_abf, ad_Wc, wc_major == MAJOR::ROW);
        for (auto &[tau, map_R_mat]: Wc_tau_R_blacs)
        {
            for (auto &[R, mat_blacs]: map_R_mat)
            {
                auto pair_mat = get_ap_map_from_blacs_dist_scheduler(mat_blacs, sched, atbasis_abf, atbasis_abf, ad_Wc);
                profiler.start("g0w0_build_spacetime_prep_Wc_all", "Prepare LibRI Wc object");
                for (auto &[pair, mat_ap]: pair_mat)
                {
                    const auto &I = as_int(pair.first);
                    const auto &J = as_int(pair.second);
                    const auto &nabf_I = atbasis_abf.get_atom_nb(I);
                    const auto &nabf_J = atbasis_abf.get_atom_nb(J);
                    // LibRI Tensor is fixed to row major
                    if (mat_ap.is_col_major()) mat_ap.swap_to_row_major();
                    if (use_complex_tensor)
                        tau_Wc_libri_cplx[tau][I][{J, {R.x, R.y, R.z}}] = RI::Tensor<cplxdb>({nabf_I, nabf_J}, mat_ap.sptr());
                    else
                        tau_Wc_libri[tau][I][{J, {R.x, R.y, R.z}}] = RI::Tensor<double>({nabf_I, nabf_J}, mat_ap.get_real().sptr());
                }
                mat_blacs.clear();
                profiler.stop("g0w0_build_spacetime_prep_Wc_all");
            }
        }
        profiler.stop("g0w0_build_spacetime_ct_ft_wc");
    }

    lib_printf_root("Time for Fourier transform of Wc in GW (seconds, Wall/CPU): %f %f\n",
            profiler.get_wall_time_last("g0w0_build_spacetime_ct_ft_wc"),
            profiler.get_cpu_time_last("g0w0_build_spacetime_ct_ft_wc"));

    RI::GW<int, int, 3, double> gw_libri;
    RI::GW<int, int, 3, cplxdb> gw_libri_cplx;

    std::map<int,std::array<double,3>> atoms_pos;
    // Dummy atoms position
    for (int i = 0; i != natom; i++)
        atoms_pos.insert(std::pair<int, std::array<double, 3>>{i, {0, 0, 0}});

    global::profiler.start("g0w0_build_spacetime_2", "Setup LibRI G0W0 object and C data");
    if (use_complex_tensor)
    {
        gw_libri_cplx.set_parallel(comm_h.comm, atoms_pos, pbc.latvec_array, pbc.period_array);
        ztensor_map data_libri;
        for (const auto &I_JR_C : LRI_Cs.data_libri)
        {
            const auto I = I_JR_C.first;
            for (const auto &JR_C : I_JR_C.second)
            {
                const auto J = JR_C.first.first;
                const auto R = JR_C.first.second;
                const auto &C = JR_C.second;
                auto JR = std::pair<int, std::array<int, 3>>(J, R);
                data_libri[I][JR] = RI::Global_Func::convert<cplxdb>(C);
            }
        }
        gw_libri_cplx.set_Cs(data_libri, this->libri_threshold_C);
    }
    else
    {
        gw_libri.set_parallel(comm_h.comm, atoms_pos, pbc.latvec_array, pbc.period_array);
        gw_libri.set_Cs(LRI_Cs.data_libri, this->libri_threshold_C);
    }

    const auto& symmetry_ctx = this->symmetry_context;
    const bool use_input_sigc_symmetry =
        this->use_symmetry_context
        && symmetry_ctx.available
        && symmetry_ctx.has_ao_shell_layout()
        && !symmetry_ctx.irreducible_sector.empty()
        && !symmetry_ctx.rspace_operations.empty()
        && symmetry_ctx.atom_to_type.size() == static_cast<std::size_t>(natom)
        && symmetry_ctx.input_coord_frac.size() == static_cast<std::size_t>(natom);
    const auto libri_sigc_irreducible_sector =
        use_input_sigc_symmetry
            ? convert_input_symmetry_irreducible_sector_to_libri_gw(
                  symmetry_ctx.irreducible_sector, this->pbc.period_array)
            : std::map<std::pair<int, int>, std::set<std::array<int, 3>>>{};
    const bool restore_input_sigc_output = use_input_sigc_symmetry;
    librpa_int::input_symmetry_rspace_sector_stars_t input_symmetry_sector_stars;
    if (use_input_sigc_symmetry)
    {
        librpa_int::build_input_symmetry_rspace_sector_stars(
            symmetry_ctx, symmetry_ctx.input_coord_frac, this->pbc.period, pbc.Rlist,
            input_symmetry_sector_stars, nullptr);
        gw_libri.set_symmetry(false, {});
        gw_libri_cplx.set_symmetry(false, {});
        if (use_complex_tensor)
        {
            global::lib_printf(
                "Reducing GW real-space self-energy outputs with ABACUS irreducible sectors\n");
            gw_libri_cplx.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_GW_Symmetry<int, std::array<int, 3>, cplxdb>>(
                    gw_libri_cplx.lri.period, libri_sigc_irreducible_sector);
        }
        else
        {
            global::lib_printf(
                "Reducing GW real-space self-energy outputs with ABACUS irreducible sectors\n");
            gw_libri.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_GW_Symmetry<int, std::array<int, 3>, double>>(
                    gw_libri.lri.period, libri_sigc_irreducible_sector);
        }
    }
    else
    {
        gw_libri.set_symmetry(false, {});
        gw_libri_cplx.set_symmetry(false, {});
    }
    profiler.stop("g0w0_build_spacetime_2");
    lib_printf_root("Time for LibRI G0W0 setup (seconds, Wall/CPU): %f %f\n",
            profiler.get_wall_time_last("g0w0_build_spacetime_2"),
            profiler.get_cpu_time_last("g0w0_build_spacetime_2"));

    const auto tot_atpair_ordered = generate_atom_pair_from_nat(natom, true);
    const auto &Rlist = this->pbc.Rlist;
    auto IJR_local_gf = dispatch_vector_prod(tot_atpair_ordered, Rlist, ad_Wc.myid(), ad_Wc.nprocs(), true, false);
    ofs_myid << "#IJR_local_gf: " << IJR_local_gf.size() << std::endl;

    ArrayDesc desc_gf;
    IndexScheduler sched_gf;
    std::vector<Vector3_Order<int>> Rs_gf;
    if (this->is_mf_eigvec_k_distributed_)
    {
        profiler.start("g0w0_build_spacetime_prepare_gf_index");
        const int n_basis_ao = mf.get_n_aos();
        desc_gf = kblacs_ctxt.create_array_desc(n_basis_ao, n_basis_ao);
        const auto map_gf_atpairs_balanced =
            get_balanced_ap_distribution_for_consec_descriptor(atbasis_wfc, atbasis_wfc, desc_gf);
        sched_gf.init(map_gf_atpairs_balanced, atbasis_wfc, atbasis_wfc, desc_gf, false);
        const auto iRs = dispatcher_balanced(0, Rlist.size(), kblacs_ctxt.kpoints_local().size(),
                                             true, kblacs_ctxt.comm_kpoint_h.comm);
        Rs_gf.reserve(iRs.size());
        for (const auto &iR: iRs)
            Rs_gf.push_back(Rlist[iR]);
        profiler.stop("g0w0_build_spacetime_prepare_gf_index");
    }

    const int nfreq = tfg.get_n_grids();

    for (auto itau = 0; itau != nfreq; itau++)
    {
        // librpa_int::global::lib_printf("task %d itau %d start\n", mpi_comm_global_h.myid, itau);
        const auto tau = tfg.get_time_nodes()[itau];
        // librpa_int::global::lib_printf("task %d Wc_tau_R.count(tau) %zu\n", mpi_comm_global_h.myid, Wc_tau_R.count(tau));
        profiler.start("g0w0_build_spacetime_3", "Setup LibRI Wc");
        size_t n_obj_wc_libri = 0;
        if (use_complex_tensor)
        {
            auto it = tau_Wc_libri_cplx.find(tau);
            const auto &Wc_libri = (it == tau_Wc_libri_cplx.end())? ztensor_map{} : it->second;
            for (const auto &w: Wc_libri)
            {
                n_obj_wc_libri += w.second.size();
            }
            gw_libri_cplx.set_Ws(Wc_libri, this->libri_threshold_Wc);
            if (it != tau_Wc_libri_cplx.end()) tau_Wc_libri_cplx.erase(it);
        }
        else
        {
            auto it = tau_Wc_libri.find(tau);
            const auto &Wc_libri = (it == tau_Wc_libri.end())? dtensor_map{} : it->second;
            for (const auto &w: Wc_libri)
            {
                n_obj_wc_libri += w.second.size();
            }
            gw_libri.set_Ws(Wc_libri, this->libri_threshold_Wc);
            if (it != tau_Wc_libri.end()) tau_Wc_libri.erase(it);
        }
        release_free_mem();
        profiler.stop("g0w0_build_spacetime_3");

        const int n_spinor = mf.get_n_spinor();

        for (int ispin = 0; ispin != mf.get_n_spins(); ispin++)
        {
            for (int ispinor_bra = 0; ispinor_bra < n_spinor; ispinor_bra++)
            {
                for (int ispinor_ket = 0; ispinor_ket < n_spinor; ispinor_ket++)
                {
                    dtensor_map sigc_posi_tau, sigc_nega_tau;
                    ztensor_map sigc_posi_tau_cplx, sigc_nega_tau_cplx;

                    profiler.start("g0w0_build_spacetime_4", "Compute G(R,t) and G(R,-t)");
                    // global::ofs_myid << "gf size " << gf.size() << endl;
                    // global::ofs_myid << "t " << gf[tau].size() << " ; -t " << gf[-tau].size() << endl;
                    std::map<double, dtensor_map> tau_gf_libri;
                    std::map<double, ztensor_map> tau_gf_libri_cplx;
                    const std::vector<double> taus{tau, -tau};
                    if (use_complex_tensor)
                    {
                        if (this->is_mf_eigvec_k_distributed_)
                        {
                            build_gf_libri_kblacs_para(
                                mf, kblacs_ctxt, desc_wfc, desc_gf, sched_gf, atbasis_wfc,
                                ispin, ispinor_bra, ispinor_ket, this->pbc.kfrac_list,
                                taus, Rs_gf, tau_gf_libri_cplx);
                        }
                        else
                        {
                            build_gf_libri_kserial(mf, atbasis_wfc, ispin, ispinor_bra, ispinor_ket, this->pbc,
                                                this->symmetry_context,
                                                this->use_symmetry_context,
                                                this->pbc.kfrac_list,
                                                taus, IJR_local_gf, tau_gf_libri_cplx);
                        }
                    }
                    else
                    {
                        if (this->is_mf_eigvec_k_distributed_)
                            build_gf_libri_kblacs_para(
                                mf, kblacs_ctxt, desc_wfc, desc_gf, sched_gf, atbasis_wfc,
                                ispin, ispinor_bra, ispinor_ket, this->pbc.kfrac_list,
                                taus, Rs_gf, tau_gf_libri);
                        else
                            build_gf_libri_kserial(mf, atbasis_wfc, ispin, ispinor_bra, ispinor_ket, this->pbc,
                                                this->symmetry_context,
                                                this->use_symmetry_context,
                                                this->pbc.kfrac_list,
                                                taus, IJR_local_gf, tau_gf_libri);
                    }
                    // if (itau == 0 && ispin == 0)  // debug
                    // {
                    //     global::ofs_myid << "tau_gf_libri itau=0 ispin=0" << std::endl;
                    //     global::ofs_myid << tau_gf_libri << std::endl;
                    // }
                    profiler.stop("g0w0_build_spacetime_4");

                    if (n_spinor > 1)
                    {
                        lib_printf_root("Time for Green's function, i_spin %d i_spinor_bra %d i_spinor_ket %d i_tau %d (seconds, Wall/CPU): %f %f\n",
                                ispin + 1, ispinor_bra + 1, ispinor_ket + 1, itau + 1,
                                profiler.get_wall_time_last("g0w0_build_spacetime_4"),
                                profiler.get_cpu_time_last("g0w0_build_spacetime_4"));
                    }
                    else
                    {
                        lib_printf_root("Time for Green's function, i_spin %d i_tau %d (seconds, Wall/CPU): %f %f\n",
                                ispin + 1, itau + 1,
                                profiler.get_wall_time_last("g0w0_build_spacetime_4"),
                                profiler.get_cpu_time_last("g0w0_build_spacetime_4"));
                    }

                    for (auto t: taus)
                    {
                        size_t n_obj_gf_libri = 0;
                        double wtime_g0w0_cal_sigc = omp_get_wtime();
                        if (use_complex_tensor)
                        {
                            const auto &gf_libri = tau_gf_libri_cplx.at(t);
                            for (const auto &gf: gf_libri)
                                n_obj_gf_libri += gf.second.size();

                            gw_libri_cplx.set_Gs(gf_libri, this->libri_threshold_G);
                            global::profiler.start("g0w0_build_spacetime_5", "Call libRI cal_Sigc");
                            gw_libri_cplx.cal_Sigmas();
                            if (restore_input_sigc_output)
                            {
                                gw_libri_cplx.Sigmas =
                                    restore_input_symmetry_ao_rspace_tensor_map_gw(
                                        gw_libri_cplx.Sigmas, symmetry_ctx,
                                        input_symmetry_sector_stars, this->atbasis_wfc);
                            }
                            release_free_mem();
                            global::profiler.stop("g0w0_build_spacetime_5");
                            global::profiler.start("g0w0_build_spacetime_5_clean");
                            gw_libri_cplx.free_Gs();
                            release_free_mem();
                            global::profiler.stop("g0w0_build_spacetime_5_clean");

                            // Check size of data
                            double mem_mb = get_tensor_map_bytes(gw_libri_cplx.Sigmas) * 1e-6;
                            global::ofs_myid << "Temporary Sigc_tau size for time " << t << " [MB]: " << mem_mb << std::endl;

                            if (t > 0)
                                sigc_posi_tau_cplx = std::move(gw_libri_cplx.Sigmas);
                            else
                                sigc_nega_tau_cplx = std::move(gw_libri_cplx.Sigmas);
                            gw_libri_cplx.Sigmas.clear();
                        }
                        else
                        {
                            // ofs_myid << tau_gf_libri << std::endl;
                            const auto &gf_libri = tau_gf_libri.at(t);
                            for (const auto &gf: gf_libri)
                                n_obj_gf_libri += gf.second.size();
                            // if (t > 0)  // debug
                            // {
                            //     global::ofs_myid << "gf_libri posi ispin=0 itau " << itau << " " << get_num_keys(gf_libri)  << std::endl;
                            //     print_keys(global::ofs_myid, gf_libri);
                            //     // global::ofs_myid << gf_libri << std::endl;
                            //     for (const auto &[I, JR_gf]: gf_libri)
                            //     {
                            //         for (const auto &[JR, gf]: JR_gf)
                            //         {
                            //             std::stringstream ss;
                            //             Vector3_Order<int> R(JR.second[0], JR.second[1], JR.second[2]);
                            //             ss << "gf_libri_posi"
                            //                 << "_itau_" << std::setfill('0') << std::setw(5) << itau
                            //                 << "_I_" << std::setfill('0') << std::setw(5) << I
                            //                 << "_J_" << std::setfill('0') << std::setw(5) << JR.first
                            //                 << "_iR_" << std::setfill('0') << std::setw(5) << get_R_index(pbc.Rlist, R) << ".dat";
                            //             librpa_int::global::ofs_myid << "Writing GF to " << ss.str() << " " << tau << " " << I << " " << JR.second << " " << R << std::endl;
                            //             std::ofstream ofs(ss.str());
                            //             ofs << gf << std::endl;
                            //             ofs.close();
                            //         }
                            //     }
                            // }
                            // else
                            // {
                            //     global::ofs_myid << "gf_libri nega ispin=0 itau " << itau << " " << get_num_keys(gf_libri) << std::endl;
                            //     print_keys(global::ofs_myid, gf_libri);
                            //     // global::ofs_myid << gf_libri << std::endl;
                            //     for (const auto &[I, JR_gf]: gf_libri)
                            //     {
                            //         for (const auto &[JR, gf]: JR_gf)
                            //         {
                            //             std::stringstream ss;
                            //             Vector3_Order<int> R(JR.second[0], JR.second[1], JR.second[2]);
                            //             ss << "gf_libri_nega"
                            //                 << "_itau_" << std::setfill('0') << std::setw(5) << itau
                            //                 << "_I_" << std::setfill('0') << std::setw(5) << I
                            //                 << "_J_" << std::setfill('0') << std::setw(5) << JR.first
                            //                 << "_iR_" << std::setfill('0') << std::setw(5) << get_R_index(pbc.Rlist, R) << ".dat";
                            //             librpa_int::global::ofs_myid << "Writing GF to " << ss.str() << " " << tau << " " << I << " " << JR.second << " " << R << std::endl;
                            //             std::ofstream ofs(ss.str());
                            //             ofs << gf << std::endl;
                            //             ofs.close();
                            //         }
                            //     }
                            // }
	                            gw_libri.set_Gs(gf_libri, this->libri_threshold_G);
	                            global::profiler.start("g0w0_build_spacetime_5", "Call libRI cal_Sigc");
	                            gw_libri.cal_Sigmas();
	                            if (restore_input_sigc_output)
	                            {
	                                gw_libri.Sigmas =
	                                    restore_input_symmetry_ao_rspace_tensor_map_gw(
	                                        gw_libri.Sigmas, symmetry_ctx,
	                                        input_symmetry_sector_stars, this->atbasis_wfc);
	                            }
	                            global::profiler.stop("g0w0_build_spacetime_5");
                            global::profiler.start("g0w0_build_spacetime_5_clean");
                            gw_libri.free_Gs();
                            global::profiler.stop("g0w0_build_spacetime_5_clean");

                            // Check size of data
                            double mem_mb = get_tensor_map_bytes(gw_libri.Sigmas) * 1e-6;
                            global::ofs_myid << "Temporary Sigc_tau size for time " << t << " [MB]: " << mem_mb << std::endl;

                            if (t > 0)
                                sigc_posi_tau = std::move(gw_libri.Sigmas);
                            else
                                sigc_nega_tau = std::move(gw_libri.Sigmas);
                            gw_libri.Sigmas.clear();
                            // if (itau == 0 && ispin == 0)
                            // {
                            //     if (t > 0)
                            //     {
                            //         global::ofs_myid << "sigc_posi_tau itau=0 ispin=0 " << get_num_keys(sigc_posi_tau) << std::endl;
                            //         print_keys(global::ofs_myid, sigc_posi_tau);
                            //         global::ofs_myid << sigc_posi_tau << std::endl;
                            //     }
                            //     if (t < 0)
                            //     {
                            //         global::ofs_myid << "sigc_nega_tau itau=0 ispin=0 " << get_num_keys(sigc_nega_tau) << std::endl;
                            //         print_keys(global::ofs_myid, sigc_nega_tau);
                            //         global::ofs_myid << sigc_nega_tau << std::endl;
                            //     }
                            // }
                        }
                        wtime_g0w0_cal_sigc = omp_get_wtime() - wtime_g0w0_cal_sigc;
                        if (n_spinor > 1)
                            librpa_int::global::lib_printf(
                                "Task %4d. libRI G0W0, spin %1d, bra %1d, ket %1d, time grid %12.6f. Wc size %zu, GF "
                                "size %zu. Wall time %f\n",
                                comm_h.myid, ispin, ispinor_bra, ispinor_ket, t, n_obj_wc_libri, n_obj_gf_libri,
                                wtime_g0w0_cal_sigc);
                        else
                            librpa_int::global::lib_printf(
                                "Task %4d. libRI G0W0, spin %1d, time grid %12.6f. Wc size %zu, GF "
                                "size %zu. Wall time %f\n",
                                comm_h.myid, ispin, t, n_obj_wc_libri, n_obj_gf_libri,
                                wtime_g0w0_cal_sigc);
                    }

                    size_t n_IJR_myid = 0; // for sigcmat output

                    if (this->output_sigc_mat_rt)
                    {
                        std::stringstream ss;
                        ss << path_as_directory(this->output_dir) << "SigcRT"
                            << "_ispin_" << std::setfill('0') << std::setw(5) << ispin;
                        if (n_spinor > 1)
                        {
                           ss << "_spinor_" << ispinor_bra << "_" << ispinor_ket;
                        }
                        ss << "_itau_" << std::setfill('0') << std::setw(5) << itau
                           << "_myid_" << std::setfill('0') << std::setw(5) << global::myid_global << ".dat";
                        ofs_sigmac_r.open(ss.str(), std::ios::out | std::ios::binary);
                        ofs_sigmac_r.write((char *) &n_IJR_myid, sizeof(size_t)); // placeholder
                    }

                    auto accumulate_sigc_tau_to_freq = [&](const auto &sigc_posi_tau_in,
                                                           const auto &sigc_nega_tau_in,
                                                           auto block_scale_factor,
                                                           auto make_freq_value, size_t elem_size)
                    {
                        for (const auto &[I, J_R_sigc_posi] : sigc_posi_tau_in)
                        {
                            const auto n_I = this->atbasis_wfc.get_atom_nb(I);

                            auto it_nega_I = sigc_nega_tau_in.find(I);
                            if (it_nega_I == sigc_nega_tau_in.cend()) continue;

                            for (const auto &[JR, sigc_posi_block] : J_R_sigc_posi)
                            {
                                const auto J = JR.first;
                                const auto n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;

                                auto it_nega_JR = it_nega_I->second.find(JR);
                                if (it_nega_JR == it_nega_I->second.cend()) continue;

                                const auto &sigc_nega_block = it_nega_JR->second;

                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};

                                const auto it_R = std::find(Rlist.cbegin(), Rlist.cend(), R);
                                const auto iR = std::distance(Rlist.cbegin(), it_R);

                                const auto sigc_cos = block_scale_factor * (sigc_posi_block + sigc_nega_block);
                                const auto sigc_sin = block_scale_factor * (sigc_posi_block - sigc_nega_block);

                                ++n_IJR_myid;

                                if (this->output_sigc_mat_rt)
                                {
                                    size_t dims[5];
                                    dims[0] = as_size(iR);
                                    dims[1] = as_size(I);
                                    dims[2] = as_size(J);
                                    dims[3] = as_size(n_I);
                                    dims[4] = as_size(n_J);

                                    ofs_sigmac_r.write(reinterpret_cast<const char *>(dims),
                                                       5 * sizeof(size_t));

                                    ofs_sigmac_r.write(
                                        reinterpret_cast<const char *>(sigc_cos.ptr()),
                                        n_I * n_J * elem_size);

                                    ofs_sigmac_r.write(
                                        reinterpret_cast<const char *>(sigc_sin.ptr()),
                                        n_I * n_J * elem_size);
                                }

                                for (size_t iomega = 0; iomega != tfg.get_n_grids(); ++iomega)
                                {
                                    const auto omega = tfg.get_freq_nodes()[iomega];
                                    const auto t2f_sin = tfg.get_sintrans_t2f()(iomega, itau);
                                    const auto t2f_cos = tfg.get_costrans_t2f()(iomega, itau);

                                    Matz sigc_temp(n_I, n_J, MAJOR::ROW);

                                    for (size_t i = 0; i != static_cast<size_t>(n_I); ++i)
                                    {
                                        for (size_t j = 0; j != static_cast<size_t>(n_J); ++j)
                                        {
                                            sigc_temp(i, j) = make_freq_value(
                                                sigc_cos(i, j), sigc_sin(i, j), t2f_cos, t2f_sin);
                                        }
                                    }

                                    const atpair_t IJ{I, J};
                                    auto &m_IJ =
                                        sigc_is_f_IJ_R[ispin][ispinor_bra][ispinor_ket][omega][IJ];

                                    auto it = m_IJ.find(R);
                                    if (it == m_IJ.cend())
                                    {
                                        m_IJ.emplace(R, std::move(sigc_temp));
                                    }
                                    else
                                    {
                                        it->second += sigc_temp;
                                    }
                                }
                            }
                        }
                    };

                    // symmetrize and perform transformation
                    global::profiler.start("g0w0_build_spacetime_6", "Transform Sigc (R,t) -> (R,w)");
                    if (use_complex_tensor)
                    {
                        accumulate_sigc_tau_to_freq(
                            sigc_posi_tau_cplx, sigc_nega_tau_cplx,
                            cplxdb(0.5, 0.0),
                            [](const cplxdb &sigc_cos, const cplxdb &sigc_sin, double t2f_cos,
                               double t2f_sin) -> cplxdb
                            {
                                return sigc_cos * t2f_cos + sigc_sin * cplxdb(0.0, t2f_sin);
                            },
                            sizeof(cplxdb));
                        sigc_posi_tau_cplx.clear();
                        sigc_nega_tau_cplx.clear();
                    }
                    else
                    {
                        accumulate_sigc_tau_to_freq(
                            sigc_posi_tau, sigc_nega_tau,
                            0.5,
                            [](double sigc_cos, double sigc_sin, double t2f_cos,
                               double t2f_sin) -> cplxdb
                            { return cplxdb{sigc_cos * t2f_cos, sigc_sin * t2f_sin}; },
                            sizeof(double));
                        sigc_posi_tau.clear();
                        sigc_nega_tau.clear();
                    }

                    global::profiler.stop("g0w0_build_spacetime_6");

                    if (this->output_sigc_mat_rt)
                    {
                        ofs_sigmac_r.seekp(0);
                        ofs_sigmac_r.write((char *) &n_IJR_myid, sizeof(size_t)); // overwrite
                        ofs_sigmac_r.close();
                    }
                }
            }
        }
        global::profiler.start("g0w0_build_spacetime_free_Ws");
        gw_libri.free_Ws();
        global::profiler.stop("g0w0_build_spacetime_free_Ws");
        // Release freed memory to OS, to resolve memory fragments in LibRI
        release_free_mem();
    }
    is_rspace_built_ = true;
#endif

    // Export real-space imaginary-frequency NAO sigma_c matrices
    if (is_rspace_built_ && this->output_sigc_mat_rf)
    {
        const int n_spinor = mf.get_n_spinor();
        for (int ispin = 0; ispin != mf.get_n_spins(); ispin++)
        {
            for (int ispinor_bra = 0; ispinor_bra < n_spinor; ispinor_bra++)
            {
                for (int ispinor_ket = 0; ispinor_ket < n_spinor; ispinor_ket++)
                {
                    for (size_t iomega = 0; iomega != tfg.get_n_grids(); iomega++)
                    {
                        size_t n_IJR_myid = 0;
                        std::stringstream ss;
                        ss << path_as_directory(this->output_dir) << "SigcRF"
                           << "_ispin_" << std::setfill('0') << std::setw(5) << ispin;
                        if (n_spinor > 1)
                        {
                           ss << "_spinor_" << ispinor_bra << "_" << ispinor_ket;
                        }
                        ss << "_iomega_" << std::setfill('0') << std::setw(5) << iomega
                           << "_myid_" << std::setfill('0') << std::setw(5) << global::myid_global << ".dat";
                        ofs_sigmac_r.open(ss.str(), std::ios::out | std::ios::binary);
                        ofs_sigmac_r.write((char *) &n_IJR_myid, sizeof(size_t)); // placeholder

                        const auto omega = tfg.get_freq_nodes()[iomega];
                        const auto &sigc_IJ_R = sigc_is_f_IJ_R[ispin][ispinor_bra][ispinor_ket][omega];
                        for (const auto &[IJ, R_sigc]: sigc_IJ_R)
                        {
                            const int I = IJ.first;
                            const int J = IJ.second;
                            for (const auto &[R, sigc]: R_sigc)
                            {
                                const auto iR = this->pbc.get_R_index(R);
                                const auto &n_I = this->atbasis_wfc.get_atom_nb(I);
                                const auto &n_J = this->atbasis_wfc.get_atom_nb(J);
                                n_IJR_myid++;
                                size_t dims[5];
                                dims[0] = iR;
                                dims[1] = I;
                                dims[2] = J;
                                dims[3] = n_I;
                                dims[4] = n_J;
                                assert (sigc.size() == n_I * n_J);
                                ofs_sigmac_r.write((char *) dims, 5 * sizeof(size_t));
                                ofs_sigmac_r.write((char *) sigc.ptr(), sigc.size() * sizeof(cplxdb));
                            }
                        }
                        ofs_sigmac_r.seekp(0);
                        ofs_sigmac_r.write((char *) &n_IJR_myid, sizeof(size_t)); // placeholder
                        ofs_sigmac_r.close();
                    }
                }
            }
        }
    }
}

void G0W0::build_sigc_matrix_KS(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                                const std::vector<Vector3_Order<double>> &kfrac_target,
                                const AtomPairBvKRemap<atom_t> &bvk_remap)
{
    throw LIBRPA_RUNTIME_ERROR("Not implemented yet");
}

void G0W0::build_sigc_matrix_KS_blacs(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                                      const std::vector<Vector3_Order<double>> &kfrac_target,
                                      const AtomPairBvKRemap<atom_t> &bvk_remap,
                                      const BlacsCtxtHandler &blacs_ctxt_h,
                                      const bool use_gpu_replace_scalapack)
{
    assert(blacs_ctxt_h.comm() == this->comm_h.comm);
    assert(this->is_rspace_built_);

    if (this->is_kspace_built_)
    {
        global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning: reset Sigmac_c k-space matrices\n");
        this->reset_kspace();
    }

    const int n_aos = mf.get_n_aos();
    const int n_bands = mf.get_n_bands();
    const int n_spins = mf.get_n_spins();
    const int n_spinor = mf.get_n_spinor();

    global::profiler.start("g0w0_build_sigc_KS");

#ifndef LIBRPA_USE_LIBRI
    if (global::mpi_comm_global_h.myid == 0)
    {
        std::cout << "LIBRA::G0W0::build_sigc_matrix_KS is only implemented on top of LibRI" << std::endl;
        std::cout << "Please recompile LibRPA with -DLIBRPA_USE_LIBRI and optionally configure include path" << std::endl;
    }
    global::mpi_comm_global_h.barrier();
    throw LIBRPA_RUNTIME_ERROR("G0W0 needs compilation with LibRI");
#else
    ArrayDesc desc_nband_nao(blacs_ctxt_h);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    ArrayDesc desc_nao_nao(blacs_ctxt_h);
    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    ArrayDesc desc_nband_nband(blacs_ctxt_h);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);

    int mb_opt = std::min(128, std::min(desc_nband_nao.mb(), desc_nband_nao.nb()));
    ArrayDesc desc_nband_nao_opt(blacs_ctxt_h), desc_nao_nao_opt(blacs_ctxt_h);
    ArrayDesc desc_nao_nband_opt(blacs_ctxt_h),  desc_nband_nband_opt(blacs_ctxt_h);
    desc_nband_nao_opt.init(n_bands, n_aos, mb_opt, mb_opt, 0, 0);
    desc_nao_nao_opt.init(n_aos, n_aos, mb_opt, mb_opt, 0, 0);
    desc_nband_nband_opt.init(n_bands, n_bands, mb_opt, mb_opt, 0, 0);
    desc_nao_nband_opt.init(n_aos, n_bands, mb_opt, mb_opt, 0, 0);

    auto wfc_bra_opt = init_local_mat<complex<double>>(desc_nao_nband_opt, MAJOR::COL);
    auto wfc_ket_opt = init_local_mat<complex<double>>(desc_nao_nband_opt, MAJOR::COL);
    auto sigc_nao_nao_opt = init_local_mat<complex<double>>(desc_nao_nao_opt, MAJOR::COL);
    auto sigc_nband_nband_opt = init_local_mat<complex<double>>(desc_nband_nband_opt, MAJOR::COL);

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    std::complex<double> *d_wfc_bra = nullptr, *d_wfc_ket = nullptr,
                         *d_sigc_nao = nullptr, *d_temp = nullptr,
                         *d_sigc_nband = nullptr;
    size_t size_wfc = 0, size_sigc_nao = 0, size_temp = 0, size_sigc_nband = 0;
    if (use_gpu_replace_scalapack)
    {
        auto &blacs_ctxt_h_nc = const_cast<BlacsCtxtHandler &>(blacs_ctxt_h);
        if (blacs_ctxt_h_nc.ddla_handle == nullptr)
            blacs_ctxt_h_nc.init_ddla_handle();
        desc_nao_nband_opt.set_ddla_desc(blacs_ctxt_h.ddla_handle);
        desc_nao_nao_opt.set_ddla_desc(blacs_ctxt_h.ddla_handle);
        desc_nband_nao_opt.set_ddla_desc(blacs_ctxt_h.ddla_handle);
        desc_nband_nband_opt.set_ddla_desc(blacs_ctxt_h.ddla_handle);

        size_wfc = static_cast<size_t>(desc_nao_nband_opt.m_loc()) *
                   desc_nao_nband_opt.n_loc();
        size_sigc_nao = static_cast<size_t>(desc_nao_nao_opt.m_loc()) *
                        desc_nao_nao_opt.n_loc();
        size_temp = static_cast<size_t>(desc_nband_nao_opt.m_loc()) *
                    desc_nband_nao_opt.n_loc();
        size_sigc_nband = static_cast<size_t>(desc_nband_nband_opt.m_loc()) *
                          desc_nband_nband_opt.n_loc();
        DEVICE_CHECK(deviceMallocAsync((void**)&d_wfc_bra, size_wfc * sizeof(std::complex<double>), blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_wfc_ket, size_wfc * sizeof(std::complex<double>), blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_sigc_nao, size_sigc_nao * sizeof(std::complex<double>), blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_temp, size_temp * sizeof(std::complex<double>), blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_sigc_nband, size_sigc_nband * sizeof(std::complex<double>), blacs_ctxt_h.ddla_handle->stream));
    }
#endif

    ArrayDesc desc_nao_nband_fb(blacs_ctxt_h);  // For k-parallel
    ArrayDesc desc_nao_nao_fb(blacs_ctxt_h);
    ArrayDesc desc_nband_nband_fb(blacs_ctxt_h);
    const int n_target_kpoints = static_cast<int>(kfrac_target.size());
    const bool use_root_dense_projection =
        this->use_symmetry_context
        && use_input_symmetry_ibz_root_projection(this->symmetry_context,
                                               this->pbc,
                                               n_target_kpoints,
                                               this->mf.get_n_kpoints());
    if (use_root_dense_projection)
    {
        desc_nao_nao_fb.init(n_aos, n_aos, n_aos, n_aos, 0, 0);
    }

    const auto set_IJ_nao_nao = get_necessary_IJ_from_block_2D(
        this->atbasis_wfc, this->atbasis_wfc, desc_nao_nao);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nao_nao);

    // Check Sigmac matrix distribution
    if (!is_rspace_redist_for_KS_)
    {
        global::profiler.start("g0w0_build_sigc_KS_rspace_redist");
        for (int isp = 0; isp < n_spins; isp++)
        {
            for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
            {
                for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
                {
                    auto sigc_orig = find_nested_int_map_3(sigc_is_f_IJ_R, isp, ispn_bra, ispn_ket);
                    for (const auto& freq: this->tfg.get_freq_nodes())
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>, Tensor<cplxdb>>> sigc_I_JR_local;
                        if (sigc_orig != nullptr)
                        {
                            auto it_sp_f = sigc_orig->find(freq);
                            if (it_sp_f != sigc_orig->cend())
                            {
                                for (const auto &IJ_R_sigc: it_sp_f->second)
                                {
                                    const auto IJ = IJ_R_sigc.first;
                                    const int I = IJ.first;
                                    const int J = IJ.second;
                                    const auto &n_I = this->atbasis_wfc.get_atom_nb(I);
                                    const auto &n_J = this->atbasis_wfc.get_atom_nb(J);
                                    for (const auto &[R, sigc]: IJ_R_sigc.second)
                                    {
                                        const std::array<int, 3> Ra{R.x, R.y, R.z};
                                        sigc_I_JR_local[I][{J, Ra}] = Tensor<complex<double>>({n_I, n_J}, sigc.sptr());
                                    }
                                }
                            }
                        }
                        // for certain spin and frequency
                        auto sigc_I_JR = comm_map2_first(blacs_ctxt_h.comm(), sigc_I_JR_local, s0_s1.first, s0_s1.second);
                        // NOTE: sigc_I_JR may be empty for some process. This is a corner case 
                        //       of very small basis and large MPI tasks. However, it must be stored
                        //       to preserve the [spin][spinor][spinor][freq] key structure,
                        //       otherwise it will lead to map error in the subsequent rotation.
                        sigc_I_JR_local.clear();
                        ap_p_map<std::map<Vector3_Order<int>, Matz>> sigc_new;
                        for (const auto &[I, JRmat]: sigc_I_JR)
                        {
                            const int n_I = this->atbasis_wfc.get_atom_nb(I);
                            for (const auto &[JR, mat]: JRmat)
                            {
                                const atom_t J = JR.first;
                                atpair_t IJ{I, J};
                                const int n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;
                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};
                                sigc_new[IJ][R] = Matz{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        if (sigc_orig == nullptr)
                            sigc_is_f_IJ_R[isp][ispn_bra][ispn_ket][freq] = std::move(sigc_new);
                        else
                        {
                            auto it_sp_f = sigc_orig->find(freq);
                            if (it_sp_f == sigc_orig->cend())
                                sigc_orig->emplace(freq, std::move(sigc_new));
                            else
                                it_sp_f->second.swap(sigc_new);
                        }
                        sigc_I_JR.clear();
                    }
                }
            }
        }

        is_rspace_redist_for_KS_ = true;
        is_rspace_redist_blacs_ = true;
        global::profiler.stop("g0w0_build_sigc_KS_rspace_redist");
    }
    else
    {
        if (!is_rspace_redist_blacs_)
            throw LIBRPA_RUNTIME_ERROR("has redistributed for non-BLACS");
    }

    sigc_is_ik_f_KS.clear();

    // local 2D-block submatrices
    auto sigc_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto sigc_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);

    for (int isp = 0; isp < n_spins; isp++)
    {
	// Initialize, make sure the map on every access has the isp key
        this->sigc_is_ik_f_KS[isp] = {};

        for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
        {
            for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
            {
                std::map<double, std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matz>>>> sigc_isp_local;
                global::profiler.start("g0w0_build_sigc_KS_find_bvk");
                for (const auto& freq: this->tfg.get_freq_nodes())
                {
                    const auto &sigc_IJ_R = sigc_is_f_IJ_R.at(isp).at(ispn_bra).at(ispn_ket).at(freq);
                    sigc_isp_local[freq] = {};
                    auto add_sigc = [](auto &R_sigc_shift, const Vector3_Order<int> &R_bvk,
                                       const auto &sigc, const double weight)
                    {
                        auto sigc_weighted = sigc.copy();
                        sigc_weighted *= weight;
                        auto [it, inserted] = R_sigc_shift.emplace(R_bvk, sigc_weighted);
                        if (!inserted) it->second += sigc_weighted;
                    };

                    // Convert each <I,<J, R>> pair to the configured BvK counterpart
                    // to speed up later Fourier transform while keeping band interpolation accurate.
                    // A missing remap entry means that R is already the desired counterpart.
                    // NOTE: from redistribution, sigc_is_f_IJ_R is ensured to have all [spin][freq] keys,
                    // but each value (atom-pair map) can be empty.
                    for (const auto &[IJ, R_sigc]: sigc_IJ_R)
                    {
                        const auto &I = IJ.first;
                        const auto &J = IJ.second;
                        auto &R_sigc_shift = sigc_isp_local[freq][I][J];
                        for (auto &[R, sigc]: R_sigc)
                        {
                            const auto *R_bvks = bvk_remap.find_R_bvk(IJ, R);
                            if (R_bvks == nullptr || R_bvks->empty())
                            {
                                R_sigc_shift[R] = sigc;
                            }
                            else if (R_bvks->size() == 1)
                            {
                                R_sigc_shift[R_bvks->front()] = sigc;
                            }
                            else
                            {
                                const auto weight = 1.0 / static_cast<double>(R_bvks->size());
                                for (const auto &R_bvk: *R_bvks)
                                {
                                    add_sigc(R_sigc_shift, R_bvk, sigc, weight);
                                }
                            }
                        }
                    }
                }
                global::profiler.stop("g0w0_build_sigc_KS_find_bvk");

                if (is_mf_eigvec_k_distributed_)
                {
                    global::profiler.start("g0w0_build_sigc_KS_rotate_kpara");
                    auto temp_nband_nao_opt = init_local_mat<complex<double>>(desc_nband_nao_opt, MAJOR::COL);
                    // collect parsed eigenvectors all all processes
                    std::vector<int> nks_all(comm_h.nprocs, 0);
                    std::vector<int> iks_local;
                    auto wfc_sp = find_nested_int_map_2(wfc_target, isp, ispn_bra);
                    if (wfc_sp != nullptr)
                    {
                        nks_all[comm_h.myid] = wfc_sp->size();
                        for (const auto &[ik, _]: *wfc_sp)
                        {
                            iks_local.emplace_back(ik);
                        }
                    }
                    MPI_Allreduce(MPI_IN_PLACE, nks_all.data(), comm_h.nprocs, mpi_datatype<int>::value, MPI_SUM, comm_h.comm);
                    const int nk_max = *std::max_element(nks_all.cbegin(), nks_all.cend());
                    std::vector<int> iks_all(nk_max * comm_h.nprocs, 0);
                    for (int i = 0; i < nks_all[comm_h.myid]; i++)
                    {
                        iks_all[comm_h.myid * nk_max + i] = iks_local[i];
                    }
                    MPI_Allreduce(MPI_IN_PLACE, iks_all.data(), comm_h.nprocs * nk_max, mpi_datatype<int>::value, MPI_SUM, comm_h.comm);
                    for (int pid = 0; pid < comm_h.nprocs; pid++)
                    {
                        const int nk_this = nks_all[pid];
                        if (nk_this == 0) continue;  // no eigenvector on this process
                        auto [irsrc, icsrc] = blacs_ctxt_h.get_pcoord(pid);
                        desc_nao_nband_fb.init(n_aos, n_bands, n_aos, n_bands, irsrc, icsrc);
                        desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, irsrc, icsrc);
                        release_free_mem();
                        auto sigc_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                        for (int ik_this = 0; ik_this < nk_this; ik_this++)
                        {
                            const int ik = iks_all[pid * nk_max + ik_this];
                            const auto& kfrac = kfrac_target[ik];
                            const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                                fourier = [kfrac](const atom_t I, const atom_t J, const Vector3_Order<int> &R)
                                {
                                    const auto ang = (kfrac * R) * TWO_PI;
                                    return complex<double>{std::cos(ang), std::sin(ang)};
                                };
                            std::vector<complex<double>> dummy(1);
                            for (const auto& freq: this->tfg.get_freq_nodes())
                            {
                                sigc_nao_nao.zero_out();
                                sigc_nband_nband_fb.zero_out();
                                collect_block_from_IJ_storage_matrix_transform(sigc_nao_nao, desc_nao_nao,
                                        this->atbasis_wfc, this->atbasis_wfc, fourier, sigc_isp_local.at(freq));
                                ScalapackConnector::pgemr2d_f(n_aos, n_aos, sigc_nao_nao.ptr(), 1, 1,
                                                              desc_nao_nao.desc, sigc_nao_nao_opt.ptr(),
                                                              1, 1, desc_nao_nao_opt.desc, blacs_ctxt_h.ictxt);
                                if (pid == comm_h.myid)
                                {
                                    const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                                    const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                                    // redistribute the k-point eigenvector at this processes
                                    ScalapackConnector::pgemr2d_f(n_aos, n_bands, wfc_bra.c, 1, 1, 
                                                                  desc_nao_nband_fb.desc, wfc_bra_opt.ptr(),
                                                                  1, 1, desc_nao_nband_opt.desc, blacs_ctxt_h.ictxt);
                                    ScalapackConnector::pgemr2d_f(n_aos, n_bands, wfc_ket.c, 1, 1, 
                                                                  desc_nao_nband_fb.desc, wfc_ket_opt.ptr(),
                                                                  1, 1, desc_nao_nband_opt.desc, blacs_ctxt_h.ictxt);
                                }
                                else
                                {
                                    // redistribute the k-point eigenvector at other processes
                                    ScalapackConnector::pgemr2d_f(n_aos, n_bands, dummy.data(), 1, 1, 
                                                                  desc_nao_nband_fb.desc, wfc_bra_opt.ptr(),
                                                                  1, 1, desc_nao_nband_opt.desc, blacs_ctxt_h.ictxt);
                                    ScalapackConnector::pgemr2d_f(n_aos, n_bands, dummy.data(), 1, 1, 
                                                                  desc_nao_nband_fb.desc, wfc_ket_opt.ptr(),
                                                                  1, 1, desc_nao_nband_opt.desc, blacs_ctxt_h.ictxt);
                                }
                                release_free_mem();
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
                                if (use_gpu_replace_scalapack)
                                {
                                    DEVICE_CHECK(deviceMemcpyAsync(d_wfc_bra, wfc_bra_opt.ptr(), size_wfc * sizeof(std::complex<double>),
                                                                   deviceMemcpyHostToDevice, blacs_ctxt_h.ddla_handle->stream));
                                    DEVICE_CHECK(deviceMemcpyAsync(d_wfc_ket, wfc_ket_opt.ptr(), size_wfc * sizeof(std::complex<double>),
                                                                   deviceMemcpyHostToDevice, blacs_ctxt_h.ddla_handle->stream));
                                    DEVICE_CHECK(deviceMemcpyAsync(d_sigc_nao, sigc_nao_nao_opt.ptr(), size_sigc_nao * sizeof(std::complex<double>),
                                                                   deviceMemcpyHostToDevice, blacs_ctxt_h.ddla_handle->stream));
                                    LaConnector::pgemm(
                                        'C', 'N', n_bands, n_aos, n_aos, std::complex<double>{1.0, 0.0},
                                        d_wfc_bra, 1, 1, desc_nao_nband_opt,
                                        d_sigc_nao, 1, 1, desc_nao_nao_opt,
                                        std::complex<double>{0.0, 0.0},
                                        d_temp, 1, 1, desc_nband_nao_opt);
                                    LaConnector::pgemm(
                                        'N', 'N', n_bands, n_bands, n_aos, std::complex<double>{1.0, 0.0},
                                        d_temp, 1, 1, desc_nband_nao_opt,
                                        d_wfc_ket, 1, 1, desc_nao_nband_opt,
                                        std::complex<double>{0.0, 0.0},
                                        d_sigc_nband, 1, 1, desc_nband_nband_opt);
                                    DEVICE_CHECK(deviceMemcpyAsync(sigc_nband_nband_opt.ptr(), d_sigc_nband,
                                                                   size_sigc_nband * sizeof(std::complex<double>),
                                                                   deviceMemcpyDeviceToHost, blacs_ctxt_h.ddla_handle->stream));
                                    DEVICE_CHECK(deviceStreamSynchronize(blacs_ctxt_h.ddla_handle->stream));
                                }
                                else
#endif
                                {
                                    ScalapackConnector::pgemm_f('C', 'N', n_bands, n_aos, n_aos, 1.0,
                                                                wfc_bra_opt.ptr(), 1, 1, desc_nao_nband_opt.desc,
                                                                sigc_nao_nao_opt.ptr(), 1, 1, desc_nao_nao_opt.desc,
                                                                0.0,
                                                                temp_nband_nao_opt.ptr(), 1, 1, desc_nband_nao_opt.desc);
                                    ScalapackConnector::pgemm_f('N', 'N', n_bands, n_bands, n_aos, 1.0,
                                                                temp_nband_nao_opt.ptr(), 1, 1, desc_nband_nao_opt.desc,
                                                                wfc_ket_opt.ptr(), 1, 1, desc_nao_nband_opt.desc,
                                                                0.0,
                                                                sigc_nband_nband_opt.ptr(), 1, 1, desc_nband_nband_opt.desc);
                                }
                                ScalapackConnector::pgemr2d_f(n_aos, n_aos, sigc_nband_nband_opt.ptr(), 1, 1,
                                                              desc_nband_nband_opt.desc, sigc_nband_nband_fb.ptr(),
                                                              1, 1, desc_nband_nband_fb.desc, blacs_ctxt_h.ictxt);
                                if (pid == comm_h.myid){
                                    auto sigc_freq = find_nested_int_map_2(this->sigc_is_ik_f_KS, isp, ik);
                                    if (sigc_freq == nullptr || sigc_freq->count(freq) == 0)
                                        this->sigc_is_ik_f_KS[isp][ik][freq] = Matz(n_bands, n_bands, MAJOR::COL);
                                    this->sigc_is_ik_f_KS[isp][ik][freq] += sigc_nband_nband_fb;
                                }
                                release_free_mem();
                            }
                        }
                    }
                    global::profiler.stop("g0w0_build_sigc_KS_rotate_kpara");
                }
                else
                {
                    global::profiler.start("g0w0_build_sigc_KS_rotate_kserial");
                    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);
                    auto sigc_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                    for (const auto& freq: this->tfg.get_freq_nodes())
                    {
                        // Perform Fourier transform and rotate
                        for (size_t ik = 0; ik < kfrac_target.size(); ik++)
                        {
                            const auto kfrac = kfrac_target[ik];
                            // const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                            const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                                fourier = [kfrac](const atom_t I, const atom_t J, const Vector3_Order<int> &R)
                                {
                                    const auto ang = (kfrac * R) * TWO_PI;
                                    return complex<double>{std::cos(ang), std::sin(ang)};
                                };

                            sigc_nao_nao.zero_out();
                            sigc_nband_nband_fb.zero_out();
                            collect_block_from_IJ_storage_matrix_transform(
                                sigc_nao_nao, desc_nao_nao,
                                this->atbasis_wfc, this->atbasis_wfc,
                                fourier, sigc_isp_local.at(freq));
                            if (use_root_dense_projection)
                            {
                                auto sigc_nao_nao_fb =
                                    init_local_mat<complex<double>>(desc_nao_nao_fb, MAJOR::COL);
                                ScalapackConnector::pgemr2d_f(
                                    n_aos, n_aos,
                                    sigc_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                    sigc_nao_nao_fb.ptr(), 1, 1, desc_nao_nao_fb.desc,
                                    desc_nao_nao_fb.ictxt());

                                ComplexMatrix sigc_nband_nband_dense;
                                if (comm_h.is_root())
                                {
                                    ComplexMatrix sigc_nao_nao_dense(n_aos, n_aos);
                                    for (int iao = 0; iao != n_aos; ++iao)
                                    {
                                        for (int jao = 0; jao != n_aos; ++jao)
                                        {
                                            sigc_nao_nao_dense(iao, jao) =
                                                sigc_nao_nao_fb(iao, jao);
                                        }
                                    }
                                    const auto &wfc_bra =
                                        wfc_target.at(isp).at(ispn_bra).at(ik);
                                    const auto &wfc_ket =
                                        wfc_target.at(isp).at(ispn_ket).at(ik);
                                    sigc_nband_nband_dense =
                                        conj(wfc_bra) * sigc_nao_nao_dense
                                        * transpose(wfc_ket, false);
                                }
                                broadcast_ComplexMatrix(
                                    sigc_nband_nband_dense, 0, comm_h.comm);
                                auto sigc_freq =
                                    find_nested_int_map_2(this->sigc_is_ik_f_KS, isp, ik);
                                if (sigc_freq == nullptr || sigc_freq->count(freq) == 0)
                                {
                                    this->sigc_is_ik_f_KS[isp][ik][freq] = Matz(
                                        n_bands, n_bands, sigc_nband_nband_dense.c,
                                        MAJOR::ROW, MAJOR::COL);
                                }
                                else
                                {
                                    sigc_freq->at(freq) += Matz(
                                        n_bands, n_bands, sigc_nband_nband_dense.c,
                                        MAJOR::ROW, MAJOR::COL);
                                }
                                continue;
                            }
                            // prepare wave function BLACS
                            // FIXME: check if bra and ket are correct
                            const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                            const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                            blacs_ctxt_h.barrier();
                            const auto wfc_block = get_local_mat(wfc_bra.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                            auto temp_nband_nao = multiply_scalapack(wfc_block, desc_nband_nao, sigc_nao_nao, desc_nao_nao, desc_nband_nao);
                            if (ispn_bra == ispn_ket)
                                ScalapackConnector::pgemm_f('N', 'C', n_bands, n_bands, n_aos, 1.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                            wfc_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                                                            sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                            else
                            {
                                const auto wfc_ket_block = get_local_mat(wfc_ket.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL);
                                ScalapackConnector::pgemm_f('N', 'T', n_bands, n_bands, n_aos, 1.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                            wfc_ket_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                                                            sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                            }
                            // collect the full matrix to master
                            // TODO: would need a different strategy for large system
                            ScalapackConnector::pgemr2d_f(n_bands, n_bands,
                                                          sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc,
                                                          sigc_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc,
                                                          desc_nband_nband_fb.ictxt());
                            // NOTE: only the matrices at master process is meaningful
                            auto sigc_freq = find_nested_int_map_2(this->sigc_is_ik_f_KS, isp, ik);
                            if (sigc_freq == nullptr || sigc_freq->count(freq) == 0)
                            {
                                this->sigc_is_ik_f_KS[isp][ik][freq] = sigc_nband_nband_fb.copy();
                            }
                            else
                            {
                                sigc_freq->at(freq) += sigc_nband_nband_fb;
                            }
                        }
                    }
                    global::profiler.stop("g0w0_build_sigc_KS_rotate_kserial");
                }
            }
        }
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (use_gpu_replace_scalapack)
    {
        DEVICE_CHECK(deviceFreeAsync(d_wfc_bra, blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_wfc_ket, blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_sigc_nao, blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_temp, blacs_ctxt_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_sigc_nband, blacs_ctxt_h.ddla_handle->stream));
    }
#endif
#endif
    this->is_kspace_built_ = true;
    global::profiler.stop("g0w0_build_sigc_KS");
}

void G0W0::build_sigc_matrix_KS_kgrid(const Atoms &geometry)
{
    comm_h.barrier();
    librpa_int::global::ofs_myid << "build_sigc_matrix_KS_kgrid: constructing self-energy matrix for SCF k-grid" << std::endl;
    this->build_sigc_matrix_KS(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {});
    if (this->output_sigc_ks_if)
    {
        const auto fn = path_as_directory(this->output_dir) + "self_energy_omega.dat";
        write_self_energy_omega(fn.c_str(), *this, this->mf.get_n_kpoints(),
                                this->mf.get_n_bands());
    }
}

void G0W0::build_sigc_matrix_KS_band(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc,
                                     const std::vector<Vector3_Order<double>> &kfrac_band,
                                     const AtomPairBvKRemap<atom_t> &bvk_remap,
                                     const std::vector<int> *output_iks)
{
    comm_h.barrier();
    if (comm_h.myid == 0)
    {
        librpa_int::global::lib_printf("build_sigc_matrix_KS_kgrid: constructing self-energy matrix for band k-path\n");
    }
    this->build_sigc_matrix_KS(wfc, kfrac_band, bvk_remap);
    if (this->output_sigc_ks_if)
    {
        const int n_bands = infer_target_n_bands(comm_h, wfc, this->mf.get_n_bands());
        const auto iks = output_iks == nullptr
                             ? collect_target_iks(comm_h, wfc, static_cast<int>(kfrac_band.size()))
                             : *output_iks;
        const auto stem = make_sigc_ks_imagfreq_band_stem(
            this->output_dir, output_sigc_ks_if_band_index_++);
        write_self_energy_omega((stem + ".dat").c_str(), *this, iks, n_bands);
        write_self_energy_omega_kpoints((stem + ".kidx").c_str(), *this, iks);
    }
}

void G0W0::build_sigc_matrix_KS_kgrid_blacs(const BlacsCtxtHandler &blacs_ctxt_h,
                                                   const bool use_gpu_replace_scalapack)
{
    comm_h.barrier();
    librpa_int::global::ofs_myid << "build_sigc_matrix_KS_kgrid: constructing self-energy matrix for SCF k-grid with BLACS" << std::endl;
    this->build_sigc_matrix_KS_blacs(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {}, blacs_ctxt_h, use_gpu_replace_scalapack);
    if (this->output_sigc_ks_if)
    {
        const auto fn = path_as_directory(this->output_dir) + "self_energy_omega.dat";
        write_self_energy_omega(fn.c_str(), *this, this->mf.get_n_kpoints(),
                                this->mf.get_n_bands());
    }
}

void G0W0::build_sigc_matrix_KS_band_blacs(
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc,
    const std::vector<Vector3_Order<double>> &kfrac_band,
    const AtomPairBvKRemap<atom_t> &bvk_remap,
    const BlacsCtxtHandler &blacs_ctxt_h,
    const bool use_gpu_replace_scalapack,
    const std::vector<int> *output_iks)
{
    comm_h.barrier();
    if (comm_h.myid == 0)
    {
        librpa_int::global::lib_printf("build_sigc_matrix_KS_band: constructing self-energy matrix for band k-path with BLACS\n");
    }
    this->build_sigc_matrix_KS_blacs(wfc, kfrac_band, bvk_remap, blacs_ctxt_h, use_gpu_replace_scalapack);
    if (this->output_sigc_ks_if)
    {
        const int n_bands = infer_target_n_bands(comm_h, wfc, this->mf.get_n_bands());
        const auto iks = output_iks == nullptr
                             ? collect_target_iks(comm_h, wfc, static_cast<int>(kfrac_band.size()))
                             : *output_iks;
        const auto stem = make_sigc_ks_imagfreq_band_stem(
            this->output_dir, output_sigc_ks_if_band_index_++);
        write_self_energy_omega((stem + ".dat").c_str(), *this, iks, n_bands);
        write_self_energy_omega_kpoints((stem + ".kidx").c_str(), *this, iks);
    }
}

} // namespace librpa_int
