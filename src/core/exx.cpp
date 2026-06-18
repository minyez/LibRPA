#include "exx.h"

#include <omp.h>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../io/global_io.h"
#include "../io/stl_io_helper.h"
#include "../math/lapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../math/vector3_order.h"
#include "../mpi/global_mpi.h"
#include "../utils/base_utility.h"
#include "../utils/constants.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "input_symmetry.h"
#include "../utils/utils_mem.h"
#include "atomic_basis.h"
#include "meanfield_mpi.h"
#include "geometry.h"
#include "librpa_enums.h"
// #include "params.h"
#include "pbc.h"
#include "utils_atomic_basis_blacs.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/physics/Exx.h>
#include <RI/physics/symmetry/Symmetry_Filter.h>
#include <RI/ri/Cell_Nearest.h>
#else
#include "../utils/libri_stub.h"
#endif

namespace librpa_int
{

namespace
{

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
convert_input_symmetry_irreducible_sector_to_libri(
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

bool exx_coulomb_uses_input_symmetry_irreducible_sector_layout(
    const atpair_R_mat_t& coul_mat,
    const librpa_int::SymmetryContext& symmetry_ctx)
{
    if (coul_mat.empty())
    {
        return false;
    }
    for (const auto& i_entry : coul_mat)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto& j_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(j_entry.first);
            const auto sector_iter = symmetry_ctx.irreducible_sector.find({ir_I, ir_J});
            if (sector_iter == symmetry_ctx.irreducible_sector.end())
            {
                return false;
            }
            for (const auto& r_entry : j_entry.second)
            {
                const auto& R = r_entry.first;
                const librpa_int::input_symmetry_R_t r_array{R.x, R.y, R.z};
                if (sector_iter->second.count(r_array) == 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

#ifdef LIBRPA_USE_LIBRI
template <typename TA, typename TC, typename Tdata>
class OutputOnlyFilter_Atom_Symmetry : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
    using TAC = std::pair<TA, TC>;

    OutputOnlyFilter_Atom_Symmetry(
        const TC& period,
        const std::map<std::pair<TA, TA>, std::set<TC>>& irreducible_sector)
        : symmetry(period, irreducible_sector)
    {
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TA& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a1b0_a2b1:
            case RI::Label::ab_ab::a1b1_a2b0:
            case RI::Label::ab_ab::a0b0_a2b1:
            case RI::Label::ab_ab::a0b1_a2b0:
            case RI::Label::ab_ab::a1b1_a2b2:
            case RI::Label::ab_ab::a0b1_a2b2:
            case RI::Label::ab_ab::a1b0_a2b2:
            case RI::Label::ab_ab::a0b0_a2b2:
                return !this->symmetry.in_irreducible_sector(A1, A3);
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
            case RI::Label::ab_ab::a0b1_a1b2:
            case RI::Label::ab_ab::a0b2_a1b0:
            case RI::Label::ab_ab::a0b2_a1b1:
                return !this->symmetry.in_irreducible_sector(A3, A1);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b1:
            case RI::Label::ab_ab::a0b1_a1b0:
                return !this->symmetry.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

  private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry;
};

template <typename Tdata>
ComplexMatrix convert_libri_tensor_to_complex_matrix(
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
#endif

} // namespace

Exx::Exx(const MeanField &mf_in, const AtomicBasis &atbasis_wfc_in,
         const PeriodicBoundaryData &pbc_in, const SymmetryContext &symmetry_context_in,
         const KPointBlacsParallelContext &kblacs_ctxt_in,
         const ArrayDesc &desc_wfc_in, bool is_mf_eigvec_k_distributed)
    : mf(mf_in),
      desc_wfc(desc_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      pbc(pbc_in),
      symmetry_context(symmetry_context_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in)
{
    is_mf_eigvec_k_distributed_ = is_mf_eigvec_k_distributed;
    is_rspace_built_ = false;
    is_kspace_built_ = false;
    is_rspace_redist_for_KS_ = false;
    is_rspace_redist_blacs_ = false;

    // Runtime options
    libri_threshold_C = 0.0;
    libri_threshold_V = 0.0;
    libri_threshold_D = 0.0;
};

static ComplexMatrix extract_dmat_cplx_R_IJblock(const ComplexMatrix& dmat_cplx, const AtomicBasis &ab_wfc, const atom_t& I, const atom_t& J)
{
    const auto I_num = ab_wfc.get_atom_nb(I);
    const auto J_num = ab_wfc.get_atom_nb(J);
    ComplexMatrix dmat_cplx_IJR(I_num, J_num);
    for (size_t i = 0; i != I_num; i++)
    {
        size_t i_glo = ab_wfc.get_global_index(I, i);
        for (size_t j = 0; j != J_num; j++)
        {
            size_t j_glo = ab_wfc.get_global_index(J, j);
            dmat_cplx_IJR(i, j) = dmat_cplx(i_glo, j_glo);
        }
    }
    return dmat_cplx_IJR;
}


static void warn_dmat_IJR_nonzero_imag(const ComplexMatrix& dmat_cplx, const int& ispin, const atom_t& I, const atom_t& J, const Vector3_Order<int> R)
{
    if (dmat_cplx.get_max_abs_imag() > 1e-2)
        global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning: complex-valued density matrix, spin %d IJR %zu %zu (%d, %d, %d)\n", ispin, I, J, R.x, R.y, R.z);
}


#ifdef LIBRPA_USE_LIBRI
static void build_dmat_libri_kserial(
    const MeanField &mf,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    const bool save_cplx,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> &dmat_libri,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> &dmat_libri_cplx)
{
    // for (const auto &[ik, mat]: mf.get_eigenvectors().at(ispin))
    // {
    //     global::ofs_myid << ik << std::endl;
    //     print_complex_matrix("eigenvector", mat, global::ofs_myid, true);
    // }
    global::profiler.start("exx_build_dmat_libri_kserial");
    std::map<Vector3_Order<int>, std::vector<atpair_t>> map_R_IJs;
    for (const auto &IJR: IJRs)
    {
        const auto &R = IJR.second;
        map_R_IJs[R].push_back(IJR.first);
    }
    const auto atom_nw = build_atom_nw_map(atbasis_wfc);
    const bool restore_input_symmetry_kstars = can_restore_input_symmetry_kstar_meanfield(
        symmetry_context, mf, kfrac_list, atom_nw, symmetry_context.input_coord_frac);
    const auto member_kfrac_targets = restore_input_symmetry_kstars
        ? build_input_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc)
        : input_symmetry_kstar_member_kfrac_targets_t{};
    for (const auto &R_IJs: map_R_IJs)
    {
        const auto &R = R_IJs.first;
        const auto &IJs = R_IJs.second;
        std::array<int,3> Ra{R.x,R.y,R.z};
        const auto dmat_cplx = restore_input_symmetry_kstars
            ? get_input_symmetry_restored_dmat_cplx_R(
                  symmetry_context, mf, ispin, ispinor_bra, ispinor_ket, kfrac_list, R, atom_nw,
                  symmetry_context.input_coord_frac, &member_kfrac_targets)
            : mf.get_dmat_cplx_R(ispin, ispinor_bra, ispinor_ket, kfrac_list, R);
        // global::ofs_myid << R << std::endl;
        // print_complex_matrix("dmat_cplx[R]", dmat_cplx, global::ofs_myid, true);
        omp_lock_t dmat_lock;
        omp_init_lock(&dmat_lock);
#pragma omp parallel for schedule(dynamic)
        for (const auto &IJ: IJs)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto dmat_IJR = extract_dmat_cplx_R_IJblock(dmat_cplx, atbasis_wfc, I, J);
            if (save_cplx)
            {
                std::valarray<cplxdb> dmat_va(dmat_IJR.c, dmat_IJR.size);
                auto pdmat = std::make_shared<std::valarray<cplxdb>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri_cplx[I][{J, Ra}] = RI::Tensor<cplxdb>({size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
            else
            {
                warn_dmat_IJR_nonzero_imag(dmat_IJR, ispin, I, J, R);
                std::valarray<double> dmat_va(dmat_IJR.real().c, dmat_IJR.size);
                auto pdmat = std::make_shared<std::valarray<double>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
        }
#pragma omp barrier
        omp_destroy_lock(&dmat_lock);
    }
    global::profiler.stop("exx_build_dmat_libri_kserial");
}

static void build_dmat_libri_kpara(
    const MeanField &mf,
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    const bool save_cplx,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> &dmat_libri,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> &dmat_libri_cplx)
{
    global::profiler.start("exx_build_dmat_libri_kpara");

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
    MPI_Allreduce(MPI_IN_PLACE, &n_Rs_max, 1, MPI_INT, MPI_MAX, comm_h.comm);
    const auto dmat_Rs_cplx = get_dmat_cplx_Rs_kpara(ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs_this, comm_h);
    // global::ofs_myid << kfrac_list << std::endl;
    for (const auto &R_dmat_cplx: dmat_Rs_cplx)
    {
        const auto &R = R_dmat_cplx.first;
        const auto &dmat_cplx = R_dmat_cplx.second;
        // global::ofs_myid << R << std::endl;
        // print_complex_matrix("test", dmat_cplx, global::ofs_myid, true);
        std::array<int,3> Ra{R.x,R.y,R.z};
        const auto &map_IJs = map_R_IJs.at(R);
        omp_lock_t dmat_lock;
        omp_init_lock(&dmat_lock);
#pragma omp parallel for schedule(dynamic)
        for (const auto &IJ: map_IJs)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto dm_block = extract_dmat_cplx_R_IJblock(dmat_cplx, atbasis_wfc, I, J);
            if (save_cplx)
            {
                std::valarray<cplxdb> dmat_va(dm_block.c, dm_block.size);
                auto pdmat = std::make_shared<std::valarray<cplxdb>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri_cplx[I][{J, Ra}] = RI::Tensor<cplxdb>({size_t(dm_block.nr), size_t(dm_block.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
            else
            {
                warn_dmat_IJR_nonzero_imag(dm_block, ispin, I, J, R);
                std::valarray<double> dmat_va(dm_block.real().c, dm_block.size);
                auto pdmat = std::make_shared<std::valarray<double>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(dm_block.nr), size_t(dm_block.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
        }
#pragma omp barrier
        omp_destroy_lock(&dmat_lock);
    }
    global::profiler.stop("exx_build_dmat_libri_kpara");
}

static void build_dmat_libri_kblacs_para(
    const MeanField &mf,
    const KPointBlacsParallelContext &kblacs_ctxt,
    const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const IndexScheduler &sched,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<Vector3_Order<int>> Rs,
    const bool save_cplx,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> &dmat_libri,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> &dmat_libri_cplx)
{
    global::profiler.start("exx_build_dmat_libri_kblacs_para");

    auto dmat_Rs_cplx = get_dmat_cplx_Rs_kblacs_para(ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs, kblacs_ctxt, desc_wfc, desc_dm);
    // global::ofs_myid << kfrac_list << std::endl;
    for (auto &R_dmat_cplx: dmat_Rs_cplx)
    {
        auto &R = R_dmat_cplx.first;
        auto &mat_blacs = R_dmat_cplx.second;
        auto pair_mat = get_ap_map_from_blacs_dist_scheduler(mat_blacs, sched, atbasis_wfc, atbasis_wfc, desc_dm);
        for (auto &[pair, mat_ap]: pair_mat)
        {
            const auto &I = as_int(pair.first);
            const auto &J = as_int(pair.second);
            const auto &n_I = atbasis_wfc.get_atom_nb(I);
            const auto &n_J = atbasis_wfc.get_atom_nb(J);
            mat_ap.swap_to_row_major();
            if (save_cplx)
                dmat_libri_cplx[I][{J, {R.x, R.y, R.z}}] = RI::Tensor<cplxdb>({n_I, n_J}, mat_ap.sptr());
            else
                dmat_libri[I][{J, {R.x, R.y, R.z}}] = RI::Tensor<double>({n_I, n_J}, mat_ap.get_real().sptr());
        }
        mat_blacs.clear();
    }
    global::profiler.stop("exx_build_dmat_libri_kblacs_para");
}
#endif

void Exx::build(const LibrpaParallelRouting routing,
                const AtomicBasis &atbasis_abf, const Cs_LRI &Cs,
                const atpair_R_mat_t &coul_mat)
{
    using std::endl;
    using global::profiler;
    using global::ofs_myid;

    assert(routing == LIBRPA_ROUTING_LIBRI);

    if (this->is_rspace_built_)
    {
        return;
    }

    const auto &n_spins = this->mf.get_n_spins();
    const auto &n_spinor = this->mf.get_n_spinor();
    const auto &ab_wfc = this->atbasis_wfc;
    const int n_atoms = ab_wfc.n_atoms;
    const auto &Rlist = this->pbc.Rlist;

    // Use complex LibRI objects when 2-component wave function is used
    const bool use_complex_exx_r = n_spinor > 1 ? true : false;

#ifdef LIBRPA_USE_LIBRI
    if (comm_h.is_root())
    {
        global::lib_printf("Computing EXX orbital energy using LibRI\n");
    }
    comm_h.barrier();

    // Use either one
    RI::Exx<int, int, 3, double> exx_libri;
    RI::Exx<int, int, 3, cplxdb> exx_libri_cplx;

    std::map<int,std::array<double,3>> atoms_pos;
    for (int i = 0; i < n_atoms; i++)
        atoms_pos.insert(std::pair<int, std::array<double, 3>>{i, {0, 0, 0}});

    if (use_complex_exx_r)
        exx_libri_cplx.set_parallel(comm_h.comm, atoms_pos, this->pbc.latvec_array, this->pbc.period_array);
    else
        exx_libri.set_parallel(comm_h.comm, atoms_pos, this->pbc.latvec_array, this->pbc.period_array);

    const auto& symmetry_ctx = this->symmetry_context;
    const bool use_input_exx_symmetry =
        symmetry_ctx.available
        && this->pbc.klist.size() < static_cast<std::size_t>(this->pbc.get_n_cells_bvk())
        && symmetry_ctx.has_ao_shell_layout()
        && !symmetry_ctx.irreducible_sector.empty()
        && !symmetry_ctx.rspace_operations.empty()
        && symmetry_ctx.atom_to_type.size() == static_cast<std::size_t>(n_atoms)
        && symmetry_ctx.input_coord_frac.size() == static_cast<std::size_t>(n_atoms);
    const auto libri_irreducible_sector =
        use_input_exx_symmetry
            ? convert_input_symmetry_irreducible_sector_to_libri(
                  symmetry_ctx.irreducible_sector, this->pbc.period_array)
            : std::map<std::pair<int, int>, std::set<std::array<int, 3>>>{};
    const bool use_libri_exx_symmetry_filter = use_input_exx_symmetry;
    librpa_int::input_symmetry_rspace_sector_stars_t input_symmetry_sector_stars;
    if (use_input_exx_symmetry)
    {
        global::lib_printf(
            "Reducing EXX real-space contractions with ABACUS irreducible sectors\n");
        librpa_int::build_input_symmetry_rspace_sector_stars(
            symmetry_ctx, symmetry_ctx.input_coord_frac, this->pbc.period, Rlist,
            input_symmetry_sector_stars, nullptr);
        exx_libri.set_symmetry(false, {});
        exx_libri_cplx.set_symmetry(false, {});
        if (use_complex_exx_r)
        {
            exx_libri_cplx.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_Atom_Symmetry<int, std::array<int, 3>, cplxdb>>(
                    exx_libri_cplx.lri.period, libri_irreducible_sector);
        }
        else
        {
            exx_libri.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_Atom_Symmetry<int, std::array<int, 3>, double>>(
                    exx_libri.lri.period, libri_irreducible_sector);
        }
    }
    else
    {
        exx_libri.set_symmetry(false, {});
        exx_libri_cplx.set_symmetry(false, {});
    }

    // Initialize Cs libRI container on each process
    // Note: we use different treatment in different routings
    //     R-tau routing:
    //         Each process has a full Cs copy.
    //         Thus in each process we only pass a few to LibRI container.
    //     atom-pair routing:
    //         Cs is already distributed across all processes.
    //         Pass the all Cs to libRI container.

    profiler.start("build_real_space_exx_1", "Prepare C libRI object");
    ofs_myid << "Number of Cs keys: " << get_num_keys(Cs.data_libri) << endl;
    // print_keys(global::ofs_myid, Cs.data_libri);
    global::ofs_myid << "comm_h.comm " << comm_h.comm << " global::mpi_comm_global_h.comm " << global::mpi_comm_global_h.comm << std::endl;
    // global::ofs_myid << Cs.data_libri << std::endl;
    if (use_complex_exx_r)
    {
        std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<cplxdb>>> data_libri;
        for (const auto &I_JR_C : Cs.data_libri)
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
        exx_libri_cplx.set_Cs(data_libri, libri_threshold_C);
    }
    else
        exx_libri.set_Cs(Cs.data_libri, libri_threshold_C);
    // exx_libri.set_Cs({}, libri_threshold_C);
    profiler.stop("build_real_space_exx_1");
    ofs_myid << "Finished setup Cs for EXX" << endl;
    std::flush(global::ofs_myid);

    // initialize Coulomb matrix
    global::profiler.start("build_real_space_exx_2", "Prepare V libRI object");

    atpair_R_mat_t exx_coul_mat_restored;
    const atpair_R_mat_t* exx_coul_mat_ptr = &coul_mat;
    const bool use_input_exx_coulomb_restore =
        use_input_exx_symmetry
        && exx_coulomb_uses_input_symmetry_irreducible_sector_layout(coul_mat, symmetry_ctx);
    if (use_input_exx_coulomb_restore)
    {
        global::lib_printf(
            "Restoring the ABACUS EXX auxiliary Coulomb blocks from the irreducible sector to the full real-space sector before LibRI contraction\n");
        for (const auto& I_JRV : coul_mat)
        {
            const auto ir_I = I_JRV.first;
            for (const auto& J_RV : I_JRV.second)
            {
                const auto ir_J = J_RV.first;
                const auto sector_pair =
                    std::make_pair(static_cast<atom_t>(ir_I), static_cast<atom_t>(ir_J));
                const auto pair_iter = input_symmetry_sector_stars.find(sector_pair);
                if (pair_iter == input_symmetry_sector_stars.end())
                {
                    throw std::runtime_error(
                        "Failed to match an irreducible EXX Coulomb atom pair with the ABACUS restore map");
                }
                for (const auto& R_V : J_RV.second)
                {
                    const auto& ir_R = R_V.first;
                    const auto star_iter = pair_iter->second.find(ir_R);
                    if (star_iter == pair_iter->second.end())
                    {
                        throw std::runtime_error(
                            "Failed to match an irreducible EXX Coulomb real-space block with the ABACUS restore map");
                    }
                    const ComplexMatrix v_ir(*R_V.second);
                    for (const auto& restore_member : star_iter->second)
                    {
                        const ComplexMatrix v_full =
                            librpa_int::rotate_input_symmetry_abf_rspace_matrix(
                                symmetry_ctx, restore_member.isym,
                                static_cast<atom_t>(ir_I),
                                static_cast<atom_t>(ir_J), v_ir);
                        const matrix v_full_real = v_full.real();
                        auto& target_pair_map =
                            exx_coul_mat_restored[restore_member.full_atom_pair.first]
                                                 [restore_member.full_atom_pair.second];
                        if (target_pair_map.count(restore_member.full_R) != 0)
                        {
                            throw std::runtime_error(
                                "Duplicate full-sector EXX Coulomb block appears during ABACUS symmetry restore");
                        }
                        target_pair_map[restore_member.full_R] =
                            std::make_shared<matrix>(v_full_real);
                    }
                }
            }
        }
        exx_coul_mat_ptr = &exx_coul_mat_restored;
    }
    const auto& exx_coul_mat = *exx_coul_mat_ptr;
    const bool use_replicated_input_symmetry_exx_coulomb =
        use_input_exx_coulomb_restore && comm_h.nprocs > 1;

    std::map<int, std::map<std::pair<int,std::array<int,3>>, RI::Tensor<double>>> V_libri;
    std::map<int, std::map<std::pair<int,std::array<int,3>>, RI::Tensor<cplxdb>>> V_libri_cplx;

    global::profiler.start("build_real_space_exx_2_1");
    if (routing == LIBRPA_ROUTING_RTAU)
    {
        // Full Coulomb case, have to re-distribute
        // TODO: remove as libri routing is enforced above
        for (auto IJR: dispatch_vector_prod(get_atom_pair(exx_coul_mat), Rlist, comm_h.myid, comm_h.nprocs, true, true))
        {
            const auto I = IJR.first.first;
            const auto J = IJR.first.second;
            const auto R = IJR.second;
            const auto& VIJR = exx_coul_mat.at(I).at(J).at(R);
            // debug
            // printf("I J R %zu %zu %d %d %d, max(V) %f\n", I, J, R.x, R.y, R.z, VIJR->max());
            std::array<int,3> Ra{R.x,R.y,R.z};
            if (use_complex_exx_r)
            {
                auto pv = std::make_shared<std::valarray<cplxdb>>(VIJR->size);
                for (size_t i = 0; i < VIJR->size; ++i)
                    (*pv)[i] = cplxdb(VIJR->c[i], 0.0);
                V_libri_cplx[I][{J, Ra}] = RI::Tensor<cplxdb>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
            }
            else
            {
                auto pv = std::make_shared<std::valarray<double>>(VIJR->c, VIJR->size);
                V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
            }
        }
    }
    else
    {
        if (use_replicated_input_symmetry_exx_coulomb)
        {
            for (const auto& IJR : dispatch_vector_prod(
                     get_atom_pair(exx_coul_mat), Rlist, comm_h.myid, comm_h.nprocs, true, true))
            {
                const auto I = IJR.first.first;
                const auto J = IJR.first.second;
                const auto& R = IJR.second;
                if (exx_coul_mat.count(I) == 0 || exx_coul_mat.at(I).count(J) == 0
                    || exx_coul_mat.at(I).at(J).count(R) == 0)
                {
                    continue;
                }
                const auto& V = exx_coul_mat.at(I).at(J).at(R);
                std::array<int, 3> Ra{R.x, R.y, R.z};
                if (use_complex_exx_r)
                {
                    auto pv = std::make_shared<std::valarray<cplxdb>>(V->size);
                    for (size_t i = 0; i < V->size; ++i)
                        (*pv)[i] = cplxdb(V->c[i], 0.0);
                    V_libri_cplx[I][{J, Ra}] =
                        RI::Tensor<cplxdb>({size_t(V->nr), size_t(V->nc)}, pv);
                }
                else
                {
                    auto pv = std::make_shared<std::valarray<double>>(V->c, V->size);
                    V_libri[I][{J, Ra}] =
                        RI::Tensor<double>({size_t(V->nr), size_t(V->nc)}, pv);
                }
            }
        }
        else
        {
            for (const auto &I_JRV: exx_coul_mat)
            {
                const auto I = I_JRV.first;
                for (const auto &J_RV: I_JRV.second)
                {
                    const auto J = J_RV.first;
                    for (const auto &R_V: J_RV.second)
                    {
                        const auto &R = R_V.first;
                        const auto &V = R_V.second;
                        std::array<int,3> Ra{R.x,R.y,R.z};
                        if (use_complex_exx_r)
                        {
                            auto pv = std::make_shared<std::valarray<cplxdb>>(V->size);
                            for (size_t i = 0; i < V->size; ++i)
                                (*pv)[i] = cplxdb(V->c[i], 0.0);
                            V_libri_cplx[I][{J, Ra}] =
                                RI::Tensor<cplxdb>({size_t(V->nr), size_t(V->nc)}, pv);
                        }
                        else
                        {
                            auto pv = std::make_shared<std::valarray<double>>(V->c, V->size);
                            V_libri[I][{J, Ra}] =
                                RI::Tensor<double>({size_t(V->nr), size_t(V->nc)}, pv);
                        }
                    }
                }
            }
        }
    }
    global::profiler.stop("build_real_space_exx_2_1");

    // global::ofs_myid << V_libri << endl;
    global::profiler.start("build_real_space_exx_2_2");
    if (use_complex_exx_r)
    {
        global::ofs_myid << "Number of V keys: " << get_num_keys(V_libri_cplx) << endl;
        exx_libri_cplx.set_Vs(V_libri_cplx, libri_threshold_V);
        V_libri_cplx.clear();
    }
    else
    {
        global::ofs_myid << "Number of V keys: " << get_num_keys(V_libri) << endl;
        exx_libri.set_Vs(V_libri, libri_threshold_V);
        V_libri.clear();
    }
    // exx_libri.set_Vs({}, libri_threshold_V);
    release_free_mem();
    global::profiler.stop("build_real_space_exx_2_2");
    global::profiler.stop("build_real_space_exx_2");
    global::lib_printf("Task %4d: V setup for EXX\n", comm_h.myid);
    // cout << V_libri << endl;

    // initialize density matrix
    global::profiler.start("build_real_space_exx_prepare_index");
    std::vector<atpair_t> atpair_dmat;
    for (atom_t I = 0; I < as_atom(n_atoms); I++)
        for (atom_t J = 0; J < as_atom(n_atoms); J++)
            atpair_dmat.push_back({I, J});
    const auto dmat_IJRs_local = dispatch_vector_prod(atpair_dmat, Rlist, comm_h.myid, comm_h.nprocs, true, true);

    // For k-BLACS two-level parallelization
    const int n_basis_ao = mf.get_n_aos();
    const auto desc_dm = kblacs_ctxt.create_array_desc(n_basis_ao, n_basis_ao);
    IndexScheduler sched;
    const auto map_atpairs_balanced =
        get_balanced_ap_distribution_for_consec_descriptor(atbasis_wfc, atbasis_wfc, desc_dm);
    sched.init(map_atpairs_balanced, atbasis_wfc, atbasis_wfc, desc_dm, false);
    const auto iRs = dispatcher_balanced(0, Rlist.size(), kblacs_ctxt.kpoints_local().size(), true,
                                         kblacs_ctxt.comm_kpoint_h.comm);
    std::vector<Vector3_Order<int>> Rs;
    Rs.reserve(iRs.size());
    for (const auto &iR : iRs)
        Rs.push_back(Rlist[iR]);
    global::profiler.stop("build_real_space_exx_prepare_index");

    for (auto isp = 0; isp != n_spins; isp++)
    {
        for (auto ispn_bra = 0; ispn_bra != n_spinor; ispn_bra++)
        {
            for (auto ispn_ket = 0; ispn_ket != n_spinor; ispn_ket++)
            {
                global::profiler.start("build_real_space_exx_3", "Prepare DM libRI object");
                std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> dmat_libri;
                std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> dmat_libri_cplx;
                if (is_mf_eigvec_k_distributed_)
                {
                    // build_dmat_libri_kpara(mf, comm_h, atbasis_wfc, isp, ispn_bra, ispn_ket,
                    //                        this->pbc.kfrac_list, dmat_IJRs_local, use_complex_exx_r,
                    //                        dmat_libri, dmat_libri_cplx);
                    build_dmat_libri_kblacs_para(mf, kblacs_ctxt, desc_wfc, desc_dm, sched,
                                                 atbasis_wfc, isp, ispn_bra, ispn_ket,
                                                 this->pbc.kfrac_list, Rs, use_complex_exx_r,
                                                 dmat_libri, dmat_libri_cplx);
                }
                else
                {
                    build_dmat_libri_kserial(mf, atbasis_wfc, isp, ispn_bra, ispn_ket,
                                             this->pbc, this->symmetry_context,
                                             this->pbc.kfrac_list, dmat_IJRs_local,
                                             use_complex_exx_r, dmat_libri, dmat_libri_cplx);
                }
                // global::ofs_myid << dmat_libri << std::endl;
                // print_keys(global::ofs_myid, dmat_libri);
                global::profiler.start("build_real_space_exx_libri_set_Ds");
                if (use_complex_exx_r)
                {
                    global::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri_cplx) << "\n";
                    exx_libri_cplx.set_Ds(dmat_libri_cplx, libri_threshold_D);
                }
                else
                {
                    global::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri) << "\n";
                    exx_libri.set_Ds(dmat_libri, libri_threshold_D);
                }
                // exx_libri.set_Ds({}, libri_threshold_D);
                global::profiler.stop("build_real_space_exx_libri_set_Ds");
                global::profiler.stop("build_real_space_exx_3");
                global::lib_printf("Task %4d: DM setup for EXX\n", comm_h.myid);

                global::profiler.start("build_real_space_exx_4", "Call libRI Hexx calculation");
                if (use_complex_exx_r)
                    exx_libri_cplx.cal_Hs();
                else
                    exx_libri.cal_Hs();
                global::profiler.stop("build_real_space_exx_4");
                release_free_mem();

                global::lib_printf("Task %4d: cal_Hs elapsed time: %f\n", comm_h.myid, global::profiler.get_wall_time_last("build_real_space_exx_4"));
                // print_keys(global::ofs_myid, exx_libri.Hs);
                // ofs_myid << "exx_libri.Hs:\n" << exx_libri.Hs << endl;

	                auto store_exx_block = [&](const atom_t full_I,
	                                           const atom_t full_J,
	                                           const Vector3_Order<int>& full_R,
	                                           const ComplexMatrix& block)
	                {
	                    const auto n_full_I = ab_wfc.get_atom_nb(full_I);
	                    const auto n_full_J = ab_wfc.get_atom_nb(full_J);
	                    if (block.nr != n_full_I || block.nc != n_full_J)
	                    {
	                        throw std::runtime_error(
	                            "ABACUS EXX real-space restore produced an AO block with an inconsistent dimension");
	                    }
	                    if (use_complex_exx_r)
	                    {
	                        this->exx_IJR_cplx[isp][ispn_bra][ispn_ket][full_I][full_J][full_R] =
	                            Matz(n_full_I, n_full_J, block.c, MAJOR::ROW);
	                    }
	                    else
	                    {
	                        const auto block_real = block.real();
	                        this->exx_IJR[isp][ispn_bra][ispn_ket][full_I][full_J][full_R] =
	                            Matd(n_full_I, n_full_J, block_real.c, MAJOR::ROW);
	                    }
	                };

	                auto copy_exx_blocks = [&](const auto &Hs)
	                {
	                    global::ofs_myid << "Number of exx_libri.Hs keys: "
	                                     << get_num_keys(Hs) << std::endl;
	                    if (use_libri_exx_symmetry_filter)
	                    {
	                        global::lib_printf(
	                            "Restoring the symmetry-filtered LibRI EXX blocks to the full AO real-space sector\n");
	                    }
	                    for (const auto &I_JR_exx : Hs)
	                    {
	                        const auto &I = I_JR_exx.first;
	                        const auto n_I = ab_wfc.get_atom_nb(I);
	                        for (const auto &JR_exx : I_JR_exx.second)
	                        {
	                            const auto &J = JR_exx.first.first;
	                            const auto n_J = ab_wfc.get_atom_nb(J);
	                            const auto &Ra = JR_exx.first.second;
	                            const auto R = Vector3_Order<int>{Ra[0], Ra[1], Ra[2]};
	                            const auto block_ir =
	                                convert_libri_tensor_to_complex_matrix(
	                                    JR_exx.second, n_I, n_J);
	                            if (use_libri_exx_symmetry_filter)
	                            {
	                                const auto sector_pair =
	                                    std::make_pair(static_cast<atom_t>(I),
	                                                   static_cast<atom_t>(J));
	                                const auto pair_iter =
	                                    input_symmetry_sector_stars.find(sector_pair);
	                                if (pair_iter == input_symmetry_sector_stars.end()
	                                    || pair_iter->second.count(R) == 0)
	                                {
	                                    std::ostringstream oss;
	                                    oss << "Failed to match a symmetry-filtered EXX real-space block"
	                                        << " with the ABACUS irreducible-sector restore map for I="
	                                        << I << " J=" << J << " R=(" << R.x << ","
	                                        << R.y << "," << R.z << ")";
	                                    throw std::runtime_error(oss.str());
	                                }
	                                for (const auto& restore_member : pair_iter->second.at(R))
	                                {
	                                    const auto block_full =
	                                        librpa_int::rotate_input_symmetry_rspace_matrix(
	                                            symmetry_ctx, restore_member.isym,
	                                            static_cast<atom_t>(I),
	                                            static_cast<atom_t>(J), block_ir);
	                                    store_exx_block(
	                                        restore_member.full_atom_pair.first,
	                                        restore_member.full_atom_pair.second,
	                                        restore_member.full_R,
	                                        block_full);
	                                }
	                            }
	                            else
	                            {
	                                store_exx_block(
	                                    static_cast<atom_t>(I),
	                                    static_cast<atom_t>(J),
	                                    R,
	                                    block_ir);
	                            }
	                        }
	                    }
	                };

	                global::profiler.start("build_real_space_exx_5");
	                if (use_complex_exx_r)
	                    copy_exx_blocks(exx_libri_cplx.Hs);
	                else
	                    copy_exx_blocks(exx_libri.Hs);
	                global::profiler.stop("build_real_space_exx_5");
            }
        }
    }

    // debug, print the Hexx matrices
    // for (const auto& isp_IJkH: this->Hexx)
    // {
    //     const auto& isp = isp_IJkH.first;
    //     for (const auto& I_JkH: isp_IJkH.second)
    //     {
    //         const auto& I = I_JkH.first;
    //         for (const auto& J_kH: I_JkH.second)
    //         {
    //             const auto& J = J_kH.first;
    //             for (const auto& k_H: J_kH.second)
    //             {
    //                 cout << isp << " " << I << " " << J << " {" << k_H.first << "} whole size: " << k_H.second->size << endl;
    //                 print_complex_matrix("", *k_H.second);
    //             }
    //         }
    //     }
    // }

#else
    if (comm_h.is_root())
    {
        global::lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error: trying build EXX orbital energy with LibRI, but the program is not compiled against LibRI\n");
    }
    throw std::logic_error("compilation");
    comm_h.barrier();
#endif

    release_free_mem();
    is_rspace_built_= true;
}

void Exx::build_KS(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                   const std::vector<Vector3_Order<double>> &kfrac_target,
                   const AtomPairBvKRemap<atom_t> &bvk_remap)
{
    throw LIBRPA_RUNTIME_ERROR("not implemented");
}

void Exx::build_KS_blacs(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                         const std::vector<Vector3_Order<double>> &kfrac_target,
                         const AtomPairBvKRemap<atom_t> &bvk_remap,
                         const BlacsCtxtHandler &blacs_ctxt_h)
{
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;

    // Ensure the communicator of BLACS context is the same as that parsed when constructed
    assert(blacs_ctxt_h.comm() == this->comm_h.comm);
    assert(this->is_rspace_built_);

    // Reset k-space matrices built from last call
    if (this->is_kspace_built_)
    {
        global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning: reset EXX k-space matrices\n");
        this->reset_kspace();
    }

    // NOTE: Here it assumes that wfc_target has the same spin, spinor and states dimensions
    const auto n_aos = this->mf.get_n_aos();
    const auto n_spins = this->mf.get_n_spins();
    const auto n_bands = this->mf.get_n_bands();
    const auto n_spinor = this->mf.get_n_spinor();
    const bool use_complex_exx_r = n_spinor > 1 ? true : false;

    // prepare scalapack array descriptors
    ArrayDesc desc_nao_nao(blacs_ctxt_h);
    ArrayDesc desc_nband_nao(blacs_ctxt_h);
    ArrayDesc desc_nband_nband(blacs_ctxt_h);

    // For communication of eigenvectors and final KS matrices
    ArrayDesc desc_nao_nband_fb(blacs_ctxt_h);  // emulate ComplexMatrix storage
    ArrayDesc desc_nao_nao_fb(blacs_ctxt_h);
    ArrayDesc desc_nband_nband_fb(blacs_ctxt_h);

    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);
    const int n_target_kpoints = static_cast<int>(kfrac_target.size());
    const bool use_root_dense_projection =
        use_input_symmetry_ibz_root_projection(this->symmetry_context,
                                               this->pbc,
                                               n_target_kpoints,
                                               this->mf.get_n_kpoints());
    if (use_root_dense_projection)
    {
        desc_nao_nao_fb.init(n_aos, n_aos, n_aos, n_aos, 0, 0);
    }

    // local 2D-block submatrices
    auto Hexx_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto Hexx_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);
    auto temp_nband_nao = init_local_mat<complex<double>>(desc_nband_nao, MAJOR::COL);

    if (!is_rspace_redist_for_KS_)
    {
        global::profiler.start("exx_build_KS_blacs_redist");
        const auto set_IJ_naonao = get_necessary_IJ_from_block_2D(
            this->atbasis_wfc, this->atbasis_wfc, desc_nao_nao);
        const auto Iset_Jset = convert_IJset_to_Iset_Jset(set_IJ_naonao);

        auto pack_real_exx_to_tensor = [&](const auto &exx_blocks)
        {
            std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> exx_tensor;
            for (const auto &[I, J_Rexx] : exx_blocks)
            {
                const auto n_I = this->atbasis_wfc.get_atom_nb(I);
                for (const auto &[J, R_exx] : J_Rexx)
                {
                    const auto n_J = this->atbasis_wfc.get_atom_nb(J);
                    for (const auto &[R, mat] : R_exx)
                    {
                        const std::array<int, 3> Ra{R.x, R.y, R.z};
                        exx_tensor[I][{J, Ra}] = RI::Tensor<double>({n_I, n_J}, mat.sptr());
                    }
                }
            }
            return exx_tensor;
        };

        auto pack_cplx_exx_to_tensor = [&](const auto &exx_blocks)
        {
            std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>> exx_tensor;

            for (const auto &[I, J_Rexx] : exx_blocks)
            {
                const auto n_I = this->atbasis_wfc.get_atom_nb(I);
                for (const auto &[J, R_exx] : J_Rexx)
                {
                    const auto n_J = this->atbasis_wfc.get_atom_nb(J);
                    for (const auto &[R, mat] : R_exx)
                    {
                        const std::array<int, 3> Ra{R.x, R.y, R.z};
                        exx_tensor[I][{J, Ra}] = RI::Tensor<cplxdb>({n_I, n_J}, mat.sptr());
                    }
                }
            }
            return exx_tensor;
        };

        for (int isp = 0; isp < n_spins; isp++)
        {
            for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
            {
                for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
                {
                    const auto exx = use_complex_exx_r
                                         ? nullptr
                                         : find_nested_int_map_3(exx_IJR, isp, ispn_bra, ispn_ket);
                    const auto exx_cplx = use_complex_exx_r
                                         ? find_nested_int_map_3(exx_IJR_cplx, isp, ispn_bra, ispn_ket)
                                         : nullptr;

                    if (use_complex_exx_r)
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>> exx_tensor_cplx;
                        if (exx_cplx != nullptr)
                            exx_tensor_cplx = pack_cplx_exx_to_tensor(*exx_cplx);
                        global::profiler.start("exx_build_KS_blacs_redist_comm_map2");
                        auto exx_redist = comm_map2_first(comm_h.comm, exx_tensor_cplx, Iset_Jset.first, Iset_Jset.second);
                        global::profiler.stop("exx_build_KS_blacs_redist_comm_map2");

                        std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matz>>> exx_is_new;
                        for (const auto &[I, JRmat]: exx_redist)
                        {
                            const int n_I = this->atbasis_wfc.get_atom_nb(I);
                            for (const auto &[JR, mat]: JRmat)
                            {
                                const atom_t J = JR.first;
                                const int n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;
                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};
                                exx_is_new[I][J][R] = Matz{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        if (exx_cplx != nullptr)
                            exx_cplx->swap(exx_is_new);
                        else if (!exx_is_new.empty())
                            exx_IJR_cplx[isp][ispn_bra][ispn_ket] = std::move(exx_is_new);
                    }
                    else
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> exx_tensor;
                        if (exx != nullptr)
                            exx_tensor = pack_real_exx_to_tensor(*exx);
                        global::profiler.start("exx_build_KS_blacs_redist_comm_map2");
                        auto exx_redist = comm_map2_first(comm_h.comm, exx_tensor, Iset_Jset.first, Iset_Jset.second);
                        global::profiler.stop("exx_build_KS_blacs_redist_comm_map2");

                        std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matd>>> exx_is_new;
                        for (const auto &[I, JRmat]: exx_redist)
                        {
                            const int n_I = this->atbasis_wfc.get_atom_nb(I);
                            for (const auto &[JR, mat]: JRmat)
                            {
                                const atom_t J = JR.first;
                                const int n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;
                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};
                                exx_is_new[I][J][R] = Matd{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        if (exx != nullptr)
                            exx->swap(exx_is_new);
                        else if (!exx_is_new.empty())
                            exx_IJR[isp][ispn_bra][ispn_ket] = std::move(exx_is_new);
                    }
                }
            }
        }
        is_rspace_redist_for_KS_ = true;
        is_rspace_redist_blacs_ = true;
        global::profiler.stop("exx_build_KS_blacs_redist");
        global::lib_printf("Task %4d: tensor communicate elapsed time: %f\n", comm_h.myid,
                           global::profiler.get_wall_time_last("exx_build_KS_blacs_redist"));
    }
    else
    {
        if (!is_rspace_redist_blacs_)
            throw LIBRPA_RUNTIME_ERROR("");
    }

    global::ofs_myid << "period:      " << this->pbc.period << std::endl;
    global::ofs_myid << "bvk_remap size: " << bvk_remap.size() << std::endl;
    global::ofs_myid << "latvec:      " << this->pbc.latvec << std::endl;

    auto shift_bvk = [&](const auto &map_orig, auto &map_shift)
    {
        auto add_mat = [](auto &R_mat_shift, const Vector3_Order<int> &R_bvk,
                          const auto &mat, const double weight)
        {
            auto mat_weighted = mat.copy();
            mat_weighted *= weight;
            auto [it, inserted] = R_mat_shift.emplace(R_bvk, mat_weighted);
            if (!inserted) it->second += mat_weighted;
        };

        for (const auto &[I, J_Rmat] : map_orig)
        {
            for (const auto &[J, R_mat] : J_Rmat)
            {
                const atpair_t IJ{I, J};
                auto &R_mat_shift = map_shift[I][J];

                for (const auto &[R, mat] : R_mat)
                {
                    const auto *R_bvks = bvk_remap.find_R_bvk(IJ, R);
                    if (R_bvks == nullptr || R_bvks->empty())
                    {
                        R_mat_shift[R] = mat;
                    }
                    else if (R_bvks->size() == 1)
                    {
                        R_mat_shift[R_bvks->front()] = mat;
                    }
                    else
                    {
                        const auto weight = 1.0 / static_cast<double>(R_bvks->size());
                        for (const auto &R_bvk: *R_bvks)
                        {
                            add_mat(R_mat_shift, R_bvk, mat, weight);
                        }
                    }
                }
            }
        }
    };

    for (int isp = 0; isp < n_spins; isp++)
    {
        this->exx_KS[isp] = {};
        this->Eexx[isp] = {};
        for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
        {
            for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
            {
                std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matd>>> exx_is_local;
                std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matz>>> exx_is_local_cplx;
                // Convert each <I,<J, R>> pair to the configured BvK counterpart to speed up later
                // Fourier transform while keeping the accuracy in further band interpolation.
                // Reuse the cleared-up exx_I_JR_local object
                auto orig = find_nested_int_map_3(exx_IJR, isp, ispn_bra, ispn_ket);
                auto orig_cplx = find_nested_int_map_3(exx_IJR_cplx, isp, ispn_bra, ispn_ket);
                if (use_complex_exx_r)
                {
                    if (orig_cplx != nullptr)
                    {
                        shift_bvk(*orig_cplx , exx_is_local_cplx);
                    }
                }
                else
                {
                    if (orig != nullptr)
                    {
                        shift_bvk(*orig, exx_is_local);
                    }
                }

                if (is_mf_eigvec_k_distributed_)
                {
                    // collect parsed eigenvectors all all processes
                    // global::ofs_myid << "isp " << isp << std::endl;
                    std::vector<int> nks_all(comm_h.nprocs, 0);
                    std::vector<int> iks_local;
                    auto it = wfc_target.find(isp);
                    if (it != wfc_target.cend())
                    {
                        const auto &wfc_sp = it->second.at(0);
                        nks_all[comm_h.myid] = wfc_sp.size();
                        for (const auto &[ik, _]: wfc_sp)
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
                        auto Hexx_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                        for (int ik_this = 0; ik_this < nk_this; ik_this++)
                        {
                            global::profiler.start("build_real_space_exx_6", "Hexx IJ -> 2D block");
                            Hexx_nao_nao.zero_out();
                            const int ik = iks_all[pid * nk_max + ik_this];
                            // global::ofs_myid << "nk_this " << nk_this << " pid " << pid << " ik_this " << ik_this << " ik " << ik << std::endl;
                            const auto& kfrac = kfrac_target[ik];
                            const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                                fourier = [kfrac](const atom_t I, const atom_t J, const Vector3_Order<int> &R)
                                {
                                    const auto ang = (kfrac * R) * TWO_PI;
                                    return complex<double>{std::cos(ang), std::sin(ang)};
                                };
                            if (use_complex_exx_r)
                                collect_block_from_IJ_storage_matrix_transform(
                                    Hexx_nao_nao, desc_nao_nao, this->atbasis_wfc,
                                    this->atbasis_wfc, fourier, exx_is_local_cplx);
                            else
                                collect_block_from_IJ_storage_matrix_transform(
                                    Hexx_nao_nao, desc_nao_nao, this->atbasis_wfc,
                                    this->atbasis_wfc, fourier, exx_is_local);
                            global::profiler.stop("build_real_space_exx_6");

                            std::vector<complex<double>> dummy(1);
                            global::profiler.start("build_real_space_exx_7", "Rotate Hexx ij -> KS");
                            if (pid == comm_h.myid)
                            {
                                // global::ofs_myid << "isp " << isp << " ik " << ik << std::endl;
                                const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                                const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                                // processing the k-point eigenvector on this process
                                ScalapackConnector::pgemm_f('C', 'N', n_bands, n_aos, n_aos, 1.0,
                                                            wfc_bra.c, 1, 1, desc_nao_nband_fb.desc,
                                                            Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                                            0.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                                ScalapackConnector::pgemm_f('N', 'N', n_bands, n_bands, n_aos, -1.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                            wfc_ket.c, 1, 1, desc_nao_nband_fb.desc,
                                                            0.0,
                                                            Hexx_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc);
                                if (this->exx_KS.count(isp) == 0 || this->exx_KS.at(isp).count(ik) == 0)
                                {
                                    this->exx_KS[isp][ik] = Hexx_nband_nband_fb.copy();
                                    this->exx_KS[isp][ik] = C_ZERO;
                                    for (int ib = 0; ib != n_bands; ib++)
                                        this->Eexx[isp][ik][ib] = 0.0;
                                }
                                this->exx_KS[isp][ik] += Hexx_nband_nband_fb;
                                // cout << "Hexx_nband_nband_fb isp " << isp  << " ik " << ik << endl << Hexx_nband_nband_fb;
                                for (int ib = 0; ib != n_bands; ib++)
                                    this->Eexx[isp][ik][ib] += Hexx_nband_nband_fb(ib, ib).real();
                            }
                            else
                            {
                                // processing the k-point eigenvector at other processes
                                ScalapackConnector::pgemm_f('C', 'N', n_bands, n_aos, n_aos, 1.0,
                                                            dummy.data(), 1, 1, desc_nao_nband_fb.desc,
                                                            Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                                            0.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                                ScalapackConnector::pgemm_f('N', 'N', n_bands, n_bands, n_aos, -1.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                            dummy.data(), 1, 1, desc_nao_nband_fb.desc,
                                                            0.0,
                                                            Hexx_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc);
                            }
                            global::profiler.stop("build_real_space_exx_7");
                        }
                    }
                }
	                else // !is_mf_eigvec_k_distributed_
	                {
	                    // Everything will be collected to rank 0
	                    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);
	                    auto Hexx_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);

                    // Each process have all KS eigenvectors, extract local blocks
                    for (size_t ik = 0; ik < kfrac_target.size(); ik++)
                    {
                        global::profiler.start("build_real_space_exx_6", "Hexx IJ -> 2D block");
                        Hexx_nao_nao.zero_out();
                        const auto& kfrac = kfrac_target[ik];
                        const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                            fourier = [kfrac](const atom_t I, const atom_t J, const Vector3_Order<int> &R)
                            {
                                const auto ang = (kfrac * R) * TWO_PI;
                                return complex<double>{std::cos(ang), std::sin(ang)};
                            };
                        if (use_complex_exx_r)
                            collect_block_from_IJ_storage_matrix_transform(Hexx_nao_nao, desc_nao_nao,
                                    this->atbasis_wfc, this->atbasis_wfc, fourier, exx_is_local_cplx);
                        else
                            collect_block_from_IJ_storage_matrix_transform(Hexx_nao_nao, desc_nao_nao,
                                    this->atbasis_wfc, this->atbasis_wfc, fourier, exx_is_local);
	                        global::profiler.stop("build_real_space_exx_6");
	                        // global::lib_printf("%s\n", str(Hexx_nao_nao).c_str());
	                        if (use_root_dense_projection)
	                        {
	                            global::profiler.start(
	                                "build_real_space_exx_7",
	                                "Rotate Hexx ij -> KS with root dense IBZ projection");
	                            auto Hexx_nao_nao_fb =
	                                init_local_mat<complex<double>>(desc_nao_nao_fb, MAJOR::COL);
	                            ScalapackConnector::pgemr2d_f(
	                                n_aos, n_aos,
	                                Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
	                                Hexx_nao_nao_fb.ptr(), 1, 1, desc_nao_nao_fb.desc,
	                                desc_nao_nao_fb.ictxt());

	                            ComplexMatrix Hexx_nband_nband_dense;
	                            if (comm_h.is_root())
	                            {
	                                ComplexMatrix Hexx_nao_nao_dense(n_aos, n_aos);
	                                for (int iao = 0; iao != n_aos; ++iao)
	                                {
	                                    for (int jao = 0; jao != n_aos; ++jao)
	                                    {
	                                        Hexx_nao_nao_dense(iao, jao) =
	                                            Hexx_nao_nao_fb(iao, jao);
	                                    }
	                                }
	                                const auto &wfc_bra =
	                                    wfc_target.at(isp).at(ispn_bra).at(ik);
	                                const auto &wfc_ket =
	                                    wfc_target.at(isp).at(ispn_ket).at(ik);
	                                Hexx_nband_nband_dense =
	                                    (-1.0) * (conj(wfc_bra) * Hexx_nao_nao_dense
	                                              * transpose(wfc_ket, false));
	                            }
	                            broadcast_ComplexMatrix(Hexx_nband_nband_dense, 0, comm_h.comm);
	                            global::profiler.stop("build_real_space_exx_7");

	                            global::profiler.start("build_real_space_exx_8",
	                                                   "Collect Eexx to root process");
	                            if (this->exx_KS.count(isp) == 0
	                                || this->exx_KS.at(isp).count(ik) == 0)
	                            {
	                                this->exx_KS[isp][ik] = Matz(
	                                    n_bands, n_bands, Hexx_nband_nband_dense.c,
	                                    MAJOR::ROW, MAJOR::COL);
	                                if (comm_h.is_root())
	                                {
	                                    for (int ib = 0; ib != n_bands; ++ib)
	                                    {
	                                        this->Eexx[isp][ik][ib] = 0.0;
	                                    }
	                                }
	                            }
	                            else
	                            {
	                                this->exx_KS[isp][ik] += Matz(
	                                    n_bands, n_bands, Hexx_nband_nband_dense.c,
	                                    MAJOR::ROW, MAJOR::COL);
	                            }
	                            if (comm_h.is_root())
	                            {
	                                for (int ib = 0; ib != n_bands; ++ib)
	                                {
	                                    this->Eexx[isp][ik][ib] +=
	                                        Hexx_nband_nband_dense(ib, ib).real();
	                                }
	                            }
	                            global::profiler.stop("build_real_space_exx_8");
	                            continue;
	                        }
	                        const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                        blacs_ctxt_h.barrier();
                        const auto wfc_bra_block = get_local_mat(wfc_bra.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                        Matz wfc_ket_block;
                        if (ispn_ket != ispn_bra)
                        {
                            const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                            wfc_ket_block = get_local_mat(wfc_ket.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                        }
                        else
                        {
                            wfc_ket_block = wfc_bra_block;
                        }
                        // global::lib_printf("%s\n", str(wfc_block).c_str());
                        // global::lib_printf("%s\n", desc_nao_nao.info_desc().c_str());
                        // global::lib_printf("%s\n", desc_nband_nao.info_desc().c_str());
                        global::profiler.start("build_real_space_exx_7", "Rotate Hexx ij -> KS");
                        ScalapackConnector::pgemm_f('N', 'N', n_bands, n_aos, n_aos, 1.0,
                                                    wfc_bra_block.ptr(), 1, 1, desc_nband_nao.desc,
                                                    Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                                    0.0,
                                                    temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                        ScalapackConnector::pgemm_f('N', 'C', n_bands, n_bands, n_aos, -1.0,
                                                    temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                    wfc_ket_block.ptr(), 1, 1, desc_nband_nao.desc,
                                                    0.0,
                                                    Hexx_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                        global::profiler.stop("build_real_space_exx_7");

                        // collect to master
                        global::profiler.start("build_real_space_exx_8", "Collect Eexx to root process");
                        global::ofs_myid << "before pgemr2d_f" << std::endl;
                        ScalapackConnector::pgemr2d_f(n_bands, n_bands,
                                                    Hexx_nband_nband.ptr(), 1, 1, desc_nband_nband.desc,
                                                    Hexx_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc,
                                                    desc_nband_nband_fb.ictxt());
                        global::ofs_myid << "after pgemr2d_f" << std::endl;
                        if (this->exx_KS.count(isp) == 0 || this->exx_KS.at(isp).count(ik) == 0)
                        {
                            this->exx_KS[isp][ik] = Hexx_nband_nband_fb.copy();
                            this->exx_KS[isp][ik] = C_ZERO;
                            if (blacs_ctxt_h.myid == 0)
                                for (int ib = 0; ib != n_bands; ib++)
                                    this->Eexx[isp][ik][ib] = 0.0;
                        }
                        this->exx_KS[isp][ik] += Hexx_nband_nband_fb;
                        // cout << "Hexx_nband_nband_fb isp " << isp  << " ik " << ik << endl << Hexx_nband_nband_fb;
                        if (blacs_ctxt_h.myid == 0)
                        {
                            for (int ib = 0; ib != n_bands; ib++)
                                this->Eexx[isp][ik][ib] += Hexx_nband_nband_fb(ib, ib).real();
                        }
                        global::ofs_myid << "after Hexx_nband_nband_fb assign" << std::endl;
                        global::profiler.stop("build_real_space_exx_8");
                    }
                }
            }
        }
    }
    global::ofs_myid << "Done Exx::build_KS_blacs" << std::endl;
}

void Exx::build_KS_kgrid()
{
    this->build_KS(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {});
}

// void Exx::build_KS0_kgrid()
// {
//     this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_);
// }

void Exx::build_KS_band(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_band,
                        const std::vector<Vector3_Order<double>> &kfrac_band,
                        const AtomPairBvKRemap<atom_t> &bvk_remap)
{
    this->build_KS(wfc_band, kfrac_band, bvk_remap);
}

void Exx::build_KS_kgrid_blacs(const BlacsCtxtHandler &blacs_ctxt_h)
{
    this->build_KS_blacs(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {}, blacs_ctxt_h);
}

// void Exx::build_KS0_kgrid_blacs()
// {
//     this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_);
// }

void Exx::build_KS_band_blacs(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_band,
                              const std::vector<Vector3_Order<double>> &kfrac_band,
                              const AtomPairBvKRemap<atom_t> &bvk_remap,
                              const BlacsCtxtHandler &blacs_ctxt_h)
{
    this->build_KS_blacs(wfc_band, kfrac_band, bvk_remap, blacs_ctxt_h);
}

void Exx::reset_rspace()
{
    this->exx_IJR.clear();
    this->is_rspace_built_ = false;
}

void Exx::reset_kspace()
{
    this->exx_KS.clear();
    this->Eexx.clear();
    this->is_kspace_built_ = false;
}

} /* end of namespace librpa_int */
