// Public API headers
#include "librpa_enums.h"

// Internal headers
#include "../utils/error.h"
#include "../utils/profiler.h"
#include "../io/global_io.h"
#include "../math/rsh.h"
#include "../math/symmetry.h"
#include "dataset_helper.h"

#include <algorithm>

namespace librpa_int
{

namespace
{

std::vector<SpeciesBasisLayout> type_layouts_from_map(
    const std::map<int, SpeciesBasisLayout> &layouts)
{
    int max_type = -1;
    for (const auto &entry : layouts)
    {
        if (entry.first < 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Atomic basis shell layout uses a negative atom type");
        }
        max_type = std::max(max_type, entry.first);
    }
    std::vector<SpeciesBasisLayout> by_type(static_cast<std::size_t>(max_type + 1));
    for (const auto &entry : layouts)
    {
        by_type[static_cast<std::size_t>(entry.first)] = entry.second;
    }
    return by_type;
}

bool append_type_layout_candidates(std::vector<std::vector<SpeciesBasisLayout>> &candidates,
                                   const std::map<int, SpeciesBasisLayout> &layouts)
{
    bool appended_any = false;
    for (const auto &entry : layouts)
    {
        if (!entry.second.is_shell_available())
        {
            continue;
        }
        const int atom_type = entry.first;
        if (atom_type < 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Atomic basis shell layout uses a negative atom type");
        }
        if (candidates.size() <= static_cast<std::size_t>(atom_type))
        {
            candidates.resize(static_cast<std::size_t>(atom_type + 1));
        }
        auto &type_candidates = candidates[static_cast<std::size_t>(atom_type)];
        const auto duplicate = std::find_if(
            type_candidates.begin(), type_candidates.end(),
            [&entry](const SpeciesBasisLayout &candidate) {
                return same_species_basis_layout(candidate, entry.second);
            });
        if (duplicate == type_candidates.end())
        {
            type_candidates.push_back(entry.second);
            appended_any = true;
        }
    }
    return appended_any;
}

void sync_input_symmetry_shell_layouts_from_cached_layouts(Dataset &ds)
{
    auto &ctx = ds.input_symmetry_ctx;
    if (!ds.atoms.types.empty())
    {
        ctx.atom_to_type = ds.atoms.types;
    }
    if (ctx.atom_to_type.empty())
    {
        return;
    }

    if (type_layouts_have_shells(ds.basis_wfc_layouts))
    {
        ctx.ao_type_layouts = type_layouts_from_map(ds.basis_wfc_layouts);
        ctx.ao_shell_layout_available = !ctx.ao_type_layouts.empty();
        ctx.ao_lmax = ds.basis_wfc.get_max_l();
    }

    const bool has_basis_abf_layouts =
        type_layouts_have_shells(ds.basis_aux_layouts)
        || type_layouts_have_shells(ds.basis_aux_shrink_layouts);
    if (has_basis_abf_layouts)
    {
        ctx.abf_type_layout_candidates.clear();
        ctx.abf_shell_layout_available = false;
        ctx.abf_lmax = -1;
    }

    int abf_lmax = -1;
    bool updated_abf_layouts = false;
    if (type_layouts_have_shells(ds.basis_aux_layouts))
    {
        updated_abf_layouts =
            append_type_layout_candidates(ctx.abf_type_layout_candidates, ds.basis_aux_layouts)
            || updated_abf_layouts;
        abf_lmax = std::max(abf_lmax, ds.basis_aux.get_max_l());
    }
    if (type_layouts_have_shells(ds.basis_aux_shrink_layouts))
    {
        updated_abf_layouts =
            append_type_layout_candidates(ctx.abf_type_layout_candidates,
                                          ds.basis_aux_shrink_layouts)
            || updated_abf_layouts;
        abf_lmax = std::max(abf_lmax, ds.basis_aux_shrink.get_max_l());
    }
    if (updated_abf_layouts)
    {
        ctx.abf_shell_layout_available = !ctx.abf_type_layout_candidates.empty();
        ctx.abf_lmax = abf_lmax;
    }
}

} // namespace

void initialize_input_symmetry_context(Dataset &ds)
{
    auto &ctx = ds.input_symmetry_ctx;
    if (ctx.rspace_operations.empty())
    {
        return;
    }

    sync_input_symmetry_shell_layouts_from_cached_layouts(ds);

    const int lmax = std::max(ctx.ao_lmax, ctx.abf_lmax);
    if (lmax < 0)
    {
        return;
    }
    if (!ctx.lattice_available)
    {
        throw LIBRPA_RUNTIME_ERROR("Cannot generate input symmetry shell rotations without lattice vectors");
    }

    auto basis_convention =
        is_basis_convention_set(ctx.basis_convention) ? ctx.basis_convention : ds.basis_convention;
    if (!is_basis_convention_set(basis_convention))
    {
        throw LIBRPA_RUNTIME_ERROR("Cannot initialize input symmetry context without basis convention");
    }
    ctx.basis_convention = basis_convention;
    ds.basis_convention = basis_convention;

    for (auto &op : ctx.rspace_operations)
    {
        const auto cartesian_rotation =
            fractional_rotation_to_cartesian(op, ctx.lattice_vectors).Inverse();
        for (int l = 0; l <= lmax; ++l)
        {
            op.shell_rotations[l] =
                real_spherical_harmonic_rotation_matrix(
                    cartesian_rotation,
                    l,
                    basis_convention.order,
                    basis_convention.coeff_m_negative,
                    basis_convention.coeff_m_positive);
        }
    }
}

void initialize_ds_tfgrids(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_tfgrids");
    ds.tfg.reset(opts.nfreq);
    double emin = opts.tfgrids_freq_min;
    double eintv = opts.tfgrids_freq_interval;
    double emax = opts.tfgrids_freq_max;
    double tmin = opts.tfgrids_time_min;
    double tintv = opts.tfgrids_time_interval;
    double regulation = opts.minimax_regulation;
    if (opts.tfgrids_type == LIBRPA_TFGRID_MINIMAX)
    {
        double emin_mf, emax_mf;
        ds.mf.get_E_min_max(emin_mf, emax_mf);
        emax = opts.minimax_emax > 0 ? opts.minimax_emax : emax_mf;
        emin = opts.minimax_emin > 0 ? opts.minimax_emin : emin_mf;
    }
    ds.tfg.generate(opts.tfgrids_type, emin, eintv, emax, tmin, tintv, regulation);
    global::profiler.stop("initialize_ds_tfgrids");
}

static void collect_atpairs_all(Dataset &ds)
{
    const auto &comm_h = ds.comm_h;
    const auto &atpairs_local = ds.atpairs_local;
    const int np_this = atpairs_local.size();
    std::vector<int> np_all(comm_h.nprocs, 0);
    np_all[comm_h.myid] = np_this;
    int np_max;
    comm_h.allreduce(&np_this, &np_max, 1, MPI_MAX);
    comm_h.allreduce(MPI_IN_PLACE, np_all.data(), comm_h.nprocs, MPI_SUM);
    if (np_max == 0) return;
    std::vector<size_t> pairs_all(comm_h.nprocs * np_max * 2, 0);
    const int st = comm_h.myid * np_max * 2;
    for (int ip = 0; ip < np_this; ip++)
    {
        pairs_all[st + ip * 2] = atpairs_local[ip].first;
        pairs_all[st + ip * 2 + 1] = atpairs_local[ip].second;
    }
    comm_h.allreduce(MPI_IN_PLACE, pairs_all.data(), pairs_all.size(), MPI_SUM);
    for (int pid = 0; pid < comm_h.nprocs; pid++)
    {
        if (pid == comm_h.myid)
        {
            std::set<atpair_t> atpairs_this(atpairs_local.cbegin(), atpairs_local.cend());
            ds.atpairs_unique_all.emplace(pid, atpairs_this);
        }
        else
        {
            const int np_this = np_all[pid];
            std::set<atpair_t> atpairs_this;
            const int st = pid * np_max * 2;
            for (int ip = 0; ip < np_this; ip++)
            {
                atpairs_this.insert({pairs_all[st + ip * 2], pairs_all[st + ip * 2 + 1]});
            }
            ds.atpairs_unique_all.emplace(pid, atpairs_this);
        }
    }
}

void initialize_ds_atpairs_local(Dataset &ds, LibrpaParallelRouting routing)
{
    global::profiler.start(__FUNCTION__);

    initialize_input_symmetry_context(ds);

    ds.atpairs_local.clear();
    const int n_atoms_basis_wfc = ds.basis_wfc.n_atoms;
    const int n_atoms_basis_aux = ds.basis_aux.n_atoms;
    const int n_atoms_struc = ds.atoms.size();
    const int n_atoms = n_atoms_struc > 0? n_atoms_struc : std::max(n_atoms_basis_aux, n_atoms_basis_wfc);
    if (n_atoms == 0)
        throw LIBRPA_RUNTIME_ERROR("Number of atoms can not be extracted, please set structure or basis first");

    if (routing == LIBRPA_ROUTING_AUTO)
    {
        throw LIBRPA_RUNTIME_ERROR("internal error: routing should be decided before initialize_ds_atpairs_local, not AUTO");
    }
    else if(routing == LIBRPA_ROUTING_ATOMPAIR || routing == LIBRPA_ROUTING_LIBRI)
    {
        auto tri_local_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
            ds.blacs_h.myprow, ds.blacs_h.mypcol);
        for (const auto &p: tri_local_atpair)
            ds.atpairs_local.emplace_back(p);
    }
    else
    {
        ds.atpairs_local = generate_atom_pair_from_nat(n_atoms, false);
    }
    collect_atpairs_all(ds);
    global::profiler.stop(__FUNCTION__);
}

void initialize_ds_exx(Dataset &ds, const LibrpaOptions &opts) noexcept
{
    global::profiler.start("initialize_ds_exx");
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    ds.p_exx = std::make_unique<librpa_int::Exx>(ds.mf, ds.basis_wfc, ds.pbc, ds.input_symmetry_ctx,
                                                 ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                 is_eigvec_k_distributed);
    ds.p_exx->libri_threshold_C = opts.libri_exx_threshold_C;
    ds.p_exx->libri_threshold_D = opts.libri_exx_threshold_D;
    ds.p_exx->libri_threshold_V = opts.libri_exx_threshold_V;
    global::profiler.stop("initialize_ds_exx");
}

void initialize_ds_chi0(Dataset &ds, const LibrpaOptions &opts) noexcept
{
    global::profiler.start("initialize_ds_chi0");
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    if (opts.use_shrink_abfs == LIBRPA_SWITCH_ON && opts.use_shrink_chi == LIBRPA_SWITCH_ON)
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux_shrink, ds.pbc,
                                                       ds.input_symmetry_ctx,
                                                       ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                       is_eigvec_k_distributed);
    else
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc,
                                                       ds.input_symmetry_ctx,
                                                       ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                       is_eigvec_k_distributed);
    ds.p_chi0->gf_threshold = opts.gf_threshold;
    ds.p_chi0->libri_collect_s0_chunk = opts.libri_chi0_collect_s0_chunk;
    ds.p_chi0->libri_collect_max_bytes = opts.libri_chi0_collect_max_bytes;
    ds.p_chi0->nbands_G = opts.n_bands_chi0;
    ds.p_chi0->libri_threshold_C = opts.libri_chi0_threshold_C;
    ds.p_chi0->libri_threshold_G = opts.libri_chi0_threshold_G;
    global::profiler.stop("initialize_ds_chi0");
}

void initialize_ds_g0w0(Dataset &ds, const LibrpaOptions &opts) noexcept
{
    global::profiler.start("initialize_ds_g0w0");
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    // global::ofs_myid << "is_eigvec_k_distributed " << is_eigvec_k_distributed << std::endl;
    ds.p_g0w0 = std::make_unique<librpa_int::G0W0>(ds.mf, ds.basis_wfc, ds.pbc,
                                                   ds.input_symmetry_ctx, ds.tfg,
                                                   ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                   is_eigvec_k_distributed);
    ds.p_g0w0->libri_threshold_C = opts.libri_g0w0_threshold_C;
    ds.p_g0w0->libri_threshold_G = opts.libri_g0w0_threshold_G;
    ds.p_g0w0->libri_threshold_Wc = opts.libri_g0w0_threshold_Wc;
    ds.p_g0w0->output_dir = opts.output_dir;
    ds.p_g0w0->output_sigc_ks_if = opts.output_gw_sigc_ks_if == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat = opts.output_gw_sigc_mat == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_rt = opts.output_gw_sigc_mat_rt == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_rf = opts.output_gw_sigc_mat_rf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_wc_rf = opts.output_wc_rf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->ifreq_output_wc_start = opts.ifreq_output_wc_start;
    ds.p_g0w0->ifreq_output_wc_end = opts.ifreq_output_wc_end;
    global::profiler.stop("initialize_ds_g0w0");
}

void initialize_ds_headwing(Dataset &ds, const LibrpaOptions &opts, const bool need_wing)
{
    if (opts.replace_w_head != LIBRPA_SWITCH_ON ||
        (opts.option_dielect_func != 3 && opts.option_dielect_func != 4))
    {
        return;
    }

    global::profiler.start("initialize_ds_headwing");

    if (ds.p_headwing && (!need_wing || ds.p_headwing->has_wing()))
    {
        global::profiler.stop("initialize_ds_headwing");
        return;
    }

    if (ds.p_headwing && need_wing)
    {
        const auto &headwing_cs =
            opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.cs_data_shrink : ds.cs_data;
        ds.p_headwing->cal_wing(headwing_cs, opts.sqrt_coulomb_threshold, ds.vq);
        if (global::should_output(LIBRPA_VERBOSE_DEBUG))
            ds.p_headwing->test_wing();
        global::profiler.stop("initialize_ds_headwing");
        return;
    }

    if (ds.velocity_matrix.empty())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "analytic head/wing requested but velocity matrix is not set");
    }

    MeanField &mf = ds.mf;
    if (!mf.initialized())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing meanfield is not initialized");

    if (static_cast<int>(ds.pbc.kfrac_list.size()) != mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing k-point list is inconsistent with meanfield");

    const auto &headwing_basis_aux =
        opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.basis_aux_shrink : ds.basis_aux;
    if (!headwing_basis_aux.initialized())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing auxiliary basis is not initialized");

    const auto &freqs = ds.tfg.get_freq_nodes();
    ds.p_headwing = std::make_unique<diele_func>(
        mf, ds.velocity_matrix, ds.pbc.kfrac_list, ds.basis_wfc,
        headwing_basis_aux, freqs, mf.get_n_aos(), mf.get_n_states(),
        mf.get_n_spins(), headwing_basis_aux.nb_total, ds.pbc, ds.comm_h, ds.blacs_h);
    ds.p_headwing->use_2d_dielectric = opts.use_2d_dielectric == LIBRPA_SWITCH_ON;
    ds.p_headwing->use_soc = mf.get_n_spinor() > 1;
    ds.p_headwing->debug = global::should_output(LIBRPA_VERBOSE_DEBUG);
    ds.p_headwing->init(opts.sqrt_coulomb_threshold, ds.vq);
    ds.p_headwing->cal_head();
    ds.epsmacs_imagfreq = ds.p_headwing->get_head_vec();
    ds.omegas_imagfreq = freqs;
    ds.p_headwing->test_head();

    if (need_wing)
    {
        const auto &headwing_cs =
            opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.cs_data_shrink : ds.cs_data;
        ds.p_headwing->cal_wing(headwing_cs, opts.sqrt_coulomb_threshold, ds.vq);
        if (global::should_output(LIBRPA_VERBOSE_DEBUG))
            ds.p_headwing->test_wing();
    }

    global::profiler.stop("initialize_ds_headwing");
}

}
