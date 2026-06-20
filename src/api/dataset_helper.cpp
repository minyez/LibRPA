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
#include <cmath>
#include <stdexcept>
#include <string>

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

std::array<double, 3> coord_frac_array(const coord_t &coord)
{
    return {coord.x, coord.y, coord.z};
}

Vector3_Order<double> coord_frac_vector(const std::map<atom_t, std::array<double, 3>> &coords,
                                        const atom_t atom)
{
    const auto iter = coords.find(atom);
    if (iter == coords.end())
    {
        throw std::runtime_error("Input symmetry is missing a fractional atom coordinate");
    }
    return {iter->second[0], iter->second[1], iter->second[2]};
}

SpaceGroupSymOps<SpaceGroupSymOp> build_kstar_operations_with_time_reversal(
    const SpaceGroupSymOps<InputSymmetryOperation> &operations)
{
    SpaceGroupSymOps<SpaceGroupSymOp> kstar_operations;
    kstar_operations.reserve(2 * operations.size());
    for (const auto &op : operations)
    {
        SpaceGroupSymOp base_op;
        base_op.rotation = op.rotation;
        base_op.translation = op.translation;
        base_op.use_row_convention = op.use_row_convention;
        kstar_operations.push_back(base_op);
    }
    for (const auto &op : operations)
    {
        SpaceGroupSymOp tr_op;
        tr_op.rotation = op.rotation * -1.0;
        tr_op.translation = op.translation;
        tr_op.use_row_convention = op.use_row_convention;
        kstar_operations.push_back(tr_op);
    }
    return kstar_operations;
}

bool same_fractional_kpoint(const Vector3_Order<double> &lhs,
                            const Vector3_Order<double> &rhs,
                            const double tol)
{
    return nearly_integer_vector(lhs - rhs, tol);
}

bool is_identity_rotation(const Matrix3 &rotation, const double tol)
{
    return std::abs(rotation.e11 - 1.0) < tol
           && std::abs(rotation.e22 - 1.0) < tol
           && std::abs(rotation.e33 - 1.0) < tol
           && std::abs(rotation.e12) < tol
           && std::abs(rotation.e13) < tol
           && std::abs(rotation.e21) < tol
           && std::abs(rotation.e23) < tol
           && std::abs(rotation.e31) < tol
           && std::abs(rotation.e32) < tol;
}

int find_identity_input_symmetry_operation(
    const SpaceGroupSymOps<InputSymmetryOperation> &operations)
{
    for (std::size_t isym = 0; isym != operations.size(); ++isym)
    {
        const auto &op = operations.at(isym);
        if (is_identity_rotation(op.rotation, 1e-8)
            && nearly_integer_vector(op.translation, 1e-8))
        {
            return static_cast<int>(isym);
        }
    }
    throw std::runtime_error("Input symmetry operations do not contain the identity operation");
}

void build_input_symmetry_pbc_index_kstars(
    SymmetryContext &ctx,
    const std::vector<Vector3_Order<double>> &kpoints)
{
    const int identity_isym = find_identity_input_symmetry_operation(ctx.rspace_operations);

    ctx.kstars.clear();
    ctx.kstar_member_fold_G.clear();
    ctx.kstars.reserve(kpoints.size());
    for (std::size_t ik = 0; ik != kpoints.size(); ++ik)
    {
        InputSymmetryKStarMember member;
        member.isym = identity_isym;
        member.k_bz = kpoints[ik];

        InputSymmetryKStar star;
        star.star_index = static_cast<int>(ik);
        star.k_ibz = kpoints[ik];
        star.members.push_back(std::move(member));

        ctx.kstar_member_fold_G[{star.star_index, 0}] = {0, 0, 0};
        ctx.kstars.push_back(std::move(star));
    }
}

std::vector<Vector3_Order<double>> build_uniform_kmesh_frac(const Vector3_Order<int> &period)
{
    std::vector<Vector3_Order<double>> kpoints;
    kpoints.reserve(static_cast<std::size_t>(period.x * period.y * period.z));
    for (int i = 0; i != period.x; ++i)
        for (int j = 0; j != period.y; ++j)
            for (int k = 0; k != period.z; ++k)
                kpoints.push_back({static_cast<double>(i) / period.x,
                                   static_cast<double>(j) / period.y,
                                   static_cast<double>(k) / period.z});
    return kpoints;
}

std::vector<Vector3_Order<double>> full_kpoints_frac_from_pbc(const PeriodicBoundaryData &pbc)
{
    if (static_cast<int>(pbc.klist.size()) < pbc.get_n_cells_bvk())
    {
        return build_uniform_kmesh_frac(pbc.period);
    }
    return pbc.kfrac_list_full.empty() ? pbc.kfrac_list : pbc.kfrac_list_full;
}

std::vector<Vector3_Order<double>> coul_kpoints_frac_from_pbc(const PeriodicBoundaryData &pbc)
{
    if (pbc.klist_coul.empty())
    {
        return pbc.kfrac_list;
    }

    std::vector<Vector3_Order<double>> kpoints;
    kpoints.reserve(pbc.klist_coul.size());
    for (const auto &k : pbc.klist_coul)
    {
        auto kfrac = pbc.latvec * k;
        if (std::abs(kfrac.x) < 1e-8)
        {
            kfrac.x = 0.0;
        }
        if (std::abs(kfrac.y) < 1e-8)
        {
            kfrac.y = 0.0;
        }
        if (std::abs(kfrac.z) < 1e-8)
        {
            kfrac.z = 0.0;
        }
        kpoints.emplace_back(kfrac);
    }
    return kpoints;
}

atom_t find_input_symmetry_atom_target(const SymmetryContext &ctx,
                                       const atom_t atom_from,
                                       const int spatial_isym,
                                       Vector3_Order<int> &return_lattice)
{
    const auto &op = ctx.rspace_operations.at(static_cast<std::size_t>(spatial_isym));
    const auto atom_type = ctx.atom_to_type.at(atom_from);
    const auto coord_from =
        restrict_fractional_coordinate(coord_frac_vector(ctx.input_coord_frac, atom_from));
    const auto transformed = apply_space_group_symmetry_operation(op, coord_from);

    atom_t matched_atom = static_cast<atom_t>(-1);
    for (const auto &[atom_to, type_to] : ctx.atom_to_type)
    {
        if (type_to != atom_type)
        {
            continue;
        }
        const auto coord_to =
            restrict_fractional_coordinate(coord_frac_vector(ctx.input_coord_frac, atom_to));
        const auto diff = transformed - coord_to;
        if (!nearly_integer_vector(diff, 1e-5))
        {
            continue;
        }
        if (matched_atom != static_cast<atom_t>(-1))
        {
            throw std::runtime_error("Input symmetry atom mapping is ambiguous");
        }
        matched_atom = atom_to;
        return_lattice = round_to_integer_vector(diff);
    }
    if (matched_atom == static_cast<atom_t>(-1))
    {
        throw std::runtime_error("Input symmetry failed to map an atom under a k-star operation");
    }
    return matched_atom;
}

void populate_input_symmetry_kstar_member_rotations(SymmetryContext &ctx,
                                                    const InputSymmetryKStar &star,
                                                    InputSymmetryKStarMember &member,
                                                    const int lmax)
{
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const int spatial_isym = member.isym >= nsym_space ? member.isym - nsym_space : member.isym;
    if (spatial_isym < 0 || spatial_isym >= nsym_space)
    {
        throw std::runtime_error("Generated input-symmetry k-star member has an invalid symmetry index");
    }
    const auto &operation = ctx.rspace_operations.at(static_cast<std::size_t>(spatial_isym));

    member.atom_rotations.clear();
    member.atom_rotations.reserve(ctx.atom_to_type.size());
    for (const auto &[atom_from, atom_type] : ctx.atom_to_type)
    {
        Vector3_Order<int> return_lattice{0, 0, 0};
        const atom_t atom_to =
            find_input_symmetry_atom_target(ctx, atom_from, spatial_isym, return_lattice);
        const auto inserted = ctx.kspace_return_lattice.emplace(
            std::make_pair(static_cast<int>(atom_from), spatial_isym), return_lattice);
        if (!inserted.second && inserted.first->second != return_lattice)
        {
            throw std::runtime_error("Input symmetry generated inconsistent atom return lattices");
        }

        InputSymmetryKAtomRotation atom_rotation;
        atom_rotation.atom_from = static_cast<int>(atom_from);
        atom_rotation.atom_to = static_cast<int>(atom_to);
        atom_rotation.atom_type = atom_type;
        atom_rotation.lmax = lmax;
        atom_rotation.shell_rotations =
            build_input_symmetry_kspace_shell_rotations(operation,
                                                        ctx.lattice_vectors,
                                                        lmax,
                                                        ctx.basis_convention,
                                                        star.k_ibz,
                                                        member.k_bz,
                                                        coord_frac_vector(ctx.input_coord_frac, atom_from),
                                                        coord_frac_vector(ctx.input_coord_frac, atom_to),
                                                        return_lattice);
        member.atom_rotations.push_back(std::move(atom_rotation));
    }
}

void generate_input_symmetry_kstars_from_pbc(SymmetryContext &ctx,
                                             const PeriodicBoundaryData &pbc)
{
    const auto full_kpoints = full_kpoints_frac_from_pbc(pbc);
    const auto coul_kpoints = coul_kpoints_frac_from_pbc(pbc);
    const bool scf_kpoints_cover_full_grid =
        static_cast<int>(pbc.kfrac_list.size()) == pbc.get_n_cells_bvk();
    const auto generated_stars = build_kpoint_stars(
        full_kpoints, build_kstar_operations_with_time_reversal(ctx.rspace_operations),
        coul_kpoints, 1e-5);
    if (generated_stars.size() != coul_kpoints.size())
    {
        if (scf_kpoints_cover_full_grid)
        {
            build_input_symmetry_pbc_index_kstars(ctx, pbc.kfrac_list);
            return;
        }
        if (coul_kpoints.size() == full_kpoints.size())
        {
            build_input_symmetry_pbc_index_kstars(ctx, coul_kpoints);
            return;
        }
        throw std::runtime_error("Generated input-symmetry k-star count does not match Coulomb k-points");
    }

    ctx.kstars.clear();
    ctx.kstar_member_fold_G.clear();
    std::vector<bool> used(generated_stars.size(), false);
    for (std::size_t ik_ibz = 0; ik_ibz != coul_kpoints.size(); ++ik_ibz)
    {
        int matched_star_index = -1;
        for (std::size_t istar = 0; istar != generated_stars.size(); ++istar)
        {
            if (used[istar])
            {
                continue;
            }
            const auto &star = generated_stars[istar];
            const auto &representative =
                star.members.at(static_cast<std::size_t>(star.representative_k_index)).kpoint;
            if (same_fractional_kpoint(representative, coul_kpoints[ik_ibz], 1e-5))
            {
                matched_star_index = static_cast<int>(istar);
                break;
            }
        }
        if (matched_star_index < 0)
        {
            if (scf_kpoints_cover_full_grid)
            {
                build_input_symmetry_pbc_index_kstars(ctx, pbc.kfrac_list);
                return;
            }
            throw std::runtime_error("Failed to order generated input-symmetry k-stars by IBZ k-points");
        }

        used[static_cast<std::size_t>(matched_star_index)] = true;
        const auto &generated_star = generated_stars[static_cast<std::size_t>(matched_star_index)];
        InputSymmetryKStar star;
        star.star_index = static_cast<int>(ik_ibz);
        star.k_ibz = coul_kpoints[ik_ibz];
        star.members.reserve(generated_star.members.size());
        for (std::size_t imember = 0; imember != generated_star.members.size(); ++imember)
        {
            InputSymmetryKStarMember member;
            member.isym = generated_star.sym_mappings.at(imember).isym;
            member.k_bz = generated_star.members[imember].kpoint;
            ctx.kstar_member_fold_G[{star.star_index, static_cast<int>(imember)}] =
                generated_star.sym_mappings.at(imember).fold_G;
            star.members.push_back(std::move(member));
        }
        ctx.kstars.push_back(std::move(star));
    }
}

void sync_input_symmetry_structure_from_dataset(Dataset &ds)
{
    auto &ctx = ds.symmetry_context;
    if (!ds.atoms.types.empty())
    {
        ctx.atom_to_type = ds.atoms.types;
    }
    if (!ds.atoms.coords_frac.empty())
    {
        ctx.input_coord_frac.clear();
        for (const auto &[atom, coord] : ds.atoms.coords_frac)
        {
            ctx.input_coord_frac[atom] = coord_frac_array(coord);
        }
    }
}

void sync_input_symmetry_shell_layouts_from_cached_layouts(Dataset &ds)
{
    auto &ctx = ds.symmetry_context;
    sync_input_symmetry_structure_from_dataset(ds);
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

void initialize_input_symmetry_context(Dataset &ds, const bool build_shell_rotations)
{
    auto &ctx = ds.symmetry_context;
    if (ctx.rspace_operations.empty())
    {
        return;
    }
    ctx.available = true;

    sync_input_symmetry_structure_from_dataset(ds);

    if (ctx.irreducible_sector.empty()
        && !ctx.atom_to_type.empty()
        && !ctx.input_coord_frac.empty()
        && !ds.pbc.Rlist.empty())
    {
        ctx.irreducible_sector =
            build_input_symmetry_rspace_irreducible_sector(ctx, ctx.input_coord_frac, ds.pbc.Rlist);
    }

    if (ctx.kstars.empty())
    {
        generate_input_symmetry_kstars_from_pbc(ctx, ds.pbc);
    }

    if (!build_shell_rotations)
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

    if (ctx.atom_to_type.empty() || ctx.input_coord_frac.empty())
    {
        return;
    }
    for (auto &star : ctx.kstars)
    {
        for (auto &member : star.members)
        {
            if (member.atom_rotations.empty())
            {
                populate_input_symmetry_kstar_member_rotations(ctx, star, member, lmax);
            }
        }
    }
}

void require_input_symmetry_shell_layouts(const Dataset &ds, const char *calculation)
{
    const auto &ctx = ds.symmetry_context;
    if (!ctx.available || ctx.rspace_operations.empty())
    {
        return;
    }

    if (!ctx.has_ao_shell_layout() || !ctx.has_abf_shell_layout())
    {
        throw LIBRPA_RUNTIME_ERROR(
            std::string("Cannot use ") + calculation
            + " symmetry without l-shell basis layouts for AO and ABF species");
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

void initialize_ds_exx(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_exx");
    const bool use_symmetry = opts.use_symmetry_exx == LIBRPA_SWITCH_ON;
    initialize_input_symmetry_context(ds, use_symmetry);
    if (use_symmetry)
    {
        require_input_symmetry_shell_layouts(ds, "EXX");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    ds.p_exx = std::make_unique<librpa_int::Exx>(ds.mf, ds.basis_wfc, ds.pbc, ds.symmetry_context,
                                                 ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                 is_eigvec_k_distributed,
                                                 use_symmetry);
    ds.p_exx->libri_threshold_C = opts.libri_exx_threshold_C;
    ds.p_exx->libri_threshold_D = opts.libri_exx_threshold_D;
    ds.p_exx->libri_threshold_V = opts.libri_exx_threshold_V;
    global::profiler.stop("initialize_ds_exx");
}

void initialize_ds_chi0(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_chi0");
    const bool use_symmetry = opts.use_symmetry_rpa == LIBRPA_SWITCH_ON;
    initialize_input_symmetry_context(ds, use_symmetry);
    if (use_symmetry)
    {
        require_input_symmetry_shell_layouts(ds, "RPA/chi0");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    if (opts.use_shrink_abfs == LIBRPA_SWITCH_ON && opts.use_shrink_chi == LIBRPA_SWITCH_ON)
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux_shrink, ds.pbc,
                                                       ds.symmetry_context,
                                                       ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                       is_eigvec_k_distributed,
                                                       use_symmetry);
    else
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc,
                                                       ds.symmetry_context,
                                                       ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                       is_eigvec_k_distributed,
                                                       use_symmetry);
    ds.p_chi0->gf_threshold = opts.gf_threshold;
    ds.p_chi0->libri_collect_s0_chunk = opts.libri_chi0_collect_s0_chunk;
    ds.p_chi0->libri_collect_max_bytes = opts.libri_chi0_collect_max_bytes;
    ds.p_chi0->nbands_G = opts.n_bands_chi0;
    ds.p_chi0->libri_threshold_C = opts.libri_chi0_threshold_C;
    ds.p_chi0->libri_threshold_G = opts.libri_chi0_threshold_G;
    global::profiler.stop("initialize_ds_chi0");
}

void initialize_ds_g0w0(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_g0w0");
    const bool use_symmetry = opts.use_symmetry_gw == LIBRPA_SWITCH_ON;
    initialize_input_symmetry_context(ds, use_symmetry);
    if (use_symmetry)
    {
        require_input_symmetry_shell_layouts(ds, "GW");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    // global::ofs_myid << "is_eigvec_k_distributed " << is_eigvec_k_distributed << std::endl;
    ds.p_g0w0 = std::make_unique<librpa_int::G0W0>(ds.mf, ds.basis_wfc, ds.pbc,
                                                   ds.symmetry_context, ds.tfg,
                                                   ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                   is_eigvec_k_distributed,
                                                   use_symmetry);
    ds.p_g0w0->libri_threshold_C = opts.libri_g0w0_threshold_C;
    ds.p_g0w0->libri_threshold_G = opts.libri_g0w0_threshold_G;
    ds.p_g0w0->libri_threshold_Wc = opts.libri_g0w0_threshold_Wc;
    ds.p_g0w0->output_dir = opts.output_dir;
    ds.p_g0w0->output_sigc_ks_kf = opts.output_gw_sigc_ks_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_ks_mat_kf = opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON;
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
