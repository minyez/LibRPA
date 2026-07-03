/*!
 * @file symmetry_context.cpp
 * @brief Utilities for generated symmetry data.
 */
#include "symmetry_context.h"

#include "../math/rsh.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "../io/stl_io_helper.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace librpa_int
{

constexpr double kSymmetryCoordTol = 1e-5;
// Real-space atom mapping is reconstructed from fractional coordinates. The looser tolerance
// below remains as a fallback for text-derived coordinates and lattice-inversion noise.
constexpr double kSymmetryRSpaceAtomMapTol = 5e-5;

static std::vector<int> build_atom_offsets(const std::map<atom_t, size_t>& atom_nw)
{
    std::vector<int> offsets(atom_nw.size() + 1, 0);
    int running = 0;
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto iter = atom_nw.find(atom);
        if (iter == atom_nw.end())
        {
            throw LIBRPA_RUNTIME_ERROR("Atomic orbital counts are not contiguous in atom_nw");
        }
        offsets[atom] = running;
        running += static_cast<int>(iter->second);
    }
    offsets.back() = running;
    return offsets;
}

static ComplexMatrix extract_atom_block(const ComplexMatrix& matrix,
                                        const atom_t atom_i,
                                        const atom_t atom_j,
                                        const std::map<atom_t, size_t>& atom_nw,
                                        const std::vector<int>& offsets)
{
    const int ni = static_cast<int>(atom_nw.at(atom_i));
    const int nj = static_cast<int>(atom_nw.at(atom_j));
    ComplexMatrix block(ni, nj);
    const int row0 = offsets[atom_i];
    const int col0 = offsets[atom_j];
    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            block(i, j) = matrix(row0 + i, col0 + j);
        }
    }
    return block;
}

static void set_atom_block(ComplexMatrix& matrix,
                           const atom_t atom_i,
                           const atom_t atom_j,
                           const ComplexMatrix& block,
                           const std::vector<int>& offsets)
{
    const int row0 = offsets[atom_i];
    const int col0 = offsets[atom_j];
    for (int i = 0; i < block.nr; ++i)
    {
        for (int j = 0; j < block.nc; ++j)
        {
            matrix(row0 + i, col0 + j) = block(i, j);
        }
    }
}

static std::vector<int> build_rspace_inverse_map(const SymmetryContext& ctx)
{
    std::vector<int> inverse_map(ctx.rspace_operations.size(), -1);
    for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
    {
        for (std::size_t jsym = 0; jsym < ctx.rspace_operations.size(); ++jsym)
        {
            const auto composed = compose_space_group_symmetry_operations(
                ctx.rspace_operations[isym], ctx.rspace_operations[jsym]);
            const bool is_inverse =
                composed.is_identity_rotation()
                && nearly_integer_vector(composed.translation, kSymmetryCoordTol);

            if (is_inverse)
            {
                inverse_map[isym] = static_cast<int>(jsym);
                break;
            }
        }
        if (inverse_map[isym] < 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Failed to build inverse symmetry-operation map for symmetry");
        }
    }
    return inverse_map;
}

static Vector3_Order<int> rotate_rspace_vector(
    const Vector3_Order<int>& R,
    const SpaceGroupAtomMapping<atom_t>& op_info,
    const SpaceGroupSymOp& op,
    const atom_t atom_from_i,
    const atom_t atom_from_j)
{
    const auto to_double = [](const Vector3_Order<int>& vec) {
        return Vector3_Order<double>(static_cast<double>(vec.x),
                                     static_cast<double>(vec.y),
                                     static_cast<double>(vec.z));
    };
    const Vector3_Order<double> R_double{static_cast<double>(R.x), static_cast<double>(R.y),
                                         static_cast<double>(R.z)};
    const Vector3_Order<double> rotated_cell = multiply_row_vector(R_double, op.rotation);
    const Vector3_Order<double> common_shift = to_double(op_info.return_lattice[atom_from_i]);
    const Vector3_Order<double> mapped_j_cell =
        to_double(op_info.return_lattice[atom_from_j]) + rotated_cell - common_shift;
    if (!nearly_integer_vector(mapped_j_cell, kSymmetryCoordTol))
    {
        throw LIBRPA_RUNTIME_ERROR("Real-space symmetry generated a non-integer lattice vector");
    }
    // Keep the raw rotated lattice vector returned by the ABACUS formula.
    // The caller is responsible for filtering against the explicit R list.
    return round_to_integer_vector(mapped_j_cell);
}

static Vector3_Order<int> canonicalize_bvk_rspace_vector(
    const Vector3_Order<int>& R,
    const Vector3_Order<int>& period)
{
    const auto centered_mod = [](const int value, const int cell_period) {
        if (cell_period <= 0)
        {
            return value;
        }
        const int upper = (cell_period - 1) / 2;
        int folded = value % cell_period;
        if (folded < 0)
        {
            folded += cell_period;
        }
        if (folded > upper)
        {
            folded -= cell_period;
        }
        return folded;
    };

    return {centered_mod(R.x, period.x),
            centered_mod(R.y, period.y),
            centered_mod(R.z, period.z)};
}

static symmetry_R_t to_symmetry_R(const Vector3_Order<int>& R)
{
    return {R.x, R.y, R.z};
}

static std::array<double, 3> coord_frac_array(const coord_t &coord)
{
    return {coord.x, coord.y, coord.z};
}

static Vector3_Order<double> coord_frac_vector(
    const std::map<atom_t, std::array<double, 3>> &coords,
    const atom_t atom)
{
    const auto iter = coords.find(atom);
    if (iter == coords.end())
    {
        throw LIBRPA_RUNTIME_ERROR("Symmetry is missing a fractional atom coordinate");
    }
    return {iter->second[0], iter->second[1], iter->second[2]};
}

struct KStarOperationInfo
{
    int spatial_isym = -1;
    bool time_reversal = false;
};

struct KStarOperations
{
    SpaceGroupSymOps operations;
    std::vector<KStarOperationInfo> info;
};

static KStarOperations build_kstar_operations_with_time_reversal(
    const SpaceGroupSymOps &operations)
{
    KStarOperations kstar_operations;
    kstar_operations.operations.reserve(2 * operations.size());
    kstar_operations.info.reserve(2 * operations.size());
    for (std::size_t isym = 0; isym != operations.size(); ++isym)
    {
        kstar_operations.operations.push_back(operations[isym]);
        kstar_operations.info.push_back({static_cast<int>(isym), false});
    }
    for (std::size_t isym = 0; isym != operations.size(); ++isym)
    {
        const auto &op = operations[isym];
        SpaceGroupSymOp tr_op;
        tr_op.rotation = op.rotation * -1.0;
        tr_op.translation = op.translation;
        tr_op.use_row_convention = op.use_row_convention;
        kstar_operations.operations.push_back(tr_op);
        kstar_operations.info.push_back({static_cast<int>(isym), true});
    }
    return kstar_operations;
}

static void build_symmetry_pbc_index_kstars(
    SymmetryContext &ctx,
    const std::vector<Vector3_Order<double>> &kpoints)
{
    const int identity_isym = find_identity_symmetry_operation(ctx.rspace_operations);

    ctx.kstars.clear();
    ctx.kstar_member_fold_G.clear();
    ctx.kstars.reserve(kpoints.size());
    for (std::size_t ik = 0; ik != kpoints.size(); ++ik)
    {
        SymmetryKStarMember member;
        member.spatial_isym = identity_isym;
        member.time_reversal = false;
        member.k_bz = kpoints[ik];

        SymmetryKStar star;
        star.star_index = static_cast<int>(ik);
        star.k_ibz = kpoints[ik];
        star.members.push_back(std::move(member));

        ctx.kstar_member_fold_G[{star.star_index, 0}] = {0, 0, 0};
        ctx.kstars.push_back(std::move(star));
    }
}

static std::vector<Vector3_Order<double>> full_kpoints_frac_from_pbc(
    const PeriodicBoundaryData &pbc)
{
    if (static_cast<int>(pbc.kfrac_list_full.size()) == pbc.get_n_cells_bvk())
    {
        return pbc.kfrac_list_full;
    }
    if (static_cast<int>(pbc.klist.size()) < pbc.get_n_cells_bvk())
    {
        return build_uniform_kmesh_frac(pbc.period);
    }
    return pbc.kfrac_list_full.empty() ? pbc.kfrac_list : pbc.kfrac_list_full;
}

static std::vector<Vector3_Order<double>> coul_kpoints_frac_from_pbc(
    const PeriodicBoundaryData &pbc)
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
        kpoints.emplace_back(restrict_fractional_coordinate(Vector3_Order<double>{kfrac}));
    }
    return kpoints;
}

static atom_t find_symmetry_atom_target(const SymmetryContext &ctx,
                                        const atom_t atom_from,
                                        const int spatial_isym,
                                        Vector3_Order<int> &return_lattice)
{
    const auto &op = ctx.rspace_operations.at(static_cast<std::size_t>(spatial_isym));
    const auto atom_type = ctx.atom_to_type.at(atom_from);
    const auto coord_from = coord_frac_vector(ctx.input_coord_frac, atom_from);
    const auto transformed = apply_space_group_symmetry_operation(op, coord_from);

    atom_t matched_atom = static_cast<atom_t>(-1);
    for (const auto &[atom_to, type_to] : ctx.atom_to_type)
    {
        if (type_to != atom_type)
        {
            continue;
        }
        const auto coord_to = coord_frac_vector(ctx.input_coord_frac, atom_to);
        const auto diff = transformed - coord_to;
        if (!nearly_integer_vector(diff, 1e-5))
        {
            continue;
        }
        if (matched_atom != static_cast<atom_t>(-1))
        {
            throw LIBRPA_RUNTIME_ERROR("Symmetry atom mapping is ambiguous");
        }
        matched_atom = atom_to;
        return_lattice = round_to_integer_vector(diff);
    }
    if (matched_atom == static_cast<atom_t>(-1))
    {
        throw LIBRPA_RUNTIME_ERROR("Symmetry failed to map an atom under a k-star operation");
    }
    return matched_atom;
}

static void store_generated_symmetry_kstars(
    SymmetryContext &ctx,
    const std::vector<KPointStar> &generated_stars,
    const std::vector<Vector3_Order<double>> &kpoints_ibz,
    const std::vector<KStarOperationInfo> &operation_info)
{
    if (kpoints_ibz.size() != generated_stars.size())
    {
        throw LIBRPA_RUNTIME_ERROR("Generated k-star representative count does not match k-star count");
    }

    ctx.kstars.clear();
    ctx.kstar_member_fold_G.clear();
    ctx.kstars.reserve(generated_stars.size());

    for (std::size_t istar = 0; istar != generated_stars.size(); ++istar)
    {
        const auto &generated_star = generated_stars[istar];
        SymmetryKStar star;
        star.star_index = static_cast<int>(istar);
        star.k_ibz = kpoints_ibz[istar];
        star.members.reserve(generated_star.members.size());
        for (std::size_t imember = 0; imember != generated_star.members.size(); ++imember)
        {
            const int generated_isym = generated_star.sym_mappings.at(imember).isym;
            if (generated_isym < 0
                || generated_isym >= static_cast<int>(operation_info.size()))
            {
                throw LIBRPA_RUNTIME_ERROR("Generated k-star member has an invalid symmetry route");
            }
            const auto &route = operation_info[static_cast<std::size_t>(generated_isym)];
            SymmetryKStarMember member;
            member.spatial_isym = route.spatial_isym;
            member.time_reversal = route.time_reversal;
            member.k_bz = generated_star.members[imember].kpoint;
            ctx.kstar_member_fold_G[{star.star_index, static_cast<int>(imember)}] =
                generated_star.sym_mappings.at(imember).fold_G;
            star.members.push_back(std::move(member));
        }
        ctx.kstars.push_back(std::move(star));
    }
}

ComplexMatrix build_symmetry_shell_rotation_from_direct_rotation(
    const SpaceGroupSymOp& operation,
    const Matrix3& lattice_vectors,
    const int l,
    const BasisConvention& basis_convention,
    const double threshold)
{
    if (l < 0)
    {
        throw LIBRPA_INVALID_ARGUMENT("angular momentum l must be non-negative");
    }
    if (!is_basis_rsh_convention_set(basis_convention))
    {
        throw LIBRPA_INVALID_ARGUMENT("basis real-spherical-harmonic convention is unset");
    }
    if (l == 0)
    {
        return real_spherical_harmonic_rotation_matrix(Vector3<double>{0.0, 0.0, 0.0},
                                                       0,
                                                       basis_convention.order,
                                                       basis_convention.coeff_m_negative,
                                                       basis_convention.coeff_m_positive);
    }
    if (std::abs(lattice_vectors.Det()) < 1e-14)
    {
        throw LIBRPA_INVALID_ARGUMENT("lattice vectors must be non-singular");
    }

    const Matrix3 cartesian_rotation = fractional_rotation_to_cartesian(operation, lattice_vectors);
    return real_spherical_harmonic_rotation_matrix(cartesian_rotation,
                                                   l,
                                                   basis_convention.order,
                                                   basis_convention.coeff_m_negative,
                                                   basis_convention.coeff_m_positive,
                                                   threshold);
}

std::map<int, ComplexMatrix> build_symmetry_shell_rotations_from_direct_rotation(
    const SpaceGroupSymOp& operation,
    const Matrix3& lattice_vectors,
    const int lmax,
    const BasisConvention& basis_convention,
    const double threshold)
{
    std::map<int, ComplexMatrix> shell_rotations;
    if (lmax < 0)
    {
        return shell_rotations;
    }

    // Eq. (40) uses T_tilde(V), the direct-space rotation part of the
    // Bloch-sum transform. Atom/k-dependent Bloch phases are applied later.
    for (int l = 0; l <= lmax; ++l)
    {
        shell_rotations[l] =
            build_symmetry_shell_rotation_from_direct_rotation(
                operation, lattice_vectors, l, basis_convention, threshold);
    }
    return shell_rotations;
}

std::complex<double> build_symmetry_kspace_phase(
    const Vector3_Order<double>& k_source,
    const Vector3_Order<double>& k_target,
    const Vector3_Order<double>& atom_from_frac,
    const Vector3_Order<double>& atom_to_frac,
    const Vector3_Order<int>& return_lattice,
    const BasisConvention& basis_convention)
{
    if (!is_basis_bloch_convention_set(basis_convention))
    {
        throw LIBRPA_INVALID_ARGUMENT("basis Bloch-sum convention is unset");
    }

    const Vector3_Order<double> return_lattice_double{
        static_cast<double>(return_lattice.x),
        static_cast<double>(return_lattice.y),
        static_cast<double>(return_lattice.z)};
    const double bracket =
        -(k_target * return_lattice_double)
        + static_cast<double>(basis_convention.bloch_ratom)
              * ((k_target * atom_to_frac) - (k_source * atom_from_frac));
    const double phase_arg =
        -TWO_PI * static_cast<double>(basis_convention.bloch_phase) * bracket;
    return {std::cos(phase_arg), std::sin(phase_arg)};
}

std::map<int, ComplexMatrix> build_symmetry_kspace_shell_rotations(
    const SpaceGroupSymOp& operation,
    const Matrix3& lattice_vectors,
    const int lmax,
    const BasisConvention& basis_convention,
    const Vector3_Order<double>& k_source,
    const Vector3_Order<double>& k_target,
    const Vector3_Order<double>& atom_from_frac,
    const Vector3_Order<double>& atom_to_frac,
    const Vector3_Order<int>& return_lattice,
    const double threshold)
{
    auto shell_rotations =
        build_symmetry_shell_rotations_from_direct_rotation(
            operation, lattice_vectors, lmax, basis_convention, threshold);
    const auto phase = build_symmetry_kspace_phase(k_source,
                                                         k_target,
                                                         atom_from_frac,
                                                         atom_to_frac,
                                                         return_lattice,
                                                         basis_convention);
    for (auto& shell_rotation : shell_rotations)
    {
        shell_rotation.second *= phase;
    }
    return shell_rotations;
}

void SymmetryContext::clear()
{
    unset_available();
    lattice_available = false;
    basis_convention = {};
    irreducible_sector.clear();
    rspace_sector_stars.clear();
    rspace_operations.clear();
    rsh_rotations.clear();
    kstars.clear();
    kstar_grid_mapping.clear();
    full_kpoint_members.clear();
    atom_to_type.clear();
    input_coord_frac.clear();
    lattice_vectors.Reset();
    reciprocal_vectors.Reset();
    kspace_return_lattice.clear();
    kstar_member_fold_G.clear();
}

void SymmetryContext::set_available()
{
    available = true;
}

void SymmetryContext::unset_available()
{
    available = false;
}

void SymmetryContext::add_rspace_operation(SymmetryOperation operation)
{
    operation.to_row_convention();
    rspace_operations.push_back(std::move(operation));
    rsh_rotations.clear();
}

void SymmetryContext::set_rspace_operations(std::vector<SymmetryOperation> operations)
{
    rspace_operations.clear();
    rsh_rotations.clear();
    rspace_operations.reserve(operations.size());
    for (auto& operation : operations)
    {
        add_rspace_operation(std::move(operation));
    }
}

void SymmetryContext::set_crystal_structure(const Matrix3& latvec,
                                            const Matrix3& reciprocal,
                                            const std::map<atom_t, int>& atom_types,
                                            const std::map<atom_t, coord_t>& coords_frac)
{
    lattice_vectors = latvec;
    reciprocal_vectors = reciprocal;
    lattice_available = true;
    atom_to_type = atom_types;
    input_coord_frac.clear();
    for (const auto &[atom, coord] : coords_frac)
    {
        input_coord_frac[atom] = coord_frac_array(coord);
    }
    irreducible_sector.clear();
    rspace_sector_stars.clear();
    kstars.clear();
    kstar_grid_mapping.clear();
    full_kpoint_members.clear();
    rsh_rotations.clear();
    kspace_return_lattice.clear();
    kstar_member_fold_G.clear();
}

void SymmetryContext::build_periodic_mappings(const PeriodicBoundaryData& pbc,
                                              const std::vector<Vector3_Order<int>>& Rlist)
{
    irreducible_sector.clear();
    rspace_sector_stars.clear();
    kstars.clear();
    kstar_grid_mapping.clear();
    full_kpoint_members.clear();
    kspace_return_lattice.clear();
    kstar_member_fold_G.clear();

    generate_irreducible_sector(Rlist);
    generate_rspace_sector_stars(pbc.period, Rlist);
    generate_kstars(pbc);
    generate_kstar_grid_mapping(pbc);
    generate_full_kpoint_members(pbc.kfrac_list);
}

void SymmetryContext::generate_irreducible_sector(const std::vector<Vector3_Order<int>> &Rlist)
{
    if (!atom_to_type.empty() && !input_coord_frac.empty() && !Rlist.empty())
    {
        irreducible_sector = build_symmetry_rspace_irreducible_sector(*this, Rlist);
    }
}

void SymmetryContext::generate_rspace_sector_stars(
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& Rlist)
{
    rspace_sector_stars.clear();
    if (irreducible_sector.empty() || rspace_operations.empty() || Rlist.empty())
    {
        return;
    }
    build_symmetry_rspace_sector_stars(*this, period, Rlist, rspace_sector_stars, nullptr);
}

void SymmetryContext::generate_kstars(const PeriodicBoundaryData &pbc)
{
    const auto full_kpoints = full_kpoints_frac_from_pbc(pbc);
    const auto coul_kpoints = coul_kpoints_frac_from_pbc(pbc);
    const bool scf_kpoints_cover_full_grid =
        static_cast<int>(pbc.kfrac_list.size()) == pbc.get_n_cells_bvk();
    const auto kstar_operations = build_kstar_operations_with_time_reversal(rspace_operations);
    const auto generated_stars = build_kpoint_stars(
        full_kpoints, kstar_operations.operations, coul_kpoints, 1e-5);
    if (generated_stars.size() != coul_kpoints.size())
    {
        if (scf_kpoints_cover_full_grid)
        {
            std::vector<Vector3_Order<double>> generated_representatives;
            generated_representatives.reserve(generated_stars.size());
            for (const auto &star : generated_stars)
            {
                generated_representatives.push_back(
                    star.members.at(static_cast<std::size_t>(star.representative_k_index)).kpoint);
            }
            store_generated_symmetry_kstars(
                *this, generated_stars, generated_representatives, kstar_operations.info);
            return;
        }
        if (coul_kpoints.size() == full_kpoints.size())
        {
            build_symmetry_pbc_index_kstars(*this, coul_kpoints);
            return;
        }
        throw LIBRPA_RUNTIME_ERROR("Generated symmetry k-star count does not match Coulomb k-points");
    }

    std::vector<Vector3_Order<double>> ordered_representatives;
    ordered_representatives.reserve(coul_kpoints.size());
    std::vector<bool> used(generated_stars.size(), false);
    std::vector<KPointStar> ordered_stars;
    ordered_stars.reserve(coul_kpoints.size());
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
                build_symmetry_pbc_index_kstars(*this, pbc.kfrac_list);
                return;
            }
            throw LIBRPA_RUNTIME_ERROR("Failed to order generated symmetry k-stars by IBZ k-points");
        }

        used[static_cast<std::size_t>(matched_star_index)] = true;
        ordered_stars.push_back(generated_stars[static_cast<std::size_t>(matched_star_index)]);
        ordered_representatives.push_back(coul_kpoints[ik_ibz]);
    }
    store_generated_symmetry_kstars(
        *this, ordered_stars, ordered_representatives, kstar_operations.info);
}

void SymmetryContext::build_rsh_rotations(const BasisConvention& basis_convention_in,
                                          const int lmax)
{
    if (lmax < 0)
    {
        return;
    }
    if (!lattice_available)
    {
        throw LIBRPA_RUNTIME_ERROR("Cannot generate symmetry shell rotations without lattice vectors");
    }
    if (!is_basis_convention_set(basis_convention_in))
    {
        throw LIBRPA_RUNTIME_ERROR("Cannot initialize symmetry context without basis convention");
    }

    basis_convention = basis_convention_in;
    rsh_rotations.resize(rspace_operations.size());
    for (std::size_t isym = 0; isym != rspace_operations.size(); ++isym)
    {
        const auto &op = rspace_operations[isym];
        const auto cartesian_rotation =
            fractional_rotation_to_cartesian(op, lattice_vectors).Inverse();
        auto &op_rsh_rotations = rsh_rotations[isym];
        for (int l = 0; l <= lmax; ++l)
        {
            if (op_rsh_rotations.find(l) == op_rsh_rotations.end())
            {
                op_rsh_rotations[l] =
                    real_spherical_harmonic_rotation_matrix(
                        cartesian_rotation,
                        l,
                        basis_convention.order,
                        basis_convention.coeff_m_negative,
                        basis_convention.coeff_m_positive);
            }
        }
    }
}

void SymmetryContext::build_kstar_member_rotations(const int lmax)
{
    if (lmax < 0 || atom_to_type.empty() || input_coord_frac.empty())
    {
        return;
    }
    for (auto &star : kstars)
    {
        for (auto &member : star.members)
        {
            if (!member.atom_rotations.empty())
            {
                continue;
            }

            const int n_spatial_operations = static_cast<int>(rspace_operations.size());
            const int spatial_isym = member.spatial_isym;
            if (spatial_isym < 0 || spatial_isym >= n_spatial_operations)
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "Generated symmetry k-star member has an invalid symmetry index");
            }
            const auto &operation = rspace_operations.at(static_cast<std::size_t>(spatial_isym));

            member.atom_rotations.clear();
            member.atom_rotations.reserve(atom_to_type.size());
            for (const auto &[atom_from, atom_type] : atom_to_type)
            {
                Vector3_Order<int> return_lattice{0, 0, 0};
                const atom_t atom_to =
                    find_symmetry_atom_target(*this, atom_from, spatial_isym, return_lattice);
                const auto inserted = kspace_return_lattice.emplace(
                    std::make_pair(static_cast<int>(atom_from), spatial_isym), return_lattice);
                if (!inserted.second && inserted.first->second != return_lattice)
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Symmetry generated inconsistent atom return lattices");
                }

                SymmetryKAtomRotation atom_rotation;
                atom_rotation.atom_from = static_cast<int>(atom_from);
                atom_rotation.atom_to = static_cast<int>(atom_to);
                atom_rotation.atom_type = atom_type;
                atom_rotation.lmax = lmax;
                atom_rotation.bloch_rsh_rotations =
                    build_symmetry_kspace_shell_rotations(
                        operation,
                        lattice_vectors,
                        lmax,
                        basis_convention,
                        star.k_ibz,
                        member.k_bz,
                        coord_frac_vector(input_coord_frac, atom_from),
                        coord_frac_vector(input_coord_frac, atom_to),
                        return_lattice);
                member.atom_rotations.push_back(std::move(atom_rotation));
            }
        }
    }
}

bool SymmetryContext::empty() const
{
    return irreducible_sector.empty() && rspace_sector_stars.empty()
           && rspace_operations.empty() && kstars.empty() && atom_to_type.empty() && rsh_rotations.empty()
           && kspace_return_lattice.empty() && kstar_member_fold_G.empty();
}

void SymmetryContext::print_summary(std::ostream& log) const
{
    log << "Detected symmetry operations from structure\n";
    if (irreducible_sector.empty())
    {
        log << "| irreducible atom pairs : pending compute initialization\n"
            << "| irreducible {pair, R}  : pending compute initialization\n";
    }
    else
    {
        log << "| irreducible atom pairs : " << count_irreducible_pairs() << "\n"
            << "| irreducible {pair, R}  : " << count_irreducible_blocks() << "\n";
    }
    log << "| real-space operations  : " << rspace_operations.size() << "\n";
    if (kstars.empty())
    {
        log << "| IBZ k-stars            : pending compute initialization\n"
            << "| total star members     : pending compute initialization\n";
    }
    else
    {
        log << "| IBZ k-stars            : " << kstars.size() << "\n"
            << "| total star members     : " << count_kstar_members() << "\n";
    }
    log << "| k rotations            : generated from symmetry operations\n"
        << "| RSH rotation cache     : " << rsh_rotations.size() << " operations\n";
    log << "| legacy sidecars        : ignored (symrot_R/symrot_k/irreducible_sector)\n";
}

std::size_t SymmetryContext::count_irreducible_pairs() const
{
    return irreducible_sector.size();
}

std::size_t SymmetryContext::count_irreducible_blocks() const
{
    std::size_t count = 0;
    for (const auto& pair_Rs : irreducible_sector)
    {
        count += pair_Rs.second.size();
    }
    return count;
}

std::size_t SymmetryContext::count_kstar_members() const
{
    std::size_t count = 0;
    for (const auto& star : kstars)
    {
        count += star.members.size();
    }
    return count;
}

bool symmetry_species_layouts_match_atom_counts(
    const std::vector<SpeciesBasisLayout>& layouts,
    const std::map<atom_t, int>& atom_to_type,
    const std::map<atom_t, size_t>& atom_nb)
{
    try
    {
        for (const auto& atom_count : atom_nb)
        {
            const auto type_iter = atom_to_type.find(atom_count.first);
            if (type_iter == atom_to_type.end())
            {
                return false;
            }
            if (get_symmetry_species_layout(layouts, type_iter->second).n_ao
                != static_cast<int>(atom_count.second))
            {
                return false;
            }
        }
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

ComplexMatrix build_symmetry_rotation_matrix(
    const SpeciesBasisLayout& layout,
    const std::map<int, ComplexMatrix>& rsh_rotations)
{
    ComplexMatrix rotation(layout.n_ao, layout.n_ao);
    const auto& shell_offsets = layout.shell_offsets;

    int filled_nao = 0;
    for (const auto& entry : layout.shell_indices)
    {
        const int l = entry.first;
        const auto rotation_iter = rsh_rotations.find(l);
        if (rotation_iter == rsh_rotations.end())
        {
            throw LIBRPA_RUNTIME_ERROR("Missing shell rotation block for l=" + std::to_string(l));
        }

        const ComplexMatrix& shell_rotation = rotation_iter->second;
        const int nm = 2 * l + 1;
        if (shell_rotation.nr != nm || shell_rotation.nc != nm)
        {
            throw LIBRPA_RUNTIME_ERROR("Shell rotation block has incompatible shape for l="
                                     + std::to_string(l));
        }

        for (const int ishell : entry.second)
        {
            if (ishell < 0 || ishell >= static_cast<int>(shell_offsets.size()))
            {
                throw LIBRPA_RUNTIME_ERROR("Species basis layout has invalid shell index");
            }
            const int offset = shell_offsets[static_cast<std::size_t>(ishell)];
            for (int row = 0; row < nm; ++row)
            {
                for (int col = 0; col < nm; ++col)
                {
                    rotation(offset + row, offset + col) = shell_rotation(row, col);
                }
            }
            filled_nao += nm;
        }
    }

    if (filled_nao != layout.n_ao)
    {
        throw LIBRPA_RUNTIME_ERROR("Failed to assemble the full AO rotation matrix");
    }
    return rotation;
}

ComplexMatrix SymmetryContext::get_rotation_matrix(
    const std::vector<SpeciesBasisLayout>& layouts,
    const int atom_type,
    const int isym) const
{
    if (isym < 0 || isym >= static_cast<int>(rspace_operations.size()))
    {
        throw std::out_of_range("Real-space symmetry index is out of range");
    }
    if (isym >= static_cast<int>(rsh_rotations.size()))
    {
        throw LIBRPA_RUNTIME_ERROR("Real-space RSH rotations are unavailable");
    }

    const auto& layout = get_symmetry_species_layout(layouts, atom_type);
    const auto& shell_rotations = rsh_rotations[static_cast<std::size_t>(isym)];
    return build_symmetry_rotation_matrix(layout, shell_rotations);
}

static int find_symmetry_kstar_index_for_kpoint(const std::vector<SymmetryKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label)
{
    int matched_index = -1;

    // Prefer the canonical IBZ representative when the incoming q-point already uses the same
    // gauge as the generated ABACUS k-star.
    for (std::size_t istar = 0; istar < kstars.size(); ++istar)
    {
        if (!same_fractional_kpoint(kstars[istar].k_ibz, k_point, kSymmetryCoordTol))
        {
            continue;
        }
        if (matched_index >= 0)
        {
            throw LIBRPA_RUNTIME_ERROR(label + " matching is ambiguous for the current q-point");
        }
        matched_index = static_cast<int>(istar);
    }

    if (matched_index >= 0)
    {
        return matched_index;
    }

    // LibRPA may keep an equivalent full-star member, rather than the ABACUS representative, as
    // the active q-point label. Fall back to a unique member match in that case.
    for (std::size_t istar = 0; istar < kstars.size(); ++istar)
    {
        const auto& star = kstars[istar];
        const bool star_contains_kpoint =
            std::any_of(star.members.begin(), star.members.end(),
                        [&k_point](const SymmetryKStarMember& member) {
                            return same_fractional_kpoint(
                                member.k_bz, k_point, kSymmetryCoordTol);
                        });
        if (!star_contains_kpoint)
        {
            continue;
        }
        if (matched_index >= 0)
        {
            throw LIBRPA_RUNTIME_ERROR(label + " member matching is ambiguous for the current q-point");
        }
        matched_index = static_cast<int>(istar);
    }

    if (matched_index < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Failed to match the current q-point with " + label);
    }
    return matched_index;
}

const SymmetryKStar& find_symmetry_kstar_for_kpoint(const std::vector<SymmetryKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label)
{
    return kstars[static_cast<std::size_t>(
        find_symmetry_kstar_index_for_kpoint(kstars, k_point, label))];
}

const SymmetryKStar& find_symmetry_kstar_for_ibz_kpoint(const SymmetryContext& ctx,
                                                    const Vector3_Order<double>& k_ibz)
{
    return find_symmetry_kstar_for_kpoint(ctx.kstars, k_ibz);
}

void SymmetryContext::generate_kstar_grid_mapping(const PeriodicBoundaryData& pbc)
{
    const auto& klist_internal = pbc.klist;
    const auto& kfrac_list = pbc.kfrac_list;
    const auto& irk_to_full_kpoints = pbc.map_irk_ks;

    kstar_grid_mapping.clear();
    if (klist_internal.size() != kfrac_list.size())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "LibRPA k-point metadata is inconsistent: `klist` and `kfrac_list` have different sizes");
    }
    if (kstars.empty())
    {
        return;
    }
    if (!lattice_available)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "k-star grid mapping requires reciprocal lattice vectors from the structure input");
    }

    auto convert_fractional_to_internal = [this](const Vector3_Order<double>& kfrac) {
        // Keep the row-vector convention used for PeriodicBoundaryData k-point keys.
        const auto& G = reciprocal_vectors;
        return Vector3_Order<double>{kfrac.x * G.e11 + kfrac.y * G.e21 + kfrac.z * G.e31,
                                     kfrac.x * G.e12 + kfrac.y * G.e22 + kfrac.z * G.e32,
                                     kfrac.x * G.e13 + kfrac.y * G.e23 + kfrac.z * G.e33};
    };

    auto find_matching_ibz_full_list =
        [&irk_to_full_kpoints](const Vector3_Order<double>& q_ibz_key)
        -> std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>>::const_iterator {
        const auto exact_iter = irk_to_full_kpoints.find(q_ibz_key);
        if (exact_iter != irk_to_full_kpoints.end())
        {
            return exact_iter;
        }

        return std::find_if(irk_to_full_kpoints.begin(), irk_to_full_kpoints.end(),
                            [&q_ibz_key](const auto& entry) {
                                return same_fractional_kpoint(
                                    entry.first, q_ibz_key, kSymmetryCoordTol);
                            });
    };

    std::vector<SymmetryKStarGridMappingEntry> mapping(kfrac_list.size());
    std::vector<bool> matched_stars(kstars.size(), false);

    for (std::size_t iq_ibz = 0; iq_ibz < kfrac_list.size(); ++iq_ibz)
    {
        const int matched_star_index = find_symmetry_kstar_index_for_kpoint(
            kstars, kfrac_list[iq_ibz], "symmetry k-stars");

        matched_stars[static_cast<std::size_t>(matched_star_index)] = true;
        auto& entry = mapping[iq_ibz];
        entry.iq_ibz = static_cast<int>(iq_ibz);
        entry.star_list_index = matched_star_index;

        const auto& star = kstars[static_cast<std::size_t>(matched_star_index)];
        const auto q_ibz_key = klist_internal[iq_ibz];
        const auto full_list_iter = find_matching_ibz_full_list(q_ibz_key);
        const std::vector<Vector3_Order<double>>* full_q_keys =
            (full_list_iter == irk_to_full_kpoints.end()) ? nullptr : &full_list_iter->second;
        std::vector<bool> matched_full_q(
            (full_q_keys == nullptr) ? 0 : full_q_keys->size(), false);
        entry.member_q_bz_keys.resize(star.members.size());

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto member_q_internal =
                convert_fractional_to_internal(star.members[imember].k_bz);
            entry.member_q_bz_keys[imember] = member_q_internal;

            // The generated ABACUS k-star is the authoritative description of the members,
            // including the exact representative chosen after symmetry and BZ folding.
            // LibRPA's `map_irk_ks` is only used here as an optional source of already
            // existing internal q keys. If a member cannot be matched back to that rebuilt
            // list, keep the symmetry-derived key instead of rejecting the star.
            if (full_q_keys == nullptr)
            {
                continue;
            }

            int matched_full_index = -1;
            for (std::size_t ifull = 0; ifull < full_q_keys->size(); ++ifull)
            {
                if (matched_full_q[ifull]
                    || !same_fractional_kpoint(
                        (*full_q_keys)[ifull], member_q_internal, kSymmetryCoordTol))
                {
                    continue;
                }
                if (matched_full_index >= 0)
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Star member to LibRPA full-q matching is ambiguous");
                }
                matched_full_index = static_cast<int>(ifull);
            }

            if (matched_full_index >= 0)
            {
                matched_full_q[static_cast<std::size_t>(matched_full_index)] = true;
                entry.member_q_bz_keys[imember] =
                    (*full_q_keys)[static_cast<std::size_t>(matched_full_index)];
            }
        }
    }

    for (std::size_t istar = 0; istar < matched_stars.size(); ++istar)
    {
        if (!matched_stars[istar])
        {
            throw LIBRPA_RUNTIME_ERROR(
                "Not every k-star could be matched to the loaded LibRPA IBZ grid");
        }
    }

    kstar_grid_mapping = std::move(mapping);
}

void SymmetryContext::generate_full_kpoint_members(
    const std::vector<Vector3_Order<double>>& kfrac_list)
{
    full_kpoint_members.clear();
    std::vector<SymmetryFullKpointMemberEntry> members;
    if (kstars.empty())
    {
        return;
    }
    if (kstars.size() != kfrac_list.size())
    {
        return;
    }

    members.reserve(count_kstar_members());
    for (int ik_ibz = 0; ik_ibz != static_cast<int>(kfrac_list.size()); ++ik_ibz)
    {
        const int matched_star_index = find_symmetry_kstar_index_for_kpoint(
            kstars, kfrac_list[static_cast<std::size_t>(ik_ibz)], "symmetry k-stars");

        const auto& star = kstars[static_cast<std::size_t>(matched_star_index)];
        for (int imember = 0; imember != static_cast<int>(star.members.size()); ++imember)
        {
            const auto& member = star.members[static_cast<std::size_t>(imember)];
            members.push_back({ik_ibz, matched_star_index, imember,
                               member.spatial_isym, member.time_reversal, member.k_bz});
        }
    }

    full_kpoint_members = std::move(members);
}

symmetry_kstar_representative_indices_t build_symmetry_full_grid_kstar_representative_indices(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list)
{
    if (!ctx.available || ctx.kstars.empty())
    {
        return {};
    }
    if (ctx.kstars.size() >= kfrac_list.size()
        || ctx.count_kstar_members() != kfrac_list.size())
    {
        return {};
    }

    symmetry_kstar_representative_indices_t representative_indices;
    representative_indices.reserve(ctx.kstars.size());
    std::vector<bool> used_kpoints(kfrac_list.size(), false);
    for (const auto& star : ctx.kstars)
    {
        FoldedKPoint folded;
        try
        {
            folded = fold_fractional_kpoint_to_targets(
                star.k_ibz, kfrac_list, kSymmetryCoordTol);
        }
        catch (const std::runtime_error&)
        {
            return {};
        }
        if (folded.target_k_index < 0
            || folded.target_k_index >= static_cast<int>(kfrac_list.size()))
        {
            return {};
        }
        const auto ik_mf = static_cast<std::size_t>(folded.target_k_index);
        if (used_kpoints[ik_mf])
        {
            return {};
        }
        used_kpoints[ik_mf] = true;
        representative_indices.push_back(folded.target_k_index);
    }
    return representative_indices;
}

symmetry_kstar_member_kfrac_targets_t build_symmetry_full_grid_kstar_member_kfrac_targets(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list)
{
    if (!ctx.available || ctx.kstars.empty())
    {
        return {};
    }
    if (ctx.kstars.size() >= kfrac_list.size()
        || ctx.count_kstar_members() != kfrac_list.size())
    {
        return {};
    }

    symmetry_kstar_member_kfrac_targets_t targets(ctx.kstars.size());
    std::vector<bool> used_kpoints(kfrac_list.size(), false);
    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto& star = ctx.kstars[istar];
        targets[istar].reserve(star.members.size());
        for (const auto& member : star.members)
        {
            FoldedKPoint folded;
            try
            {
                folded = fold_fractional_kpoint_to_targets(
                    member.k_bz, kfrac_list, kSymmetryCoordTol);
            }
            catch (const std::runtime_error&)
            {
                return {};
            }
            if (folded.target_k_index < 0
                || folded.target_k_index >= static_cast<int>(kfrac_list.size()))
            {
                return {};
            }
            const auto ik_mf = static_cast<std::size_t>(folded.target_k_index);
            if (used_kpoints[ik_mf])
            {
                return {};
            }
            used_kpoints[ik_mf] = true;
            targets[istar].push_back(folded.kpoint);
        }
    }
    return targets;
}

static Vector3_Order<int> build_symmetry_kspace_return_lattice(
    const SymmetryContext& ctx,
    const SymmetryKAtomRotation& atom_rotation,
    const int spatial_isym)
{
    const auto stored_return_lattice =
        ctx.kspace_return_lattice.find({atom_rotation.atom_from, spatial_isym});
    if (stored_return_lattice != ctx.kspace_return_lattice.end())
    {
        return stored_return_lattice->second;
    }

    const auto coord_from_iter =
        ctx.input_coord_frac.find(static_cast<atom_t>(atom_rotation.atom_from));
    const auto coord_to_iter =
        ctx.input_coord_frac.find(static_cast<atom_t>(atom_rotation.atom_to));
    if (coord_from_iter == ctx.input_coord_frac.end()
        || coord_to_iter == ctx.input_coord_frac.end())
    {
        throw LIBRPA_RUNTIME_ERROR("Missing fractional coordinate for k-space phase correction");
    }

    if (spatial_isym < 0 || spatial_isym >= static_cast<int>(ctx.rspace_operations.size()))
    {
        throw LIBRPA_RUNTIME_ERROR("K-space phase correction uses an invalid symmetry index");
    }

    const auto& op = ctx.rspace_operations[static_cast<std::size_t>(spatial_isym)];
    const Vector3_Order<double> coord_from{coord_from_iter->second};
    const Vector3_Order<double> coord_to{coord_to_iter->second};
    const Vector3_Order<double> transformed =
        apply_space_group_symmetry_operation(op, coord_from);
    const Vector3_Order<double> return_lattice = transformed - coord_to;
    if (!nearly_integer_vector(return_lattice, kSymmetryCoordTol))
    {
        throw LIBRPA_RUNTIME_ERROR("K-space phase correction produced a non-integer return lattice");
    }
    return round_to_integer_vector(return_lattice);
}

static Vector3_Order<int> build_symmetry_equivalent_kpoint_shift(
    const Vector3_Order<double>& k_bz_source,
    const Vector3_Order<double>& k_bz_target)
{
    const Vector3_Order<double> k_shift{
        k_bz_target.x - k_bz_source.x,
        k_bz_target.y - k_bz_source.y,
        k_bz_target.z - k_bz_source.z,
    };
    if (!nearly_integer_vector(k_shift, kSymmetryCoordTol))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Symmetry restore encountered non-equivalent full-k representatives");
    }
    return round_to_integer_vector(k_shift);
}

static std::pair<atom_t, atom_t> canonicalize_symmetry_upper_atom_pair(
    const atom_t atom_i,
    const atom_t atom_j)
{
    return (atom_i <= atom_j) ? std::make_pair(atom_i, atom_j)
                              : std::make_pair(atom_j, atom_i);
}

static std::vector<const SymmetryKAtomRotation*> build_symmetry_rotations_by_from(
    const SymmetryKStarMember& member)
{
    int max_atom_index = -1;
    for (const auto& atom_rotation : member.atom_rotations)
    {
        max_atom_index = std::max(max_atom_index, atom_rotation.atom_from);
        max_atom_index = std::max(max_atom_index, atom_rotation.atom_to);
    }
    std::vector<const SymmetryKAtomRotation*> rotations_by_from(
        static_cast<std::size_t>(max_atom_index + 1), nullptr);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        rotations_by_from.at(static_cast<std::size_t>(atom_rotation.atom_from)) = &atom_rotation;
    }
    return rotations_by_from;
}

std::set<std::pair<atom_t, atom_t>> build_symmetry_upper_atom_pair_closure(
    const SymmetryKStar& star,
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs)
{
    std::set<std::pair<atom_t, atom_t>> closure_pairs;
    for (const auto& atom_pair : target_atom_pairs)
    {
        closure_pairs.insert(canonicalize_symmetry_upper_atom_pair(
            atom_pair.first, atom_pair.second));
    }

    bool changed = true;
    while (changed)
    {
        changed = false;
        const auto snapshot_pairs = closure_pairs;
        for (const auto& member : star.members)
        {
            const auto rotations_by_from = build_symmetry_rotations_by_from(member);
            for (const auto& atom_pair : snapshot_pairs)
            {
                if (static_cast<std::size_t>(atom_pair.first) >= rotations_by_from.size()
                    || static_cast<std::size_t>(atom_pair.second) >= rotations_by_from.size())
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Atom-pair closure requested an atom outside the loaded star");
                }
                const auto* rot_i = rotations_by_from[static_cast<std::size_t>(atom_pair.first)];
                const auto* rot_j = rotations_by_from[static_cast<std::size_t>(atom_pair.second)];
                if (rot_i == nullptr || rot_j == nullptr)
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Atom-pair closure found an incomplete atom permutation");
                }
                const auto source_pair = canonicalize_symmetry_upper_atom_pair(
                    static_cast<atom_t>(rot_i->atom_to),
                    static_cast<atom_t>(rot_j->atom_to));
                if (closure_pairs.insert(source_pair).second)
                {
                    changed = true;
                }
            }
        }
    }
    return closure_pairs;
}

static std::complex<double> build_symmetry_reciprocal_gauge_phase(
    const Vector3_Order<int>& k_shift,
    const atom_t atom,
    const std::map<atom_t, std::array<double, 3>>& coord_frac_map)
{
    const auto coord_iter = coord_frac_map.find(atom);
    if (coord_iter == coord_frac_map.end())
    {
        throw LIBRPA_RUNTIME_ERROR("Missing fractional coordinate for reciprocal-gauge phase");
    }

    const Vector3_Order<double> tau{coord_iter->second};
    const double phase_arg =
        TWO_PI * (static_cast<double>(k_shift.x) * tau.x
                  + static_cast<double>(k_shift.y) * tau.y
                  + static_cast<double>(k_shift.z) * tau.z);
    return std::complex<double>(std::cos(phase_arg), std::sin(phase_arg));
}

struct KSpaceRotationPlan
{
    bool apply_time_reversal = false;
    bool apply_target_gauge = false;
    std::vector<const SymmetryKAtomRotation*> rotations_by_from;
    std::vector<ComplexMatrix> atom_M_blocks;
    std::vector<std::complex<double>> atom_target_phases;
};

static KSpaceRotationPlan build_kspace_rotation_plan(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& layouts,
    const SymmetryKStarMember& member,
    const std::map<atom_t, size_t>& atom_counts,
    const Vector3_Order<double>& k_ibz,
    const bool use_time_reversal,
    const Vector3_Order<double>* k_bz_target,
    const char* label)
{
    KSpaceRotationPlan plan;
    plan.apply_time_reversal = use_time_reversal || member.time_reversal;
    plan.rotations_by_from.assign(atom_counts.size(), nullptr);

    std::vector<bool> visited_to(atom_counts.size(), false);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from < 0
            || atom_rotation.atom_from >= static_cast<int>(atom_counts.size())
            || atom_rotation.atom_to < 0
            || atom_rotation.atom_to >= static_cast<int>(atom_counts.size()))
        {
            throw LIBRPA_RUNTIME_ERROR(std::string(label) + " atom mapping is out of range");
        }
        plan.rotations_by_from[static_cast<std::size_t>(atom_rotation.atom_from)] =
            &atom_rotation;
        visited_to[static_cast<std::size_t>(atom_rotation.atom_to)] = true;
    }

    for (std::size_t atom = 0; atom < atom_counts.size(); ++atom)
    {
        if (plan.rotations_by_from[atom] == nullptr)
        {
            throw LIBRPA_RUNTIME_ERROR(std::string(label)
                                      + " atom rotations do not cover every atom");
        }
        if (!visited_to[atom])
        {
            throw LIBRPA_RUNTIME_ERROR(std::string(label)
                                      + " atom mapping is not a full permutation");
        }
    }

    const int spatial_isym = member.spatial_isym;
    if (spatial_isym < 0 || spatial_isym >= static_cast<int>(ctx.rspace_operations.size()))
    {
        throw LIBRPA_RUNTIME_ERROR(std::string(label) + " uses an invalid symmetry index");
    }

    const Vector3_Order<double> delta_k{k_ibz.x - member.k_bz.x,
                                        k_ibz.y - member.k_bz.y,
                                        k_ibz.z - member.k_bz.z};
    plan.atom_M_blocks.resize(atom_counts.size());
    for (std::size_t atom = 0; atom < atom_counts.size(); ++atom)
    {
        const auto* atom_rotation = plan.rotations_by_from[atom];
        const auto& layout = get_symmetry_species_layout(layouts, atom_rotation->atom_type);
        plan.atom_M_blocks[atom] =
            build_symmetry_rotation_matrix(layout, atom_rotation->bloch_rsh_rotations);
        const auto return_lattice =
            build_symmetry_kspace_return_lattice(ctx,
                                                 *atom_rotation,
                                                 spatial_isym);
        const double phase_arg =
            TWO_PI * (delta_k.x * static_cast<double>(return_lattice.x)
                      + delta_k.y * static_cast<double>(return_lattice.y)
                      + delta_k.z * static_cast<double>(return_lattice.z));
        plan.atom_M_blocks[atom] *=
            std::complex<double>(std::cos(phase_arg), std::sin(phase_arg));
    }

    plan.apply_target_gauge = (k_bz_target != nullptr);
    plan.atom_target_phases.assign(atom_counts.size(), {1.0, 0.0});
    if (plan.apply_target_gauge)
    {
        const auto k_shift = build_symmetry_equivalent_kpoint_shift(member.k_bz, *k_bz_target);
        for (std::size_t atom = 0; atom < atom_counts.size(); ++atom)
        {
            plan.atom_target_phases[atom] = build_symmetry_reciprocal_gauge_phase(
                k_shift, static_cast<atom_t>(atom), ctx.input_coord_frac);
        }
    }

    return plan;
}

symmetry_atom_block_matrix_map_t rotate_symmetry_kspace_operator_blocks(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& layouts,
    const SymmetryKStarMember& member,
    const symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nbasis,
    const Vector3_Order<double>& k_ibz,
    const bool use_time_reversal,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs,
    const Vector3_Order<double>* k_bz_target)
{
    const bool apply_time_reversal = use_time_reversal || member.time_reversal;

    auto get_block_or_hermitian = [&blocks_ibz](const atom_t atom_i, const atom_t atom_j) {
        const auto atom_i_iter = blocks_ibz.find(atom_i);
        if (atom_i_iter != blocks_ibz.end())
        {
            const auto atom_j_iter = atom_i_iter->second.find(atom_j);
            if (atom_j_iter != atom_i_iter->second.end())
            {
                return atom_j_iter->second;
            }
        }

        const auto atom_j_iter = blocks_ibz.find(atom_j);
        if (atom_j_iter != blocks_ibz.end())
        {
            const auto atom_i_iter_fallback = atom_j_iter->second.find(atom_i);
            if (atom_i_iter_fallback != atom_j_iter->second.end())
            {
                return transpose(atom_i_iter_fallback->second, true);
            }
        }

        throw LIBRPA_RUNTIME_ERROR("Missing atom block while rotating the q-space operator");
    };
    // The identity member at the IBZ representative is an exact no-op as long as no
    // target-representative gauge shift is requested. Returning the original/hermitian-completed
    // blocks here avoids rebuilding an M matrix that should mathematically be the identity.
    if (!apply_time_reversal && member.spatial_isym == 0 && k_bz_target == nullptr
        && same_fractional_kpoint(member.k_bz, k_ibz, kSymmetryCoordTol))
    {
        symmetry_atom_block_matrix_map_t identity_blocks;
        for (std::size_t atom_i = 0; atom_i < atom_nbasis.size(); ++atom_i)
        {
            for (std::size_t atom_j = 0; atom_j < atom_nbasis.size(); ++atom_j)
            {
                if (target_atom_pairs != nullptr
                    && target_atom_pairs->count({static_cast<atom_t>(atom_i),
                                                 static_cast<atom_t>(atom_j)}) == 0)
                {
                    continue;
                }
                identity_blocks[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                    get_block_or_hermitian(static_cast<atom_t>(atom_i),
                                           static_cast<atom_t>(atom_j));
            }
        }
        return identity_blocks;
    }

    const auto plan = build_kspace_rotation_plan(ctx,
                                                 layouts,
                                                 member,
                                                 atom_nbasis,
                                                 k_ibz,
                                                 use_time_reversal,
                                                 k_bz_target,
                                                 "K-space rotation");

    symmetry_atom_block_matrix_map_t rotated_blocks;
    for (std::size_t atom_i = 0; atom_i < atom_nbasis.size(); ++atom_i)
    {
        const auto* rot_i = plan.rotations_by_from[atom_i];
        const auto& M_i = plan.atom_M_blocks[atom_i];
        for (std::size_t atom_j = 0; atom_j < atom_nbasis.size(); ++atom_j)
        {
            if (target_atom_pairs != nullptr
                && target_atom_pairs->count({static_cast<atom_t>(atom_i),
                                             static_cast<atom_t>(atom_j)}) == 0)
            {
                continue;
            }
            const auto* rot_j = plan.rotations_by_from[atom_j];
            const auto& M_j = plan.atom_M_blocks[atom_j];
            const auto target_i = static_cast<atom_t>(atom_i);
            const auto target_j = static_cast<atom_t>(atom_j);
            const auto source_i = static_cast<atom_t>(rot_i->atom_to);
            const auto source_j = static_cast<atom_t>(rot_j->atom_to);
            ComplexMatrix block_ibz;
            try
            {
                block_ibz = get_block_or_hermitian(source_i, source_j);
            }
            catch (const std::exception&)
            {
                std::ostringstream oss;
                oss << "Missing atom block while rotating the q-space operator: "
                    << "target_pair=(" << target_i << "," << target_j << "), "
                    << "source_pair=(" << source_i << "," << source_j << "), "
                    << "spatial_isym=" << member.spatial_isym << ", "
                    << "time_reversal=" << (plan.apply_time_reversal ? "true" : "false");
                throw LIBRPA_RUNTIME_ERROR(oss.str());
            }

            if (block_ibz.nr != static_cast<int>(atom_nbasis.at(source_i))
                || block_ibz.nc != static_cast<int>(atom_nbasis.at(source_j)))
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "The atom block dimension is incompatible with the rotated source atom pair");
            }

            ComplexMatrix block_rotated;
            if (plan.apply_time_reversal)
            {
                // Use the same row-major convention as the AO-side k-space rotation:
                //   TRS: O_bz[I,J] = M_I^dagger · conj(O_ibz[S(I),S(J)]) · M_J
                block_rotated = transpose(M_i, true) * conj(block_ibz) * M_j;
            }
            else
            {
                // Row-major equivalent of the ABACUS col-major rotation:
                //   non-TRS: O_bz[I,J] = M_I^T · O_ibz[S(I),S(J)] · conj(M_J)
                block_rotated = transpose(M_i, false) * block_ibz * conj(M_j);
            }
            if (plan.apply_target_gauge)
            {
                const auto left_phase = plan.atom_target_phases[atom_i];
                const auto right_phase = plan.atom_target_phases[atom_j];
                block_rotated *= left_phase * std::conj(right_phase);
            }
            rotated_blocks[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                std::move(block_rotated);
        }
    }

    return rotated_blocks;
}

symmetry_atom_block_matrix_map_t symmetrize_symmetry_ibz_kspace_operator_blocks(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& layouts,
    const Vector3_Order<double>& k_ibz,
    const symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nbasis,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs)
{
    std::set<std::pair<atom_t, atom_t>> inferred_target_pairs;
    if (target_atom_pairs == nullptr)
    {
        for (const auto& atom_i_pair : blocks_ibz)
        {
            for (const auto& atom_j_pair : atom_i_pair.second)
            {
                inferred_target_pairs.insert({atom_i_pair.first, atom_j_pair.first});
            }
        }
        target_atom_pairs = &inferred_target_pairs;
    }

    if (target_atom_pairs->empty())
    {
        return blocks_ibz;
    }

    const auto& star = find_symmetry_kstar_for_ibz_kpoint(ctx, k_ibz);
    symmetry_atom_block_matrix_map_t accumulated_blocks;
    int n_members_used = 0;
    for (const auto& member : star.members)
    {
        if (member.time_reversal)
        {
            continue;
        }
        if (!same_fractional_kpoint(member.k_bz, k_ibz, kSymmetryCoordTol))
        {
            continue;
        }

        // Little-group members can return an equivalent IBZ representative that differs from the
        // active LibRPA label by a reciprocal-lattice vector G. Re-apply the target gauge so the
        // averaged operator is accumulated in the same k_ibz representative used by LibRPA.
        const auto rotated_blocks = rotate_symmetry_kspace_operator_blocks(
            ctx, layouts, member, blocks_ibz, atom_nbasis, k_ibz, false,
            target_atom_pairs, &k_ibz);

        for (const auto& atom_i_pair : rotated_blocks)
        {
            for (const auto& atom_j_pair : atom_i_pair.second)
            {
                auto& block = accumulated_blocks[atom_i_pair.first][atom_j_pair.first];
                if (block.nr == 0 && block.nc == 0)
                {
                    block = atom_j_pair.second;
                }
                else
                {
                    block += atom_j_pair.second;
                }
            }
        }
        ++n_members_used;
    }

    if (n_members_used == 0)
    {
        return blocks_ibz;
    }

    const std::complex<double> inv_count(1.0 / static_cast<double>(n_members_used), 0.0);
    for (auto& atom_i_pair : accumulated_blocks)
    {
        for (auto& atom_j_pair : atom_i_pair.second)
        {
            atom_j_pair.second *= inv_count;
        }
    }
    return accumulated_blocks;
}

ComplexMatrix rotate_symmetry_kspace_matrix(const SymmetryContext& ctx,
                                            const std::vector<SpeciesBasisLayout>& layouts,
                                            const SymmetryKStarMember& member,
                                            const ComplexMatrix& matrix_ibz,
                                            const std::map<atom_t, size_t>& atom_nw,
                                            const Vector3_Order<double>& k_ibz,
                                            const bool use_time_reversal,
                                            const Vector3_Order<double>* k_bz_target)
{
    // -------------------------------------------------------------------------
    // Rotate D(k_ibz) to D(k_bz) using the input-convention Bloch rotation matrix M.
    //
    // Important: ABACUS defines the Bloch phase with k_bz, while the
    // internal restore_dm() path uses k_ibz when constructing M(R, k). We must
    // therefore rebuild the internal matrix by multiplying the exported block
    // with exp[i (k_ibz - k_bz) · O], where O is the atom-resolved return
    // lattice. This is the AO counterpart of the already validated side
    // phase correction.
    //
    // ABACUS col-major:  D^T(k_bz) = M† · D^T(k_ibz) · M
    // Row-major:         D(k_bz)   = M^T · D(k_ibz) · M*
    //
    // Block formula (M_I = M[S(I), I]):
    //   non-TRS:  D_bz[I, J] = M_I^T  · D_ibz[S(I), S(J)]  · conj(M_J)
    //   TRS:      D_bz[I, J] = M_I†   · conj(D_ibz[S(I), S(J)]) · M_J
    // -------------------------------------------------------------------------
    const auto offsets = build_atom_offsets(atom_nw);
    const int nao_total = offsets.back();
    if (matrix_ibz.nr != nao_total || matrix_ibz.nc != nao_total)
    {
        throw LIBRPA_RUNTIME_ERROR("The input matrix dimension is incompatible with the AO basis layout");
    }

    const auto plan = build_kspace_rotation_plan(ctx,
                                                 layouts,
                                                 member,
                                                 atom_nw,
                                                 k_ibz,
                                                 use_time_reversal,
                                                 k_bz_target,
                                                 "AO k-space rotation");

    ComplexMatrix rotated_matrix(nao_total, nao_total);

    // Apply the block-level rotation formula.
    //
    // ABACUS col-major formula:  D^T(k_bz) = M† · D^T(k_ibz) · M
    // Row-major equivalent:      D(k_bz)   = M^T · D(k_ibz) · M*
    //
    // M[S(I), I] is the internal ABACUS Bloch rotation block reconstructed from
    // the generated shell rotation times the return-lattice phase correction.
    //
    // Block formulas (M_I denotes M[S(I), I]):
    //   non-TRS:  D_bz[I, J] = M_I^T  · D_ibz[S(I), S(J)]  · conj(M_J)
    //   TRS:      D_bz[I, J] = M_I†   · D_ibz[S(I), S(J)]* · M_J
    //
    // Source indices: S(I) = atom_to,  destination indices: I = atom_from.
    for (std::size_t atom_i = 0; atom_i < atom_nw.size(); ++atom_i)
    {
        const auto* rot_i = plan.rotations_by_from[atom_i];
        const auto& M_i = plan.atom_M_blocks[atom_i];
        for (std::size_t atom_j = 0; atom_j < atom_nw.size(); ++atom_j)
        {
            const auto* rot_j = plan.rotations_by_from[atom_j];
            const auto& M_j = plan.atom_M_blocks[atom_j];
            // Read from D_ibz at the MAPPED atom positions S(I), S(J)
            const ComplexMatrix block_ibz =
                extract_atom_block(matrix_ibz,
                                   static_cast<atom_t>(rot_i->atom_to),
                                   static_cast<atom_t>(rot_j->atom_to),
                                   atom_nw, offsets);
            ComplexMatrix block_rotated;
            if (plan.apply_time_reversal)
            {
                // TRS: D_bz[I,J] = M_I† · conj(D_ibz[S(I),S(J)]) · M_J
                block_rotated = transpose(M_i, true) * conj(block_ibz) * M_j;
            }
            else
            {
                // Space group: D_bz[I,J] = M_I^T · D_ibz[S(I),S(J)] · conj(M_J)
                block_rotated = transpose(M_i, false) * block_ibz * conj(M_j);
            }
            if (plan.apply_target_gauge)
            {
                const auto left_phase = plan.atom_target_phases[atom_i];
                const auto right_phase = plan.atom_target_phases[atom_j];
                block_rotated *= left_phase * std::conj(right_phase);
            }
            // Write to D_bz at the ORIGINAL atom positions I, J
            set_atom_block(rotated_matrix,
                           static_cast<atom_t>(atom_i),
                           static_cast<atom_t>(atom_j),
                           block_rotated,
                           offsets);
        }
    }

    return rotated_matrix;
}

symmetry_irreducible_sector_t build_symmetry_rspace_irreducible_sector(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<int>>& Rlist)
{
    if (ctx.input_coord_frac.size() != ctx.atom_to_type.size())
    {
        throw LIBRPA_RUNTIME_ERROR("Fractional coordinates are unavailable for real-space symmetry");
    }

    std::map<int, Vector3_Order<double>> generic_coord_frac;
    for (const auto& [atom, coord] : ctx.input_coord_frac)
    {
        generic_coord_frac[static_cast<int>(atom)] = {coord[0], coord[1], coord[2]};
    }
    std::map<int, int> generic_atom_to_type;
    for (const auto& [atom, type] : ctx.atom_to_type)
    {
        generic_atom_to_type[static_cast<int>(atom)] = type;
    }

    const auto generic_sector = build_space_group_rspace_irreducible_sector(
        ctx.rspace_operations,
        generic_coord_frac,
        generic_atom_to_type,
        Rlist,
        ctx.lattice_available ? &ctx.lattice_vectors : nullptr,
        kSymmetryRSpaceAtomMapTol,
        kSymmetryCoordTol);

    symmetry_irreducible_sector_t irreducible_sector;
    for (const auto& [pair, Rs] : generic_sector)
    {
        auto& target_Rs =
            irreducible_sector[{static_cast<atom_t>(pair.first), static_cast<atom_t>(pair.second)}];
        for (const auto& R : Rs)
        {
            target_Rs.insert(to_symmetry_R(R));
        }
    }

    return irreducible_sector;
}

void build_symmetry_rspace_sector_stars(const SymmetryContext& ctx,
                                        const Vector3_Order<int>& period,
                                        const std::vector<Vector3_Order<int>>& Rlist,
                                        symmetry_rspace_sector_stars_t& sector_stars,
                                        std::ostream* log)
{
    if (ctx.irreducible_sector.empty() || ctx.rspace_operations.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("Real-space symmetry metadata is incomplete");
    }
    if (ctx.atom_to_type.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("Atom-to-type mapping is unavailable for real-space symmetry");
    }
    if (ctx.input_coord_frac.size() != ctx.atom_to_type.size())
    {
        throw LIBRPA_RUNTIME_ERROR("Fractional coordinates are unavailable for real-space symmetry");
    }

    std::vector<SpaceGroupAtomMapping<atom_t>> op_infos;
    op_infos.reserve(ctx.rspace_operations.size());
    for (const auto &op: ctx.rspace_operations)
    {
        op_infos.push_back(get_space_group_atom_mapping(
            op, ctx.input_coord_frac, ctx.atom_to_type, kSymmetryRSpaceAtomMapTol));
    }
    std::vector<bool> use_operation(ctx.rspace_operations.size(), true);
    if (ctx.lattice_available)
    {
        for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
        {
            use_operation[isym] =
                preserves_lattice_metric(ctx.rspace_operations[isym].rotation, ctx.lattice_vectors);
        }
    }
    const auto inverse_map = build_rspace_inverse_map(ctx);

    sector_stars.clear();
    std::set<Vector3_Order<int>> Rset(Rlist.begin(), Rlist.end());
    using full_key_t = std::tuple<atom_t, atom_t, Vector3_Order<int>>;
    std::set<full_key_t> covered;

    for (const auto& pair_Rs : ctx.irreducible_sector)
    {
        const atpair_t& ir_pair = pair_Rs.first;
        for (const auto& ir_R_array : pair_Rs.second)
        {
            const Vector3_Order<int> ir_R{ir_R_array[0], ir_R_array[1], ir_R_array[2]};
            auto& star_members = sector_stars[ir_pair][ir_R];
            std::vector<std::string> candidate_debug;
            bool saw_duplicate_member = false;
            for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
            {
                const int inv = inverse_map[isym];
                if (!use_operation[static_cast<std::size_t>(inv)])
                {
                    continue;
                }
                const auto& op_info = op_infos[static_cast<std::size_t>(inv)];
                const auto full_I = op_info.atom_map[ir_pair.first];
                const auto full_J = op_info.atom_map[ir_pair.second];
                const auto full_R = canonicalize_bvk_rspace_vector(
                    rotate_rspace_vector(ir_R,
                                         op_info,
                                         ctx.rspace_operations[static_cast<std::size_t>(inv)],
                                         ir_pair.first,
                                         ir_pair.second),
                    period);
                const bool in_rset = Rset.count(full_R) != 0;
                const full_key_t full_key{full_I, full_J, full_R};
                const bool is_duplicate = covered.count(full_key) != 0;

                std::ostringstream oss;
                oss << "isym=" << isym << " inv=" << inv << " -> (" << full_I << ", " << full_J
                    << "), R=(" << full_R.x << ", " << full_R.y << ", " << full_R.z
                    << "), in_Rset=" << (in_rset ? "true" : "false")
                    << ", duplicate=" << (is_duplicate ? "true" : "false");
                candidate_debug.push_back(oss.str());

                if (!in_rset)
                {
                    continue;
                }

                saw_duplicate_member = saw_duplicate_member || is_duplicate;
                if (covered.insert(full_key).second)
                {
                    star_members.push_back(
                        {static_cast<int>(isym), {full_I, full_J}, full_R});
                }
            }

            if (star_members.empty())
            {
                if (saw_duplicate_member)
                {
                    continue;
                }
                std::ostringstream oss;
                oss << "Failed to build a real-space symmetry star from symmetry for "
                    << "irreducible pair (" << ir_pair.first << ", " << ir_pair.second << ")"
                    << " and R=(" << ir_R.x << ", " << ir_R.y << ", " << ir_R.z << ")";
                for (const auto& line : candidate_debug)
                {
                    oss << "\n  " << line;
                }
                throw LIBRPA_RUNTIME_ERROR(oss.str());
            }
        }
    }

    if (log != nullptr)
    {
        std::size_t total_members = 0;
        for (const auto& pair_star : sector_stars)
        {
            for (const auto& R_star : pair_star.second)
            {
                total_members += R_star.second.size();
            }
        }
        (*log) << "| real-space sector stars: " << total_members << " full members restored from "
               << ctx.count_irreducible_blocks() << " irreducible blocks (" << covered.size()
               << " unique full {atom pair, R} blocks)\n";
    }
}

ComplexMatrix rotate_symmetry_rspace_block(const SymmetryContext& ctx,
                                           const std::vector<SpeciesBasisLayout>& layouts_i,
                                           const std::vector<SpeciesBasisLayout>& layouts_j,
                                           const int isym,
                                           const atom_t atom_from_i,
                                           const atom_t atom_from_j,
                                           const ComplexMatrix& matrix_source)
{
    const int type_i = ctx.atom_to_type.at(atom_from_i);
    const int type_j = ctx.atom_to_type.at(atom_from_j);
    const ComplexMatrix T_i = ctx.get_rotation_matrix(layouts_i, type_i, isym);
    const ComplexMatrix T_j = ctx.get_rotation_matrix(layouts_j, type_j, isym);

    if (matrix_source.nr != T_i.nr || matrix_source.nc != T_j.nr)
    {
        throw LIBRPA_RUNTIME_ERROR("Real-space rotation has incompatible AO dimensions");
    }

    // Keep the same row-major convention as the AO/ABF k-space operator restore:
    //   H_bz[I,J] = T_I^T · H_ir[S(I),S(J)] · conj(T_J)
    // for ordinary spatial operations without time reversal.
    return transpose(T_i, false) * matrix_source * conj(T_j);
}

ComplexMatrix rotate_symmetry_rspace_block(const SymmetryContext& ctx,
                                           const std::vector<SpeciesBasisLayout>& layouts,
                                           const int isym,
                                           const atom_t atom_from_i,
                                           const atom_t atom_from_j,
                                           const ComplexMatrix& matrix_source)
{
    return rotate_symmetry_rspace_block(
        ctx, layouts, layouts, isym, atom_from_i, atom_from_j, matrix_source);
}

}  // namespace librpa_int
