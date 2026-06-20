#include "symmetry.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace librpa_int
{

namespace
{

bool try_fold_fractional_kpoint_to_target(const Vector3_Order<double>& kpoint,
                                          const Vector3_Order<double>& target_kpoint,
                                          const double tol,
                                          FoldedKPoint& folded)
{
    const auto fold = kpoint - target_kpoint;
    if (!nearly_integer_vector(fold, tol))
    {
        return false;
    }
    folded.kpoint = target_kpoint;
    folded.fold_G = round_to_integer_vector(fold);
    return true;
}

bool find_atom_to_target_mapping(const std::vector<Vector3_Order<double>>& atom_positions,
                                 const SpaceGroupSymOps<SpaceGroupSymOp>& operations,
                                 const int atom,
                                 const int target,
                                 const double tol,
                                 AtomSymMapping& mapping)
{
    const auto atom_position = restrict_fractional_coordinate(atom_positions[atom], tol);
    const auto target_position = restrict_fractional_coordinate(atom_positions[target], tol);

    for (std::size_t isym = 0; isym < operations.size(); ++isym)
    {
        const auto& operation = operations[isym];
        const Vector3_Order<double> transformed =
            apply_space_group_symmetry_operation(operation, atom_position);
        const Vector3_Order<double> return_lattice = transformed - target_position;
        if (!nearly_integer_vector(return_lattice, tol))
        {
            continue;
        }
        mapping.inequivalent_atom = target;
        mapping.isym = static_cast<int>(isym);
        mapping.return_lattice = round_to_integer_vector(return_lattice);
        return true;
    }
    return false;
}

bool find_kpoint_sym_mapping(const Vector3_Order<double>& representative_kpoint,
                             const Vector3_Order<double>& member_kpoint,
                             const SpaceGroupSymOps<SpaceGroupSymOp>& operations,
                             const double tol,
                             KPointSymMapping& mapping)
{
    for (std::size_t isym = 0; isym < operations.size(); ++isym)
    {
        const auto rotated =
            apply_space_group_rotation_to_kpoint(operations[isym], representative_kpoint);
        FoldedKPoint folded;
        if (!try_fold_fractional_kpoint_to_target(rotated, member_kpoint, tol, folded))
        {
            continue;
        }
        mapping.isym = static_cast<int>(isym);
        mapping.fold_G = folded.fold_G;
        return true;
    }
    return false;
}

int choose_kpoint_star_representative(
    const KPointStar& star,
    const std::size_t fallback_full_k_index,
    const std::vector<Vector3_Order<double>>& preferred_representative_kpoints,
    const double tol)
{
    for (const auto& preferred_kpoint : preferred_representative_kpoints)
    {
        for (std::size_t member_index = 0; member_index < star.members.size(); ++member_index)
        {
            FoldedKPoint folded;
            if (try_fold_fractional_kpoint_to_target(
                    preferred_kpoint, star.members[member_index].kpoint, tol, folded))
            {
                return static_cast<int>(member_index);
            }
        }
    }

    const auto representative_member = std::find_if(
        star.members.begin(),
        star.members.end(),
        [fallback_full_k_index](const KPointStarMember& member) {
            return member.full_k_index == static_cast<int>(fallback_full_k_index);
        });
    if (representative_member == star.members.end())
    {
        throw std::runtime_error("k-star does not contain its representative k-point");
    }
    return static_cast<int>(std::distance(star.members.begin(), representative_member));
}

} // namespace

SpaceGroupSymOp compose_space_group_symmetry_operations(
    const SpaceGroupSymOp& lhs,
    const SpaceGroupSymOp& rhs)
{
    if (lhs.use_row_convention != rhs.use_row_convention)
    {
        throw std::invalid_argument("cannot compose symmetry operations with mixed conventions");
    }

    SpaceGroupSymOp composed;
    composed.use_row_convention = lhs.use_row_convention;
    if (composed.use_row_convention)
    {
        composed.rotation = multiply_space_group_rotation_matrices(lhs.rotation, rhs.rotation);
        composed.translation = multiply_row_vector(lhs.translation, rhs.rotation) + rhs.translation;
    }
    else
    {
        composed.rotation = multiply_space_group_rotation_matrices(rhs.rotation, lhs.rotation);
        composed.translation =
            Vector3_Order<double>(rhs.rotation * lhs.translation) + rhs.translation;
    }
    return composed;
}

Vector3_Order<double> apply_space_group_symmetry_operation(
    const SpaceGroupSymOp& operation,
    const Vector3_Order<double>& coord)
{
    if (operation.use_row_convention)
    {
        return multiply_row_vector(coord, operation.rotation) + operation.translation;
    }
    return Vector3_Order<double>(operation.rotation * coord) + operation.translation;
}

Vector3_Order<double> apply_space_group_rotation_to_kpoint(
    const SpaceGroupSymOp& operation,
    const Vector3_Order<double>& kpoint)
{
    const Matrix3 reciprocal_rotation = operation.rotation.Inverse().Transpose();
    if (operation.use_row_convention)
    {
        return multiply_row_vector(kpoint, reciprocal_rotation);
    }
    return Vector3_Order<double>(reciprocal_rotation * kpoint);
}

FoldedKPoint fold_fractional_kpoint_to_targets(
    const Vector3_Order<double>& kpoint,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const double tol)
{
    FoldedKPoint matched;
    bool found = false;
    for (std::size_t ik = 0; ik < target_kpoints.size(); ++ik)
    {
        FoldedKPoint candidate;
        if (!try_fold_fractional_kpoint_to_target(kpoint, target_kpoints[ik], tol, candidate))
        {
            continue;
        }
        if (found)
        {
            throw std::runtime_error("fractional k-point folds to more than one target k-point");
        }
        candidate.target_k_index = static_cast<int>(ik);
        matched = candidate;
        found = true;
    }
    if (!found)
    {
        throw std::runtime_error("fractional k-point does not fold to any target k-point");
    }
    return matched;
}

namespace
{

std::vector<AtomSymMapping> build_fractional_atom_to_inequivalent_symmetry_mapping(
    const std::vector<Vector3_Order<double>>& atom_positions,
    const SpaceGroupSymOps<SpaceGroupSymOp>& operations,
    const double tol)
{
    std::vector<AtomSymMapping> mappings(atom_positions.size());
    std::vector<int> inequivalent_atoms;

    for (std::size_t atom_index = 0; atom_index < atom_positions.size(); ++atom_index)
    {
        const int atom = static_cast<int>(atom_index);
        auto& mapping = mappings[atom_index];
        for (const int inequivalent_atom : inequivalent_atoms)
        {
            if (find_atom_to_target_mapping(atom_positions,
                                            operations,
                                            atom,
                                            inequivalent_atom,
                                            tol,
                                            mapping))
            {
                break;
            }
        }
        if (mapping.inequivalent_atom >= 0)
        {
            continue;
        }

        mapping.inequivalent_atom = atom;
        find_atom_to_target_mapping(atom_positions, operations, atom, atom, tol, mapping);
        inequivalent_atoms.push_back(atom);
    }

    return mappings;
}

} // namespace

const SpaceGroupSymOp SpaceGroupSymOp::IDENTITY{
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, true};

const SpaceGroupSymOp SpaceGroupSymOp::INVERSE{
    {-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, true};

const SpaceGroupSymOp SpaceGroupSymOp::C41_Z{
    {0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, true};

std::vector<AtomSymMapping> build_atom_to_inequivalent_symmetry_mapping(
    const std::vector<Vector3_Order<double>>& atom_positions_frac,
    const Matrix3& lattice_vectors,
    const SpaceGroupSymOps<SpaceGroupSymOp>& fractional_operations,
    const double tol)
{
    if (std::fabs(lattice_vectors.Det()) < 1e-14)
    {
        throw std::invalid_argument("lattice vectors must be non-singular");
    }

    return build_fractional_atom_to_inequivalent_symmetry_mapping(
        atom_positions_frac, fractional_operations, tol);
}

std::vector<int> collect_inequivalent_atoms(
    const std::vector<AtomSymMapping>& mappings)
{
    std::vector<int> atoms;
    for (const auto& mapping : mappings)
    {
        if (mapping.inequivalent_atom < 0
            || std::find(atoms.begin(), atoms.end(), mapping.inequivalent_atom) != atoms.end())
        {
            continue;
        }
        atoms.push_back(mapping.inequivalent_atom);
    }
    return atoms;
}

std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps<SpaceGroupSymOp>& fractional_operations,
    const double tol)
{
    return build_kpoint_stars(
        full_kpoints_frac, fractional_operations, std::vector<Vector3_Order<double>>{}, tol);
}

std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps<SpaceGroupSymOp>& fractional_operations,
    const std::vector<Vector3_Order<double>>& preferred_representative_kpoints,
    const double tol)
{
    if (fractional_operations.empty())
    {
        throw std::invalid_argument("at least one symmetry operation is required to build k-stars");
    }

    const auto& kpoints = full_kpoints_frac;

    std::vector<KPointStar> stars;
    std::vector<bool> used(kpoints.size(), false);
    for (std::size_t ik_rep = 0; ik_rep < kpoints.size(); ++ik_rep)
    {
        if (used[ik_rep])
        {
            continue;
        }

        KPointStar star;
        const auto seed_kpoint = kpoints[ik_rep];

        // NOTE: O(n_k^2 * n_sym); add a grid hash if this becomes hot.
        for (std::size_t ik = 0; ik < kpoints.size(); ++ik)
        {
            if (used[ik])
            {
                continue;
            }
            for (std::size_t isym = 0; isym < fractional_operations.size(); ++isym)
            {
                const auto rotated =
                    apply_space_group_rotation_to_kpoint(fractional_operations[isym], seed_kpoint);
                FoldedKPoint folded;
                if (!try_fold_fractional_kpoint_to_target(rotated, kpoints[ik], tol, folded))
                {
                    continue;
                }

                KPointStarMember member;
                member.full_k_index = static_cast<int>(ik);
                member.kpoint = kpoints[ik];
                star.members.push_back(member);
                break;
            }
        }

        if (star.members.empty())
        {
            throw std::runtime_error("failed to build a k-star member for the representative k-point");
        }
        star.representative_k_index =
            choose_kpoint_star_representative(star, ik_rep, preferred_representative_kpoints, tol);
        const auto chosen_representative_kpoint =
            star.members[star.representative_k_index].kpoint;
        star.sym_mappings.reserve(star.members.size());
        for (const auto& member : star.members)
        {
            KPointSymMapping sym_mapping;
            if (!find_kpoint_sym_mapping(
                    chosen_representative_kpoint,
                    member.kpoint,
                    fractional_operations,
                    tol,
                    sym_mapping))
            {
                throw std::runtime_error("failed to build symmetry mapping for k-star member");
            }
            star.sym_mappings.push_back(sym_mapping);
        }
        for (const auto& member : star.members)
        {
            used[static_cast<std::size_t>(member.full_k_index)] = true;
        }
        stars.push_back(std::move(star));
    }

    return stars;
}

} // namespace librpa_int
