#include "symmetry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace librpa_int
{

namespace
{

bool find_atom_to_target_mapping(const std::vector<Vector3_Order<double>>& atom_positions,
                                 const SpaceGroupSymOps<SpaceGroupSymOp>& operations,
                                 const int atom,
                                 const int target,
                                 const double tol,
                                 AtomInequivalentSymmetryMapping& mapping)
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

} // namespace

Matrix3 multiply_space_group_rotation_matrices(const Matrix3& lhs,
                                               const Matrix3& rhs)
{
    return lhs * rhs;
}

Vector3_Order<double> multiply_row_vector(const Vector3_Order<double>& vec,
                                          const Matrix3& matrix)
{
    return Vector3_Order<double>(vec * matrix);
}

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

namespace
{

std::vector<AtomInequivalentSymmetryMapping> build_fractional_atom_to_inequivalent_symmetry_mapping(
    const std::vector<Vector3_Order<double>>& atom_positions,
    const SpaceGroupSymOps<SpaceGroupSymOp>& operations,
    const double tol)
{
    std::vector<AtomInequivalentSymmetryMapping> mappings(atom_positions.size());
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

std::vector<AtomInequivalentSymmetryMapping> build_atom_to_inequivalent_symmetry_mapping(
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
    const std::vector<AtomInequivalentSymmetryMapping>& mappings)
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

} // namespace librpa_int
