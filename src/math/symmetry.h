/*!
 * @file symmetry.h
 * @brief Space-group symmetry operation primitives.
 */
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "matrix3.h"
#include "vector3_order.h"

namespace librpa_int
{

/*!
 * @brief Space-group operation in fractional coordinates.
 *
 * This follows the LibRPA row-fractional convention of lattice vectors
 * and atom positions:
 *
 * x' = x * rotation + translation.
 *
 * Set use_row_convention=false for column-fractional operations:
 *
 * x' = rotation * x + translation.
 *
 * The latter convention is used, e.g. in Spglib.
 */
struct SpaceGroupSymOp
{
    Matrix3 rotation{0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0};
    Vector3_Order<double> translation{0.0, 0.0, 0.0};
    //! Whether to treat lattice vectors and fractional coordinates as row vectors.
    bool use_row_convention = true;
};

/*!
 * @brief Container for a set of space-group symmetry operations.
 */
template <typename OperationType = SpaceGroupSymOp>
struct SpaceGroupSymOps
{
    using operation_type = OperationType;

    std::vector<operation_type> operations;

    void clear() { operations.clear(); }
    bool empty() const { return operations.empty(); }
    std::size_t size() const { return operations.size(); }
    void reserve(const std::size_t count) { operations.reserve(count); }

    void push_back(const operation_type& operation) { operations.push_back(operation); }
    void push_back(operation_type&& operation) { operations.push_back(std::move(operation)); }

    operation_type& at(const std::size_t index) { return operations.at(index); }
    const operation_type& at(const std::size_t index) const { return operations.at(index); }

    operation_type& operator[](const std::size_t index) { return operations[index]; }
    const operation_type& operator[](const std::size_t index) const { return operations[index]; }

    auto begin() { return operations.begin(); }
    auto begin() const { return operations.begin(); }
    auto cbegin() const { return operations.cbegin(); }
    auto end() { return operations.end(); }
    auto end() const { return operations.end(); }
    auto cend() const { return operations.cend(); }
};

/*!
 * @brief Mapping from one atom to its inequivalent representative.
 *
 * For the fractional atom at index `atom`, stored implicitly by the vector
 * position of this entry, the mapped fractional operation satisfies:
 *
 * apply_space_group_symmetry_operation(operations[isym], atom_positions[atom])
 *   = atom_positions[inequivalent_atom] + return_lattice
 *
 * `isym == -1` is used only when no operation is available for a self mapping.
 */
struct AtomSymMapping
{
    int inequivalent_atom = -1;
    int isym = -1;
    Vector3_Order<int> return_lattice{0, 0, 0};
};

//! Fractional k-point folded by an integer reciprocal-lattice vector.
struct FoldedKPoint
{
    int target_k_index = -1;
    Vector3_Order<double> kpoint{0.0, 0.0, 0.0};
    Vector3_Order<int> fold_G{0, 0, 0};
};

//! One member of a fractional-k star generated from a full fractional k-grid.
struct KPointStarMember
{
    int full_k_index = -1;
    Vector3_Order<double> kpoint{0.0, 0.0, 0.0};
};

//! Symmetry mapping from the representative k-point to the same-index star member.
struct KPointSymMapping
{
    int isym = -1;
    //! Integer G where rotated representative k equals kpoint + G.
    Vector3_Order<int> fold_G{0, 0, 0};
};

//! One fractional-k star generated from a full fractional k-grid.
struct KPointStar
{
    //! Index into members for the representative k-point.
    int representative_k_index = -1;
    std::vector<KPointStarMember> members;
    std::vector<KPointSymMapping> sym_mappings;
};

inline Matrix3 multiply_space_group_rotation_matrices(const Matrix3& lhs, const Matrix3& rhs)
{
    return lhs * rhs;
}

inline Vector3_Order<double> multiply_row_vector(const Vector3_Order<double>& vec,
                                                 const Matrix3& matrix)
{
    return Vector3_Order<double>(vec * matrix);
}

SpaceGroupSymOp compose_space_group_symmetry_operations(
    const SpaceGroupSymOp& lhs,
    const SpaceGroupSymOp& rhs);

Vector3_Order<double> apply_space_group_symmetry_operation(
    const SpaceGroupSymOp& operation,
    const Vector3_Order<double>& coord);

//! Apply the reciprocal-space dual of a fractional direct-space rotation to fractional k.
Vector3_Order<double> apply_space_group_rotation_to_kpoint(
    const SpaceGroupSymOp& operation,
    const Vector3_Order<double>& kpoint);

FoldedKPoint fold_fractional_kpoint_to_targets(
    const Vector3_Order<double>& kpoint,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    double tol = 1e-8);

inline Matrix3 row_fractional_rotation_to_cartesian(const Matrix3& row_fractional_rotation,
                                                    const Matrix3& row_lattice_vectors)
{
    return row_lattice_vectors.Transpose() * row_fractional_rotation.Transpose() *
           row_lattice_vectors.Inverse().Transpose();
}

inline Matrix3 col_fractional_rotation_to_cartesian(const Matrix3& col_fractional_rotation,
                                                    const Matrix3& col_lattice_vectors)
{
    return col_lattice_vectors * col_fractional_rotation * col_lattice_vectors.Inverse();
}

inline Matrix3 fractional_rotation_to_cartesian(const SpaceGroupSymOp& symop,
                                                const Matrix3& lattice_vectors)
{
    return symop.use_row_convention
               ? row_fractional_rotation_to_cartesian(symop.rotation, lattice_vectors)
               : col_fractional_rotation_to_cartesian(symop.rotation, lattice_vectors);
}

//! Build atom mapping from fractional atom positions and fractional symmetry operations.
std::vector<AtomSymMapping> build_atom_to_inequivalent_symmetry_mapping(
    const std::vector<Vector3_Order<double>>& atom_positions_frac, const Matrix3& lattice_vectors,
    const SpaceGroupSymOps<SpaceGroupSymOp>& fractional_operations, double tol = 1e-5);

template <typename OperationType>
std::vector<AtomSymMapping> build_atom_to_inequivalent_symmetry_mapping(
    const std::vector<Vector3_Order<double>>& atom_positions_frac,
    const Matrix3& lattice_vectors,
    const SpaceGroupSymOps<OperationType>& fractional_operations,
    const double tol = 1e-5)
{
    SpaceGroupSymOps<SpaceGroupSymOp> base_operations;
    base_operations.reserve(fractional_operations.size());
    for (const auto& operation : fractional_operations)
    {
        SpaceGroupSymOp base_operation;
        base_operation.rotation = operation.rotation;
        base_operation.translation = operation.translation;
        base_operation.use_row_convention = operation.use_row_convention;
        base_operations.push_back(base_operation);
    }
    return build_atom_to_inequivalent_symmetry_mapping(
        atom_positions_frac, lattice_vectors, base_operations, tol);
}

std::vector<int> collect_inequivalent_atoms(
    const std::vector<AtomSymMapping>& mappings);

std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps<SpaceGroupSymOp>& fractional_operations,
    double tol = 1e-8);

//! Preferred representatives are tried in order; non-member hints are ignored.
std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps<SpaceGroupSymOp>& fractional_operations,
    const std::vector<Vector3_Order<double>>& preferred_representative_kpoints,
    double tol = 1e-8);

template <typename OperationType>
std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps<OperationType>& fractional_operations,
    const double tol = 1e-8)
{
    SpaceGroupSymOps<SpaceGroupSymOp> base_operations;
    base_operations.reserve(fractional_operations.size());
    for (const auto& operation : fractional_operations)
    {
        SpaceGroupSymOp base_operation;
        base_operation.rotation = operation.rotation;
        base_operation.translation = operation.translation;
        base_operation.use_row_convention = operation.use_row_convention;
        base_operations.push_back(base_operation);
    }
    return build_kpoint_stars(full_kpoints_frac, base_operations, tol);
}

template <typename OperationType>
std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps<OperationType>& fractional_operations,
    const std::vector<Vector3_Order<double>>& preferred_representative_kpoints,
    const double tol = 1e-8)
{
    SpaceGroupSymOps<SpaceGroupSymOp> base_operations;
    base_operations.reserve(fractional_operations.size());
    for (const auto& operation : fractional_operations)
    {
        SpaceGroupSymOp base_operation;
        base_operation.rotation = operation.rotation;
        base_operation.translation = operation.translation;
        base_operation.use_row_convention = operation.use_row_convention;
        base_operations.push_back(base_operation);
    }
    return build_kpoint_stars(
        full_kpoints_frac, base_operations, preferred_representative_kpoints, tol);
}

} // namespace librpa_int
