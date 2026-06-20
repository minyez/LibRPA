/*!
 * @file symmetry.h
 * @brief Space-group symmetry operation primitives.
 */
#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <map>

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
    static const SpaceGroupSymOp IDENTITY;
    static const SpaceGroupSymOp INVERSE;
    static const SpaceGroupSymOp C41_Z;

    // Default to identity operation
    Matrix3 rotation{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0};
    Vector3_Order<double> translation{0.0, 0.0, 0.0};
    //! Whether to treat lattice vectors and fractional coordinates as row vectors.
    bool use_row_convention = true;

    bool is_identity_rotation() const
    {
        return is_same_matrix(this->rotation, Matrix3::IDENTITY, 1e-8);
    }
};

inline bool operator==(const SpaceGroupSymOp &op1, const SpaceGroupSymOp &op2)
{
    return (op1.use_row_convention == op2.use_row_convention) &&
           is_same_matrix(op1.rotation, op2.rotation, 1e-5) &&
           (op1.translation == op2.translation);
}

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
 * @brief Atom mapping induced by one space-group operation.
 *
 * For the atom at vector index `atom`, applying the operation maps it to
 * `atom_map[atom] + return_lattice[atom]` in fractional coordinates.
 */
template <typename AtomIndex>
struct SpaceGroupAtomMapping
{
    using atom_index_type = AtomIndex;

    std::vector<atom_index_type> atom_map;
    std::vector<Vector3_Order<int>> return_lattice;
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

template <typename AtomIndex, typename coord_t, typename AtomType>
SpaceGroupAtomMapping<AtomIndex> get_space_group_atom_mapping(
    const SpaceGroupSymOp &op,
    const std::map<AtomIndex, coord_t>& coord_frac,
    const std::map<AtomIndex, AtomType> &atom_to_type, double tol = 1e-5)
{
    if (coord_frac.size() != atom_to_type.size())
    {
        throw std::runtime_error("Fractional coordinates and atom mapping have inconsistent sizes");
    }

    SpaceGroupAtomMapping<AtomIndex> info;
    info.atom_map.resize(coord_frac.size(), static_cast<AtomIndex>(-1));
    info.return_lattice.resize(coord_frac.size(), {0, 0, 0});

    {
        for (AtomIndex atom_from = 0; atom_from < coord_frac.size(); ++atom_from)
        {
            const auto& coord_from = coord_frac.at(atom_from);
            const Vector3_Order<double> coord_from_vec =
                restrict_fractional_coordinate(coord_from, tol);
            // Keep the unwrapped rotated position so that the integer return lattice is preserved
            // exactly as in the ABACUS irreducible-sector construction.
            const Vector3_Order<double> transformed =
                apply_space_group_symmetry_operation(op, coord_from_vec);

            AtomIndex matched_atom;
            bool matched = false;
            Vector3_Order<int> matched_return{0, 0, 0};
            for (AtomIndex atom_to = 0; atom_to < coord_frac.size(); ++atom_to)
            {
                if (atom_to_type.at(atom_from) != atom_to_type.at(atom_to))
                {
                    continue;
                }
                const auto& coord_to = coord_frac.at(atom_to);
                const Vector3_Order<double> coord_to_vec =
                    restrict_fractional_coordinate(coord_to, tol);
                const Vector3_Order<double> diff = transformed - coord_to_vec;
                if (!nearly_integer_vector(diff, tol))
                {
                    continue;
                }
                matched = true;
                matched_atom = atom_to;
                matched_return = round_to_integer_vector(diff);
            }

            if (!matched)
            {
                throw std::runtime_error("Failed to match real-space symmetry atom mapping");
            }

            info.atom_map[atom_from] = matched_atom;
            info.return_lattice[atom_from] = matched_return;
        }
    }
    return info;
}

} // namespace librpa_int
