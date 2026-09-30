/*!
 * @file symmetry.h
 * @brief Space-group symmetry operation primitives.
 */
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
#include <map>
#include <set>

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

    bool is_identity() const
    {
        return is_identity_rotation() && nearly_integer_vector(translation, 1e-8);
    }

    void to_row_convention();

    void to_col_convention();
};

inline bool operator==(const SpaceGroupSymOp &op1, const SpaceGroupSymOp &op2)
{
    return (op1.use_row_convention == op2.use_row_convention) &&
           is_same_matrix(op1.rotation, op2.rotation, 1e-5) &&
           (op1.translation == op2.translation);
}

inline void SpaceGroupSymOp::to_row_convention()
{
    if (use_row_convention) return;
    rotation = rotation.Transpose();
    use_row_convention = true;
}

inline void SpaceGroupSymOp::to_col_convention()
{
    if (!use_row_convention) return;
    rotation = rotation.Transpose();
    use_row_convention = false;
}

using SpaceGroupSymOps = std::vector<SpaceGroupSymOp>;

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

using SpaceGroupRSpaceSector =
    std::map<std::pair<int, int>, std::set<Vector3_Order<int>>>;

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

//! True when two fractional k-points differ by an integer reciprocal vector.
bool same_fractional_kpoint(const Vector3_Order<double>& lhs,
                            const Vector3_Order<double>& rhs,
                            double tol = 1e-8);

bool try_fold_fractional_kpoint_to_target(
    const Vector3_Order<double>& kpoint,
    const Vector3_Order<double>& target_kpoint,
    double tol,
    FoldedKPoint& folded);

FoldedKPoint fold_fractional_kpoint_to_targets(
    const Vector3_Order<double>& kpoint,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    double tol = 1e-8);

bool preserves_lattice_metric(const Matrix3& rotation,
                              const Matrix3& lattice_vectors,
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

/*!
 * @brief Proper (axial-vector) part of a Cartesian rotation: det(R) * R.
 *
 * For an improper operation Q (det = -1, e.g. mirror or inversion) the spin
 * transforms as an axial vector, i.e. only through the proper part det(Q) Q.
 * The determinant is identical in fractional and Cartesian coordinates, so the
 * caller may pass either representation as long as it is Cartesian-basis
 * orthonormal when fed to so3_to_su2.
 */
Matrix3 axial_rotation_of(const Matrix3 &cartesian_rotation);

/*!
 * @brief Lift a proper Cartesian SO(3) rotation to SU(2), row-major [u00 u01; u10 u11].
 *
 * Axis-angle formula U = cos(theta/2) I - i sin(theta/2) (n . sigma) with
 * theta = arccos((Tr R - 1) / 2) and n proportional to the antisymmetric part
 * (R_zy - R_yz, R_xz - R_zx, R_yx - R_xy) — the sign that makes
 * su2_to_so3(so3_to_su2(R)) == R for a matrix acting on column vectors,
 * (R u)_i = R_ij u_j. The input is the Cartesian proper rotation matrix
 * itself; no transpose guessing is involved. (The planning report writes the
 * tuple in the swapped order (R_yz - R_zy, ...); that is the row-vector
 * reading x' = x R and yields the adjoint U^dagger for the same numerical
 * matrix.) LibRPA stores lattice vectors and fractional coordinates as row
 * vectors (C1), but a Cartesian rotation converted by
 * fractional_rotation_to_cartesian is a plain Cartesian matrix and is fed
 * here as-is. ABACUS instead builds spin_so3 = proper_part(g).Transpose()
 * because its g acts on column fractional/Cartesian vectors with the opposite
 * convention; the transpose there converts conventions, it is not part of the
 * axis-angle map.
 *
 * theta = pi branch: R = 2 n n^T - I, the axis sign is anchored on the
 * largest diagonal element and the remaining components are recovered from
 * R_ij + R_ji = 4 n_i n_j (aligned with the ABACUS fix in commit 1ed54ee23).
 * Away from pi the axis is the normalized antisymmetric part and
 * sin(theta/2) = sqrt(1 - cos(theta/2)^2), so theta = 0 returns the identity
 * without any 0/0. Conditioning limit: the trace-based cos(theta/2) loses
 * significance within ~1e-7 of pi, so genuine rotations in the narrow sliver
 * pi - 2e-7 < theta < pi cannot be round-tripped to 1e-12 in double
 * precision; exact pi rotations are handled by the pi branch.
 *
 * The result satisfies su2_to_so3(so3_to_su2(R)) == R and is defined up to the
 * SU(2) double-cover sign: U and -U induce the same bilinear X -> U X U^dag.
 */
std::array<std::complex<double>, 4> so3_to_su2(const Matrix3 &proper_rotation);

/*!
 * @brief Covering map SU(2) -> SO(3): W_ij = (1/2) Tr(sigma_i U sigma_j U^dag).
 *
 * Inverse check of so3_to_su2, also used to cross-check Pauli-based rotation
 * formulas. Row-major [u00 u01; u10 u11] input, Cartesian Matrix3 output.
 */
Matrix3 su2_to_so3(const std::array<std::complex<double>, 4> &U);

int find_identity_symmetry_operation(const SpaceGroupSymOps &operations);

//! Build atom mapping from fractional atom positions and fractional symmetry operations.
std::vector<AtomSymMapping> build_atom_to_inequivalent_symmetry_mapping(
    const std::vector<Vector3_Order<double>>& atom_positions_frac, const Matrix3& lattice_vectors,
    const SpaceGroupSymOps& fractional_operations, double tol = 1e-5);

std::vector<int> collect_inequivalent_atoms(
    const std::vector<AtomSymMapping>& mappings);

SpaceGroupRSpaceSector build_space_group_rspace_irreducible_sector(
    const SpaceGroupSymOps& fractional_operations,
    const std::map<int, Vector3_Order<double>>& coord_frac,
    const std::map<int, int>& atom_to_type,
    const std::vector<Vector3_Order<int>>& Rlist,
    const Matrix3* lattice_vectors = nullptr,
    double atom_map_tol = 5e-5,
    double coord_tol = 1e-5);

std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps& fractional_operations,
    double tol = 1e-8);

//! Preferred representatives are tried in order; non-member hints are ignored.
std::vector<KPointStar> build_kpoint_stars(
    const std::vector<Vector3_Order<double>>& full_kpoints_frac,
    const SpaceGroupSymOps& fractional_operations,
    const std::vector<Vector3_Order<double>>& preferred_representative_kpoints,
    double tol = 1e-8);

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
        const auto atom_count = static_cast<AtomIndex>(coord_frac.size());
        for (AtomIndex atom_from = 0; atom_from < atom_count; ++atom_from)
        {
            const auto& coord_from = coord_frac.at(atom_from);
            const Vector3_Order<double> coord_from_vec(coord_from);
            const Vector3_Order<double> transformed =
                apply_space_group_symmetry_operation(op, coord_from_vec);

            AtomIndex matched_atom;
            bool matched = false;
            Vector3_Order<int> matched_return{0, 0, 0};
            for (AtomIndex atom_to = 0; atom_to < atom_count; ++atom_to)
            {
                if (atom_to_type.at(atom_from) != atom_to_type.at(atom_to))
                {
                    continue;
                }
                const auto& coord_to = coord_frac.at(atom_to);
                const Vector3_Order<double> coord_to_vec(coord_to);
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
