/*!
 * @file symmetry_types.h
 * @brief Shared symmetry data structures.
 */
#pragma once

#include <array>
#include <map>
#include <set>
#include <vector>

#include "atom.h"
#include "../math/complexmatrix.h"
#include "../math/symmetry.h"
#include "../math/vector3_order.h"

namespace librpa_int
{

using symmetry_R_t = std::array<int, 3>;
using symmetry_irreducible_sector_t = std::map<atpair_t, std::set<symmetry_R_t>>;

using SymmetryOperation = SpaceGroupSymOp;

/*!
 * @brief Atom-resolved k-space symmetry information exported by a symmetry convention.
 */
struct SymmetryKAtomRotation
{
    int atom_from = -1;
    int atom_to = -1;
    int atom_type = -1;
    int lmax = -1;
    std::map<int, ComplexMatrix> bloch_rsh_rotations;
};

/*!
 * @brief One member of an irreducible k-star.
 */
struct SymmetryKStarMember
{
    int spatial_isym = -1;
    bool time_reversal = false;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
    std::vector<SymmetryKAtomRotation> atom_rotations;
};

/*!
 * @brief One irreducible k-star exported by a symmetry convention.
 */
struct SymmetryKStar
{
    int star_index = -1;
    Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::vector<SymmetryKStarMember> members;
};

/*!
 * @brief Explicit mapping between one loaded LibRPA IBZ q-index and one symmetry k-star.
 *
 * The mapping stores the full-BZ q keys that LibRPA should use for every star member.
 * Symmetry coordinates are treated as the source of truth; when LibRPA already
 * has an equivalent internal q key, that exact key is reused to keep later lookups
 * aligned with existing storage.
 */
struct SymmetryKStarGridMappingEntry
{
    int iq_ibz = -1;
    int star_list_index = -1;
    std::vector<Vector3_Order<double>> member_q_bz_keys;
};

/*!
 * @brief One full-BZ k-point member expanded from a symmetry IBZ k-star.
 */
struct SymmetryFullKpointMemberEntry
{
    int ik_ibz = -1;
    int star_list_index = -1;
    int member_index = -1;
    int spatial_isym = -1;
    bool time_reversal = false;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
};

/*!
 * @brief One full real-space member generated from an irreducible {atom pair, R}.
 */
struct SymmetryRSpaceRestoreMember
{
    int isym = -1;
    atpair_t full_atom_pair;
    Vector3_Order<int> full_R{0, 0, 0};
};

using symmetry_rspace_sector_stars_t =
    std::map<atpair_t, std::map<Vector3_Order<int>, std::vector<SymmetryRSpaceRestoreMember>>>;
using symmetry_atom_block_matrix_map_t = std::map<atom_t, std::map<atom_t, ComplexMatrix>>;
using symmetry_kstar_member_kfrac_targets_t =
    std::vector<std::vector<Vector3_Order<double>>>;
using symmetry_kstar_representative_indices_t = std::vector<int>;

} // namespace librpa_int
