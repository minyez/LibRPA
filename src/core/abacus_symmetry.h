/*!
 * @file abacus_symmetry.h
 * @brief Utilities for reading ABACUS symmetry sidecar files.
 */
#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "atom.h"
#include "../math/complexmatrix.h"
#include "../math/matrix3.h"
#include "../math/vector3_order.h"

namespace LIBRPA
{

using librpa_int::atom_t;
using librpa_int::atpair_t;
using librpa_int::ComplexMatrix;
using librpa_int::Matrix3;
using librpa_int::Vector3_Order;

using abacus_R_t = std::array<int, 3>;
using abacus_irreducible_sector_t = std::map<atpair_t, std::set<abacus_R_t>>;

/*!
 * @brief Real-space symmetry operation exported by ABACUS.
 */
struct AbacusSymmetryOperation
{
    int isym = -1;
    std::array<std::array<double, 3>, 3> rotation{{{{0.0, 0.0, 0.0}},
                                                    {{0.0, 0.0, 0.0}},
                                                    {{0.0, 0.0, 0.0}}}};
    Vector3_Order<double> translation{0.0, 0.0, 0.0};
    std::map<int, ComplexMatrix> shell_rotations;
};

/*!
 * @brief Atom-resolved k-space symmetry information exported by ABACUS.
 */
struct AbacusKAtomRotation
{
    int atom_from = -1;
    int atom_to = -1;
    int atom_type = -1;
    int lmax = -1;
    std::map<int, ComplexMatrix> shell_rotations;
};

/*!
 * @brief One member of an irreducible k-star.
 */
struct AbacusKStarMember
{
    int isym = -1;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
    std::vector<AbacusKAtomRotation> atom_rotations;
};

/*!
 * @brief One irreducible k-star exported by ABACUS.
 */
struct AbacusKStar
{
    int star_index = -1;
    Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::vector<AbacusKStarMember> members;
};

/*!
 * @brief Explicit mapping between one loaded LibRPA IBZ q-index and one ABACUS k-star.
 *
 * The mapping stores the full-BZ q keys that LibRPA should use for every star member.
 * ABACUS sidecar coordinates are treated as the source of truth; when LibRPA already
 * has an equivalent internal q key, that exact key is reused to keep later lookups
 * aligned with existing storage.
 */
struct AbacusKStarGridMappingEntry
{
    int iq_ibz = -1;
    int star_list_index = -1;
    std::vector<Vector3_Order<double>> member_q_bz_keys;
};

/*!
 * @brief One full-BZ k-point member expanded from an ABACUS IBZ k-star.
 */
struct AbacusFullKpointMemberEntry
{
    int ik_ibz = -1;
    int star_list_index = -1;
    int member_index = -1;
    int isym = -1;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
};

/*!
 * @brief AO shell layout of one ABACUS atom type.
 *
 * The shell multiplicities follow the ABACUS orbital ordering:
 * increasing angular momentum `l`, then zeta index, then magnetic index.
 */
struct AbacusAOTypeLayout
{
    std::string label;
    std::string orbital_file;
    std::vector<int> shell_counts;
    int nao = 0;
};

/*!
 * @brief One full real-space member generated from an irreducible {atom pair, R}.
 */
struct AbacusRSpaceRestoreMember
{
    int isym = -1;
    atpair_t full_atom_pair;
    Vector3_Order<int> full_R{0, 0, 0};
};

using abacus_rspace_sector_stars_t =
    std::map<atpair_t, std::map<Vector3_Order<int>, std::vector<AbacusRSpaceRestoreMember>>>;
using abacus_atom_block_matrix_map_t = std::map<atom_t, std::map<atom_t, ComplexMatrix>>;

/*!
 * @brief In-memory representation of ABACUS symmetry sidecar files.
 *
 * The context is intentionally read-only after loading. It will be used by later
 * EXX/GW symmetry implementations to avoid reparsing the sidecar files.
 */
struct AbacusSymmetryContext
{
    bool available = false;
    bool lattice_available = false;
    bool ao_shell_layout_available = false;
    bool abf_shell_layout_available = false;
    int ao_lmax = -1;
    int abf_lmax = -1;
    abacus_irreducible_sector_t irreducible_sector;
    std::vector<AbacusSymmetryOperation> rspace_operations;
    std::vector<AbacusKStar> kstars;
    std::vector<AbacusKStar> abf_kstars;
    std::vector<AbacusAOTypeLayout> ao_type_layouts;
    std::vector<std::vector<AbacusAOTypeLayout>> abf_type_layout_candidates;
    std::map<atom_t, int> atom_to_type;
    std::map<atom_t, std::array<double, 3>> input_coord_frac;
    Matrix3 lattice_vectors;
    Matrix3 reciprocal_vectors;
    std::map<std::pair<int, int>, Vector3_Order<int>> kspace_return_lattice;
    std::map<std::pair<int, int>, Vector3_Order<int>> kstar_member_fold_G;

    void clear();
    void set_lattice(const Matrix3& latvec, const Matrix3& G);
    bool empty() const;
    bool has_ao_shell_layout() const;
    bool has_abf_shell_layout() const;
    std::size_t count_irreducible_pairs() const;
    std::size_t count_irreducible_blocks() const;
    std::size_t count_kstar_members() const;
    std::size_t count_atoms_with_layout() const;
    std::size_t count_abf_layout_candidates() const;
    const AbacusAOTypeLayout& get_ao_type_layout(int atom_type) const;
    const AbacusAOTypeLayout& find_abf_type_layout(int atom_type, int nao_hint) const;
};

extern AbacusSymmetryContext abacus_symmetry_ctx;

bool load_abacus_symmetry_context(const std::string& dir_path,
                                  AbacusSymmetryContext& ctx,
                                  std::ostream* log = nullptr);

bool load_global_abacus_symmetry_context(const std::string& dir_path,
                                         std::ostream* log = nullptr);

ComplexMatrix build_abacus_ao_rotation_matrix(const AbacusSymmetryContext& ctx,
                                              int atom_type,
                                              const std::map<int, ComplexMatrix>& shell_rotations);

ComplexMatrix build_abacus_abf_rotation_matrix(
    const AbacusSymmetryContext& ctx,
    int atom_type,
    int nao_hint,
    const std::map<int, ComplexMatrix>& shell_rotations,
    const std::array<std::array<double, 3>, 3>& direct_rotation);

const AbacusKStar& find_abacus_kstar_for_kpoint(const std::vector<AbacusKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label = "ABACUS k-stars");

const AbacusKStar& find_abacus_kstar_for_ibz_kpoint(const AbacusSymmetryContext& ctx,
                                                    const Vector3_Order<double>& k_ibz);

std::vector<AbacusKStarGridMappingEntry> build_abacus_kstar_grid_mapping(
    const AbacusSymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& klist_internal,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>>& irk_to_full_kpoints);

std::vector<AbacusFullKpointMemberEntry> build_abacus_full_kpoint_member_list(
    const AbacusSymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list);

std::set<std::pair<atom_t, atom_t>> build_abacus_upper_atom_pair_closure(
    const AbacusKStar& star,
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs);

abacus_atom_block_matrix_map_t rotate_abacus_abf_kspace_operator_blocks(
    const AbacusSymmetryContext& ctx,
    const AbacusKStarMember& member,
    const abacus_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    bool use_time_reversal = false,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr,
    const Vector3_Order<double>* k_bz_target = nullptr);

abacus_atom_block_matrix_map_t symmetrize_abacus_abf_ibz_kspace_operator_blocks(
    const AbacusSymmetryContext& ctx,
    const Vector3_Order<double>& k_ibz,
    const abacus_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const AbacusKStar* abf_star = nullptr,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr);

ComplexMatrix rotate_abacus_abf_kspace_operator_matrix(
    const AbacusSymmetryContext& ctx,
    const AbacusKStarMember& member,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    bool use_time_reversal = false,
    const Vector3_Order<double>* k_bz_target = nullptr);

ComplexMatrix symmetrize_abacus_abf_ibz_kspace_operator_matrix(
    const AbacusSymmetryContext& ctx,
    const Vector3_Order<double>& k_ibz,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac);

ComplexMatrix rotate_abacus_kspace_matrix(const AbacusSymmetryContext& ctx,
                                          const AbacusKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const std::map<atom_t, std::array<double, 3>>& coord_frac,
                                          bool use_time_reversal = false,
                                          const Vector3_Order<double>* k_bz_target = nullptr);

void build_abacus_rspace_sector_stars(
    const AbacusSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& Rlist,
    abacus_rspace_sector_stars_t& sector_stars,
    std::ostream* log = nullptr);

ComplexMatrix rotate_abacus_rspace_matrix(const AbacusSymmetryContext& ctx,
                                          int isym,
                                          atom_t atom_from_i,
                                          atom_t atom_from_j,
                                          const ComplexMatrix& matrix_source);

ComplexMatrix rotate_abacus_abf_rspace_matrix(const AbacusSymmetryContext& ctx,
                                              int isym,
                                              atom_t atom_from_i,
                                              atom_t atom_from_j,
                                              const ComplexMatrix& matrix_source);

} // namespace LIBRPA
