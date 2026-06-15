/*!
 * @file input_symmetry.h
 * @brief Utilities for reading input symmetry sidecar files.
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
#include "atomic_basis.h"
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

using input_symmetry_R_t = std::array<int, 3>;
using input_symmetry_irreducible_sector_t = std::map<atpair_t, std::set<input_symmetry_R_t>>;

enum class InputSymmetryConvention
{
    NONE = 0,
    AUTO = 1,
    ABACUS = 2,
};

InputSymmetryConvention parse_input_symmetry_convention(const std::string& convention);
std::string input_symmetry_convention_name(InputSymmetryConvention convention);

/*!
 * @brief Real-space symmetry operation exported by an input symmetry convention.
 */
struct InputSymmetryOperation
{
    int isym = -1;
    std::array<std::array<double, 3>, 3> rotation{{{{0.0, 0.0, 0.0}},
                                                    {{0.0, 0.0, 0.0}},
                                                    {{0.0, 0.0, 0.0}}}};
    Vector3_Order<double> translation{0.0, 0.0, 0.0};
    std::map<int, ComplexMatrix> shell_rotations;
};

/*!
 * @brief Atom-resolved k-space symmetry information exported by an input symmetry convention.
 */
struct InputSymmetryKAtomRotation
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
struct InputSymmetryKStarMember
{
    int isym = -1;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
    std::vector<InputSymmetryKAtomRotation> atom_rotations;
};

/*!
 * @brief One irreducible k-star exported by an input symmetry convention.
 */
struct InputSymmetryKStar
{
    int star_index = -1;
    Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::vector<InputSymmetryKStarMember> members;
};

/*!
 * @brief Explicit mapping between one loaded LibRPA IBZ q-index and one input symmetry k-star.
 *
 * The mapping stores the full-BZ q keys that LibRPA should use for every star member.
 * Input sidecar coordinates are treated as the source of truth; when LibRPA already
 * has an equivalent internal q key, that exact key is reused to keep later lookups
 * aligned with existing storage.
 */
struct InputSymmetryKStarGridMappingEntry
{
    int iq_ibz = -1;
    int star_list_index = -1;
    std::vector<Vector3_Order<double>> member_q_bz_keys;
};

/*!
 * @brief One full-BZ k-point member expanded from an input symmetry IBZ k-star.
 */
struct InputSymmetryFullKpointMemberEntry
{
    int ik_ibz = -1;
    int star_list_index = -1;
    int member_index = -1;
    int isym = -1;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
};

/*!
 * @brief AO shell layout of one input symmetry atom type.
 *
 * The shell multiplicities follow the selected input symmetry orbital ordering:
 * increasing angular momentum `l`, then zeta index, then magnetic index.
 */
struct InputSymmetryAOTypeLayout
{
    std::string label;
    std::string orbital_file;
    std::vector<int> shell_counts;
    int nao = 0;
};

/*!
 * @brief One full real-space member generated from an irreducible {atom pair, R}.
 */
struct InputSymmetryRSpaceRestoreMember
{
    int isym = -1;
    atpair_t full_atom_pair;
    Vector3_Order<int> full_R{0, 0, 0};
};

using input_symmetry_rspace_sector_stars_t =
    std::map<atpair_t, std::map<Vector3_Order<int>, std::vector<InputSymmetryRSpaceRestoreMember>>>;
using input_symmetry_atom_block_matrix_map_t = std::map<atom_t, std::map<atom_t, ComplexMatrix>>;

/*!
 * @brief In-memory representation of input symmetry sidecar files.
 *
 * The context is intentionally read-only after loading. It will be used by later
 * EXX/GW symmetry implementations to avoid reparsing the sidecar files.
 */
struct InputSymmetryContext
{
    InputSymmetryConvention convention = InputSymmetryConvention::NONE;
    bool available = false;
    bool lattice_available = false;
    bool ao_shell_layout_available = false;
    bool abf_shell_layout_available = false;
    int ao_lmax = -1;
    int abf_lmax = -1;
    librpa_int::BasisConvention basis_convention;
    input_symmetry_irreducible_sector_t irreducible_sector;
    std::vector<InputSymmetryOperation> rspace_operations;
    std::vector<InputSymmetryKStar> kstars;
    std::vector<InputSymmetryKStar> abf_kstars;
    std::vector<InputSymmetryAOTypeLayout> ao_type_layouts;
    std::vector<std::vector<InputSymmetryAOTypeLayout>> abf_type_layout_candidates;
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
    const InputSymmetryAOTypeLayout& get_ao_type_layout(int atom_type) const;
    const InputSymmetryAOTypeLayout& find_abf_type_layout(int atom_type, int nao_hint) const;
};

extern InputSymmetryContext input_symmetry_ctx;

bool load_input_symmetry_context(const std::string& dir_path,
                                  InputSymmetryConvention convention,
                                  InputSymmetryContext& ctx,
                                  std::ostream* log = nullptr);

bool load_global_input_symmetry_context(const std::string& dir_path,
                                         InputSymmetryConvention convention,
                                         std::ostream* log = nullptr);

ComplexMatrix build_input_symmetry_ao_rotation_matrix(const InputSymmetryContext& ctx,
                                              int atom_type,
                                              const std::map<int, ComplexMatrix>& shell_rotations);

ComplexMatrix build_input_symmetry_abf_rotation_matrix(
    const InputSymmetryContext& ctx,
    int atom_type,
    int nao_hint,
    const std::map<int, ComplexMatrix>& shell_rotations,
    const std::array<std::array<double, 3>, 3>& direct_rotation);

const InputSymmetryKStar& find_input_symmetry_kstar_for_kpoint(const std::vector<InputSymmetryKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label = "input symmetry k-stars");

const InputSymmetryKStar& find_input_symmetry_kstar_for_ibz_kpoint(const InputSymmetryContext& ctx,
                                                    const Vector3_Order<double>& k_ibz);

std::vector<InputSymmetryKStarGridMappingEntry> build_input_symmetry_kstar_grid_mapping(
    const InputSymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& klist_internal,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>>& irk_to_full_kpoints);

std::vector<InputSymmetryFullKpointMemberEntry> build_input_symmetry_full_kpoint_member_list(
    const InputSymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list);

std::set<std::pair<atom_t, atom_t>> build_input_symmetry_upper_atom_pair_closure(
    const InputSymmetryKStar& star,
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs);

input_symmetry_atom_block_matrix_map_t rotate_input_symmetry_abf_kspace_operator_blocks(
    const InputSymmetryContext& ctx,
    const InputSymmetryKStarMember& member,
    const input_symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    bool use_time_reversal = false,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr,
    const Vector3_Order<double>* k_bz_target = nullptr);

input_symmetry_atom_block_matrix_map_t symmetrize_input_symmetry_abf_ibz_kspace_operator_blocks(
    const InputSymmetryContext& ctx,
    const Vector3_Order<double>& k_ibz,
    const input_symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const InputSymmetryKStar* abf_star = nullptr,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr);

ComplexMatrix rotate_input_symmetry_abf_kspace_operator_matrix(
    const InputSymmetryContext& ctx,
    const InputSymmetryKStarMember& member,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    bool use_time_reversal = false,
    const Vector3_Order<double>* k_bz_target = nullptr);

ComplexMatrix symmetrize_input_symmetry_abf_ibz_kspace_operator_matrix(
    const InputSymmetryContext& ctx,
    const Vector3_Order<double>& k_ibz,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac);

ComplexMatrix rotate_input_symmetry_kspace_matrix(const InputSymmetryContext& ctx,
                                          const InputSymmetryKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const std::map<atom_t, std::array<double, 3>>& coord_frac,
                                          bool use_time_reversal = false,
                                          const Vector3_Order<double>* k_bz_target = nullptr);

void build_input_symmetry_rspace_sector_stars(
    const InputSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& Rlist,
    input_symmetry_rspace_sector_stars_t& sector_stars,
    std::ostream* log = nullptr);

ComplexMatrix rotate_input_symmetry_rspace_matrix(const InputSymmetryContext& ctx,
                                          int isym,
                                          atom_t atom_from_i,
                                          atom_t atom_from_j,
                                          const ComplexMatrix& matrix_source);

ComplexMatrix rotate_input_symmetry_abf_rspace_matrix(const InputSymmetryContext& ctx,
                                              int isym,
                                              atom_t atom_from_i,
                                              atom_t atom_from_j,
                                              const ComplexMatrix& matrix_source);

} // namespace LIBRPA
