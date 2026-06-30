/*!
 * @file symmetry_context.h
 * @brief Utilities for generated symmetry data.
 */
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "atom.h"
#include "atomic_basis.h"
#include "geometry.h"
#include "../math/complexmatrix.h"
#include "../math/matrix3.h"
#include "../math/symmetry.h"
#include "../math/vector3_order.h"

namespace librpa_int
{

class PeriodicBoundaryData;

using symmetry_R_t = std::array<int, 3>;
using symmetry_irreducible_sector_t = std::map<atpair_t, std::set<symmetry_R_t>>;

/*!
 * @brief Real-space symmetry operation exported by a symmetry convention.
 */
struct SymmetryOperation : SpaceGroupSymOp
{
    std::map<int, ComplexMatrix> shell_rotations;
};

/*!
 * @brief Atom-resolved k-space symmetry information exported by a symmetry convention.
 */
struct SymmetryKAtomRotation
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
struct SymmetryKStarMember
{
    int isym = -1;
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
    int isym = -1;
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

/*!
 * @brief In-memory representation of the system symmetry.
 *
 * Built from the structure and full k-point grid, independent of which k-points
 * are stored in the PBC object. The context is populated from Dataset metadata
 * when a compute API initializes its calculation objects.
 */
struct SymmetryContext
{
    bool available = false;
    bool lattice_available = false;
    int ao_lmax = -1;
    int abf_lmax = -1;
    BasisConvention basis_convention;
    symmetry_irreducible_sector_t irreducible_sector;
    SpaceGroupSymOps<SymmetryOperation> rspace_operations;
    std::vector<SymmetryKStar> kstars;
    std::vector<SymmetryKStar> abf_kstars;
    std::map<std::string, std::vector<SpeciesBasisLayout>> map_key_layouts;
    std::map<atom_t, int> atom_to_type;
    std::map<atom_t, std::array<double, 3>> input_coord_frac;
    Matrix3 lattice_vectors;
    Matrix3 reciprocal_vectors;
    std::map<std::pair<int, int>, Vector3_Order<int>> kspace_return_lattice;
    std::map<std::pair<int, int>, Vector3_Order<int>> kstar_member_fold_G;

    // void build_shell_rotmat_T(int lmax, BasisConvention conv);

    void clear();
    void set_available();
    void unset_available();
    void add_rspace_operation(SymmetryOperation operation);
    void set_rspace_operations(std::vector<SymmetryOperation> operations);
    void set_lattice(const Matrix3& latvec, const Matrix3& G);
    void set_structure(const std::map<atom_t, int>& types,
                       const std::map<atom_t, coord_t>& coords_frac);
    void clear_basis_layouts();
    void add_basis_layouts(const std::string &label,
                           const std::vector<SpeciesBasisLayout> &layouts);
    void add_basis_layouts(const AtomicBasis &basis,
                           const std::map<atom_t, int> &types);
    void set_basis_layouts(const std::vector<const AtomicBasis*> &bases,
                           const std::map<atom_t, int> &types);
    int max_l() const;
    void generate_irreducible_sector(const std::vector<Vector3_Order<int>> &Rlist);
    void generate_kstars(const PeriodicBoundaryData &pbc);
    void generate_shell_rotations(const BasisConvention &basis_convention);
    void generate_kstar_member_rotations(int lmax);
    void print_summary(std::ostream& log) const;
    bool empty() const;
    bool has_shell_layout(const std::string &key) const;
    std::size_t count_irreducible_pairs() const;
    std::size_t count_irreducible_blocks() const;
    std::size_t count_kstar_members() const;
    std::size_t count_atoms_with_layout() const;
    const SpeciesBasisLayout& get_shell_layout(const std::string &key, const int atom_type) const;
};

std::string find_symmetry_shell_layout_key(
    const SymmetryContext& ctx,
    const std::map<atom_t, size_t>& atom_nb,
    const std::string& preferred_key);

std::vector<SymmetryOperation> make_symmetry_operations(
    int n_symops,
    bool use_row_convention,
    const int* rotmats,
    const double* translations = nullptr);

ComplexMatrix build_symmetry_shell_rotation_from_direct_rotation(
    const SpaceGroupSymOp& operation,
    const Matrix3& lattice_vectors,
    int l,
    const BasisConvention& basis_convention,
    double threshold = 1e-5);

std::map<int, ComplexMatrix> build_symmetry_shell_rotations_from_direct_rotation(
    const SpaceGroupSymOp& operation,
    const Matrix3& lattice_vectors,
    int lmax,
    const BasisConvention& basis_convention,
    double threshold = 1e-5);

std::complex<double> build_symmetry_kspace_phase(
    const Vector3_Order<double>& k_source,
    const Vector3_Order<double>& k_target,
    const Vector3_Order<double>& atom_from_frac,
    const Vector3_Order<double>& atom_to_frac,
    const Vector3_Order<int>& return_lattice,
    const BasisConvention& basis_convention);

std::map<int, ComplexMatrix> build_symmetry_kspace_shell_rotations(
    const SpaceGroupSymOp& operation,
    const Matrix3& lattice_vectors,
    int lmax,
    const BasisConvention& basis_convention,
    const Vector3_Order<double>& k_source,
    const Vector3_Order<double>& k_target,
    const Vector3_Order<double>& atom_from_frac,
    const Vector3_Order<double>& atom_to_frac,
    const Vector3_Order<int>& return_lattice,
    double threshold = 1e-5);

ComplexMatrix build_symmetry_rotation_matrix(
    const SymmetryContext& ctx, const std::string& key, const int atom_type,
    const std::map<int, ComplexMatrix>& shell_rotations);

ComplexMatrix build_symmetry_rotation_matrix(
    const SymmetryContext& ctx,
    const std::string &key,
    const int atom_type,
    const std::map<int, ComplexMatrix>& shell_rotations,
    const Matrix3& direct_rotation);

const SymmetryKStar& find_symmetry_kstar_for_kpoint(const std::vector<SymmetryKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label = "symmetry k-stars");

const SymmetryKStar& find_symmetry_kstar_for_ibz_kpoint(const SymmetryContext& ctx,
                                                    const Vector3_Order<double>& k_ibz);

std::vector<SymmetryKStarGridMappingEntry> build_symmetry_kstar_grid_mapping(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& klist_internal,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>>& irk_to_full_kpoints);

std::vector<SymmetryFullKpointMemberEntry> build_symmetry_full_kpoint_member_list(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list);

symmetry_kstar_representative_indices_t build_symmetry_full_grid_kstar_representative_indices(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list);

symmetry_kstar_member_kfrac_targets_t build_symmetry_full_grid_kstar_member_kfrac_targets(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list);

std::set<std::pair<atom_t, atom_t>> build_symmetry_upper_atom_pair_closure(
    const SymmetryKStar& star,
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs);

symmetry_atom_block_matrix_map_t rotate_symmetry_kspace_operator_blocks(
    const SymmetryContext& ctx,
    const std::string &key,
    const SymmetryKStarMember& member,
    const symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    bool use_time_reversal = false,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr,
    const Vector3_Order<double>* k_bz_target = nullptr);

symmetry_atom_block_matrix_map_t symmetrize_symmetry_ibz_kspace_operator_blocks(
    const SymmetryContext& ctx,
    const std::string &key,
    const Vector3_Order<double>& k_ibz,
    const symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const SymmetryKStar* abf_star = nullptr,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr);

ComplexMatrix rotate_symmetry_kspace_operator_matrix(
    const SymmetryContext& ctx,
    const std::string &key,
    const SymmetryKStarMember& member,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    bool use_time_reversal = false,
    const Vector3_Order<double>* k_bz_target = nullptr);

ComplexMatrix symmetrize_symmetry_ibz_kspace_operator_matrix(
    const SymmetryContext& ctx,
    const std::string &key,
    const Vector3_Order<double>& k_ibz,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac);

ComplexMatrix rotate_symmetry_kspace_matrix(const SymmetryContext& ctx,
                                          const std::string &key,
                                          const SymmetryKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const std::map<atom_t, std::array<double, 3>>& coord_frac,
                                          const bool use_time_reversal = false,
                                          const Vector3_Order<double>* k_bz_target = nullptr);

symmetry_irreducible_sector_t build_symmetry_rspace_irreducible_sector(
    const SymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const std::vector<Vector3_Order<int>>& Rlist);

void build_symmetry_rspace_sector_stars(
    const SymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& Rlist,
    symmetry_rspace_sector_stars_t& sector_stars,
    std::ostream* log = nullptr);

ComplexMatrix rotate_symmetry_rspace_matrix(const SymmetryContext& ctx,
                                          const std::string& key,
                                          const int isym,
                                          atom_t atom_from_i,
                                          atom_t atom_from_j,
                                          const ComplexMatrix& matrix_source);

} // namespace librpa_int
