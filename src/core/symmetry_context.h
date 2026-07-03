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
#include <vector>

#include "atom.h"
#include "atomic_basis.h"
#include "geometry.h"
#include "pbc.h"
#include "symmetry_types.h"
#include "../math/complexmatrix.h"
#include "../math/matrix3.h"
#include "../math/symmetry.h"
#include "../math/vector3_order.h"

namespace librpa_int
{

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
    BasisConvention basis_convention;
    symmetry_irreducible_sector_t irreducible_sector;
    SpaceGroupSymOps rspace_operations;
    std::vector<std::map<int, ComplexMatrix>> rsh_rotations;
    std::vector<SymmetryKStar> kstars;
    std::vector<SymmetryKStarGridMappingEntry> kstar_grid_mapping;
    std::vector<SymmetryFullKpointMemberEntry> full_kpoint_members;
    std::map<atom_t, int> atom_to_type;
    std::map<atom_t, std::array<double, 3>> input_coord_frac;
    Matrix3 lattice_vectors;
    Matrix3 reciprocal_vectors;
    std::map<std::pair<int, int>, Vector3_Order<int>> kspace_return_lattice;
    std::map<std::pair<int, int>, Vector3_Order<int>> kstar_member_fold_G;

    void clear();
    void set_available();
    void unset_available();
    void add_rspace_operation(SymmetryOperation operation);
    void set_rspace_operations(std::vector<SymmetryOperation> operations);
    void set_crystal_structure(const Matrix3& latvec,
                               const Matrix3& reciprocal,
                               const std::map<atom_t, int>& atom_types,
                               const std::map<atom_t, coord_t>& coords_frac);
    void build_periodic_mappings(const PeriodicBoundaryData& pbc,
                                 const std::vector<Vector3_Order<int>>& Rlist);
    void build_rsh_rotations(const BasisConvention &basis_convention, int lmax);
    void build_kstar_member_rotations(int lmax);
    ComplexMatrix get_rotation_matrix(const std::vector<SpeciesBasisLayout>& layouts,
                                      int atom_type,
                                      int isym) const;
    void print_summary(std::ostream& log) const;
    bool empty() const;
    std::size_t count_irreducible_pairs() const;
    std::size_t count_irreducible_blocks() const;
    std::size_t count_kstar_members() const;

private:
    void generate_irreducible_sector(const std::vector<Vector3_Order<int>> &Rlist);
    void generate_kstars(const PeriodicBoundaryData &pbc);
    void generate_kstar_grid_mapping(const PeriodicBoundaryData &pbc);
    void generate_full_kpoint_members(const std::vector<Vector3_Order<double>>& kfrac_list);
};

bool symmetry_species_layouts_match_atom_counts(
    const std::vector<SpeciesBasisLayout>& layouts,
    const std::map<atom_t, int>& atom_to_type,
    const std::map<atom_t, size_t>& atom_nb);

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
    const SpeciesBasisLayout& layout,
    const std::map<int, ComplexMatrix>& shell_rotations);

const SymmetryKStar& find_symmetry_kstar_for_kpoint(const std::vector<SymmetryKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label = "symmetry k-stars");

const SymmetryKStar& find_symmetry_kstar_for_ibz_kpoint(const SymmetryContext& ctx,
                                                    const Vector3_Order<double>& k_ibz);

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
    const std::vector<SpeciesBasisLayout>& layouts,
    const SymmetryKStarMember& member,
    const symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    bool use_time_reversal = false,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr,
    const Vector3_Order<double>* k_bz_target = nullptr);

symmetry_atom_block_matrix_map_t symmetrize_symmetry_ibz_kspace_operator_blocks(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& layouts,
    const Vector3_Order<double>& k_ibz,
    const symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs = nullptr);

ComplexMatrix rotate_symmetry_kspace_matrix(const SymmetryContext& ctx,
                                          const std::vector<SpeciesBasisLayout>& layouts,
                                          const SymmetryKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const bool use_time_reversal = false,
                                          const Vector3_Order<double>* k_bz_target = nullptr);

symmetry_irreducible_sector_t build_symmetry_rspace_irreducible_sector(
    const SymmetryContext& ctx,
    const std::vector<Vector3_Order<int>>& Rlist);

void build_symmetry_rspace_sector_stars(
    const SymmetryContext& ctx,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& Rlist,
    symmetry_rspace_sector_stars_t& sector_stars,
    std::ostream* log = nullptr);

/*!
 * @brief Rotate one dense atom-pair block between symmetry-related real-space sectors.
 *
 * `matrix_source` contains all basis rows on `atom_from_i` and all basis columns
 * on `atom_from_j`. Shell RSH rotations are assembled into one atom-level basis
 * rotation for each side before applying the block transform.
 */
ComplexMatrix rotate_symmetry_rspace_block(const SymmetryContext& ctx,
                                           const std::vector<SpeciesBasisLayout>& layouts_i,
                                           const std::vector<SpeciesBasisLayout>& layouts_j,
                                           const int isym,
                                           const atom_t atom_from_i,
                                           const atom_t atom_from_j,
                                           const ComplexMatrix& matrix_source);

ComplexMatrix rotate_symmetry_rspace_block(const SymmetryContext& ctx,
                                           const std::vector<SpeciesBasisLayout>& layouts,
                                           const int isym,
                                           const atom_t atom_from_i,
                                           const atom_t atom_from_j,
                                           const ComplexMatrix& matrix_source);

} // namespace librpa_int
