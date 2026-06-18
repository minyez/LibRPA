#include "../math/symmetry.h"

#include "testutils.h"
#include <cassert>

using namespace librpa_int;

static void test_rotation()
{
    Vector3_Order<double> a1{0.3, 0.2, 0.1};
    const Matrix3 rotation(0.0, -1.0, 0.0,
                           1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0);
    SpaceGroupSymOp op1{rotation, {0.0, 0.0, 0.0}};
    auto a2 = apply_space_group_symmetry_operation(op1, a1);
    assert(fequal(a2.x, 0.2));
    assert(fequal(a2.y, -0.3));
    assert(fequal(a2.z, 0.1));
}

static void test_composition()
{
    const Matrix3 lhs_rotation(0.0, -1.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0);
    const Matrix3 rhs_rotation(1.0, 0.0, 0.0,
                               0.0, 0.0, -1.0,
                               0.0, 1.0, 0.0);
    const SpaceGroupSymOp lhs{lhs_rotation, {0.1, 0.2, 0.3}};
    const SpaceGroupSymOp rhs{rhs_rotation, {0.4, 0.5, 0.6}};
    const auto composed = compose_space_group_symmetry_operations(lhs, rhs);

    const Vector3_Order<double> coord{0.7, 0.8, 0.9};
    const auto sequential =
        apply_space_group_symmetry_operation(rhs,
                                             apply_space_group_symmetry_operation(lhs, coord));
    const auto direct = apply_space_group_symmetry_operation(composed, coord);

    assert(fequal(direct.x, sequential.x));
    assert(fequal(direct.y, sequential.y));
    assert(fequal(direct.z, sequential.z));
}

static void test_column_convention()
{
    const Matrix3 lhs_rotation(0.0, -1.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0);
    const Matrix3 rhs_rotation(1.0, 0.0, 0.0,
                               0.0, 0.0, -1.0,
                               0.0, 1.0, 0.0);
    const SpaceGroupSymOp lhs{lhs_rotation, {0.1, 0.2, 0.3}, false};
    const SpaceGroupSymOp rhs{rhs_rotation, {0.4, 0.5, 0.6}, false};

    const Vector3_Order<double> coord{0.7, 0.8, 0.9};
    const auto applied = apply_space_group_symmetry_operation(lhs, coord);
    assert(fequal(applied.x, -0.7));
    assert(fequal(applied.y, 0.9));
    assert(fequal(applied.z, 1.2));

    const auto composed = compose_space_group_symmetry_operations(lhs, rhs);
    const auto sequential =
        apply_space_group_symmetry_operation(rhs,
                                             apply_space_group_symmetry_operation(lhs, coord));
    const auto direct = apply_space_group_symmetry_operation(composed, coord);
    assert(fequal(direct.x, sequential.x));
    assert(fequal(direct.y, sequential.y));
    assert(fequal(direct.z, sequential.z));
}

static void test_row_fractional_rotation_to_cartesian()
{
    const Matrix3 lattice_vectors(2.0, 0.0, 0.0,
                                  0.5, 1.5, 0.0,
                                  0.2, 0.3, 2.0);
    const Matrix3 rotation(0.0, -1.0, 0.0,
                           1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0);
    const Vector3_Order<double> frac{0.2, 0.3, 0.4};

    const auto cartesian_rotation =
        row_fractional_rotation_to_cartesian(rotation, lattice_vectors);
    const auto cart = Vector3_Order<double>(lattice_vectors.Transpose() * frac);
    const auto rotated_frac = multiply_row_vector(frac, rotation);
    const auto rotated_cart = Vector3_Order<double>(lattice_vectors.Transpose() * rotated_frac);
    const auto direct_cart = Vector3_Order<double>(cartesian_rotation * cart);

    assert(fequal(direct_cart.x, rotated_cart.x));
    assert(fequal(direct_cart.y, rotated_cart.y));
    assert(fequal(direct_cart.z, rotated_cart.z));
}

static void test_atom_to_inequivalent_symmetry_mapping()
{
    const Matrix3 lattice_vectors;
    const std::vector<Vector3_Order<double>> atom_positions{
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.25, 0.25, 0.0},
        {0.75, 0.75, 0.0},
    };

    SpaceGroupSymOps<SpaceGroupSymOp> operations;
    operations.push_back(SpaceGroupSymOp{Matrix3(), {0.0, 0.0, 0.0}});
    operations.push_back(SpaceGroupSymOp{Matrix3(), {0.5, 0.5, 0.0}});

    const auto mappings = build_atom_to_inequivalent_symmetry_mapping(
        atom_positions, lattice_vectors, operations);
    const auto inequivalent_atoms = collect_inequivalent_atoms(mappings);

    assert(inequivalent_atoms.size() == 2);
    assert(inequivalent_atoms[0] == 0);
    assert(inequivalent_atoms[1] == 2);

    assert(mappings[0].inequivalent_atom == 0);
    assert(mappings[0].isym == 0);
    assert(mappings[0].return_lattice == Vector3_Order<int>(0, 0, 0));

    assert(mappings[1].inequivalent_atom == 0);
    assert(mappings[1].isym == 1);
    assert(mappings[1].return_lattice == Vector3_Order<int>(1, 1, 0));

    assert(mappings[2].inequivalent_atom == 2);
    assert(mappings[2].isym == 0);

    assert(mappings[3].inequivalent_atom == 2);
    assert(mappings[3].isym == 1);
    assert(mappings[3].return_lattice == Vector3_Order<int>(1, 1, 0));
}

static void test_symmetry_mapping_keeps_operations_fractional()
{
    const Matrix3 lattice_vectors(2.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.2, 0.3, 2.0);
    const Matrix3 rotation(0.0, -1.0, 0.0,
                           1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0);
    const Vector3_Order<double> source_frac{0.2, 0.3, 0.4};
    const Vector3_Order<double> translation_frac{0.25, 0.1, 0.0};
    const Vector3_Order<double> target = restrict_fractional_coordinate(
        multiply_row_vector(source_frac, rotation) + translation_frac, 1e-10);

    const std::vector<Vector3_Order<double>> atom_positions{target, source_frac};
    SpaceGroupSymOps<SpaceGroupSymOp> fractional_operations;
    fractional_operations.push_back(SpaceGroupSymOp{Matrix3(), {0.0, 0.0, 0.0}});
    fractional_operations.push_back(SpaceGroupSymOp{rotation, translation_frac});

    const auto mappings = build_atom_to_inequivalent_symmetry_mapping(
        atom_positions, lattice_vectors, fractional_operations, 1e-10);
    const auto inequivalent_atoms = collect_inequivalent_atoms(mappings);

    assert(inequivalent_atoms.size() == 1);
    assert(inequivalent_atoms[0] == 0);
    assert(mappings[1].inequivalent_atom == 0);
    assert(mappings[1].isym == 1);
    assert(mappings[1].return_lattice == Vector3_Order<int>(0, -1, 0));
}

static void test_kpoint_rotation_and_target_fold()
{
    const Matrix3 rotation(1.0, 1.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0);
    const SpaceGroupSymOp op{rotation, {0.0, 0.0, 0.0}};
    const Vector3_Order<double> rpoint{0.2, 0.3, 0.0};
    const Vector3_Order<double> kpoint{0.4, 0.6, 0.0};

    const auto rotated_r = apply_space_group_symmetry_operation(op, rpoint);
    const auto rotated_k = apply_space_group_rotation_to_kpoint(op, kpoint);
    assert(fequal(kpoint * rpoint, rotated_k * rotated_r));
    assert(fequal(rotated_k.x, -0.2));
    assert(fequal(rotated_k.y, 0.6));
    assert(fequal(rotated_k.z, 0.0));

    const std::vector<Vector3_Order<double>> target_kpoints{
        {1.8, 0.6, 0.0},
        {0.4, 0.6, 0.0},
    };
    const auto target_folded = fold_fractional_kpoint_to_targets(rotated_k, target_kpoints);
    assert(target_folded.target_k_index == 0);
    assert(target_folded.kpoint == Vector3_Order<double>(1.8, 0.6, 0.0));
    assert(target_folded.fold_G == Vector3_Order<int>(-2, 0, 0));
}

static void test_kpoint_stars_from_full_grid()
{
    const Matrix3 identity;
    // Clockwise C4 rotations; check x' = x R
    const Matrix3 rot90(0.0, -1.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0);
    const Matrix3 rot180(-1.0, 0.0, 0.0,
                         0.0, -1.0, 0.0,
                         0.0, 0.0, 1.0);
    const Matrix3 rot270(0.0, 1.0, 0.0,
                         -1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0);
    SpaceGroupSymOps<SpaceGroupSymOp> operations;
    operations.push_back(SpaceGroupSymOp{identity, {0.0, 0.0, 0.0}});
    operations.push_back(SpaceGroupSymOp{rot90, {0.0, 0.0, 0.0}});
    operations.push_back(SpaceGroupSymOp{rot180, {0.0, 0.0, 0.0}});
    operations.push_back(SpaceGroupSymOp{rot270, {0.0, 0.0, 0.0}});

    const std::vector<Vector3_Order<double>> full_kpoints{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, -0.5, 0.0},
        {0.5, 0.5, 0.0},
    };

    const auto rotate_k = apply_space_group_rotation_to_kpoint(operations[1], full_kpoints[1]);
    assert(fequal(rotate_k.x, 0.0));
    assert(fequal(rotate_k.y, -0.5));
    assert(fequal(rotate_k.z, 0.0));

    const auto stars = build_kpoint_stars(full_kpoints, operations);
    assert(stars.size() == 3);
    assert(stars[0].members.size() == 1);
    assert(stars[0].sym_mappings.size() == stars[0].members.size());
    assert(stars[0].representative_k_index == 0);
    assert(stars[0].members[stars[0].representative_k_index].full_k_index == 0);
    assert(stars[1].members.size() == 2);
    assert(stars[1].sym_mappings.size() == stars[1].members.size());
    assert(stars[1].representative_k_index == 0);
    assert(stars[1].members[stars[1].representative_k_index].full_k_index == 1);
    assert(stars[1].members[0].full_k_index == 1);
    assert(stars[1].sym_mappings[0].isym == 0);
    assert(stars[1].sym_mappings[0].fold_G == Vector3_Order<int>(0, 0, 0));
    assert(stars[1].members[1].full_k_index == 2);
    assert(stars[1].members[1].kpoint == Vector3_Order<double>(0.0, -0.5, 0.0));
    assert(stars[1].sym_mappings[1].isym == 1);
    assert(stars[1].sym_mappings[1].fold_G == Vector3_Order<int>(0, 0, 0));
    assert(stars[2].members.size() == 1);
    assert(stars[2].sym_mappings.size() == stars[2].members.size());
    assert(stars[2].representative_k_index == 0);
    assert(stars[2].members[stars[2].representative_k_index].full_k_index == 3);

    const std::vector<Vector3_Order<double>> preferred_representatives{{0.0, -0.5, 0.0}};
    const auto hinted_stars = build_kpoint_stars(full_kpoints, operations, preferred_representatives);
    assert(hinted_stars[1].members.size() == 2);
    assert(hinted_stars[1].representative_k_index == 1);
    assert(hinted_stars[1].members[hinted_stars[1].representative_k_index].full_k_index == 2);
    assert(hinted_stars[1].sym_mappings[0].isym == 1);
    assert(hinted_stars[1].sym_mappings[0].fold_G == Vector3_Order<int>(-1, 0, 0));
    assert(hinted_stars[1].sym_mappings[1].isym == 0);
    assert(hinted_stars[1].sym_mappings[1].fold_G == Vector3_Order<int>(0, 0, 0));
}

int main()
{
    test_rotation();
    test_composition();
    test_column_convention();
    test_row_fractional_rotation_to_cartesian();
    test_atom_to_inequivalent_symmetry_mapping();
    test_symmetry_mapping_keeps_operations_fractional();
    test_kpoint_rotation_and_target_fold();
    test_kpoint_stars_from_full_grid();
    return 0;
}
