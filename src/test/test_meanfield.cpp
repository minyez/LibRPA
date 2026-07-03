#include "../core/meanfield.h"
#include "../core/symmetry_context.h"
#include <cassert>
#include <array>
#include <map>
#include <stdexcept>
#include <utility>

#include "testutils.h"

void test_BCC_He_gamma_minimal_basis_aims()
{
    using namespace librpa_int;

    const int nk = 1;
    MeanField mf(1, 1, 8, 8);
    mf.get_efermi() = 0.240386888648512;
    mf.get_weight()[0].zero_out();
    mf.get_weight()[0](0, 0) = mf.get_weight()[0](0, 1) = 2.0 / nk;
    std::vector<double> eig {
         -0.649240864,
         -0.577333356,
          0.783349882,
          0.783349885,
          0.783349885,
          1.130014638,
          1.130014642,
          1.130014642,
    };
    std::vector<complex<double>> wkc_gamma_T
    {
       0.635395184437372,  0.747426907197624, -0.000000000000000,  0.000000000000000,  0.000000000000000, -0.000000000000000,  0.000000000000000, -0.000000000000000,
       0.000000000000000,  0.000000000000000,  0.000000046406191, -0.379780880349053, -0.566034156121301, -0.000000021962747, -0.074956542623863,  0.783355936987987,
       0.000000000000000, -0.000000000000000,  0.000000013452817,  0.566034156121300, -0.379780880349051, -0.000000008499556,  0.783355936987989,  0.074956542623864,
       0.000000000000000,  0.000000000000000, -0.681636400858004, -0.000000014684412, -0.000000046031303, -0.786933928164525, -0.000000006368929, -0.000000022672482,
       0.635395184437320, -0.747426907197668, -0.000000000000000, -0.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000001,
      -0.000000000000000,  0.000000000000000, -0.000000046406191,  0.379780880349045,  0.566034156121291, -0.000000021962747, -0.074956542623865,  0.783355936987997,
      -0.000000000000000,  0.000000000000000, -0.000000013452817, -0.566034156121295,  0.379780880349048, -0.000000008499557,  0.783355936987995,  0.074956542623863,
       0.000000000000000, -0.000000000000000,  0.681636400857993,  0.000000014684413,  0.000000046031304, -0.786933928164535, -0.000000006368930, -0.000000022672483,
    };
    mf.get_eigenvectors()[0][0][0].create(8, 8);
    for (int ib = 0; ib < 8; ib++)
        mf.get_eigenvals()[0](0, ib) = eig[ib];
    for (int iw = 0; iw < 8; iw++)
        for (int ib = 0; ib < 8; ib++)
            mf.get_eigenvectors()[0][0][0](ib, iw) = wkc_gamma_T[iw * 8 + ib];

    // test density matrix
    const auto dmat_gamma = mf.get_dmat_cplx(0, 0, 0, 0);
    const complex<double> thres = 1e-10;
    assert(fequal(dmat_gamma(0, 0), { 0.962374022009208e-00, 0}, thres));
    assert(fequal(dmat_gamma(4, 0), {-1.549199411968699e-01, 0}, thres));
    assert(fequal(dmat_gamma(0, 4), {-1.549199411968699e-01, 0}, thres));
    assert(fequal(dmat_gamma(4, 4), { 0.962374022009208e+00, 0}, thres));

    // test Green's function G(i \tau). approaches to (minus) density matrix for \tau -> 0^-
    const auto gf_gamma = mf.get_gf_cplx_imagtime(0, 0, 0, 0, -1e-12);
    assert(fequal(gf_gamma(0, 0), {-0.962374022009208e-00, 0}, thres));
    assert(fequal(gf_gamma(4, 0), { 1.549199411968699e-01, 0}, thres));
    assert(fequal(gf_gamma(0, 4), { 1.549199411968699e-01, 0}, thres));
    assert(fequal(gf_gamma(4, 4), {-0.962374022009208e+00, 0}, thres));
}

void test_state_index_energy_bounds()
{
    using namespace librpa_int;

    MeanField mf(1, 2, 4, 4);
    const std::vector<std::vector<double>> eig {
        {-3.0, -1.0, 0.5, 2.0},
        {-2.0, -0.5, 1.5, 3.0},
    };
    for (int ik = 0; ik != 2; ++ik)
    {
        for (int ist = 0; ist != 4; ++ist)
        {
            mf.get_eigenvals()[0](ik, ist) = eig[ik][ist];
        }
    }

    assert(mf.get_max_state_below_energy(-3.5) == -1);
    assert(mf.get_max_state_below_energy(-1.5) == 0);
    assert(mf.get_max_state_below_energy(0.75) == 1);
    assert(mf.get_max_state_below_energy(4.0) == 3);

    assert(mf.get_min_state_above_energy(4.0) == 4);
    assert(mf.get_min_state_above_energy(1.0) == 3);
    assert(mf.get_min_state_above_energy(-0.75) == 2);
    assert(mf.get_min_state_above_energy(-4.0) == 0);
}

void test_find_highest_occupied_state()
{
    using namespace librpa_int;

    MeanField mf(1, 2, 4, 1);
    mf.get_eigenvals()[0](0, 0) = -4.0;
    mf.get_eigenvals()[0](0, 1) = -1.0;
    mf.get_eigenvals()[0](0, 2) = 0.0;
    mf.get_eigenvals()[0](0, 3) = 1.0;
    mf.get_eigenvals()[0](1, 0) = -3.0;
    mf.get_eigenvals()[0](1, 1) = -0.5;
    mf.get_eigenvals()[0](1, 2) = 0.5;
    mf.get_eigenvals()[0](1, 3) = 2.0;
    mf.get_weight()[0].zero_out();
    mf.get_weight()[0](0, 1) = 1.0;
    mf.get_weight()[0](1, 0) = 1.0;

    assert(mf.find_highest_occupied_state(0) == std::make_pair(0, 1));
    assert(mf.find_highest_occupied_state(0, 1) == std::make_pair(1, 0));

    mf.get_weight()[0].zero_out();
    assert(mf.find_highest_occupied_state(0) == std::make_pair(-1, -1));

    mf.get_weight()[0](1, 2) = 0.75e-8;
    assert(mf.find_highest_occupied_state(0) == std::make_pair(1, 2));
}

void test_dmat_cplx_Rs_matches_single_R_accumulation()
{
    using namespace librpa_int;

    const int nk = 2;
    MeanField mf(1, nk, 1, 1);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](1, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0 / nk;
    mf.get_weight()[0](1, 0) = 2.0 / nk;
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][1].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};
    mf.get_eigenvectors()[0][0][1](0, 0) = {1.0, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> Rs {
        {0, 0, 0},
        {1, 0, 0},
    };

    const auto dmat_Rs = mf.get_dmat_cplx_Rs(0, 0, 0, kfrac_list, Rs);
    if (dmat_Rs.size() != Rs.size())
        throw std::runtime_error("get_dmat_cplx_Rs returned an unexpected number of R blocks");
    for (const auto &R : Rs)
    {
        const auto dmat_R = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
        if (!fequal(dmat_Rs.at(R)(0, 0), dmat_R(0, 0), {1e-12, 0.0}))
            throw std::runtime_error("get_dmat_cplx_Rs differs from get_dmat_cplx_R");
    }
}

void test_symmetry_context_kstar_restored_dmat_uses_full_star_phases()
{
    using namespace librpa_int;

    librpa_int::SymmetryContext ctx;
    ctx.set_available();
    ctx.add_basis_layouts("WFC", {{"X", {0}}});
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    librpa_int::SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    librpa_int::SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    librpa_int::SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 1);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const Vector3_Order<int> R{1, 0, 0};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    const std::map<atom_t, std::array<double, 3>> coord_frac{{0, {0.0, 0.0, 0.0}}};

    const auto direct_ibz = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
    const auto restored = get_symmetry_restored_dmat_cplx_R(
        ctx, mf, 0, 0, 0, kfrac_list, R, atom_nw, coord_frac);

    if (std::abs(direct_ibz(0, 0)) < 1e-12)
        throw std::runtime_error("direct IBZ density matrix unexpectedly vanished");
    if (std::abs(restored(0, 0)) > 1e-12)
        throw std::runtime_error("ABACUS k-star restored density matrix did not use full-star phases");
}

void test_symmetry_context_kstar_restore_skips_full_grid()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    ctx.add_basis_layouts("WFC", {{"X", {0}}});
    SymmetryKStar star;
    star.members.resize(2);
    ctx.kstars.push_back(star);

    MeanField mf(1, 2, 1, 1);
    const std::vector<Vector3_Order<double>> kfrac_list{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };

    assert(!can_restore_symmetry_kstar_meanfield(
        ctx, mf, kfrac_list, {{0, 1}}, {{0, {0.0, 0.0, 0.0}}}));
}

void test_symmetry_context_full_grid_kstar_route_matches_direct_full_k()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    ctx.add_basis_layouts("WFC", {{"X", {0}}});
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    ctx.kstars.push_back(star);

    MeanField mf(1, 2, 2, 1);
    mf.get_efermi() = 0.0;
    for (int ik = 0; ik != 2; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
        mf.get_weight()[0](ik, 0) = 1.0;
        mf.get_weight()[0](ik, 1) = 0.0;
        mf.get_eigenvectors()[0][0][ik].create(2, 1);
        mf.get_eigenvectors()[0][0][ik](0, 0) = {1.0, 0.0};
        mf.get_eigenvectors()[0][0][ik](1, 0) = {1.0, 0.0};
    }

    const std::vector<Vector3_Order<double>> kfrac_list{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> Rs{
        {0, 0, 0},
        {1, 0, 0},
    };
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    const std::map<atom_t, std::array<double, 3>> coord_frac{{0, {0.0, 0.0, 0.0}}};
    const auto representative_indices =
        build_symmetry_full_grid_kstar_representative_indices(
            ctx, kfrac_list);
    if (representative_indices.size() != 1 || representative_indices[0] != 0)
        throw std::runtime_error("full-grid k-star representative lookup failed");
    const auto member_kfrac_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, kfrac_list);

    for (const auto &R : Rs)
    {
        const auto direct = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
        const auto restored = get_symmetry_restored_dmat_cplx_R(
            ctx, mf, 0, 0, 0, kfrac_list, R, atom_nw, coord_frac,
            &member_kfrac_targets, &representative_indices);
        if (!fequal(direct(0, 0), restored(0, 0), {1e-12, 0.0}))
            throw std::runtime_error("full-grid k-star density matrix route differs from direct full-k");
    }

    const std::vector<double> taus{-1e-12, 1e-12};
    const auto direct_gf = mf.get_gf_cplx_imagtimes_Rs(0, 0, 0, kfrac_list, taus, Rs);
    const auto restored_gf = get_symmetry_restored_gf_cplx_imagtimes_Rs(
        ctx, mf, 0, 0, 0, kfrac_list, taus, Rs, atom_nw, coord_frac, -1,
        &member_kfrac_targets, &representative_indices);
    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            if (!fequal(direct_gf.at(tau).at(R)(0, 0),
                        restored_gf.at(tau).at(R)(0, 0), {1e-12, 0.0}))
            {
                throw std::runtime_error(
                    "full-grid k-star Green's-function route differs from direct full-k");
            }
        }
    }
}

void test_symmetry_context_kstar_restored_dmat_uses_target_kpoint_gauge()
{
    using namespace librpa_int;

    librpa_int::SymmetryContext ctx;
    ctx.set_available();
    ctx.ao_lmax = 0;
    ctx.atom_to_type[0] = 0;
    ctx.atom_to_type[1] = 0;
    ctx.add_basis_layouts("WFC", {{"X", {0}}});

    librpa_int::SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    auto make_atom_rotation = [](const atom_t atom) {
        librpa_int::SymmetryKAtomRotation atom_rotation;
        atom_rotation.atom_from = static_cast<int>(atom);
        atom_rotation.atom_to = static_cast<int>(atom);
        atom_rotation.atom_type = 0;
        atom_rotation.lmax = 0;
        atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
        atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};
        return atom_rotation;
    };

    librpa_int::SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_atom_rotation(0));
    star.members[0].atom_rotations.push_back(make_atom_rotation(1));
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(make_atom_rotation(0));
    star.members[1].atom_rotations.push_back(make_atom_rotation(1));
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_eigenvectors()[0][0][0].create(1, 2);
    mf.get_eigenvectors()[0][0][0](0, 0) = {std::sqrt(0.5), 0.0};
    mf.get_eigenvectors()[0][0][0](0, 1) = {std::sqrt(0.5), 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const Vector3_Order<int> R{0, 0, 0};
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const std::map<atom_t, std::array<double, 3>> coord_frac{
        {0, {0.0, 0.0, 0.0}},
        {1, {0.25, 0.0, 0.0}},
    };
    const std::vector<std::vector<Vector3_Order<double>>> target_kfrac_list{
        {{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}}};

    const auto restored = get_symmetry_restored_dmat_cplx_R(
        ctx, mf, 0, 0, 0, kfrac_list, R, atom_nw, coord_frac, &target_kfrac_list);

    const std::complex<double> expected_offdiag{0.25, -0.25};
    if (std::abs(restored(0, 1) - expected_offdiag) > 1e-12)
        throw std::runtime_error("ABACUS k-star restored density matrix ignored target k-point gauge");
}

int main(int argc, char *argv[])
{
    test_BCC_He_gamma_minimal_basis_aims();
    test_state_index_energy_bounds();
    test_find_highest_occupied_state();
    test_dmat_cplx_Rs_matches_single_R_accumulation();
    test_symmetry_context_kstar_restored_dmat_uses_full_star_phases();
    test_symmetry_context_kstar_restore_skips_full_grid();
    test_symmetry_context_full_grid_kstar_route_matches_direct_full_k();
    test_symmetry_context_kstar_restored_dmat_uses_target_kpoint_gauge();
    return 0;
}
