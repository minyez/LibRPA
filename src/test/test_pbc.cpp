#include "../core/pbc.h"
#include "../utils/constants.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cmath>

using namespace librpa_int;

static void test_is_gamma_point()
{
    assert(is_gamma_point(Vector3_Order<double>{0.0, 0.0, 0.0}));
    assert(!is_gamma_point(Vector3_Order<double>{0.3, 0.2, 0.1}));
    assert(is_gamma_point(Vector3_Order<int>{0, 0, 0}));
}

static void test_get_R_index()
{
    Vector3_Order<int> period{2, 2, 2};
    std::vector<Vector3_Order<int>> sc222 = construct_R_grid(period);
    for (size_t i = 0; i != sc222.size(); i++ )
        printf("i=%zu: x %d, y %d, z %d\n", i, sc222[i].x, sc222[i].y, sc222[i].z);
    Vector3_Order<int> R {-1, -1, -1};
    printf(" R: x %d, y %d, z %d\n", R.x, R.y, R.z);
    auto mR = -R;
    printf("-R: x %d, y %d, z %d\n", mR.x, mR.y, mR.z);
    assert(get_R_index(sc222, R) == 0);
    assert(get_R_index(sc222, Vector3_Order<int>{-1, 0, -1}) == 2);
    assert(get_R_index(sc222, Vector3_Order<int>{3, 3, -1}) < 0);
    assert(get_R_index(sc222, Vector3_Order<int>{3, 3, -1} % period) == 0);
}

static void test_periodic_boundary_data()
{
    PeriodicBoundaryData pbc;
    const Vector3_Order<int> gamma_period{1, 1, 1};
    const std::array<int, 3> gamma_period_array{1, 1, 1};
    assert(pbc.period == gamma_period);
    assert(pbc.period_array == gamma_period_array);
    assert(pbc.Rlist == construct_R_grid(pbc.period));

    pbc.set_period(2, 3, 1);
    const Vector3_Order<int> period{2, 3, 1};
    const std::array<int, 3> period_array{2, 3, 1};
    assert(pbc.period == period);
    assert(pbc.period_array == period_array);
    assert(pbc.Rlist == construct_R_grid(pbc.period));

    pbc.set_latvec_and_G({1, 2, 3, 4, 5, 6, 7, 8, 9},
                         {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

static void test_kgrids_with_weighted_coulomb_mapping()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        librpa_int::TWO_PI * 2.0 / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 0, 2});

    assert(pbc.klist.size() == 3);
    assert(pbc.klist_full.size() == 3);
    assert(std::abs(pbc.weight_k[0] - 1.0 / 3.0) < 1e-12);
    assert(std::abs(pbc.weight_k[1] - 1.0 / 3.0) < 1e-12);
    assert(std::abs(pbc.weight_k[2] - 1.0 / 3.0) < 1e-12);
    assert(pbc.klist_coul.size() == 2);
    assert(std::abs(pbc.weight_q[0] - 2.0 / 3.0) < 1e-12);
    assert(std::abs(pbc.weight_q[1] - 1.0 / 3.0) < 1e-12);
    assert(pbc.map_irk_ks.at(pbc.klist_coul[0]).size() == 2);
    assert(pbc.map_irk_ks.at(pbc.klist_coul[1]).size() == 1);

    pbc.set_period(3, 1, 1);
    assert(pbc.klist.size() == 3);
    pbc.set_period(1, 1, 1);
    assert(pbc.klist.empty());
    assert(pbc.klist_full.empty());
    assert(pbc.klist_coul.empty());
}

static void test_full_scf_kgrids_keep_loaded_order()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI * 2.0 / 3.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);

    assert(pbc.klist_full == pbc.klist);
    assert(pbc.kfrac_list_full == pbc.kfrac_list);
    assert(pbc.k_to_kfull == std::vector<int>({0, 1, 2}));
    assert(pbc.kfull_to_k == std::vector<int>({0, 1, 2}));
    assert(pbc.kfull_to_k_relation == std::vector<KFullToKRelation>({
        KFullToKRelation::DIRECT,
        KFullToKRelation::DIRECT,
        KFullToKRelation::DIRECT,
    }));
    assert(!pbc.kgrid_uses_time_reversal);
}

static void test_reduced_scf_kgrids()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs, {1.0 / 3.0, 2.0 / 3.0});
    pbc.set_kq_mapping({0, 1});

    assert(pbc.klist.size() == 2);
    assert(pbc.klist_full.size() == 3);
    assert(pbc.k_to_kfull == std::vector<int>({0, 1}));
    assert(pbc.kfull_to_k == std::vector<int>({0, 1, 1}));
    assert(pbc.kfull_to_k_relation == std::vector<KFullToKRelation>({
        KFullToKRelation::DIRECT,
        KFullToKRelation::DIRECT,
        KFullToKRelation::TIME_REVERSAL,
    }));
    assert(pbc.kgrid_uses_time_reversal);
    assert(pbc.klist_coul.size() == 2);
    assert(std::abs(pbc.weight_q[0] - 1.0 / 3.0) < 1e-12);
    assert(std::abs(pbc.weight_q[1] - 2.0 / 3.0) < 1e-12);
}

static void test_incomplete_time_reversal_reduced_scf_kgrids()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI * 0.25, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(4, 1, 1, kvecs, {0.25, 0.75});

    assert(pbc.kfull_to_k == std::vector<int>({0, 1, -1, 1}));
    assert(pbc.weight_k == std::vector<double>({0.25, 0.75}));
    assert(pbc.kfull_to_k_relation == std::vector<KFullToKRelation>({
        KFullToKRelation::DIRECT,
        KFullToKRelation::DIRECT,
        KFullToKRelation::NONE,
        KFullToKRelation::TIME_REVERSAL,
    }));
    assert(!pbc.kgrid_uses_time_reversal);

    pbc.set_kgrids_kvec(4, 1, 1, kvecs, {1.0, 3.0});
    assert(std::abs(pbc.weight_k[0] - 0.25) < 1e-12);
    assert(std::abs(pbc.weight_k[1] - 0.75) < 1e-12);

    PeriodicBoundaryData pbc_without_weights;
    pbc_without_weights.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    pbc_without_weights.set_kgrids_kvec(4, 1, 1, kvecs);
    assert(pbc_without_weights.klist_coul.size() == 2);
    assert(pbc_without_weights.weight_k.empty());
    assert(pbc_without_weights.weight_q.empty());
    assert(pbc_without_weights.map_q_weight.empty());
}

static void test_irreducible_kgrids_from_symmetry_stars()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2, 0, 0, 0, 3, 0, 0, 0, 4});

    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    assert(pbc.klist.size() == 2);
    assert(pbc.klist_full.size() == 3);
    assert(std::abs(pbc.kfrac_list[1].x - 1.0 / 3.0) < 1e-12);
    assert(std::abs(pbc.weight_q[0] - 1.0 / 3.0) < 1e-12);
    assert(std::abs(pbc.weight_q[1] - 2.0 / 3.0) < 1e-12);
    assert(pbc.map_irk_ks.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(pbc.klist_full[1].x - 1.0 / 6.0) < 1e-12);
    assert(std::abs(pbc.klist_full[2].x + 1.0 / 6.0) < 1e-12);
}

static void test_atom_pair_bvk_remap()
{
    typedef std::size_t atom_t;
    typedef std::pair<atom_t, atom_t> atpair_t;

    const std::map<atom_t, Vector3<double>> coord_fracs{
        {0, {0.1, 0.0, 0.0}},
        {1, {0.9, 0.0, 0.0}},
    };
    const std::vector<Vector3_Order<int>> Rs{
        {0, 0, 0},
        {1, 0, 0},
    };
    const Vector3_Order<int> period{3, 3, 3};
    const Matrix3 latvec;

    const AtomPairBvKRemap<atom_t, atpair_t> remap(coord_fracs, Rs, period, latvec);
    const atpair_t pair00{0, 0};
    const atpair_t pair01{0, 1};
    const atpair_t pair10{1, 0};
    const Vector3_Order<int> R0{0, 0, 0};
    const Vector3_Order<int> R1{1, 0, 0};
    const Vector3_Order<int> Rm2{-2, 0, 0};

    const auto *R_bvk = remap.find_R_bvk(pair01, R1);
    assert(R_bvk != nullptr);
    assert(R_bvk->size() == 1);
    assert(R_bvk->front() == Rm2);
    assert(remap.find_R_bvk(pair00, R1) == nullptr);
    assert(remap.find_R_bvk(pair01, R0) == nullptr);
    assert(remap.find_R_bvk(pair10, R1) == nullptr);
    assert(remap.find_R_bvk({2, 0}, R1) == nullptr);

    const Vector3_Order<int> period_2{2, 2, 2};
    const Vector3_Order<int> Rm1{-1, 0, 0};
    const AtomPairBvKRemap<atom_t, atpair_t> remap_ws(coord_fracs, Rs, period_2, latvec, 1);
    const auto *R_bvks_ws = remap_ws.find_R_bvk(pair00, R1);
    assert(R_bvks_ws != nullptr);
    assert(R_bvks_ws->size() == 2);
    assert(std::find(R_bvks_ws->cbegin(), R_bvks_ws->cend(), R1) != R_bvks_ws->cend());
    assert(std::find(R_bvks_ws->cbegin(), R_bvks_ws->cend(), Rm1) != R_bvks_ws->cend());
    assert(remap_ws.find_R_bvk(pair00, R0) == nullptr);
}

int main (int argc, char *argv[])
{
    test_is_gamma_point();
    test_get_R_index();
    test_periodic_boundary_data();
    test_kgrids_with_weighted_coulomb_mapping();
    test_full_scf_kgrids_keep_loaded_order();
    test_reduced_scf_kgrids();
    test_incomplete_time_reversal_reduced_scf_kgrids();
    test_irreducible_kgrids_from_symmetry_stars();
    test_atom_pair_bvk_remap();
    return 0;
}
