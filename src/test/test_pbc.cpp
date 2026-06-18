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
        librpa_int::TWO_PI * 0.5, 0.0, 0.0,
        0.0, librpa_int::TWO_PI * 0.5, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_ibz_mapping({0, 0, 2}, {}, {0.25, 0.25, 0.5});

    assert(pbc.klist.size() == 3);
    assert(pbc.klist_full.size() == 3);
    assert(pbc.klist_coul.size() == 2);
    assert(std::abs(pbc.kweight_ibz[0] - 0.5) < 1e-12);
    assert(std::abs(pbc.kweight_ibz[1] - 0.5) < 1e-12);
    assert(pbc.map_irk_ks.at(pbc.klist_coul[0]).size() == 2);
    assert(pbc.map_irk_ks.at(pbc.klist_coul[1]).size() == 1);
}

static void test_reduced_scf_kgrids()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        0.0, librpa_int::TWO_PI * 0.5, 0.0,
    };
    pbc.set_kgrids_kvec(2, 2, 1, kvecs);
    pbc.set_ibz_mapping({0, 1}, {}, {0.5, 0.5});

    assert(pbc.klist.size() == 2);
    assert(pbc.klist_full.size() == 2);
    assert(pbc.klist_coul.size() == 2);
    assert(std::abs(pbc.kweight_ibz[0] - 0.5) < 1e-12);
    assert(std::abs(pbc.kweight_ibz[1] - 0.5) < 1e-12);
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
    test_reduced_scf_kgrids();
    test_atom_pair_bvk_remap();
    return 0;
}
