#include "../core/pbc.h"
#include <cassert>
#include <cstddef>

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
    pbc.set_latvec_and_G({1, 2, 3, 4, 5, 6, 7, 8, 9},
                         {1, 2, 3, 4, 5, 6, 7, 8, 9});
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
    const Vector3_Order<int> period{2, 2, 2};
    const Matrix3 latvec;

    const AtomPairBvKRemap<atom_t, atpair_t> remap(coord_fracs, Rs, period, latvec);
    const atpair_t pair00{0, 0};
    const atpair_t pair01{0, 1};
    const atpair_t pair10{1, 0};
    const Vector3_Order<int> R0{0, 0, 0};
    const Vector3_Order<int> R1{1, 0, 0};
    const Vector3_Order<int> Rm1{-1, 0, 0};

    assert(remap.size() == 4);
    assert(remap.at(pair00).empty());
    assert(remap.at(pair01).size() == 1);
    assert(remap.at(pair01).count(R0) == 0);
    const auto *R_bvk = remap.find_R_bvk(pair01, R1);
    assert(R_bvk != nullptr);
    assert(*R_bvk == Rm1);
    assert(remap.find_R_bvk(pair01, R0) == nullptr);
    assert(remap.at(pair10).empty());
    assert(remap.find_R_bvk(pair10, R1) == nullptr);
    assert(remap.find_R_bvk({2, 0}, R1) == nullptr);
}

int main (int argc, char *argv[])
{
    test_is_gamma_point();
    test_get_R_index();
    test_periodic_boundary_data();
    test_atom_pair_bvk_remap();
    return 0;
}
