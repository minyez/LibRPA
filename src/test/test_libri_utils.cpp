#include <array>
#include <cassert>
#include <set>
#include <utility>
#include <vector>

#include "../math/vector3_order.h"
#include "../utils/libri_utils.h"

void test_comm_map2_targets_from_vector()
{
    using namespace librpa_int;

    const std::vector<std::pair<int, int>> atpairs{{0, 0}, {0, 1}, {2, 3}};
    const std::vector<Vector3_Order<int>> cells{{0, 0, 0}, {1, -1, 2}};
    const auto targets = get_s0_s1_for_comm_map2<int, int>(atpairs, cells);

    assert(targets.first.size() == 2);
    assert(targets.first.count(0) == 1);
    assert(targets.first.count(2) == 1);

    assert(targets.second.size() == 6);
    assert(targets.second.count({0, {0, 0, 0}}) == 1);
    assert(targets.second.count({0, {1, -1, 2}}) == 1);
    assert(targets.second.count({1, {0, 0, 0}}) == 1);
    assert(targets.second.count({1, {1, -1, 2}}) == 1);
    assert(targets.second.count({3, {0, 0, 0}}) == 1);
    assert(targets.second.count({3, {1, -1, 2}}) == 1);
}

void test_comm_map2_targets_from_set()
{
    using namespace librpa_int;

    const std::set<std::pair<int, int>> atpairs{{0, 0}, {0, 1}, {2, 3}};
    const std::vector<Vector3_Order<int>> cells{{0, 0, 0}, {1, -1, 2}};
    const auto targets = get_s0_s1_for_comm_map2<int, int>(atpairs, cells);

    const std::set<int> expected_s0{0, 2};
    const std::set<libri_types<int, int>::TAC> expected_s1{
        {0, {0, 0, 0}}, {0, {1, -1, 2}},
        {1, {0, 0, 0}}, {1, {1, -1, 2}},
        {3, {0, 0, 0}}, {3, {1, -1, 2}},
    };

    assert(targets.first == expected_s0);
    assert(targets.second == expected_s1);
}

void test_comm_map2_targets_output_types()
{
    using namespace librpa_int;

    const std::vector<std::pair<int, int>> atpairs{{1, 2}};
    const std::vector<Vector3_Order<int>> cells{{3, 4, 5}};
    const auto targets = get_s0_s1_for_comm_map2<int, int, long, long>(atpairs, cells);

    const std::set<long> expected_s0{1L};
    const std::set<libri_types<long, long>::TAC> expected_s1{{2L, {3L, 4L, 5L}}};
    assert(targets.first == expected_s0);
    assert(targets.second == expected_s1);
}

void test_comm_map2_targets_empty_cells()
{
    using namespace librpa_int;

    const std::vector<std::pair<int, int>> atpairs{{0, 1}, {2, 3}};
    const std::vector<Vector3_Order<int>> cells;
    const auto targets = get_s0_s1_for_comm_map2<int, int>(atpairs, cells);

    const std::set<int> expected_s0{0, 2};
    assert(targets.first == expected_s0);
    assert(targets.second.empty());
}

int main()
{
    test_comm_map2_targets_from_vector();
    test_comm_map2_targets_from_set();
    test_comm_map2_targets_output_types();
    test_comm_map2_targets_empty_cells();

    return 0;
}
