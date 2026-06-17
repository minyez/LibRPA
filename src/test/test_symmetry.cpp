#include "../math/symmetry.h"

#include <cassert>

using namespace librpa_int;

static void test_atom_to_inequivalent_symmetry_mapping()
{
    const std::vector<Vector3_Order<double>> atom_positions{
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.25, 0.25, 0.0},
        {0.75, 0.75, 0.0},
    };

    SpaceGroupSymOps<SpaceGroupSymOp> operations;
    operations.push_back(SpaceGroupSymOp{0, Matrix3(), {0.0, 0.0, 0.0}});
    operations.push_back(SpaceGroupSymOp{1, Matrix3(), {0.5, 0.5, 0.0}});

    const auto mappings =
        build_atom_to_inequivalent_symmetry_mapping(atom_positions, operations);
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

int main()
{
    test_atom_to_inequivalent_symmetry_mapping();
    return 0;
}
