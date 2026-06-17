#include "../math/mathtools.h"
#include "../math/vector3_order.h"

#include <cassert>
#include <type_traits>

using namespace librpa_int;

int main()
{
    assert(nearly_integer(1.0 + 5e-9, 1e-8));
    assert(!nearly_integer(1.0 + 2e-8, 1e-8));

    assert(nearly_integer_vector(Vector3<double>{1.0 + 5e-9, -2.0 - 5e-9, 0.0}, 1e-8));
    assert(!nearly_integer_vector(Vector3<double>{1.0, 2.0 + 2e-8, 0.0}, 1e-8));

    assert(round_to_integer_vector(Vector3<double>{0.49, 1.51, -1.51}) == Vector3<int>(0, 2, -2));

    static_assert(std::is_same_v<decltype(restrict_fractional_coordinate(Vector3<double>{})),
                                 Vector3<double>>);
    assert(restrict_fractional_coordinate(Vector3<double>{-0.25, 1.0, 1.25})
           == Vector3<double>(0.75, 0.0, 0.25));

    static_assert(std::is_same_v<decltype(restrict_fractional_coordinate(Vector3_Order<double>{})),
                                 Vector3_Order<double>>);
    assert(restrict_fractional_coordinate(Vector3_Order<double>{-0.25, 1.0, 1.25})
           == Vector3_Order<double>(0.75, 0.0, 0.25));

    return 0;
}
