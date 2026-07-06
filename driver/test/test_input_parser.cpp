#include "../input_parser.h"

#include <cassert>
#include <string>

void test_key_prefixes()
{
    InputParser parser(
        "tfgrids_type = minimax\n"
        "anacon_tfgrids_type = GL\n"
        "fn_basis = basis_out\n"
        "fn_basis_wfc = basis_wfc_out\n");

    int flag = 1;
    std::string value;

    parser.parse_string("tfgrids_type", value, flag);
    assert(flag == 0);
    assert(value == "minimax");

    parser.parse_string("anacon_tfgrids_type", value, flag);
    assert(flag == 0);
    assert(value == "GL");

    parser.parse_string("fn_basis", value, flag);
    assert(flag == 0);
    assert(value == "basis_out");
}

int main()
{
    test_key_prefixes();
}
