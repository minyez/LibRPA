#include "../stru_symops_parser.h"

#include <cassert>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<std::string> tokenize(const std::string &text)
{
    std::vector<std::string> tokens;
    std::string token;
    for (const char ch : text)
    {
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            if (!token.empty())
            {
                tokens.push_back(token);
                token.clear();
            }
        }
        else
        {
            token.push_back(ch);
        }
    }
    if (!token.empty())
    {
        tokens.push_back(token);
    }
    return tokens;
}

bool throws_runtime(const std::vector<std::string> &tokens, int n_kpoints = 0)
{
    driver::StruSymopsTail tail;
    try
    {
        driver::parse_stru_symops_tail(tokens, "test_stru", n_kpoints, tail);
    }
    catch (const std::runtime_error &)
    {
        return true;
    }
    return false;
}

void test_empty_tail()
{
    driver::StruSymopsTail tail;
    assert(!driver::parse_stru_symops_tail({}, "test_stru", 0, tail));
}

void test_spatial_only()
{
    // Identity + inversion, row convention, no spin block (legacy format).
    const auto tokens = tokenize(
        "2 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "-1 0 0 0 -1 0 0 0 -1 0.5 0.5 0.5");
    driver::StruSymopsTail tail;
    assert(driver::parse_stru_symops_tail(tokens, "test_stru", 0, tail));
    assert(!tail.has_spin_block);
    assert(tail.n_symops == 2);
    assert(tail.row_conv == 1);
    assert(tail.rotmats.size() == 18);
    assert(tail.rotmats[0] == 1 && tail.rotmats[9] == -1);
    assert(tail.trans.size() == 6);
    assert(std::abs(tail.trans[3] - 0.5) < 1e-15);
}

void test_spin_block_derived_soc()
{
    // Magnetic-group style: two spatial ops, spin block with one antiunitary.
    const auto tokens = tokenize(
        "2 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "0 -1 0 1 0 0 0 0 1 0.0 0.0 0.0 "
        "spin_symmetry 0 2 "
        "0 "
        "1");
    driver::StruSymopsTail tail;
    assert(driver::parse_stru_symops_tail(tokens, "test_stru", 0, tail));
    assert(tail.has_spin_block);
    assert(tail.grey_group == 0);
    assert(tail.spin_source == 2);
    assert(tail.antiunitary.size() == 2);
    assert(tail.antiunitary[0] == 0 && tail.antiunitary[1] == 1);
    assert(tail.spin_u.empty());
}

void test_spin_block_explicit()
{
    // ExplicitSpinSpace: 8 doubles of U_s per operation.
    const auto tokens = tokenize(
        "2 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "1 0 0 0 1 0 0 0 1 0.25 0.0 0.0 "
        "spin_symmetry 1 1 "
        "0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 "
        "1 0.0 0.0 0.0 -1.0 0.0 1.0 0.0 0.0");
    driver::StruSymopsTail tail;
    assert(driver::parse_stru_symops_tail(tokens, "test_stru", 0, tail));
    assert(tail.has_spin_block);
    assert(tail.grey_group == 1);
    assert(tail.spin_source == 1);
    assert(tail.antiunitary.size() == 2);
    assert(tail.spin_u.size() == 16);
    assert(std::abs(tail.spin_u[0] - 1.0) < 1e-15);
    assert(std::abs(tail.spin_u[6] - 1.0) < 1e-15);
    assert(std::abs(tail.spin_u[11] + 1.0) < 1e-15);
    assert(std::abs(tail.spin_u[13] - 1.0) < 1e-15);
    assert(std::abs(tail.trans[3] - 0.25) < 1e-15);
}

void test_garbage_after_symops_throws()
{
    const auto tokens = tokenize(
        "1 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "nonsense 1 2 3");
    assert(throws_runtime(tokens));
}

void test_garbage_after_spin_block_throws()
{
    const auto tokens = tokenize(
        "1 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "spin_symmetry 0 2 0 "
        "extra");
    assert(throws_runtime(tokens));
}

void test_bad_source_throws()
{
    const auto tokens = tokenize(
        "1 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "spin_symmetry 0 5 0");
    assert(throws_runtime(tokens));
}

void test_truncated_spin_block_throws()
{
    // Two operations declared, only one spin row given.
    const auto tokens = tokenize(
        "2 row "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 "
        "spin_symmetry 0 2 "
        "0");
    assert(throws_runtime(tokens));
}

void test_trailing_data_without_header_throws()
{
    const auto tokens = tokenize("17 23");
    assert(throws_runtime(tokens));
}

void test_legacy_kpoint_section_then_spin_tail()
{
    // 2x2x2 grid + 8 k-rows + 8 mapping ints, then symops with spin block.
    std::string text = "2 2 2 ";
    for (int i = 0; i != 8; ++i)
    {
        text += "0.0 0.0 0.0 ";
    }
    for (int i = 0; i != 8; ++i)
    {
        text += "1 ";
    }
    text += "1 row 1 0 0 0 1 0 0 0 1 0.0 0.0 0.0 spin_symmetry 1 2 0";
    const auto tokens = tokenize(text);
    driver::StruSymopsTail tail;
    assert(driver::parse_stru_symops_tail(tokens, "test_stru", 0, tail));
    assert(tail.has_spin_block);
    assert(tail.n_symops == 1);
    assert(tail.grey_group == 1);
    assert(tail.spin_source == 2);
}

void test_legacy_kpoint_section_ibz_rows_only()
{
    // 2x2x2 grid with only the loaded IBZ rows (n_kpoints = 2), then symops.
    const auto tokens = tokenize(
        "2 2 2 "
        "0.0 0.0 0.0 0.5 0.0 0.0 "
        "1 row 1 0 0 0 1 0 0 0 1 0.0 0.0 0.0");
    driver::StruSymopsTail tail;
    assert(driver::parse_stru_symops_tail(tokens, "test_stru", 2, tail));
    assert(tail.n_symops == 1);
    assert(!tail.has_spin_block);
}

} // namespace

int main()
{
    test_empty_tail();
    test_spatial_only();
    test_spin_block_derived_soc();
    test_spin_block_explicit();
    test_garbage_after_symops_throws();
    test_garbage_after_spin_block_throws();
    test_bad_source_throws();
    test_truncated_spin_block_throws();
    test_trailing_data_without_header_throws();
    test_legacy_kpoint_section_then_spin_tail();
    test_legacy_kpoint_section_ibz_rows_only();
}
