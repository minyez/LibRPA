#include "stru_symops_parser.h"

#include "../src/utils/error.h"

#include <algorithm>
#include <cctype>
#include <exception>

namespace
{

std::string lowercase_token(std::string token)
{
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return token;
}

bool is_stru_symop_convention(const std::string &token)
{
    const auto convention = lowercase_token(token);
    return convention == "row" || convention == "col";
}

int parse_stru_int_token(const std::string &token, const std::string &context)
{
    try
    {
        std::size_t used = 0;
        const int value = std::stoi(token, &used);
        if (used == token.size())
        {
            return value;
        }
    }
    catch (const std::exception &)
    {
    }
    throw LIBRPA_RUNTIME_ERROR("Invalid integer in " + context + ": " + token);
}

double parse_stru_double_token(const std::string &token, const std::string &context)
{
    try
    {
        std::size_t used = 0;
        const double value = std::stod(token, &used);
        if (used == token.size())
        {
            return value;
        }
    }
    catch (const std::exception &)
    {
    }
    throw LIBRPA_RUNTIME_ERROR("Invalid floating-point value in " + context + ": " + token);
}

void require_stru_tail_tokens(const std::vector<std::string> &tokens,
                              const std::size_t pos,
                              const std::size_t count,
                              const std::string &context)
{
    if (pos > tokens.size() || tokens.size() - pos < count)
    {
        throw LIBRPA_RUNTIME_ERROR("Unexpected end of stru_out while reading " + context);
    }
}

bool is_stru_symop_header_at(const std::vector<std::string> &tokens, const std::size_t pos)
{
    return pos + 1 < tokens.size() && is_stru_symop_convention(tokens[pos + 1]);
}

bool is_stru_tail_boundary_at(const std::vector<std::string> &tokens, const std::size_t pos)
{
    return pos == tokens.size() || (pos < tokens.size() && is_stru_symop_header_at(tokens, pos));
}

std::size_t skip_legacy_stru_kpoint_section(const std::vector<std::string> &tokens,
                                            std::size_t pos,
                                            int n_kpoints,
                                            const std::string &file_path)
{
    require_stru_tail_tokens(tokens, pos, 3, "legacy k-point grid");
    const int nk0 = parse_stru_int_token(tokens[pos], file_path);
    const int nk1 = parse_stru_int_token(tokens[pos + 1], file_path);
    const int nk2 = parse_stru_int_token(tokens[pos + 2], file_path);
    pos += 3;
    if (nk0 <= 0 || nk1 <= 0 || nk2 <= 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid legacy k-point grid in " + file_path);
    }

    const int nk_full = nk0 * nk1 * nk2;
    if (n_kpoints > 0 && n_kpoints <= nk_full)
    {
        const auto after_ibz_rows = pos + static_cast<std::size_t>(3 * n_kpoints);
        if (is_stru_tail_boundary_at(tokens, after_ibz_rows))
        {
            return after_ibz_rows;
        }
        const auto after_ibz_mapping = after_ibz_rows + static_cast<std::size_t>(nk_full);
        if (is_stru_tail_boundary_at(tokens, after_ibz_mapping))
        {
            return after_ibz_mapping;
        }
    }

    require_stru_tail_tokens(tokens, pos, static_cast<std::size_t>(3 * nk_full),
                             "legacy k-point rows");
    pos += static_cast<std::size_t>(3 * nk_full);
    if (is_stru_tail_boundary_at(tokens, pos))
    {
        return pos;
    }

    require_stru_tail_tokens(tokens, pos, static_cast<std::size_t>(nk_full),
                             "legacy k-point mapping");
    return pos + static_cast<std::size_t>(nk_full);
}

std::size_t read_stru_symops_from_tokens(const std::vector<std::string> &tokens,
                                         std::size_t pos,
                                         const std::string &file_path,
                                         driver::StruSymopsTail &tail)
{
    require_stru_tail_tokens(tokens, pos, 2, "symmetry operation header");
    tail.n_symops = parse_stru_int_token(tokens[pos], file_path);
    if (tail.n_symops < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid number of symmetry operations in " + file_path);
    }
    const auto convention = lowercase_token(tokens[pos + 1]);
    if (!is_stru_symop_convention(convention))
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid symmetry operation convention in " + file_path
                                  + ": " + tokens[pos + 1]);
    }
    tail.row_conv = convention == "row" ? 1 : 0;
    pos += 2;

    tail.rotmats.clear();
    tail.trans.clear();
    tail.rotmats.reserve(static_cast<std::size_t>(9 * tail.n_symops));
    tail.trans.reserve(static_cast<std::size_t>(3 * tail.n_symops));
    for (int isym = 0; isym != tail.n_symops; ++isym)
    {
        require_stru_tail_tokens(tokens, pos, 12, "symmetry operation");
        for (int i = 0; i != 9; ++i)
        {
            tail.rotmats.push_back(parse_stru_int_token(tokens[pos + static_cast<std::size_t>(i)],
                                                        file_path));
        }
        pos += 9;
        for (int i = 0; i != 3; ++i)
        {
            tail.trans.push_back(parse_stru_double_token(tokens[pos + static_cast<std::size_t>(i)],
                                                         file_path));
        }
        pos += 3;
    }
    return pos;
}

std::size_t read_stru_spin_block_from_tokens(const std::vector<std::string> &tokens,
                                             std::size_t pos,
                                             const std::string &file_path,
                                             driver::StruSymopsTail &tail)
{
    require_stru_tail_tokens(tokens, pos, 3, "spin symmetry header");
    if (lowercase_token(tokens[pos]) != "spin_symmetry")
    {
        throw LIBRPA_RUNTIME_ERROR("Unexpected data after symmetry operations in " + file_path
                                  + ": " + tokens[pos]);
    }
    tail.grey_group = parse_stru_int_token(tokens[pos + 1], file_path);
    tail.spin_source = parse_stru_int_token(tokens[pos + 2], file_path);
    if (tail.spin_source < 0 || tail.spin_source > 2)
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid spin action source in " + file_path
                                  + ": " + tokens[pos + 2]);
    }
    pos += 3;

    tail.antiunitary.clear();
    tail.spin_u.clear();
    tail.antiunitary.reserve(static_cast<std::size_t>(tail.n_symops));
    const int n_spin_u = tail.spin_source == 1 ? 8 : 0;
    tail.spin_u.reserve(static_cast<std::size_t>(n_spin_u * tail.n_symops));
    for (int isym = 0; isym != tail.n_symops; ++isym)
    {
        require_stru_tail_tokens(tokens, pos, static_cast<std::size_t>(1 + n_spin_u),
                                 "spin symmetry operation");
        tail.antiunitary.push_back(parse_stru_int_token(tokens[pos], file_path));
        pos += 1;
        for (int i = 0; i != n_spin_u; ++i)
        {
            tail.spin_u.push_back(parse_stru_double_token(tokens[pos + static_cast<std::size_t>(i)],
                                                          file_path));
        }
        pos += static_cast<std::size_t>(n_spin_u);
    }
    tail.has_spin_block = true;
    return pos;
}

} // namespace

namespace driver
{

bool parse_stru_symops_tail(const std::vector<std::string> &tokens,
                            const std::string &file_path,
                            int n_kpoints,
                            StruSymopsTail &tail)
{
    tail = StruSymopsTail{};
    if (tokens.empty())
    {
        return false;
    }

    std::size_t pos = 0;
    if (tokens.size() < 2 || !is_stru_symop_convention(tokens[1]))
    {
        pos = skip_legacy_stru_kpoint_section(tokens, pos, n_kpoints, file_path);
    }
    if (pos == tokens.size())
    {
        return false;
    }
    if (pos + 1 >= tokens.size() || !is_stru_symop_convention(tokens[pos + 1]))
    {
        throw LIBRPA_RUNTIME_ERROR("Unexpected trailing data in " + file_path);
    }

    pos = read_stru_symops_from_tokens(tokens, pos, file_path, tail);
    if (pos != tokens.size())
    {
        pos = read_stru_spin_block_from_tokens(tokens, pos, file_path, tail);
    }
    if (pos != tokens.size())
    {
        throw LIBRPA_RUNTIME_ERROR("Unexpected data after symmetry operations in " + file_path);
    }
    return true;
}

} // namespace driver
