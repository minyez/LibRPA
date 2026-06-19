#include "reader_structure.h"

#include "driver.h"

#include "../src/api/instance_manager.h"
#include "../src/core/input_symmetry.h"
#include "../src/io/global_io.h"
#include "../src/utils/error.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <utility>
#include <vector>

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

std::size_t skip_legacy_stru_kpoint_section(const std::vector<std::string> &tokens,
                                            std::size_t pos,
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
    if (driver::n_kpoints > 0 && driver::n_kpoints <= nk_full)
    {
        const auto after_ibz_rows = pos + static_cast<std::size_t>(3 * driver::n_kpoints);
        if (is_stru_symop_header_at(tokens, after_ibz_rows))
        {
            return after_ibz_rows;
        }
        const auto after_ibz_mapping = after_ibz_rows + static_cast<std::size_t>(nk_full);
        if (is_stru_symop_header_at(tokens, after_ibz_mapping))
        {
            return after_ibz_mapping;
        }
    }

    require_stru_tail_tokens(tokens, pos, static_cast<std::size_t>(3 * nk_full),
                             "legacy k-point rows");
    pos += static_cast<std::size_t>(3 * nk_full);
    if (is_stru_symop_header_at(tokens, pos))
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
                                         std::vector<librpa_int::InputSymmetryOperation> &symops)
{
    require_stru_tail_tokens(tokens, pos, 2, "symmetry operation header");
    const int n_symop = parse_stru_int_token(tokens[pos], file_path);
    if (n_symop < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid number of symmetry operations in " + file_path);
    }
    const auto convention = lowercase_token(tokens[pos + 1]);
    if (!is_stru_symop_convention(convention))
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid symmetry operation convention in " + file_path
                                  + ": " + tokens[pos + 1]);
    }
    pos += 2;

    symops.clear();
    symops.reserve(static_cast<std::size_t>(n_symop));
    for (int isym = 0; isym != n_symop; ++isym)
    {
        require_stru_tail_tokens(tokens, pos, 12, "symmetry operation");
        int rotation[9];
        for (int i = 0; i != 9; ++i)
        {
            rotation[i] = parse_stru_int_token(tokens[pos + static_cast<std::size_t>(i)], file_path);
        }
        pos += 9;

        librpa_int::InputSymmetryOperation op;
        op.rotation = librpa_int::Matrix3(rotation[0], rotation[1], rotation[2],
                                          rotation[3], rotation[4], rotation[5],
                                          rotation[6], rotation[7], rotation[8]);
        op.translation = {std::stod(tokens[pos]),
                          std::stod(tokens[pos + 1]),
                          std::stod(tokens[pos + 2])};
        op.use_row_convention = convention == "row";
        pos += 3;
        symops.push_back(std::move(op));
    }
    return pos;
}

void read_stru_tail_symops(std::ifstream &infile, const std::string &file_path)
{
    std::vector<std::string> tokens;
    std::string token;
    while (infile >> token)
    {
        tokens.push_back(token);
    }
    if (tokens.empty())
    {
        return;
    }

    std::size_t pos = 0;
    if (tokens.size() < 2 || !is_stru_symop_convention(tokens[1]))
    {
        pos = skip_legacy_stru_kpoint_section(tokens, pos, file_path);
    }
    if (pos == tokens.size())
    {
        return;
    }
    if (pos + 1 >= tokens.size() || !is_stru_symop_convention(tokens[pos + 1]))
    {
        throw LIBRPA_RUNTIME_ERROR("Unexpected trailing data in " + file_path);
    }

    std::vector<librpa_int::InputSymmetryOperation> stru_symops;
    pos = read_stru_symops_from_tokens(tokens, pos, file_path, stru_symops);
    if (pos != tokens.size())
    {
        throw LIBRPA_RUNTIME_ERROR("Unexpected data after symmetry operations in " + file_path);
    }

    auto pds = librpa_int::api::get_dataset_instance(driver::h);
    auto &operations = pds->symmetry_context.rspace_operations;
    if (!operations.empty() && operations.size() != stru_symops.size())
    {
        throw LIBRPA_RUNTIME_ERROR("stru_out symmetry operation count conflicts with existing symmetry context");
    }
    if (operations.size() == stru_symops.size())
    {
        for (std::size_t isym = 0; isym != operations.size(); ++isym)
        {
            stru_symops[isym].shell_rotations = operations[isym].shell_rotations;
        }
    }
    pds->symmetry_context.set_rspace_operations(std::move(stru_symops));
}

} // namespace

void reader_structure(const std::string &file_path)
{
    using namespace librpa_int;
    global::lib_printf_root("Reading structure file: %s\n", file_path.c_str());

    std::ifstream infile(file_path);
    if (!infile.good())
        throw LIBRPA_RUNTIME_ERROR("Fail to open structure file " + file_path);
    std::string x, y, z;

    std::vector<double> lat_mat(9);
    std::vector<double> G_mat(9);

    for (int i = 0; i < 3; i++)
    {
        infile >> x >> y >> z;
        lat_mat[i * 3] = stod(x);
        lat_mat[i * 3 + 1] = stod(y);
        lat_mat[i * 3 + 2] = stod(z);
    }

    for (int i = 0; i < 3; i++)
    {
        infile >> x >> y >> z;
        G_mat[i * 3] = stod(x);
        G_mat[i * 3 + 1] = stod(y);
        G_mat[i * 3 + 2] = stod(z);
    }

    driver::h.set_latvec_and_G(lat_mat.data(), G_mat.data());

    infile >> driver::n_atoms;
    const auto n_atoms = driver::n_atoms;
    driver::atom_types.resize(n_atoms);
    std::vector<double> coords(n_atoms * 3);
    int type;
    for (size_t iat = 0; iat < n_atoms; iat++)
    {
        for (int i = 0; i < 3; i++) infile >> coords[3 * iat + i];
        infile >> type;
        driver::atom_types[iat] = type - 1;
    }
    driver::h.set_atoms(driver::atom_types, coords);
    {
        auto pds = api::get_dataset_instance(driver::h);
        pds->symmetry_context.set_lattice(pds->pbc.latvec, pds->pbc.G);
    }
    read_stru_tail_symops(infile, file_path);
}
