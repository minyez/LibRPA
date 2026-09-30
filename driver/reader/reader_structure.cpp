#include "reader_structure.h"

#include "reader_context.h"
#include "../stru_symops_parser.h"

#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/utils/error.h"

#include <fstream>
#include <vector>

namespace librpa::reader
{

namespace
{

void read_stru_tail_symops(ReaderContext &ctx, std::ifstream &infile, const std::string &file_path)
{
    std::vector<std::string> tokens;
    std::string token;
    while (infile >> token)
    {
        tokens.push_back(token);
    }
    driver::StruSymopsTail tail;
    if (!driver::parse_stru_symops_tail(tokens, file_path, ctx.state.n_kpoints, tail)) return;
    if (tail.has_spin_block)
    {
        ctx.h.set_symmetry_spin_operations(
            tail.n_symops, tail.row_conv,
            tail.rotmats.empty() ? nullptr : tail.rotmats.data(),
            tail.trans.empty() ? nullptr : tail.trans.data(),
            tail.antiunitary.empty() ? nullptr : tail.antiunitary.data(),
            tail.spin_u.empty() ? nullptr : tail.spin_u.data(), tail.spin_source, tail.grey_group);
    }
    else
    {
        ctx.h.set_symmetry_operations(tail.n_symops, tail.row_conv,
                                      tail.rotmats.empty() ? nullptr : tail.rotmats.data(),
                                      tail.trans.empty() ? nullptr : tail.trans.data());
    }
}

} // namespace

void reader_structure(ReaderContext &ctx, const std::string &file_path)
{
    using namespace librpa_int;
    global::lib_printf_root("Reading structure file: %s\n", file_path.c_str());

    require_readable_file(file_path);
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

    ctx.h.set_latvec_and_G(lat_mat.data(), G_mat.data());

    infile >> ctx.state.n_atoms;
    const auto n_atoms = ctx.state.n_atoms;
    ctx.state.atom_types.resize(n_atoms);
    std::vector<double> coords(n_atoms * 3);
    int type;
    for (size_t iat = 0; iat < n_atoms; iat++)
    {
        for (int i = 0; i < 3; i++) infile >> coords[3 * iat + i];
        infile >> type;
        ctx.state.atom_types[iat] = type - 1;
    }
    ctx.h.set_atoms(ctx.state.atom_types, coords);
    read_stru_tail_symops(ctx, infile, file_path);
}

} // namespace librpa::reader
