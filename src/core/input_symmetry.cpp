/*!
 * @file input_symmetry.cpp
 * @brief Utilities for reading input symmetry sidecar files.
 */
#include "input_symmetry.h"

#include "pbc.h"
#include "../math/rsh.h"
#include "../utils/constants.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace librpa_int
{

namespace
{

constexpr double kInputSymmetryCoordTol = 1e-5;
// Real-space atom mapping is reconstructed from ABACUS text sidecars. When exact fractional
// coordinates are available from the input `STRU`, LibRPA uses them directly. The looser
// tolerance below remains as a fallback for text-derived coordinates and lattice-inversion noise.
constexpr double kInputSymmetryRSpaceAtomMapTol = 5e-5;

std::string trim(const std::string& text)
{
    const auto begin = std::find_if_not(text.begin(), text.end(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; });
    if (begin == text.end())
    {
        return "";
    }
    const auto end = std::find_if_not(text.rbegin(), text.rend(),
                                      [](unsigned char ch) { return std::isspace(ch) != 0; })
                         .base();
    return std::string(begin, end);
}

std::string strip_comment(const std::string& text)
{
    const auto comment_pos = text.find('#');
    return trim(text.substr(0, comment_pos));
}

bool starts_with(const std::string& text, const std::string& prefix)
{
    return text.rfind(prefix, 0) == 0;
}

std::string parent_path(const std::string& file_path)
{
    const auto pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return ".";
    }
    if (pos == 0)
    {
        return file_path.substr(0, 1);
    }
    return file_path.substr(0, pos);
}

std::string base_name(const std::string& file_path)
{
    const auto pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return file_path;
    }
    return file_path.substr(pos + 1);
}

bool is_absolute_path(const std::string& file_path)
{
    if (file_path.empty())
    {
        return false;
    }
    if (file_path.front() == '/' || file_path.front() == '\\')
    {
        return true;
    }
    return file_path.size() > 1 && std::isalpha(static_cast<unsigned char>(file_path[0])) != 0
           && file_path[1] == ':';
}

std::string join_path(const std::string& dir_path, const std::string& file_name)
{
    if (dir_path.empty())
    {
        return file_name;
    }
    if (dir_path.back() == '/' || dir_path.back() == '\\')
    {
        return dir_path + file_name;
    }
    return dir_path + "/" + file_name;
}

bool file_exists(const std::string& file_path)
{
    std::ifstream ifs(file_path);
    return ifs.good();
}

std::vector<std::string> build_input_symmetry_path_candidates(const std::string& dir_path)
{
    std::vector<std::string> dirs;
    auto append_unique = [&dirs](const std::string& dir) {
        if (!dir.empty() && std::find(dirs.begin(), dirs.end(), dir) == dirs.end())
        {
            dirs.push_back(dir);
        }
    };

    append_unique(dir_path);
    append_unique(join_path(dir_path, "OUT.ABACUS"));
    if (base_name(dir_path) == "OUT.ABACUS")
    {
        append_unique(parent_path(dir_path));
    }
    return dirs;
}

std::vector<std::string> split_fields(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream iss(line);
    std::string field;
    while (iss >> field)
    {
        fields.push_back(field);
    }
    return fields;
}

bool is_section_header(const std::string& line)
{
    static const std::set<std::string> section_headers{
        "ATOMIC_SPECIES",
        "NUMERICAL_ORBITAL",
        "ABFS_ORBITAL",
        "NUMERICAL_DESCRIPTOR",
        "LATTICE_CONSTANT",
        "LATTICE_PARAMETER",
        "LATTICE_VECTORS",
        "ATOMIC_POSITIONS",
        "NUMERICAL_DESCRIPTOR_VNA",
    };
    return section_headers.count(line) != 0;
}

std::vector<double> extract_doubles(const std::string& text)
{
    std::vector<double> values;
    const char* begin = text.c_str();
    char* end = nullptr;
    while (*begin != '\0')
    {
        const double value = std::strtod(begin, &end);
        if (end != begin)
        {
            values.push_back(value);
            begin = end;
        }
        else
        {
            ++begin;
        }
    }
    return values;
}

std::vector<long> extract_integers(const std::string& text)
{
    std::vector<long> values;
    const char* begin = text.c_str();
    char* end = nullptr;
    while (*begin != '\0')
    {
        const long value = std::strtol(begin, &end, 10);
        if (end != begin)
        {
            values.push_back(value);
            begin = end;
        }
        else
        {
            ++begin;
        }
    }
    return values;
}

bool is_integer_line(const std::string& text)
{
    const std::string stripped = trim(text);
    if (stripped.empty())
    {
        return false;
    }
    std::size_t start = (stripped.front() == '+' || stripped.front() == '-') ? 1 : 0;
    if (start == stripped.size())
    {
        return false;
    }
    return std::all_of(stripped.begin() + static_cast<std::ptrdiff_t>(start), stripped.end(),
                       [](unsigned char ch) { return std::isdigit(ch) != 0; });
}

bool starts_with_integer_token(const std::string& text)
{
    const std::string stripped = trim(text);
    if (stripped.empty())
    {
        return false;
    }

    std::size_t index = (stripped.front() == '+' || stripped.front() == '-') ? 1 : 0;
    if (index == stripped.size() || std::isdigit(static_cast<unsigned char>(stripped[index])) == 0)
    {
        return false;
    }
    while (index < stripped.size() && std::isdigit(static_cast<unsigned char>(stripped[index])) != 0)
    {
        ++index;
    }
    return index == stripped.size()
           || std::isspace(static_cast<unsigned char>(stripped[index])) != 0;
}

bool nearly_integer(const double value, const double tol = kInputSymmetryCoordTol)
{
    return std::abs(value - std::round(value)) < tol;
}

int shell_symbol_to_l(const char symbol)
{
    const std::string shells = "SPDFGHIJKLMNO";
    const auto pos = shells.find(static_cast<char>(std::toupper(static_cast<unsigned char>(symbol))));
    if (pos == std::string::npos)
    {
        return -1;
    }
    return static_cast<int>(pos);
}

int compute_nao_from_shell_counts(const std::vector<int>& shell_counts)
{
    int nao = 0;
    for (int l = 0; l < static_cast<int>(shell_counts.size()); ++l)
    {
        nao += shell_counts[l] * (2 * l + 1);
    }
    return nao;
}

std::vector<int> build_atom_offsets(const std::map<atom_t, size_t>& atom_nw)
{
    std::vector<int> offsets(atom_nw.size() + 1, 0);
    int running = 0;
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto iter = atom_nw.find(atom);
        if (iter == atom_nw.end())
        {
            throw std::runtime_error("Atomic orbital counts are not contiguous in atom_nw");
        }
        offsets[atom] = running;
        running += static_cast<int>(iter->second);
    }
    offsets.back() = running;
    return offsets;
}

Vector3_Order<double> restrict_fractional_coordinate(const Vector3_Order<double>& vec,
                                                     const double tol = kInputSymmetryCoordTol)
{
    auto wrap = [tol](const double x) {
        double wrapped = std::fmod(x + 100.0 + tol, 1.0) - tol;
        if (std::abs(wrapped) < tol)
        {
            wrapped = 0.0;
        }
        return wrapped;
    };
    return {wrap(vec.x), wrap(vec.y), wrap(vec.z)};
}

bool nearly_same_kpoint(const Vector3_Order<double>& lhs,
                        const Vector3_Order<double>& rhs,
                        const double tol = kInputSymmetryCoordTol)
{
    const auto is_same_component = [tol](const double lhs_component, const double rhs_component) {
        return std::abs((lhs_component - rhs_component) - std::round(lhs_component - rhs_component))
               < tol;
    };
    return is_same_component(lhs.x, rhs.x) && is_same_component(lhs.y, rhs.y)
           && is_same_component(lhs.z, rhs.z);
}

Matrix3 build_matrix3_from_array(const std::array<std::array<double, 3>, 3>& matrix)
{
    return Matrix3(matrix[0][0], matrix[0][1], matrix[0][2],
                   matrix[1][0], matrix[1][1], matrix[1][2],
                   matrix[2][0], matrix[2][1], matrix[2][2]);
}

ComplexMatrix build_input_symmetry_shell_rotation_from_direct_rotation(
    const InputSymmetryContext& ctx,
    const int l,
    const Matrix3& direct_rotation)
{
    const auto& basis_convention = ctx.basis_convention;
    if (l == 0)
    {
        return real_spherical_harmonic_rotation_matrix(Vector3<double>{0.0, 0.0, 0.0},
                                                       0,
                                                       basis_convention.order,
                                                       basis_convention.coeff_m_negative,
                                                       basis_convention.coeff_m_positive);
    }
    if (!ctx.lattice_available)
    {
        throw std::runtime_error(
            "ABACUS shell rotation fallback requires lattice vectors from the structure input");
    }

    const Matrix3 cartesian_matrix =
        ctx.lattice_vectors.Inverse() * direct_rotation * ctx.lattice_vectors;
    return real_spherical_harmonic_rotation_matrix(cartesian_matrix,
                                                   l,
                                                   basis_convention.order,
                                                   basis_convention.coeff_m_negative,
                                                   basis_convention.coeff_m_positive,
                                                   kInputSymmetryCoordTol);
}

bool is_identity_rotation(const Matrix3& matrix,
                          const double tol = 1e-8)
{
    return std::abs(matrix.e11 - 1.0) < tol && std::abs(matrix.e12) < tol
           && std::abs(matrix.e13) < tol && std::abs(matrix.e21) < tol
           && std::abs(matrix.e22 - 1.0) < tol && std::abs(matrix.e23) < tol
           && std::abs(matrix.e31) < tol && std::abs(matrix.e32) < tol
           && std::abs(matrix.e33 - 1.0) < tol;
}

Vector3_Order<int> round_vec3_to_int(const Vector3_Order<double>& vec)
{
    return {static_cast<int>(std::lround(vec.x)),
            static_cast<int>(std::lround(vec.y)),
            static_cast<int>(std::lround(vec.z))};
}

bool is_nearly_integer_vec3(const Vector3_Order<double>& vec,
                            const double tol = kInputSymmetryCoordTol)
{
    return nearly_integer(vec.x, tol) && nearly_integer(vec.y, tol) && nearly_integer(vec.z, tol);
}

ComplexMatrix extract_atom_block(const ComplexMatrix& matrix,
                                 const atom_t atom_i,
                                 const atom_t atom_j,
                                 const std::map<atom_t, size_t>& atom_nw,
                                 const std::vector<int>& offsets)
{
    const int ni = static_cast<int>(atom_nw.at(atom_i));
    const int nj = static_cast<int>(atom_nw.at(atom_j));
    ComplexMatrix block(ni, nj);
    const int row0 = offsets[atom_i];
    const int col0 = offsets[atom_j];
    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            block(i, j) = matrix(row0 + i, col0 + j);
        }
    }
    return block;
}

void set_atom_block(ComplexMatrix& matrix,
                    const atom_t atom_i,
                    const atom_t atom_j,
                    const ComplexMatrix& block,
                    const std::vector<int>& offsets)
{
    const int row0 = offsets[atom_i];
    const int col0 = offsets[atom_j];
    for (int i = 0; i < block.nr; ++i)
    {
        for (int j = 0; j < block.nc; ++j)
        {
            matrix(row0 + i, col0 + j) = block(i, j);
        }
    }
}

struct InputSymmetryRSpaceOperationInfo
{
    std::vector<atom_t> atom_map;
    std::vector<Vector3_Order<int>> return_lattice;
};

std::vector<InputSymmetryRSpaceOperationInfo> build_rspace_operation_info(
    const InputSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac)
{
    if (coord_frac.size() != ctx.atom_to_type.size())
    {
        throw std::runtime_error("Fractional coordinates and ABACUS atom mapping have inconsistent sizes");
    }

    std::vector<InputSymmetryRSpaceOperationInfo> infos(ctx.rspace_operations.size());
    for (auto& info : infos)
    {
        info.atom_map.resize(coord_frac.size(), static_cast<atom_t>(-1));
        info.return_lattice.resize(coord_frac.size(), {0, 0, 0});
    }

    for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
    {
        const auto& op = ctx.rspace_operations[isym];
        const double atom_map_tol = kInputSymmetryRSpaceAtomMapTol;
        for (atom_t atom_from = 0; atom_from < coord_frac.size(); ++atom_from)
        {
            const auto& coord_from = coord_frac.at(atom_from);
            const Vector3_Order<double> coord_from_vec =
                restrict_fractional_coordinate({coord_from[0], coord_from[1], coord_from[2]},
                                               atom_map_tol);
            // Keep the unwrapped rotated position so that the integer return lattice is preserved
            // exactly as in the ABACUS irreducible-sector construction.
            const Vector3_Order<double> transformed =
                multiply_row_vector(coord_from_vec, op.rotation)
                + restrict_fractional_coordinate(op.translation, atom_map_tol);

            atom_t matched_atom = static_cast<atom_t>(-1);
            Vector3_Order<int> matched_return{0, 0, 0};
            for (atom_t atom_to = 0; atom_to < coord_frac.size(); ++atom_to)
            {
                if (ctx.atom_to_type.at(atom_from) != ctx.atom_to_type.at(atom_to))
                {
                    continue;
                }
                const auto& coord_to = coord_frac.at(atom_to);
                const Vector3_Order<double> coord_to_vec =
                    restrict_fractional_coordinate({coord_to[0], coord_to[1], coord_to[2]},
                                                   atom_map_tol);
                const Vector3_Order<double> diff =
                    transformed - coord_to_vec;
                if (!is_nearly_integer_vec3(diff, atom_map_tol))
                {
                    continue;
                }
                if (matched_atom != static_cast<atom_t>(-1))
                {
                    throw std::runtime_error("ABACUS real-space symmetry atom mapping is ambiguous");
                }
                matched_atom = atom_to;
                matched_return = round_vec3_to_int(diff);
            }

            if (matched_atom == static_cast<atom_t>(-1))
            {
                throw std::runtime_error("Failed to match ABACUS real-space symmetry atom mapping");
            }

            infos[isym].atom_map[atom_from] = matched_atom;
            infos[isym].return_lattice[atom_from] = matched_return;
        }
    }
    return infos;
}

std::vector<int> build_rspace_inverse_map(
    const InputSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac)
{
    (void)coord_frac;
    std::vector<int> inverse_map(ctx.rspace_operations.size(), -1);
    for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
    {
        for (std::size_t jsym = 0; jsym < ctx.rspace_operations.size(); ++jsym)
        {
            const auto composed_rotation = multiply_space_group_rotation_matrices(
                ctx.rspace_operations[isym].rotation, ctx.rspace_operations[jsym].rotation);
            const auto composed_translation =
                multiply_row_vector(ctx.rspace_operations[isym].translation,
                                    ctx.rspace_operations[jsym].rotation)
                + ctx.rspace_operations[jsym].translation;
            const bool is_inverse =
                is_identity_rotation(composed_rotation) && is_nearly_integer_vec3(composed_translation);

            if (is_inverse)
            {
                inverse_map[isym] = static_cast<int>(jsym);
                break;
            }
        }
        if (inverse_map[isym] < 0)
        {
            throw std::runtime_error("Failed to build inverse symmetry-operation map for ABACUS sidecars");
        }
    }
    return inverse_map;
}

Vector3_Order<int> rotate_rspace_vector(
    const Vector3_Order<int>& R,
    const InputSymmetryRSpaceOperationInfo& op_info,
    const InputSymmetryOperation& op,
    const atom_t atom_from_i,
    const atom_t atom_from_j)
{
    const Vector3_Order<double> R_double{static_cast<double>(R.x), static_cast<double>(R.y),
                                         static_cast<double>(R.z)};
    const Vector3_Order<double> rotated_double =
        multiply_row_vector(R_double, op.rotation)
        + Vector3_Order<double>(static_cast<double>(op_info.return_lattice[atom_from_j].x),
                                static_cast<double>(op_info.return_lattice[atom_from_j].y),
                                static_cast<double>(op_info.return_lattice[atom_from_j].z))
        - Vector3_Order<double>(static_cast<double>(op_info.return_lattice[atom_from_i].x),
                                static_cast<double>(op_info.return_lattice[atom_from_i].y),
                                static_cast<double>(op_info.return_lattice[atom_from_i].z));
    if (!is_nearly_integer_vec3(rotated_double))
    {
        throw std::runtime_error("ABACUS real-space symmetry generated a non-integer lattice vector");
    }
    // Keep the raw rotated lattice vector returned by the ABACUS formula.
    // The caller is responsible for filtering against the explicit R list.
    return round_vec3_to_int(rotated_double);
}

Vector3_Order<double> parse_vec3_double(const std::string& line, const std::string& context)
{
    const auto values = extract_doubles(line);
    if (values.size() != 3)
    {
        throw std::runtime_error("Failed to parse 3-vector in " + context + ": " + line);
    }
    return {values[0], values[1], values[2]};
}

input_symmetry_R_t parse_vec3_int(const std::string& line, const std::string& context)
{
    const auto values = extract_integers(line);
    if (values.size() < 3)
    {
        throw std::runtime_error("Failed to parse integer 3-vector in " + context + ": " + line);
    }
    return {static_cast<int>(values[values.size() - 3]),
            static_cast<int>(values[values.size() - 2]),
            static_cast<int>(values[values.size() - 1])};
}

ComplexMatrix parse_complex_row_matrix(const std::string& line,
                                       const int expected_count,
                                       const std::string& context)
{
    const auto values = extract_doubles(line);
    ComplexMatrix row(1, expected_count);
    if (static_cast<int>(values.size()) == expected_count)
    {
        for (int i = 0; i < expected_count; ++i)
        {
            row(0, i) = std::complex<double>(values[i], 0.0);
        }
    }
    else if (static_cast<int>(values.size()) == 2 * expected_count)
    {
        for (int i = 0; i < expected_count; ++i)
        {
            row(0, i) = std::complex<double>(values[2 * i], values[2 * i + 1]);
        }
    }
    else
    {
        throw std::runtime_error("Failed to parse complex row in " + context + ": " + line);
    }
    return row;
}

ComplexMatrix parse_shell_rotation(const std::vector<std::string>& lines,
                                   std::size_t& index,
                                   const int nm,
                                   const std::string& context)
{
    ComplexMatrix mat(nm, nm);
    for (int row = 0; row < nm; ++row)
    {
        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            throw std::runtime_error("Unexpected end of file while reading " + context);
        }
        const auto parsed_row = parse_complex_row_matrix(lines[index], nm, context);
        for (int col = 0; col < nm; ++col)
        {
            mat(row, col) = parsed_row(0, col);
        }
        ++index;
    }
    return mat;
}

void load_irreducible_sector_file(const std::string& file_path,
                                  input_symmetry_irreducible_sector_t& irreducible_sector)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + file_path);
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        if (trim(line).empty())
        {
            continue;
        }
        const auto values = extract_integers(line);
        if (values.size() != 5)
        {
            throw std::runtime_error("Failed to parse irreducible-sector line: " + line);
        }
        const atpair_t atom_pair{static_cast<atom_t>(values[0]), static_cast<atom_t>(values[1])};
        const input_symmetry_R_t R{static_cast<int>(values[2]), static_cast<int>(values[3]),
                           static_cast<int>(values[4])};
        irreducible_sector[atom_pair].insert(R);
    }
}

void load_symrot_R_file(const std::string& file_path, InputSymmetryContext& ctx)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + file_path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        lines.push_back(line);
    }

    std::size_t index = 0;
    while (index < lines.size() && !starts_with(trim(lines[index]), "Lmax of AOs:"))
    {
        ++index;
    }
    if (index >= lines.size())
    {
        throw std::runtime_error("Missing AO Lmax header in " + file_path);
    }
    ctx.ao_lmax = static_cast<int>(extract_integers(lines[index]).front());
    ++index;

    while (index < lines.size() && !starts_with(trim(lines[index]), "Lmax of ABFs:"))
    {
        ++index;
    }
    if (index >= lines.size())
    {
        throw std::runtime_error("Missing ABF Lmax header in " + file_path);
    }
    ctx.abf_lmax = static_cast<int>(extract_integers(lines[index]).front());
    ++index;

    while (index < lines.size() && !is_integer_line(lines[index]))
    {
        ++index;
    }

    const int lmax = std::max(ctx.ao_lmax, ctx.abf_lmax);
    while (index < lines.size())
    {
        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            break;
        }
        if (!is_integer_line(lines[index]))
        {
            throw std::runtime_error("Expected symmetry index in " + file_path + ": " + lines[index]);
        }

        InputSymmetryOperation op;
        op.isym = std::stoi(trim(lines[index]));
        ++index;

        std::array<std::array<double, 3>, 3> rotation_rows{{{{0.0, 0.0, 0.0}},
                                                            {{0.0, 0.0, 0.0}},
                                                            {{0.0, 0.0, 0.0}}}};
        for (int row = 0; row < 3; ++row)
        {
            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index >= lines.size())
            {
                throw std::runtime_error("Unexpected end of file while reading rotation matrix");
            }
            const auto values = extract_doubles(lines[index]);
            if (values.size() != 3)
            {
                throw std::runtime_error("Failed to parse symmetry rotation matrix row: " + lines[index]);
            }
            for (int col = 0; col < 3; ++col)
            {
                rotation_rows[row][col] = values[col];
            }
            ++index;
        }
        op.rotation = build_matrix3_from_array(rotation_rows);

        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            throw std::runtime_error("Unexpected end of file while reading symmetry translation");
        }
        op.translation = parse_vec3_double(lines[index], file_path);
        ++index;

        for (int l = 0; l <= lmax; ++l)
        {
            const int nm = 2 * l + 1;
            op.shell_rotations[l] =
                parse_shell_rotation(lines, index, nm, "symrot_R l=" + std::to_string(l));
        }
        ctx.rspace_operations.push_back(std::move(op));
    }
}

bool append_unique_abf_layout(std::vector<InputSymmetryAOTypeLayout>& candidates,
                              const InputSymmetryAOTypeLayout& layout);

InputSymmetryAOTypeLayout parse_symrot_type_layout_line(const std::string& line,
                                                 int& atom_type,
                                                 const std::string& file_path,
                                                 const std::string& layout_kind)
{
    const auto fields = split_fields(strip_comment(line));
    if (fields.size() < 10 || fields[0] != "type" || fields[2] != "label" || fields[4] != "nao"
        || fields[6] != "lmax" || fields[8] != "shell_counts")
    {
        throw std::runtime_error("Failed to parse " + layout_kind + " shell-layout header line in "
                                 + file_path + ": " + line);
    }

    atom_type = std::stoi(fields[1]) - 1;
    if (atom_type < 0)
    {
        throw std::runtime_error(layout_kind
                                 + " shell-layout header uses an invalid atom type in "
                                 + file_path + ": " + line);
    }

    InputSymmetryAOTypeLayout layout;
    layout.label = fields[3];
    layout.nao = std::stoi(fields[5]);
    const int lmax = std::stoi(fields[7]);
    if (lmax < 0)
    {
        throw std::runtime_error(layout_kind + " shell-layout header uses a negative lmax in "
                                 + file_path + ": " + line);
    }

    layout.shell_counts.reserve(static_cast<std::size_t>(lmax + 1));
    for (std::size_t index = 9; index < fields.size(); ++index)
    {
        layout.shell_counts.push_back(std::stoi(fields[index]));
    }
    if (static_cast<int>(layout.shell_counts.size()) != lmax + 1)
    {
        throw std::runtime_error(layout_kind
                                 + " shell-layout header has inconsistent shell_counts in "
                                 + file_path + ": " + line);
    }
    if (compute_nao_from_shell_counts(layout.shell_counts) != layout.nao)
    {
        throw std::runtime_error(layout_kind + " shell-layout header has inconsistent nao in "
                                 + file_path + ": " + line);
    }

    return layout;
}

void parse_symrot_ao_layout_header(const std::vector<std::string>& lines,
                                   const std::string& file_path,
                                   std::vector<InputSymmetryAOTypeLayout>& layouts_by_type)
{
    layouts_by_type.clear();

    std::size_t index = 0;
    while (index < lines.size() && !starts_with(trim(lines[index]), "Star "))
    {
        const std::string cleaned = strip_comment(lines[index]);
        if (cleaned.empty())
        {
            ++index;
            continue;
        }

        if (!starts_with(cleaned, "AO shell layouts:"))
        {
            ++index;
            continue;
        }

        ++index;
        while (index < lines.size())
        {
            const std::string layout_line = strip_comment(lines[index]);
            if (layout_line.empty())
            {
                ++index;
                continue;
            }
            if (starts_with(layout_line, "End AO shell layouts"))
            {
                return;
            }

            int atom_type = -1;
            const auto layout =
                parse_symrot_type_layout_line(layout_line, atom_type, file_path, "AO");
            if (static_cast<int>(layouts_by_type.size()) <= atom_type)
            {
                layouts_by_type.resize(static_cast<std::size_t>(atom_type + 1));
            }
            layouts_by_type[static_cast<std::size_t>(atom_type)] = layout;
            ++index;
        }

        throw std::runtime_error("AO shell-layout header in " + file_path
                                 + " is missing `End AO shell layouts`");
    }
}

void parse_symrot_abf_layout_header(
    const std::vector<std::string>& lines,
    const std::string& file_path,
    std::vector<std::vector<InputSymmetryAOTypeLayout>>& candidates_by_type)
{
    candidates_by_type.clear();

    std::size_t index = 0;
    while (index < lines.size() && !starts_with(trim(lines[index]), "Star "))
    {
        const std::string cleaned = strip_comment(lines[index]);
        if (cleaned.empty())
        {
            ++index;
            continue;
        }

        if (!starts_with(cleaned, "ABF shell layouts:"))
        {
            ++index;
            continue;
        }

        ++index;
        while (index < lines.size())
        {
            const std::string layout_line = strip_comment(lines[index]);
            if (layout_line.empty())
            {
                ++index;
                continue;
            }
            if (starts_with(layout_line, "End ABF shell layouts"))
            {
                return;
            }

            int atom_type = -1;
            const auto layout =
                parse_symrot_type_layout_line(layout_line, atom_type, file_path, "ABF");
            if (static_cast<int>(candidates_by_type.size()) <= atom_type)
            {
                candidates_by_type.resize(static_cast<std::size_t>(atom_type + 1));
            }
            append_unique_abf_layout(candidates_by_type[static_cast<std::size_t>(atom_type)],
                                     layout);
            ++index;
        }

        throw std::runtime_error("ABF shell-layout header in " + file_path
                                 + " is missing `End ABF shell layouts`");
    }
}

void parse_symrot_k_file(const std::string& file_path,
                         std::vector<InputSymmetryKStar>& kstars,
                         std::map<std::pair<int, int>, Vector3_Order<int>>* kspace_return_lattice = nullptr,
                         std::map<std::pair<int, int>, Vector3_Order<int>>* kstar_member_fold_G = nullptr,
                         const int nsym_space = -1,
                         std::vector<InputSymmetryAOTypeLayout>* ao_layouts = nullptr,
                         std::vector<std::vector<InputSymmetryAOTypeLayout>>* abf_layout_candidates = nullptr)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + file_path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        lines.push_back(line);
    }

    if (ao_layouts != nullptr)
    {
        parse_symrot_ao_layout_header(lines, file_path, *ao_layouts);
    }
    if (abf_layout_candidates != nullptr)
    {
        parse_symrot_abf_layout_header(lines, file_path, *abf_layout_candidates);
    }

    std::size_t index = 0;
    while (index < lines.size() && !starts_with(trim(lines[index]), "Star "))
    {
        ++index;
    }

    while (index < lines.size())
    {
        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            break;
        }
        if (!starts_with(trim(lines[index]), "Star "))
        {
            throw std::runtime_error("Expected star header in " + file_path + ": " + lines[index]);
        }

        InputSymmetryKStar star;
        const auto star_numbers = extract_integers(lines[index]);
        if (star_numbers.empty())
        {
            throw std::runtime_error("Failed to parse star index in " + lines[index]);
        }
        star.star_index = static_cast<int>(star_numbers.front()) - 1;
        const auto left = lines[index].find('(');
        const auto right = lines[index].find(')', left == std::string::npos ? 0 : left);
        if (left == std::string::npos || right == std::string::npos)
        {
            throw std::runtime_error("Failed to parse IBZ k-vector in " + lines[index]);
        }
        star.k_ibz = parse_vec3_double(lines[index].substr(left, right - left + 1), file_path);
        ++index;

        while (index < lines.size())
        {
            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index >= lines.size() || starts_with(trim(lines[index]), "Star "))
            {
                break;
            }
            if (!starts_with_integer_token(lines[index]))
            {
                throw std::runtime_error("Expected symmetry index in " + file_path + ": " + lines[index]);
            }

            InputSymmetryKStarMember member;
            member.isym = std::stoi(trim(lines[index]));
            ++index;

            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index >= lines.size())
            {
                throw std::runtime_error("Unexpected end of file while reading k-star member");
            }
            member.k_bz = parse_vec3_double(lines[index], file_path);
            ++index;

            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index < lines.size() && starts_with(trim(lines[index]), "fold_G"))
            {
                const auto fold_G = parse_vec3_int(lines[index], file_path);
                if (kstar_member_fold_G != nullptr)
                {
                    (*kstar_member_fold_G)[{star.star_index,
                                            static_cast<int>(star.members.size())}] =
                        {fold_G[0], fold_G[1], fold_G[2]};
                }
                ++index;
            }

            while (index < lines.size())
            {
                while (index < lines.size() && trim(lines[index]).empty())
                {
                    ++index;
                }
                if (index >= lines.size() || starts_with(trim(lines[index]), "Star ")
                    || starts_with_integer_token(lines[index]))
                {
                    break;
                }
                if (!starts_with(trim(lines[index]), "atom "))
                {
                    throw std::runtime_error("Expected atom header in " + file_path + ": " + lines[index]);
                }

                InputSymmetryKAtomRotation atom_rotation;
                const auto values = extract_integers(lines[index]);
                if (values.size() < 4)
                {
                    throw std::runtime_error("Failed to parse atom symmetry header: " + lines[index]);
                }
                atom_rotation.atom_from = static_cast<int>(values[0]) - 1;
                atom_rotation.atom_to = static_cast<int>(values[1]) - 1;
                atom_rotation.atom_type = static_cast<int>(values[2]) - 1;
                atom_rotation.lmax = static_cast<int>(values[3]);
                ++index;

                while (index < lines.size() && trim(lines[index]).empty())
                {
                    ++index;
                }
                if (index < lines.size() && starts_with(trim(lines[index]), "return_lattice"))
                {
                    const auto return_lattice = parse_vec3_int(lines[index], file_path);
                    if (kspace_return_lattice != nullptr)
                    {
                        int spatial_isym = member.isym;
                        if (nsym_space > 0 && spatial_isym >= nsym_space)
                        {
                            spatial_isym -= nsym_space;
                        }
                        (*kspace_return_lattice)[{atom_rotation.atom_from, spatial_isym}] =
                            {return_lattice[0], return_lattice[1], return_lattice[2]};
                    }
                    ++index;
                }

                for (int l = 0; l <= atom_rotation.lmax; ++l)
                {
                    const int nm = 2 * l + 1;
                    atom_rotation.shell_rotations[l] =
                        parse_shell_rotation(lines, index, nm, "symrot_k l=" + std::to_string(l));
                }
                member.atom_rotations.push_back(std::move(atom_rotation));
            }
            star.members.push_back(std::move(member));
        }

        kstars.push_back(std::move(star));
    }
}

void infer_atom_to_type_from_kstars(const std::vector<InputSymmetryKStar>& kstars,
                                    std::map<atom_t, int>& atom_to_type)
{
    atom_to_type.clear();
    for (const auto& star : kstars)
    {
        for (const auto& member : star.members)
        {
            for (const auto& atom_rotation : member.atom_rotations)
            {
                for (const atom_t atom_index : {atom_rotation.atom_from, atom_rotation.atom_to})
                {
                    const auto inserted =
                        atom_to_type.emplace(atom_index, atom_rotation.atom_type);
                    if (!inserted.second && inserted.first->second != atom_rotation.atom_type)
                    {
                        throw std::runtime_error(
                            "ABACUS symrot_k.txt contains inconsistent atom-type metadata");
                    }
                }
            }
        }
    }
}

void load_symrot_k_file(const std::string& file_path, InputSymmetryContext& ctx)
{
    ctx.kstars.clear();
    ctx.ao_type_layouts.clear();
    ctx.kspace_return_lattice.clear();
    ctx.kstar_member_fold_G.clear();
    parse_symrot_k_file(file_path, ctx.kstars, &ctx.kspace_return_lattice,
                        &ctx.kstar_member_fold_G,
                        static_cast<int>(ctx.rspace_operations.size()),
                        &ctx.ao_type_layouts, nullptr);
    ctx.ao_shell_layout_available = !ctx.ao_type_layouts.empty();
    infer_atom_to_type_from_kstars(ctx.kstars, ctx.atom_to_type);
}

std::string find_first_existing_file(const std::vector<std::string>& candidates)
{
    for (const auto& candidate : candidates)
    {
        if (!candidate.empty() && file_exists(candidate))
        {
            return candidate;
        }
    }
    return "";
}

std::string read_input_symmetry_keyword(const std::string& input_file, const std::string& keyword)
{
    std::ifstream ifs(input_file);
    if (!ifs.good())
    {
        return "";
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (cleaned.empty())
        {
            continue;
        }
        const auto fields = split_fields(cleaned);
        if (!fields.empty() && fields.front() == keyword && fields.size() >= 2)
        {
            return fields[1];
        }
    }
    return "";
}

struct ParsedInputSymmetryStru
{
    std::vector<std::string> species_labels;
    std::vector<std::string> orbital_files;
    std::map<atom_t, int> atom_to_type;
    std::map<atom_t, std::array<double, 3>> coord_frac;
};

ParsedInputSymmetryStru parse_input_symmetry_stru_file(const std::string& stru_file)
{
    std::ifstream ifs(stru_file);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + stru_file);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (!cleaned.empty())
        {
            lines.push_back(cleaned);
        }
    }

    ParsedInputSymmetryStru parsed;
    double lattice_constant = 1.0;
    Matrix3 lattice_vectors;
    bool has_lattice_vectors = false;
    std::size_t index = 0;
    while (index < lines.size())
    {
        const std::string& current = lines[index];
        if (current == "LATTICE_CONSTANT")
        {
            ++index;
            if (index >= lines.size())
            {
                throw std::runtime_error("Missing LATTICE_CONSTANT value in " + stru_file);
            }
            const auto fields = split_fields(lines[index]);
            if (fields.empty())
            {
                throw std::runtime_error("Missing LATTICE_CONSTANT value in " + stru_file);
            }
            lattice_constant = std::stod(fields.front());
            ++index;
            continue;
        }
        if (current == "LATTICE_VECTORS")
        {
            ++index;
            if (index + 2 >= lines.size())
            {
                throw std::runtime_error("Incomplete LATTICE_VECTORS block in " + stru_file);
            }
            std::array<std::array<double, 3>, 3> lattice_rows{{{{0.0, 0.0, 0.0}},
                                                                {{0.0, 0.0, 0.0}},
                                                                {{0.0, 0.0, 0.0}}}};
            for (int row = 0; row < 3; ++row, ++index)
            {
                const auto values = extract_doubles(lines[index]);
                if (values.size() != 3)
                {
                    throw std::runtime_error("Failed to parse LATTICE_VECTORS row in "
                                             + stru_file + ": " + lines[index]);
                }
                for (int col = 0; col < 3; ++col)
                {
                    lattice_rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                        values[static_cast<std::size_t>(col)] * lattice_constant;
                }
            }
            lattice_vectors = build_matrix3_from_array(lattice_rows);
            has_lattice_vectors = true;
            continue;
        }
        if (current == "ATOMIC_SPECIES")
        {
            ++index;
            while (index < lines.size() && !is_section_header(lines[index]))
            {
                const auto fields = split_fields(lines[index]);
                if (!fields.empty())
                {
                    parsed.species_labels.push_back(fields.front());
                }
                ++index;
            }
            continue;
        }
        if (current == "NUMERICAL_ORBITAL")
        {
            ++index;
            while (index < lines.size() && !is_section_header(lines[index]))
            {
                parsed.orbital_files.push_back(lines[index]);
                ++index;
            }
            continue;
        }
        if (current == "ATOMIC_POSITIONS")
        {
            ++index;
            if (index >= lines.size())
            {
                throw std::runtime_error("Missing ATOMIC_POSITIONS mode in " + stru_file);
            }
            const std::string position_mode = lines[index];
            ++index;

            atom_t atom_index = 0;
            while (index < lines.size() && !is_section_header(lines[index]))
            {
                const auto species_fields = split_fields(lines[index]);
                if (species_fields.empty())
                {
                    ++index;
                    continue;
                }
                const std::string& species_label = species_fields.front();
                const auto type_iter =
                    std::find(parsed.species_labels.begin(), parsed.species_labels.end(), species_label);
                if (type_iter == parsed.species_labels.end())
                {
                    throw std::runtime_error("Failed to match atom type label " + species_label
                                             + " in " + stru_file);
                }
                const int atom_type =
                    static_cast<int>(std::distance(parsed.species_labels.begin(), type_iter));
                ++index;
                if (index >= lines.size())
                {
                    throw std::runtime_error("Unexpected end of file while reading magnetic moment block in "
                                             + stru_file);
                }
                ++index;
                if (index >= lines.size())
                {
                    throw std::runtime_error("Unexpected end of file while reading atom count in "
                                             + stru_file);
                }
                const auto count_fields = split_fields(lines[index]);
                if (count_fields.empty())
                {
                    throw std::runtime_error("Missing atom count in " + stru_file);
                }
                const int nat_this_type = std::stoi(count_fields.front());
                ++index;
                for (int i = 0; i < nat_this_type; ++i)
                {
                    if (index >= lines.size())
                    {
                        throw std::runtime_error("Unexpected end of file while reading atomic positions in "
                                                 + stru_file);
                    }
                    const auto atom_fields = split_fields(lines[index]);
                    if (atom_fields.size() < 3)
                    {
                        throw std::runtime_error("Failed to parse atomic coordinate in "
                                                 + stru_file + ": " + lines[index]);
                    }
                    std::array<double, 3> coord_raw{
                        std::stod(atom_fields[0]),
                        std::stod(atom_fields[1]),
                        std::stod(atom_fields[2]),
                    };

                    std::array<double, 3> coord_frac = coord_raw;
                    std::string mode_lower = position_mode;
                    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(),
                                   [](unsigned char ch) {
                                       return static_cast<char>(std::tolower(ch));
                                   });
                    if (mode_lower == "direct")
                    {
                        coord_frac = coord_raw;
                    }
                    else if (starts_with(mode_lower, "cartesian"))
                    {
                        if (!has_lattice_vectors)
                        {
                            throw std::runtime_error("ATOMIC_POSITIONS uses Cartesian coordinates but "
                                                     "LATTICE_VECTORS is unavailable in "
                                                     + stru_file);
                        }
                        Vector3<double> coord_cart(coord_raw[0], coord_raw[1], coord_raw[2]);
                        if (mode_lower.find("angstrom") != std::string::npos)
                        {
                            coord_cart.x *= ANG2BOHR;
                            coord_cart.y *= ANG2BOHR;
                            coord_cart.z *= ANG2BOHR;
                        }
                        const Vector3<double> coord_frac_vec =
                            coord_cart * lattice_vectors.Inverse();
                        coord_frac = {coord_frac_vec.x, coord_frac_vec.y, coord_frac_vec.z};
                    }
                    else
                    {
                        throw std::runtime_error("Unsupported ATOMIC_POSITIONS mode `" + position_mode
                                                 + "` in " + stru_file);
                    }

                    parsed.atom_to_type[atom_index] = atom_type;
                    parsed.coord_frac[atom_index] = coord_frac;
                    ++atom_index;
                    ++index;
                }
            }
            continue;
        }
        ++index;
    }

    return parsed;
}

ParsedInputSymmetryStru parse_input_symmetry_stru_out_file(const std::string& stru_file)
{
    std::ifstream ifs(stru_file);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + stru_file);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (!cleaned.empty())
        {
            lines.push_back(cleaned);
        }
    }
    if (lines.size() < 7)
    {
        throw std::runtime_error("Incomplete ABACUS stru_out file " + stru_file);
    }

    std::array<std::array<double, 3>, 3> lattice_rows{{{{0.0, 0.0, 0.0}},
                                                        {{0.0, 0.0, 0.0}},
                                                        {{0.0, 0.0, 0.0}}}};
    for (int row = 0; row < 3; ++row)
    {
        const auto values = extract_doubles(lines[static_cast<std::size_t>(row)]);
        if (values.size() != 3)
        {
            throw std::runtime_error("Failed to parse lattice row in " + stru_file);
        }
        for (int col = 0; col < 3; ++col)
        {
            lattice_rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                values[static_cast<std::size_t>(col)];
        }
    }
    const Matrix3 lattice_vectors = build_matrix3_from_array(lattice_rows);

    const auto natom_fields = split_fields(lines[6]);
    if (natom_fields.empty())
    {
        throw std::runtime_error("Failed to parse atom count in " + stru_file);
    }
    const int natoms = std::stoi(natom_fields.front());
    if (natoms < 0 || static_cast<std::size_t>(7 + natoms) > lines.size())
    {
        throw std::runtime_error("Incomplete atomic coordinate block in " + stru_file);
    }

    ParsedInputSymmetryStru parsed;
    for (int iatom = 0; iatom != natoms; ++iatom)
    {
        const auto fields = split_fields(lines[static_cast<std::size_t>(7 + iatom)]);
        if (fields.size() < 3)
        {
            throw std::runtime_error("Failed to parse atomic coordinate row in " + stru_file);
        }
        const Vector3<double> coord_cart{
            std::stod(fields[0]), std::stod(fields[1]), std::stod(fields[2])};
        const Vector3<double> coord_frac_vec = coord_cart * lattice_vectors.Inverse();
        parsed.coord_frac[static_cast<atom_t>(iatom)] =
            {coord_frac_vec.x, coord_frac_vec.y, coord_frac_vec.z};
    }
    return parsed;
}

ParsedInputSymmetryStru parse_input_symmetry_coord_source_file(const std::string& file_path)
{
    if (base_name(file_path) == "stru_out")
    {
        return parse_input_symmetry_stru_out_file(file_path);
    }
    return parse_input_symmetry_stru_file(file_path);
}

void try_load_input_symmetry_coord_frac(const std::string& dir_path,
                                      InputSymmetryContext& ctx,
                                      std::ostream* log)
{
    const auto candidate_dirs = build_input_symmetry_path_candidates(dir_path);
    std::vector<std::string> stru_candidates;
    for (const auto& dir : candidate_dirs)
    {
        stru_candidates.push_back(join_path(dir, "STRU"));
        stru_candidates.push_back(join_path(dir, "stru_out"));
    }

    std::vector<std::string> existing_candidates;
    for (const auto& candidate : stru_candidates)
    {
        if (!candidate.empty() && file_exists(candidate))
        {
            existing_candidates.push_back(candidate);
        }
    }
    if (existing_candidates.empty())
    {
        if (log != nullptr)
        {
            (*log) << "| Input fractional coords: unavailable (STRU/stru_out not found)\n";
        }
        return;
    }

    std::string last_error;
    for (const auto& stru_file : existing_candidates)
    {
        try
        {
            const ParsedInputSymmetryStru parsed = parse_input_symmetry_coord_source_file(stru_file);
            if (parsed.coord_frac.empty())
            {
                throw std::runtime_error("atomic coordinates not found in " + stru_file);
            }
            if (!parsed.atom_to_type.empty() && !ctx.atom_to_type.empty()
                && parsed.atom_to_type.size() != ctx.atom_to_type.size())
            {
                throw std::runtime_error("atom count does not match symrot_k.txt in " + stru_file);
            }
            for (const auto& atom_type : parsed.atom_to_type)
            {
                const auto iter = ctx.atom_to_type.find(atom_type.first);
                if (iter != ctx.atom_to_type.end() && iter->second != atom_type.second)
                {
                    throw std::runtime_error("atom ordering/type mapping is inconsistent with "
                                             "symrot_k.txt in " + stru_file);
                }
            }
            ctx.input_coord_frac = parsed.coord_frac;
            if (log != nullptr)
            {
                (*log) << "| Input fractional coords: loaded from " << base_name(stru_file)
                       << " for " << ctx.input_coord_frac.size() << " atoms\n";
            }
            return;
        }
        catch (const std::exception& ex)
        {
            last_error = ex.what();
        }
    }

    if (log != nullptr)
    {
        (*log) << "| Input fractional coords: unavailable (" << last_error << ")\n";
    }
}

InputSymmetryAOTypeLayout parse_input_symmetry_orbital_file(const std::string& orbital_file,
                                             const std::string& species_label)
{
    std::ifstream ifs(orbital_file);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open orbital file " + orbital_file);
    }

    InputSymmetryAOTypeLayout layout;
    layout.label = species_label;
    layout.orbital_file = orbital_file;

    int lmax = -1;
    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (cleaned.empty())
        {
            continue;
        }
        if (starts_with(cleaned, "Lmax"))
        {
            const auto values = extract_integers(cleaned);
            if (values.empty())
            {
                throw std::runtime_error("Failed to parse Lmax in orbital file " + orbital_file);
            }
            lmax = static_cast<int>(values.front());
            layout.shell_counts.resize(static_cast<std::size_t>(lmax + 1), 0);
            continue;
        }
        if (starts_with(cleaned, "Number of "))
        {
            const auto prefix_size = std::string("Number of ").size();
            const auto token = cleaned.substr(prefix_size);
            char shell_symbol = '\0';
            for (const char ch : token)
            {
                if (std::isalpha(static_cast<unsigned char>(ch)) != 0)
                {
                    shell_symbol = ch;
                    break;
                }
            }
            const int l = shell_symbol_to_l(shell_symbol);
            const auto values = extract_integers(cleaned);
            if (l < 0 || values.empty())
            {
                continue;
            }
            if (static_cast<int>(layout.shell_counts.size()) <= l)
            {
                layout.shell_counts.resize(static_cast<std::size_t>(l + 1), 0);
            }
            layout.shell_counts[static_cast<std::size_t>(l)] = static_cast<int>(values.back());
        }
        if (starts_with(cleaned, "SUMMARY"))
        {
            break;
        }
    }

    if (lmax >= 0 && static_cast<int>(layout.shell_counts.size()) < lmax + 1)
    {
        layout.shell_counts.resize(static_cast<std::size_t>(lmax + 1), 0);
    }
    layout.nao = compute_nao_from_shell_counts(layout.shell_counts);
    if (layout.nao <= 0)
    {
        throw std::runtime_error("Parsed zero AO functions from orbital file " + orbital_file);
    }
    return layout;
}

std::string resolve_input_symmetry_file(const std::string& file_name,
                                const std::vector<std::string>& search_dirs)
{
    if (is_absolute_path(file_name))
    {
        return file_exists(file_name) ? file_name : "";
    }

    for (const auto& dir : search_dirs)
    {
        if (dir.empty())
        {
            continue;
        }
        const std::string candidate = join_path(dir, file_name);
        if (file_exists(candidate))
        {
            return candidate;
        }
    }
    return file_exists(file_name) ? file_name : "";
}

bool append_unique_abf_layout(std::vector<InputSymmetryAOTypeLayout>& candidates,
                              const InputSymmetryAOTypeLayout& layout)
{
    const auto duplicate = std::find_if(candidates.begin(), candidates.end(),
                                        [&layout](const InputSymmetryAOTypeLayout& candidate) {
                                            return candidate.label == layout.label
                                                   && candidate.shell_counts == layout.shell_counts
                                                   && candidate.nao == layout.nao;
                                        });
    if (duplicate != candidates.end())
    {
        return false;
    }
    candidates.push_back(layout);
    return true;
}

[[maybe_unused]] void try_load_input_symmetry_ao_shell_layout(const std::string& dir_path,
                                                      InputSymmetryContext& ctx,
                                                      std::ostream* log)
{
    const auto candidate_dirs = build_input_symmetry_path_candidates(dir_path);
    std::vector<std::string> stru_candidates;
    std::vector<std::string> input_candidates;
    for (const auto& dir : candidate_dirs)
    {
        stru_candidates.push_back(join_path(dir, "STRU"));
        input_candidates.push_back(join_path(dir, "INPUT"));
    }

    const std::string stru_file = find_first_existing_file(stru_candidates);
    if (stru_file.empty())
    {
        if (log != nullptr)
        {
            (*log) << "| AO shell layout        : unavailable (STRU not found)\n";
        }
        return;
    }

    const std::string input_file = find_first_existing_file(input_candidates);
    const std::string orbital_dir =
        input_file.empty() ? "" : read_input_symmetry_keyword(input_file, "orbital_dir");

    try
    {
        const ParsedInputSymmetryStru parsed = parse_input_symmetry_stru_file(stru_file);
        if (parsed.species_labels.empty() || parsed.orbital_files.empty())
        {
            throw std::runtime_error("Failed to find ATOMIC_SPECIES / NUMERICAL_ORBITAL sections in "
                                     + stru_file);
        }
        if (parsed.species_labels.size() != parsed.orbital_files.size())
        {
            throw std::runtime_error("The number of NUMERICAL_ORBITAL entries does not match "
                                     "ATOMIC_SPECIES in " + stru_file);
        }

        std::vector<std::string> search_dirs;
        search_dirs.push_back(parent_path(stru_file));
        search_dirs.push_back(dir_path);
        if (!input_file.empty())
        {
            search_dirs.push_back(parent_path(input_file));
        }
        if (!orbital_dir.empty())
        {
            if (is_absolute_path(orbital_dir))
            {
                search_dirs.push_back(orbital_dir);
            }
            else
            {
                if (!input_file.empty())
                {
                    search_dirs.push_back(join_path(parent_path(input_file), orbital_dir));
                }
                search_dirs.push_back(join_path(parent_path(stru_file), orbital_dir));
                search_dirs.push_back(join_path(dir_path, orbital_dir));
            }
        }

        ctx.ao_type_layouts.clear();
        ctx.ao_type_layouts.reserve(parsed.species_labels.size());
        for (std::size_t itype = 0; itype < parsed.species_labels.size(); ++itype)
        {
            const std::string resolved_orbital =
                resolve_input_symmetry_file(parsed.orbital_files[itype], search_dirs);
            if (resolved_orbital.empty())
            {
                throw std::runtime_error("Failed to resolve orbital file "
                                         + parsed.orbital_files[itype] + " for species "
                                         + parsed.species_labels[itype]);
            }
            ctx.ao_type_layouts.push_back(
                parse_input_symmetry_orbital_file(resolved_orbital, parsed.species_labels[itype]));
        }

        ctx.atom_to_type = parsed.atom_to_type;
        ctx.input_coord_frac = parsed.coord_frac;
        ctx.ao_shell_layout_available = true;
        if (log != nullptr)
        {
            (*log) << "| AO shell layout        : loaded for " << ctx.ao_type_layouts.size()
                   << " atom types and " << ctx.atom_to_type.size() << " atoms\n";
            for (std::size_t itype = 0; itype < ctx.ao_type_layouts.size(); ++itype)
            {
                const auto& layout = ctx.ao_type_layouts[itype];
                (*log) << "|   type " << itype << " (" << layout.label << ")"
                       << " nao=" << layout.nao << " shell_counts=";
                for (std::size_t l = 0; l < layout.shell_counts.size(); ++l)
                {
                    if (l != 0)
                    {
                        (*log) << ",";
                    }
                    (*log) << layout.shell_counts[l];
                }
                (*log) << "\n";
            }
        }
    }
    catch (const std::exception& ex)
    {
        ctx.ao_type_layouts.clear();
        ctx.atom_to_type.clear();
        ctx.ao_shell_layout_available = false;
        if (log != nullptr)
        {
            (*log) << "| AO shell layout        : unavailable (" << ex.what() << ")\n";
        }
    }
}

} // namespace

InputSymmetryConvention parse_input_symmetry_convention(const std::string& convention)
{
    std::string normalized = trim(convention);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (normalized == "auto" || normalized.empty())
    {
        return InputSymmetryConvention::AUTO;
    }
    if (normalized == "abacus")
    {
        return InputSymmetryConvention::ABACUS;
    }
    if (normalized == "none" || normalized == "off" || normalized == "false")
    {
        return InputSymmetryConvention::NONE;
    }
    throw std::runtime_error("Unknown input symmetry convention: " + convention);
}

std::string input_symmetry_convention_name(const InputSymmetryConvention convention)
{
    switch (convention)
    {
    case InputSymmetryConvention::NONE:
        return "none";
    case InputSymmetryConvention::AUTO:
        return "auto";
    case InputSymmetryConvention::ABACUS:
        return "abacus";
    }
    return "unknown";
}

void InputSymmetryContext::clear()
{
    convention = InputSymmetryConvention::NONE;
    available = false;
    lattice_available = false;
    ao_shell_layout_available = false;
    abf_shell_layout_available = false;
    ao_lmax = -1;
    abf_lmax = -1;
    basis_convention = {};
    irreducible_sector.clear();
    rspace_operations.clear();
    kstars.clear();
    abf_kstars.clear();
    ao_type_layouts.clear();
    abf_type_layout_candidates.clear();
    atom_to_type.clear();
    input_coord_frac.clear();
    lattice_vectors.Reset();
    reciprocal_vectors.Reset();
    kspace_return_lattice.clear();
    kstar_member_fold_G.clear();
}

void InputSymmetryContext::set_lattice(const Matrix3& latvec, const Matrix3& G)
{
    lattice_vectors = latvec;
    reciprocal_vectors = G;
    lattice_available = true;
}

bool InputSymmetryContext::empty() const
{
    return irreducible_sector.empty() && rspace_operations.empty() && kstars.empty()
           && abf_kstars.empty()
           && ao_type_layouts.empty() && abf_type_layout_candidates.empty()
           && atom_to_type.empty()
           && kspace_return_lattice.empty() && kstar_member_fold_G.empty();
}

bool InputSymmetryContext::has_ao_shell_layout() const
{
    return ao_shell_layout_available && !ao_type_layouts.empty();
}

bool InputSymmetryContext::has_abf_shell_layout() const
{
    return abf_shell_layout_available && !abf_type_layout_candidates.empty();
}

std::size_t InputSymmetryContext::count_irreducible_pairs() const
{
    return irreducible_sector.size();
}

std::size_t InputSymmetryContext::count_irreducible_blocks() const
{
    std::size_t count = 0;
    for (const auto& pair_Rs : irreducible_sector)
    {
        count += pair_Rs.second.size();
    }
    return count;
}

std::size_t InputSymmetryContext::count_kstar_members() const
{
    std::size_t count = 0;
    for (const auto& star : kstars)
    {
        count += star.members.size();
    }
    return count;
}

std::size_t InputSymmetryContext::count_atoms_with_layout() const
{
    return atom_to_type.size();
}

std::size_t InputSymmetryContext::count_abf_layout_candidates() const
{
    std::size_t count = 0;
    for (const auto& candidates : abf_type_layout_candidates)
    {
        count += candidates.size();
    }
    return count;
}

const InputSymmetryAOTypeLayout& InputSymmetryContext::get_ao_type_layout(const int atom_type) const
{
    if (atom_type < 0 || atom_type >= static_cast<int>(ao_type_layouts.size()))
    {
        throw std::out_of_range("ABACUS atom type is out of range in AO shell layout");
    }
    return ao_type_layouts[static_cast<std::size_t>(atom_type)];
}

const InputSymmetryAOTypeLayout& InputSymmetryContext::find_abf_type_layout(const int atom_type,
                                                                      const int nao_hint) const
{
    if (atom_type < 0 || atom_type >= static_cast<int>(abf_type_layout_candidates.size()))
    {
        throw std::out_of_range("ABACUS atom type is out of range in ABF shell layout");
    }

    const auto& candidates = abf_type_layout_candidates[static_cast<std::size_t>(atom_type)];
    if (candidates.empty())
    {
        throw std::runtime_error("ABACUS ABF shell layout is unavailable for atom type "
                                 + std::to_string(atom_type));
    }

    if (nao_hint > 0)
    {
        const auto matched = std::find_if(candidates.begin(), candidates.end(),
                                          [nao_hint](const InputSymmetryAOTypeLayout& candidate) {
                                              return candidate.nao == nao_hint;
                                          });
        if (matched != candidates.end())
        {
            return *matched;
        }
    }

    if (candidates.size() == 1)
    {
        return candidates.front();
    }

    std::ostringstream oss;
    oss << "Failed to resolve ABF shell layout for atom type " << atom_type
        << " with nao_hint=" << nao_hint << ". Candidate dimensions:";
    for (const auto& candidate : candidates)
    {
        oss << " " << candidate.nao;
    }
    throw std::runtime_error(oss.str());
}

bool load_input_symmetry_context(const std::string& dir_path,
                                  const InputSymmetryConvention convention,
                                  InputSymmetryContext& ctx,
                                  std::ostream* log)
{
    if (convention == InputSymmetryConvention::NONE)
    {
        ctx.clear();
        return false;
    }
    if (convention != InputSymmetryConvention::AUTO
        && convention != InputSymmetryConvention::ABACUS)
    {
        throw std::runtime_error("Unsupported input symmetry convention: "
                                 + input_symmetry_convention_name(convention));
    }

    const auto candidate_dirs = build_input_symmetry_path_candidates(dir_path);
    std::string sidecar_dir;
    for (const auto& dir : candidate_dirs)
    {
        if (file_exists(join_path(dir, "irreducible_sector.txt"))
            || file_exists(join_path(dir, "symrot_R.txt"))
            || file_exists(join_path(dir, "symrot_k.txt"))
            || file_exists(join_path(dir, "symrot_abf_k.txt")))
        {
            sidecar_dir = dir;
            break;
        }
    }

    const std::string irreducible_sector_file = join_path(sidecar_dir, "irreducible_sector.txt");
    const std::string symrot_R_file = join_path(sidecar_dir, "symrot_R.txt");
    const std::string symrot_k_file = join_path(sidecar_dir, "symrot_k.txt");
    const std::string symrot_abf_k_file = join_path(sidecar_dir, "symrot_abf_k.txt");

    const bool has_irreducible_sector = !sidecar_dir.empty() && file_exists(irreducible_sector_file);
    const bool has_symrot_R = !sidecar_dir.empty() && file_exists(symrot_R_file);
    const bool has_symrot_k = !sidecar_dir.empty() && file_exists(symrot_k_file);
    const bool has_symrot_abf_k = !sidecar_dir.empty() && file_exists(symrot_abf_k_file);

    if (!has_irreducible_sector && !has_symrot_R && !has_symrot_k)
    {
        ctx.clear();
        return false;
    }

    if (!(has_irreducible_sector && has_symrot_R && has_symrot_k))
    {
        std::ostringstream oss;
        oss << "Incomplete ABACUS symmetry sidecar set near " << dir_path
            << ". Expected irreducible_sector.txt, symrot_R.txt and symrot_k.txt together.";
        throw std::runtime_error(oss.str());
    }

    ctx.clear();
    ctx.convention = InputSymmetryConvention::ABACUS;
    load_irreducible_sector_file(irreducible_sector_file, ctx.irreducible_sector);
    load_symrot_R_file(symrot_R_file, ctx);
    load_symrot_k_file(symrot_k_file, ctx);
    try_load_input_symmetry_coord_frac(dir_path, ctx, log);
    if (has_symrot_abf_k)
    {
        parse_symrot_k_file(symrot_abf_k_file, ctx.abf_kstars, nullptr, nullptr, -1, nullptr,
                            &ctx.abf_type_layout_candidates);
    }
    ctx.abf_shell_layout_available = !ctx.abf_type_layout_candidates.empty();
    if (ctx.has_ao_shell_layout())
    {
        if (ctx.abf_type_layout_candidates.size() != ctx.ao_type_layouts.size())
        {
            ctx.abf_shell_layout_available = false;
        }
        else
        {
            for (const auto& candidates : ctx.abf_type_layout_candidates)
            {
                if (candidates.empty())
                {
                    ctx.abf_shell_layout_available = false;
                    break;
                }
            }
        }
    }
    ctx.available = true;

    if (log != nullptr)
    {
        (*log) << "Detected ABACUS symmetry sidecar files\n"
               << "| irreducible atom pairs : " << ctx.count_irreducible_pairs() << "\n"
               << "| irreducible {pair, R}  : " << ctx.count_irreducible_blocks() << "\n"
               << "| real-space operations  : " << ctx.rspace_operations.size() << "\n"
               << "| IBZ k-stars            : " << ctx.kstars.size() << "\n"
               << "| total star members     : " << ctx.count_kstar_members() << "\n"
               << "| ABF k rotations        : "
               << (ctx.abf_kstars.empty() ? std::string("fallback to symrot_k.txt")
                                         : std::string("loaded from symrot_abf_k.txt"))
               << "\n"
               << "| AO / ABF lmax          : " << ctx.ao_lmax << " / " << ctx.abf_lmax << "\n";
        if (ctx.has_ao_shell_layout())
        {
            (*log) << "| AO shell layout        : loaded for " << ctx.ao_type_layouts.size()
                   << " atom types and " << ctx.atom_to_type.size() << " atoms\n";
            for (std::size_t itype = 0; itype < ctx.ao_type_layouts.size(); ++itype)
            {
                const auto& layout = ctx.ao_type_layouts[itype];
                (*log) << "|   type " << itype << " (" << layout.label << ")"
                       << " nao=" << layout.nao << " shell_counts=";
                for (std::size_t l = 0; l < layout.shell_counts.size(); ++l)
                {
                    if (l != 0)
                    {
                        (*log) << ",";
                    }
                    (*log) << layout.shell_counts[l];
                }
                (*log) << "\n";
            }
        }
        else
        {
            (*log) << "| AO shell layout        : unavailable (missing in symrot_k.txt)\n";
        }
        if (ctx.has_abf_shell_layout())
        {
            (*log) << "| ABF shell layout       : loaded from symrot_abf_k.txt with "
                   << ctx.count_abf_layout_candidates() << " candidate type layouts\n";
        }
        else
        {
            (*log) << "| ABF shell layout       : unavailable\n";
        }
    }
    return true;
}

ComplexMatrix build_input_symmetry_ao_rotation_matrix(const InputSymmetryContext& ctx,
                                              const int atom_type,
                                              const std::map<int, ComplexMatrix>& shell_rotations)
{
    const auto& layout = ctx.get_ao_type_layout(atom_type);
    ComplexMatrix rotation(layout.nao, layout.nao);

    int offset = 0;
    for (int l = 0; l < static_cast<int>(layout.shell_counts.size()); ++l)
    {
        const int shell_count = layout.shell_counts[static_cast<std::size_t>(l)];
        if (shell_count == 0)
        {
            continue;
        }

        const auto rotation_iter = shell_rotations.find(l);
        if (rotation_iter == shell_rotations.end())
        {
            throw std::runtime_error("Missing shell rotation block for l=" + std::to_string(l));
        }

        const ComplexMatrix& shell_rotation = rotation_iter->second;
        const int nm = 2 * l + 1;
        if (shell_rotation.nr != nm || shell_rotation.nc != nm)
        {
            throw std::runtime_error("Shell rotation block has incompatible shape for l="
                                     + std::to_string(l));
        }

        for (int ishell = 0; ishell < shell_count; ++ishell)
        {
            for (int row = 0; row < nm; ++row)
            {
                for (int col = 0; col < nm; ++col)
                {
                    rotation(offset + row, offset + col) = shell_rotation(row, col);
                }
            }
            offset += nm;
        }
    }

    if (offset != layout.nao)
    {
        throw std::runtime_error("Failed to assemble the full AO rotation matrix for atom type "
                                 + std::to_string(atom_type));
    }
    return rotation;
}

ComplexMatrix build_input_symmetry_abf_rotation_matrix(
    const InputSymmetryContext& ctx,
    const int atom_type,
    const int nao_hint,
    const std::map<int, ComplexMatrix>& shell_rotations,
    const Matrix3& direct_rotation)
{
    const auto& layout = ctx.find_abf_type_layout(atom_type, nao_hint);
    ComplexMatrix rotation(layout.nao, layout.nao);

    int offset = 0;
    for (int l = 0; l < static_cast<int>(layout.shell_counts.size()); ++l)
    {
        const int shell_count = layout.shell_counts[static_cast<std::size_t>(l)];
        if (shell_count == 0)
        {
            continue;
        }

        ComplexMatrix shell_rotation;
        const auto rotation_iter = shell_rotations.find(l);
        if (rotation_iter != shell_rotations.end())
        {
            shell_rotation = rotation_iter->second;
        }
        else
        {
            shell_rotation =
                build_input_symmetry_shell_rotation_from_direct_rotation(ctx, l, direct_rotation);
        }

        const int nm = 2 * l + 1;
        if (shell_rotation.nr != nm || shell_rotation.nc != nm)
        {
            throw std::runtime_error("ABF shell rotation block has incompatible shape for l="
                                     + std::to_string(l));
        }

        for (int ishell = 0; ishell < shell_count; ++ishell)
        {
            for (int row = 0; row < nm; ++row)
            {
                for (int col = 0; col < nm; ++col)
                {
                    rotation(offset + row, offset + col) = shell_rotation(row, col);
                }
            }
            offset += nm;
        }
    }

    if (offset != layout.nao)
    {
        throw std::runtime_error("Failed to assemble the full ABF rotation matrix for atom type "
                                 + std::to_string(atom_type));
    }
    return rotation;
}

namespace
{

int find_input_symmetry_kstar_index_for_kpoint(const std::vector<InputSymmetryKStar>& kstars,
                                       const Vector3_Order<double>& k_point,
                                       const std::string& label)
{
    int matched_index = -1;

    // Prefer the canonical IBZ representative when the incoming q-point already uses the same
    // gauge as the ABACUS sidecar. This keeps the old fast path intact.
    for (std::size_t istar = 0; istar < kstars.size(); ++istar)
    {
        if (!nearly_same_kpoint(kstars[istar].k_ibz, k_point))
        {
            continue;
        }
        if (matched_index >= 0)
        {
            throw std::runtime_error(label + " matching is ambiguous for the current q-point");
        }
        matched_index = static_cast<int>(istar);
    }

    if (matched_index >= 0)
    {
        return matched_index;
    }

    // LibRPA may keep an equivalent full-star member, rather than the ABACUS representative, as
    // the active q-point label. Fall back to a unique member match in that case.
    for (std::size_t istar = 0; istar < kstars.size(); ++istar)
    {
        const auto& star = kstars[istar];
        const bool star_contains_kpoint =
            std::any_of(star.members.begin(), star.members.end(),
                        [&k_point](const InputSymmetryKStarMember& member) {
                            return nearly_same_kpoint(member.k_bz, k_point);
                        });
        if (!star_contains_kpoint)
        {
            continue;
        }
        if (matched_index >= 0)
        {
            throw std::runtime_error(label + " member matching is ambiguous for the current q-point");
        }
        matched_index = static_cast<int>(istar);
    }

    if (matched_index < 0)
    {
        throw std::runtime_error("Failed to match the current q-point with " + label);
    }
    return matched_index;
}

} // namespace

const InputSymmetryKStar& find_input_symmetry_kstar_for_kpoint(const std::vector<InputSymmetryKStar>& kstars,
                                                const Vector3_Order<double>& k_point,
                                                const std::string& label)
{
    return kstars[static_cast<std::size_t>(
        find_input_symmetry_kstar_index_for_kpoint(kstars, k_point, label))];
}

const InputSymmetryKStar& find_input_symmetry_kstar_for_ibz_kpoint(const InputSymmetryContext& ctx,
                                                    const Vector3_Order<double>& k_ibz)
{
    return find_input_symmetry_kstar_for_kpoint(ctx.kstars, k_ibz);
}

std::vector<InputSymmetryKStarGridMappingEntry> build_input_symmetry_kstar_grid_mapping(
    const InputSymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& klist_internal,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>>& irk_to_full_kpoints)
{
    if (klist_internal.size() != kfrac_list.size())
    {
        throw std::runtime_error(
            "LibRPA k-point metadata is inconsistent: `klist` and `kfrac_list` have different sizes");
    }
    if (ctx.kstars.size() != kfrac_list.size())
    {
        throw std::runtime_error(
            "ABACUS k-star metadata does not match the loaded LibRPA IBZ k-point count");
    }
    if (!ctx.lattice_available)
    {
        throw std::runtime_error(
            "ABACUS k-star grid mapping requires reciprocal lattice vectors from the structure input");
    }

    auto convert_fractional_to_internal = [&ctx](const Vector3_Order<double>& kfrac) {
        // Keep the same row-vector convention as `read_data.cpp::convert_fractional_kpoint_to_klist_units`
        // so the sidecar-derived q/k keys are identical to the keys used in `klist` and `map_irk_ks`.
        const auto& G = ctx.reciprocal_vectors;
        return Vector3_Order<double>{kfrac.x * G.e11 + kfrac.y * G.e21 + kfrac.z * G.e31,
                                     kfrac.x * G.e12 + kfrac.y * G.e22 + kfrac.z * G.e32,
                                     kfrac.x * G.e13 + kfrac.y * G.e23 + kfrac.z * G.e33};
    };

    auto find_matching_ibz_full_list =
        [&irk_to_full_kpoints](const Vector3_Order<double>& q_ibz_key)
        -> std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>>::const_iterator {
        const auto exact_iter = irk_to_full_kpoints.find(q_ibz_key);
        if (exact_iter != irk_to_full_kpoints.end())
        {
            return exact_iter;
        }

        return std::find_if(irk_to_full_kpoints.begin(), irk_to_full_kpoints.end(),
                            [&q_ibz_key](const auto& entry) {
                                return nearly_same_kpoint(entry.first, q_ibz_key);
                            });
    };

    std::vector<InputSymmetryKStarGridMappingEntry> mapping(kfrac_list.size());
    std::vector<bool> matched_stars(ctx.kstars.size(), false);

    for (std::size_t iq_ibz = 0; iq_ibz < kfrac_list.size(); ++iq_ibz)
    {
        const int matched_star_index = find_input_symmetry_kstar_index_for_kpoint(
            ctx.kstars, kfrac_list[iq_ibz], "input symmetry k-stars");

        matched_stars[static_cast<std::size_t>(matched_star_index)] = true;
        auto& entry = mapping[iq_ibz];
        entry.iq_ibz = static_cast<int>(iq_ibz);
        entry.star_list_index = matched_star_index;

        const auto& star = ctx.kstars[static_cast<std::size_t>(matched_star_index)];
        const auto q_ibz_key = klist_internal[iq_ibz];
        const auto full_list_iter = find_matching_ibz_full_list(q_ibz_key);
        const std::vector<Vector3_Order<double>>* full_q_keys =
            (full_list_iter == irk_to_full_kpoints.end()) ? nullptr : &full_list_iter->second;
        std::vector<bool> matched_full_q(
            (full_q_keys == nullptr) ? 0 : full_q_keys->size(), false);
        entry.member_q_bz_keys.resize(star.members.size());

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto member_q_internal =
                convert_fractional_to_internal(star.members[imember].k_bz);
            entry.member_q_bz_keys[imember] = member_q_internal;

            // The ABACUS sidecar is the authoritative description of the k-star members,
            // including the exact representative chosen after symmetry and BZ folding.
            // LibRPA's `map_irk_ks` is only used here as an optional source of already
            // existing internal q keys. If a member cannot be matched back to that rebuilt
            // list, keep the sidecar-derived key instead of rejecting the star.
            if (full_q_keys == nullptr)
            {
                continue;
            }

            int matched_full_index = -1;
            for (std::size_t ifull = 0; ifull < full_q_keys->size(); ++ifull)
            {
                if (matched_full_q[ifull]
                    || !nearly_same_kpoint((*full_q_keys)[ifull], member_q_internal))
                {
                    continue;
                }
                if (matched_full_index >= 0)
                {
                    throw std::runtime_error(
                        "ABACUS star member to LibRPA full-q matching is ambiguous");
                }
                matched_full_index = static_cast<int>(ifull);
            }

            if (matched_full_index >= 0)
            {
                matched_full_q[static_cast<std::size_t>(matched_full_index)] = true;
                entry.member_q_bz_keys[imember] =
                    (*full_q_keys)[static_cast<std::size_t>(matched_full_index)];
            }
        }
    }

    for (std::size_t istar = 0; istar < matched_stars.size(); ++istar)
    {
        if (!matched_stars[istar])
        {
            throw std::runtime_error(
                "Not every ABACUS k-star could be matched to the loaded LibRPA IBZ grid");
        }
    }

    return mapping;
}

std::vector<InputSymmetryFullKpointMemberEntry> build_input_symmetry_full_kpoint_member_list(
    const InputSymmetryContext& ctx,
    const std::vector<Vector3_Order<double>>& kfrac_list)
{
    std::vector<InputSymmetryFullKpointMemberEntry> members;
    if (!ctx.available || ctx.kstars.empty())
    {
        return members;
    }

    members.reserve(ctx.count_kstar_members());
    for (int ik_ibz = 0; ik_ibz != static_cast<int>(kfrac_list.size()); ++ik_ibz)
    {
        const int matched_star_index = find_input_symmetry_kstar_index_for_kpoint(
            ctx.kstars, kfrac_list[static_cast<std::size_t>(ik_ibz)], "input symmetry k-stars");

        const auto& star = ctx.kstars[static_cast<std::size_t>(matched_star_index)];
        for (int imember = 0; imember != static_cast<int>(star.members.size()); ++imember)
        {
            const auto& member = star.members[static_cast<std::size_t>(imember)];
            members.push_back({ik_ibz, matched_star_index, imember, member.isym, member.k_bz});
        }
    }

    return members;
}

Vector3_Order<int> build_input_symmetry_kspace_return_lattice(
    const InputSymmetryContext& ctx,
    const InputSymmetryKAtomRotation& atom_rotation,
    const std::map<atom_t, std::array<double, 3>>& coord_frac_map,
    const int spatial_isym)
{
    const auto stored_return_lattice =
        ctx.kspace_return_lattice.find({atom_rotation.atom_from, spatial_isym});
    if (stored_return_lattice != ctx.kspace_return_lattice.end())
    {
        return stored_return_lattice->second;
    }

    const auto coord_from_iter = coord_frac_map.find(static_cast<atom_t>(atom_rotation.atom_from));
    const auto coord_to_iter = coord_frac_map.find(static_cast<atom_t>(atom_rotation.atom_to));
    if (coord_from_iter == coord_frac_map.end() || coord_to_iter == coord_frac_map.end())
    {
        throw std::runtime_error("Missing fractional coordinate for ABACUS k-space phase correction");
    }

    if (spatial_isym < 0 || spatial_isym >= static_cast<int>(ctx.rspace_operations.size()))
    {
        throw std::runtime_error("ABACUS k-space phase correction uses an invalid symmetry index");
    }

    const auto& op = ctx.rspace_operations[static_cast<std::size_t>(spatial_isym)];
    const Vector3_Order<double> coord_from =
        restrict_fractional_coordinate({coord_from_iter->second[0],
                                        coord_from_iter->second[1],
                                        coord_from_iter->second[2]});
    const Vector3_Order<double> coord_to =
        restrict_fractional_coordinate({coord_to_iter->second[0],
                                        coord_to_iter->second[1],
                                        coord_to_iter->second[2]});
    const Vector3_Order<double> transformed =
        multiply_row_vector(coord_from, op.rotation) + restrict_fractional_coordinate(op.translation);
    const Vector3_Order<double> return_lattice = transformed - coord_to;
    if (!is_nearly_integer_vec3(return_lattice))
    {
        throw std::runtime_error("ABACUS k-space phase correction produced a non-integer return lattice");
    }
    return round_vec3_to_int(return_lattice);
}

Vector3_Order<int> build_input_symmetry_equivalent_kpoint_shift(
    const Vector3_Order<double>& k_bz_source,
    const Vector3_Order<double>& k_bz_target)
{
    const Vector3_Order<double> k_shift{
        k_bz_target.x - k_bz_source.x,
        k_bz_target.y - k_bz_source.y,
        k_bz_target.z - k_bz_source.z,
    };
    if (!is_nearly_integer_vec3(k_shift))
    {
        throw std::runtime_error(
            "ABACUS symmetry restore encountered non-equivalent full-k representatives");
    }
    return round_vec3_to_int(k_shift);
}

std::pair<atom_t, atom_t> canonicalize_input_symmetry_upper_atom_pair(const atom_t atom_i,
                                                              const atom_t atom_j)
{
    return (atom_i <= atom_j) ? std::make_pair(atom_i, atom_j)
                              : std::make_pair(atom_j, atom_i);
}

std::vector<const InputSymmetryKAtomRotation*> build_input_symmetry_rotations_by_from(
    const InputSymmetryKStarMember& member)
{
    int max_atom_index = -1;
    for (const auto& atom_rotation : member.atom_rotations)
    {
        max_atom_index = std::max(max_atom_index, atom_rotation.atom_from);
        max_atom_index = std::max(max_atom_index, atom_rotation.atom_to);
    }
    std::vector<const InputSymmetryKAtomRotation*> rotations_by_from(
        static_cast<std::size_t>(max_atom_index + 1), nullptr);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        rotations_by_from.at(static_cast<std::size_t>(atom_rotation.atom_from)) = &atom_rotation;
    }
    return rotations_by_from;
}

std::set<std::pair<atom_t, atom_t>> build_input_symmetry_upper_atom_pair_closure(
    const InputSymmetryKStar& star,
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs)
{
    std::set<std::pair<atom_t, atom_t>> closure_pairs;
    for (const auto& atom_pair : target_atom_pairs)
    {
        closure_pairs.insert(canonicalize_input_symmetry_upper_atom_pair(
            atom_pair.first, atom_pair.second));
    }

    bool changed = true;
    while (changed)
    {
        changed = false;
        const auto snapshot_pairs = closure_pairs;
        for (const auto& member : star.members)
        {
            const auto rotations_by_from = build_input_symmetry_rotations_by_from(member);
            for (const auto& atom_pair : snapshot_pairs)
            {
                if (static_cast<std::size_t>(atom_pair.first) >= rotations_by_from.size()
                    || static_cast<std::size_t>(atom_pair.second) >= rotations_by_from.size())
                {
                    throw std::runtime_error(
                        "ABACUS atom-pair closure requested an atom outside the loaded star");
                }
                const auto* rot_i = rotations_by_from[static_cast<std::size_t>(atom_pair.first)];
                const auto* rot_j = rotations_by_from[static_cast<std::size_t>(atom_pair.second)];
                if (rot_i == nullptr || rot_j == nullptr)
                {
                    throw std::runtime_error(
                        "ABACUS atom-pair closure found an incomplete atom permutation");
                }
                const auto source_pair = canonicalize_input_symmetry_upper_atom_pair(
                    static_cast<atom_t>(rot_i->atom_to),
                    static_cast<atom_t>(rot_j->atom_to));
                if (closure_pairs.insert(source_pair).second)
                {
                    changed = true;
                }
            }
        }
    }
    return closure_pairs;
}

namespace
{

std::complex<double> build_input_symmetry_reciprocal_gauge_phase(
    const Vector3_Order<int>& k_shift,
    const atom_t atom,
    const std::map<atom_t, std::array<double, 3>>& coord_frac_map)
{
    const auto coord_iter = coord_frac_map.find(atom);
    if (coord_iter == coord_frac_map.end())
    {
        throw std::runtime_error("Missing fractional coordinate for ABACUS reciprocal-gauge phase");
    }

    const Vector3_Order<double> tau =
        restrict_fractional_coordinate({coord_iter->second[0],
                                        coord_iter->second[1],
                                        coord_iter->second[2]});
    const double phase_arg =
        TWO_PI * (static_cast<double>(k_shift.x) * tau.x
                  + static_cast<double>(k_shift.y) * tau.y
                  + static_cast<double>(k_shift.z) * tau.z);
    return std::complex<double>(std::cos(phase_arg), std::sin(phase_arg));
}

} // namespace

input_symmetry_atom_block_matrix_map_t rotate_input_symmetry_abf_kspace_operator_blocks(
    const InputSymmetryContext& ctx,
    const InputSymmetryKStarMember& member,
    const input_symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac_map,
    const bool use_time_reversal,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs,
    const Vector3_Order<double>* k_bz_target)
{
    if (!ctx.has_abf_shell_layout())
    {
        throw std::runtime_error("ABF shell layout is required before rotating ABACUS k-space operators");
    }

    auto get_block_or_hermitian = [&blocks_ibz](const atom_t atom_i, const atom_t atom_j) {
        const auto atom_i_iter = blocks_ibz.find(atom_i);
        if (atom_i_iter != blocks_ibz.end())
        {
            const auto atom_j_iter = atom_i_iter->second.find(atom_j);
            if (atom_j_iter != atom_i_iter->second.end())
            {
                return atom_j_iter->second;
            }
        }

        const auto atom_j_iter = blocks_ibz.find(atom_j);
        if (atom_j_iter != blocks_ibz.end())
        {
            const auto atom_i_iter_fallback = atom_j_iter->second.find(atom_i);
            if (atom_i_iter_fallback != atom_j_iter->second.end())
            {
                return transpose(atom_i_iter_fallback->second, true);
            }
        }

        throw std::runtime_error("Missing ABF atom block while rotating the ABACUS q-space operator");
    };
    // The identity member at the IBZ representative is an exact no-op as long as no
    // target-representative gauge shift is requested. Returning the original/hermitian-completed
    // blocks here avoids rebuilding an M matrix that should mathematically be the identity.
    if (!use_time_reversal && member.isym == 0 && k_bz_target == nullptr
        && nearly_same_kpoint(member.k_bz, k_ibz))
    {
        input_symmetry_atom_block_matrix_map_t identity_blocks;
        for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
        {
            for (std::size_t atom_j = 0; atom_j < atom_nabf.size(); ++atom_j)
            {
                if (target_atom_pairs != nullptr
                    && target_atom_pairs->count({static_cast<atom_t>(atom_i),
                                                 static_cast<atom_t>(atom_j)}) == 0)
                {
                    continue;
                }
                identity_blocks[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                    get_block_or_hermitian(static_cast<atom_t>(atom_i),
                                           static_cast<atom_t>(atom_j));
            }
        }
        return identity_blocks;
    }

    std::vector<const InputSymmetryKAtomRotation*> rotations_by_from(atom_nabf.size(), nullptr);
    std::vector<bool> visited_to(atom_nabf.size(), false);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from < 0 || atom_rotation.atom_from >= static_cast<int>(atom_nabf.size())
            || atom_rotation.atom_to < 0 || atom_rotation.atom_to >= static_cast<int>(atom_nabf.size()))
        {
            throw std::runtime_error("ABACUS k-space atom mapping is out of range for ABF rotation");
        }
        rotations_by_from[static_cast<std::size_t>(atom_rotation.atom_from)] = &atom_rotation;
        visited_to[static_cast<std::size_t>(atom_rotation.atom_to)] = true;
    }
    for (std::size_t atom = 0; atom < atom_nabf.size(); ++atom)
    {
        if (rotations_by_from[atom] == nullptr)
        {
            throw std::runtime_error("ABACUS k-space ABF rotations do not cover every atom");
        }
        if (!visited_to[atom])
        {
            throw std::runtime_error("ABACUS k-space ABF atom mapping is not a full permutation");
        }
    }

    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const int spatial_isym = use_time_reversal ? member.isym - nsym_space : member.isym;
    if (spatial_isym < 0 || spatial_isym >= nsym_space)
    {
        throw std::runtime_error("ABACUS q-space operator rotation uses an invalid symmetry index");
    }
    const auto& direct_rotation =
        ctx.rspace_operations.at(static_cast<std::size_t>(spatial_isym)).rotation;
    const Vector3_Order<double> delta_k{k_ibz.x - member.k_bz.x,
                                        k_ibz.y - member.k_bz.y,
                                        k_ibz.z - member.k_bz.z};

    std::vector<ComplexMatrix> atom_M_blocks(atom_nabf.size());
    for (std::size_t atom = 0; atom < atom_nabf.size(); ++atom)
    {
        const auto* atom_rotation = rotations_by_from[atom];
        atom_M_blocks[atom] = build_input_symmetry_abf_rotation_matrix(
            ctx, atom_rotation->atom_type, static_cast<int>(atom_nabf.at(atom)),
            atom_rotation->shell_rotations, direct_rotation);
        const auto return_lattice =
            build_input_symmetry_kspace_return_lattice(ctx, *atom_rotation, coord_frac_map, spatial_isym);
        const double phase_arg =
            TWO_PI * (delta_k.x * static_cast<double>(return_lattice.x)
                      + delta_k.y * static_cast<double>(return_lattice.y)
                      + delta_k.z * static_cast<double>(return_lattice.z));
        atom_M_blocks[atom] *= std::complex<double>(std::cos(phase_arg), std::sin(phase_arg));
    }

    const bool apply_target_gauge = (k_bz_target != nullptr);
    std::vector<std::complex<double>> atom_target_phases(atom_nabf.size(), {1.0, 0.0});
    if (apply_target_gauge)
    {
        const auto k_shift = build_input_symmetry_equivalent_kpoint_shift(member.k_bz, *k_bz_target);
        for (std::size_t atom = 0; atom < atom_nabf.size(); ++atom)
        {
            atom_target_phases[atom] = build_input_symmetry_reciprocal_gauge_phase(
                k_shift, static_cast<atom_t>(atom), coord_frac_map);
        }
    }

    input_symmetry_atom_block_matrix_map_t rotated_blocks;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        const auto* rot_i = rotations_by_from[atom_i];
        const auto& M_i = atom_M_blocks[atom_i];
        for (std::size_t atom_j = 0; atom_j < atom_nabf.size(); ++atom_j)
        {
            if (target_atom_pairs != nullptr
                && target_atom_pairs->count({static_cast<atom_t>(atom_i),
                                             static_cast<atom_t>(atom_j)}) == 0)
            {
                continue;
            }
            const auto* rot_j = rotations_by_from[atom_j];
            const auto& M_j = atom_M_blocks[atom_j];
            const auto target_i = static_cast<atom_t>(atom_i);
            const auto target_j = static_cast<atom_t>(atom_j);
            const auto source_i = static_cast<atom_t>(rot_i->atom_to);
            const auto source_j = static_cast<atom_t>(rot_j->atom_to);
            ComplexMatrix block_ibz;
            try
            {
                block_ibz = get_block_or_hermitian(source_i, source_j);
            }
            catch (const std::exception&)
            {
                std::ostringstream oss;
                oss << "Missing ABF atom block while rotating the ABACUS q-space operator: "
                    << "target_pair=(" << target_i << "," << target_j << "), "
                    << "source_pair=(" << source_i << "," << source_j << "), "
                    << "member_isym=" << member.isym << ", "
                    << "use_time_reversal=" << (use_time_reversal ? "true" : "false");
                throw std::runtime_error(oss.str());
            }

            if (block_ibz.nr != static_cast<int>(atom_nabf.at(source_i))
                || block_ibz.nc != static_cast<int>(atom_nabf.at(source_j)))
            {
                throw std::runtime_error(
                    "The ABF atom block dimension is incompatible with the rotated source atom pair");
            }

            ComplexMatrix block_rotated;
            if (use_time_reversal)
            {
                // Use the same row-major convention as the AO-side k-space rotation:
                //   TRS: O_bz[I,J] = M_I^dagger · conj(O_ibz[S(I),S(J)]) · M_J
                block_rotated = transpose(M_i, true) * conj(block_ibz) * M_j;
            }
            else
            {
                // Row-major equivalent of the ABACUS col-major rotation:
                //   non-TRS: O_bz[I,J] = M_I^T · O_ibz[S(I),S(J)] · conj(M_J)
                block_rotated = transpose(M_i, false) * block_ibz * conj(M_j);
            }
            if (apply_target_gauge)
            {
                const auto left_phase = atom_target_phases[atom_i];
                const auto right_phase = atom_target_phases[atom_j];
                block_rotated *= left_phase * std::conj(right_phase);
            }
            rotated_blocks[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                std::move(block_rotated);
        }
    }

    return rotated_blocks;
}

const InputSymmetryKStarMember& find_matching_abf_kstar_member(const InputSymmetryKStar& abf_star,
                                                        const InputSymmetryKStarMember& ao_member)
{
    const auto matched = std::find_if(
        abf_star.members.begin(), abf_star.members.end(),
        [&ao_member](const InputSymmetryKStarMember& candidate) {
            return candidate.isym == ao_member.isym
                   && nearly_same_kpoint(candidate.k_bz, ao_member.k_bz);
        });
    if (matched == abf_star.members.end())
    {
        throw std::runtime_error(
            "Failed to match an ABF k-star member with the AO-side symmetry member");
    }
    return *matched;
}

input_symmetry_atom_block_matrix_map_t symmetrize_input_symmetry_abf_ibz_kspace_operator_blocks(
    const InputSymmetryContext& ctx,
    const Vector3_Order<double>& k_ibz,
    const input_symmetry_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const InputSymmetryKStar* abf_star,
    const std::set<std::pair<atom_t, atom_t>>* target_atom_pairs)
{
    if (!ctx.has_abf_shell_layout())
    {
        throw std::runtime_error("ABF shell layout is required before symmetrizing ABACUS q-space operators");
    }

    std::set<std::pair<atom_t, atom_t>> inferred_target_pairs;
    if (target_atom_pairs == nullptr)
    {
        for (const auto& atom_i_pair : blocks_ibz)
        {
            for (const auto& atom_j_pair : atom_i_pair.second)
            {
                inferred_target_pairs.insert({atom_i_pair.first, atom_j_pair.first});
            }
        }
        target_atom_pairs = &inferred_target_pairs;
    }

    if (target_atom_pairs->empty())
    {
        return blocks_ibz;
    }

    const auto& star = find_input_symmetry_kstar_for_ibz_kpoint(ctx, k_ibz);
    input_symmetry_atom_block_matrix_map_t accumulated_blocks;
    int n_members_used = 0;
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    for (const auto& member : star.members)
    {
        if (member.isym < 0 || member.isym >= nsym_space)
        {
            continue;
        }
        if (!nearly_same_kpoint(member.k_bz, k_ibz))
        {
            continue;
        }

        const auto& abf_member =
            (abf_star == nullptr) ? member : find_matching_abf_kstar_member(*abf_star, member);
        // Little-group members can return an equivalent IBZ representative that differs from the
        // active LibRPA label by a reciprocal-lattice vector G. Re-apply the target gauge so the
        // averaged operator is accumulated in the same k_ibz representative used by LibRPA.
        const auto rotated_blocks = rotate_input_symmetry_abf_kspace_operator_blocks(
            ctx, abf_member, blocks_ibz, atom_nabf, k_ibz, coord_frac, false, target_atom_pairs,
            &k_ibz);

        for (const auto& atom_i_pair : rotated_blocks)
        {
            for (const auto& atom_j_pair : atom_i_pair.second)
            {
                auto& block = accumulated_blocks[atom_i_pair.first][atom_j_pair.first];
                if (block.nr == 0 && block.nc == 0)
                {
                    block = atom_j_pair.second;
                }
                else
                {
                    block += atom_j_pair.second;
                }
            }
        }
        ++n_members_used;
    }

    if (n_members_used == 0)
    {
        return blocks_ibz;
    }

    const std::complex<double> inv_count(1.0 / static_cast<double>(n_members_used), 0.0);
    for (auto& atom_i_pair : accumulated_blocks)
    {
        for (auto& atom_j_pair : atom_i_pair.second)
        {
            atom_j_pair.second *= inv_count;
        }
    }
    return accumulated_blocks;
}

ComplexMatrix rotate_input_symmetry_abf_kspace_operator_matrix(
    const InputSymmetryContext& ctx,
    const InputSymmetryKStarMember& member,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const Vector3_Order<double>& k_ibz,
    const std::map<atom_t, std::array<double, 3>>& coord_frac_map,
    const bool use_time_reversal,
    const Vector3_Order<double>* k_bz_target)
{
    if (!ctx.has_abf_shell_layout())
    {
        throw std::runtime_error("ABF shell layout is required before rotating ABACUS k-space operators");
    }

    const auto offsets = build_atom_offsets(atom_nabf);
    const int nabf_total = offsets.back();
    if (matrix_ibz.nr != nabf_total || matrix_ibz.nc != nabf_total)
    {
        throw std::runtime_error("The input matrix dimension is incompatible with the ABF basis layout");
    }

    input_symmetry_atom_block_matrix_map_t blocks_ibz;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        for (std::size_t atom_j = 0; atom_j < atom_nabf.size(); ++atom_j)
        {
            blocks_ibz[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                extract_atom_block(matrix_ibz, static_cast<atom_t>(atom_i), static_cast<atom_t>(atom_j),
                                   atom_nabf, offsets);
        }
    }

    const auto rotated_blocks = rotate_input_symmetry_abf_kspace_operator_blocks(
        ctx, member, blocks_ibz, atom_nabf, k_ibz, coord_frac_map, use_time_reversal,
        nullptr, k_bz_target);

    ComplexMatrix rotated_matrix(nabf_total, nabf_total);
    for (const auto& atom_i_pair : rotated_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            set_atom_block(rotated_matrix, atom_i_pair.first, atom_j_pair.first, atom_j_pair.second,
                           offsets);
        }
    }
    return rotated_matrix;
}

ComplexMatrix symmetrize_input_symmetry_abf_ibz_kspace_operator_matrix(
    const InputSymmetryContext& ctx,
    const Vector3_Order<double>& k_ibz,
    const ComplexMatrix& matrix_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::map<atom_t, std::array<double, 3>>& coord_frac)
{
    if (!ctx.has_abf_shell_layout())
    {
        throw std::runtime_error("ABF shell layout is required before symmetrizing ABACUS q-space operators");
    }

    const auto offsets = build_atom_offsets(atom_nabf);
    input_symmetry_atom_block_matrix_map_t blocks_ibz;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        for (std::size_t atom_j = 0; atom_j < atom_nabf.size(); ++atom_j)
        {
            blocks_ibz[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                extract_atom_block(matrix_ibz, static_cast<atom_t>(atom_i), static_cast<atom_t>(atom_j),
                                   atom_nabf, offsets);
        }
    }

    const auto rotated_blocks = symmetrize_input_symmetry_abf_ibz_kspace_operator_blocks(
        ctx, k_ibz, blocks_ibz, atom_nabf, coord_frac);

    ComplexMatrix accumulated(matrix_ibz.nr, matrix_ibz.nc);
    for (const auto& atom_i_pair : rotated_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            set_atom_block(accumulated, atom_i_pair.first, atom_j_pair.first, atom_j_pair.second,
                           offsets);
        }
    }
    return accumulated;
}

ComplexMatrix rotate_input_symmetry_kspace_matrix(const InputSymmetryContext& ctx,
                                          const InputSymmetryKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const std::map<atom_t, std::array<double, 3>>& coord_frac_map,
                                          const bool use_time_reversal,
                                          const Vector3_Order<double>* k_bz_target)
{
    // -------------------------------------------------------------------------
    // Rotate D(k_ibz) to D(k_bz) using the Bloch rotation matrix M from
    // symrot_k.txt.
    //
    // Important: ABACUS prints the sidecar Bloch phase with k_bz, while the
    // internal restore_dm() path uses k_ibz when constructing M(R, k). We must
    // therefore rebuild the internal matrix by multiplying the exported block
    // with exp[i (k_ibz - k_bz) · O], where O is the atom-resolved return
    // lattice. This is the AO counterpart of the already validated ABF-side
    // phase correction.
    //
    // ABACUS col-major:  D^T(k_bz) = M† · D^T(k_ibz) · M
    // Row-major:         D(k_bz)   = M^T · D(k_ibz) · M*
    //
    // Block formula (M_I = M[S(I), I]):
    //   non-TRS:  D_bz[I, J] = M_I^T  · D_ibz[S(I), S(J)]  · conj(M_J)
    //   TRS:      D_bz[I, J] = M_I†   · conj(D_ibz[S(I), S(J)]) · M_J
    // -------------------------------------------------------------------------
    if (!ctx.has_ao_shell_layout())
    {
        throw std::runtime_error("AO shell layout is required before rotating ABACUS k-space matrices");
    }

    const auto offsets = build_atom_offsets(atom_nw);
    const int nao_total = offsets.back();
    if (matrix_ibz.nr != nao_total || matrix_ibz.nc != nao_total)
    {
        throw std::runtime_error("The input matrix dimension is incompatible with the AO basis layout");
    }

    // Build atom permutation: rotations_by_from[I] gives the rotation entry for atom I.
    std::vector<const InputSymmetryKAtomRotation*> rotations_by_from(atom_nw.size(), nullptr);
    std::vector<bool> visited_to(atom_nw.size(), false);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from < 0 || atom_rotation.atom_from >= static_cast<int>(atom_nw.size())
            || atom_rotation.atom_to < 0 || atom_rotation.atom_to >= static_cast<int>(atom_nw.size()))
        {
            throw std::runtime_error("ABACUS k-space atom mapping is out of range");
        }
        rotations_by_from[static_cast<std::size_t>(atom_rotation.atom_from)] = &atom_rotation;
        visited_to[static_cast<std::size_t>(atom_rotation.atom_to)] = true;
    }

    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        if (rotations_by_from[atom] == nullptr)
        {
            throw std::runtime_error("ABACUS k-space atom rotations do not cover every atom");
        }
        if (!visited_to[atom])
        {
            throw std::runtime_error("ABACUS k-space atom mapping is not a full permutation");
        }
    }

    // Build the AO rotation blocks M_I for each atom.
    ComplexMatrix rotated_matrix(nao_total, nao_total);

    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const int spatial_isym = use_time_reversal ? member.isym - nsym_space : member.isym;
    if (spatial_isym < 0 || spatial_isym >= nsym_space)
    {
        throw std::runtime_error("ABACUS AO k-space rotation uses an invalid symmetry index");
    }
    const Vector3_Order<double> delta_k{k_ibz.x - member.k_bz.x,
                                        k_ibz.y - member.k_bz.y,
                                        k_ibz.z - member.k_bz.z};

    std::vector<ComplexMatrix> atom_M_blocks(atom_nw.size());
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto* atom_rotation = rotations_by_from[atom];
        atom_M_blocks[atom] = build_input_symmetry_ao_rotation_matrix(ctx,
                                                               atom_rotation->atom_type,
                                                               atom_rotation->shell_rotations);
        const auto return_lattice =
            build_input_symmetry_kspace_return_lattice(ctx, *atom_rotation, coord_frac_map, spatial_isym);
        const double phase_arg =
            TWO_PI * (delta_k.x * static_cast<double>(return_lattice.x)
                      + delta_k.y * static_cast<double>(return_lattice.y)
                      + delta_k.z * static_cast<double>(return_lattice.z));
        atom_M_blocks[atom] *= std::complex<double>(std::cos(phase_arg), std::sin(phase_arg));
    }

    const bool apply_target_gauge = (k_bz_target != nullptr);
    std::vector<std::complex<double>> atom_target_phases(atom_nw.size(), {1.0, 0.0});
    if (apply_target_gauge)
    {
        const auto k_shift = build_input_symmetry_equivalent_kpoint_shift(member.k_bz, *k_bz_target);
        for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
        {
            atom_target_phases[atom] = build_input_symmetry_reciprocal_gauge_phase(
                k_shift, static_cast<atom_t>(atom), coord_frac_map);
        }
    }

    // Apply the block-level rotation formula.
    //
    // ABACUS col-major formula:  D^T(k_bz) = M† · D^T(k_ibz) · M
    // Row-major equivalent:      D(k_bz)   = M^T · D(k_ibz) · M*
    //
    // M[S(I), I] is the internal ABACUS Bloch rotation block reconstructed from
    // the sidecar shell rotation times the return-lattice phase correction.
    //
    // Block formulas (M_I denotes M[S(I), I]):
    //   non-TRS:  D_bz[I, J] = M_I^T  · D_ibz[S(I), S(J)]  · conj(M_J)
    //   TRS:      D_bz[I, J] = M_I†   · D_ibz[S(I), S(J)]* · M_J
    //
    // Source indices: S(I) = atom_to,  destination indices: I = atom_from.
    for (std::size_t atom_i = 0; atom_i < atom_nw.size(); ++atom_i)
    {
        const auto* rot_i = rotations_by_from[atom_i];
        const auto& M_i = atom_M_blocks[atom_i];
        for (std::size_t atom_j = 0; atom_j < atom_nw.size(); ++atom_j)
        {
            const auto* rot_j = rotations_by_from[atom_j];
            const auto& M_j = atom_M_blocks[atom_j];
            // Read from D_ibz at the MAPPED atom positions S(I), S(J)
            const ComplexMatrix block_ibz =
                extract_atom_block(matrix_ibz,
                                   static_cast<atom_t>(rot_i->atom_to),
                                   static_cast<atom_t>(rot_j->atom_to),
                                   atom_nw, offsets);
            ComplexMatrix block_rotated;
            if (use_time_reversal)
            {
                // TRS: D_bz[I,J] = M_I† · conj(D_ibz[S(I),S(J)]) · M_J
                block_rotated = transpose(M_i, true) * conj(block_ibz) * M_j;
            }
            else
            {
                // Space group: D_bz[I,J] = M_I^T · D_ibz[S(I),S(J)] · conj(M_J)
                block_rotated = transpose(M_i, false) * block_ibz * conj(M_j);
            }
            if (apply_target_gauge)
            {
                const auto left_phase = atom_target_phases[atom_i];
                const auto right_phase = atom_target_phases[atom_j];
                block_rotated *= left_phase * std::conj(right_phase);
            }
            // Write to D_bz at the ORIGINAL atom positions I, J
            set_atom_block(rotated_matrix,
                           static_cast<atom_t>(atom_i),
                           static_cast<atom_t>(atom_j),
                           block_rotated,
                           offsets);
        }
    }

    return rotated_matrix;
}

void build_input_symmetry_rspace_sector_stars(const InputSymmetryContext& ctx,
                                      const std::map<atom_t, std::array<double, 3>>& coord_frac,
                                      const Vector3_Order<int>& period,
                                      const std::vector<Vector3_Order<int>>& Rlist,
                                      input_symmetry_rspace_sector_stars_t& sector_stars,
                                      std::ostream* log)
{
    (void)period;
    if (!ctx.available || ctx.irreducible_sector.empty() || ctx.rspace_operations.empty())
    {
        throw std::runtime_error("ABACUS real-space symmetry metadata is incomplete");
    }
    if (ctx.atom_to_type.empty())
    {
        throw std::runtime_error("ABACUS atom-to-type mapping is unavailable for real-space symmetry");
    }

    const auto& rspace_coord_frac =
        (ctx.input_coord_frac.size() == ctx.atom_to_type.size()) ? ctx.input_coord_frac : coord_frac;
    const auto op_infos = build_rspace_operation_info(ctx, rspace_coord_frac);
    const auto inverse_map = build_rspace_inverse_map(ctx, rspace_coord_frac);

    sector_stars.clear();
    std::set<Vector3_Order<int>> Rset(Rlist.begin(), Rlist.end());
    using full_key_t = std::tuple<atom_t, atom_t, Vector3_Order<int>>;
    std::set<full_key_t> covered;

    for (const auto& pair_Rs : ctx.irreducible_sector)
    {
        const atpair_t& ir_pair = pair_Rs.first;
        for (const auto& ir_R_array : pair_Rs.second)
        {
            const Vector3_Order<int> ir_R{ir_R_array[0], ir_R_array[1], ir_R_array[2]};
            auto& star_members = sector_stars[ir_pair][ir_R];
            std::vector<std::string> candidate_debug;
            for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
            {
                const int inv = inverse_map[isym];
                const auto& op_info = op_infos[static_cast<std::size_t>(inv)];
                const auto full_I = op_info.atom_map[ir_pair.first];
                const auto full_J = op_info.atom_map[ir_pair.second];
                const auto full_R = rotate_rspace_vector(ir_R,
                                                         op_info,
                                                         ctx.rspace_operations[static_cast<std::size_t>(inv)],
                                                         ir_pair.first,
                                                         ir_pair.second);
                const bool in_rset = Rset.count(full_R) != 0;
                const full_key_t full_key{full_I, full_J, full_R};
                const bool is_duplicate = covered.count(full_key) != 0;

                std::ostringstream oss;
                oss << "isym=" << isym << " inv=" << inv << " -> (" << full_I << ", " << full_J
                    << "), R=(" << full_R.x << ", " << full_R.y << ", " << full_R.z
                    << "), in_Rset=" << (in_rset ? "true" : "false")
                    << ", duplicate=" << (is_duplicate ? "true" : "false");
                candidate_debug.push_back(oss.str());

                if (!in_rset)
                {
                    continue;
                }

                if (covered.insert(full_key).second)
                {
                    star_members.push_back(
                        {static_cast<int>(isym), {full_I, full_J}, full_R});
                }
            }

            if (star_members.empty())
            {
                std::ostringstream oss;
                oss << "Failed to build a real-space symmetry star from ABACUS sidecars for "
                    << "irreducible pair (" << ir_pair.first << ", " << ir_pair.second << ")"
                    << " and R=(" << ir_R.x << ", " << ir_R.y << ", " << ir_R.z << ")";
                for (const auto& line : candidate_debug)
                {
                    oss << "\n  " << line;
                }
                throw std::runtime_error(oss.str());
            }
        }
    }

    if (log != nullptr)
    {
        std::size_t total_members = 0;
        for (const auto& pair_star : sector_stars)
        {
            for (const auto& R_star : pair_star.second)
            {
                total_members += R_star.second.size();
            }
        }
        (*log) << "| real-space sector stars: " << total_members << " full members restored from "
               << ctx.count_irreducible_blocks() << " irreducible blocks (" << covered.size()
               << " unique full {atom pair, R} blocks)\n";
    }
}

ComplexMatrix rotate_input_symmetry_rspace_matrix(const InputSymmetryContext& ctx,
                                          const int isym,
                                          const atom_t atom_from_i,
                                          const atom_t atom_from_j,
                                          const ComplexMatrix& matrix_source)
{
    if (!ctx.has_ao_shell_layout())
    {
        throw std::runtime_error("AO shell layout is required before rotating ABACUS real-space matrices");
    }
    if (isym < 0 || isym >= static_cast<int>(ctx.rspace_operations.size()))
    {
        throw std::out_of_range("ABACUS real-space symmetry index is out of range");
    }

    const int type_i = ctx.atom_to_type.at(atom_from_i);
    const int type_j = ctx.atom_to_type.at(atom_from_j);
    const auto& op = ctx.rspace_operations[static_cast<std::size_t>(isym)];
    const ComplexMatrix T_i = build_input_symmetry_ao_rotation_matrix(ctx, type_i, op.shell_rotations);
    const ComplexMatrix T_j = build_input_symmetry_ao_rotation_matrix(ctx, type_j, op.shell_rotations);

    if (matrix_source.nr != T_i.nr || matrix_source.nc != T_j.nr)
    {
        throw std::runtime_error("ABACUS real-space rotation has incompatible AO dimensions");
    }

    // Keep the same row-major convention as the AO/ABF k-space operator restore:
    //   H_bz[I,J] = T_I^T · H_ir[S(I),S(J)] · conj(T_J)
    // for ordinary spatial operations without time reversal.
    return transpose(T_i, false) * matrix_source * conj(T_j);
}

ComplexMatrix rotate_input_symmetry_abf_rspace_matrix(const InputSymmetryContext& ctx,
                                              const int isym,
                                              const atom_t atom_from_i,
                                              const atom_t atom_from_j,
                                              const ComplexMatrix& matrix_source)
{
    if (!ctx.has_abf_shell_layout())
    {
        throw std::runtime_error("ABF shell layout is required before rotating ABACUS real-space matrices");
    }
    if (isym < 0 || isym >= static_cast<int>(ctx.rspace_operations.size()))
    {
        throw std::out_of_range("ABACUS real-space symmetry index is out of range");
    }

    const int type_i = ctx.atom_to_type.at(atom_from_i);
    const int type_j = ctx.atom_to_type.at(atom_from_j);
    const auto& op = ctx.rspace_operations[static_cast<std::size_t>(isym)];
    const ComplexMatrix T_i = build_input_symmetry_abf_rotation_matrix(
        ctx, type_i, matrix_source.nr, op.shell_rotations, op.rotation);
    const ComplexMatrix T_j = build_input_symmetry_abf_rotation_matrix(
        ctx, type_j, matrix_source.nc, op.shell_rotations, op.rotation);

    if (matrix_source.nr != T_i.nr || matrix_source.nc != T_j.nr)
    {
        throw std::runtime_error("ABACUS real-space ABF rotation has incompatible dimensions");
    }

    return transpose(T_i, false) * matrix_source * conj(T_j);
}

} // namespace librpa_int
