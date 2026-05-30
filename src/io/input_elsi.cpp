#include "input_elsi.h"

#include <iostream>
#include <regex>

namespace librpa_int
{

namespace
{

std::string get_file_name(const std::string& file_path)
{
    const auto pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos) return file_path;
    return file_path.substr(pos + 1);
}

bool parse_regular_csc_key(const std::string& file_name, std::string& key)
{
    static const std::regex pattern(
        R"(^([[:alnum:]_]+)_spin_([0-9]+)_kpt_([0-9]{6})(?:_freq_([0-9]+))?\.csc$)");

    std::smatch match;
    if (!std::regex_match(file_name, match, pattern)) return false;

    key = match[1].str() + "_spin_" + match[2].str() + "_kpt_" + match[3].str();
    if (match[4].matched) key += "_freq_" + match[4].str();
    return true;
}

bool parse_band_vxc_csc_key(const std::string& file_name, std::string& key)
{
    static const std::regex pattern(R"(^band_vxc_mat_spin_([0-9]+)_k_([0-9]{5})\.csc$)");

    std::smatch match;
    if (!std::regex_match(file_name, match, pattern)) return false;

    key = "band_vxc_mat_spin_" + match[1].str() + "_k_" + match[2].str();
    return true;
}

bool parse_csc_key(const std::string& file_path, std::string& key)
{
    const auto file_name = get_file_name(file_path);
    return parse_band_vxc_csc_key(file_name, key) || parse_regular_csc_key(file_name, key);
}

}  // namespace

// Read ELSI CSC file into standard CSC format
bool read_elsi_to_csc(const std::string& file_path, std::vector<int>& col_ptr,
                      std::vector<int>& row_idx, std::vector<double>& nnz_val,
                      std::vector<std::complex<double>>& nnz_val_cplx, int& n_basis,
                      const bool force_cplx)
{
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs)
        throw LIBRPA_RUNTIME_ERROR("Cannot open file " + file_path);

    bool is_complex = true;

    // Read file content
    ifs.seekg(0, std::ios::end);
    std::streampos size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    ifs.read(buffer.data(), size);
    ifs.close();

    // parse header
    int64_t header[16];
    std::memcpy(header, buffer.data(), 128);

    n_basis = header[3];
    int64_t nnz = header[5];

    // column
    int64_t* col_ptr_raw = reinterpret_cast<int64_t*>(buffer.data() + 128);
    col_ptr.assign(col_ptr_raw, col_ptr_raw + n_basis);
    col_ptr.push_back(nnz + 1);

    // row indices
    int32_t* row_idx_raw = reinterpret_cast<int32_t*>(buffer.data() + 128 + n_basis * 8);
    row_idx.assign(row_idx_raw, row_idx_raw + nnz);

    // non-zero values
    char* nnz_val_raw = buffer.data() + 128 + n_basis * 8 + nnz * 4;
    if (header[2] == 0)
    {
        double* nnz_ptr = reinterpret_cast<double*>(nnz_val_raw);
        if (force_cplx)
        {
            is_complex = true;
            nnz_val_cplx.resize(nnz);
            for (int64_t i = 0; i < nnz; ++i)
            {
                nnz_val_cplx[i] = std::complex<double>(nnz_ptr[i], 0.0);
            }
        }
        else
        {
            is_complex = false;
            nnz_val.assign(nnz_ptr, nnz_ptr + nnz);
        }
    }
    else
    {
        is_complex = true;
        auto *nnz_ptr_cplx = reinterpret_cast<std::complex<double> *>(nnz_val_raw);
        nnz_val_cplx.assign(nnz_ptr_cplx, nnz_ptr_cplx + nnz);
    }

    // Convert indices to 0-based
    for (int32_t& idx : row_idx) {
        idx -= 1;
    }
    for (int32_t& ptr : col_ptr) {
        ptr -= 1;
    }

    return is_complex;
}

Matz load_matrix_cplx(const std::string& file_path, const MAJOR major)
{
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
    std::vector<double> nnz_val;
    std::vector<std::complex<double>> nnz_val_cplx;
    int n_basis;
    bool is_cplx;

    is_cplx = read_elsi_to_csc(file_path, col_ptr, row_idx, nnz_val, nnz_val_cplx, n_basis, true);
    if (is_cplx)
        return load_csc_to_matrix(n_basis, col_ptr, row_idx, nnz_val_cplx, major);
    return load_csc_to_matrix(n_basis, col_ptr, row_idx, nnz_val, major).to_complex();
}

Matd load_matrix_real(const std::string& file_path, const MAJOR major)
{
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
    std::vector<double> nnz_val;
    std::vector<std::complex<double>> nnz_val_cplx;
    int n_basis;
    bool is_cplx;

    is_cplx = read_elsi_to_csc(file_path, col_ptr, row_idx, nnz_val, nnz_val_cplx, n_basis, true);
    if (is_cplx)
        return load_csc_to_matrix(n_basis, col_ptr, row_idx, nnz_val_cplx, major).get_real();
    return load_csc_to_matrix(n_basis, col_ptr, row_idx, nnz_val, major);
}

bool convert_csc(const std::string& filePath, std::map<std::string, Matz>& matrices,
                 std::string& key, const MAJOR major)
{
    if (!parse_csc_key(filePath, key))
    {
        std::cerr << "Failed to parse CSC file name: " << filePath << std::endl;
        return false;
    }

    try
    {
        matrices[key] = load_matrix_cplx(filePath, major);
        std::cout << "Matrix loaded and stored successfully under key: " << key << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load matrix from file: " << filePath << " Error: " << e.what()
                  << std::endl;
        return false;
    }

    return true;
}

}
