// Compile the code via
//   c++ -std=c++17 -O2 -o convert_legacy_Cs_v1.exe convert_legacy_Cs_v1.cpp
// Run as
//   convert_legacy_Cs_v1.exe Cs_data_1.txt v1_Cs_data_1.dat
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

constexpr std::int32_t READER_LRICOEF_V1_MARKER = 10267453;
constexpr std::int64_t V1_HEADER_BASE_SIZE =
    3 * static_cast<std::int64_t>(sizeof(std::int32_t)) +
    2 * static_cast<std::int64_t>(sizeof(std::int64_t));
constexpr std::int64_t V1_BLOCK_RECORD_SIZE =
    5 * static_cast<std::int64_t>(sizeof(std::int32_t)) +
    static_cast<std::int64_t>(sizeof(double)) +
    static_cast<std::int64_t>(sizeof(std::int64_t));
constexpr std::size_t COPY_BUFFER_BYTES = 8ULL * 1024ULL * 1024ULL;

struct Options
{
    fs::path input_file;
    fs::path output_file;
    std::string input_format = "auto";
    std::int64_t n_apcell_file_max = -1;
    bool overwrite = false;
    bool quiet = false;
};

struct BlockRecord
{
    std::int32_t ia1 = 0;
    std::int32_t ia2 = 0;
    std::int32_t R[3] = {0, 0, 0};
    std::int32_t n_i = 0;
    std::int32_t n_j = 0;
    std::int32_t n_mu = 0;
    double max_abs = 0.0;
    std::int64_t offset = 0;
    std::int64_t nbytes = 0;
};

struct LegacyScan
{
    std::int32_t natom = 0;
    std::int32_t ncell = 0;
    bool binary = false;
    std::vector<BlockRecord> blocks;
};

std::string usage()
{
    return
        "Usage: convert_legacy_Cs_v1 INPUT_FILE OUTPUT_FILE [options]\n"
        "\n"
        "Converts one legacy LRI coefficient (Cs) file to reader-v1 format.\n"
        "Both legacy text and unformatted binary Cs files are accepted.\n"
        "\n"
        "Options:\n"
        "  -T, --target-reader-version 1\n"
        "                              Accepted for compatibility; only v1 is supported\n"
        "      --input-format FORMAT   auto, text, or binary (default: auto)\n"
        "      --n-apcell-file-max N   Reserved v1 header record count\n"
        "                              (default: number of converted blocks)\n"
        "      --reserved-blocks N     Alias of --n-apcell-file-max\n"
        "      --overwrite            Replace OUTPUT_FILE if it already exists\n"
        "      --quiet                Suppress conversion summary\n"
        "  -h, --help                 Show this message\n";
}

bool starts_with(const std::string &text, const std::string &prefix)
{
    return text.rfind(prefix, 0) == 0;
}

std::string next_value(int &i, int argc, char **argv, const std::string &name)
{
    if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + name);
    return argv[++i];
}

std::int64_t parse_i64(const std::string &text, const std::string &name)
{
    std::size_t pos = 0;
    std::int64_t value = 0;
    try
    {
        value = std::stoll(text, &pos);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error("invalid integer for " + name + ": " + text);
    }
    if (pos != text.size())
        throw std::runtime_error("invalid integer for " + name + ": " + text);
    return value;
}

std::int32_t parse_i32_token(const std::string &text, const std::string &name)
{
    const auto value = parse_i64(text, name);
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
    {
        throw std::runtime_error("integer out of int32 range for " + name + ": " + text);
    }
    return static_cast<std::int32_t>(value);
}

double parse_double_token(std::string text, const std::string &name)
{
    for (auto &ch: text)
        if (ch == 'd' || ch == 'D') ch = 'e';
    std::size_t pos = 0;
    double value = 0.0;
    try
    {
        value = std::stod(text, &pos);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error("invalid floating-point value for " + name + ": " + text);
    }
    if (pos != text.size())
        throw std::runtime_error("invalid floating-point value for " + name + ": " + text);
    return value;
}

Options parse_args(int argc, char **argv)
{
    Options opts;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            std::cout << usage();
            std::exit(0);
        }
        else if (arg == "-T" || arg == "--target-reader-version")
        {
            const auto version = next_value(i, argc, argv, arg);
            if (version != "1")
                throw std::runtime_error("this converter only supports -T 1");
        }
        else if (arg == "--input-format")
        {
            opts.input_format = next_value(i, argc, argv, arg);
            if (opts.input_format != "auto" && opts.input_format != "text" &&
                opts.input_format != "binary")
            {
                throw std::runtime_error("--input-format must be auto, text, or binary");
            }
        }
        else if (arg == "--n-apcell-file-max" || arg == "--reserved-blocks")
        {
            opts.n_apcell_file_max = parse_i64(next_value(i, argc, argv, arg), arg);
            if (opts.n_apcell_file_max < 0)
                throw std::runtime_error(arg + " must be non-negative");
        }
        else if (arg == "--overwrite")
        {
            opts.overwrite = true;
        }
        else if (arg == "--quiet")
        {
            opts.quiet = true;
        }
        else if (starts_with(arg, "-"))
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
        else
        {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 2)
        throw std::runtime_error("expected INPUT_FILE and OUTPUT_FILE arguments\n" + usage());
    opts.input_file = positional[0];
    opts.output_file = positional[1];
    return opts;
}

void read_exact(std::istream &in, void *buffer, std::size_t nbytes,
                const std::string &label)
{
    in.read(static_cast<char *>(buffer), static_cast<std::streamsize>(nbytes));
    if (!in.good())
        throw std::runtime_error("failed to read " + label);
}

void write_exact(std::ostream &out, const void *buffer, std::size_t nbytes,
                 const std::string &label)
{
    out.write(static_cast<const char *>(buffer), static_cast<std::streamsize>(nbytes));
    if (!out.good())
        throw std::runtime_error("failed to write " + label);
}

template <typename T>
void write_scalar(std::ostream &out, const T &value, const std::string &label)
{
    write_exact(out, &value, sizeof(T), label);
}

std::int64_t checked_mul_i64(std::int64_t a, std::int64_t b, const std::string &label)
{
    if (a < 0 || b < 0 ||
        (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a))
    {
        throw std::runtime_error("size overflow while computing " + label);
    }
    return a * b;
}

std::int64_t payload_bytes_for(const std::int32_t n_i, const std::int32_t n_j,
                               const std::int32_t n_mu)
{
    if (n_i <= 0 || n_j <= 0 || n_mu <= 0)
        throw std::runtime_error("invalid Cs block dimensions");
    const auto n_ij = checked_mul_i64(n_i, n_j, "n_i*n_j");
    const auto n_values = checked_mul_i64(n_ij, n_mu, "n_i*n_j*n_mu");
    return checked_mul_i64(n_values, static_cast<std::int64_t>(sizeof(double)),
                           "Cs payload bytes");
}

std::int64_t header_bytes_for(const std::int64_t n_apcell_file_max)
{
    if (n_apcell_file_max < 0)
        throw std::runtime_error("negative n_apcell_file_max");
    const auto table_bytes =
        checked_mul_i64(n_apcell_file_max, V1_BLOCK_RECORD_SIZE, "v1 block table size");
    if (V1_HEADER_BASE_SIZE >
        std::numeric_limits<std::int64_t>::max() - table_bytes)
    {
        throw std::runtime_error("v1 header size overflow");
    }
    return V1_HEADER_BASE_SIZE + table_bytes;
}

void validate_block_indices(const BlockRecord &block, std::int32_t natom,
                            const fs::path &path)
{
    if (block.ia1 <= 0 || block.ia1 > natom ||
        block.ia2 <= 0 || block.ia2 > natom)
    {
        throw std::runtime_error(path.string() + ": atom index out of range in Cs block");
    }
    if (block.n_i <= 0 || block.n_j <= 0 || block.n_mu <= 0)
        throw std::runtime_error(path.string() + ": invalid Cs block dimensions");
}

bool legacy_binary_layout_matches(const fs::path &path)
{
    const auto file_size = fs::file_size(path);
    if (file_size < 3 * sizeof(std::int32_t)) return false;

    std::ifstream in(path, std::ios::binary);
    std::int32_t natom = 0;
    std::int32_t ncell = 0;
    std::int32_t nblocks = 0;
    in.read(reinterpret_cast<char *>(&natom), sizeof(natom));
    in.read(reinterpret_cast<char *>(&ncell), sizeof(ncell));
    in.read(reinterpret_cast<char *>(&nblocks), sizeof(nblocks));
    if (!in.good() || natom <= 0 || ncell < 0 || nblocks < 0) return false;

    std::uintmax_t pos = 3 * sizeof(std::int32_t);
    for (std::int32_t iblock = 0; iblock != nblocks; ++iblock)
    {
        std::int32_t dims[8];
        in.read(reinterpret_cast<char *>(dims), sizeof(dims));
        if (!in.good()) return false;
        if (dims[0] <= 0 || dims[0] > natom || dims[1] <= 0 || dims[1] > natom)
            return false;
        if (dims[5] <= 0 || dims[6] <= 0 || dims[7] <= 0) return false;
        std::int64_t payload = 0;
        try
        {
            payload = payload_bytes_for(dims[5], dims[6], dims[7]);
        }
        catch (const std::exception &)
        {
            return false;
        }
        pos += sizeof(dims) + static_cast<std::uintmax_t>(payload);
        if (pos > file_size) return false;
        in.seekg(static_cast<std::streamoff>(payload), std::ios::cur);
    }
    return pos == file_size;
}

LegacyScan scan_binary_legacy(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.good())
        throw std::runtime_error("failed to open " + path.string());

    LegacyScan scan;
    scan.binary = true;
    std::int32_t nblocks = 0;
    read_exact(in, &scan.natom, sizeof(scan.natom), "binary Cs natom");
    read_exact(in, &scan.ncell, sizeof(scan.ncell), "binary Cs ncell");
    read_exact(in, &nblocks, sizeof(nblocks), "binary Cs n_apcell_file");
    if (scan.natom <= 0 || scan.ncell < 0 || nblocks < 0)
        throw std::runtime_error(path.string() + ": invalid binary Cs header");

    scan.blocks.reserve(static_cast<std::size_t>(nblocks));
    std::vector<double> buffer(COPY_BUFFER_BYTES / sizeof(double));
    if (buffer.empty()) buffer.resize(1);
    for (std::int32_t iblock = 0; iblock != nblocks; ++iblock)
    {
        std::int32_t dims[8];
        read_exact(in, dims, sizeof(dims), "binary Cs block header");
        BlockRecord block;
        block.ia1 = dims[0];
        block.ia2 = dims[1];
        block.R[0] = dims[2];
        block.R[1] = dims[3];
        block.R[2] = dims[4];
        block.n_i = dims[5];
        block.n_j = dims[6];
        block.n_mu = dims[7];
        validate_block_indices(block, scan.natom, path);
        block.nbytes = payload_bytes_for(block.n_i, block.n_j, block.n_mu);

        std::int64_t bytes_left = block.nbytes;
        while (bytes_left > 0)
        {
            const auto chunk_bytes =
                std::min<std::int64_t>(bytes_left,
                                       static_cast<std::int64_t>(buffer.size() *
                                                                 sizeof(double)));
            if (chunk_bytes % static_cast<std::int64_t>(sizeof(double)) != 0)
                throw std::runtime_error("internal error: non-double Cs chunk");
            read_exact(in, buffer.data(), static_cast<std::size_t>(chunk_bytes),
                       "binary Cs payload");
            const auto nvalues = chunk_bytes / static_cast<std::int64_t>(sizeof(double));
            for (std::int64_t i = 0; i != nvalues; ++i)
                block.max_abs = std::max(block.max_abs, std::abs(buffer[i]));
            bytes_left -= chunk_bytes;
        }
        scan.blocks.push_back(block);
    }
    return scan;
}

void require_token(std::istream &in, std::string &token, const fs::path &path,
                   const std::string &label)
{
    in >> token;
    if (!in.good())
        throw std::runtime_error(path.string() + ": unexpected EOF while reading " + label);
}

LegacyScan scan_text_legacy(const fs::path &path)
{
    std::ifstream in(path);
    if (!in.good())
        throw std::runtime_error("failed to open " + path.string());

    LegacyScan scan;
    scan.binary = false;
    std::string token;
    require_token(in, token, path, "text Cs natom");
    scan.natom = parse_i32_token(token, "text Cs natom");
    require_token(in, token, path, "text Cs ncell");
    scan.ncell = parse_i32_token(token, "text Cs ncell");
    if (scan.natom <= 0 || scan.ncell < 0)
        throw std::runtime_error(path.string() + ": invalid text Cs header");

    while (in >> token)
    {
        BlockRecord block;
        block.ia1 = parse_i32_token(token, "Cs ia1");
        require_token(in, token, path, "Cs ia2");
        block.ia2 = parse_i32_token(token, "Cs ia2");
        for (int idim = 0; idim != 3; ++idim)
        {
            require_token(in, token, path, "Cs R");
            block.R[idim] = parse_i32_token(token, "Cs R");
        }
        require_token(in, token, path, "Cs n_i");
        block.n_i = parse_i32_token(token, "Cs n_i");
        require_token(in, token, path, "Cs n_j");
        block.n_j = parse_i32_token(token, "Cs n_j");
        require_token(in, token, path, "Cs n_mu");
        block.n_mu = parse_i32_token(token, "Cs n_mu");
        validate_block_indices(block, scan.natom, path);
        block.nbytes = payload_bytes_for(block.n_i, block.n_j, block.n_mu);

        const auto nvalues = block.nbytes / static_cast<std::int64_t>(sizeof(double));
        for (std::int64_t i = 0; i != nvalues; ++i)
        {
            require_token(in, token, path, "Cs payload");
            const auto value = parse_double_token(token, "Cs payload");
            block.max_abs = std::max(block.max_abs, std::abs(value));
        }
        scan.blocks.push_back(block);
    }
    return scan;
}

LegacyScan scan_legacy_input(const Options &opts)
{
    if (!fs::exists(opts.input_file))
        throw std::runtime_error("input file does not exist: " + opts.input_file.string());
    if (!fs::is_regular_file(opts.input_file))
        throw std::runtime_error("input path is not a regular file: " +
                                 opts.input_file.string());

    if (opts.input_format == "binary") return scan_binary_legacy(opts.input_file);
    if (opts.input_format == "text") return scan_text_legacy(opts.input_file);
    if (legacy_binary_layout_matches(opts.input_file)) return scan_binary_legacy(opts.input_file);
    return scan_text_legacy(opts.input_file);
}

void assign_offsets(LegacyScan &scan, std::int64_t n_apcell_file_max)
{
    auto offset = header_bytes_for(n_apcell_file_max);
    for (auto &block: scan.blocks)
    {
        block.offset = offset;
        if (offset > std::numeric_limits<std::int64_t>::max() - block.nbytes)
            throw std::runtime_error("output file size overflow");
        offset += block.nbytes;
    }
}

void write_v1_record(std::ostream &out, const BlockRecord &block)
{
    write_scalar(out, block.ia1, "v1 Cs ia1");
    write_scalar(out, block.ia2, "v1 Cs ia2");
    write_scalar(out, block.R[0], "v1 Cs R0");
    write_scalar(out, block.R[1], "v1 Cs R1");
    write_scalar(out, block.R[2], "v1 Cs R2");
    write_scalar(out, block.max_abs, "v1 Cs max_abs");
    write_scalar(out, block.offset, "v1 Cs offset");
}

void write_v1_header(std::ostream &out, const LegacyScan &scan,
                     std::int64_t n_apcell_file_max)
{
    write_scalar(out, READER_LRICOEF_V1_MARKER, "v1 Cs marker");
    write_scalar(out, scan.natom, "v1 Cs natom");
    write_scalar(out, scan.ncell, "v1 Cs ncell");
    const auto n_apcell_file = static_cast<std::int64_t>(scan.blocks.size());
    write_scalar(out, n_apcell_file, "v1 Cs n_apcell_file");
    write_scalar(out, n_apcell_file_max, "v1 Cs n_apcell_file_max");
    for (const auto &block: scan.blocks) write_v1_record(out, block);

    const char zero_record[V1_BLOCK_RECORD_SIZE] = {};
    for (std::int64_t i = n_apcell_file; i != n_apcell_file_max; ++i)
    {
        write_exact(out, zero_record, sizeof(zero_record), "v1 Cs zero padding record");
    }
}

void compare_block_header(const BlockRecord &block, const std::int32_t dims[8],
                          const fs::path &path)
{
    if (block.ia1 != dims[0] || block.ia2 != dims[1] ||
        block.R[0] != dims[2] || block.R[1] != dims[3] || block.R[2] != dims[4] ||
        block.n_i != dims[5] || block.n_j != dims[6] || block.n_mu != dims[7])
    {
        throw std::runtime_error(path.string() +
                                 ": legacy binary Cs changed between scan and copy");
    }
}

void copy_binary_payloads(const Options &opts, const LegacyScan &scan, std::ostream &out)
{
    std::ifstream in(opts.input_file, std::ios::binary);
    if (!in.good())
        throw std::runtime_error("failed to reopen " + opts.input_file.string());
    std::int32_t header[3];
    read_exact(in, header, sizeof(header), "binary Cs header");

    std::vector<char> buffer(COPY_BUFFER_BYTES);
    if (buffer.empty()) buffer.resize(1);
    for (const auto &block: scan.blocks)
    {
        std::int32_t dims[8];
        read_exact(in, dims, sizeof(dims), "binary Cs block header");
        compare_block_header(block, dims, opts.input_file);
        std::int64_t bytes_left = block.nbytes;
        while (bytes_left > 0)
        {
            const auto chunk_bytes =
                std::min<std::int64_t>(bytes_left,
                                       static_cast<std::int64_t>(buffer.size()));
            read_exact(in, buffer.data(), static_cast<std::size_t>(chunk_bytes),
                       "binary Cs payload");
            write_exact(out, buffer.data(), static_cast<std::size_t>(chunk_bytes),
                        "v1 Cs payload");
            bytes_left -= chunk_bytes;
        }
    }
}

void copy_text_payloads(const Options &opts, const LegacyScan &scan, std::ostream &out)
{
    std::ifstream in(opts.input_file);
    if (!in.good())
        throw std::runtime_error("failed to reopen " + opts.input_file.string());

    std::string token;
    require_token(in, token, opts.input_file, "text Cs natom");
    require_token(in, token, opts.input_file, "text Cs ncell");

    std::vector<double> buffer(COPY_BUFFER_BYTES / sizeof(double));
    if (buffer.empty()) buffer.resize(1);
    for (const auto &block: scan.blocks)
    {
        std::int32_t dims[8];
        for (auto &dim: dims)
        {
            require_token(in, token, opts.input_file, "text Cs block header");
            dim = parse_i32_token(token, "text Cs block header");
        }
        compare_block_header(block, dims, opts.input_file);

        std::int64_t nvalues_left =
            block.nbytes / static_cast<std::int64_t>(sizeof(double));
        while (nvalues_left > 0)
        {
            const auto chunk_values =
                std::min<std::int64_t>(nvalues_left,
                                       static_cast<std::int64_t>(buffer.size()));
            for (std::int64_t i = 0; i != chunk_values; ++i)
            {
                require_token(in, token, opts.input_file, "text Cs payload");
                buffer[static_cast<std::size_t>(i)] =
                    parse_double_token(token, "text Cs payload");
            }
            write_exact(out, buffer.data(),
                        static_cast<std::size_t>(chunk_values) * sizeof(double),
                        "v1 Cs payload");
            nvalues_left -= chunk_values;
        }
    }
}

void check_output_path(const Options &opts)
{
    const auto input_abs = fs::weakly_canonical(opts.input_file);
    const auto output_abs = fs::absolute(opts.output_file).lexically_normal();
    if (input_abs == output_abs ||
        (fs::exists(opts.output_file) && fs::equivalent(opts.input_file, opts.output_file)))
    {
        throw std::runtime_error("input and output files must be different");
    }
    if (fs::exists(opts.output_file) && !opts.overwrite)
    {
        throw std::runtime_error(opts.output_file.string() +
                                 " already exists; use --overwrite to replace it");
    }
    const auto parent = opts.output_file.parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

void convert(const Options &opts)
{
    auto scan = scan_legacy_input(opts);
    auto n_apcell_file_max = opts.n_apcell_file_max;
    if (n_apcell_file_max < 0)
        n_apcell_file_max = static_cast<std::int64_t>(scan.blocks.size());
    if (n_apcell_file_max < static_cast<std::int64_t>(scan.blocks.size()))
    {
        throw std::runtime_error("n_apcell_file_max is smaller than the number of Cs blocks");
    }
    assign_offsets(scan, n_apcell_file_max);
    check_output_path(opts);

    std::ofstream out(opts.output_file, std::ios::binary | std::ios::trunc);
    if (!out.good())
        throw std::runtime_error("failed to create " + opts.output_file.string() +
                                 ": " + std::strerror(errno));

    write_v1_header(out, scan, n_apcell_file_max);
    if (scan.binary)
        copy_binary_payloads(opts, scan, out);
    else
        copy_text_payloads(opts, scan, out);
    out.close();
    if (!out.good())
        throw std::runtime_error("failed to close " + opts.output_file.string());

    if (!opts.quiet)
    {
        std::int64_t payload_bytes = 0;
        for (const auto &block: scan.blocks) payload_bytes += block.nbytes;
        std::cout << "Converted " << scan.blocks.size() << " legacy Cs block(s) from "
                  << (scan.binary ? "binary" : "text") << " to reader v1." << std::endl;
        std::cout << "natom=" << scan.natom
                  << ", ncell=" << scan.ncell
                  << ", n_apcell_file_max=" << n_apcell_file_max
                  << ", header_bytes=" << header_bytes_for(n_apcell_file_max)
                  << ", payload_bytes=" << payload_bytes << std::endl;
    }
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        convert(parse_args(argc, argv));
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "error: " << exc.what() << std::endl;
        return 1;
    }
}
