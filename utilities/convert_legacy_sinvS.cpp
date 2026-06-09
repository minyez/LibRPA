// Compile the code via
//   c++ -std=c++17 -O2 -o convert_legacy_sinvS.exe convert_legacy_sinvS.cpp
// Run as
//   convert_legacy_sinvS.exe shrink_sinvS_0.txt v1_shrink_sinvS_0.txt
#include <cstdint>
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

constexpr std::int32_t READER_SHRINK_SINVS_V1_MARKER = -30241621;
constexpr std::int64_t HEADER_BASE_SIZE = 2 * static_cast<std::int64_t>(sizeof(std::int32_t));
constexpr std::int64_t RECORD_SIZE =
    7 * static_cast<std::int64_t>(sizeof(std::int32_t)) +
    static_cast<std::int64_t>(sizeof(double)) +
    static_cast<std::int64_t>(sizeof(std::int64_t));

struct Options
{
    fs::path input_file;
    fs::path output_file;
    bool overwrite = false;
};

struct Record
{
    std::int32_t iq = 0;
    std::int32_t nrow_total = 0;
    std::int32_t ncol_total = 0;
    std::int32_t begin_row = 0;
    std::int32_t end_row = 0;
    std::int32_t begin_col = 0;
    std::int32_t end_col = 0;
    double q_weight = 0.0;
    std::int64_t offset = 0;
    std::vector<double> payload;
};

std::string usage()
{
    return
        "Usage: convert_legacy_sinvS INPUT_FILE OUTPUT_FILE [options]\n"
        "\n"
        "Converts one legacy ABACUS shrink_sinvS text file to reader-v1 format.\n"
        "\n"
        "Options:\n"
        "      --overwrite    Replace OUTPUT_FILE if it already exists\n"
        "  -h, --help         Show this message\n";
}

std::string next_value(int &i, int argc, char **argv, const std::string &name)
{
    if (i + 1 >= argc)
    {
        throw std::runtime_error("missing value for " + name);
    }
    return argv[++i];
}

Options parse_args(int argc, char **argv)
{
    Options options;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help")
        {
            std::cout << usage();
            std::exit(0);
        }
        if (arg == "--overwrite")
        {
            options.overwrite = true;
            continue;
        }
        if (arg == "-o" || arg == "--output")
        {
            options.output_file = next_value(i, argc, argv, arg);
            continue;
        }
        positional.push_back(arg);
    }
    if (positional.size() < 1 || positional.size() > 2)
    {
        throw std::runtime_error("expected INPUT_FILE and OUTPUT_FILE");
    }
    options.input_file = positional[0];
    if (positional.size() == 2)
    {
        options.output_file = positional[1];
    }
    if (options.output_file.empty())
    {
        throw std::runtime_error("missing OUTPUT_FILE");
    }
    return options;
}

template <typename T>
void write_scalar(std::ofstream &out, const T &value, const std::string &context)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!out.good())
    {
        throw std::runtime_error("failed to write " + context);
    }
}

std::int32_t parse_i32(const std::string &text, const std::string &name)
{
    std::size_t pos = 0;
    const long long value = std::stoll(text, &pos);
    if (pos != text.size() ||
        value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
    {
        throw std::runtime_error("invalid int32 for " + name + ": " + text);
    }
    return static_cast<std::int32_t>(value);
}

double parse_double(std::string text, const std::string &name)
{
    for (auto &ch: text)
    {
        if (ch == 'd' || ch == 'D')
        {
            ch = 'e';
        }
    }
    std::size_t pos = 0;
    const double value = std::stod(text, &pos);
    if (pos != text.size())
    {
        throw std::runtime_error("invalid double for " + name + ": " + text);
    }
    return value;
}

std::vector<Record> read_legacy_text(const fs::path &input_file)
{
    std::ifstream in(input_file);
    if (!in.good())
    {
        throw std::runtime_error("failed to open " + input_file.string());
    }

    int n_q = 0;
    in >> n_q;
    if (!in.good() || n_q < 0)
    {
        throw std::runtime_error("invalid shrink_sinvS text header in " + input_file.string());
    }

    std::vector<Record> records;
    while (true)
    {
        std::string nrow_s;
        std::string ncol_s;
        std::string begin_row_s;
        std::string end_row_s;
        std::string begin_col_s;
        std::string end_col_s;
        in >> nrow_s >> ncol_s >> begin_row_s >> end_row_s >> begin_col_s >> end_col_s;
        if (!in.good())
        {
            break;
        }

        std::string iq_s;
        std::string q_weight_s;
        in >> iq_s >> q_weight_s;
        if (!in.good())
        {
            throw std::runtime_error("truncated shrink_sinvS block header in " + input_file.string());
        }

        const auto nrow_total = parse_i32(nrow_s, "nrow");
        const auto ncol_total = parse_i32(ncol_s, "ncol");
        const auto begin_row = parse_i32(begin_row_s, "begin_row");
        const auto end_row = parse_i32(end_row_s, "end_row");
        const auto begin_col = parse_i32(begin_col_s, "begin_col");
        const auto end_col = parse_i32(end_col_s, "end_col");
        if (begin_row < 1 || begin_col < 1 ||
            end_row < begin_row || end_col < begin_col ||
            end_row > nrow_total || end_col > ncol_total)
        {
            throw std::runtime_error("invalid shrink_sinvS rectangular range in " + input_file.string());
        }

        const auto iq = parse_i32(iq_s, "iq");
        Record record;
        record.iq = iq;
        record.nrow_total = nrow_total;
        record.ncol_total = ncol_total;
        record.begin_row = begin_row;
        record.end_row = end_row;
        record.begin_col = begin_col;
        record.end_col = end_col;
        record.q_weight = parse_double(q_weight_s, "q_weight");
        const auto nrow_block = static_cast<std::size_t>(end_row - begin_row + 1);
        const auto ncol_block = static_cast<std::size_t>(end_col - begin_col + 1);
        if (nrow_block == 0 || ncol_block == 0)
        {
            throw std::runtime_error("invalid shrink_sinvS rectangular block dimensions in " +
                                     input_file.string());
        }
        record.payload.reserve(2 * nrow_block * ncol_block);
        for (std::size_t i = 0; i != nrow_block; ++i)
        {
            for (std::size_t j = 0; j != ncol_block; ++j)
            {
                std::string real_s;
                std::string imag_s;
                in >> real_s >> imag_s;
                if (!in.good())
                {
                    throw std::runtime_error("truncated shrink_sinvS payload in " + input_file.string());
                }
                record.payload.push_back(parse_double(real_s, "real"));
                record.payload.push_back(parse_double(imag_s, "imag"));
            }
        }
        records.push_back(std::move(record));
    }
    return records;
}

void write_v1(const fs::path &output_file, std::vector<Record> &records, bool overwrite)
{
    if (fs::exists(output_file) && !overwrite)
    {
        throw std::runtime_error(output_file.string() + " exists; use --overwrite");
    }

    std::int64_t offset = HEADER_BASE_SIZE + static_cast<std::int64_t>(records.size()) * RECORD_SIZE;
    for (auto &record: records)
    {
        record.offset = offset;
        offset += static_cast<std::int64_t>(record.payload.size() * sizeof(double));
    }

    std::ofstream out(output_file, std::ios::binary | std::ios::trunc);
    if (!out.good())
    {
        throw std::runtime_error("failed to open " + output_file.string());
    }

    const auto marker = READER_SHRINK_SINVS_V1_MARKER;
    const auto nrecords = static_cast<std::int32_t>(records.size());
    write_scalar(out, marker, "v1 shrink_sinvS marker");
    write_scalar(out, nrecords, "v1 shrink_sinvS record count");
    for (const auto &record: records)
    {
        write_scalar(out, record.iq, "v1 shrink_sinvS iq");
        write_scalar(out, record.nrow_total, "v1 shrink_sinvS nrow_total");
        write_scalar(out, record.ncol_total, "v1 shrink_sinvS ncol_total");
        write_scalar(out, record.begin_row, "v1 shrink_sinvS begin_row");
        write_scalar(out, record.end_row, "v1 shrink_sinvS end_row");
        write_scalar(out, record.begin_col, "v1 shrink_sinvS begin_col");
        write_scalar(out, record.end_col, "v1 shrink_sinvS end_col");
        write_scalar(out, record.q_weight, "v1 shrink_sinvS q_weight");
        write_scalar(out, record.offset, "v1 shrink_sinvS offset");
    }
    for (const auto &record: records)
    {
        out.write(reinterpret_cast<const char *>(record.payload.data()),
                  static_cast<std::streamsize>(record.payload.size() * sizeof(double)));
        if (!out.good())
        {
            throw std::runtime_error("failed to write v1 shrink_sinvS payload");
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        auto options = parse_args(argc, argv);
        auto records = read_legacy_text(options.input_file);
        write_v1(options.output_file, records, options.overwrite);
        std::cerr << "Converted " << records.size() << " shrink_sinvS block(s) to "
                  << options.output_file << "\n";
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "convert_legacy_sinvS: " << exc.what() << "\n";
        std::cerr << usage();
        return 1;
    }
}
