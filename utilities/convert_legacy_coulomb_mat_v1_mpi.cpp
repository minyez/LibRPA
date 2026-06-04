// Compile the code via
//   mpicxx -std=c++17 -o convert_legacy_coulomb_mat_v1_mpi.exe convert_legacy_coulomb_mat_v1_mpi.cpp
// Run as
//   mpirun -np 4 convert_legacy_coulomb_mat_v1_mpi.exe
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{

constexpr int COULOMB_V1_MARKER = 20129433;
constexpr int COMPLEX_FLAG = 1;
constexpr int V1_HEADER_BASE_SIZE = 6 * static_cast<int>(sizeof(int));
constexpr int V1_BLOCK_RECORD_SIZE = sizeof(int) + sizeof(std::int64_t);
constexpr int LEGACY_COMPLEX_BYTES = 2 * static_cast<int>(sizeof(double));
constexpr std::uint64_t CHECKPOINT_MAGIC = 0x31544b435f415052ULL; // "RPA_CKT1", little-endian
constexpr std::uint64_t CHECKPOINT_VERSION = 1;
constexpr int CHECKPOINT_HEADER_WORDS = 8;
constexpr MPI_Offset CHECKPOINT_HEADER_BYTES =
    CHECKPOINT_HEADER_WORDS * static_cast<MPI_Offset>(sizeof(std::int64_t));
constexpr std::size_t MAX_CACHED_FILES = 64;

struct ComplexValue
{
    double real = 0.0;
    double imag = 0.0;
};

static_assert(sizeof(ComplexValue) == 2 * sizeof(double),
              "ComplexValue must be two contiguous doubles");

struct Options
{
    fs::path input_dir = ".";
    fs::path output_dir;
    std::string input_prefix = "coulomb_mat";
    std::string output_prefix = "coulomb_full_iq";
    std::string basis_name = "basis_out";
    std::string stru_name = "stru_out";
    std::string ri_prefix = "Cs_data";
    std::string atom_naux_arg;
    fs::path checkpoint_path;
    int rows_per_task = 16;
    int progress_blocks = 500;
    bool quiet = false;
    bool restart = false;
};

struct AtomLayout
{
    std::vector<int> atom_naux;
    std::vector<int> atom_offsets;
    std::vector<std::int64_t> pair_offsets;
    int naux = 0;
    int natoms = 0;

    explicit AtomLayout(std::vector<int> atom_naux_in = {})
        : atom_naux(std::move(atom_naux_in))
    {
        if (atom_naux.empty()) return;
        natoms = static_cast<int>(atom_naux.size());
        atom_offsets.resize(natoms + 1, 0);
        for (int i = 0; i != natoms; ++i)
        {
            if (atom_naux[i] <= 0)
                throw std::runtime_error("per-atom auxiliary sizes must be positive");
            atom_offsets[i + 1] = atom_offsets[i] + atom_naux[i];
        }
        naux = atom_offsets.back();

        pair_offsets.resize(static_cast<std::size_t>(natoms) * (natoms + 1) / 2 + 1);
        std::int64_t offset = 0;
        for (int I = 0; I != natoms; ++I)
        {
            for (int J = I; J != natoms; ++J)
            {
                pair_offsets[atom_pair_index(I, J)] = offset;
                offset += static_cast<std::int64_t>(atom_naux[I]) * atom_naux[J];
            }
        }
        pair_offsets.back() = offset;
    }

    std::size_t atom_pair_index(int I, int J) const
    {
        assert(I <= J);
        return static_cast<std::size_t>(I) * natoms
               - static_cast<std::size_t>(I) * (I - 1) / 2 + (J - I);
    }

    std::pair<int, int> atom_for_aux(int index) const
    {
        if (index < 0 || index >= naux)
            throw std::runtime_error("auxiliary index out of range");
        const auto it = std::upper_bound(atom_offsets.begin(), atom_offsets.end(), index);
        const int atom = static_cast<int>(it - atom_offsets.begin()) - 1;
        return {atom, index - atom_offsets[atom]};
    }

    MPI_Offset header_bytes() const
    {
        return V1_HEADER_BASE_SIZE + static_cast<MPI_Offset>(natoms) * sizeof(int)
               + pair_count() * V1_BLOCK_RECORD_SIZE;
    }

    MPI_Offset payload_value_count() const
    {
        return static_cast<MPI_Offset>(pair_offsets.back());
    }

    MPI_Offset pair_count() const
    {
        return static_cast<MPI_Offset>(natoms) * (natoms + 1) / 2;
    }

    MPI_Offset block_table_offset() const
    {
        return V1_HEADER_BASE_SIZE + static_cast<MPI_Offset>(natoms) * sizeof(int);
    }

    MPI_Offset pair_value_offset(int I, int J, int i_local, int j_local) const
    {
        return static_cast<MPI_Offset>(pair_offsets[atom_pair_index(I, J)])
               + static_cast<MPI_Offset>(i_local) * atom_naux[J] + j_local;
    }

    MPI_Offset byte_offset(int I, int J, int i_local, int j_local) const
    {
        return header_bytes() + pair_value_offset(I, J, i_local, j_local)
                                  * LEGACY_COMPLEX_BYTES;
    }

    MPI_Offset file_size_bytes() const
    {
        return header_bytes() + payload_value_count() * LEGACY_COMPLEX_BYTES;
    }
};

struct BinaryTask
{
    int file_index = 0;
    int iq = 0;
    int naux = 0;
    int row_begin = 0;
    int nrows = 0;
    int col_begin = 0;
    int ncol = 0;
    MPI_Offset payload_offset = 0;
};

struct TextTask
{
    int file_index = 0;
    int iq = 0;
    int naux = 0;
    int row_begin = 0;
    int nrows = 0;
    int col_begin = 0;
    int ncol = 0;
    std::int64_t payload_offset = 0;
};

struct Timing
{
    double read_seconds = 0.0;
    double write_seconds = 0.0;
};

struct ProgressKey
{
    int iq = 0;
    std::size_t pair_index = 0;

    bool operator==(const ProgressKey &other) const
    {
        return iq == other.iq && pair_index == other.pair_index;
    }
};

struct ProgressKeyHash
{
    std::size_t operator()(const ProgressKey &key) const
    {
        const auto h1 = std::hash<int>{}(key.iq);
        const auto h2 = std::hash<std::size_t>{}(key.pair_index);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

class ProgressTracker
{
public:
    ProgressTracker(int rank_in, int interval_in)
        : rank(rank_in), interval(interval_in), start_time(MPI_Wtime())
    {
    }

    void note(int iq, std::size_t pair_index)
    {
        if (interval <= 0) return;
        const auto inserted = seen.insert({iq, pair_index}).second;
        if (!inserted) return;

        const auto count = seen.size();
        if (count % static_cast<std::size_t>(interval) != 0) return;

        std::cout << "[rank " << rank << "] processed " << count
                  << " atom-pair block(s); latest iq=" << iq
                  << ", pair_index=" << pair_index
                  << ", elapsed=" << std::fixed << std::setprecision(2)
                  << (MPI_Wtime() - start_time) << " s" << std::endl;
    }

private:
    int rank = 0;
    int interval = 0;
    double start_time = 0.0;
    std::unordered_set<ProgressKey, ProgressKeyHash> seen;
};

std::string usage()
{
    return
        "Usage: mpirun -np N convert_legacy_coulomb_mat_v1_mpi [input_dir] [options]\n"
        "\n"
        "Converts legacy Coulomb files to reader-v1 atom-pair-block files.\n"
        "Assumptions are fixed: stream complex output, skip lower input, target v1.\n"
        "\n"
        "Options:\n"
        "  -O, --output-dir    DIR     Output directory (default: input_dir)\n"
        "  -i, --input-prefix  PREFIX  Legacy input prefix (default: coulomb_mat)\n"
        "  -o, --output-prefix PREFIX  Output prefix (default: coulomb_full_iq)\n"
        "  -T, --target-reader-version 1\n"
        "                              Accepted for compatibility; only v1 is supported\n"
        "      --atom-naux  LIST       Per-atom auxiliary sizes, comma/space separated\n"
        "      --basis-name NAME       Basis metadata filename (default: basis_out)\n"
        "      --stru-name  NAME       Structure metadata filename (default: stru_out)\n"
        "      --ri-prefix  PREFIX     RI coefficient prefix for basis fallback (default: Cs_data)\n"
        "      --rows-per-task N       Binary legacy row chunk size (default: 16)\n"
        "      --progress-blocks N     Print per-rank progress every N atom-pair blocks (default: 500; 0 disables)\n"
        "      --restart              Resume from the temporary checkpoint file\n"
        "      --checkpoint PATH      Checkpoint file (default: output_dir/.<output_prefix>.v1_mpi.ckpt)\n"
        "      --stream-complex        Accepted for compatibility; always enabled\n"
        "      --skip-lower            Accepted for compatibility; always enabled\n"
        "      --quiet                 Suppress conversion summary\n"
        "  -h, --help                  Show this message\n";
}

bool starts_with(const std::string &text, const std::string &prefix)
{
    return text.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string &text, const std::string &suffix)
{
    return text.size() >= suffix.size()
           && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

double parse_double_token(std::string token)
{
    for (auto &ch: token)
        if (ch == 'd' || ch == 'D') ch = 'e';
    return std::stod(token);
}

std::vector<int> parse_atom_naux_list(const std::string &text)
{
    std::string normalized = text;
    for (auto &ch: normalized)
        if (ch == ',') ch = ' ';
    std::istringstream iss(normalized);
    std::vector<int> values;
    int value = 0;
    while (iss >> value) values.push_back(value);
    if (values.empty())
        throw std::runtime_error("--atom-naux did not contain any sizes");
    for (const auto nb: values)
        if (nb <= 0) throw std::runtime_error("--atom-naux sizes must be positive");
    return values;
}

Options parse_args(int argc, char **argv)
{
    Options opts;
    bool input_seen = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help")
        {
            std::cout << usage();
            MPI_Finalize();
            std::exit(0);
        }
        else if (arg == "-O" || arg == "--output-dir")
        {
            opts.output_dir = next_value(arg);
        }
        else if (arg == "-i" || arg == "--input-prefix")
        {
            opts.input_prefix = next_value(arg);
        }
        else if (arg == "-o" || arg == "--output-prefix")
        {
            opts.output_prefix = next_value(arg);
        }
        else if (arg == "-T" || arg == "--target-reader-version")
        {
            const auto version = next_value(arg);
            if (version != "1")
                throw std::runtime_error("this MPI converter only supports -T 1");
        }
        else if (arg == "--atom-naux")
        {
            opts.atom_naux_arg = next_value(arg);
        }
        else if (arg == "--basis-name")
        {
            opts.basis_name = next_value(arg);
        }
        else if (arg == "--stru-name")
        {
            opts.stru_name = next_value(arg);
        }
        else if (arg == "--ri-prefix")
        {
            opts.ri_prefix = next_value(arg);
        }
        else if (arg == "--rows-per-task")
        {
            opts.rows_per_task = std::stoi(next_value(arg));
            if (opts.rows_per_task <= 0)
                throw std::runtime_error("--rows-per-task must be positive");
        }
        else if (arg == "--progress-blocks")
        {
            opts.progress_blocks = std::stoi(next_value(arg));
            if (opts.progress_blocks < 0)
                throw std::runtime_error("--progress-blocks must be non-negative");
        }
        else if (arg == "--restart")
        {
            opts.restart = true;
        }
        else if (arg == "--checkpoint")
        {
            opts.checkpoint_path = next_value(arg);
        }
        else if (arg == "--stream-complex" || arg == "--skip-lower")
        {
            // Fixed fast path: complex-valued streaming output with lower input skipped.
        }
        else if (arg == "--quiet")
        {
            opts.quiet = true;
        }
        else if (!starts_with(arg, "-") && !input_seen)
        {
            opts.input_dir = arg;
            input_seen = true;
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.output_dir.empty()) opts.output_dir = opts.input_dir;
    if (opts.checkpoint_path.empty())
        opts.checkpoint_path = opts.output_dir
                               / ("." + opts.output_prefix + ".v1_mpi.ckpt");
    return opts;
}

std::vector<int> read_atom_types_from_stru(const fs::path &path)
{
    std::ifstream in(path);
    if (!in.good())
        throw std::runtime_error("failed to open " + path.string());

    std::string tok;
    for (int i = 0; i != 18; ++i) in >> tok;
    int natoms = 0;
    in >> natoms;
    if (!in.good() || natoms <= 0)
        throw std::runtime_error("invalid structure file " + path.string());

    std::vector<int> atom_types(natoms);
    for (int i = 0; i != natoms; ++i)
    {
        std::string x, y, z;
        int type = 0;
        in >> x >> y >> z >> type;
        if (!in.good() || type <= 0)
            throw std::runtime_error("invalid atom entry in " + path.string());
        atom_types[i] = type - 1;
    }
    return atom_types;
}

std::vector<int> read_atom_naux_from_basis_and_stru(const fs::path &basis_path,
                                                    const fs::path &stru_path)
{
    const auto atom_types = read_atom_types_from_stru(stru_path);
    std::ifstream in(basis_path);
    if (!in.good())
        throw std::runtime_error("failed to open " + basis_path.string());

    int ntypes = 0;
    int total_wfc = 0;
    int total_aux = 0;
    std::string kind;
    in >> ntypes >> total_wfc >> total_aux >> kind;
    if (!in.good() || ntypes <= 0)
        throw std::runtime_error("invalid basis file " + basis_path.string());

    std::map<int, int> type_naux;
    for (int i = 0; i != ntypes; ++i)
    {
        int type = 0;
        int n_wfc = 0;
        int n_aux = 0;
        in >> type >> n_wfc >> n_aux;
        if (!in.good() || type <= 0 || n_aux <= 0)
            throw std::runtime_error("invalid basis type entry in " + basis_path.string());
        type_naux[type - 1] = n_aux;
    }

    std::vector<int> atom_naux;
    atom_naux.reserve(atom_types.size());
    for (const auto type: atom_types)
    {
        const auto it = type_naux.find(type);
        if (it == type_naux.end())
            throw std::runtime_error("basis file lacks atom type " + std::to_string(type + 1));
        atom_naux.push_back(it->second);
    }
    return atom_naux;
}

bool cs_binary_layout_matches(const fs::path &path)
{
    const auto file_size = fs::file_size(path);
    if (file_size < 12) return false;

    std::ifstream in(path, std::ios::binary);
    int natom = 0;
    int ncell = 0;
    int nblocks = 0;
    in.read(reinterpret_cast<char *>(&natom), sizeof(int));
    in.read(reinterpret_cast<char *>(&ncell), sizeof(int));
    in.read(reinterpret_cast<char *>(&nblocks), sizeof(int));
    if (!in.good() || natom <= 0 || ncell < 0 || nblocks < 0) return false;

    std::uintmax_t pos = 12;
    for (int ib = 0; ib != nblocks; ++ib)
    {
        int dims[8];
        in.read(reinterpret_cast<char *>(dims), sizeof(dims));
        if (!in.good()) return false;
        if (dims[0] <= 0 || dims[0] > natom || dims[1] <= 0 || dims[1] > natom)
            return false;
        if (dims[5] <= 0 || dims[6] <= 0 || dims[7] <= 0) return false;
        const auto payload = static_cast<std::uintmax_t>(dims[5]) * dims[6] * dims[7]
                             * sizeof(double);
        pos += sizeof(dims) + payload;
        if (pos > file_size) return false;
        in.seekg(static_cast<std::streamoff>(payload), std::ios::cur);
    }
    return pos == file_size;
}

void parse_cs_binary_atom_naux(const fs::path &path, std::map<int, int> &atom_naux,
                               int &natoms_seen)
{
    std::ifstream in(path, std::ios::binary);
    int natom = 0;
    int ncell = 0;
    int nblocks = 0;
    in.read(reinterpret_cast<char *>(&natom), sizeof(int));
    in.read(reinterpret_cast<char *>(&ncell), sizeof(int));
    in.read(reinterpret_cast<char *>(&nblocks), sizeof(int));
    if (!in.good() || natom <= 0 || nblocks < 0)
        throw std::runtime_error("invalid binary Cs header in " + path.string());
    if (natoms_seen < 0) natoms_seen = natom;
    if (natoms_seen != natom)
        throw std::runtime_error("inconsistent atom count in Cs files");

    for (int ib = 0; ib != nblocks; ++ib)
    {
        int dims[8];
        in.read(reinterpret_cast<char *>(dims), sizeof(dims));
        if (!in.good())
            throw std::runtime_error("truncated binary Cs header in " + path.string());
        const int ia1 = dims[0] - 1;
        if (ia1 < 0 || ia1 >= natom || dims[5] <= 0 || dims[6] <= 0 || dims[7] <= 0)
            throw std::runtime_error("invalid binary Cs block in " + path.string());
        atom_naux[ia1] = dims[7];
        const auto payload = static_cast<std::streamoff>(dims[5]) * dims[6] * dims[7]
                             * sizeof(double);
        in.seekg(payload, std::ios::cur);
    }
}

void parse_cs_text_atom_naux(const fs::path &path, std::map<int, int> &atom_naux,
                             int &natoms_seen)
{
    std::ifstream in(path);
    int natom = 0;
    int ncell = 0;
    in >> natom >> ncell;
    if (!in.good() || natom <= 0)
        throw std::runtime_error("invalid text Cs header in " + path.string());
    if (natoms_seen < 0) natoms_seen = natom;
    if (natoms_seen != natom)
        throw std::runtime_error("inconsistent atom count in Cs files");

    while (true)
    {
        int ia1 = 0, ia2 = 0, r1 = 0, r2 = 0, r3 = 0;
        int n_i = 0, n_j = 0, n_mu = 0;
        in >> ia1;
        if (!in.good()) break;
        in >> ia2 >> r1 >> r2 >> r3 >> n_i >> n_j >> n_mu;
        if (!in.good() || ia1 <= 0 || ia1 > natom || n_i <= 0 || n_j <= 0 || n_mu <= 0)
            throw std::runtime_error("invalid text Cs block in " + path.string());
        atom_naux[ia1 - 1] = n_mu;
        std::string value;
        for (std::int64_t i = 0; i != static_cast<std::int64_t>(n_i) * n_j * n_mu; ++i)
            in >> value;
    }
}

std::vector<int> read_atom_naux_from_cs(const fs::path &input_dir,
                                        const std::string &ri_prefix)
{
    std::vector<fs::path> files;
    for (const auto &entry: fs::directory_iterator(input_dir))
    {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (starts_with(name, ri_prefix)) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("could not infer atom auxiliary sizes; no "
                                 + ri_prefix + "* files found");

    std::map<int, int> atom_naux_map;
    int natoms_seen = -1;
    for (const auto &path: files)
    {
        if (cs_binary_layout_matches(path))
            parse_cs_binary_atom_naux(path, atom_naux_map, natoms_seen);
        else
            parse_cs_text_atom_naux(path, atom_naux_map, natoms_seen);
    }

    std::vector<int> atom_naux(natoms_seen);
    for (int i = 0; i != natoms_seen; ++i)
    {
        const auto it = atom_naux_map.find(i);
        if (it == atom_naux_map.end())
            throw std::runtime_error("could not infer auxiliary size for atom "
                                     + std::to_string(i + 1));
        atom_naux[i] = it->second;
    }
    return atom_naux;
}

AtomLayout resolve_atom_layout(const Options &opts)
{
    if (!opts.atom_naux_arg.empty())
        return AtomLayout(parse_atom_naux_list(opts.atom_naux_arg));

    const auto basis_path = opts.input_dir / opts.basis_name;
    const auto stru_path = opts.input_dir / opts.stru_name;
    if (fs::exists(basis_path) && fs::exists(stru_path))
        return AtomLayout(read_atom_naux_from_basis_and_stru(basis_path, stru_path));

    return AtomLayout(read_atom_naux_from_cs(opts.input_dir, opts.ri_prefix));
}

std::vector<fs::path> discover_input_files(const Options &opts)
{
    std::vector<fs::path> files;
    const auto prefix = opts.input_prefix + "_";
    for (const auto &entry: fs::directory_iterator(opts.input_dir))
    {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (starts_with(name, prefix) && ends_with(name, ".txt"))
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("no input files found with prefix " + opts.input_prefix);
    return files;
}

void validate_legacy_block(const fs::path &path, int naux, int row_begin, int row_end,
                           int col_begin, int col_end, int iq)
{
    if (naux <= 0 || iq <= 0)
        throw std::runtime_error("invalid legacy Coulomb block in " + path.string());
    if (row_begin <= 0 || row_end < row_begin || row_end > naux)
        throw std::runtime_error("invalid row range in " + path.string());
    if (col_begin <= 0 || col_end < col_begin || col_end > naux)
        throw std::runtime_error("invalid column range in " + path.string());
}

bool legacy_binary_layout_matches(const fs::path &path)
{
    const auto file_size = fs::file_size(path);
    if (file_size < 8) return false;
    std::ifstream in(path, std::ios::binary);
    int nirk = 0;
    int nblocks = 0;
    in.read(reinterpret_cast<char *>(&nirk), sizeof(int));
    in.read(reinterpret_cast<char *>(&nblocks), sizeof(int));
    if (!in.good() || nirk <= 0 || nblocks < 0 || nblocks > nirk) return false;

    std::uintmax_t pos = 8;
    for (int ib = 0; ib != nblocks; ++ib)
    {
        int ints[6];
        double weight = 0.0;
        in.read(reinterpret_cast<char *>(ints), sizeof(ints));
        in.read(reinterpret_cast<char *>(&weight), sizeof(double));
        if (!in.good()) return false;
        const int naux = ints[0], row_begin = ints[1], row_end = ints[2];
        const int col_begin = ints[3], col_end = ints[4], iq = ints[5];
        if (naux <= 0 || iq <= 0 || row_begin <= 0 || row_end < row_begin ||
            row_end > naux || col_begin <= 0 || col_end < col_begin || col_end > naux)
            return false;
        const auto nrow = static_cast<std::uintmax_t>(row_end - row_begin + 1);
        const auto ncol = static_cast<std::uintmax_t>(col_end - col_begin + 1);
        const auto payload = nrow * ncol * LEGACY_COMPLEX_BYTES;
        pos += 32 + payload;
        if (pos > file_size) return false;
        in.seekg(static_cast<std::streamoff>(payload), std::ios::cur);
    }
    return pos == file_size;
}

void set_q_naux(std::map<int, int> &q_naux, int iq, int naux)
{
    const auto it = q_naux.find(iq);
    if (it == q_naux.end())
    {
        q_naux[iq] = naux;
        return;
    }
    if (it->second != naux)
        throw std::runtime_error("inconsistent Naux for q-point " + std::to_string(iq));
}

void scan_binary_file(const fs::path &path, int file_index, int rows_per_task,
                      std::vector<BinaryTask> &tasks, std::map<int, int> &q_naux,
                      std::int64_t &nblocks_total)
{
    std::ifstream in(path, std::ios::binary);
    int nirk = 0;
    int nblocks = 0;
    in.read(reinterpret_cast<char *>(&nirk), sizeof(int));
    in.read(reinterpret_cast<char *>(&nblocks), sizeof(int));
    if (!in.good()) throw std::runtime_error("failed to read " + path.string());

    MPI_Offset pos = 8;
    for (int ib = 0; ib != nblocks; ++ib)
    {
        int naux = 0, row_begin = 0, row_end = 0, col_begin = 0, col_end = 0, iq = 0;
        double q_weight = 0.0;
        in.read(reinterpret_cast<char *>(&naux), sizeof(int));
        in.read(reinterpret_cast<char *>(&row_begin), sizeof(int));
        in.read(reinterpret_cast<char *>(&row_end), sizeof(int));
        in.read(reinterpret_cast<char *>(&col_begin), sizeof(int));
        in.read(reinterpret_cast<char *>(&col_end), sizeof(int));
        in.read(reinterpret_cast<char *>(&iq), sizeof(int));
        in.read(reinterpret_cast<char *>(&q_weight), sizeof(double));
        validate_legacy_block(path, naux, row_begin, row_end, col_begin, col_end, iq);
        set_q_naux(q_naux, iq, naux);

        const int nrow = row_end - row_begin + 1;
        const int ncol = col_end - col_begin + 1;
        const int row_begin0 = row_begin - 1;
        const int col_begin0 = col_begin - 1;
        const int col_end0 = col_end - 1;
        const MPI_Offset payload_offset = pos + 32;

        for (int local_row = 0; local_row < nrow;)
        {
            const int global_row = row_begin0 + local_row;
            if (global_row > col_end0) break; // strictly lower rows under --skip-lower
            const int usable_rows = std::min({rows_per_task, nrow - local_row,
                                              col_end0 + 1 - global_row});
            tasks.push_back({
                file_index, iq, naux, global_row, usable_rows, col_begin0, ncol,
                payload_offset + static_cast<MPI_Offset>(local_row) * ncol
                                 * LEGACY_COMPLEX_BYTES
            });
            local_row += usable_rows;
        }

        const MPI_Offset payload_bytes = static_cast<MPI_Offset>(nrow) * ncol
                                         * LEGACY_COMPLEX_BYTES;
        pos += 32 + payload_bytes;
        in.seekg(static_cast<std::streamoff>(payload_bytes), std::ios::cur);
        ++nblocks_total;
    }
}

void skip_text_values(std::ifstream &in, std::int64_t nvalues, const fs::path &path)
{
    std::string dummy;
    for (std::int64_t i = 0; i != 2 * nvalues; ++i)
    {
        in >> dummy;
        if (!in.good())
            throw std::runtime_error("truncated text Coulomb values in " + path.string());
    }
}

void scan_text_file(const fs::path &path, int file_index, int rows_per_task,
                    std::vector<TextTask> &tasks, std::map<int, int> &q_naux,
                    std::int64_t &nblocks_total)
{
    std::ifstream in(path);
    int nirk = 0;
    in >> nirk;
    if (!in.good() || nirk <= 0)
        throw std::runtime_error("invalid text Coulomb header in " + path.string());

    while (true)
    {
        int naux = 0, row_begin = 0, row_end = 0, col_begin = 0, col_end = 0, iq = 0;
        std::string q_weight;
        in >> naux;
        if (!in.good()) break;
        in >> row_begin >> row_end >> col_begin >> col_end >> iq >> q_weight;
        validate_legacy_block(path, naux, row_begin, row_end, col_begin, col_end, iq);
        set_q_naux(q_naux, iq, naux);
        const int nrow = row_end - row_begin + 1;
        const int ncol = col_end - col_begin + 1;
        const int row_begin0 = row_begin - 1;
        const int col_begin0 = col_begin - 1;
        const int col_end0 = col_end - 1;
        for (int local_row = 0; local_row < nrow;)
        {
            const int global_row = row_begin0 + local_row;
            const int rows_this =
                (global_row <= col_end0)
                    ? std::min({rows_per_task, nrow - local_row,
                                col_end0 + 1 - global_row})
                    : nrow - local_row;
            const auto payload_pos = in.tellg();
            if (payload_pos == std::streampos(-1))
                throw std::runtime_error("failed to record text offset in "
                                         + path.string());
            if (global_row <= col_end0)
            {
                tasks.push_back({
                    file_index, iq, naux, global_row, rows_this, col_begin0, ncol,
                    static_cast<std::int64_t>(payload_pos)
                });
            }
            skip_text_values(in, static_cast<std::int64_t>(rows_this) * ncol, path);
            local_row += rows_this;
        }
        ++nblocks_total;
    }
}

std::vector<char> pack_strings(const std::vector<fs::path> &paths,
                               std::vector<int> &lengths)
{
    std::vector<char> blob;
    lengths.clear();
    for (const auto &path: paths)
    {
        const auto text = path.string();
        lengths.push_back(static_cast<int>(text.size()));
        blob.insert(blob.end(), text.begin(), text.end());
    }
    return blob;
}

std::vector<fs::path> unpack_strings(const std::vector<int> &lengths,
                                     const std::vector<char> &blob)
{
    std::vector<fs::path> paths;
    std::size_t pos = 0;
    for (const auto len: lengths)
    {
        paths.emplace_back(std::string(blob.data() + pos, blob.data() + pos + len));
        pos += len;
    }
    return paths;
}

template <typename T>
void bcast_vector(std::vector<T> &values, int root, MPI_Comm comm)
{
    int size = static_cast<int>(values.size());
    MPI_Bcast(&size, 1, MPI_INT, root, comm);
    values.resize(size);
    if (size > 0)
        MPI_Bcast(values.data(), static_cast<int>(sizeof(T) * values.size()),
                  MPI_BYTE, root, comm);
}

void bcast_paths(std::vector<fs::path> &paths, int root, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::vector<int> lengths;
    std::vector<char> blob;
    if (rank == root) blob = pack_strings(paths, lengths);
    bcast_vector(lengths, root, comm);
    int blob_size = static_cast<int>(blob.size());
    MPI_Bcast(&blob_size, 1, MPI_INT, root, comm);
    blob.resize(blob_size);
    if (blob_size > 0) MPI_Bcast(blob.data(), blob_size, MPI_BYTE, root, comm);
    if (rank != root) paths = unpack_strings(lengths, blob);
}

std::vector<std::int64_t> pack_tasks(const std::vector<BinaryTask> &tasks)
{
    std::vector<std::int64_t> packed;
    packed.reserve(tasks.size() * 8);
    for (const auto &task: tasks)
    {
        packed.push_back(task.file_index);
        packed.push_back(task.iq);
        packed.push_back(task.naux);
        packed.push_back(task.row_begin);
        packed.push_back(task.nrows);
        packed.push_back(task.col_begin);
        packed.push_back(task.ncol);
        packed.push_back(static_cast<std::int64_t>(task.payload_offset));
    }
    return packed;
}

std::vector<BinaryTask> unpack_tasks(const std::vector<std::int64_t> &packed)
{
    if (packed.size() % 8 != 0)
        throw std::runtime_error("internal error: malformed packed task list");
    std::vector<BinaryTask> tasks;
    tasks.reserve(packed.size() / 8);
    for (std::size_t i = 0; i != packed.size(); i += 8)
    {
        tasks.push_back({
            static_cast<int>(packed[i]),
            static_cast<int>(packed[i + 1]),
            static_cast<int>(packed[i + 2]),
            static_cast<int>(packed[i + 3]),
            static_cast<int>(packed[i + 4]),
            static_cast<int>(packed[i + 5]),
            static_cast<int>(packed[i + 6]),
            static_cast<MPI_Offset>(packed[i + 7])
        });
    }
    return tasks;
}

std::vector<std::int64_t> pack_text_tasks(const std::vector<TextTask> &tasks)
{
    std::vector<std::int64_t> packed;
    packed.reserve(tasks.size() * 8);
    for (const auto &task: tasks)
    {
        packed.push_back(task.file_index);
        packed.push_back(task.iq);
        packed.push_back(task.naux);
        packed.push_back(task.row_begin);
        packed.push_back(task.nrows);
        packed.push_back(task.col_begin);
        packed.push_back(task.ncol);
        packed.push_back(task.payload_offset);
    }
    return packed;
}

std::vector<TextTask> unpack_text_tasks(const std::vector<std::int64_t> &packed)
{
    if (packed.size() % 8 != 0)
        throw std::runtime_error("internal error: malformed packed text task list");
    std::vector<TextTask> tasks;
    tasks.reserve(packed.size() / 8);
    for (std::size_t i = 0; i != packed.size(); i += 8)
    {
        tasks.push_back({
            static_cast<int>(packed[i]),
            static_cast<int>(packed[i + 1]),
            static_cast<int>(packed[i + 2]),
            static_cast<int>(packed[i + 3]),
            static_cast<int>(packed[i + 4]),
            static_cast<int>(packed[i + 5]),
            static_cast<int>(packed[i + 6]),
            packed[i + 7]
        });
    }
    return tasks;
}

fs::path output_path_for(const Options &opts, int iq)
{
    return opts.output_dir / (opts.output_prefix + "_" + std::to_string(iq) + ".dat");
}

bool has_existing_outputs(const Options &opts)
{
    if (!fs::exists(opts.output_dir)) return false;
    const auto prefix = opts.output_prefix + "_";
    for (const auto &entry: fs::directory_iterator(opts.output_dir))
    {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (starts_with(name, prefix) && ends_with(name, ".dat"))
            return true;
    }
    return false;
}

void check_no_existing_outputs(const Options &opts)
{
    if (has_existing_outputs(opts))
        throw std::runtime_error("existing output file found for prefix "
                                 + opts.output_prefix
                                 + "; please clean up manually or use --restart");
    if (fs::exists(opts.checkpoint_path))
        throw std::runtime_error("existing checkpoint file found: "
                                 + opts.checkpoint_path.string()
                                 + "; please clean up manually or use --restart");
}

void create_output_file(const Options &opts, const AtomLayout &layout, int iq,
                        Timing &timing)
{
    const auto path = output_path_for(opts, iq);
    MPI_File file = MPI_FILE_NULL;
    double t0 = MPI_Wtime();
    const int ierr = MPI_File_open(MPI_COMM_SELF, path.string().c_str(),
                                   MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_EXCL,
                                   MPI_INFO_NULL, &file);
    timing.write_seconds += MPI_Wtime() - t0;
    if (ierr != MPI_SUCCESS)
        throw std::runtime_error("failed to create output file " + path.string());

    const int header[6] = {
        COULOMB_V1_MARKER, iq, layout.naux, COMPLEX_FLAG, layout.natoms,
        static_cast<int>(layout.pair_count())
    };
    t0 = MPI_Wtime();
    MPI_File_write_at(file, 0, const_cast<int *>(header), sizeof(header),
                      MPI_BYTE, MPI_STATUS_IGNORE);
    timing.write_seconds += MPI_Wtime() - t0;
    t0 = MPI_Wtime();
    MPI_File_write_at(file, V1_HEADER_BASE_SIZE,
                      const_cast<int *>(layout.atom_naux.data()),
                      static_cast<int>(layout.atom_naux.size() * sizeof(int)),
                      MPI_BYTE, MPI_STATUS_IGNORE);
    timing.write_seconds += MPI_Wtime() - t0;
    MPI_Offset record_offset = layout.block_table_offset();
    for (int I = 0; I != layout.natoms; ++I)
    {
        for (int J = I; J != layout.natoms; ++J)
        {
            const int ipair = static_cast<int>(layout.atom_pair_index(I, J));
            const std::int64_t block_offset =
                static_cast<std::int64_t>(layout.byte_offset(I, J, 0, 0));
            t0 = MPI_Wtime();
            MPI_File_write_at(file, record_offset, const_cast<int *>(&ipair),
                              sizeof(ipair), MPI_BYTE, MPI_STATUS_IGNORE);
            timing.write_seconds += MPI_Wtime() - t0;
            t0 = MPI_Wtime();
            MPI_File_write_at(file, record_offset + static_cast<MPI_Offset>(sizeof(int)),
                              const_cast<std::int64_t *>(&block_offset),
                              sizeof(block_offset), MPI_BYTE, MPI_STATUS_IGNORE);
            timing.write_seconds += MPI_Wtime() - t0;
            record_offset += V1_BLOCK_RECORD_SIZE;
        }
    }
    t0 = MPI_Wtime();
    MPI_File_set_size(file, layout.file_size_bytes());
    timing.write_seconds += MPI_Wtime() - t0;
    t0 = MPI_Wtime();
    MPI_File_close(&file);
    timing.write_seconds += MPI_Wtime() - t0;
}

void validate_output_file(const Options &opts, const AtomLayout &layout, int iq)
{
    const auto path = output_path_for(opts, iq);
    if (!fs::exists(path))
        throw std::runtime_error("missing restart output file " + path.string());
    if (fs::file_size(path) != static_cast<std::uintmax_t>(layout.file_size_bytes()))
        throw std::runtime_error("restart output file has unexpected size: "
                                 + path.string());

    std::ifstream in(path, std::ios::binary);
    int header[6];
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!in.good())
        throw std::runtime_error("failed to read restart output header: "
                                 + path.string());
    if (header[0] != COULOMB_V1_MARKER || header[1] != iq ||
        header[2] != layout.naux || header[3] != COMPLEX_FLAG ||
        header[4] != layout.natoms || header[5] != static_cast<int>(layout.pair_count()))
        throw std::runtime_error("restart output header does not match this run: "
                                 + path.string());

    std::vector<int> atom_naux(layout.natoms);
    in.read(reinterpret_cast<char *>(atom_naux.data()),
            static_cast<std::streamsize>(atom_naux.size() * sizeof(int)));
    if (!in.good() || atom_naux != layout.atom_naux)
        throw std::runtime_error("restart output atom layout does not match this run: "
                                 + path.string());

    for (int I = 0; I != layout.natoms; ++I)
    {
        for (int J = I; J != layout.natoms; ++J)
        {
            int ipair = -1;
            std::int64_t block_offset = -1;
            in.read(reinterpret_cast<char *>(&ipair), sizeof(ipair));
            in.read(reinterpret_cast<char *>(&block_offset), sizeof(block_offset));
            if (!in.good() ||
                ipair != static_cast<int>(layout.atom_pair_index(I, J)) ||
                block_offset != static_cast<std::int64_t>(layout.byte_offset(I, J, 0, 0)))
                throw std::runtime_error(
                    "restart output block table does not match this run: "
                    + path.string());
        }
    }
}

void hash_byte(std::uint64_t &hash, unsigned char byte)
{
    constexpr std::uint64_t prime = 1099511628211ULL;
    hash ^= byte;
    hash *= prime;
}

template <typename T>
void hash_scalar(std::uint64_t &hash, T value)
{
    static_assert(std::is_integral<T>::value, "hash_scalar expects an integer type");
    using U = typename std::make_unsigned<T>::type;
    U unsigned_value = static_cast<U>(value);
    for (std::size_t i = 0; i != sizeof(U); ++i)
    {
        hash_byte(hash, static_cast<unsigned char>(unsigned_value & 0xffU));
        unsigned_value >>= 8;
    }
}

void hash_string(std::uint64_t &hash, const std::string &text)
{
    hash_scalar(hash, static_cast<std::uint64_t>(text.size()));
    for (const auto ch: text)
        hash_byte(hash, static_cast<unsigned char>(ch));
}

void hash_path_metadata(std::uint64_t &hash, const fs::path &path)
{
    hash_string(hash, path.string());
    hash_scalar(hash, static_cast<std::uint64_t>(fs::file_size(path)));
    const auto stamp = fs::last_write_time(path).time_since_epoch().count();
    hash_scalar(hash, static_cast<std::int64_t>(stamp));
}

std::uint64_t conversion_signature(const Options &opts, const AtomLayout &layout,
                                   const std::vector<fs::path> &input_files,
                                   const std::vector<BinaryTask> &binary_tasks,
                                   const std::vector<TextTask> &text_tasks,
                                   const std::map<int, int> &q_naux)
{
    std::uint64_t hash = 1469598103934665603ULL;
    hash_scalar(hash, static_cast<std::int64_t>(COULOMB_V1_MARKER));
    hash_string(hash, opts.input_prefix);
    hash_string(hash, opts.output_prefix);
    hash_scalar(hash, static_cast<std::int64_t>(opts.rows_per_task));
    hash_scalar(hash, static_cast<std::int64_t>(layout.natoms));
    hash_scalar(hash, static_cast<std::int64_t>(layout.naux));
    for (const auto naux: layout.atom_naux)
        hash_scalar(hash, static_cast<std::int64_t>(naux));
    for (const auto &path: input_files)
        hash_path_metadata(hash, path);
    for (const auto &task: binary_tasks)
    {
        hash_scalar(hash, static_cast<std::int64_t>(task.file_index));
        hash_scalar(hash, static_cast<std::int64_t>(task.iq));
        hash_scalar(hash, static_cast<std::int64_t>(task.naux));
        hash_scalar(hash, static_cast<std::int64_t>(task.row_begin));
        hash_scalar(hash, static_cast<std::int64_t>(task.nrows));
        hash_scalar(hash, static_cast<std::int64_t>(task.col_begin));
        hash_scalar(hash, static_cast<std::int64_t>(task.ncol));
        hash_scalar(hash, static_cast<std::int64_t>(task.payload_offset));
    }
    for (const auto &task: text_tasks)
    {
        hash_scalar(hash, static_cast<std::int64_t>(task.file_index));
        hash_scalar(hash, static_cast<std::int64_t>(task.iq));
        hash_scalar(hash, static_cast<std::int64_t>(task.naux));
        hash_scalar(hash, static_cast<std::int64_t>(task.row_begin));
        hash_scalar(hash, static_cast<std::int64_t>(task.nrows));
        hash_scalar(hash, static_cast<std::int64_t>(task.col_begin));
        hash_scalar(hash, static_cast<std::int64_t>(task.ncol));
        hash_scalar(hash, task.payload_offset);
    }
    for (const auto &[iq, naux]: q_naux)
    {
        hash_scalar(hash, static_cast<std::int64_t>(iq));
        hash_scalar(hash, static_cast<std::int64_t>(naux));
    }
    return hash;
}

std::vector<std::uint64_t> checkpoint_header(std::size_t nbinary_tasks,
                                             std::size_t ntext_tasks,
                                             const AtomLayout &layout,
                                             std::uint64_t signature)
{
    return {
        CHECKPOINT_MAGIC,
        CHECKPOINT_VERSION,
        static_cast<std::uint64_t>(COULOMB_V1_MARKER),
        static_cast<std::uint64_t>(nbinary_tasks),
        static_cast<std::uint64_t>(ntext_tasks),
        static_cast<std::uint64_t>(layout.naux),
        static_cast<std::uint64_t>(layout.natoms),
        signature
    };
}

std::vector<unsigned char> read_checkpoint_file(const Options &opts,
                                                std::size_t nbinary_tasks,
                                                std::size_t ntext_tasks,
                                                const AtomLayout &layout,
                                                std::uint64_t signature)
{
    const auto expected_header =
        checkpoint_header(nbinary_tasks, ntext_tasks, layout, signature);
    const std::uintmax_t expected_size =
        static_cast<std::uintmax_t>(CHECKPOINT_HEADER_BYTES + nbinary_tasks
                                    + ntext_tasks);
    if (fs::file_size(opts.checkpoint_path) != expected_size)
        throw std::runtime_error("checkpoint file has unexpected size: "
                                 + opts.checkpoint_path.string());

    std::ifstream in(opts.checkpoint_path, std::ios::binary);
    std::vector<std::uint64_t> header(CHECKPOINT_HEADER_WORDS);
    in.read(reinterpret_cast<char *>(header.data()),
            static_cast<std::streamsize>(header.size() * sizeof(std::uint64_t)));
    if (!in.good() || header != expected_header)
        throw std::runtime_error("checkpoint file does not match this conversion: "
                                 + opts.checkpoint_path.string());

    std::vector<unsigned char> completed(nbinary_tasks + ntext_tasks);
    in.read(reinterpret_cast<char *>(completed.data()),
            static_cast<std::streamsize>(completed.size()));
    if (!in.good())
        throw std::runtime_error("failed to read checkpoint flags: "
                                 + opts.checkpoint_path.string());
    for (const auto flag: completed)
        if (flag != 0 && flag != 1)
            throw std::runtime_error("checkpoint file contains invalid flags: "
                                     + opts.checkpoint_path.string());
    return completed;
}

void write_checkpoint_file(const Options &opts, std::size_t nbinary_tasks,
                           std::size_t ntext_tasks, const AtomLayout &layout,
                           std::uint64_t signature,
                           const std::vector<unsigned char> &completed)
{
    if (completed.size() != nbinary_tasks + ntext_tasks)
        throw std::runtime_error("internal error: checkpoint flag count mismatch");
    std::ofstream out(opts.checkpoint_path, std::ios::binary | std::ios::trunc);
    if (!out.good())
        throw std::runtime_error("failed to create checkpoint file "
                                 + opts.checkpoint_path.string());
    const auto header = checkpoint_header(nbinary_tasks, ntext_tasks, layout, signature);
    out.write(reinterpret_cast<const char *>(header.data()),
              static_cast<std::streamsize>(header.size() * sizeof(std::uint64_t)));
    out.write(reinterpret_cast<const char *>(completed.data()),
              static_cast<std::streamsize>(completed.size()));
    if (!out.good())
        throw std::runtime_error("failed to write checkpoint file "
                                 + opts.checkpoint_path.string());
}

class MpiFileCache
{
public:
    ~MpiFileCache()
    {
        close_all();
    }

    MPI_File get_input(int index, const std::vector<fs::path> &paths, Timing &timing)
    {
        const auto it = input_files.find(index);
        if (it != input_files.end()) return it->second;

        evict_one_input_if_needed(timing);
        MPI_File file = MPI_FILE_NULL;
        const double t0 = MPI_Wtime();
        const int ierr = MPI_File_open(MPI_COMM_SELF, paths.at(index).string().c_str(),
                                       MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        timing.read_seconds += MPI_Wtime() - t0;
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("failed to open input file "
                                     + paths.at(index).string());
        input_files[index] = file;
        return file;
    }

    MPI_File get_output(int iq, const Options &opts, Timing &timing)
    {
        const auto it = output_files.find(iq);
        if (it != output_files.end()) return it->second;

        evict_one_output_if_needed(timing);
        MPI_File file = MPI_FILE_NULL;
        const auto path = output_path_for(opts, iq);
        const double t0 = MPI_Wtime();
        const int ierr = MPI_File_open(MPI_COMM_SELF, path.string().c_str(),
                                       MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        timing.write_seconds += MPI_Wtime() - t0;
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("failed to open output file " + path.string());
        output_files[iq] = file;
        return file;
    }

    void close_all(Timing *timing = nullptr)
    {
        for (auto &[_, file]: input_files)
        {
            const double t0 = MPI_Wtime();
            MPI_File_close(&file);
            if (timing != nullptr) timing->read_seconds += MPI_Wtime() - t0;
        }
        for (auto &[_, file]: output_files)
        {
            const double t0 = MPI_Wtime();
            MPI_File_close(&file);
            if (timing != nullptr) timing->write_seconds += MPI_Wtime() - t0;
        }
        input_files.clear();
        output_files.clear();
    }

private:
    void evict_one_input_if_needed(Timing &timing)
    {
        if (input_files.size() < MAX_CACHED_FILES) return;
        auto it = input_files.begin();
        const double t0 = MPI_Wtime();
        MPI_File_close(&it->second);
        timing.read_seconds += MPI_Wtime() - t0;
        input_files.erase(it);
    }

    void evict_one_output_if_needed(Timing &timing)
    {
        if (output_files.size() < MAX_CACHED_FILES) return;
        auto it = output_files.begin();
        const double t0 = MPI_Wtime();
        MPI_File_close(&it->second);
        timing.write_seconds += MPI_Wtime() - t0;
        output_files.erase(it);
    }

    std::unordered_map<int, MPI_File> input_files;
    std::unordered_map<int, MPI_File> output_files;
};

class TextFileCache
{
public:
    std::ifstream &get_input(int index, const std::vector<fs::path> &paths,
                             Timing &timing)
    {
        const auto it = input_files.find(index);
        if (it != input_files.end()) return it->second;

        evict_one_input_if_needed();
        const double t0 = MPI_Wtime();
        auto inserted = input_files.emplace(index, std::ifstream(paths.at(index)));
        timing.read_seconds += MPI_Wtime() - t0;
        auto &stream = inserted.first->second;
        if (!stream.good())
            throw std::runtime_error("failed to open text input file "
                                     + paths.at(index).string());
        return stream;
    }

private:
    void evict_one_input_if_needed()
    {
        if (input_files.size() < MAX_CACHED_FILES) return;
        input_files.erase(input_files.begin());
    }

    std::unordered_map<int, std::ifstream> input_files;
};

class CheckpointWriter
{
public:
    ~CheckpointWriter()
    {
        close();
    }

    void open(const fs::path &path)
    {
        close();
        const int ierr = MPI_File_open(MPI_COMM_SELF, path.string().c_str(),
                                       MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("failed to open checkpoint file " + path.string());
    }

    void mark(std::size_t work_index)
    {
        if (file == MPI_FILE_NULL) return;
        const unsigned char done = 1;
        const int ierr = MPI_File_write_at(file,
                                           CHECKPOINT_HEADER_BYTES
                                               + static_cast<MPI_Offset>(work_index),
                                           const_cast<unsigned char *>(&done),
                                           1, MPI_BYTE, MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("failed to update checkpoint file");
    }

    void close()
    {
        if (file != MPI_FILE_NULL)
        {
            MPI_File_close(&file);
            file = MPI_FILE_NULL;
        }
    }

private:
    MPI_File file = MPI_FILE_NULL;
};

void mpi_read_exact(MPI_File file, MPI_Offset offset, void *buffer, std::size_t nbytes,
                    const std::string &label, Timing &timing)
{
    char *ptr = static_cast<char *>(buffer);
    std::size_t left = nbytes;
    while (left > 0)
    {
        const int chunk = static_cast<int>(std::min<std::size_t>(
            left, static_cast<std::size_t>(std::numeric_limits<int>::max())));
        MPI_Status status;
        const double t0 = MPI_Wtime();
        const int ierr = MPI_File_read_at(file, offset, ptr, chunk, MPI_BYTE, &status);
        timing.read_seconds += MPI_Wtime() - t0;
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("MPI read failed for " + label);
        int got = 0;
        MPI_Get_count(&status, MPI_BYTE, &got);
        if (got != chunk)
            throw std::runtime_error("truncated MPI read for " + label);
        offset += chunk;
        ptr += chunk;
        left -= chunk;
    }
}

void mpi_write_exact(MPI_File file, MPI_Offset offset, const void *buffer, std::size_t nbytes,
                     const std::string &label, Timing &timing)
{
    const char *ptr = static_cast<const char *>(buffer);
    std::size_t left = nbytes;
    while (left > 0)
    {
        const int chunk = static_cast<int>(std::min<std::size_t>(
            left, static_cast<std::size_t>(std::numeric_limits<int>::max())));
        const double t0 = MPI_Wtime();
        const int ierr = MPI_File_write_at(file, offset, const_cast<char *>(ptr),
                                           chunk, MPI_BYTE, MPI_STATUS_IGNORE);
        timing.write_seconds += MPI_Wtime() - t0;
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("MPI write failed for " + label);
        offset += chunk;
        ptr += chunk;
        left -= chunk;
    }
}

void write_pair_conjugate(MPI_File output, const AtomLayout &layout,
                          int atom, int row_local, int col_local,
                          const ComplexValue &value, Timing &timing)
{
    if (row_local == col_local) return;
    const ComplexValue conj_value{value.real, -value.imag};
    mpi_write_exact(output, layout.byte_offset(atom, atom, col_local, row_local),
                    &conj_value, sizeof(conj_value), "diagonal conjugate", timing);
}

void write_row_v1(MPI_File output, const AtomLayout &layout, int iq, int row,
                  int col_begin, int ncol, const ComplexValue *values,
                  Timing &timing, ProgressTracker &progress)
{
    const auto [row_atom, row_local] = layout.atom_for_aux(row);
    int col = std::max(row, col_begin);
    const int col_end = col_begin + ncol;

    while (col < col_end)
    {
        const auto [col_atom, col_local] = layout.atom_for_aux(col);
        const int span_end = std::min(col_end, layout.atom_offsets[col_atom + 1]);
        const int span_len = span_end - col;
        const auto *span_values = values + (col - col_begin);

        if (row_atom < col_atom)
        {
            mpi_write_exact(output,
                            layout.byte_offset(row_atom, col_atom, row_local, col_local),
                            span_values,
                            static_cast<std::size_t>(span_len) * sizeof(ComplexValue),
                            "off-diagonal atom-pair row", timing);
            progress.note(iq, layout.atom_pair_index(row_atom, col_atom));
        }
        else if (row_atom == col_atom)
        {
            mpi_write_exact(output,
                            layout.byte_offset(row_atom, row_atom, row_local, col_local),
                            span_values,
                            static_cast<std::size_t>(span_len) * sizeof(ComplexValue),
                            "diagonal atom-pair row", timing);
            for (int offset = 0; offset != span_len; ++offset)
            {
                write_pair_conjugate(output, layout, row_atom, row_local,
                                     col_local + offset, span_values[offset], timing);
            }
            progress.note(iq, layout.atom_pair_index(row_atom, row_atom));
        }

        col = span_end;
    }
}

void process_binary_task(const BinaryTask &task, const Options &opts, const AtomLayout &layout,
                         const std::vector<fs::path> &input_files, MpiFileCache &cache,
                         Timing &timing, ProgressTracker &progress)
{
    if (task.naux != layout.naux)
        throw std::runtime_error("legacy Naux does not match v1 atom layout");

    MPI_File input = cache.get_input(task.file_index, input_files, timing);
    MPI_File output = cache.get_output(task.iq, opts, timing);
    std::vector<ComplexValue> buffer(static_cast<std::size_t>(task.nrows) * task.ncol);
    mpi_read_exact(input, task.payload_offset, buffer.data(),
                   buffer.size() * sizeof(ComplexValue), "legacy binary Coulomb rows",
                   timing);

    for (int local_row = 0; local_row != task.nrows; ++local_row)
    {
        write_row_v1(output, layout, task.iq, task.row_begin + local_row, task.col_begin,
                     task.ncol,
                     buffer.data() + static_cast<std::size_t>(local_row) * task.ncol,
                     timing, progress);
    }
}

void process_text_task(const TextTask &task, const Options &opts, const AtomLayout &layout,
                       const std::vector<fs::path> &input_files, MpiFileCache &cache,
                       TextFileCache &text_cache, Timing &timing,
                       ProgressTracker &progress)
{
    if (task.naux != layout.naux)
        throw std::runtime_error("legacy Naux does not match v1 atom layout");

    auto &in = text_cache.get_input(task.file_index, input_files, timing);
    double t0 = MPI_Wtime();
    in.clear();
    in.seekg(static_cast<std::streamoff>(task.payload_offset), std::ios::beg);
    timing.read_seconds += MPI_Wtime() - t0;
    if (!in.good())
        throw std::runtime_error("failed to seek text Coulomb task in "
                                 + input_files.at(task.file_index).string());

    MPI_File output = cache.get_output(task.iq, opts, timing);
    for (int local_row = 0; local_row != task.nrows; ++local_row)
    {
        std::vector<ComplexValue> row_values(task.ncol);
        t0 = MPI_Wtime();
        for (int col = 0; col != task.ncol; ++col)
        {
            std::string real_s, imag_s;
            in >> real_s >> imag_s;
            if (!in.good())
                throw std::runtime_error("truncated text Coulomb task in "
                                         + input_files.at(task.file_index).string());
            row_values[col].real = parse_double_token(real_s);
            row_values[col].imag = parse_double_token(imag_s);
        }
        timing.read_seconds += MPI_Wtime() - t0;
        write_row_v1(output, layout, task.iq, task.row_begin + local_row,
                     task.col_begin, task.ncol, row_values.data(), timing, progress);
    }
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try
    {
        Options opts;
        AtomLayout layout;
        std::vector<fs::path> input_files;
        std::vector<BinaryTask> binary_tasks;
        std::vector<TextTask> text_tasks;
        std::vector<unsigned char> completed_flags;
        std::map<int, int> q_naux;
        std::int64_t total_blocks = 0;
        Timing timing;

        if (rank == 0)
        {
            opts = parse_args(argc, argv);
            fs::create_directories(opts.output_dir);
            const bool checkpoint_exists = fs::exists(opts.checkpoint_path);
            if (!opts.restart)
            {
                check_no_existing_outputs(opts);
            }
            else if (!checkpoint_exists && has_existing_outputs(opts))
            {
                throw std::runtime_error(
                    "existing output files require checkpoint file "
                    + opts.checkpoint_path.string());
            }
            layout = resolve_atom_layout(opts);
            input_files = discover_input_files(opts);

            for (int ifile = 0; ifile != static_cast<int>(input_files.size()); ++ifile)
            {
                const auto &path = input_files[ifile];
                double t0 = MPI_Wtime();
                const bool is_binary = legacy_binary_layout_matches(path);
                timing.read_seconds += MPI_Wtime() - t0;
                if (is_binary)
                {
                    t0 = MPI_Wtime();
                    scan_binary_file(path, ifile, opts.rows_per_task, binary_tasks,
                                     q_naux, total_blocks);
                    timing.read_seconds += MPI_Wtime() - t0;
                }
                else
                {
                    t0 = MPI_Wtime();
                    scan_text_file(path, ifile, opts.rows_per_task, text_tasks,
                                   q_naux, total_blocks);
                    timing.read_seconds += MPI_Wtime() - t0;
                }
            }

            const auto signature =
                conversion_signature(opts, layout, input_files, binary_tasks,
                                     text_tasks, q_naux);
            if (opts.restart && checkpoint_exists)
            {
                completed_flags =
                    read_checkpoint_file(opts, binary_tasks.size(), text_tasks.size(),
                                         layout, signature);
            }
            else
            {
                completed_flags.assign(binary_tasks.size() + text_tasks.size(), 0);
            }

            for (const auto &[iq, naux]: q_naux)
            {
                if (naux != layout.naux)
                    throw std::runtime_error("q-point " + std::to_string(iq)
                                             + " has Naux=" + std::to_string(naux)
                                             + ", but atom layout sums to "
                                             + std::to_string(layout.naux));
                if (opts.restart && checkpoint_exists)
                    validate_output_file(opts, layout, iq);
                else
                    create_output_file(opts, layout, iq, timing);
            }
            write_checkpoint_file(opts, binary_tasks.size(), text_tasks.size(),
                                  layout, signature, completed_flags);

            if (!opts.quiet)
            {
                std::cout << "Discovered " << input_files.size() << " input file(s), "
                          << binary_tasks.size() << " binary row task(s), "
                          << text_tasks.size() << " text row task(s), "
                          << q_naux.size() << " q-point output file(s)." << std::endl;
                const auto completed_count =
                    std::count(completed_flags.begin(), completed_flags.end(), 1);
                if (opts.restart && checkpoint_exists)
                    std::cout << "Restart checkpoint: " << completed_count << "/"
                              << completed_flags.size()
                              << " work item(s) already complete." << std::endl;
                std::cout << "Checkpoint file: " << opts.checkpoint_path.string()
                          << std::endl;
            }
        }

        // Broadcast options as strings/primitive values needed by all ranks.
        // Reparse command line on non-root ranks to keep this simple and deterministic.
        if (rank != 0) opts = parse_args(argc, argv);

        std::vector<int> atom_naux = layout.atom_naux;
        bcast_vector(atom_naux, 0, MPI_COMM_WORLD);
        if (rank != 0) layout = AtomLayout(atom_naux);
        bcast_paths(input_files, 0, MPI_COMM_WORLD);
        bcast_vector(completed_flags, 0, MPI_COMM_WORLD);

        auto packed_tasks = pack_tasks(binary_tasks);
        bcast_vector(packed_tasks, 0, MPI_COMM_WORLD);
        if (rank != 0) binary_tasks = unpack_tasks(packed_tasks);
        auto packed_text_tasks = pack_text_tasks(text_tasks);
        bcast_vector(packed_text_tasks, 0, MPI_COMM_WORLD);
        if (rank != 0) text_tasks = unpack_text_tasks(packed_text_tasks);

        MPI_Barrier(MPI_COMM_WORLD);

        MpiFileCache cache;
        TextFileCache text_cache;
        CheckpointWriter checkpoint;
        checkpoint.open(opts.checkpoint_path);
        ProgressTracker progress(rank, opts.progress_blocks);
        for (std::size_t itask = rank; itask < binary_tasks.size(); itask += nprocs)
        {
            if (completed_flags[itask] != 0) continue;
            process_binary_task(binary_tasks[itask], opts, layout, input_files, cache,
                                timing, progress);
            checkpoint.mark(itask);
        }
        const auto text_offset = binary_tasks.size();
        for (std::size_t itext = rank; itext < text_tasks.size(); itext += nprocs)
        {
            const auto work_index = text_offset + itext;
            if (completed_flags[work_index] != 0) continue;
            process_text_task(text_tasks[itext], opts, layout, input_files, cache,
                              text_cache, timing, progress);
            checkpoint.mark(work_index);
        }
        cache.close_all(&timing);
        checkpoint.close();

        MPI_Barrier(MPI_COMM_WORLD);
        const double local_timing[2] = {timing.read_seconds, timing.write_seconds};
        double max_timing[2] = {0.0, 0.0};
        MPI_Reduce(local_timing, max_timing, 2, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0 && !opts.quiet)
        {
            std::cout << "Converted " << total_blocks << " legacy Coulomb block(s) to v1."
                      << std::endl;
            std::cout << std::fixed << std::setprecision(6)
                      << "Input read time (max over ranks): " << max_timing[0] << " s\n"
                      << "Output write time (max over ranks): " << max_timing[1] << " s"
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            fs::remove(opts.checkpoint_path);

        MPI_Finalize();
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "rank " << rank << ": error: " << exc.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
}
