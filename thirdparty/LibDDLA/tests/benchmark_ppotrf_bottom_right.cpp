#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

namespace {

using Complex = std::complex<double>;
using namespace ddla;

struct Options {
    std::vector<int> sizes = {4097, 8193};
    char uplo = 'U';
    int block_size = 128;
    int warmup = 1;
    int repetitions = 3;
    int nprows = 0;
    int npcols = 0;
    int irsrc = 0;
    int icsrc = 0;
    bool grid_set = false;
    bool show_help = false;
};

enum class Algorithm {
    standard,
    bottom_right,
};

const char* algorithm_name(Algorithm algorithm, char uplo)
{
    if(algorithm == Algorithm::bottom_right) return "ppotrf_bottom_right";
    return uplo == 'U' ? "ppotrf_upper" : "ppotrf_lower";
}

bool parse_integer(const std::string& text, bool allow_zero, int& value)
{
    if(text.empty()) return false;

    long long parsed = 0;
    for(char ch : text){
        if(ch < '0' || ch > '9') return false;
        parsed = parsed * 10 + (ch - '0');
        if(parsed > std::numeric_limits<int>::max()) return false;
    }
    if(parsed == 0 && !allow_zero) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parse_grid(const std::string& text, int& rows, int& cols)
{
    const std::size_t separator = text.find_first_of("xX");
    if(separator == std::string::npos || separator == 0
       || separator + 1 >= text.size()
       || text.find_first_of("xX", separator + 1) != std::string::npos){
        return false;
    }
    return parse_integer(text.substr(0, separator), false, rows)
        && parse_integer(text.substr(separator + 1), false, cols);
}

bool parse_source(const std::string& text, int& row, int& col)
{
    const std::size_t separator = text.find_first_of("xX,:");
    if(separator == std::string::npos || separator == 0
       || separator + 1 >= text.size()
       || text.find_first_of("xX,:", separator + 1) != std::string::npos){
        return false;
    }
    return parse_integer(text.substr(0, separator), true, row)
        && parse_integer(text.substr(separator + 1), true, col);
}

bool parse_uplo(const std::string& text, char& uplo)
{
    if(text != "U" && text != "L") return false;
    uplo = text.front();
    return true;
}

std::string usage(const char* program)
{
    std::ostringstream os;
    os << "Usage: " << program << " [options] [N ...]\n"
       << "  --n N              matrix size (repeatable; defaults: 4097 8193)\n"
       << "  --uplo U|L         triangle and factor mode (default: U)\n"
       << "  --nb NB            block size (default: 128)\n"
       << "  --warmup N         warm-up pairs (default: 1; zero is allowed)\n"
       << "  --reps N           measured pairs (default: 3)\n"
       << "  --grid RxR         square process grid (default: inferred from MPI size)\n"
       << "  --source R,C       descriptor source coordinates (default: 0,0)\n"
       << "  --help              show this message";
    return os.str();
}

bool parse_options(int argc, char** argv, Options& options, std::string& error)
{
    bool sizes_set = false;
    bool block_size_set = false;
    bool warmup_set = false;
    bool repetitions_set = false;
    bool source_set = false;
    bool uplo_set = false;

    auto add_size = [&](const std::string& text) {
        int n = 0;
        if(!parse_integer(text, false, n)){
            error = "invalid matrix size: " + text;
            return false;
        }
        if(!sizes_set){
            options.sizes.clear();
            sizes_set = true;
        }
        options.sizes.push_back(n);
        return true;
    };

    auto require_value = [&](int& index, const std::string& option,
                             std::string& value) {
        if(index + 1 >= argc){
            error = option + " requires a value";
            return false;
        }
        value = argv[++index];
        return true;
    };

    for(int i = 1; i < argc; ++i){
        const std::string arg(argv[i]);
        if(arg == "--help" || arg == "-h"){
            options.show_help = true;
        }else if(arg == "--n"){
            std::string value;
            if(!require_value(i, arg, value) || !add_size(value)) return false;
        }else if(arg.rfind("--n=", 0) == 0){
            if(!add_size(arg.substr(4))) return false;
        }else if(arg == "--uplo"){
            if(uplo_set){
                error = "--uplo was provided more than once";
                return false;
            }
            std::string value;
            uplo_set = true;
            if(!require_value(i, arg, value) || !parse_uplo(value, options.uplo)){
                if(error.empty()) error = "invalid --uplo value: " + value;
                return false;
            }
        }else if(arg.rfind("--uplo=", 0) == 0){
            if(uplo_set){
                error = "--uplo was provided more than once";
                return false;
            }
            uplo_set = true;
            if(!parse_uplo(arg.substr(7), options.uplo)){
                error = "invalid --uplo value: " + arg.substr(7);
                return false;
            }
        }else if(arg == "--nb"){
            if(block_size_set){
                error = "--nb was provided more than once";
                return false;
            }
            std::string value;
            block_size_set = true;
            if(!require_value(i, arg, value)
               || !parse_integer(value, false, options.block_size)){
                if(error.empty()) error = "invalid --nb value: " + value;
                return false;
            }
        }else if(arg.rfind("--nb=", 0) == 0){
            if(block_size_set){
                error = "--nb was provided more than once";
                return false;
            }
            block_size_set = true;
            if(!parse_integer(arg.substr(5), false, options.block_size)){
                error = "invalid --nb value: " + arg.substr(5);
                return false;
            }
        }else if(arg == "--warmup"){
            if(warmup_set){
                error = "--warmup was provided more than once";
                return false;
            }
            std::string value;
            warmup_set = true;
            if(!require_value(i, arg, value)
               || !parse_integer(value, true, options.warmup)){
                if(error.empty()) error = "invalid --warmup value: " + value;
                return false;
            }
        }else if(arg.rfind("--warmup=", 0) == 0){
            if(warmup_set){
                error = "--warmup was provided more than once";
                return false;
            }
            warmup_set = true;
            if(!parse_integer(arg.substr(9), true, options.warmup)){
                error = "invalid --warmup value: " + arg.substr(9);
                return false;
            }
        }else if(arg == "--reps" || arg == "--repeats"){
            if(repetitions_set){
                error = "--reps/--repeats was provided more than once";
                return false;
            }
            std::string value;
            repetitions_set = true;
            if(!require_value(i, arg, value)
               || !parse_integer(value, false, options.repetitions)){
                if(error.empty()) error = "invalid " + arg + " value: " + value;
                return false;
            }
        }else if(arg.rfind("--reps=", 0) == 0
                 || arg.rfind("--repeats=", 0) == 0){
            if(repetitions_set){
                error = "--reps/--repeats was provided more than once";
                return false;
            }
            repetitions_set = true;
            const std::size_t offset = arg.rfind("--reps=", 0) == 0 ? 7 : 10;
            if(!parse_integer(arg.substr(offset), false, options.repetitions)){
                error = "invalid repetitions value: " + arg.substr(offset);
                return false;
            }
        }else if(arg == "--grid"){
            if(options.grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            std::string value;
            options.grid_set = true;
            if(!require_value(i, arg, value)
               || !parse_grid(value, options.nprows, options.npcols)){
                if(error.empty()) error = "invalid --grid value: " + value;
                return false;
            }
        }else if(arg.rfind("--grid=", 0) == 0){
            if(options.grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            options.grid_set = true;
            if(!parse_grid(arg.substr(7), options.nprows, options.npcols)){
                error = "invalid --grid value: " + arg.substr(7);
                return false;
            }
        }else if(arg == "--source"){
            if(source_set){
                error = "--source was provided more than once";
                return false;
            }
            std::string value;
            source_set = true;
            if(!require_value(i, arg, value)
               || !parse_source(value, options.irsrc, options.icsrc)){
                if(error.empty()) error = "invalid --source value: " + value;
                return false;
            }
        }else if(arg.rfind("--source=", 0) == 0){
            if(source_set){
                error = "--source was provided more than once";
                return false;
            }
            source_set = true;
            if(!parse_source(arg.substr(9), options.irsrc, options.icsrc)){
                error = "invalid --source value: " + arg.substr(9);
                return false;
            }
        }else if(!arg.empty() && arg[0] == '-'){
            error = "unknown option: " + arg;
            return false;
        }else if(!add_size(arg)){
            return false;
        }
    }
    return true;
}

bool validate_options(Options& options, int nranks, std::string& error)
{
    if(!options.grid_set){
        const int side = static_cast<int>(std::llround(std::sqrt(static_cast<double>(nranks))));
        if(side <= 0 || side * side != nranks){
            error = "MPI size must be a perfect square for this benchmark";
            return false;
        }
        options.nprows = side;
        options.npcols = side;
    }

    if(options.nprows != options.npcols){
        error = "bottom-right Cholesky benchmark requires a square process grid";
        return false;
    }
    if(static_cast<long long>(options.nprows) * options.npcols != nranks){
        std::ostringstream os;
        os << "--grid " << options.nprows << "x" << options.npcols
           << " requires " << options.nprows * options.npcols
           << " MPI ranks, but this run has " << nranks;
        error = os.str();
        return false;
    }
    if(options.irsrc >= options.nprows || options.icsrc >= options.npcols){
        std::ostringstream os;
        os << "--source " << options.irsrc << "," << options.icsrc
           << " lies outside grid " << options.nprows << "x" << options.npcols;
        error = os.str();
        return false;
    }
    if(options.irsrc != options.icsrc){
        error = "the ppotrf comparison baseline requires aligned row/column sources";
        return false;
    }
    return true;
}

std::size_t local_size(const ddla::DdlaDesc& desc)
{
    return static_cast<std::size_t>(desc.lld()) * desc.n_loc();
}

std::vector<Complex> make_local_hpd(const ddla::DdlaDesc& desc)
{
    const int n = desc.n();
    const double scale = 1.0 / static_cast<double>(n);
    const double two_pi = 2.0 * std::acos(-1.0);

    std::vector<Complex> phase(n);
    for(int i = 0; i < n; ++i){
        const double x = static_cast<double>(i) / static_cast<double>(n);
        const double angle = two_pi * (x + 0.125 * x * x);
        phase[i] = Complex(std::cos(angle), std::sin(angle));
    }

    // A = I + v*v^H/n has eigenvalues 1 and 2, so both factorizations see
    // exactly the same dense, well-conditioned Hermitian positive-definite A.
    std::vector<Complex> local(local_size(desc), Complex(0.0, 0.0));
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        if(j >= n) continue;
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            if(i >= n) continue;
            Complex value = scale * phase[i] * std::conj(phase[j]);
            if(i == j) value += Complex(1.0, 0.0);
            local[iloc + static_cast<std::size_t>(jloc) * desc.lld()] = value;
        }
    }
    return local;
}

void reset_matrix(const ddla::DdlaHandle_t& handle, Complex* d_work,
                  const Complex* d_reference, std::size_t count)
{
    if(count > 0){
        DEVICE_CHECK(deviceMemcpyAsync(d_work, d_reference, count * sizeof(Complex),
                                       deviceMemcpyDeviceToDevice, handle->stream));
    }
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

double run_once(Algorithm algorithm, int n, Complex* d_work,
                const Complex* d_reference, std::size_t count,
                const ddla::DdlaDesc& desc,
                const Options& options, const ddla::DdlaHandle_t& handle)
{
    reset_matrix(handle, d_work, d_reference, count);

    int info = -1;
    bool sign_correction = false;
    MPI_CHECK(MPI_Barrier(handle->comm));
    const double start = MPI_Wtime();
    if(algorithm == Algorithm::standard){
        sign_correction = ddla::ppotrf(
            options.uplo, n, d_work, 1, 1, desc, info);
    }else{
        ddla::ppotrf_bottom_right(options.uplo, n, d_work, desc, info);
    }
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    const double local_elapsed = MPI_Wtime() - start;

    const int local_failure = (info != 0 || sign_correction) ? 1 : 0;
    int global_failure = 0;
    MPI_CHECK(MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT,
                            MPI_MAX, handle->comm));
    if(global_failure != 0){
        if(handle->myid == 0){
            std::cerr << algorithm_name(algorithm, options.uplo)
                      << " failed for n=" << n
                      << ", info=" << info
                      << ", sign_correction=" << sign_correction << std::endl;
        }
        MPI_Abort(handle->comm, 1);
    }

    double max_elapsed = 0.0;
    MPI_CHECK(MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE,
                            MPI_MAX, handle->comm));
    return max_elapsed;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if(values.size() % 2 != 0) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

void print_run(const char* kind, Algorithm algorithm, int iteration,
               int n, const Options& options, double elapsed)
{
    std::cout << kind
              << " algorithm=" << algorithm_name(algorithm, options.uplo)
              << " iteration=" << iteration + 1
              << " n=" << n
              << " nb=" << options.block_size
              << " uplo=" << options.uplo
              << " grid=" << options.nprows << "x" << options.npcols
              << " source=" << options.irsrc << "," << options.icsrc
              << " time_s=" << std::fixed << std::setprecision(6) << elapsed
              << std::endl;
}

void benchmark_size(int n, const Options& options,
                    const ddla::DdlaHandle_t& handle)
{
    ddla::DdlaDesc desc(handle);
    desc.init(n, n, options.block_size, options.block_size,
              options.irsrc, options.icsrc);

    const std::size_t count = local_size(desc);
    const std::size_t allocation_count = std::max<std::size_t>(count, 1);
    Complex* d_reference = nullptr;
    Complex* d_work = nullptr;
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_reference),
                                   allocation_count * sizeof(Complex), handle->stream));
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_work),
                                   allocation_count * sizeof(Complex), handle->stream));

    {
        const std::vector<Complex> h_reference = make_local_hpd(desc);
        if(count > 0){
            DEVICE_CHECK(deviceMemcpyAsync(d_reference, h_reference.data(),
                                           count * sizeof(Complex),
                                           deviceMemcpyHostToDevice, handle->stream));
        }
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    }

    for(int iteration = 0; iteration < options.warmup; ++iteration){
        const double standard = run_once(Algorithm::standard, n, d_work,
                                         d_reference, count, desc, options, handle);
        const double bottom_right = run_once(Algorithm::bottom_right, n, d_work,
                                             d_reference, count, desc, options, handle);
        if(handle->myid == 0){
            print_run("WARMUP", Algorithm::standard, iteration,
                      n, options, standard);
            print_run("WARMUP", Algorithm::bottom_right, iteration,
                      n, options, bottom_right);
        }
    }

    std::vector<double> standard_times;
    std::vector<double> bottom_right_times;
    standard_times.reserve(options.repetitions);
    bottom_right_times.reserve(options.repetitions);
    for(int iteration = 0; iteration < options.repetitions; ++iteration){
        double standard = 0.0;
        double bottom_right = 0.0;
        // Alternate call order to avoid giving either algorithm a persistent
        // first/second-run cache or thermal advantage.
        if(iteration % 2 == 0){
            standard = run_once(Algorithm::standard, n, d_work,
                                d_reference, count, desc, options, handle);
            bottom_right = run_once(Algorithm::bottom_right, n, d_work,
                                    d_reference, count, desc, options, handle);
        }else{
            bottom_right = run_once(Algorithm::bottom_right, n, d_work,
                                    d_reference, count, desc, options, handle);
            standard = run_once(Algorithm::standard, n, d_work,
                                d_reference, count, desc, options, handle);
        }
        standard_times.push_back(standard);
        bottom_right_times.push_back(bottom_right);
        if(handle->myid == 0){
            print_run("RUN", Algorithm::standard, iteration,
                      n, options, standard);
            print_run("RUN", Algorithm::bottom_right, iteration,
                      n, options, bottom_right);
        }
    }

    if(handle->myid == 0){
        const double standard_median = median(standard_times);
        const double bottom_right_median = median(bottom_right_times);
        std::cout << "RESULT"
                  << " n=" << n
                  << " type=complex<double>"
                  << " nb=" << options.block_size
                  << " uplo=" << options.uplo
                  << " grid=" << options.nprows << "x" << options.npcols
                  << " ranks=" << options.nprows * options.npcols
                  << " source=" << options.irsrc << "," << options.icsrc
                  << " warmup=" << options.warmup
                  << " reps=" << options.repetitions
                  << (options.uplo == 'U'
                      ? " ppotrf_upper_median_s="
                      : " ppotrf_lower_median_s=")
                  << std::fixed << std::setprecision(6)
                  << standard_median
                  << " ppotrf_bottom_right_median_s=" << bottom_right_median
                  << " speedup=" << standard_median / bottom_right_median
                  << std::endl;
    }

    DEVICE_CHECK(deviceFreeAsync(d_reference, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_work, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nranks = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Options options;
    std::string error;
    if(!parse_options(argc, argv, options, error)){
        if(rank == 0){
            std::cerr << error << "\n" << usage(argv[0]) << std::endl;
        }
        MPI_Finalize();
        return 2;
    }
    if(options.show_help){
        if(rank == 0) std::cout << usage(argv[0]) << std::endl;
        MPI_Finalize();
        return 0;
    }
    if(!validate_options(options, nranks, error)){
        if(rank == 0){
            std::cerr << error << "\n" << usage(argv[0]) << std::endl;
        }
        MPI_Finalize();
        return 2;
    }

    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddla_init(handle);
    ddla::ddla_set(handle, MPI_COMM_WORLD, options.nprows, options.npcols);

    if(handle->myid == 0){
        std::cout << "=== distributed Cholesky benchmark: complex<double> ===\n"
                  << "matrix=I+v*v^H/n"
                  << " uplo=" << options.uplo
                  << " grid=" << options.nprows << "x" << options.npcols
                  << " source=" << options.irsrc << "," << options.icsrc
                  << " nb=" << options.block_size
                  << " warmup=" << options.warmup
                  << " reps=" << options.repetitions
                  << std::endl;
    }

    for(int n : options.sizes){
        benchmark_size(n, options, handle);
    }

    ddla::ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
