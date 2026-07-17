#ifndef DDLA_BENCHMARK_GRID_OPTIONS_H
#define DDLA_BENCHMARK_GRID_OPTIONS_H

#include <cctype>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace benchmark_cli {

struct Options {
    int nprows = 2;
    int npcols = 2;
    int repeats = 1;
    std::vector<int> sizes = {500, 5000, 10000, 15000};
};

inline bool parse_positive_int(const std::string& text, int& value)
{
    if(text.empty()) return false;
    long long parsed = 0;
    for(char ch : text){
        if(!std::isdigit(static_cast<unsigned char>(ch))) return false;
        parsed = parsed * 10 + (ch - '0');
        if(parsed > std::numeric_limits<int>::max()) return false;
    }
    if(parsed <= 0) return false;
    value = static_cast<int>(parsed);
    return true;
}

inline bool parse_grid_spec(const std::string& spec, int& nprows, int& npcols)
{
    const size_t separator = spec.find_first_of("xX");
    if(separator == std::string::npos || separator == 0
       || separator + 1 >= spec.size()
       || spec.find_first_of("xX", separator + 1) != std::string::npos){
        return false;
    }
    return parse_positive_int(spec.substr(0, separator), nprows)
        && parse_positive_int(spec.substr(separator + 1), npcols);
}

inline bool parse(int argc, char** argv, bool allow_repeats,
                  int default_repeats, Options& options, std::string& error)
{
    options.repeats = default_repeats;
    bool sizes_set = false;
    bool grid_set = false;
    bool repeats_set = false;

    for(int i = 1; i < argc; ++i){
        const std::string arg(argv[i]);
        if(arg == "--grid"){
            if(grid_set || i + 1 >= argc){
                error = grid_set ? "--grid was provided more than once"
                                 : "--grid requires a value like 2x3";
                return false;
            }
            grid_set = true;
            if(!parse_grid_spec(argv[++i], options.nprows, options.npcols)){
                error = "invalid --grid value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--grid=", 0) == 0){
            if(grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            grid_set = true;
            if(!parse_grid_spec(arg.substr(7), options.nprows, options.npcols)){
                error = "invalid --grid value: " + arg.substr(7);
                return false;
            }
        }else if(arg == "--repeats"){
            if(!allow_repeats){
                error = "--repeats is not supported by this benchmark";
                return false;
            }
            if(repeats_set || i + 1 >= argc){
                error = repeats_set ? "--repeats was provided more than once"
                                    : "--repeats requires a positive integer";
                return false;
            }
            repeats_set = true;
            if(!parse_positive_int(argv[++i], options.repeats)){
                error = "invalid --repeats value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--repeats=", 0) == 0){
            if(!allow_repeats || repeats_set){
                error = !allow_repeats ? "--repeats is not supported by this benchmark"
                                       : "--repeats was provided more than once";
                return false;
            }
            repeats_set = true;
            if(!parse_positive_int(arg.substr(10), options.repeats)){
                error = "invalid --repeats value: " + arg.substr(10);
                return false;
            }
        }else if(!arg.empty() && arg[0] == '-'){
            error = "unknown option: " + arg;
            return false;
        }else{
            int size = 0;
            if(!parse_positive_int(arg, size)){
                error = "invalid matrix size: " + arg;
                return false;
            }
            if(!sizes_set){
                options.sizes.clear();
                sizes_set = true;
            }
            options.sizes.push_back(size);
        }
    }
    return true;
}

inline std::string grid_name(const Options& options)
{
    return std::to_string(options.nprows) + "x" + std::to_string(options.npcols);
}

inline std::string usage(const char* program, bool allow_repeats)
{
    std::ostringstream os;
    os << "Usage: " << program << " [--grid RxC]";
    if(allow_repeats) os << " [--repeats N]";
    os << " [500 5000 10000 15000]";
    return os.str();
}

} // namespace benchmark_cli

#endif // DDLA_BENCHMARK_GRID_OPTIONS_H
