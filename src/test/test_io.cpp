#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "../src/io/aux_basis_summary.h"
#include "../src/io/fs.h"
#include "../src/io/global_io.h"
#include "../src/io/stl_io_helper.h"

int main (int argc, char *argv[])
{
    using namespace librpa_int;

    assert(LIBRPA_VERBOSE_INFO > LIBRPA_VERBOSE_WARN);
    global::set_output_level(LIBRPA_VERBOSE_WARN);
    assert(!global::should_output(LIBRPA_VERBOSE_INFO));
    assert(global::should_output(LIBRPA_VERBOSE_WARN));
    assert(global::should_output(LIBRPA_VERBOSE_CRITICAL));
    global::lib_printf(LIBRPA_VERBOSE_INFO, "");
    global::lib_printf(LIBRPA_VERBOSE_WARN, "");

    global::set_output_level(LIBRPA_VERBOSE_CRITICAL);
    assert(!global::should_output(LIBRPA_VERBOSE_WARN));
    assert(global::should_output(LIBRPA_VERBOSE_CRITICAL));
    global::lib_printf(LIBRPA_VERBOSE_CRITICAL, "");

    global::set_output_level(LIBRPA_VERBOSE_DEBUG);
    assert(global::should_output(LIBRPA_VERBOSE_INFO));

    int myid = 0;
    create_directories("librpa.d", myid);
    create_directories("librpa.d/nested/path", myid);
    assert(path_exists("librpa.d/nested/path"));
    create_directories("librpa.d/fs_discovery", myid);

    assert(join_path("", "file.dat") == "file.dat");
    assert(join_path("librpa.d/fs_discovery", "alpha_001.dat") ==
           "librpa.d/fs_discovery/alpha_001.dat");

    std::ofstream(join_path("librpa.d/fs_discovery", "alpha_001.dat")).close();
    std::ofstream(join_path("librpa.d/fs_discovery", "alpha_002.txt")).close();
    std::ofstream(join_path("librpa.d/fs_discovery", "beta_001.dat")).close();
    assert(is_readable_file(join_path("librpa.d/fs_discovery", "alpha_001.dat")));
    assert(!is_readable_file(join_path("librpa.d/fs_discovery", "missing.dat")));

    const auto alpha_all = discover_files_with_prefix("librpa.d/fs_discovery", "alpha_");
    assert(alpha_all.size() == 2);

    const auto dat_all = discover_files_with_suffix("librpa.d/fs_discovery", ".dat");
    assert(dat_all.size() == 2);

    const auto alpha_dat = discover_files("librpa.d/fs_discovery", "alpha_", ".dat");
    assert(alpha_dat.size() == 1);
    assert(alpha_dat[0] == join_path("librpa.d/fs_discovery", "alpha_001.dat"));

    auto throws_with = [](const std::string &path, const std::string &text)
    {
        try
        {
            require_readable_file(path);
        }
        catch (const std::runtime_error &err)
        {
            return std::string(err.what()).find(text) != std::string::npos;
        }
        return false;
    };
    assert(throws_with(join_path("librpa.d/fs_discovery", "missing.perm"), "does not exist"));
    assert(throws_with("librpa.d/nested/path", "not a regular file"));

    const auto shrink_summary = format_aux_basis_compression_summary(
        {397, 397, 210}, {167, 167, 96}, {{0, 0}, {1, 0}, {2, 1}});
    const std::string expected_shrink_summary =
        "Auxiliary basis compression summary:\n"
        "+------+----------------+----------------+\n"
        "| type | large ABF/atom | small ABF/atom |\n"
        "+------+----------------+----------------+\n"
        "|    1 |            397 |            167 |\n"
        "|    2 |            210 |             96 |\n"
        "+------+----------------+----------------+\n";
    if (shrink_summary != expected_shrink_summary)
    {
        std::cerr << "Expected shrink summary:\n" << expected_shrink_summary
                  << "Actual shrink summary:\n" << shrink_summary;
    }
    assert(shrink_summary == expected_shrink_summary);

    const auto unreadable_path = join_path("librpa.d/fs_discovery", "unreadable.perm");
    if (path_exists(unreadable_path.c_str()))
    {
        std::filesystem::permissions(
            unreadable_path,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    }
    std::ofstream(unreadable_path).close();
    std::filesystem::permissions(unreadable_path, std::filesystem::perms::none);
    assert(!is_readable_file(unreadable_path));
    assert(throws_with(unreadable_path, "lacks read permission"));
    std::filesystem::permissions(
        unreadable_path,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);

    std::map<int, std::map<int, std::map<int, double>>> nested_map
    {
        {0, {
                {0, {
                        {1, 1.0},
                        {2, 2.0},
                    }
                },
                {1, {
                        {2, 3.0},
                    }
                },
            }
        },
        {1, {
                {4, {
                        {0, -1.0},
                        {1, 1.0},
                        {2, 2.0},
                    }
                },
            }
        },
    };
    assert(get_num_keys(nested_map) == 6);

    std::ostringstream tuple_out;
    tuple_out << std::make_tuple(1, 2, 3);
    assert(tuple_out.str() == "{1,2,3}");

    std::ostringstream empty_tuple_out;
    empty_tuple_out << std::tuple<>{};
    assert(empty_tuple_out.str() == "{}");

    return 0;
}
