#include <cassert>
#include <fstream>

#include "../src/io/fs.h"
#include "../src/io/stl_io_helper.h"

int main (int argc, char *argv[])
{
    using namespace librpa_int;

    int myid = 0;
    create_directories("librpa.d", myid);
    create_directories("librpa.d/nested/path", myid);
    assert(path_exists("librpa.d/nested/path"));
    create_directories("librpa.d/fs_discovery", myid);

    assert(join_dir_file("", "file.dat") == "file.dat");
    assert(join_dir_file("librpa.d/fs_discovery", "alpha_001.dat") ==
           "librpa.d/fs_discovery/alpha_001.dat");

    std::ofstream(join_dir_file("librpa.d/fs_discovery", "alpha_001.dat")).close();
    std::ofstream(join_dir_file("librpa.d/fs_discovery", "alpha_002.txt")).close();
    std::ofstream(join_dir_file("librpa.d/fs_discovery", "beta_001.dat")).close();

    const auto alpha_all = discover_files_with_prefix("librpa.d/fs_discovery", "alpha_");
    assert(alpha_all.size() == 2);

    const auto dat_all = discover_files_with_suffix("librpa.d/fs_discovery", ".dat");
    assert(dat_all.size() == 2);

    const auto alpha_dat = discover_files("librpa.d/fs_discovery", "alpha_", ".dat");
    assert(alpha_dat.size() == 1);
    assert(alpha_dat[0] == join_dir_file("librpa.d/fs_discovery", "alpha_001.dat"));

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

    return 0;
}
