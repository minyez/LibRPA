#include "fs.h"

#include <filesystem>
#include <system_error>

#include "../utils/error.h"

namespace librpa_int
{

std::string path_as_directory(const std::string &path)
{
    if (path.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("dirpath is empty");
    }

    if (path.find(":") != std::string::npos)
    {
        throw LIBRPA_RUNTIME_ERROR("dirpath contains invalid character (:) for POSIX path");
    }

    if (path.back() != '/')
    {
        return path + '/';
    }

    return path;
}

bool path_exists(const char *path_cstr)
{
    return path_cstr != nullptr && std::filesystem::exists(path_cstr);
}

void create_directories(const char *dname, int root_process)
{
    if (dname == nullptr || dname[0] == '\0')
    {
        throw LIBRPA_RUNTIME_ERROR("directory path is empty");
    }

    if (std::filesystem::is_directory(dname) || root_process != 0) return;

    std::error_code ec;
    std::filesystem::create_directories(dname, ec);
    if (!std::filesystem::is_directory(dname))
    {
        throw LIBRPA_RUNTIME_ERROR(std::string("Failed to create directories ") + dname);
    }
}

}
