#include "fs.h"

#include <algorithm>
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

std::string join_dir_file(const std::string &dir_path, const std::string &filename)
{
    if (dir_path.empty())
    {
        return filename;
    }

    return (std::filesystem::path(dir_path) / filename).string();
}

namespace
{

bool starts_with(const std::string &text, const std::string &prefix)
{
    return prefix.empty() || text.find(prefix) == 0;
}

bool ends_with(const std::string &text, const std::string &suffix)
{
    return suffix.empty() || (
        text.size() >= suffix.size() &&
        text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0);
}

} // namespace

std::vector<std::string> discover_files(const std::string &dir_path,
                                        const std::string &prefix,
                                        const std::string &suffix)
{
    std::vector<std::string> files;
    for (const auto &entry: std::filesystem::directory_iterator(dir_path))
    {
        const auto filename = entry.path().filename().string();
        if (starts_with(filename, prefix) && ends_with(filename, suffix))
        {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> discover_files_with_prefix(const std::string &dir_path,
                                                    const std::string &prefix)
{
    return discover_files(dir_path, prefix, "");
}

std::vector<std::string> discover_files_with_suffix(const std::string &dir_path,
                                                    const std::string &suffix)
{
    return discover_files(dir_path, "", suffix);
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
