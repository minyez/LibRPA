#include "fs.h"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <string>
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

    if (path.back() != '/' && path.back() != '\\')
    {
        return path + std::filesystem::path::preferred_separator;
    }

    return path;
}

std::string parent_path(const std::string &file_path)
{
    const auto parent = std::filesystem::path(file_path).parent_path().string();
    return parent.empty() ? "." : parent;
}

std::string base_name(const std::string &file_path)
{
    return std::filesystem::path(file_path).filename().string();
}

bool is_absolute_path(const std::string &file_path)
{
    return std::filesystem::path(file_path).is_absolute();
}

std::string join_path(const std::string &dir_path, const std::string &file_name)
{
    if (dir_path.empty())
    {
        return file_name;
    }

    return (std::filesystem::path(dir_path) / file_name).string();
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

std::string unreadable_file_reason(const std::string &file_path)
{
    if (file_path.empty())
    {
        return "File path is empty";
    }
    if (ends_with(std::filesystem::path(file_path).filename().string(), ".lock"))
    {
        return "Lock file is not valid input: " + file_path;
    }

    std::error_code ec;
    if (!std::filesystem::exists(file_path, ec))
    {
        if (ec)
        {
            return "Cannot inspect file " + file_path + ": " + ec.message();
        }
        return "File does not exist: " + file_path;
    }

    const auto status = std::filesystem::status(file_path, ec);
    if (ec)
    {
        return "Cannot inspect file permissions: " + file_path + ": " + ec.message();
    }
    if (!std::filesystem::is_regular_file(status))
    {
        return "Path is not a regular file: " + file_path;
    }

    errno = 0;
    std::ifstream input(file_path, std::ios::binary);
    if (!input.is_open())
    {
        const int open_errno = errno;
        std::string reason = "Cannot open file for reading: " + file_path;
        if (open_errno != 0)
        {
            reason += ": " + std::error_code(open_errno, std::generic_category()).message();
        }
        return reason;
    }
    return "";
}

} // namespace

bool file_exists(const std::string &file_path)
{
    std::error_code ec;
    return std::filesystem::exists(file_path, ec);
}

bool is_readable_file(const std::string &file_path)
{
    return unreadable_file_reason(file_path).empty();
}

void require_readable_file(const std::string &file_path)
{
    const auto reason = unreadable_file_reason(file_path);
    if (!reason.empty())
    {
        throw LIBRPA_RUNTIME_ERROR(reason);
    }
}

std::vector<std::string> discover_files(const std::string &dir_path,
                                        const std::string &prefix,
                                        const std::string &suffix)
{
    std::vector<std::string> files;
    for (const auto &entry: std::filesystem::directory_iterator(dir_path))
    {
        const auto filename = entry.path().filename().string();
        if (ends_with(filename, ".lock")) continue;
        if (starts_with(filename, prefix) && ends_with(filename, suffix))
        {
            const auto file_path = entry.path().string();
            require_readable_file(file_path);
            files.push_back(file_path);
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
    return path_cstr != nullptr && file_exists(path_cstr);
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
