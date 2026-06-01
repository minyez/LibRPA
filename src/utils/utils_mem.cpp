#include "utils_mem.h"

#include <cstdio>
#include <cstdlib>
// For memory cleanup
#if defined(__linux__)
#include <malloc.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <malloc/malloc.h>  // for malloc_zone_pressure_relief and malloc_default_zone
#endif

// For memory query
#if defined(__linux__)
#include <sys/sysinfo.h>
#define PROC_DIR_AVAILABLE
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef PROC_DIR_AVAILABLE
#include <fstream>
#endif

namespace librpa_int
{

#if defined(__GLIBC__) && (__GLIBC__ * 1000 + __GLIBC_MINOR__ >= 2033)
#define USE_MALLINFO2 1
#endif

#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
namespace
{

template <typename T>
int get_bsd_hw_mem_kb(int mib_name, unsigned long long &value)
{
    T value_bytes = 0;
    int mib[2] = {CTL_HW, mib_name};
    size_t len = sizeof(value_bytes);
    const int retcode = sysctl(mib, 2, &value_bytes, &len, NULL, 0);
    if (retcode == 0)
    {
        value = static_cast<unsigned long long>(value_bytes) / 1024;
    }
    return retcode;
}

} /* end anonymous namespace */
#endif

void display_free_mem()
{
#if defined(__linux__)
    std::system("free -m");
#elif defined(__APPLE__) && defined(__MACH__)
    std::system("vm_stat | awk 'NR==1{page_size=$8} NR>1{gsub(\".\", \"\"); free+=$3} END {print \"Free memory (in MB): \" page_size*4096/1024/1024}'");
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
    std::system("sysctl hw.physmem hw.usermem");
#endif
}

void release_free_mem()
{
#if defined(__linux__)
    malloc_trim(0);
#elif defined(__APPLE__) && defined(__MACH__)
    malloc_zone_pressure_relief(malloc_default_zone(), 0);
#endif
}

int get_node_total_mem(double &total_mem)
{
    int retcode = 1;
    // value in KB unit
    unsigned long long value = 0ULL;

#if defined(__linux__)
    struct sysinfo mem_info;
    retcode = sysinfo(&mem_info);
    if (retcode == 0)
    {
        value = static_cast<unsigned long long>(mem_info.totalram) *
                mem_info.mem_unit / 1024;
    }
#elif defined(__APPLE__) && defined(__MACH__)
    unsigned long long value_bytes = 0ULL;
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    size_t len = sizeof(value_bytes);
    retcode = sysctl(mib, 2, &value_bytes, &len, NULL, 0);
    if (retcode == 0)
    {
        value = value_bytes / 1024;
    }
#elif defined(__FreeBSD__)
    retcode = get_bsd_hw_mem_kb<unsigned long>(HW_PHYSMEM, value);
#elif defined(__NetBSD__) || defined(__OpenBSD__)
#ifdef HW_PHYSMEM64
    retcode = get_bsd_hw_mem_kb<long long>(HW_PHYSMEM64, value);
#else
    retcode = get_bsd_hw_mem_kb<int>(HW_PHYSMEM, value);
#endif
#endif

    total_mem = static_cast<double>(value) * 1.e-6;
    return retcode;
}

int get_node_free_mem(double &free_mem)
{
    int retcode = 1;
    // value in KB unit
    unsigned long long value = 0ULL;

#if defined(__linux__)
    char line[1024];
    FILE *fp = NULL;

    if ((fp = fopen("/proc/meminfo", "r")) != NULL)
    {
        while (fgets(line, sizeof(line), fp))
        {
            if (sscanf(line, "MemAvailable: %llu kB", &value) == 1)
            {
                retcode = 0;
                break;
            }
        }
        fclose(fp);
    }
#elif defined(__APPLE__) && defined(__MACH__)
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    retcode = host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                reinterpret_cast<host_info64_t>(&vm_stat),
                                &count);
    if (retcode == KERN_SUCCESS)
    {
        vm_size_t page_size = 0;
        retcode = host_page_size(mach_host_self(), &page_size);
        if (retcode == KERN_SUCCESS)
        {
            const auto available_pages = static_cast<unsigned long long>(vm_stat.free_count) +
                                         vm_stat.inactive_count +
                                         vm_stat.speculative_count;
            value = available_pages * page_size / 1024;
        }
    }
#elif defined(__FreeBSD__)
    retcode = get_bsd_hw_mem_kb<unsigned long>(HW_USERMEM, value);
#elif defined(__NetBSD__) || defined(__OpenBSD__)
#ifdef HW_USERMEM64
    retcode = get_bsd_hw_mem_kb<long long>(HW_USERMEM64, value);
#else
    retcode = get_bsd_hw_mem_kb<int>(HW_USERMEM, value);
#endif
#endif

    free_mem = static_cast<double>(value) * 1.e-6;
    return retcode;
}

void report_virtual_pages(std::ostream &os)
{
#ifndef __linux__
    os << "report_virtual_pages only available on Linux, skip" << std::endl;
#else
#ifdef USE_MALLINFO2
    struct mallinfo2 mi = mallinfo2();  // field as size_t
#else
    struct mallinfo mi = mallinfo();    // field as int, may overflow
#endif
    std::ifstream ifs("/proc/self/status");
    std::string line;
    std::string vmrss, vmdata;
    while (std::getline(ifs, line)) {
        if (line.rfind("VmRSS:", 0) == 0) vmrss = line;
        if (line.rfind("VmData:", 0) == 0) vmdata = line;
    }
    os << vmrss << std::endl << vmdata << std::endl
       << "Heap arena: " << mi.arena / (1024.0*1024)
       << " MB, Free: " << mi.fordblks / (1024.0*1024)
       << " MB, Mmap space: " << mi.hblkhd / (1024.0*1024)
       << " MB" << std::endl;
#endif
}

} /* end of namespace librpa_int */
