cmake_minimum_required(VERSION 3.13)

project(LibDDLA NONE)

file(READ ${PROJECT_SOURCE_DIR}/src/version.h version_file)
string(REGEX MATCH "LIBDDLA_MAJOR_VERSION ([0-9]+)" libddla_major_version ${version_file})
set(libddla_major_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBDDLA_MINOR_VERSION ([0-9]+)" libddla_minor_version ${version_file})
set(libddla_minor_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBDDLA_MICRO_VERSION ([0-9]+)" libddla_micro_version ${version_file})
set(libddla_micro_version ${CMAKE_MATCH_1})
set(version "${libddla_major_version}.${libddla_minor_version}.${libddla_micro_version}")

message(STATUS "LibDDLA version: ${version}")

set(CMAKE_PROJECT_SOVERSION_MAJOR "${libddla_major_version}")
set(CMAKE_PROJECT_VERSION "${version}")
set(LIB_NAME            "ddla")
set(math_libs "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_TESTS "Build test programs" OFF) 
option(DDLA_USE_CUDA "Build with CUDA" OFF)
option(DDLA_USE_HIP "Build with HIP" OFF)
option(DDLA_USE_DEBUG "DEBUG for program" OFF)
option(DDLA_USE_CCL "ENABLE NCCL or RCCL" OFF)
option(DDLA_USE_GPU_CPU_TUNNEL "use gpu cpu tunnel for communication" OFF)
if(DDLA_USE_DEBUG)
add_compile_definitions("DDLA_USE_DEBUG")
endif()
if(DDLA_USE_CCL)
add_compile_definitions("DDLA_USE_CCL")
endif()
if(DDLA_USE_GPU_CPU_TUNNEL)
add_compile_definitions("DDLA_USE_GPU_CPU_TUNNEL")
endif()
if(DDLA_USE_CUDA AND DDLA_USE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP")
endif()
if ((NOT DDLA_USE_CUDA) AND (NOT DDLA_USE_HIP))
    message(FATAL_ERROR "Must enable either CUDA or HIP")
endif()
if(DDLA_USE_CUDA)
    add_compile_definitions("DDLA_USE_CUDA")
endif()
if(DDLA_USE_HIP)
    add_compile_definitions("DDLA_USE_HIP")
    add_compile_definitions("__HIP_PLATFORM_AMD__")
endif()

enable_language(CXX)
include(GNUInstallDirs)
include_directories(src)
include_directories(include)
find_package(MPI REQUIRED)
list(APPEND math_libs MPI::MPI_CXX)



if(BUILD_TESTS)
    # enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "DDLA_USE_CCL: ${DDLA_USE_CCL}")
message(STATUS "DDLA_USE_GPU_CPU_TUNNEL: ${DDLA_USE_GPU_CPU_TUNNEL}")
if (DDLA_USE_CCL AND DDLA_USE_GPU_CPU_TUNNEL)
    message(WARNING "DDLA_USE_CCL and DDLA_USE_GPU_CPU_TUNNEL are both enabled, the library will need ccl library but it will not be used, if you want use ccl communication, please set DDLA_USE_GPU_CPU_TUNNEL to OFF")
endif()
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")


if(DDLA_USE_HIP)
    enable_language(HIP)
    # if(NOT EXISTS ${ROCM_PATH})
    #     message(FATAL_ERROR "ROCm not found at ${ROCM_PATH}")
    # endif()
    # list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    # find_package(hip REQUIRED)
    # find_package(amd_comgr REQUIRED)
    # set(CMAKE_HIP_STANDARD 17)
    # set(CMAKE_HIP_STANDARD_REQUIRED True)
    add_compile_definitions("DDLA_USE_HIP")
    add_compile_definitions("__HIP_PLATFORM_AMD__")
    set(HIP_LIBRARIES "-lhipblas -lrccl -lhipsolver -lamdhip64 -lhiprand -lgalaxyhip -lomp")
    # -lamdhip64 -lhsa-runtime64
    # find_package(HIP REQUIRED)
    list(APPEND math_libs
        ${HIP_LIBRARIES}
    )
endif()
# cuda setting
if(DDLA_USE_CUDA)
    add_compile_definitions("DDLA_USE_CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    # Use CUDAToolkit for the runtime and to locate the toolkit, then pick up
    # the math libraries (cublas, curand, cusolver) from the HPC SDK math_libs tree.
    find_package(CUDAToolkit REQUIRED)
    get_filename_component(_cuda_lib_parent "${CUDAToolkit_LIBRARY_DIR}" DIRECTORY)
    get_filename_component(_cuda_version "${_cuda_lib_parent}" NAME)
    get_filename_component(_cuda_root "${_cuda_lib_parent}" DIRECTORY)
    get_filename_component(_sdk_root "${_cuda_root}" DIRECTORY)
    set(_math_lib_dir "${_sdk_root}/math_libs/${_cuda_version}/lib64")
    message(STATUS "Looking for CUDA math libraries in: ${_math_lib_dir}")

    find_library(CUBLAS_LIBRARY NAMES cublas PATHS ${_math_lib_dir} NO_DEFAULT_PATH REQUIRED)
    find_library(CURAND_LIBRARY NAMES curand PATHS ${_math_lib_dir} NO_DEFAULT_PATH REQUIRED)
    find_library(CUSOLVER_LIBRARY NAMES cusolver PATHS ${_math_lib_dir} NO_DEFAULT_PATH REQUIRED)

    list(APPEND math_libs
        ${CUBLAS_LIBRARY}
        ${CURAND_LIBRARY}
        ${CUSOLVER_LIBRARY}
        ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    )

    if(DDLA_USE_CCL)
        # NCCL is not shipped with the CUDA toolkit in this HPC SDK layout;
        # try the CMake config first, otherwise fall back to the plain library.
        find_package(NCCL QUIET)
        if(NCCL_FOUND)
            list(APPEND math_libs NCCL::nccl)
        else()
            message(STATUS "NCCL CMake package not found, linking -lnccl directly")
            list(APPEND math_libs nccl)
        endif()
    endif()
endif(DDLA_USE_CUDA)
add_subdirectory(src)
