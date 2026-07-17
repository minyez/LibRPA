add_library(ddla_lib SHARED)


set_target_properties(ddla_lib PROPERTIES
  OUTPUT_NAME ${LIB_NAME}
  VERSION ${CMAKE_PROJECT_VERSION}
  SOVERSION ${CMAKE_PROJECT_SOVERSION_MAJOR}
)
message(STATUS "Building ddla library version ${CMAKE_PROJECT_VERSION}")



list(APPEND lib_sources
    ddla_stream.cpp
    ddla_utils.cpp
    ptrtrs.cpp
    pgetf2.cpp
    pgetf2_kernels.cpp
    pgetf2_panel.cpp
    pgetrf.cpp
    pgetrf_bpiv.cpp
    pgetrf_nopiv.cpp
    getrf_nopiv.cpp
    plapiv.cpp
    pswap.cpp
    pgetrs.cpp
    pgetrs_nopiv.cpp
    pgesv.cpp
    pgesv_nopiv.cpp
    pgemm.cpp
    pgeadd.cpp
    pdam.cpp
    ppotrf.cpp
    potrf_bottom_right.cpp
    potrf_bottom_right_transform.cpp
    ppotrf_bottom_right.cpp
    ppotrs.cpp
    pposv.cpp
    transport_block.cpp
    ptran.cpp
    ddla_handle_t.cpp
    ddla_desc.cpp
)

if(DDLA_USE_CUDA)
    set_source_files_properties(
        getrf_nopiv.cpp
        pdam.cpp
        pgetf2_kernels.cpp
        potrf_bottom_right_transform.cpp
        ptran.cpp
        PROPERTIES
        LANGUAGE CUDA
    )
endif()

if(DDLA_USE_HIP)
    set_source_files_properties(
        getrf_nopiv.cpp
        pdam.cpp
        pgetf2_kernels.cpp
        potrf_bottom_right_transform.cpp
        ptran.cpp
        PROPERTIES
        LANGUAGE HIP
    )
endif()

target_sources(ddla_lib PRIVATE ${lib_sources})
target_link_libraries(ddla_lib PUBLIC ${math_libs})

install(TARGETS ddla_lib
    ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
