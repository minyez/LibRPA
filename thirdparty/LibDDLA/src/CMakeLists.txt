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
    pgetf2_panel.cpp
    pgetrf.cpp
    pgetrf_bpiv.cpp
    plapiv.cpp
    pswap.cpp
    pgetrs.cpp
    pgesv.cpp
    pgemm.cpp
    pgeadd.cpp
    ppotrf.cpp
    ppotrs.cpp
    pposv.cpp
    transport_block.cpp
    ddla_handle_t.cpp
    ddla_desc.cpp
)

target_sources(ddla_lib PRIVATE ${lib_sources})
target_link_libraries(ddla_lib PUBLIC ${math_libs})

install(TARGETS ddla_lib
    ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)