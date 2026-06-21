add_executable(rpa_exe)
set_target_properties(rpa_exe PROPERTIES
  OUTPUT_NAME ${LIBRPA_DRIVER_NAME}
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

list(APPEND exe_sources
  driver.cpp
  inputfile.cpp
  main.cpp
  read_data.cpp
  reader_eigenvec.cpp
  reader_basis.cpp
  reader_structure.cpp
  reader_lri.cpp
  reader_coulomb.cpp
  task.cpp
  # Specific task implementations
  tasks/task_helper.cpp
  tasks/exx.cpp
  tasks/exx_band.cpp
  tasks/g0w0.cpp
  tasks/g0w0_band.cpp
  tasks/rpa.cpp
  tasks/print_minimax.cpp
  # tasks/test.cpp
  # task_qsgw.cpp
  # task_qsgwA.cpp
  # task_qsgw_band.cpp
  # task_hf_band.cpp
  # task_scRPA.cpp
  # task_scRPA_band.cpp
  # task_screened_coulomb.cpp
  )

target_sources(rpa_exe PRIVATE ${exe_sources})

target_link_libraries(rpa_exe PRIVATE rpa_lib)
if(BUILD_SHARED_LIBS)
  target_link_libraries(rpa_exe PRIVATE ${math_libs} ${parallel_libs})
endif()

install(TARGETS rpa_exe DESTINATION ${CMAKE_INSTALL_BINDIR})
