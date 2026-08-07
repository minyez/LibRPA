add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(
  test_qsgw_inputfile
  test_qsgw_inputfile.cpp
  ../driver.cpp
  ../input_parser.cpp
  ../inputfile.cpp
)
target_link_libraries(test_qsgw_inputfile PRIVATE rpa_lib)
add_test(NAME test_qsgw_inputfile COMMAND $<TARGET_FILE:test_qsgw_inputfile>)
