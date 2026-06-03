# 定义宏简化重复操作
function(add_test_executable test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE ddla)
    # add_custom_command(TARGET ${test_name} POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E echo 
    # "Built at: $<TARGET_FILE:${test_name}>"
    # )
    add_test(NAME ${test_name} COMMAND ${test_name})
    # 可选：复制测试数据到输出目录
    # file(COPY data/ DESTINATION ${CMAKE_BINARY_DIR}/tests/data)
endfunction()

# 调用宏添加测试
add_test_executable(test_pzgemm)