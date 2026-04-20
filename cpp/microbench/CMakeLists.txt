set(MICROBENCH_COMMON_SOURCES
    common.cpp
    dummy_data.cpp
)

add_executable(kvtensor_kv_bench
    kv_bench.cpp
    ${MICROBENCH_COMMON_SOURCES}
)
target_link_libraries(kvtensor_kv_bench PRIVATE simpledb)
target_include_directories(kvtensor_kv_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(leveldb_FOUND)
    target_link_libraries(kvtensor_kv_bench PRIVATE leveldb::leveldb)
endif()

add_executable(kvtensor_compute_bench
    compute_bench.cpp
    ${MICROBENCH_COMMON_SOURCES}
)
target_link_libraries(kvtensor_compute_bench PRIVATE kvtensor)
target_include_directories(kvtensor_compute_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(kvtensor_runtime_bench
    runtime_bench.cpp
    ${MICROBENCH_COMMON_SOURCES}
)
target_link_libraries(kvtensor_runtime_bench PRIVATE kvtensor)
target_include_directories(kvtensor_runtime_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
