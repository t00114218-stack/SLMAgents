cmake_minimum_required(VERSION 3.22.1)
project("slm_agents_offline")

add_library(
    slm_llama_bridge
    SHARED
    llama-android-bridge.cpp
)

find_library(
    log-lib
    log
)

target_include_directories(
    slm_llama_bridge
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    slm_llama_bridge
    ${log-lib}
)
