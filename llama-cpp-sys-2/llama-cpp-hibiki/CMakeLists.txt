cmake_minimum_required(VERSION 3.14)
project(llama_cpp_hibiki)

set(CMAKE_CXX_STANDARD 14)

add_library(llama_cpp_hibiki STATIC llama_cpp_hibiki.cpp)

target_include_directories(
    llama_cpp_hibiki PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
    llama.cpp/common
)