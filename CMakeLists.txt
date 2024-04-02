cmake_minimum_required(VERSION 3.27)

project(others)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,cppcoreguidelines-*)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


link_directories(/opt/homebrew/lib)
include_directories(/opt/homebrew/include)

add_executable(others src/list2_9.cpp)

target_link_libraries(others PRIVATE fmt)
