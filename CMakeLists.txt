cmake_minimum_required(VERSION 3.15)
project(capstone)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(capstone test_cam.cpp)
target_link_libraries(capstone ${OpenCV_LIBS})

