cmake_minimum_required(VERSION 3.10)
project(OpenCVApp)

set(CMAKE_CXX_STANDARD 17)

# IMPORTANT: MSYS2 OpenCV path
set(OpenCV_DIR "C:/msys64/mingw64/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(app main.cpp)

target_link_libraries(app ${OpenCV_LIBS})