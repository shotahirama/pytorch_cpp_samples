cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet_cpp_sample)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(example-app main.cpp)
target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
