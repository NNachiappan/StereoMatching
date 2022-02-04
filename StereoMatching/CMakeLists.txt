# CMakeList.txt : CMake project for StereoDepth, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
enable_language(CUDA)

find_package(CUDA  REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add source to this project's executable.
add_executable (StereoMatching "StereoMatching.cpp" "disparity_gpu.h" "disparity_gpu.cu")

target_link_libraries( StereoMatching ${OpenCV_LIBS} )
target_link_libraries( StereoMatching ${CUDA_LIBS} )

# TODO: Add tests and install targets if needed.
