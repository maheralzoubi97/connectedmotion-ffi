cmake_minimum_required(VERSION 3.18.1)
project(connectedmotion_ffi LANGUAGES C CXX)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20230816-android/${ANDROID_ABI}/include/ncnn
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

# Import OpenCV library
add_library(lib_opencv STATIC IMPORTED)
set_target_properties(lib_opencv PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jni/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so
)

# Enable OpenMP for C and C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

# Static linking for OpenMP on newer NDK versions
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()

# Find and configure ncnn
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230816-android/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Configure ncnn properties
set_target_properties(
    ncnn PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-frtti;-fexceptions"
)

# Define source files
set(SOURCES
    ../ios/Classes/opencv-cpp/main.cpp
)

# Find the log library
find_library(
    log-lib
    log
)

# Create the shared library
add_library(OpenCV_ffi SHARED ${SOURCES})

# Set compile definitions
target_compile_definitions(OpenCV_ffi PUBLIC USE_NCNN_SIMPLEOCV)

# Link libraries
target_link_libraries(OpenCV_ffi
    lib_opencv
    ncnn
    ${log-lib}
)