# Vehicle Inspection App

## Description
The Vehicle Inspection App leverages artificial intelligence to automatically detect vehicle damages and dents, streamlining the inspection process. This app simplifies vehicle assessments, providing accurate results and improving inspection efficiency.

## Setup Instructions
1. **Clean the project**: Run the following command to clean the project and remove any cached data:
   ```bash
   flutter clean
   flutter pub get


## setup for Android
1. Add this in the android/app/build.gradle file under the appropriate section.

```java
buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig signingConfigs.debug
        }
    }
```

```java
 externalNativeBuild {
        cmake {
            path "CMakeLists.txt"
        }
    }

    defaultConfig {
        externalNativeBuild {
            cmake {
                cppFlags '-frtti -fexceptions -std=c++17'
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
    
```
2. Add openCv2 to the Android folder
3. Add ncnn-20230816-android to the  Android folder.
4. Add JNI to the Android folder
5. Add the CMakeLists.txt file to the android/app directory.
6. Add this code inside CMakeLists.txt


```java
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
```



## setup for IOS
This project integrates NCNN and OpenCV with Flutter for iOS platforms using ffi (Foreign Function Interface), enabling advanced image processing capabilities required for AI-powered vehicle inspection.

1. Create a required Flutter plugin to install OpenCV and NCNN, as the plugin contains a podspec file.

2. In the podspec file, input the following code:


```swift
#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint ncnn_yolox_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'dartffiplugin'
  s.version          = '0.0.1'
  s.summary          = 'A new flutter plugin project.'
  s.description      = <<-DESC
  A new flutter plugin project.
                         DESC
  s.homepage         = 'https://github.com/AutomotionJo/dartFFIPlugin/'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'connectedmotion' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.dependency 'OpenCV', '4.3.0'
  s.platform = :ios, '14'

  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 arm64' }
  s.swift_version = '5.0'

  ## If you do not need to download the ncnn library, remove it. 
  ## From here
   s.prepare_command = <<-CMD
      rm -rf "ncnn.xcframework"
      rm -rf "openmp.xcframework"
      curl "https://github.com/KoheiKanagu/ncnn_yolox_flutter/releases/download/0.0.6/ncnn-ios-bitcode_xcframework.zip" -L -o "ncnn-ios-bitcode_xcframework.zip"
      unzip "ncnn-ios-bitcode_xcframework.zip"
      rm "ncnn-ios-bitcode_xcframework.zip"
    CMD
  ## Up to here

  s.preserve_paths = 'ncnn.xcframework', 'openmp.xcframework'
  s.xcconfig = { 
    'OTHER_LDFLAGS' => '-framework ncnn -framework openmp',
    'OTHER_CFLAGS' => '-DUSE_NCNN_SIMPLEOCV -DDART_FFI_PLUGIN_IOS',
  }
  s.ios.vendored_frameworks = 'ncnn.xcframework', 'openmp.xcframework'
  s.library = 'c++'
end
```

3. In the plugin directory (e.g., dartFFIPlugin/ios/dartffiplugin.podspec), run this command:
```swift
pod lib lint dartffiplugin.podspec --verbose
```

4. Navigate to the vehicle_inspection_dev directory inside the project, then run this code on iOS:
```swift
cd ios
arch -x86_64 pod update
pod install
arch -x86_64 pod update
```

5. Install ffi: ^2.0.1 in the pubspec.yaml file.
6. Please open this link to access the C++ file inside the project
https://bensonthew.medium.com/failed-to-lookup-symbol-dlsym-rtld-default-xxxx-symbol-not-found-e40216370345



## Note
If you see this issue:

Install the latest versions of OpenCV and NCNN from this link https://github.com/Tencent/ncnn/blob/master/README.md?plain=1

After downloading, replace the existing files in the 'plugin/ios' directory with the new file.

## Note
To work in archive mode, navigate to Xcode build settings. Select 'All', then 'Deployment', and finally set 'Strip Style' to 'Non-Global Symbols'.

