#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint connectedmotion_ffi.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'connectedmotion_ffi'
  s.version          = '0.0.1+1'
  s.summary          = 'A Flutter plugin integrating OpenCV and NCNN for AI and image processing.'
  s.description      = <<-DESC
  This Flutter plugin provides an integration with OpenCV and NCNN for AI-powered image processing.
  It enables optimized computer vision and deep learning functionalities using Dart FFI.
  DESC
  s.homepage         = 'https://github.com/maheralzoubi97/connectedmotion-ffi'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'connectedmotion' }
  s.source           = { :git => 'https://github.com/maheralzoubi97/connectedmotion-ffi.git', :tag => s.version.to_s }
  
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.dependency 'OpenCV', '4.3.0'

  s.platform = :ios, '14'
  
  # Fix transitive static library issue
  s.static_framework = true

  s.pod_target_xcconfig = { 
    'DEFINES_MODULE' => 'YES', 
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 arm64'
  }
  s.swift_version = '5.0'

  ## Download NCNN framework
  # s.prepare_command = <<-CMD
  #     rm -rf "ncnn.xcframework"
  #     rm -rf "openmp.xcframework"
  #     curl "https://github.com/KoheiKanagu/ncnn_yolox_flutter/releases/download/0.0.6/ncnn-ios-bitcode_xcframework.zip" -L -o "ncnn-ios-bitcode_xcframework.zip"
  #     unzip "ncnn-ios-bitcode_xcframework.zip"
  #     rm "ncnn-ios-bitcode_xcframework.zip"
  # CMD

  s.preserve_paths = 'ncnn.xcframework', 'openmp.xcframework'
  
  s.xcconfig = { 
    'OTHER_LDFLAGS' => '-framework ncnn -framework openmp',
    'OTHER_CFLAGS' => '-DUSE_NCNN_SIMPLEOCV -DDART_FFI_PLUGIN_IOS',
    'ENABLE_BITCODE' => 'NO' # Disable bitcode to avoid OpenCV compatibility issues
  }
  
  s.ios.vendored_frameworks = 'ncnn.xcframework', 'openmp.xcframework'
  s.library = 'c++'
end
