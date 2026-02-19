import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:connectedmotion_ffi/functions/functions.dart';
import 'package:ffi/ffi.dart';

Map<String, Uint8List>? _processBgra8888Image(
    CameraImage cameraImage, int newWidth, int newHeight) {
  final s = cameraImage.planes[0].bytes.length;
  final p = malloc.allocate<Uint8>(4 * cameraImage.height * cameraImage.width);
  p.asTypedList(s).setRange(0, s, cameraImage.planes[0].bytes);

  final segBoundary =
      malloc.allocate<Int32>(cameraImage.height * cameraImage.width);
  final segBoundarySize = malloc.allocate<Int32>(1);
  final jpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final jpegSize = malloc.allocate<Int32>(1);
  final resizedJpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final resizedJpegSize = malloc.allocate<Int32>(1);

  // Lookup the FFI function for BGRA8888 image processing with resizing
  final imageFfi = dylib.lookupFunction<
      Void Function(
        Pointer<Uint8>,
        Int32,
        Int32,
        Int32,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        Int32,
        Int32,
      ),
      void Function(
        Pointer<Uint8>,
        int,
        int,
        int,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        int,
        int,
      )>('bgra88882jpg');

  try {
    imageFfi(
      p,
      s,
      cameraImage.width,
      cameraImage.height,
      jpegBuf,
      jpegSize,
      resizedJpegBuf,
      resizedJpegSize,
      newWidth,
      newHeight,
    );

    final originalImageBytes = jpegBuf.value.asTypedList(jpegSize.value);
    final resizedImageBytes =
        resizedJpegBuf.value.asTypedList(resizedJpegSize.value);
    return {
      "originalImage": originalImageBytes,
      "resizedImage": resizedImageBytes,
    };
  } finally {
    malloc.free(p);
    malloc.free(segBoundary);
    malloc.free(segBoundarySize);
    if (jpegBuf.value != nullptr) malloc.free(jpegBuf.value);
    malloc.free(jpegBuf);
    malloc.free(jpegSize);
    if (resizedJpegBuf.value != nullptr) malloc.free(resizedJpegBuf.value);
    malloc.free(resizedJpegBuf);
    malloc.free(resizedJpegSize);
  }
}

// Native C function signature typedef (matches native exactly)
typedef YUV2JPGFunction = Void Function(
  Pointer<Uint8> yData,
  Pointer<Uint8> uData,
  Pointer<Uint8> vData,
  Int32 width,
  Int32 height,
  Int32 uvRowStride,
  Int32 uvPixelStride,
  Pointer<Pointer<Uint8>> originalJpegBuf,
  Pointer<Int32> originalJpegSize,
  Pointer<Pointer<Uint8>> mediumJpegBuf,
  Pointer<Int32> mediumJpegSize,
  Pointer<Pointer<Uint8>> lowJpegBuf,
  Pointer<Int32> lowJpegSize,
  Int32 newWidthMedium,
  Int32 newHeightMedium,
  Int32 newWidthLow,
  Int32 newHeightLow,
  Int32 isPortrait, // 👈 added
);

// Dart callable function typedef (must match param count & order)
typedef YUV2JPG = void Function(
  Pointer<Uint8> yData,
  Pointer<Uint8> uData,
  Pointer<Uint8> vData,
  int width,
  int height,
  int uvRowStride,
  int uvPixelStride,
  Pointer<Pointer<Uint8>> originalJpegBuf,
  Pointer<Int32> originalJpegSize,
  Pointer<Pointer<Uint8>> mediumJpegBuf,
  Pointer<Int32> mediumJpegSize,
  Pointer<Pointer<Uint8>> lowJpegBuf,
  Pointer<Int32> lowJpegSize,
  int newWidthMedium,
  int newHeightMedium,
  int newWidthLow,
  int newHeightLow,
  int isPortrait,
);

Map<String, Uint8List>? _processYuv420Image(
  CameraImage cameraImage,
  int newWidthMedium,
  int newHeightMedium,
  int newWidthLow,
  int newHeightLow,
  bool isPortrait,
) {
  final yPlane = cameraImage.planes[0];
  final uPlane = cameraImage.planes[1];
  final vPlane = cameraImage.planes[2];

  final yData = malloc.allocate<Uint8>(yPlane.bytes.length);
  final uData = malloc.allocate<Uint8>(uPlane.bytes.length);
  final vData = malloc.allocate<Uint8>(vPlane.bytes.length);

  yData.asTypedList(yPlane.bytes.length).setAll(0, yPlane.bytes);
  uData.asTypedList(uPlane.bytes.length).setAll(0, uPlane.bytes);
  vData.asTypedList(vPlane.bytes.length).setAll(0, vPlane.bytes);

  final originalJpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final originalJpegSize = malloc.allocate<Int32>(1);
  final mediumJpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final mediumJpegSize = malloc.allocate<Int32>(1);
  final lowJpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final lowJpegSize = malloc.allocate<Int32>(1);

  final yuv2jpgFunc = dylib.lookupFunction<YUV2JPGFunction, YUV2JPG>('YUV2JPG');

  try {
    final isPortraitInt = isPortrait ? 1 : 0;

    yuv2jpgFunc(
      yData,
      uData,
      vData,
      cameraImage.width,
      cameraImage.height,
      uPlane.bytesPerRow,
      uPlane.bytesPerPixel ?? 1,
      originalJpegBuf,
      originalJpegSize,
      mediumJpegBuf,
      mediumJpegSize,
      lowJpegBuf,
      lowJpegSize,
      newWidthMedium,
      newHeightMedium,
      newWidthLow,
      newHeightLow,
      isPortraitInt, // 👈 pass it here
    );

    final originalImageBytes = Uint8List.fromList(
      originalJpegBuf.value.asTypedList(originalJpegSize.value),
    );
    final mediumImageBytes = Uint8List.fromList(
      mediumJpegBuf.value.asTypedList(mediumJpegSize.value),
    );
    final lowImageBytes = Uint8List.fromList(
      lowJpegBuf.value.asTypedList(lowJpegSize.value),
    );

    return {
      "originalImage": originalImageBytes,
      "mediumImage": mediumImageBytes,
      "lowImage": lowImageBytes,
    };
  } finally {
    malloc.free(yData);
    malloc.free(uData);
    malloc.free(vData);
    if (originalJpegBuf.value != nullptr) malloc.free(originalJpegBuf.value);
    malloc.free(originalJpegBuf);
    malloc.free(originalJpegSize);
    if (mediumJpegBuf.value != nullptr) malloc.free(mediumJpegBuf.value);
    malloc.free(mediumJpegBuf);
    malloc.free(mediumJpegSize);
    if (lowJpegBuf.value != nullptr) malloc.free(lowJpegBuf.value);
    malloc.free(lowJpegBuf);
    malloc.free(lowJpegSize);
  }
}

void processImageInIsolate(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  receivePort.listen((message) {
    if (message is Map<String, dynamic>) {
      final responsePort = message['responsePort'] as SendPort;
      final cameraImage = message['cameraImage'] as CameraImage;
      final newWidthMedium = message['newWidthMedium'] as int;
      final newHeightMedium = message['newHeightMedium'] as int;
      final newWidthLow = message['newWidthLow'] as int;
      final newHeightLow = message['newHeightLow'] as int;
      final isAndroid = message['isAndroid'] as bool;
      final isPortrait = message['isPortrait'] as bool;

      final stopwatch = Stopwatch()..start();
      try {
        Map<String, Uint8List>? result;

        if (isAndroid) {
          result = _processYuv420Image(
            cameraImage,
            newWidthMedium,
            newHeightMedium,
            newWidthLow,
            newHeightLow,
            isPortrait,
          );
        } else {
          // Process iOS BGRA image
          result = _processBgra8888Image(
            cameraImage,
            newWidthLow,
            newHeightLow,
          );
        }

        if (result != null) {
          responsePort.send({
            'success': true,
            'originalImage': result['originalImage'],
            'mediumImage': result['mediumImage'],
            'lowImage': result['lowImage'],
            'processingTimeMs': stopwatch.elapsedMilliseconds,
          });
        } else {
          responsePort.send({
            'success': false,
            'error': 'Image processing failed',
            'processingTimeMs': stopwatch.elapsedMilliseconds,
          });
        }
      } catch (e) {
        responsePort.send({
          'success': false,
          'error': e.toString(),
          'processingTimeMs': stopwatch.elapsedMilliseconds,
        });
      } finally {
        stopwatch.stop();
      }
    }
  });
}

void convertBgra88882Jpg(Map<String, dynamic> data) {
  final port = data['send_port'] as SendPort;
  final cameraImage = data['cameraImage'] as CameraImage;
  final index = data['index'];
  final isolateTimeStamp = data['isolateTimeStamp'];
  final newWidth = data['newWidth'] as int;
  final newHeight = data['newHeight'] as int;

  final s = cameraImage.planes[0].bytes.length;
  final p = malloc.allocate<Uint8>(4 * cameraImage.height * cameraImage.width);
  p.asTypedList(s).setRange(0, s, cameraImage.planes[0].bytes);
  final segBoundary =
      malloc.allocate<Int32>(cameraImage.height * cameraImage.width);
  final segBoundarySize = malloc.allocate<Int32>(1);
  final jpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final jpegSize = malloc.allocate<Int32>(1);
  final resizedJpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final resizedJpegSize = malloc.allocate<Int32>(1);

  // Lookup the FFI function for BGRA8888 image processing with resizing
  final imageFfi = dylib.lookupFunction<
      Void Function(
        Pointer<Uint8>,
        Int32,
        Int32,
        Int32,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        Int32,
        Int32,
      ),
      void Function(
        Pointer<Uint8>,
        int,
        int,
        int,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        Pointer<Pointer<Uint8>>,
        Pointer<Int32>,
        int,
        int,
      )>('bgra88882jpg');

  try {
    imageFfi(
      p,
      s,
      cameraImage.width,
      cameraImage.height,
      jpegBuf,
      jpegSize,
      resizedJpegBuf,
      resizedJpegSize,
      newWidth,
      newHeight,
    );

    final originalImageBytes = jpegBuf.value.asTypedList(jpegSize.value);
    final resizedImageBytes =
        resizedJpegBuf.value.asTypedList(resizedJpegSize.value);

    port.send({
      "index": index,
      "isolateTimeStamp": isolateTimeStamp,
      "originalImage": originalImageBytes,
      "resizedImage": resizedImageBytes,
    });
  } finally {
    malloc.free(p);
    malloc.free(segBoundary);
    malloc.free(segBoundarySize);
    malloc.free(jpegBuf.value);
    malloc.free(jpegBuf);
    malloc.free(jpegSize);
    malloc.free(resizedJpegBuf.value);
    malloc.free(resizedJpegBuf);
    malloc.free(resizedJpegSize);
  }
}

typedef DetectImageOrientationC = Int32 Function(Int32 width, Int32 height);
typedef DetectImageOrientationDart = int Function(int width, int height);

// Fast dimension extraction function
typedef GetImageDimensionsC = Void Function(
  Pointer<Uint8> imageBytes,
  Int32 imageSize,
  Pointer<Int32> width,
  Pointer<Int32> height,
);
typedef GetImageDimensionsDart = void Function(
  Pointer<Uint8> imageBytes,
  int imageSize,
  Pointer<Int32> width,
  Pointer<Int32> height,
);

// --- resizeImageTo1920x1080WithAspectRatio ---
typedef ResizeImage1920x1080C = Void Function(
  Pointer<Uint8> imageBytes,
  Int32 imageSize,
  Int32 orientation,
  Pointer<Pointer<Uint8>> outputBuffer,
  Pointer<Int32> outputSize,
);
typedef ResizeImage1920x1080Dart = void Function(
  Pointer<Uint8> imageBytes,
  int imageSize,
  int orientation,
  Pointer<Pointer<Uint8>> outputBuffer,
  Pointer<Int32> outputSize,
);

// Lookups
final detectImageOrientation =
    dylib.lookupFunction<DetectImageOrientationC, DetectImageOrientationDart>(
        'detectImageOrientation');

final getImageDimensions =
    dylib.lookupFunction<GetImageDimensionsC, GetImageDimensionsDart>(
        'getImageDimensions');

final resizeImageTo1920x1080 =
    dylib.lookupFunction<ResizeImage1920x1080C, ResizeImage1920x1080Dart>(
        'resizeImageTo1920x1080WithAspectRatio');

// Fast version - no Dart image decoding
Uint8List resizeJpegTo1920x1080(Uint8List jpegBytes) {
  final inputPointer = malloc.allocate<Uint8>(jpegBytes.length);
  inputPointer.asTypedList(jpegBytes.length).setAll(0, jpegBytes);

  final outputBufferPtr = malloc.allocate<Pointer<Uint8>>(1);
  final outputSizePtr = malloc.allocate<Int32>(1);
  final widthPtr = malloc.allocate<Int32>(1);
  final heightPtr = malloc.allocate<Int32>(1);

  try {
    // Fast dimension extraction in C++ (no full decode)
    getImageDimensions(inputPointer, jpegBytes.length, widthPtr, heightPtr);

    // Get orientation from dimensions
    final orientation = detectImageOrientation(widthPtr.value, heightPtr.value);

    // Resize image
    resizeImageTo1920x1080(
      inputPointer,
      jpegBytes.length,
      orientation,
      outputBufferPtr,
      outputSizePtr,
    );

    final resized = outputBufferPtr.value.asTypedList(outputSizePtr.value);
    return Uint8List.fromList(resized);
  } finally {
    malloc.free(inputPointer);
    malloc.free(widthPtr);
    malloc.free(heightPtr);
    if (outputBufferPtr.value != nullptr) malloc.free(outputBufferPtr.value);
    malloc.free(outputBufferPtr);
    malloc.free(outputSizePtr);
  }
}
