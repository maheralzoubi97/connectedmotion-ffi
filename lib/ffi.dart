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

typedef YUV2JPGFunction = Void Function(
    Pointer<Uint8>,
    Pointer<Uint8>,
    Pointer<Uint8>,
    Int32,
    Int32,
    Int32,
    Int32,
    Pointer<Pointer<Uint8>>,
    Pointer<Int32>,
    Pointer<Pointer<Uint8>>,
    Pointer<Int32>,
    Int32,
    Int32);

typedef YUV2JPG = void Function(
    Pointer<Uint8>,
    Pointer<Uint8>,
    Pointer<Uint8>,
    int,
    int,
    int,
    int,
    Pointer<Pointer<Uint8>>,
    Pointer<Int32>,
    Pointer<Pointer<Uint8>>,
    Pointer<Int32>,
    int,
    int);

Map<String, Uint8List>? _processYuv420Image(
    CameraImage cameraImage, int newWidth, int newHeight) {
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
  final resizedJpegBuf = malloc.allocate<Pointer<Uint8>>(1);
  final resizedJpegSize = malloc.allocate<Int32>(1);

  final yuv2jpgFunc = dylib.lookupFunction<YUV2JPGFunction, YUV2JPG>('YUV2JPG');

  try {
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
      resizedJpegBuf,
      resizedJpegSize,
      newWidth,
      newHeight,
    );

    final originalImageBytes = Uint8List.fromList(
        originalJpegBuf.value.asTypedList(originalJpegSize.value));
    final resizedImageBytes = Uint8List.fromList(
        resizedJpegBuf.value.asTypedList(resizedJpegSize.value));

    return {
      "originalImage": originalImageBytes,
      "resizedImage": resizedImageBytes,
    };
  } finally {
    malloc.free(yData);
    malloc.free(uData);
    malloc.free(vData);
    if (originalJpegBuf.value != nullptr) malloc.free(originalJpegBuf.value);
    malloc.free(originalJpegBuf);
    malloc.free(originalJpegSize);
    if (resizedJpegBuf.value != nullptr) malloc.free(resizedJpegBuf.value);
    malloc.free(resizedJpegBuf);
    malloc.free(resizedJpegSize);
  }
}

void processImageInIsolate(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  // Send the port to receive messages on
  mainSendPort.send(receivePort.sendPort);

  receivePort.listen((message) {
    if (message is Map<String, dynamic>) {
      final responsePort = message['responsePort'] as SendPort;
      final cameraImage = message['cameraImage'] as CameraImage;
      final newWidth = message['newWidth'] as int;
      final newHeight = message['newHeight'] as int;
      final isAndroid = message['isAndroid'] as bool;

      final stopwatch = Stopwatch()..start();
      try {
        Map<String, Uint8List>? result;

        if (isAndroid) {
          // Process Android YUV image
          result = _processYuv420Image(cameraImage, newWidth, newHeight);
        } else {
          // Process iOS BGRA image
          result = _processBgra8888Image(cameraImage, newWidth, newHeight);
        }

        if (result != null) {
          responsePort.send({
            'success': true,
            'originalImage': result['originalImage'],
            'resizedImage': result['resizedImage'],
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
