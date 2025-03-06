import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:connectedmotion_ffi/functions/functions.dart';
import 'package:ffi/ffi.dart';

void bgra88882Jpg(Map<String, dynamic> data) {
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

void yuv2jpg(Map<String, dynamic> data) {
  var port = data['send_port'] as SendPort;
  var cameraImage = data['cameraImage'] as CameraImage;
  var index = data['index'];
  var isolateTimeStamp = data['isolateTimeStamp'];
  var newWidth = data['newWidth'] as int;
  var newHeight = data['newHeight'] as int;

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

  final yuv2jpg = dylib.lookupFunction<YUV2JPGFunction, YUV2JPG>('YUV2JPG');

  try {
    yuv2jpg(
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

    final originalImageBytes =
        originalJpegBuf.value.asTypedList(originalJpegSize.value);
    final resizedImageBytes =
        resizedJpegBuf.value.asTypedList(resizedJpegSize.value);

    port.send({
      "index": index,
      "isolateTimeStamp": isolateTimeStamp,
      "originalImage": originalImageBytes,
      "resizedImage": resizedImageBytes,
    });
  } finally {
    malloc.free(yData);
    malloc.free(uData);
    malloc.free(vData);
    malloc.free(originalJpegBuf.value);
    malloc.free(originalJpegBuf);
    malloc.free(originalJpegSize);
    malloc.free(resizedJpegBuf.value);
    malloc.free(resizedJpegBuf);
    malloc.free(resizedJpegSize);
  }
}
