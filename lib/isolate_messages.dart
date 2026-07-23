import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:camera_platform_interface/camera_platform_interface.dart';

/// A single camera image plane packaged for zero-copy transfer across an
/// isolate boundary. Sending a [Uint8List] directly through a [SendPort]
/// deep-copies its bytes; wrapping it in [TransferableTypedData] instead
/// transfers ownership without copying, which matters here since a single
/// frame's planes can be several MB.
class TransferablePlane {
  TransferablePlane({
    required Uint8List bytes,
    required this.bytesPerRow,
    this.bytesPerPixel,
    this.height,
    this.width,
  }) : transferableBytes = TransferableTypedData.fromList([bytes]);

  final TransferableTypedData transferableBytes;
  final int bytesPerRow;
  final int? bytesPerPixel;
  final int? height;
  final int? width;

  /// Materializes the transferred bytes back into a [CameraImagePlane].
  /// Can only be called once — [TransferableTypedData.materialize] consumes
  /// the underlying buffer.
  CameraImagePlane materialize() {
    return CameraImagePlane(
      bytes: transferableBytes.materialize().asUint8List(),
      bytesPerRow: bytesPerRow,
      bytesPerPixel: bytesPerPixel,
      height: height,
      width: width,
    );
  }
}

/// A request sent to a worker isolate to process one camera frame.
///
/// Carries the frame as [TransferablePlane]s rather than a [CameraImage]
/// directly, so the planes' bytes are moved into the worker isolate instead
/// of being deep-copied by the default isolate message serialization.
class CameraFrameRequest {
  CameraFrameRequest({
    required this.responsePort,
    required CameraImage cameraImage,
    required this.newWidthMedium,
    required this.newHeightMedium,
    required this.newWidthLow,
    required this.newHeightLow,
    required this.isAndroid,
    required this.isPortrait,
  })  : formatGroupIndex = cameraImage.format.group.index,
        formatRaw = cameraImage.format.raw,
        imageWidth = cameraImage.width,
        imageHeight = cameraImage.height,
        planes = cameraImage.planes
            .map((plane) => TransferablePlane(
                  bytes: plane.bytes,
                  bytesPerRow: plane.bytesPerRow,
                  bytesPerPixel: plane.bytesPerPixel,
                  height: plane.height,
                  width: plane.width,
                ))
            .toList(growable: false);

  final SendPort responsePort;
  final int formatGroupIndex;
  final dynamic formatRaw;
  final int imageWidth;
  final int imageHeight;
  final List<TransferablePlane> planes;
  final int newWidthMedium;
  final int newHeightMedium;
  final int newWidthLow;
  final int newHeightLow;
  final bool isAndroid;
  final bool isPortrait;

  /// Reconstructs a real [CameraImage] from the transferred plane data.
  /// Consumes the planes' transferred buffers — only call once per request.
  CameraImage materializeCameraImage() {
    final cameraImageData = CameraImageData(
      format: CameraImageFormat(
        ImageFormatGroup.values[formatGroupIndex],
        raw: formatRaw,
      ),
      planes: planes.map((p) => p.materialize()).toList(growable: false),
      height: imageHeight,
      width: imageWidth,
    );
    return CameraImage.fromPlatformInterface(cameraImageData);
  }
}

/// The result of processing one [CameraFrameRequest].
class CameraFrameResponse {
  CameraFrameResponse.success({
    required this.originalImage,
    required this.mediumImage,
    required this.lowImage,
    required this.processingTimeMs,
  })  : success = true,
        error = null;

  CameraFrameResponse.failure({
    required this.error,
    required this.processingTimeMs,
  })  : success = false,
        originalImage = null,
        mediumImage = null,
        lowImage = null;

  final bool success;
  final Uint8List? originalImage;
  final Uint8List? mediumImage;
  final Uint8List? lowImage;
  final int processingTimeMs;
  final String? error;
}
