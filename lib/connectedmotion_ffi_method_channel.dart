import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'connectedmotion_ffi_platform_interface.dart';

/// An implementation of [ConnectedmotionFfiPlatform] that uses method channels.
class MethodChannelConnectedmotionFfi extends ConnectedmotionFfiPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('connectedmotion_ffi');

  @override
  Future<String?> getPlatformVersion() async {
    final version =
        await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
