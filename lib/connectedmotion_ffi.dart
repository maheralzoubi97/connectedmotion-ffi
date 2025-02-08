import 'connectedmotion_ffi_platform_interface.dart';

class ConnectedmotionFfi {
  Future<String?> getPlatformVersion() {
    return ConnectedmotionFfiPlatform.instance.getPlatformVersion();
  }
}
