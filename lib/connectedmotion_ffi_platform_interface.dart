import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'connectedmotion_ffi_method_channel.dart';

abstract class ConnectedmotionFfiPlatform extends PlatformInterface {
  /// Constructs a ConnectedmotionFfiPlatform.
  ConnectedmotionFfiPlatform() : super(token: _token);

  static final Object _token = Object();

  static ConnectedmotionFfiPlatform _instance =
      MethodChannelConnectedmotionFfi();

  /// The default instance of [ConnectedmotionFfiPlatform] to use.
  ///
  /// Defaults to [MethodChannelConnectedmotionFfi].
  static ConnectedmotionFfiPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [ConnectedmotionFfiPlatform] when
  /// they register themselves.
  static set instance(ConnectedmotionFfiPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
