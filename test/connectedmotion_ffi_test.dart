import 'package:flutter_test/flutter_test.dart';
import 'package:connectedmotion_ffi/connectedmotion_ffi.dart';
import 'package:connectedmotion_ffi/connectedmotion_ffi_platform_interface.dart';
import 'package:connectedmotion_ffi/connectedmotion_ffi_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockConnectedmotionFfiPlatform
    with MockPlatformInterfaceMixin
    implements ConnectedmotionFfiPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final ConnectedmotionFfiPlatform initialPlatform = ConnectedmotionFfiPlatform.instance;

  test('$MethodChannelConnectedmotionFfi is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelConnectedmotionFfi>());
  });

  test('getPlatformVersion', () async {
    ConnectedmotionFfi connectedmotionFfiPlugin = ConnectedmotionFfi();
    MockConnectedmotionFfiPlatform fakePlatform = MockConnectedmotionFfiPlatform();
    ConnectedmotionFfiPlatform.instance = fakePlatform;

    expect(await connectedmotionFfiPlugin.getPlatformVersion(), '42');
  });
}
