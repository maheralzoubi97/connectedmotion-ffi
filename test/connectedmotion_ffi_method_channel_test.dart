import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:connectedmotion_ffi/connectedmotion_ffi_method_channel.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  MethodChannelConnectedmotionFfi platform = MethodChannelConnectedmotionFfi();
  const MethodChannel channel = MethodChannel('connectedmotion_ffi');

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(
      channel,
      (MethodCall methodCall) async {
        return '42';
      },
    );
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(channel, null);
  });

  test('getPlatformVersion', () async {
    expect(await platform.getPlatformVersion(), '42');
  });
}
