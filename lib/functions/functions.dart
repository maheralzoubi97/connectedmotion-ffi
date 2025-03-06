import 'dart:ffi';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:logger/logger.dart';
import 'package:path_provider/path_provider.dart';

Logger logger = Logger();

const String _libName = 'libOpenCV_ffi';
final DynamicLibrary dylib = () {
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.process();
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('$_libName.so');
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('$_libName.dll');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

Future<String> copy(String assetsPath) async {
  final documentDir = await getApplicationDocumentsDirectory();

  final data = await rootBundle.load(assetsPath);

  final file = File('${documentDir.path}/$assetsPath')
    ..createSync(recursive: true)
    ..writeAsBytesSync(
      data.buffer.asUint8List(),
      flush: true,
    );
  return file.path;
}
