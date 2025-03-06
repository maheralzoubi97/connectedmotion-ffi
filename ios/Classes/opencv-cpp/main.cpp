
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void image_ffi_path(char *path, int *objectCnt);
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void image_ffi(unsigned char *buf, int size, int *segBoundary, int *segBoundarySize);
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void initYolo8(char *modelPath, char *paramPath);
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void disposeYolo8();
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void image_ffi_yuv24(unsigned char *yData, unsigned char *uData, unsigned char *vData, int width, int height, int uvRowStride, int uvPixelStride, int *segBoundary, int *segBoundarySize, unsigned char **jpegBuf, int *jpegSize);
extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char *
classifyPathImage(unsigned char *buf, int size);
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void initClassifier(char *modelPath, char *paramPath);
extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char *
classifyImageYUV240(unsigned char *yData, unsigned char *uData, unsigned char *vData, int width, int height, int uvRowStride, int uvPixelStride);
extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char *
blurImage(unsigned char *buf, int size, double threshold, unsigned char *yData, unsigned char *uData, unsigned char *vData, int width, int height, int uvRowStride, int uvPixelStride, unsigned char **outputImageBuffer, int *outputImageSize);
extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char *
blurPathImage(unsigned char *buf, int size, double threshold);
extern "C" __attribute__((visibility("default"))) __attribute__((used))
const char *
blurImageYUV240(unsigned char *buf, int size, double threshold, unsigned char *yData, unsigned char *uData, unsigned char *vData, int width, int height, int uvRowStride, int uvPixelStride);
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void resizeImageYUV240(unsigned char *yData, unsigned char *uData, unsigned char *vData, int width, int height, int uvRowStride, int uvPixelStride, int newWidth, int newHeight, unsigned char **resizedRGBImageData, int *resizedRGBImageSize);
extern "C" __attribute__((visibility("default"))) __attribute__((used)) void YUV2JPG(unsigned char *yData, unsigned char *uData, unsigned char *vData, int width, int height, int uvRowStride, int uvPixelStride, unsigned char **originalJpegBuf, int *originalJpegSize, unsigned char **resizedJpegBuf, int *resizedJpegSize, int newWidth, int newHeight);

#include "main-seg.cpp"