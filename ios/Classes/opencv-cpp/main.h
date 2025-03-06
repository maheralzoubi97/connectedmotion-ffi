#ifndef YOUR_HEADER_H
#define YOUR_HEADER_H

#ifdef __cplusplus
extern "C"
{
#endif

    // Declaration for the "Gaussian" function
    __attribute__((visibility("default"))) __attribute__((used)) void Gaussian(char *);

    // Declaration for the "image_ffi" function
    __attribute__((visibility("default"))) __attribute__((used)) void image_ffi(unsigned char *, unsigned int *);

#ifdef __cplusplus
}
#endif

#endif // YOUR_HEADER_H
