#ifndef H3_LATENT_IO_H
#define H3_LATENT_IO_H

#include "h3_video_encoder.h"

#include <stddef.h>

/* Stable little-endian interchange for normalized H3 video latents:
 *   magic "H3LATF32", u32 time, u32 height, u32 width,
 *   then channel-major F32 [24,time,height,width]. */
int h3_video_latent_file_write(const char *path,
                               const float *values,
                               int time, int height, int width,
                               char *error, size_t error_size);

int h3_video_latent_file_read(const char *path,
                              h3_video_latent *latent,
                              char *error, size_t error_size);

#endif
