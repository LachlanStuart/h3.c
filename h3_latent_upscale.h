#ifndef H3_LATENT_UPSCALE_H
#define H3_LATENT_UPSCALE_H

#include <stddef.h>

/* Stream a normalized [24,T,H,W] F32 H3 video latent through the released
 * MPS upscaler.  No latent is materialized on disk.  When python is NULL the
 * executable script is launched directly, preserving its uv shebang. */
int h3_latent_upscale_pipe(const char *python, const char *script,
                           const char *source, const char *checkpoint,
                           const float *input, int input_time,
                           int input_height, int input_width,
                           float **output, int *output_time,
                           int *output_height, int *output_width,
                           char *error, size_t error_size);

#endif
