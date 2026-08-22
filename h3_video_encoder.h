#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef void (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

typedef struct {
    int chunks;
    int padded_frames;
    int encoded_tokens;
    int latent_tokens;
} h3_ref2va_video_encode_plan;

/* Released Ref2VA encode_temporal contract: right-pad each input timeline to
 * 17-frame clips, encode five tokens per clip, then drop three tail tokens. */
int h3_ref2va_video_encode_plan_build(int frames,
                                      h3_ref2va_video_encode_plan *plan);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. */
int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
/* Ref2VA's temporal wrapper around the causal encoder. Its temporal chunk
 * stitching and final token drop stay in Metal; only source RGB upload and
 * the final channel-major latent read cross the host boundary. */
int h3_video_vae_encode_ref2va_temporal(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
void h3_video_latent_free(h3_video_latent *latent);

#endif
