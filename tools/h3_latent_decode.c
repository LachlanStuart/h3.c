#include "h3_ffmpeg.h"
#include "h3_latent_io.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static void die(const char *message) {
    fprintf(stderr, "h3-latent-decode: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed == 0 || completed % 8 == 0)
        fprintf(stderr, "h3-latent-decode: video VAE %d/%d\n",
                completed, total);
}

int main(int argc, char **argv) {
    if (argc != 5)
        die("MODEL_ROOT SHADER INPUT_LATENT OUTPUT_MP4");
    char weights[4096];
    if (snprintf(weights, sizeof(weights), "%s/Ref2VA/video_vae/source",
                 argv[1]) >= (int)sizeof(weights))
        die("model path is too long");

    char error[1024];
    h3_video_latent latent;
    if (!h3_video_latent_file_read(argv[3], &latent, error, sizeof(error)))
        die(error);
    double started = seconds_now();
    h3_video_frames frames;
    if (!h3_video_vae_decode(weights, argv[2], latent.values,
            latent.time, latent.height, latent.width,
            progress, NULL, &frames, error, sizeof(error)))
        die(error);
    double decoded = seconds_now();

    size_t count = (size_t)frames.frames * (size_t)frames.height *
                   (size_t)frames.width * 3;
    uint8_t *rgb = malloc(count);
    if (!rgb) die("out of memory converting RGB frames");
    size_t finite = 0;
    double mean = 0.0, m2 = 0.0;
    for (size_t index = 0; index < count; index++) {
        float value = frames.rgb[index];
        if (isfinite(value)) finite++;
        double delta = value - mean;
        mean += delta / (double)(index + 1);
        m2 += delta * (value - mean);
        value = fminf(1.0f, fmaxf(0.0f, value));
        rgb[index] = (uint8_t)lrintf(value * 255.0f);
    }
    if (finite != count) die("VideoVAE returned non-finite RGB");

    int encoded = h3_ffmpeg_write_rgb24(
        argv[4], rgb, frames.frames, frames.width, frames.height, 24,
        error, sizeof(error));
    if (!encoded) die(error);
    double finished = seconds_now();
    printf("decode latent_shape=1x24x%dx%dx%d frames=%d pixels=%dx%d "
           "finite=%zu/%zu rgb_mean=%.9g rgb_std=%.9g decode_seconds=%.3f "
           "total_seconds=%.3f gpu_seconds=%.3f peak_metal_gib=%.3f "
           "submissions=%llu output=%s\n",
           latent.time, latent.height, latent.width, frames.frames,
           frames.width, frames.height, finite, count, mean,
           count > 1 ? sqrt(m2 / (double)(count - 1)) : 0.0,
           decoded - started, finished - started,
           frames.gpu_stats.gpu_seconds,
           (double)frames.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (unsigned long long)frames.gpu_stats.submissions, argv[4]);
    free(rgb);
    h3_video_frames_free(&frames);
    h3_video_latent_free(&latent);
    return 0;
}
