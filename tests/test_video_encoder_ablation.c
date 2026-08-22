#include "h3_video_encoder.h"
#include "h3_video_vae.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_video_encoder_ablation.c: %s\n", message);
    exit(1);
}

static double now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) die("clock_gettime failed");
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int integer(const char *value, const char *label) {
    errno = 0;
    char *end = NULL;
    long result = strtol(value, &end, 10);
    if (errno || end == value || *end || result < 1 || result > 1000000) {
        fprintf(stderr, "invalid %s: %s\n", label, value);
        exit(2);
    }
    return (int)result;
}

static void write_f32(const char *path, const float *values, size_t count) {
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(values, sizeof(*values), count, file) != count ||
        fclose(file)) die("cannot write oracle");
}

static float *read_f32(const char *path, size_t count) {
    float *values = malloc(count * sizeof(*values));
    FILE *file = fopen(path, "rb");
    if (!values || !file || fread(values, sizeof(*values), count, file) != count ||
        fgetc(file) != EOF || fclose(file)) die("cannot read oracle");
    return values;
}

static void compare(const char *label, const float *got, const float *want,
                    size_t count, double max_limit, double l2_limit) {
    double maximum = 0.0, error = 0.0, value = 0.0;
    uint64_t different = 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(got[index])) die("encoder produced a non-finite value");
        double delta = (double)got[index] - (double)want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        error += delta * delta;
        value += (double)want[index] * (double)want[index];
        if (memcmp(got + index, want + index, sizeof(*got))) different++;
    }
    double relative_l2 = sqrt(error / (value > 1e-24 ? value : 1e-24));
    printf("%s: max-abs=%.9g rel-L2=%.9g differing=%llu/%zu\n", label,
           maximum, relative_l2, (unsigned long long)different, count);
    if (maximum > max_limit || relative_l2 > l2_limit)
        die("oracle comparison exceeded its numerical tolerance");
}

int main(int argc, char **argv) {
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: %s WEIGHTS (--write|--compare) ORACLE "
                "FRAMES HEIGHT WIDTH EXPECTED_SUBMISSIONS [DECODED_ORACLE]\n",
                argv[0]);
        return 2;
    }
    const char *weights = argv[1];
    int writing = !strcmp(argv[2], "--write");
    if (!writing && strcmp(argv[2], "--compare")) die("invalid oracle mode");
    int frames = integer(argv[4], "frame count");
    int height = integer(argv[5], "height");
    int width = integer(argv[6], "width");
    int expected_submissions = integer(argv[7], "submission count");
    if (height % 16 || width % 16) die("geometry must be divisible by 16");
    size_t pixel_count = (size_t)3 * (size_t)frames * (size_t)height *
                         (size_t)width;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    if (!pixels) die("cannot allocate synthetic pixels");
    for (size_t index = 0; index < pixel_count; index++) {
        uint32_t bits = (uint32_t)index * UINT32_C(1664525) +
                        UINT32_C(1013904223);
        pixels[index] = (float)(bits >> 8) * (1.0f / 16777215.0f);
    }
    char error[512];
    h3_video_latent latent;
    double started = now();
    if (!h3_video_vae_encode_ref2va_temporal(
            weights, "h3_shaders.metal", pixels, frames, height, width,
            NULL, NULL, &latent, error, sizeof(error))) die(error);
    double elapsed = now() - started;
    free(pixels);
    h3_ref2va_video_encode_plan plan;
    if (!h3_ref2va_video_encode_plan_build(frames, &plan) ||
        latent.time != plan.latent_tokens || latent.height != height / 16 ||
        latent.width != width / 16) die("Ref2VA temporal geometry changed");
    if (latent.gpu_stats.submissions != (uint64_t)expected_submissions)
        die("unexpected command-buffer submission count");
    if (latent.gpu_stats.host_tensor_reads != 1 ||
        latent.gpu_stats.host_tensor_writes != 0)
        die("unexpected host tensor transfer count");
    printf("encoder: wall=%.3fs gpu=%.3fs wait=%.3fs submissions=%llu "
           "host-rw=%llu/%llu bytes=%llu/%llu peak=%.3fGiB alloc=%.3fGiB "
           "conv=%llu direct=%llu shape=24x%dx%dx%d chunks=%d pad=%d\n",
           elapsed, latent.gpu_stats.gpu_seconds,
           latent.gpu_stats.command_wait_seconds,
           (unsigned long long)latent.gpu_stats.submissions,
           (unsigned long long)latent.gpu_stats.host_tensor_reads,
           (unsigned long long)latent.gpu_stats.host_tensor_writes,
           (unsigned long long)latent.gpu_stats.host_tensor_read_bytes,
           (unsigned long long)latent.gpu_stats.host_tensor_write_bytes,
           (double)latent.gpu_stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)latent.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           (unsigned long long)latent.gpu_stats.mps_conv_dispatches,
           (unsigned long long)latent.gpu_stats.direct_dispatches,
           latent.time, latent.height, latent.width, plan.chunks,
           plan.padded_frames);
    size_t latent_count = (size_t)24 * (size_t)latent.time *
                          (size_t)latent.height * (size_t)latent.width;
    double max_limit = 1e-6, l2_limit = 1e-7;
    const char *limit = getenv("H3_TEST_MAX_ABS");
    if (limit) max_limit = strtod(limit, NULL);
    limit = getenv("H3_TEST_REL_L2");
    if (limit) l2_limit = strtod(limit, NULL);
    if (writing) {
        write_f32(argv[3], latent.values, latent_count);
    } else {
        float *want = read_f32(argv[3], latent_count);
        compare("latent oracle", latent.values, want, latent_count,
                max_limit, l2_limit);
        free(want);
    }
    if (argc == 9) {
        h3_video_frames decoded;
        started = now();
        if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent.values,
                latent.time, latent.height, latent.width, NULL, NULL,
                &decoded, error, sizeof(error))) die(error);
        elapsed = now() - started;
        size_t decoded_count = (size_t)decoded.frames * (size_t)decoded.height *
                               (size_t)decoded.width * 3;
        printf("decoder: wall=%.3fs gpu=%.3fs submissions=%llu peak=%.3fGiB "
               "shape=%dx%dx%dx3\n", elapsed, decoded.gpu_stats.gpu_seconds,
               (unsigned long long)decoded.gpu_stats.submissions,
               (double)decoded.gpu_stats.peak_live_bytes /
                   (1024.0 * 1024.0 * 1024.0), decoded.frames,
               decoded.height, decoded.width);
        if (writing) {
            write_f32(argv[8], decoded.rgb, decoded_count);
        } else {
            float *want = read_f32(argv[8], decoded_count);
            compare("decoded oracle", decoded.rgb, want, decoded_count,
                    max_limit, l2_limit);
            free(want);
        }
        h3_video_frames_free(&decoded);
    }
    h3_video_latent_free(&latent);
    puts("ok: video encoder ablation matches its contract");
    return 0;
}
