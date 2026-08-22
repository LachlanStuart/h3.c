#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_video_encoder_fp16.c: %s\n", message);
    exit(1);
}

static void gpu_check(h3_gpu *gpu, int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_video_encoder_fp16.c: %s: %s\n",
                operation, h3_gpu_error(gpu));
        exit(1);
    }
}

static float sample(uint32_t index) {
    uint32_t bits = index * UINT32_C(1664525) + UINT32_C(1013904223);
    return ((float)(bits >> 8) * (2.0f / 16777215.0f) - 1.0f) * 0.75f;
}

static void compare(const char *label, const float *fp16, const float *fp32,
                    size_t count, double max_limit, double l2_limit) {
    double maximum = 0.0, error = 0.0, value = 0.0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(fp16[index]) || !isfinite(fp32[index]))
            fail("non-finite oracle result");
        double delta = (double)fp16[index] - fp32[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        error += delta * delta;
        value += (double)fp32[index] * fp32[index];
    }
    double relative_l2 = sqrt(error / (value > 1e-24 ? value : 1e-24));
    printf("%s: max-abs=%.9g rel-L2=%.9g count=%zu\n",
           label, maximum, relative_l2, count);
    if (maximum > max_limit || relative_l2 > l2_limit)
        fail("FP16 oracle tolerance exceeded");
}

static h3_gpu_tensor *new_f32(h3_gpu *gpu, float *values, size_t count,
                              uint32_t seed) {
    for (size_t index = 0; index < count; index++)
        values[index] = sample((uint32_t)index + seed);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, count);
    if (!tensor) fail("cannot allocate F32 input");
    return tensor;
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);

    enum {
        NORM_D = 2, NORM_H = 4, NORM_W = 5, NORM_C = 32,
        NORM_COUNT = NORM_D * NORM_H * NORM_W * NORM_C,
        PIXEL_D = 2, PIXEL_H = 4, PIXEL_W = 5,
        PIXEL_COUNT = 3 * PIXEL_D * PIXEL_H * PIXEL_W,
        PAD_D = 2, PAD_H = 4, PAD_W = 5, PAD_C = 7,
        PAD_OUT_COUNT = (PAD_D + 2) * (PAD_H + 2) * (PAD_W + 2) * PAD_C,
        CONV_D = 4, CONV_H = 5, CONV_W = 6, CONV_I = 4, CONV_O = 5,
        CONV_INPUT_COUNT = CONV_D * CONV_H * CONV_W * CONV_I,
        CONV_WEIGHT_COUNT = CONV_O * CONV_I * 3 * 3 * 3,
        CONV_OUTPUT_COUNT = (CONV_D - 2) * (CONV_H - 2) *
                            (CONV_W - 2) * CONV_O
    };
    float source[NORM_COUNT], norm_weight[NORM_C], norm_bias[NORM_C];
    float pixels[PIXEL_COUNT];
    float pad_source[PAD_D * PAD_H * PAD_W * PAD_C];
    float conv_source[CONV_INPUT_COUNT], conv_weight[CONV_WEIGHT_COUNT];
    float conv_bias[CONV_O];
    h3_gpu_tensor *source_f32 = new_f32(gpu, source, NORM_COUNT, 1);
    h3_gpu_tensor *weight_f32 = new_f32(gpu, norm_weight, NORM_C, 20);
    h3_gpu_tensor *bias_f32 = new_f32(gpu, norm_bias, NORM_C, 40);
    h3_gpu_tensor *pixels_f32 = new_f32(gpu, pixels, PIXEL_COUNT, 50);
    h3_gpu_tensor *pad_source_f32 = new_f32(
        gpu, pad_source, PAD_D * PAD_H * PAD_W * PAD_C, 60);
    h3_gpu_tensor *conv_source_f32 = new_f32(
        gpu, conv_source, CONV_INPUT_COUNT, 80);
    h3_gpu_tensor *conv_weight_f32 = new_f32(
        gpu, conv_weight, CONV_WEIGHT_COUNT, 100);
    h3_gpu_tensor *conv_bias_f32 = new_f32(gpu, conv_bias, CONV_O, 120);

#define NEW_F16(name, count) \
    h3_gpu_tensor *name = h3_gpu_tensor_new_f16(gpu, (count)); \
    if (!(name)) fail("cannot allocate F16 tensor")
#define NEW_F32(name, count) \
    h3_gpu_tensor *name = h3_gpu_tensor_new_f32(gpu, (count)); \
    if (!(name)) fail("cannot allocate F32 tensor")
    NEW_F16(source_f16, NORM_COUNT);
    NEW_F16(weight_f16, NORM_C);
    NEW_F16(bias_f16, NORM_C);
    NEW_F16(norm_f16, NORM_COUNT);
    NEW_F32(norm_fp16_f32, NORM_COUNT);
    NEW_F32(norm_f32, NORM_COUNT);
    NEW_F16(sum_f16, NORM_COUNT);
    NEW_F32(sum_fp16_f32, NORM_COUNT);
    NEW_F32(sum_f32, NORM_COUNT);
    NEW_F16(pixels_f16, PIXEL_COUNT);
    NEW_F32(pixels_fp16_f32, PIXEL_COUNT);
    NEW_F16(pad_source_f16, PAD_D * PAD_H * PAD_W * PAD_C);
    NEW_F16(pad_f16, PAD_OUT_COUNT);
    NEW_F32(pad_fp16_f32, PAD_OUT_COUNT);
    NEW_F32(pad_f32, PAD_OUT_COUNT);
    NEW_F16(conv_source_f16, CONV_INPUT_COUNT);
    NEW_F16(conv_weight_f16, CONV_WEIGHT_COUNT);
    NEW_F16(conv_bias_f16, CONV_O);
    NEW_F16(conv_f16, CONV_OUTPUT_COUNT);
    NEW_F32(conv_fp16_f32, CONV_OUTPUT_COUNT);
    NEW_F32(conv_f32, CONV_OUTPUT_COUNT);

    gpu_check(gpu, h3_gpu_begin(gpu), "begin");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, source_f16, source_f32, NORM_COUNT), "cast norm input");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, weight_f16, weight_f32, NORM_C), "cast norm weight");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, bias_f16, bias_f32, NORM_C), "cast norm bias");
    gpu_check(gpu, h3_gpu_vae_encoder_group_norm_silu_f32(
        gpu, norm_f32, source_f32, weight_f32, bias_f32,
        1, NORM_D, NORM_H, NORM_W, NORM_C, 4, 1e-6f), "F32 norm");
    gpu_check(gpu, h3_gpu_vae_encoder_group_norm_silu_f16(
        gpu, norm_f16, source_f16, weight_f16, bias_f16,
        1, NORM_D, NORM_H, NORM_W, NORM_C, 4, 1e-6f), "F16 norm");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, norm_fp16_f32, norm_f16, NORM_COUNT), "cast norm result");
    gpu_check(gpu, h3_gpu_add_scaled_f32(
        gpu, sum_f32, norm_f32, norm_f32, 1.0f, 1.0f, NORM_COUNT),
        "F32 residual add");
    gpu_check(gpu, h3_gpu_add_f16(
        gpu, sum_f16, norm_f16, norm_f16, NORM_COUNT), "F16 residual add");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, sum_fp16_f32, sum_f16, NORM_COUNT), "cast residual result");

    gpu_check(gpu, h3_gpu_vae_encoder_normalize_pixels_f16(
        gpu, pixels_f16, pixels_f32, 1, PIXEL_D, PIXEL_H, PIXEL_W),
        "normalize source pixels");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, pixels_fp16_f32, pixels_f16, PIXEL_COUNT),
        "cast normalized pixels");

    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, pad_source_f16, pad_source_f32,
        PAD_D * PAD_H * PAD_W * PAD_C), "cast pad input");
    gpu_check(gpu, h3_gpu_vae_encoder_pad_f32(
        gpu, pad_f32, pad_source_f32, 1, PAD_D, PAD_H, PAD_W, PAD_C,
        2, 1, 1, 1, 1), "F32 pad");
    gpu_check(gpu, h3_gpu_vae_encoder_pad_f16(
        gpu, pad_f16, pad_source_f16, 1, PAD_D, PAD_H, PAD_W, PAD_C,
        2, 1, 1, 1, 1), "F16 pad");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, pad_fp16_f32, pad_f16, PAD_OUT_COUNT), "cast pad result");

    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, conv_source_f16, conv_source_f32, CONV_INPUT_COUNT),
        "cast Conv3D input");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, conv_weight_f16, conv_weight_f32, CONV_WEIGHT_COUNT),
        "cast Conv3D weight");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, conv_bias_f16, conv_bias_f32, CONV_O), "cast Conv3D bias");
    gpu_check(gpu, h3_gpu_conv3d_f32(
        gpu, conv_f32, conv_source_f32, conv_weight_f32, conv_bias_f32,
        1, CONV_D, CONV_H, CONV_W, CONV_I, CONV_O, 3, 3, 3, 1, 1, 1),
        "F32 Conv3D");
    gpu_check(gpu, h3_gpu_conv3d_f16(
        gpu, conv_f16, conv_source_f16, conv_weight_f16, conv_bias_f16,
        1, CONV_D, CONV_H, CONV_W, CONV_I, CONV_O, 3, 3, 3, 1, 1, 1),
        "F16 Conv3D");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, conv_fp16_f32, conv_f16, CONV_OUTPUT_COUNT),
        "cast Conv3D result");
    gpu_check(gpu, h3_gpu_submit(gpu), "submit");

    float got_norm[NORM_COUNT], want_norm[NORM_COUNT];
    float got_sum[NORM_COUNT], want_sum[NORM_COUNT];
    float got_pixels[PIXEL_COUNT], want_pixels[PIXEL_COUNT];
    float got_pad[PAD_OUT_COUNT], want_pad[PAD_OUT_COUNT];
    float got_conv[CONV_OUTPUT_COUNT], want_conv[CONV_OUTPUT_COUNT];
    if (!h3_gpu_tensor_read_f32(norm_fp16_f32, got_norm, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(norm_f32, want_norm, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(sum_fp16_f32, got_sum, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(sum_f32, want_sum, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(
            pixels_fp16_f32, got_pixels, PIXEL_COUNT) ||
        !h3_gpu_tensor_read_f32(pad_fp16_f32, got_pad, PAD_OUT_COUNT) ||
        !h3_gpu_tensor_read_f32(pad_f32, want_pad, PAD_OUT_COUNT) ||
        !h3_gpu_tensor_read_f32(conv_fp16_f32, got_conv, CONV_OUTPUT_COUNT) ||
        !h3_gpu_tensor_read_f32(conv_f32, want_conv, CONV_OUTPUT_COUNT))
        fail("cannot read oracle outputs");
    const float pixel_mean[] = {0.485f, 0.456f, 0.406f};
    const float pixel_std[] = {0.229f, 0.224f, 0.225f};
    size_t target = 0;
    for (int time = 0; time < PIXEL_D; time++)
        for (int y = 0; y < PIXEL_H; y++)
            for (int x = 0; x < PIXEL_W; x++)
                for (int channel = 0; channel < 3; channel++) {
                    size_t source_index =
                        (((size_t)channel * PIXEL_D + (size_t)time) *
                         PIXEL_H + (size_t)y) * PIXEL_W + (size_t)x;
                    want_pixels[target++] =
                        (pixels[source_index] - pixel_mean[channel]) /
                        pixel_std[channel];
                }
    compare("pixel normalize/layout", got_pixels, want_pixels, PIXEL_COUNT,
            0.001, 0.0003);
    compare("GroupNorm+SiLU", got_norm, want_norm, NORM_COUNT, 0.002, 0.001);
    compare("residual add", got_sum, want_sum, NORM_COUNT, 0.004, 0.001);
    compare("causal/reflect pad", got_pad, want_pad, PAD_OUT_COUNT,
            0.00025, 0.0003);
    compare("Conv3D", got_conv, want_conv, CONV_OUTPUT_COUNT, 0.01, 0.0015);

    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(gpu, &stats) || stats.submissions != 1 ||
        stats.blit_copies != 0 || stats.host_tensor_writes != 0 ||
        stats.host_tensor_reads != 9)
        fail("unexpected submission/copy/transfer contract");
    printf("stats: submissions=%llu blits=%llu host-rw=%llu/%llu "
           "bytes=%llu/%llu peak=%.3fMiB conv=%llu direct=%llu\n",
           (unsigned long long)stats.submissions,
           (unsigned long long)stats.blit_copies,
           (unsigned long long)stats.host_tensor_reads,
           (unsigned long long)stats.host_tensor_writes,
           (unsigned long long)stats.host_tensor_read_bytes,
           (unsigned long long)stats.host_tensor_write_bytes,
           (double)stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)stats.mps_conv_dispatches,
           (unsigned long long)stats.direct_dispatches);
    h3_gpu_free(gpu);
    puts("ok: native FP16 VideoVAE encoder kernels match F32 oracles");
    return 0;
}
