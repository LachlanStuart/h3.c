#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
                            (CONV_W - 2) * CONV_O,
        /* Decoder attention uses this exact layout: [row,head,3,dimension]. */
        DECODER_SEQUENCE = 3, DECODER_HEADS = 2, DECODER_HEAD_DIM = 4,
        DECODER_ROPE_HALF = 1,
        DECODER_QKV_COUNT = DECODER_SEQUENCE * DECODER_HEADS *
                            DECODER_HEAD_DIM * 3,
        DECODER_Q_COUNT = DECODER_SEQUENCE * DECODER_HEADS * DECODER_HEAD_DIM,
        DECODER_ROPE_COUNT = DECODER_SEQUENCE * DECODER_ROPE_HALF,
        PACK_PATCH_ROWS = 2, PACK_REGISTERS = 4, PACK_SUFFIX = 1,
        PACK_WIDTH = 4, PACK_COUNT = (PACK_PATCH_ROWS + PACK_REGISTERS +
                                      PACK_SUFFIX) * PACK_WIDTH,
        UNPACK_SEQUENCE = 12, UNPACK_PATCH = 3 * 4 * 16 * 16,
        UNPACK_PROJECTED = UNPACK_SEQUENCE * UNPACK_PATCH,
        UNPACK_RGB = 22 * 16 * 16 * 3
    };
    float source[NORM_COUNT], norm_weight[NORM_C], norm_bias[NORM_C];
    float pixels[PIXEL_COUNT];
    float pad_source[PAD_D * PAD_H * PAD_W * PAD_C];
    float conv_source[CONV_INPUT_COUNT], conv_weight[CONV_WEIGHT_COUNT];
    float conv_bias[CONV_O];
    float decoder_qkv[DECODER_QKV_COUNT], decoder_rope_cos[DECODER_ROPE_COUNT];
    float decoder_rope_sin[DECODER_ROPE_COUNT];
    float pack_patches[PACK_PATCH_ROWS * PACK_WIDTH];
    float pack_registers[PACK_REGISTERS * PACK_WIDTH];
    float unpack_projected[UNPACK_PROJECTED];
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
    h3_gpu_tensor *decoder_qkv_f32 = new_f32(
        gpu, decoder_qkv, DECODER_QKV_COUNT, 140);
    h3_gpu_tensor *decoder_rope_cos_f32 = new_f32(
        gpu, decoder_rope_cos, DECODER_ROPE_COUNT, 160);
    h3_gpu_tensor *decoder_rope_sin_f32 = new_f32(
        gpu, decoder_rope_sin, DECODER_ROPE_COUNT, 180);
    h3_gpu_tensor *pack_patches_f32 = new_f32(gpu, pack_patches,
        PACK_PATCH_ROWS * PACK_WIDTH, 200);
    h3_gpu_tensor *pack_registers_f32 = new_f32(gpu, pack_registers,
        PACK_REGISTERS * PACK_WIDTH, 220);
    memset(unpack_projected, 0, sizeof(unpack_projected));
    for (int patch = 0; patch < 7; patch++)
        for (int component = 0; component < UNPACK_PATCH; component++)
            unpack_projected[patch * UNPACK_PATCH + component] =
                (float)(patch & 1);
    h3_gpu_tensor *unpack_projected_f32 = h3_gpu_tensor_from_f32(
        gpu, unpack_projected, UNPACK_PROJECTED);
    if (!unpack_projected_f32) fail("cannot allocate unpack source");

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
    NEW_F16(decoder_qkv_f16, DECODER_QKV_COUNT);
    NEW_F16(decoder_query_f16, DECODER_Q_COUNT);
    NEW_F16(decoder_key_f16, DECODER_Q_COUNT);
    NEW_F16(decoder_value_f16, DECODER_Q_COUNT);
    NEW_F32(decoder_query_fp16_f32, DECODER_Q_COUNT);
    NEW_F32(decoder_key_fp16_f32, DECODER_Q_COUNT);
    NEW_F32(decoder_value_fp16_f32, DECODER_Q_COUNT);
    NEW_F32(decoder_query_f32, DECODER_Q_COUNT);
    NEW_F32(decoder_key_f32, DECODER_Q_COUNT);
    NEW_F32(decoder_value_f32, DECODER_Q_COUNT);
    NEW_F16(pack_patches_f16, PACK_PATCH_ROWS * PACK_WIDTH);
    NEW_F16(pack_registers_f16, PACK_REGISTERS * PACK_WIDTH);
    NEW_F16(pack_output_f16, PACK_COUNT);
    NEW_F32(pack_output_f32, PACK_COUNT);
    NEW_F16(unpack_projected_f16, UNPACK_PROJECTED);
    NEW_F32(unpack_rgb_f32, UNPACK_RGB);
    NEW_F32(captured_tiles_f32, UNPACK_RGB * 2);

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
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, decoder_qkv_f16, decoder_qkv_f32, DECODER_QKV_COUNT),
        "cast decoder QKV");
    gpu_check(gpu, h3_gpu_video_qkv_rope_f32(
        gpu, decoder_query_f32, decoder_key_f32, decoder_value_f32,
        decoder_qkv_f32, decoder_rope_cos_f32, decoder_rope_sin_f32,
        DECODER_SEQUENCE, DECODER_HEADS, DECODER_HEAD_DIM,
        DECODER_ROPE_HALF, 1e-5f), "F32 decoder QKV/RoPE");
    gpu_check(gpu, h3_gpu_video_qkv_rope_f16(
        gpu, decoder_query_f16, decoder_key_f16, decoder_value_f16,
        decoder_qkv_f16, decoder_rope_cos_f32, decoder_rope_sin_f32,
        DECODER_SEQUENCE, DECODER_HEADS, DECODER_HEAD_DIM,
        DECODER_ROPE_HALF, 1e-5f), "F16 decoder QKV/RoPE");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, decoder_query_fp16_f32, decoder_query_f16, DECODER_Q_COUNT),
        "cast decoder query result");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, decoder_key_fp16_f32, decoder_key_f16, DECODER_Q_COUNT),
        "cast decoder key result");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, decoder_value_fp16_f32, decoder_value_f16, DECODER_Q_COUNT),
        "cast decoder value result");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, pack_patches_f16, pack_patches_f32, PACK_PATCH_ROWS * PACK_WIDTH),
        "cast pack patches");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, pack_registers_f16, pack_registers_f32, PACK_REGISTERS * PACK_WIDTH),
        "cast pack registers");
    gpu_check(gpu, h3_gpu_video_vae_pack_f16(
        gpu, pack_output_f16, pack_patches_f16, pack_registers_f16,
        PACK_PATCH_ROWS, PACK_REGISTERS, PACK_SUFFIX, PACK_WIDTH),
        "pack FP16 decoder sequence");
    gpu_check(gpu, h3_gpu_cast_f16_to_f32(
        gpu, pack_output_f32, pack_output_f16, PACK_COUNT), "cast pack result");
    gpu_check(gpu, h3_gpu_cast_f32_to_f16(
        gpu, unpack_projected_f16, unpack_projected_f32, UNPACK_PROJECTED),
        "cast unpack projected patches");
    gpu_check(gpu, h3_gpu_video_vae_unpack_rgb_f16(
        gpu, unpack_rgb_f32, unpack_projected_f16, 1, 1, 22),
        "unpack FP16 decoder RGB");
    /* This is deliberately dispatched, not merely declared: omitting its
     * pipeline from h3_gpu_create must fail this GPU construction test. */
    gpu_check(gpu, h3_gpu_video_vae_capture_tile_f32(
        gpu, captured_tiles_f32, unpack_rgb_f32, 1, UNPACK_RGB),
        "capture GPU-resident decoder tile");
    gpu_check(gpu, h3_gpu_submit(gpu), "submit");

    float got_norm[NORM_COUNT], want_norm[NORM_COUNT];
    float got_sum[NORM_COUNT], want_sum[NORM_COUNT];
    float got_pixels[PIXEL_COUNT], want_pixels[PIXEL_COUNT];
    float got_pad[PAD_OUT_COUNT], want_pad[PAD_OUT_COUNT];
    float got_conv[CONV_OUTPUT_COUNT], want_conv[CONV_OUTPUT_COUNT];
    float got_decoder_query[DECODER_Q_COUNT], want_decoder_query[DECODER_Q_COUNT];
    float got_decoder_key[DECODER_Q_COUNT], want_decoder_key[DECODER_Q_COUNT];
    float got_decoder_value[DECODER_Q_COUNT], want_decoder_value[DECODER_Q_COUNT];
    float got_pack[PACK_COUNT], got_unpack[UNPACK_RGB];
    if (!h3_gpu_tensor_read_f32(norm_fp16_f32, got_norm, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(norm_f32, want_norm, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(sum_fp16_f32, got_sum, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(sum_f32, want_sum, NORM_COUNT) ||
        !h3_gpu_tensor_read_f32(
            pixels_fp16_f32, got_pixels, PIXEL_COUNT) ||
        !h3_gpu_tensor_read_f32(pad_fp16_f32, got_pad, PAD_OUT_COUNT) ||
        !h3_gpu_tensor_read_f32(pad_f32, want_pad, PAD_OUT_COUNT) ||
        !h3_gpu_tensor_read_f32(conv_fp16_f32, got_conv, CONV_OUTPUT_COUNT) ||
        !h3_gpu_tensor_read_f32(conv_f32, want_conv, CONV_OUTPUT_COUNT) ||
        !h3_gpu_tensor_read_f32(decoder_query_fp16_f32, got_decoder_query,
                                DECODER_Q_COUNT) ||
        !h3_gpu_tensor_read_f32(decoder_query_f32, want_decoder_query,
                                DECODER_Q_COUNT) ||
        !h3_gpu_tensor_read_f32(decoder_key_fp16_f32, got_decoder_key,
                                DECODER_Q_COUNT) ||
        !h3_gpu_tensor_read_f32(decoder_key_f32, want_decoder_key,
                                DECODER_Q_COUNT) ||
        !h3_gpu_tensor_read_f32(decoder_value_fp16_f32, got_decoder_value,
                                DECODER_Q_COUNT) ||
        !h3_gpu_tensor_read_f32(decoder_value_f32, want_decoder_value,
                                DECODER_Q_COUNT) ||
        !h3_gpu_tensor_read_f32(pack_output_f32, got_pack, PACK_COUNT) ||
        !h3_gpu_tensor_read_f32(unpack_rgb_f32, got_unpack, UNPACK_RGB))
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
    compare("decoder QKV/RoPE query", got_decoder_query, want_decoder_query,
            DECODER_Q_COUNT, 0.002, 0.001);
    compare("decoder QKV/RoPE key", got_decoder_key, want_decoder_key,
            DECODER_Q_COUNT, 0.002, 0.001);
    compare("decoder QKV/RoPE value", got_decoder_value, want_decoder_value,
            DECODER_Q_COUNT, 0.00025, 0.0003);
    for (int row = 0; row < PACK_PATCH_ROWS; row++)
        compare("decoder pack patches", got_pack + row * PACK_WIDTH,
                pack_patches + row * PACK_WIDTH, PACK_WIDTH, 0.00025, 0.0003);
    for (int row = 0; row < PACK_REGISTERS; row++)
        compare("decoder pack registers", got_pack + (row + PACK_PATCH_ROWS) * PACK_WIDTH,
                pack_registers + row * PACK_WIDTH, PACK_WIDTH, 0.00025, 0.0003);
    for (int index = (PACK_PATCH_ROWS + PACK_REGISTERS) * PACK_WIDTH;
         index < PACK_COUNT; index++) if (got_pack[index] != 0.0f)
        fail("decoder pack suffix is not zero");
    for (int frame = 0; frame < 22; frame++) {
        int decoded_t = frame + 3;
        if (frame >= 17) decoded_t += 3;
        float expected = (decoded_t / 4 & 1) ? 0.714f : 0.485f;
        float observed = got_unpack[(size_t)frame * 16 * 16 * 3];
        if (fabsf(observed - expected) > 0.0005f)
            fail("decoder unpack temporal patch mapping is wrong");
    }

    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(gpu, &stats) || stats.submissions != 1 ||
        stats.blit_copies != 0 || stats.host_tensor_writes != 0 ||
        stats.host_tensor_reads != 17)
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
