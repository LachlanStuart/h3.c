#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROWS = 128, INPUT = 256, OUTPUT = 128, RANK = 8 };

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_convrot.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static uint32_t random_state = UINT32_C(0x6d2b79f5);

static uint32_t random_u32(void) {
    uint32_t value = random_state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    random_state = value;
    return value;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void hadamard256(float values[256]) {
    for (unsigned stride = 1; stride <= 64; stride *= 4)
        for (unsigned operation = 0; operation < 64; operation++) {
            unsigned quartet = operation / stride;
            unsigned offset = operation % stride;
            unsigned index = quartet * stride * 4 + offset;
            float a = values[index];
            float b = values[index + stride];
            float c = values[index + stride * 2];
            float d = values[index + stride * 3];
            values[index] = a + b + c - d;
            values[index + stride] = a + b - c + d;
            values[index + stride * 2] = a - b + c + d;
            values[index + stride * 3] = -a + b + c + d;
        }
}

static void test_convrot_projection(h3_gpu *gpu) {
    const size_t input_count = (size_t)ROWS * INPUT;
    const size_t weight_count = (size_t)OUTPUT * INPUT;
    const size_t output_count = (size_t)ROWS * OUTPUT;
    uint16_t *input = malloc(input_count * sizeof(*input));
    uint16_t *rotated = malloc(input_count * sizeof(*rotated));
    int8_t *quantized = malloc(input_count);
    int8_t *weight = malloc(weight_count);
    float *input_scales = malloc(ROWS * sizeof(*input_scales));
    float *weight_scales = malloc(OUTPUT * sizeof(*weight_scales));
    uint16_t *output = malloc(output_count * sizeof(*output));
    require(input && rotated && quantized && weight && input_scales &&
            weight_scales && output, "host allocation failed");
    for (size_t index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            ((float)(int32_t)(random_u32() % 4001) - 2000.0f) / 997.0f);
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (int8_t)((int32_t)(random_u32() % 63) - 31);
    for (unsigned index = 0; index < OUTPUT; index++)
        weight_scales[index] = 0.001f +
            (float)(random_u32() % 4000) * 0.000001f;
    for (unsigned row = 0; row < ROWS; row++) {
        float values[INPUT];
        float maximum = 0.0f;
        for (unsigned column = 0; column < INPUT; column++)
            values[column] = bf16_to_f32(input[(size_t)row * INPUT + column]);
        hadamard256(values);
        for (unsigned column = 0; column < INPUT; column++) {
            uint16_t rounded = f32_to_bf16(values[column] / 16.0f);
            rotated[(size_t)row * INPUT + column] = rounded;
            maximum = fmaxf(maximum, fabsf(bf16_to_f32(rounded)));
        }
        float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f / 127.0f;
        float inverse = 1.0f / bf16_to_f32(f32_to_bf16(scale));
        input_scales[row] = scale;
        for (unsigned column = 0; column < INPUT; column++) {
            float ratio = bf16_to_f32(f32_to_bf16(
                bf16_to_f32(rotated[(size_t)row * INPUT + column]) * inverse));
            long value = lrintf(ratio);
            if (value < -128) value = -128;
            if (value > 127) value = 127;
            quantized[(size_t)row * INPUT + column] = (int8_t)value;
        }
    }
    h3_gpu_tensor *input_gpu = h3_gpu_tensor_from_bf16(gpu, input, input_count);
    h3_gpu_tensor *rotated_gpu = h3_gpu_tensor_new_bf16(gpu, input_count);
    h3_gpu_tensor *quantized_gpu = h3_gpu_tensor_new_i8(gpu, input_count);
    h3_gpu_tensor *input_scales_gpu = h3_gpu_tensor_new_f32(gpu, ROWS);
    h3_gpu_tensor *weight_gpu = h3_gpu_tensor_from_i8(gpu, weight, weight_count);
    h3_gpu_tensor *weight_scales_gpu = h3_gpu_tensor_from_f32(
        gpu, weight_scales, OUTPUT);
    h3_gpu_tensor *output_gpu = h3_gpu_tensor_new_bf16(gpu, output_count);
    require(input_gpu && rotated_gpu && quantized_gpu && input_scales_gpu &&
            weight_gpu && weight_scales_gpu && output_gpu,
            "Metal allocation failed");
    require(h3_gpu_begin(gpu) && h3_gpu_linear_convrot_int8_bf16(
                gpu, output_gpu, rotated_gpu, quantized_gpu, input_scales_gpu,
                input_gpu, weight_gpu, weight_scales_gpu, ROWS, INPUT, OUTPUT) &&
            h3_gpu_submit(gpu), "ConvRot projection failed");
    uint16_t *rotated_got = malloc(input_count * sizeof(*rotated_got));
    int8_t *quantized_got = malloc(input_count);
    float *scales_got = malloc(ROWS * sizeof(*scales_got));
    require(rotated_got && quantized_got && scales_got,
            "result allocation failed");
    require(h3_gpu_tensor_read_bf16(rotated_gpu, rotated_got, input_count) &&
            h3_gpu_tensor_read_i8(quantized_gpu, quantized_got, input_count) &&
            h3_gpu_tensor_read_f32(input_scales_gpu, scales_got, ROWS) &&
            h3_gpu_tensor_read_bf16(output_gpu, output, output_count),
            "cannot read ConvRot results");
    require(!memcmp(rotated, rotated_got, input_count * sizeof(*rotated)),
            "Hadamard boundary differs from the BF16 reference");
    require(!memcmp(quantized, quantized_got, input_count),
            "ConvRot quantization boundary differs from TensorWiseINT8");
    for (unsigned row = 0; row < ROWS; row++)
        require(scales_got[row] == input_scales[row],
                "ConvRot activation scale differs from reference");
    for (unsigned row = 0; row < ROWS; row++)
        for (unsigned column = 0; column < OUTPUT; column++) {
            int32_t sum = 0;
            for (unsigned inner = 0; inner < INPUT; inner++)
                sum += (int32_t)quantized[(size_t)row * INPUT + inner] *
                       (int32_t)weight[(size_t)column * INPUT + inner];
            float expected = bf16_to_f32(f32_to_bf16((float)sum *
                input_scales[row] * weight_scales[column]));
            require(bf16_to_f32(output[(size_t)row * OUTPUT + column]) ==
                    expected, "ConvRot projection differs from BF16 reference");
        }
    h3_gpu_tensor_free(input_gpu); h3_gpu_tensor_free(rotated_gpu);
    h3_gpu_tensor_free(quantized_gpu); h3_gpu_tensor_free(input_scales_gpu);
    h3_gpu_tensor_free(weight_gpu); h3_gpu_tensor_free(weight_scales_gpu);
    h3_gpu_tensor_free(output_gpu);
    free(input); free(rotated); free(quantized); free(weight); free(input_scales);
    free(weight_scales); free(output); free(rotated_got); free(quantized_got);
    free(scales_got);
}

static void test_qkv_layout(h3_gpu *gpu) {
    enum { HEADS = 2, HEAD_DIM = 128, INNER = HEADS * HEAD_DIM, ROPE = 48 };
    uint16_t qkv[INNER * 3], norm[HEAD_DIM], cosine[ROPE], sine[ROPE];
    for (unsigned index = 0; index < INNER; index++) {
        qkv[index] = f32_to_bf16(1.0f);
        qkv[INNER + index] = f32_to_bf16(2.0f);
        qkv[INNER * 2 + index] = f32_to_bf16(3.0f);
    }
    for (unsigned index = 0; index < HEAD_DIM; index++) norm[index] = f32_to_bf16(1.0f);
    for (unsigned index = 0; index < ROPE; index++) {
        cosine[index] = f32_to_bf16(1.0f);
        sine[index] = 0;
    }
    h3_gpu_tensor *qkv_gpu = h3_gpu_tensor_from_bf16(gpu, qkv, INNER * 3);
    h3_gpu_tensor *norm_gpu = h3_gpu_tensor_from_bf16(gpu, norm, HEAD_DIM);
    h3_gpu_tensor *cos_gpu = h3_gpu_tensor_from_bf16(gpu, cosine, ROPE);
    h3_gpu_tensor *sin_gpu = h3_gpu_tensor_from_bf16(gpu, sine, ROPE);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, INNER);
    require(qkv_gpu && norm_gpu && cos_gpu && sin_gpu && query && key && value,
            "QKV allocation failed");
    require(h3_gpu_begin(gpu) && h3_gpu_qkv_rope_bf16(
                gpu, query, key, value, qkv_gpu, norm_gpu, norm_gpu, cos_gpu,
                sin_gpu, 1, HEADS, HEAD_DIM, ROPE, 1e-5f) && h3_gpu_submit(gpu),
            "contiguous QKV decode failed");
    uint16_t values[INNER];
    require(h3_gpu_tensor_read_bf16(value, values, INNER),
            "cannot read contiguous QKV result");
    for (unsigned index = 0; index < INNER; index++)
        require(values[index] == f32_to_bf16(3.0f),
                "contiguous QKV value range was decoded as grouped");
    h3_gpu_tensor_free(qkv_gpu); h3_gpu_tensor_free(norm_gpu);
    h3_gpu_tensor_free(cos_gpu); h3_gpu_tensor_free(sin_gpu);
    h3_gpu_tensor_free(query); h3_gpu_tensor_free(key); h3_gpu_tensor_free(value);
}

static void test_rank8_schedule(h3_gpu *gpu) {
    enum { TABLE_ROWS = 1025, TIME_ROWS = 3, OUTPUTS = 4 };
    const float times[TIME_ROWS] = {0.0f, 0.5f, 1.0f};
    float *table = malloc((size_t)TABLE_ROWS * RANK * sizeof(*table));
    uint16_t weights[OUTPUTS * RANK];
    uint16_t bias[OUTPUTS] = {0};
    require(table != NULL, "rank-8 table allocation failed");
    for (unsigned row = 0; row < TABLE_ROWS; row++)
        for (unsigned rank = 0; rank < RANK; rank++)
            table[(size_t)row * RANK + rank] = (float)(row + rank);
    for (unsigned index = 0; index < OUTPUTS * RANK; index++)
        weights[index] = UINT16_C(0x3c00);
    h3_gpu_tensor *time = h3_gpu_tensor_from_f32(gpu, times, TIME_ROWS);
    h3_gpu_tensor *table_gpu = h3_gpu_tensor_from_f32(
        gpu, table, (size_t)TABLE_ROWS * RANK);
    h3_gpu_tensor *features = h3_gpu_tensor_new_f32(gpu, TIME_ROWS * RANK);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f16(gpu, weights,
                                                    OUTPUTS * RANK);
    h3_gpu_tensor *bias_gpu = h3_gpu_tensor_from_f16(gpu, bias, OUTPUTS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, TIME_ROWS * OUTPUTS);
    require(time && table_gpu && features && weight && bias_gpu && output,
            "rank-8 Metal allocation failed");
    require(h3_gpu_begin(gpu) && h3_gpu_adaln_table_interpolate_f32(
                gpu, features, time, table_gpu, TIME_ROWS, TABLE_ROWS) &&
            h3_gpu_linear_rank8_f16_bf16(
                gpu, output, features, weight, bias_gpu, TIME_ROWS, OUTPUTS) &&
            h3_gpu_submit(gpu), "rank-8 schedule projection failed");
    float actual_features[TIME_ROWS * RANK];
    uint16_t actual_output[TIME_ROWS * OUTPUTS];
    require(h3_gpu_tensor_read_f32(features, actual_features,
                                   TIME_ROWS * RANK) &&
            h3_gpu_tensor_read_bf16(output, actual_output,
                                     TIME_ROWS * OUTPUTS),
            "cannot read rank-8 results");
    for (unsigned row = 0; row < TIME_ROWS; row++) {
        unsigned source_row = row * 512;
        for (unsigned rank = 0; rank < RANK; rank++)
            require(actual_features[row * RANK + rank] ==
                    (float)(source_row + rank),
                    "rank-8 table interpolation differs from reference");
        uint16_t expected = f32_to_bf16((float)(source_row * RANK + 28));
        for (unsigned column = 0; column < OUTPUTS; column++)
            require(actual_output[row * OUTPUTS + column] == expected,
                    "rank-8 BF16 projection differs from reference");
    }
    h3_gpu_tensor_free(time); h3_gpu_tensor_free(table_gpu);
    h3_gpu_tensor_free(features); h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(bias_gpu); h3_gpu_tensor_free(output);
    free(table);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);
    require(h3_gpu_has_int8_mlp(gpu),
            "ConvRot requires Apple Metal INT8 tensor support");
    test_convrot_projection(gpu);
    test_qkv_layout(gpu);
    test_rank8_schedule(gpu);
    h3_gpu_free(gpu);
    puts("ConvRot tests passed");
    return 0;
}
