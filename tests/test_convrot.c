#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { FULL_ROWS = 128, SHORT_ROWS = 87, INPUT = 256, OUTPUT = 128, RANK = 8 };

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

static uint16_t f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & UINT32_C(0x8000);
    int32_t exponent = (int32_t)((bits >> 23) & UINT32_C(0xff)) - 127 + 15;
    uint32_t mantissa = bits & UINT32_C(0x7fffff);
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa |= UINT32_C(0x800000);
        uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t half = mantissa >> shift;
        uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1);
        uint32_t halfway = UINT32_C(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    if (exponent >= 31) return (uint16_t)(sign | UINT32_C(0x7c00));
    uint32_t half = mantissa >> 13;
    uint32_t remainder = mantissa & UINT32_C(0x1fff);
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) && (half & 1))) {
        half++;
        if (half == UINT32_C(0x400)) {
            half = 0;
            exponent++;
            if (exponent >= 31) return (uint16_t)(sign | UINT32_C(0x7c00));
        }
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | half);
}

static float f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & UINT16_C(0x8000)) << 16;
    uint32_t exponent = (value >> 10) & UINT16_C(0x1f);
    uint32_t mantissa = value & UINT16_C(0x03ff);
    uint32_t bits;
    if (!exponent) {
        if (!mantissa) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & UINT32_C(0x400)) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= UINT32_C(0x3ff);
            bits = sign | (uint32_t)(127 - 14 - shift) << 23 |
                   mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | UINT32_C(0x7f800000) | mantissa << 13;
    } else {
        bits = sign | (exponent + 112) << 23 | mantissa << 13;
    }
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

static void test_dense_hadamard_reference(void) {
    static const int h4[4][4] = {
        {1, 1, 1, -1}, {1, 1, -1, 1},
        {1, -1, 1, 1}, {-1, 1, 1, 1}
    };
    float input[256], fast[256];
    for (unsigned index = 0; index < 256; index++)
        input[index] = fast[index] =
            ((float)(int32_t)(random_u32() % 4001) - 2000.0f) / 997.0f;
    hadamard256(fast);
    float maximum_error = 0.0f;
    for (unsigned output = 0; output < 256; output++) {
        float dense = 0.0f;
        for (unsigned inner = 0; inner < 256; inner++) {
            int sign = 1;
            unsigned left = inner, right = output;
            for (unsigned digit = 0; digit < 4; digit++) {
                sign *= h4[left & 3u][right & 3u];
                left >>= 2;
                right >>= 2;
            }
            dense += input[inner] * (float)sign;
        }
        maximum_error = fmaxf(maximum_error, fabsf(dense - fast[output]));
    }
    require(maximum_error <= 0.0001f,
            "radix-4 Hadamard differs from independent dense Kronecker reference");
}

static void test_convrot(h3_gpu *gpu, unsigned rows) {
    size_t input_count = (size_t)rows * INPUT;
    size_t weight_count = (size_t)OUTPUT * INPUT;
    size_t output_count = (size_t)rows * OUTPUT;
    size_t padded_rows = ((size_t)rows + 127u) & ~(size_t)127u;
    uint16_t *input = malloc(input_count * sizeof(*input));
    uint16_t *rotated_want = malloc(input_count * sizeof(*rotated_want));
    uint16_t *rotated_got = malloc(input_count * sizeof(*rotated_got));
    int8_t *quantized_want = malloc(input_count);
    int8_t *quantized_got = malloc(input_count);
    int8_t *weight = malloc(weight_count);
    float *input_scales_want = malloc((size_t)rows * sizeof(float));
    float *input_scales_got = malloc((size_t)rows * sizeof(float));
    float *weight_scales = malloc(OUTPUT * sizeof(float));
    uint16_t *output_got = malloc(output_count * sizeof(*output_got));
    require(input && rotated_want && rotated_got && quantized_want &&
            quantized_got && weight && input_scales_want && input_scales_got &&
            weight_scales && output_got, "host allocation failed");
    for (size_t index = 0; index < input_count; index++) {
        float value = ((float)(int32_t)(random_u32() % 4001) - 2000.0f) /
                      997.0f;
        input[index] = f32_to_bf16(value);
    }
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (int8_t)((int32_t)(random_u32() % 63) - 31);
    for (unsigned column = 0; column < OUTPUT; column++)
        weight_scales[column] = 0.001f +
            (float)(random_u32() % 4000) * 0.000001f;
    for (unsigned row = 0; row < rows; row++) {
        float values[256];
        float maximum = 0.0f;
        for (unsigned column = 0; column < INPUT; column++)
            values[column] = bf16_to_f32(input[(size_t)row * INPUT + column]);
        hadamard256(values);
        for (unsigned column = 0; column < INPUT; column++) {
            uint16_t rounded = f32_to_bf16(values[column] / 16.0f);
            rotated_want[(size_t)row * INPUT + column] = rounded;
            maximum = fmaxf(maximum, fabsf(bf16_to_f32(rounded)));
        }
        float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f / 127.0f;
        float quantization_scale = bf16_to_f32(f32_to_bf16(scale));
        float inverse = 1.0f / quantization_scale;
        input_scales_want[row] = scale;
        for (unsigned column = 0; column < INPUT; column++) {
            float ratio = bf16_to_f32(f32_to_bf16(bf16_to_f32(
                rotated_want[(size_t)row * INPUT + column]) * inverse));
            long rounded = lrintf(ratio);
            if (rounded < -128) rounded = -128;
            if (rounded > 127) rounded = 127;
            quantized_want[(size_t)row * INPUT + column] = (int8_t)rounded;
        }
    }
    h3_gpu_tensor *input_gpu = h3_gpu_tensor_from_bf16(gpu, input, input_count);
    h3_gpu_tensor *rotated_gpu = h3_gpu_tensor_new_bf16(gpu, input_count);
    /* The ConvRot quantizer pads only its int8 data and per-row scale tiles.
     * The source, rotated, and output tensors keep their semantic row count. */
    h3_gpu_tensor *quantized_gpu = h3_gpu_tensor_new_i8(
        gpu, padded_rows * INPUT);
    h3_gpu_tensor *input_scales_gpu = h3_gpu_tensor_new_f32(gpu, padded_rows);
    h3_gpu_tensor *weight_gpu = h3_gpu_tensor_from_i8(gpu, weight, weight_count);
    h3_gpu_tensor *weight_scales_gpu = h3_gpu_tensor_from_f32(
        gpu, weight_scales, OUTPUT);
    h3_gpu_tensor *output_gpu = h3_gpu_tensor_new_bf16(gpu, output_count);
    require(input_gpu && rotated_gpu && quantized_gpu && input_scales_gpu &&
            weight_gpu && weight_scales_gpu && output_gpu,
            "Metal tensor allocation failed");
    require(h3_gpu_begin(gpu), "cannot begin ConvRot command buffer");
    if (!h3_gpu_linear_convrot_int8_bf16(
            gpu, output_gpu, rotated_gpu, quantized_gpu, input_scales_gpu,
            input_gpu, weight_gpu, weight_scales_gpu,
            rows, INPUT, OUTPUT)) {
        fprintf(stderr, "FAIL tests/test_convrot.c: ConvRot dispatch: %s\n",
                h3_gpu_error(gpu));
        exit(1);
    }
    require(h3_gpu_submit(gpu), "cannot submit ConvRot command buffer");
    require(h3_gpu_tensor_read_bf16(rotated_gpu, rotated_got, input_count),
            "cannot read rotated activation");
    require(h3_gpu_tensor_read_i8(quantized_gpu, quantized_got, input_count),
            "cannot read quantized activation");
    require(h3_gpu_tensor_read_f32(input_scales_gpu, input_scales_got, rows),
            "cannot read ConvRot scales");
    require(h3_gpu_tensor_read_bf16(output_gpu, output_got, output_count),
            "cannot read ConvRot output");
    require(memcmp(rotated_want, rotated_got,
                   input_count * sizeof(*rotated_want)) == 0,
            "Metal Hadamard rotation differs from the scalar reference");
    require(memcmp(quantized_want, quantized_got, input_count) == 0,
            "Metal ConvRot quantization differs from the scalar reference");
    float maximum_scale_error = 0.0f;
    float maximum_output_error = 0.0f;
    for (unsigned row = 0; row < rows; row++)
        maximum_scale_error = fmaxf(maximum_scale_error,
            fabsf(input_scales_got[row] - input_scales_want[row]));
    for (unsigned row = 0; row < rows; row++)
        for (unsigned column = 0; column < OUTPUT; column++) {
            int32_t sum = 0;
            for (unsigned inner = 0; inner < INPUT; inner++)
                sum += (int32_t)quantized_want[(size_t)row * INPUT + inner] *
                       (int32_t)weight[(size_t)column * INPUT + inner];
            float want = (float)sum * input_scales_want[row] *
                         weight_scales[column];
            float got = bf16_to_f32(
                output_got[(size_t)row * OUTPUT + column]);
            maximum_output_error = fmaxf(maximum_output_error,
                                         fabsf(got - want));
        }
    require(maximum_scale_error <= 1e-7f,
            "Metal ConvRot row scale exceeds tolerance");
    require(maximum_output_error <= 0.02f,
            "Metal ConvRot matrix output exceeds tolerance");
    printf("ConvRot %u-row matrix max abs %.7g, scale %.7g\n", rows,
           maximum_output_error, maximum_scale_error);
    h3_gpu_tensor_free(input_gpu); h3_gpu_tensor_free(rotated_gpu);
    h3_gpu_tensor_free(quantized_gpu); h3_gpu_tensor_free(input_scales_gpu);
    h3_gpu_tensor_free(weight_gpu); h3_gpu_tensor_free(weight_scales_gpu);
    h3_gpu_tensor_free(output_gpu);
    free(input); free(rotated_want); free(rotated_got); free(quantized_want);
    free(quantized_got); free(weight); free(input_scales_want);
    free(input_scales_got); free(weight_scales); free(output_got);
}

static void test_rank8(h3_gpu *gpu) {
    enum { CURVE_ROWS = 1025, TIME_ROWS = 5, OUTPUTS = 64 };
    const float times[TIME_ROWS] = {0.0f, 1.0f, 0.5f, 0.12345f, 0.9997f};
    float *table = malloc((size_t)CURVE_ROWS * RANK * sizeof(*table));
    float features_want[TIME_ROWS * RANK];
    float features_got[TIME_ROWS * RANK];
    uint16_t weights[OUTPUTS * RANK];
    uint16_t biases[OUTPUTS];
    uint16_t output_got[TIME_ROWS * OUTPUTS];
    require(table != NULL, "curve allocation failed");
    for (unsigned row = 0; row < CURVE_ROWS; row++)
        for (unsigned rank = 0; rank < RANK; rank++)
            table[(size_t)row * RANK + rank] =
                sinf((float)(row * (rank + 1)) * 0.0031f) +
                (float)rank * 0.01f;
    for (unsigned row = 0; row < TIME_ROWS; row++) {
        float position = fminf(fmaxf(times[row], 0.0f), 1.0f) * 1024.0f;
        unsigned lower = (unsigned)floorf(position);
        if (lower > 1023) lower = 1023;
        float fraction = position - (float)lower;
        for (unsigned rank = 0; rank < RANK; rank++) {
            float left = table[(size_t)lower * RANK + rank];
            float right = table[(size_t)(lower + 1) * RANK + rank];
            features_want[row * RANK + rank] =
                left + (right - left) * fraction;
        }
    }
    for (unsigned index = 0; index < OUTPUTS * RANK; index++)
        weights[index] = f32_to_f16(
            ((float)(int32_t)(random_u32() % 257) - 128.0f) / 128.0f);
    for (unsigned index = 0; index < OUTPUTS; index++)
        biases[index] = f32_to_f16(
            ((float)(int32_t)(random_u32() % 65) - 32.0f) / 64.0f);
    h3_gpu_tensor *times_gpu = h3_gpu_tensor_from_f32(gpu, times, TIME_ROWS);
    h3_gpu_tensor *table_gpu = h3_gpu_tensor_from_f32(
        gpu, table, (size_t)CURVE_ROWS * RANK);
    h3_gpu_tensor *features_gpu = h3_gpu_tensor_new_f32(
        gpu, TIME_ROWS * RANK);
    h3_gpu_tensor *weights_gpu = h3_gpu_tensor_from_f16(
        gpu, weights, OUTPUTS * RANK);
    h3_gpu_tensor *biases_gpu = h3_gpu_tensor_from_f16(gpu, biases, OUTPUTS);
    h3_gpu_tensor *output_gpu = h3_gpu_tensor_new_bf16(
        gpu, TIME_ROWS * OUTPUTS);
    require(times_gpu && table_gpu && features_gpu && weights_gpu &&
            biases_gpu && output_gpu, "rank-8 Metal allocation failed");
    require(h3_gpu_begin(gpu), "cannot begin rank-8 command buffer");
    require(h3_gpu_adaln_table_interpolate_f32(
                gpu, features_gpu, times_gpu, table_gpu, TIME_ROWS, CURVE_ROWS),
            "cannot interpolate rank-8 curve");
    require(h3_gpu_linear_rank8_f16_bf16(
                gpu, output_gpu, features_gpu, weights_gpu, biases_gpu,
                TIME_ROWS, OUTPUTS), "cannot project rank-8 AdaLN");
    require(h3_gpu_submit(gpu), "cannot submit rank-8 command buffer");
    require(h3_gpu_tensor_read_f32(
                features_gpu, features_got, TIME_ROWS * RANK),
            "cannot read rank-8 features");
    require(h3_gpu_tensor_read_bf16(
                output_gpu, output_got, TIME_ROWS * OUTPUTS),
            "cannot read rank-8 output");
    float maximum_curve_error = 0.0f;
    float maximum_projection_error = 0.0f;
    for (unsigned index = 0; index < TIME_ROWS * RANK; index++)
        maximum_curve_error = fmaxf(maximum_curve_error,
            fabsf(features_got[index] - features_want[index]));
    for (unsigned row = 0; row < TIME_ROWS; row++)
        for (unsigned column = 0; column < OUTPUTS; column++) {
            float want = f16_to_f32(biases[column]);
            for (unsigned rank = 0; rank < RANK; rank++)
                want = fmaf(features_want[row * RANK + rank],
                    f16_to_f32(weights[column * RANK + rank]), want);
            float got = bf16_to_f32(output_got[row * OUTPUTS + column]);
            maximum_projection_error = fmaxf(maximum_projection_error,
                                              fabsf(got - want));
        }
    require(maximum_curve_error <= 2e-6f,
            "rank-8 curve interpolation exceeds tolerance");
    require(maximum_projection_error <= 0.02f,
            "rank-8 AdaLN projection exceeds tolerance");
    printf("rank-8 random timestep max abs %.7g, curve %.7g\n",
           maximum_projection_error, maximum_curve_error);
    h3_gpu_tensor_free(times_gpu); h3_gpu_tensor_free(table_gpu);
    h3_gpu_tensor_free(features_gpu); h3_gpu_tensor_free(weights_gpu);
    h3_gpu_tensor_free(biases_gpu); h3_gpu_tensor_free(output_gpu);
    free(table);
}

static void test_comfy_qkv_layout(h3_gpu *gpu) {
    enum { Q_ROWS = 1, HEADS = 2, HEAD_DIM = 128,
           INNER = HEADS * HEAD_DIM, ROPE_HALF = 48 };
    uint16_t qkv[INNER * 3], norm[HEAD_DIM];
    uint16_t rope_cos[Q_ROWS * ROPE_HALF], rope_sin[Q_ROWS * ROPE_HALF];
    for (unsigned index = 0; index < INNER; index++) {
        qkv[index] = f32_to_bf16(1.0f);
        qkv[INNER + index] = f32_to_bf16(2.0f);
        qkv[INNER * 2 + index] = f32_to_bf16(3.0f);
    }
    for (unsigned index = 0; index < HEAD_DIM; index++)
        norm[index] = f32_to_bf16(1.0f);
    for (unsigned index = 0; index < Q_ROWS * ROPE_HALF; index++) {
        rope_cos[index] = f32_to_bf16(1.0f);
        rope_sin[index] = f32_to_bf16(0.0f);
    }
    h3_gpu_tensor *qkv_gpu = h3_gpu_tensor_from_bf16(gpu, qkv, INNER * 3);
    h3_gpu_tensor *norm_gpu = h3_gpu_tensor_from_bf16(gpu, norm, HEAD_DIM);
    h3_gpu_tensor *cos_gpu = h3_gpu_tensor_from_bf16(
        gpu, rope_cos, Q_ROWS * ROPE_HALF);
    h3_gpu_tensor *sin_gpu = h3_gpu_tensor_from_bf16(
        gpu, rope_sin, Q_ROWS * ROPE_HALF);
    h3_gpu_tensor *query_gpu = h3_gpu_tensor_new_bf16(gpu, Q_ROWS * INNER);
    h3_gpu_tensor *key_gpu = h3_gpu_tensor_new_bf16(gpu, Q_ROWS * INNER);
    h3_gpu_tensor *value_gpu = h3_gpu_tensor_new_bf16(gpu, Q_ROWS * INNER);
    require(qkv_gpu && norm_gpu && cos_gpu && sin_gpu && query_gpu && key_gpu &&
            value_gpu, "QKV layout Metal allocation failed");
    require(h3_gpu_begin(gpu) && h3_gpu_qkv_rope_bf16(
                gpu, query_gpu, key_gpu, value_gpu, qkv_gpu, norm_gpu,
                norm_gpu, cos_gpu, sin_gpu, Q_ROWS, HEADS, HEAD_DIM, ROPE_HALF,
                1e-5f) && h3_gpu_submit(gpu),
            "cannot decode Comfy contiguous QKV layout");
    uint16_t values[Q_ROWS * INNER];
    require(h3_gpu_tensor_read_bf16(value_gpu, values, Q_ROWS * INNER),
            "cannot read Comfy QKV layout result");
    for (unsigned index = 0; index < Q_ROWS * INNER; index++)
        require(values[index] == f32_to_bf16(3.0f),
                "Comfy contiguous QKV value range was decoded as grouped");
    h3_gpu_tensor_free(qkv_gpu); h3_gpu_tensor_free(norm_gpu);
    h3_gpu_tensor_free(cos_gpu); h3_gpu_tensor_free(sin_gpu);
    h3_gpu_tensor_free(query_gpu); h3_gpu_tensor_free(key_gpu);
    h3_gpu_tensor_free(value_gpu);
}

static void test_qkv_head_major_sdpa_layout(h3_gpu *gpu) {
    enum { ROWS = 87, HEADS = 8, HEAD_DIM = 128, ROPE_HALF = 48 };
    size_t elements = (size_t)ROWS * HEADS * HEAD_DIM;
    size_t qkv_elements = elements * 3;
    size_t rope_elements = (size_t)ROWS * ROPE_HALF;
    uint16_t *qkv = malloc(qkv_elements * sizeof(*qkv));
    uint16_t *q_norm = malloc(HEAD_DIM * sizeof(*q_norm));
    uint16_t *k_norm = malloc(HEAD_DIM * sizeof(*k_norm));
    uint16_t *rope_cos = malloc(rope_elements * sizeof(*rope_cos));
    uint16_t *rope_sin = malloc(rope_elements * sizeof(*rope_sin));
    uint16_t *row_q = malloc(elements * sizeof(*row_q));
    uint16_t *row_k = malloc(elements * sizeof(*row_k));
    uint16_t *row_v = malloc(elements * sizeof(*row_v));
    uint16_t *head_q = malloc(elements * sizeof(*head_q));
    uint16_t *head_k = malloc(elements * sizeof(*head_k));
    uint16_t *head_v = malloc(elements * sizeof(*head_v));
    uint16_t *row_attention = malloc(elements * sizeof(*row_attention));
    uint16_t *head_attention = malloc(elements * sizeof(*head_attention));
    require(qkv && q_norm && k_norm && rope_cos && rope_sin && row_q && row_k &&
            row_v && head_q && head_k && head_v && row_attention &&
            head_attention, "QKV head-major fixture allocation failed");
    for (size_t index = 0; index < qkv_elements; index++)
        qkv[index] = f32_to_bf16(
            ((float)(int)(index % 113) - 56.0f) * 0.03125f +
            (float)(index % 17) * 0.001f);
    for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++) {
        q_norm[dimension] = f32_to_bf16(0.7f +
            (float)(dimension % 19) * 0.013f);
        k_norm[dimension] = f32_to_bf16(0.9f +
            (float)(dimension % 23) * 0.009f);
    }
    for (unsigned row = 0; row < ROWS; row++)
        for (unsigned dimension = 0; dimension < ROPE_HALF; dimension++) {
            size_t index = (size_t)row * ROPE_HALF + dimension;
            rope_cos[index] = f32_to_bf16(0.55f +
                (float)((row * 3 + dimension * 5) % 23) * 0.011f);
            rope_sin[index] = f32_to_bf16(-0.35f +
                (float)((row * 7 + dimension * 2) % 19) * 0.017f);
        }
    h3_gpu_tensor *qkv_gpu = h3_gpu_tensor_from_bf16(
        gpu, qkv, qkv_elements);
    h3_gpu_tensor *q_norm_gpu = h3_gpu_tensor_from_bf16(
        gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *k_norm_gpu = h3_gpu_tensor_from_bf16(
        gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *cos_gpu = h3_gpu_tensor_from_bf16(
        gpu, rope_cos, rope_elements);
    h3_gpu_tensor *sin_gpu = h3_gpu_tensor_from_bf16(
        gpu, rope_sin, rope_elements);
    h3_gpu_tensor *row_q_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *row_k_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *row_v_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *head_q_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *head_k_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *head_v_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *row_attention_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *head_attention_gpu = h3_gpu_tensor_new_bf16(gpu, elements);
    require(qkv_gpu && q_norm_gpu && k_norm_gpu && cos_gpu && sin_gpu &&
            row_q_gpu && row_k_gpu && row_v_gpu && head_q_gpu && head_k_gpu &&
            head_v_gpu && row_attention_gpu && head_attention_gpu,
            "QKV head-major tensor allocation failed");
    const char *saved_reference_value = getenv("H3_REFERENCE_SDPA");
    char *saved_reference = saved_reference_value ?
        strdup(saved_reference_value) : NULL;
    require(!saved_reference_value || saved_reference,
            "QKV head-major environment save failed");
    /* Force the tested cached paths, then leave all diagnostic switches clear. */
    unsetenv("H3_DISABLE_COOP_QKV");
    unsetenv("H3_DISABLE_CACHED_QKV");
    unsetenv("H3_DISABLE_CONVROT_HEAD_MAJOR");
    unsetenv("H3_DISABLE_HEAD_MAJOR_SDPA");
    unsetenv("H3_REFERENCE_SDPA");
    require(h3_gpu_begin(gpu) &&
            h3_gpu_qkv_rope_bf16(
                gpu, row_q_gpu, row_k_gpu, row_v_gpu, qkv_gpu, q_norm_gpu,
                k_norm_gpu, cos_gpu, sin_gpu, ROWS, HEADS, HEAD_DIM,
                ROPE_HALF, 1e-5f) && h3_gpu_submit(gpu),
            "row-major QKV/RoPE regression dispatch failed");
    require(h3_gpu_tensor_read_bf16(row_q_gpu, row_q, elements) &&
            h3_gpu_tensor_read_bf16(row_k_gpu, row_k, elements) &&
            h3_gpu_tensor_read_bf16(row_v_gpu, row_v, elements),
            "row-major QKV/RoPE regression read failed");
    require(h3_gpu_begin(gpu) && h3_gpu_sdpa_bf16(
                gpu, row_attention_gpu, row_q_gpu, row_k_gpu, row_v_gpu,
                ROWS, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)) &&
            h3_gpu_submit(gpu) && h3_gpu_tensor_read_bf16(
                row_attention_gpu, row_attention, elements),
            "row-major SDPA regression dispatch failed");
    require(h3_gpu_begin(gpu) &&
            h3_gpu_qkv_rope_bf16_for_sdpa(
                gpu, head_q_gpu, head_k_gpu, head_v_gpu, qkv_gpu, q_norm_gpu,
                k_norm_gpu, cos_gpu, sin_gpu, ROWS, HEADS, HEAD_DIM,
                ROPE_HALF, 1e-5f) && h3_gpu_submit(gpu),
            "head-major QKV/RoPE regression dispatch failed");
    require(h3_gpu_tensor_read_bf16(head_q_gpu, head_q, elements) &&
            h3_gpu_tensor_read_bf16(head_k_gpu, head_k, elements) &&
            h3_gpu_tensor_read_bf16(head_v_gpu, head_v, elements),
            "head-major QKV/RoPE regression read failed");
    for (unsigned row = 0; row < ROWS; row++)
        for (unsigned head = 0; head < HEADS; head++)
            for (unsigned dimension = 0; dimension < HEAD_DIM; dimension++) {
                size_t row_index = ((size_t)row * HEADS + head) * HEAD_DIM +
                                   dimension;
                size_t head_index = ((size_t)head * ROWS + row) * HEAD_DIM +
                                    dimension;
                require(head_q[head_index] == row_q[row_index] &&
                        head_k[head_index] == row_k[row_index] &&
                        head_v[head_index] == row_v[row_index],
                        "head-major QKV/RoPE layout differs from row-major");
            }
    require(h3_gpu_begin(gpu) &&
            h3_gpu_qkv_rope_bf16_for_sdpa(
                gpu, head_q_gpu, head_k_gpu, head_v_gpu, qkv_gpu, q_norm_gpu,
                k_norm_gpu, cos_gpu, sin_gpu, ROWS, HEADS, HEAD_DIM,
                ROPE_HALF, 1e-5f) && h3_gpu_sdpa_bf16(
                gpu, head_attention_gpu, head_q_gpu, head_k_gpu, head_v_gpu,
                ROWS, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)) &&
            h3_gpu_submit(gpu) && h3_gpu_tensor_read_bf16(
                head_attention_gpu, head_attention, elements),
            "head-major SDPA regression dispatch failed");
    require(!memcmp(row_attention, head_attention,
                    elements * sizeof(*row_attention)),
            "head-major SDPA output differs from row-major baseline");
    unsetenv("H3_DISABLE_COOP_QKV");
    unsetenv("H3_DISABLE_CACHED_QKV");
    unsetenv("H3_DISABLE_CONVROT_HEAD_MAJOR");
    unsetenv("H3_DISABLE_HEAD_MAJOR_SDPA");
    if (saved_reference) {
        setenv("H3_REFERENCE_SDPA", saved_reference, 1);
        free(saved_reference);
    } else {
        unsetenv("H3_REFERENCE_SDPA");
    }
    h3_gpu_tensor_free(qkv_gpu); h3_gpu_tensor_free(q_norm_gpu);
    h3_gpu_tensor_free(k_norm_gpu); h3_gpu_tensor_free(cos_gpu);
    h3_gpu_tensor_free(sin_gpu); h3_gpu_tensor_free(row_q_gpu);
    h3_gpu_tensor_free(row_k_gpu); h3_gpu_tensor_free(row_v_gpu);
    h3_gpu_tensor_free(head_q_gpu); h3_gpu_tensor_free(head_k_gpu);
    h3_gpu_tensor_free(head_v_gpu); h3_gpu_tensor_free(row_attention_gpu);
    h3_gpu_tensor_free(head_attention_gpu);
    free(qkv); free(q_norm); free(k_norm); free(rope_cos); free(rope_sin);
    free(row_q); free(row_k); free(row_v); free(head_q); free(head_k);
    free(head_v); free(row_attention); free(head_attention);
}

static void test_checkpoint_qkv(h3_gpu *gpu, const char *path) {
    enum { CHECKPOINT_INPUT = 5376, CHECKPOINT_OUTPUT = 21504,
           CHECKPOINT_ROWS = 128, CHECK_COLUMNS = 128 };
    char error[512];
    h3_weight_store *store = h3_weight_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL tests/test_convrot.c: %s\n", error);
        exit(1);
    }
    uint64_t weight_shape[] = {CHECKPOINT_OUTPUT, CHECKPOINT_INPUT};
    uint64_t scale_shape[] = {CHECKPOINT_OUTPUT, 1};
    h3_gpu_tensor *weight = h3_weight_load_i8(
        store, gpu, "blocks.0.attn.qkv_proj.weight", 2, weight_shape,
        error, sizeof(error));
    h3_gpu_tensor *weight_scales = h3_weight_load_f32(
        store, gpu, "blocks.0.attn.qkv_proj.weight_scale", 2, scale_shape,
        error, sizeof(error));
    require(weight && weight_scales, "cannot load checkpoint QKV tensors");

    size_t input_count = (size_t)CHECKPOINT_ROWS * CHECKPOINT_INPUT;
    size_t output_count = (size_t)CHECKPOINT_ROWS * CHECKPOINT_OUTPUT;
    uint16_t *input = malloc(input_count * sizeof(*input));
    uint16_t *rotated = malloc(input_count * sizeof(*rotated));
    int8_t *quantized = malloc(input_count);
    int8_t *weight_prefix = malloc((size_t)CHECK_COLUMNS * CHECKPOINT_INPUT);
    float *input_scales = malloc(CHECKPOINT_ROWS * sizeof(*input_scales));
    float *checkpoint_scales = malloc(CHECK_COLUMNS * sizeof(*checkpoint_scales));
    uint16_t *output = malloc(output_count * sizeof(*output));
    require(input && rotated && quantized && weight_prefix && input_scales &&
            checkpoint_scales && output, "checkpoint test allocation failed");
    for (size_t index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            ((float)(int32_t)(random_u32() % 4001) - 2000.0f) / 997.0f);

    h3_gpu_tensor *input_gpu = h3_gpu_tensor_from_bf16(gpu, input, input_count);
    h3_gpu_tensor *rotated_gpu = h3_gpu_tensor_new_bf16(gpu, input_count);
    h3_gpu_tensor *quantized_gpu = h3_gpu_tensor_new_i8(gpu, input_count);
    h3_gpu_tensor *input_scales_gpu = h3_gpu_tensor_new_f32(gpu, CHECKPOINT_ROWS);
    h3_gpu_tensor *output_gpu = h3_gpu_tensor_new_bf16(gpu, output_count);
    require(input_gpu && rotated_gpu && quantized_gpu && input_scales_gpu &&
            output_gpu, "checkpoint Metal allocation failed");
    require(h3_gpu_begin(gpu), "cannot begin checkpoint ConvRot command buffer");
    require(h3_gpu_linear_convrot_int8_bf16(
                gpu, output_gpu, rotated_gpu, quantized_gpu, input_scales_gpu,
                input_gpu, weight, weight_scales, CHECKPOINT_ROWS,
                CHECKPOINT_INPUT, CHECKPOINT_OUTPUT),
            "cannot project checkpoint QKV matrix");
    require(h3_gpu_submit(gpu), "cannot submit checkpoint ConvRot command buffer");
    require(h3_gpu_tensor_read_bf16(rotated_gpu, rotated, input_count) &&
            h3_gpu_tensor_read_i8(quantized_gpu, quantized, input_count) &&
            h3_gpu_tensor_read_i8(weight, weight_prefix,
                                  (size_t)CHECK_COLUMNS * CHECKPOINT_INPUT) &&
            h3_gpu_tensor_read_f32(weight_scales, checkpoint_scales,
                                   CHECK_COLUMNS) &&
            h3_gpu_tensor_read_f32(input_scales_gpu, input_scales,
                                   CHECKPOINT_ROWS) &&
            h3_gpu_tensor_read_bf16(output_gpu, output, output_count),
            "cannot read checkpoint ConvRot results");

    float maximum_error = 0.0f;
    float maximum_rounded_error = 0.0f;
    for (unsigned row = 0; row < CHECKPOINT_ROWS; row++)
        for (unsigned column = 0; column < CHECK_COLUMNS; column++) {
            int32_t sum = 0;
            for (unsigned inner = 0; inner < CHECKPOINT_INPUT; inner++)
                sum += (int32_t)quantized[(size_t)row * CHECKPOINT_INPUT + inner] *
                       (int32_t)weight_prefix[(size_t)column * CHECKPOINT_INPUT + inner];
            float want = (float)sum * input_scales[row] * checkpoint_scales[column];
            float got = bf16_to_f32(output[(size_t)row * CHECKPOINT_OUTPUT + column]);
            maximum_error = fmaxf(maximum_error, fabsf(got - want));
            float rounded_want = bf16_to_f32(f32_to_bf16(want));
            maximum_rounded_error = fmaxf(maximum_rounded_error,
                                          fabsf(got - rounded_want));
        }
    printf("checkpoint QKV mapped-weight max abs %.7g, rounded %.7g\n",
           maximum_error, maximum_rounded_error);
    require(maximum_rounded_error == 0.0f,
            "checkpoint QKV Metal output differs from BF16 reference");

    h3_gpu_tensor_free(input_gpu); h3_gpu_tensor_free(rotated_gpu);
    h3_gpu_tensor_free(quantized_gpu); h3_gpu_tensor_free(input_scales_gpu);
    h3_gpu_tensor_free(output_gpu); h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(weight_scales); h3_weight_store_free(store);
    free(input); free(rotated); free(quantized); free(weight_prefix);
    free(input_scales); free(checkpoint_scales); free(output);
}

static void test_checkpoint_rank8(h3_gpu *gpu, const char *path) {
    enum { CURVE_ROWS = 1025, OUTPUTS = 96768, TIME_ROWS = 5 };
    const float times[TIME_ROWS] = {0.0f, 1.0f, 0.5f, 0.12345f, 0.9997f};
    char error[512];
    h3_weight_store *store = h3_weight_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL tests/test_convrot.c: %s\n", error);
        exit(1);
    }
    uint64_t table_shape[] = {CURVE_ROWS, RANK};
    uint64_t weight_shape[] = {OUTPUTS, RANK};
    uint64_t bias_shape[] = {OUTPUTS};
    h3_gpu_tensor *table_gpu = h3_weight_load_f32(
        store, gpu, "adaln_t_table", 2, table_shape, error, sizeof(error));
    h3_gpu_tensor *weight_gpu = h3_weight_load_f16(
        store, gpu, "blocks.0.adaln_proj.linear.weight", 2, weight_shape,
        error, sizeof(error));
    h3_gpu_tensor *bias_gpu = h3_weight_load_f16(
        store, gpu, "blocks.0.adaln_proj.linear.bias", 1, bias_shape,
        error, sizeof(error));
    require(table_gpu && weight_gpu && bias_gpu,
            "cannot load checkpoint rank-8 tensors");
    h3_gpu_tensor *times_gpu = h3_gpu_tensor_from_f32(gpu, times, TIME_ROWS);
    h3_gpu_tensor *features_gpu = h3_gpu_tensor_new_f32(gpu, TIME_ROWS * RANK);
    h3_gpu_tensor *output_gpu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)TIME_ROWS * OUTPUTS);
    require(times_gpu && features_gpu && output_gpu,
            "checkpoint rank-8 Metal allocation failed");
    require(h3_gpu_begin(gpu) && h3_gpu_adaln_table_interpolate_f32(
                gpu, features_gpu, times_gpu, table_gpu, TIME_ROWS, CURVE_ROWS) &&
            h3_gpu_linear_rank8_f16_bf16(
                gpu, output_gpu, features_gpu, weight_gpu, bias_gpu,
                TIME_ROWS, OUTPUTS) && h3_gpu_submit(gpu),
            "cannot run checkpoint rank-8 projection");

    float features[TIME_ROWS * RANK];
    uint16_t *weights = malloc((size_t)OUTPUTS * RANK * sizeof(*weights));
    uint16_t *biases = malloc(OUTPUTS * sizeof(*biases));
    uint16_t *output = malloc((size_t)TIME_ROWS * OUTPUTS * sizeof(*output));
    require(weights && biases && output,
            "checkpoint rank-8 host allocation failed");
    require(h3_gpu_tensor_read_f32(features_gpu, features, TIME_ROWS * RANK) &&
            h3_gpu_tensor_read_bf16(output_gpu, output,
                                    (size_t)TIME_ROWS * OUTPUTS),
            "cannot read checkpoint rank-8 results");
    const h3_st_header *header = NULL;
    const h3_st_tensor *weight_st = h3_weight_find(
        store, "blocks.0.adaln_proj.linear.weight", &header);
    require(weight_st && header && h3_st_read_data(
                header, weight_st, weights,
                (size_t)OUTPUTS * RANK * sizeof(*weights), error,
                sizeof(error)), "cannot read checkpoint rank-8 weights");
    const h3_st_tensor *bias_st = h3_weight_find(
        store, "blocks.0.adaln_proj.linear.bias", &header);
    require(bias_st && header && h3_st_read_data(
                header, bias_st, biases, OUTPUTS * sizeof(*biases), error,
                sizeof(error)), "cannot read checkpoint rank-8 biases");
    float maximum_error = 0.0f;
    for (unsigned row = 0; row < TIME_ROWS; row++)
        for (unsigned column = 0; column < OUTPUTS; column++) {
            float want = f16_to_f32(biases[column]);
            for (unsigned rank = 0; rank < RANK; rank++)
                want = fmaf(features[row * RANK + rank],
                    f16_to_f32(weights[(size_t)column * RANK + rank]), want);
            float rounded = bf16_to_f32(f32_to_bf16(want));
            float got = bf16_to_f32(output[(size_t)row * OUTPUTS + column]);
            maximum_error = fmaxf(maximum_error, fabsf(got - rounded));
        }
    printf("checkpoint rank-8 BF16-boundary max abs %.7g\n", maximum_error);
    require(maximum_error == 0.0f,
            "checkpoint rank-8 Metal output differs from BF16 reference");

    h3_gpu_tensor_free(table_gpu); h3_gpu_tensor_free(weight_gpu);
    h3_gpu_tensor_free(bias_gpu); h3_gpu_tensor_free(times_gpu);
    h3_gpu_tensor_free(features_gpu); h3_gpu_tensor_free(output_gpu);
    h3_weight_store_free(store); free(weights); free(biases); free(output);
}

static void print_bf16_stats(const char *name, const h3_gpu_tensor *tensor,
                             size_t elements) {
    uint16_t *values = malloc(elements * sizeof(*values));
    require(values && h3_gpu_tensor_read_bf16(tensor, values, elements),
            "cannot read block-probe tensor");
    double sum = 0.0, squares = 0.0;
    float minimum = INFINITY, maximum = -INFINITY;
    for (size_t index = 0; index < elements; index++) {
        float value = bf16_to_f32(values[index]);
        minimum = fminf(minimum, value);
        maximum = fmaxf(maximum, value);
        sum += value;
        squares += (double)value * value;
    }
    printf("block %-20s min=% .7g max=% .7g mean=% .7g rms=% .7g\n",
           name, minimum, maximum, sum / (double)elements,
           sqrt(squares / (double)elements));
    free(values);
}

/* The CUDA reference can optionally write the BF16 value at every block
 * boundary.  Keep this outside the normal numerical smoke so a missing oracle
 * never turns a local unit test into an accidental pass/fail gate. */
static void compare_bf16_stage(const char *directory, const char *name,
                               const h3_gpu_tensor *tensor, size_t elements) {
    if (!directory || !*directory) return;
    size_t path_size = strlen(directory) + strlen(name) + 7;
    char *path = malloc(path_size);
    uint16_t *actual = malloc(elements * sizeof(*actual));
    uint16_t *expected = malloc(elements * sizeof(*expected));
    require(path && actual && expected, "cannot allocate stage comparison");
    snprintf(path, path_size, "%s/%s.bf16", directory, name);
    FILE *stream = fopen(path, "rb");
    require(stream && fread(expected, sizeof(*expected), elements, stream) == elements &&
            h3_gpu_tensor_read_bf16(tensor, actual, elements),
            "cannot read block stage oracle");
    fclose(stream);
    double absolute_sum = 0.0, reference_sum = 0.0;
    float maximum_error = 0.0f;
    size_t mismatches = 0;
    for (size_t index = 0; index < elements; index++) {
        float got = bf16_to_f32(actual[index]);
        float want = bf16_to_f32(expected[index]);
        float error = fabsf(got - want);
        maximum_error = fmaxf(maximum_error, error);
        absolute_sum += error;
        reference_sum += fabs((double)want);
        mismatches += actual[index] != expected[index];
    }
    printf("block stage %-14s mismatches=%zu/%zu max_abs=%.7g mean_abs=%.7g relative_l1=%.7g\n",
           name, mismatches, elements, maximum_error,
           absolute_sum / (double)elements, absolute_sum / reference_sum);
    if (getenv("H3_CONVROT_BLOCK_DIAG") && mismatches) {
        size_t shown = 0;
        for (size_t index = 0; index < elements && shown < 8; index++) {
            if (actual[index] == expected[index]) continue;
            printf("block stage %s first_diff[%zu]=%.7g expected=%.7g\n",
                   name, index, bf16_to_f32(actual[index]),
                   bf16_to_f32(expected[index]));
            shown++;
        }
    }
    free(path); free(actual); free(expected);
}

static void compare_f32_stage(const char *directory, const char *name,
                              const h3_gpu_tensor *tensor, size_t elements) {
    if (!directory || !*directory) return;
    size_t path_size = strlen(directory) + strlen(name) + 6;
    char *path = malloc(path_size);
    float *actual = malloc(elements * sizeof(*actual));
    float *expected = malloc(elements * sizeof(*expected));
    require(path && actual && expected, "cannot allocate F32 stage comparison");
    snprintf(path, path_size, "%s/%s.f32", directory, name);
    FILE *stream = fopen(path, "rb");
    require(stream && fread(expected, sizeof(*expected), elements, stream) == elements &&
            h3_gpu_tensor_read_f32(tensor, actual, elements),
            "cannot read F32 block stage oracle");
    fclose(stream);
    double absolute_sum = 0.0, reference_sum = 0.0;
    float maximum_error = 0.0f;
    size_t exact = 0;
    for (size_t index = 0; index < elements; index++) {
        float error = fabsf(actual[index] - expected[index]);
        maximum_error = fmaxf(maximum_error, error);
        absolute_sum += error;
        reference_sum += fabs((double)expected[index]);
        exact += actual[index] == expected[index];
    }
    printf("block stage %-14s exact=%zu/%zu max_abs=%.7g mean_abs=%.7g relative_l1=%.7g\n",
           name, exact, elements, maximum_error, absolute_sum / (double)elements,
           absolute_sum / reference_sum);
    if (getenv("H3_CONVROT_BLOCK_DIAG")) {
        size_t shown = 0;
        for (size_t index = 0; index < elements && shown < 8; index++) {
            if (actual[index] == expected[index]) continue;
            printf("block stage %s first_diff[%zu]=%.7g expected=%.7g\n",
                   name, index, actual[index], expected[index]);
            shown++;
        }
    }
    free(path); free(actual); free(expected);
}

static h3_gpu_tensor *load_bf16_stage_oracle(h3_gpu *gpu,
                                              const char *directory,
                                              const char *name,
                                              size_t elements) {
    size_t path_size = strlen(directory) + strlen(name) + 7;
    char *path = malloc(path_size);
    uint16_t *values = malloc(elements * sizeof(*values));
    require(path && values, "cannot allocate stage oracle");
    snprintf(path, path_size, "%s/%s.bf16", directory, name);
    FILE *stream = fopen(path, "rb");
    require(stream && fread(values, sizeof(*values), elements, stream) == elements,
            "cannot read stage oracle");
    fclose(stream);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, elements);
    free(path); free(values);
    require(tensor != NULL, "cannot upload stage oracle");
    return tensor;
}

static h3_gpu_tensor *checkpoint_matrix(h3_weight_store *store, h3_gpu *gpu,
                                        const char *stem, uint64_t rows,
                                        uint64_t columns, int scales,
                                        char *error, size_t error_size) {
    char name[160];
    snprintf(name, sizeof(name), "blocks.0.%s.%s", stem,
             scales ? "weight_scale" : "weight");
    uint64_t shape[] = {rows, scales ? 1 : columns};
    return scales ? h3_weight_load_f32(
        store, gpu, name, 2, shape, error, error_size) : h3_weight_load_i8(
        store, gpu, name, 2, shape, error, error_size);
}

static void test_checkpoint_block(h3_gpu *gpu, const char *checkpoint,
                                  const char *input_path,
                                  const char *expected_path) {
    enum { B_ROWS = 128, HIDDEN = 5376, HEADS = 56, HEAD_DIM = 128,
           INNER = HEADS * HEAD_DIM, FFN = 14336, ROPE_HALF = 48,
           MODULATION = 3 * 6 * HIDDEN };
    char error[512];
    h3_weight_store *store = h3_weight_store_open(
        checkpoint, error, sizeof(error));
    require(store != NULL, "cannot open checkpoint for block probe");
    size_t hidden_count = (size_t)B_ROWS * HIDDEN;
    uint16_t *input = malloc(hidden_count * sizeof(*input));
    FILE *stream = fopen(input_path, "rb");
    require(input && stream &&
            fread(input, sizeof(*input), hidden_count, stream) == hidden_count,
            "cannot read deterministic block input");
    fclose(stream);
    h3_gpu_tensor *hidden = h3_gpu_tensor_from_bf16(gpu, input, hidden_count);
    free(input);

    uint64_t hidden_shape[] = {HIDDEN};
    uint64_t head_shape[] = {HEAD_DIM};
    h3_gpu_tensor *norm1 = h3_weight_load_bf16(
        store, gpu, "blocks.0.norm1.weight", 1, hidden_shape, error,
        sizeof(error));
    h3_gpu_tensor *norm2 = h3_weight_load_bf16(
        store, gpu, "blocks.0.norm2.weight", 1, hidden_shape, error,
        sizeof(error));
    h3_gpu_tensor *q_norm = h3_weight_load_bf16(
        store, gpu, "blocks.0.attn.q_norm.weight", 1, head_shape, error,
        sizeof(error));
    h3_gpu_tensor *k_norm = h3_weight_load_bf16(
        store, gpu, "blocks.0.attn.k_norm.weight", 1, head_shape, error,
        sizeof(error));
    h3_gpu_tensor *qkv_w = checkpoint_matrix(
        store, gpu, "attn.qkv_proj", INNER * 3, HIDDEN, 0, error,
        sizeof(error));
    h3_gpu_tensor *qkv_s = checkpoint_matrix(
        store, gpu, "attn.qkv_proj", INNER * 3, HIDDEN, 1, error,
        sizeof(error));
    h3_gpu_tensor *out_w = checkpoint_matrix(
        store, gpu, "attn.out_proj", HIDDEN, INNER, 0, error, sizeof(error));
    h3_gpu_tensor *out_s = checkpoint_matrix(
        store, gpu, "attn.out_proj", HIDDEN, INNER, 1, error, sizeof(error));
    h3_gpu_tensor *fc1_w = checkpoint_matrix(
        store, gpu, "mlp.fc1", FFN * 2, HIDDEN, 0, error, sizeof(error));
    h3_gpu_tensor *fc1_s = checkpoint_matrix(
        store, gpu, "mlp.fc1", FFN * 2, HIDDEN, 1, error, sizeof(error));
    h3_gpu_tensor *fc2_w = checkpoint_matrix(
        store, gpu, "mlp.fc2", HIDDEN, FFN, 0, error, sizeof(error));
    h3_gpu_tensor *fc2_s = checkpoint_matrix(
        store, gpu, "mlp.fc2", HIDDEN, FFN, 1, error, sizeof(error));
    require(hidden && norm1 && norm2 && q_norm && k_norm && qkv_w && qkv_s &&
            out_w && out_s && fc1_w && fc1_s && fc2_w && fc2_s,
            "cannot load deterministic block weights");

    float time = 0.5f;
    uint64_t table_shape[] = {1025, 8};
    uint64_t modulation_weight_shape[] = {MODULATION, 8};
    uint64_t modulation_bias_shape[] = {MODULATION};
    h3_gpu_tensor *time_gpu = h3_gpu_tensor_from_f32(gpu, &time, 1);
    h3_gpu_tensor *table = h3_weight_load_f32(
        store, gpu, "adaln_t_table", 2, table_shape, error, sizeof(error));
    h3_gpu_tensor *features = h3_gpu_tensor_new_f32(gpu, 8);
    h3_gpu_tensor *mod_w = h3_weight_load_f16(
        store, gpu, "blocks.0.adaln_proj.linear.weight", 2,
        modulation_weight_shape, error, sizeof(error));
    h3_gpu_tensor *mod_b = h3_weight_load_f16(
        store, gpu, "blocks.0.adaln_proj.linear.bias", 1,
        modulation_bias_shape, error, sizeof(error));
    h3_gpu_tensor *modulation = h3_gpu_tensor_new_bf16(gpu, MODULATION);
    uint32_t zero_rows[B_ROWS] = {0};
    h3_gpu_tensor *row_map = h3_gpu_tensor_from_u32(gpu, zero_rows, B_ROWS);
    uint16_t cosine[B_ROWS * ROPE_HALF], sine[B_ROWS * ROPE_HALF];
    for (unsigned index = 0; index < B_ROWS * ROPE_HALF; index++) {
        cosine[index] = f32_to_bf16(1.0f);
        sine[index] = f32_to_bf16(0.0f);
    }
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_bf16(
        gpu, cosine, B_ROWS * ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_bf16(
        gpu, sine, B_ROWS * ROPE_HALF);
    h3_gpu_tensor *mod_attention = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(
        gpu, (size_t)B_ROWS * INNER * 3);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, (size_t)B_ROWS * INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, (size_t)B_ROWS * INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, (size_t)B_ROWS * INNER);
    h3_gpu_tensor *attention_heads = h3_gpu_tensor_new_bf16(
        gpu, (size_t)B_ROWS * INNER);
    h3_gpu_tensor *oracle_attention_heads = NULL;
    const char *oracle_directory = getenv("H3_CONVROT_BLOCK_INTERMEDIATES");
    if (getenv("H3_CONVROT_USE_ORACLE_ATTENTION_HEADS") &&
        oracle_directory && *oracle_directory)
        oracle_attention_heads = load_bf16_stage_oracle(
            gpu, oracle_directory, "attention_heads", (size_t)B_ROWS * INNER);
    h3_gpu_tensor *oracle_query = NULL;
    h3_gpu_tensor *oracle_key = NULL;
    h3_gpu_tensor *oracle_value = NULL;
    if (getenv("H3_CONVROT_USE_ORACLE_QKV") && oracle_directory &&
        *oracle_directory) {
        oracle_query = load_bf16_stage_oracle(
            gpu, oracle_directory, "query", (size_t)B_ROWS * INNER);
        oracle_key = load_bf16_stage_oracle(
            gpu, oracle_directory, "key", (size_t)B_ROWS * INNER);
        oracle_value = load_bf16_stage_oracle(
            gpu, oracle_directory, "value", (size_t)B_ROWS * INNER);
    }
    int use_f32_sdpa = getenv("H3_F32_SDPA") != NULL;
    uint32_t head_elements = B_ROWS * INNER;
    h3_gpu_tensor *query_f32 = use_f32_sdpa ?
        h3_gpu_tensor_new_f32(gpu, head_elements) : NULL;
    h3_gpu_tensor *key_f32 = use_f32_sdpa ?
        h3_gpu_tensor_new_f32(gpu, head_elements) : NULL;
    h3_gpu_tensor *value_f32 = use_f32_sdpa ?
        h3_gpu_tensor_new_f32(gpu, head_elements) : NULL;
    h3_gpu_tensor *heads_f32 = use_f32_sdpa ?
        h3_gpu_tensor_new_f32(gpu, head_elements) : NULL;
    int use_reference_lse = getenv("H3_CONVROT_REFERENCE_LSE") != NULL;
    h3_gpu_tensor *attention_lse = use_reference_lse ?
        h3_gpu_tensor_new_f32(gpu, B_ROWS * HEADS) : NULL;
    h3_gpu_tensor *attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *after_attention = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *mod_mlp = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)B_ROWS * FFN * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)B_ROWS * FFN);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *rotated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)B_ROWS * FFN);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(
        gpu, (size_t)B_ROWS * FFN);
    h3_gpu_tensor *activation_scales = h3_gpu_tensor_new_f32(gpu, B_ROWS);
    int use_dequantized_out = getenv("H3_CONVROT_DEQUANTIZED_OUT") != NULL;
    h3_gpu_tensor *out_bf16 = use_dequantized_out ?
        h3_gpu_tensor_new_bf16(gpu, (size_t)HIDDEN * INNER) : NULL;
    require(time_gpu && table && features && mod_w && mod_b && modulation &&
            row_map && rope_cos && rope_sin && mod_attention && qkv && query &&
            key && value && attention_heads && attention_output &&
            after_attention && mod_mlp && fc1 && activated && mlp_output &&
            output && rotated && quantized && activation_scales &&
            (!use_dequantized_out || out_bf16) &&
            (!use_f32_sdpa || (query_f32 && key_f32 && value_f32 && heads_f32)) &&
            (!use_reference_lse || attention_lse),
            "cannot allocate deterministic block tensors");

    require(h3_gpu_begin(gpu) && h3_gpu_adaln_table_interpolate_f32(
                gpu, features, time_gpu, table, 1, 1025) &&
            (!use_dequantized_out || h3_gpu_dequantize_rows_i8_bf16(
                gpu, out_bf16, out_w, out_s, HIDDEN, INNER)) &&
            h3_gpu_linear_rank8_f16_bf16(
                gpu, modulation, features, mod_w, mod_b, 1, MODULATION) &&
            h3_gpu_adaln_bf16(
                gpu, mod_attention, hidden, norm1, modulation, row_map,
                B_ROWS, HIDDEN, 6, 0, 1, 1e-5f) &&
            h3_gpu_linear_convrot_int8_bf16(
                gpu, qkv, rotated, quantized, activation_scales,
                mod_attention, qkv_w, qkv_s, B_ROWS, HIDDEN, INNER * 3) &&
            h3_gpu_qkv_rope_bf16(
                gpu, query, key, value, qkv, q_norm, k_norm, rope_cos,
                rope_sin, B_ROWS, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f) &&
            (use_reference_lse ?
                h3_gpu_sdpa_reference_bf16_lse(
                    gpu, attention_heads, attention_lse,
                    oracle_query ? oracle_query : query,
                    oracle_key ? oracle_key : key,
                    oracle_value ? oracle_value : value, B_ROWS, HEADS,
                    HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)) :
             use_f32_sdpa ?
                (h3_gpu_cast_bf16_to_f32(
                    gpu, query_f32, oracle_query ? oracle_query : query,
                    head_elements) &&
                 h3_gpu_cast_bf16_to_f32(
                    gpu, key_f32, oracle_key ? oracle_key : key,
                    head_elements) &&
                 h3_gpu_cast_bf16_to_f32(
                    gpu, value_f32, oracle_value ? oracle_value : value,
                    head_elements) &&
                 h3_gpu_sdpa_f32(
                    gpu, heads_f32, query_f32, key_f32, value_f32, B_ROWS,
                    HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)) &&
                 h3_gpu_cast_f32_to_bf16(
                    gpu, attention_heads, heads_f32, head_elements)) :
                h3_gpu_sdpa_bf16(
                    gpu, attention_heads, oracle_query ? oracle_query : query,
                    oracle_key ? oracle_key : key,
                    oracle_value ? oracle_value : value, B_ROWS, HEADS,
                    HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM))) &&
            (use_dequantized_out ?
             (h3_gpu_linear_convrot_int8_bf16(
                gpu, attention_output, rotated, quantized, activation_scales,
                oracle_attention_heads ? oracle_attention_heads : attention_heads,
                out_w, out_s, B_ROWS, INNER, HIDDEN) &&
              h3_gpu_linear_bf16(
                gpu, attention_output, rotated, out_bf16, NULL, B_ROWS,
                INNER, HIDDEN)) :
             h3_gpu_linear_convrot_int8_bf16(
                gpu, attention_output, rotated, quantized, activation_scales,
                oracle_attention_heads ? oracle_attention_heads : attention_heads,
                out_w, out_s, B_ROWS, INNER, HIDDEN)) &&
            h3_gpu_gate_adaln_bf16(
                gpu, after_attention, mod_mlp, hidden, attention_output,
                norm2, modulation, modulation, row_map, B_ROWS, HIDDEN, 6,
                2, 3, 4, 1e-5f) &&
            h3_gpu_linear_convrot_int8_bf16(
                gpu, fc1, rotated, quantized, activation_scales, mod_mlp,
                fc1_w, fc1_s, B_ROWS, HIDDEN, FFN * 2) &&
            h3_gpu_swiglu_bf16(gpu, activated, fc1, B_ROWS, FFN) &&
            h3_gpu_linear_convrot_int8_bf16(
                gpu, mlp_output, rotated, quantized, activation_scales,
                activated, fc2_w, fc2_s, B_ROWS, FFN, HIDDEN) &&
            h3_gpu_gate_bf16(
                gpu, output, after_attention, mlp_output, modulation, row_map,
                B_ROWS, HIDDEN, 6, 5) && h3_gpu_submit(gpu),
            "deterministic ConvRot block dispatch failed");

    print_bf16_stats("input", hidden, hidden_count);
    print_bf16_stats("modulation", modulation, MODULATION);
    print_bf16_stats("attention_adaln", mod_attention, hidden_count);
    print_bf16_stats("qkv", qkv, (size_t)B_ROWS * INNER * 3);
    print_bf16_stats("attention_output", attention_output, hidden_count);
    print_bf16_stats("attention_residual", after_attention, hidden_count);
    print_bf16_stats("mlp_adaln", mod_mlp, hidden_count);
    print_bf16_stats("swiglu", activated, (size_t)B_ROWS * FFN);
    print_bf16_stats("mlp_output", mlp_output, hidden_count);
    print_bf16_stats("output", output, hidden_count);
    const char *stage_directory = getenv("H3_CONVROT_BLOCK_INTERMEDIATES");
    if (stage_directory && *stage_directory) {
        compare_bf16_stage(stage_directory, "attention_adaln", mod_attention,
                           hidden_count);
        compare_bf16_stage(stage_directory, "qkv", qkv,
                           (size_t)B_ROWS * INNER * 3);
        compare_bf16_stage(stage_directory, "query", query,
                           (size_t)B_ROWS * INNER);
        compare_bf16_stage(stage_directory, "key", key,
                           (size_t)B_ROWS * INNER);
        compare_bf16_stage(stage_directory, "value", value,
                           (size_t)B_ROWS * INNER);
        if (use_reference_lse)
            compare_f32_stage(stage_directory, "attention_lse_flash",
                              attention_lse, B_ROWS * HEADS);
        compare_bf16_stage(stage_directory, "attention_heads", attention_heads,
                           (size_t)B_ROWS * INNER);
        compare_bf16_stage(stage_directory, "attention_output", attention_output,
                           hidden_count);
        compare_bf16_stage(stage_directory, "attention_residual", after_attention,
                           hidden_count);
        compare_bf16_stage(stage_directory, "mlp_adaln", mod_mlp, hidden_count);
        compare_bf16_stage(stage_directory, "fc1", fc1,
                           (size_t)B_ROWS * FFN * 2);
        compare_bf16_stage(stage_directory, "swiglu", activated,
                           (size_t)B_ROWS * FFN);
        compare_bf16_stage(stage_directory, "mlp_output", mlp_output, hidden_count);
        compare_bf16_stage(stage_directory, "output", output, hidden_count);
    }
    if (expected_path && *expected_path) {
        uint16_t *actual = malloc(hidden_count * sizeof(*actual));
        uint16_t *expected = malloc(hidden_count * sizeof(*expected));
        FILE *expected_stream = fopen(expected_path, "rb");
        require(actual && expected && expected_stream &&
                fread(expected, sizeof(*expected), hidden_count,
                      expected_stream) == hidden_count &&
                h3_gpu_tensor_read_bf16(output, actual, hidden_count),
                "cannot compare deterministic block output");
        fclose(expected_stream);
        double absolute_sum = 0.0, reference_sum = 0.0;
        float maximum_error = 0.0f;
        for (size_t index = 0; index < hidden_count; index++) {
            float got = bf16_to_f32(actual[index]);
            float want = bf16_to_f32(expected[index]);
            maximum_error = fmaxf(maximum_error, fabsf(got - want));
            absolute_sum += fabs((double)got - want);
            reference_sum += fabs((double)want);
        }
        printf("block output comparison max_abs=%.7g mean_abs=%.7g relative_l1=%.7g\n",
               maximum_error, absolute_sum / (double)hidden_count,
               absolute_sum / reference_sum);
        require(absolute_sum / reference_sum <= 1e-3,
                "native block-0 output diverges from the captured Comfy oracle");
        free(actual); free(expected);
    }

#define FREE_BLOCK_TENSOR(name) h3_gpu_tensor_free(name)
    FREE_BLOCK_TENSOR(hidden); FREE_BLOCK_TENSOR(norm1); FREE_BLOCK_TENSOR(norm2);
    FREE_BLOCK_TENSOR(q_norm); FREE_BLOCK_TENSOR(k_norm);
    FREE_BLOCK_TENSOR(qkv_w); FREE_BLOCK_TENSOR(qkv_s);
    FREE_BLOCK_TENSOR(out_w); FREE_BLOCK_TENSOR(out_s);
    FREE_BLOCK_TENSOR(out_bf16);
    FREE_BLOCK_TENSOR(fc1_w); FREE_BLOCK_TENSOR(fc1_s);
    FREE_BLOCK_TENSOR(fc2_w); FREE_BLOCK_TENSOR(fc2_s);
    FREE_BLOCK_TENSOR(time_gpu); FREE_BLOCK_TENSOR(table);
    FREE_BLOCK_TENSOR(features); FREE_BLOCK_TENSOR(mod_w);
    FREE_BLOCK_TENSOR(mod_b); FREE_BLOCK_TENSOR(modulation);
    FREE_BLOCK_TENSOR(row_map); FREE_BLOCK_TENSOR(rope_cos);
    FREE_BLOCK_TENSOR(rope_sin); FREE_BLOCK_TENSOR(mod_attention);
    FREE_BLOCK_TENSOR(qkv); FREE_BLOCK_TENSOR(query); FREE_BLOCK_TENSOR(key);
    FREE_BLOCK_TENSOR(value); FREE_BLOCK_TENSOR(attention_heads);
    FREE_BLOCK_TENSOR(oracle_attention_heads);
    FREE_BLOCK_TENSOR(oracle_query); FREE_BLOCK_TENSOR(oracle_key);
    FREE_BLOCK_TENSOR(oracle_value);
    FREE_BLOCK_TENSOR(query_f32); FREE_BLOCK_TENSOR(key_f32);
    FREE_BLOCK_TENSOR(value_f32); FREE_BLOCK_TENSOR(heads_f32);
    FREE_BLOCK_TENSOR(attention_lse);
    FREE_BLOCK_TENSOR(attention_output); FREE_BLOCK_TENSOR(after_attention);
    FREE_BLOCK_TENSOR(mod_mlp); FREE_BLOCK_TENSOR(fc1);
    FREE_BLOCK_TENSOR(activated); FREE_BLOCK_TENSOR(mlp_output);
    FREE_BLOCK_TENSOR(output); FREE_BLOCK_TENSOR(rotated);
    FREE_BLOCK_TENSOR(quantized); FREE_BLOCK_TENSOR(activation_scales);
#undef FREE_BLOCK_TENSOR
    h3_weight_store_free(store);
}

static void test_adapter_qkv_layout(h3_gpu *gpu, int grouped) {
    enum { ROWS = 3, WIDTH = 256, COMPONENT = 1, ELEMENTS = ROWS * WIDTH };
    uint16_t output[ROWS * WIDTH * 3], branch[ELEMENTS], expected[ROWS * WIDTH * 3];
    for (unsigned index = 0; index < ROWS * WIDTH * 3; index++)
        output[index] = f32_to_bf16((float)((int)(index % 19) - 9) * 0.125f);
    for (unsigned index = 0; index < ELEMENTS; index++)
        branch[index] = f32_to_bf16((float)((int)(index % 13) - 6) * 0.0625f);
    memcpy(expected, output, sizeof(expected));
    for (unsigned index = 0; index < ELEMENTS; index++) {
        unsigned row = index / WIDTH, column = index % WIDTH;
        unsigned target_column = grouped ?
            ((column / 128) * 3 + COMPONENT) * 128 + column % 128 :
            COMPONENT * WIDTH + column;
        unsigned target = row * WIDTH * 3 + target_column;
        expected[target] = f32_to_bf16(bf16_to_f32(expected[target]) +
                                       bf16_to_f32(branch[index]));
    }
    h3_gpu_tensor *out = h3_gpu_tensor_from_bf16(gpu, output,
                                                    ROWS * WIDTH * 3);
    h3_gpu_tensor *delta = h3_gpu_tensor_from_bf16(gpu, branch, ELEMENTS);
    require(out && delta, "adapter QKV tensor allocation");
    require(h3_gpu_begin(gpu), "begin adapter QKV layout test");
    require(h3_gpu_add_qkv_component_bf16(gpu, out, delta, ROWS, WIDTH,
                                           COMPONENT, grouped),
            "adapter QKV layout dispatch");
    require(h3_gpu_submit(gpu), "submit adapter QKV layout test");
    require(h3_gpu_tensor_read_bf16(out, output, ROWS * WIDTH * 3),
            "read adapter QKV layout output");
    require(!memcmp(output, expected, sizeof(expected)),
            grouped ? "grouped QKV adapter layout mismatch" :
                      "contiguous QKV adapter layout mismatch");
    h3_gpu_tensor_free(out); h3_gpu_tensor_free(delta);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_convrot.c: %s\n", error);
        return 1;
    }
    require(h3_gpu_has_int8_mlp(gpu),
            "this numerical test requires Apple INT8 tensor support");
    test_dense_hadamard_reference();
    test_convrot(gpu, FULL_ROWS);
    test_convrot(gpu, SHORT_ROWS);
    test_rank8(gpu);
    test_comfy_qkv_layout(gpu);
    test_qkv_head_major_sdpa_layout(gpu);
    test_adapter_qkv_layout(gpu, 0);
    test_adapter_qkv_layout(gpu, 1);
    const char *checkpoint = getenv("H3_CONVROT_CHECKPOINT");
    if (checkpoint && *checkpoint) {
        test_checkpoint_qkv(gpu, checkpoint);
        test_checkpoint_rank8(gpu, checkpoint);
        const char *block_input = getenv("H3_CONVROT_BLOCK_INPUT");
        if (block_input && *block_input)
            test_checkpoint_block(
                gpu, checkpoint, block_input,
                getenv("H3_CONVROT_BLOCK_EXPECTED"));
    }
    h3_gpu_free(gpu);
    puts("ConvRot Metal/reference tests passed");
    return 0;
}
