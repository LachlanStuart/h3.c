#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *name;
    uint32_t input_dim;
    uint32_t output_dim;
    int qkv;
    int grouped;
} lora_shape;

static const lora_shape shapes[] = {
    {"qkv", 5376, 7168, 1, 0},
    {"qkv-grouped", 5376, 7168, 1, 1},
    {"attention-out", 7168, 5376, 0, 0},
    {"fc1", 5376, 28672, 0, 0},
    {"fc2", 14336, 5376, 0, 0},
};

typedef enum {
    VARIANT_BASELINE,
    VARIANT_DIRECT,
    VARIANT_MPS
} lora_variant;

static void die(const char *message) {
    fprintf(stderr, "h3_lora_bench: %s\n", message);
    exit(1);
}

static double now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
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

static uint16_t *make_values(size_t elements, uint32_t seed) {
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values) die("out of memory allocating benchmark data");
    uint32_t state = seed;
    for (size_t index = 0; index < elements; index++) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        float value = ((float)(state >> 8) / 16777216.0f - 0.5f) * 0.2f;
        values[index] = f32_to_bf16(value);
    }
    return values;
}

static int run_variant(h3_gpu *gpu, lora_variant variant,
                        h3_gpu_tensor *base,
                        h3_gpu_tensor *rank, h3_gpu_tensor *delta,
                        h3_gpu_tensor *input, h3_gpu_tensor *down,
                        h3_gpu_tensor *up, uint32_t rows,
                        const lora_shape *shape, float scale) {
    size_t branch_elements = (size_t)rows * shape->output_dim;
    if (variant == VARIANT_MPS)
        return h3_gpu_begin(gpu) &&
            h3_gpu_lora_bf16(
                gpu, base, input, down, up, scale, rows, shape->input_dim,
                128, shape->output_dim, shape->qkv ? 0 : -1,
                shape->grouped) &&
            h3_gpu_submit(gpu);
    int ok = h3_gpu_begin(gpu) &&
        h3_gpu_linear_bf16(gpu, rank, input, down, NULL, rows,
                           shape->input_dim, 128) &&
        (variant == VARIANT_BASELINE ?
            (h3_gpu_linear_bf16(gpu, delta, rank, up, NULL, rows, 128,
                                shape->output_dim) &&
             h3_gpu_scale_bf16(gpu, delta, delta, scale,
                               (uint32_t)branch_elements) &&
             (shape->qkv ?
                h3_gpu_add_qkv_component_bf16(
                    gpu, base, delta, rows, shape->output_dim, 0,
                    shape->grouped) :
                h3_gpu_add_bf16(gpu, base, base, delta,
                                (uint32_t)branch_elements))) :
            (shape->qkv ?
                h3_gpu_linear_add_qkv_component_bf16(
                    gpu, base, rank, up, scale, rows, 128, shape->output_dim,
                    0, shape->grouped) :
                h3_gpu_linear_add_bf16(
                    gpu, base, rank, up, scale, rows, 128,
                    shape->output_dim))) &&
        h3_gpu_submit(gpu);
    return ok;
}

static void compare_output(h3_gpu_tensor *actual, const uint16_t *reference,
                           size_t elements, const char *variant,
                           const char *shape) {
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values) die("out of memory comparing benchmark output");
    if (!h3_gpu_tensor_read_bf16(actual, values, elements))
        die("cannot read benchmark output");
    float max_abs = 0.0f;
    double sum_squared = 0.0;
    for (size_t index = 0; index < elements; index++) {
        float expected = bf16_to_f32(reference[index]);
        float actual_value = bf16_to_f32(values[index]);
        float difference = fabsf(actual_value - expected);
        float allowed = 0.02f + 0.02f * fabsf(expected);
        if (difference > max_abs) max_abs = difference;
        sum_squared += (double)difference * difference;
        if (difference > allowed) {
            fprintf(stderr, "h3_lora_bench: %s %s exceeds tolerance at "
                    "%zu: %.7g vs %.7g (allowed %.7g)\n", shape, variant,
                    index, actual_value, expected, allowed);
            free(values);
            exit(1);
        }
    }
    printf("parity %-8s %-14s max=%.6g rms=%.6g tol=0.02+2%%\n", variant,
           shape, max_abs, (float)sqrt(sum_squared / (double)elements));
    free(values);
}

static void benchmark_variant(h3_gpu *gpu, lora_variant variant,
                              h3_gpu_tensor *base, h3_gpu_tensor *rank,
                              h3_gpu_tensor *delta, h3_gpu_tensor *input,
                              h3_gpu_tensor *down, h3_gpu_tensor *up,
                              const uint16_t *base_values, size_t output_elements,
                              uint32_t rows, const lora_shape *shape,
                              uint32_t warmup, uint32_t repeats) {
    const char *name = variant == VARIANT_BASELINE ? "baseline" :
        variant == VARIANT_DIRECT ? "direct" : "mps";
    char error[512];
    for (uint32_t index = 0; index < warmup; index++) {
        if (!h3_gpu_tensor_write_bf16(base, base_values, output_elements) ||
            !run_variant(gpu, variant, base, rank, delta, input, down, up,
                         rows, shape, 0.0625f)) {
            snprintf(error, sizeof(error), "%s warmup failed: %s", name,
                     h3_gpu_error(gpu));
            die(error);
        }
    }
    h3_gpu_stats before, after;
    if (!h3_gpu_get_stats(gpu, &before)) die("cannot read initial statistics");
    double elapsed = 0.0;
    for (uint32_t index = 0; index < repeats; index++) {
        if (!h3_gpu_tensor_write_bf16(base, base_values, output_elements))
            die("cannot reset benchmark output");
        double started = now();
        if (!run_variant(gpu, variant, base, rank, delta, input, down, up,
                         rows, shape, 0.0625f)) {
            snprintf(error, sizeof(error), "%s benchmark failed: %s", name,
                     h3_gpu_error(gpu));
            die(error);
        }
        elapsed += now() - started;
    }
    if (!h3_gpu_get_stats(gpu, &after)) die("cannot read final statistics");
    printf("bench %-8s %-14s rows=%u wall=%.4fms gpu=%.4fms direct=%llu "
           "mps-linear=%llu\n", name, shape->name, rows,
           elapsed * 1000.0 / repeats,
           (after.gpu_seconds - before.gpu_seconds) * 1000.0 / repeats,
           (unsigned long long)(after.direct_dispatches -
                                before.direct_dispatches) / repeats,
           (unsigned long long)(after.mps_linear_dispatches -
                                before.mps_linear_dispatches) / repeats);
}

static void benchmark_shape(h3_gpu *gpu, uint32_t rows,
                            const lora_shape *shape, uint32_t warmup,
                            uint32_t repeats) {
    size_t input_elements = (size_t)rows * shape->input_dim;
    size_t rank_elements = (size_t)rows * 128;
    size_t branch_elements = (size_t)rows * shape->output_dim;
    size_t output_elements = branch_elements * (shape->qkv ? 3 : 1);
    size_t weight_elements = (size_t)shape->output_dim * 128;
    uint16_t *input_values = make_values(input_elements, 11);
    uint16_t *down_values = make_values((size_t)128 * shape->input_dim, 17);
    uint16_t *up_values = make_values(weight_elements, 23);
    uint16_t *base_values = make_values(output_elements, 29);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, input_values, input_elements);
    h3_gpu_tensor *down = h3_gpu_tensor_from_bf16(
        gpu, down_values, (size_t)128 * shape->input_dim);
    h3_gpu_tensor *up = h3_gpu_tensor_from_bf16(gpu, up_values,
                                                weight_elements);
    h3_gpu_tensor *baseline = h3_gpu_tensor_from_bf16(
        gpu, base_values, output_elements);
    h3_gpu_tensor *direct = h3_gpu_tensor_from_bf16(
        gpu, base_values, output_elements);
    h3_gpu_tensor *mps = h3_gpu_tensor_from_bf16(
        gpu, base_values, output_elements);
    h3_gpu_tensor *rank = h3_gpu_tensor_new_bf16(gpu, rank_elements);
    h3_gpu_tensor *delta = h3_gpu_tensor_new_bf16(gpu, branch_elements);
    if (!input || !down || !up || !baseline || !direct || !mps || !rank ||
        !delta)
        die("out of memory allocating benchmark tensors");
    if (!run_variant(gpu, VARIANT_BASELINE, baseline, rank, delta, input,
                     down, up, rows, shape, 0.0625f))
        die(h3_gpu_error(gpu));
    uint16_t *reference = malloc(output_elements * sizeof(*reference));
    if (!reference || !h3_gpu_tensor_read_bf16(baseline, reference,
                                               output_elements))
        die("cannot read baseline output");
    if (!run_variant(gpu, VARIANT_DIRECT, direct, rank, delta, input, down,
                     up, rows, shape, 0.0625f))
        die(h3_gpu_error(gpu));
    compare_output(direct, reference, output_elements, "direct", shape->name);
    if (!run_variant(gpu, VARIANT_MPS, mps, rank, delta, input, down, up,
                     rows, shape, 0.0625f))
        die(h3_gpu_error(gpu));
    compare_output(mps, reference, output_elements, "mps", shape->name);
    benchmark_variant(gpu, VARIANT_BASELINE, baseline, rank, delta, input,
                      down, up, base_values, output_elements, rows, shape,
                      warmup, repeats);
    benchmark_variant(gpu, VARIANT_DIRECT, direct, rank, delta, input, down,
                      up, base_values, output_elements, rows, shape, warmup,
                      repeats);
    benchmark_variant(gpu, VARIANT_MPS, mps, rank, delta, input, down, up,
                      base_values, output_elements, rows, shape, warmup,
                      repeats);
    free(reference);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(down);
    h3_gpu_tensor_free(up);
    h3_gpu_tensor_free(baseline);
    h3_gpu_tensor_free(direct);
    h3_gpu_tensor_free(mps);
    h3_gpu_tensor_free(rank);
    h3_gpu_tensor_free(delta);
    free(input_values);
    free(down_values);
    free(up_values);
    free(base_values);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    uint32_t rows = 512;
    const char *rows_text = getenv("H3_LORA_BENCH_ROWS");
    if (rows_text && *rows_text) rows = (uint32_t)strtoul(rows_text, NULL, 10);
    uint32_t warmup = 2;
    uint32_t repeats = 5;
    for (size_t index = 0; index < sizeof(shapes) / sizeof(*shapes); index++)
        benchmark_shape(gpu, rows, &shapes[index], warmup, repeats);
    h3_gpu_free(gpu);
    return 0;
}
