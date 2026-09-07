/* Exercise the private runtime application helper directly.  Core DiT blocks
 * and text-refiner blocks both call adapter_apply, so separate labelled calls
 * prove the alpha/rank factor and strength-zero handling at the actual shared
 * GPU contribution boundary rather than only as scalar arithmetic. */
#define fail h3_adapter_private_fail
#include "../h3_adapter.c"
#undef fail
#include "../h3_dit.c"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_adapter_scale.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) test_fail(message);
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

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) <=
        0.02f * fmaxf(1.0f, fabsf(expected));
}

/* A and B contain one nonzero component, making B(A(1)) exactly one. */
static float run_contribution(h3_gpu *gpu, h3_adapter_kind kind,
                              float strength, const char *label) {
    uint16_t input_host[] = {f32_to_bf16(1.0f)};
    uint16_t output_host[] = {f32_to_bf16(1.0f)};
    uint16_t down_host[128] = {0};
    uint16_t up_host[128] = {0};
    down_host[0] = up_host[0] = f32_to_bf16(1.0f);
    h3_dit dit;
    memset(&dit, 0, sizeof(dit));
    h3_adapter_runtime runtime;
    memset(&runtime, 0, sizeof(runtime));
    runtime.kind = kind;
    runtime.strength = strength;
    runtime.rank = 128;
    dit.gpu = gpu;
    dit.adapter.runtime = &runtime;
    h3_dit_adapter_factor factor = {
        .input = 1,
        .output = 1,
        .down = h3_gpu_tensor_from_bf16(gpu, down_host, 128),
        .up = h3_gpu_tensor_from_bf16(gpu, up_host, 128),
    };
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, input_host, 1);
    h3_gpu_tensor *output = h3_gpu_tensor_from_bf16(gpu, output_host, 1);
    dit.adapter_rank = h3_gpu_tensor_new_bf16(gpu, 128);
    char error[256] = {0};
    require(factor.down && factor.up && input && output && dit.adapter_rank,
            "allocate scale-test tensors");
    require(h3_gpu_begin(gpu), "begin adapter-scale GPU command");
    require(adapter_apply(&dit, &factor, input, output, 1, -1, 0,
                          error, sizeof(error), label), error);
    require(h3_gpu_submit(gpu), "submit adapter-scale GPU command");
    require(h3_gpu_tensor_read_bf16(output, output_host, 1),
            "read adapter-scale output");
    h3_gpu_tensor_free(factor.down);
    h3_gpu_tensor_free(factor.up);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(dit.adapter_rank);
    return bf16_to_f32(output_host[0]);
}

int main(void) {
    char error[256] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    require(gpu != NULL, error);
    float expected_turbo = 1.0f + 8.0f / 128.0f;
    require(close_enough(run_contribution(gpu, H3_ADAPTER_MODELTC_TURBO, 1.0f,
                             "core Turbo adapter"), expected_turbo),
            "core Turbo contribution must use alpha/rank");
    require(close_enough(run_contribution(gpu, H3_ADAPTER_MODELTC_TURBO, 1.0f,
                             "refiner Turbo adapter"), expected_turbo),
            "refiner Turbo contribution must use alpha/rank");
    float baseline = 1.0f;
    require(close_enough(run_contribution(gpu, H3_ADAPTER_MODELTC_TURBO, 0.0f,
                             "core zero-strength Turbo adapter"), baseline),
            "zero-strength core Turbo adapter must leave the base output unchanged");
    require(close_enough(run_contribution(gpu, H3_ADAPTER_MODELTC_TURBO, 0.0f,
                             "refiner zero-strength Turbo adapter"), baseline),
            "zero-strength refiner Turbo adapter must leave the base output unchanged");
    require(close_enough(run_contribution(gpu, H3_ADAPTER_ALIBABA_PAI_PDD, 1.0f,
                             "PAI adapter"), 2.0f),
            "PAI adapter scale must remain unchanged");
    h3_gpu_free(gpu);
    puts("runtime adapter scale tests passed");
    return 0;
}
