/*
 * The ConvRot exporter leaves the text refiner in BF16, but changes its QKV
 * output rows from the released model's per-head Q/K/V groups to contiguous
 * Q, K, then V ranges.  Compile the private setup function into this test so
 * the production call site, rather than only the generic QKV kernel, selects
 * the correct decoder layout.
 */
#include "../h3_dit.c"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEST_TEXT_ROWS = 6, TEST_TEXT_DIM = 5120 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_refiner_layout.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static uint32_t random_state = UINT32_C(0x6d2b79f5);

static uint32_t random_u32(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static uint16_t *refine(const char *weights_path, int convrot,
                        const h3_text_embedding *text) {
    char error[512] = {0};
    h3_weight_store *weights = h3_weight_store_open(
        weights_path, error, sizeof(error));
    if (!weights) die(error);
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        h3_weight_store_free(weights);
        die(error);
    }
    h3_dit dit;
    memset(&dit, 0, sizeof(dit));
    dit.gpu = gpu;
    dit.weights = weights;
    dit.convrot = convrot;
    dit.text_rows = TEST_TEXT_ROWS;
    if (!refine_text(&dit, text, error, sizeof(error))) {
        h3_gpu_free(gpu);
        h3_weight_store_free(weights);
        die(error);
    }
    size_t elements = (size_t)TEST_TEXT_ROWS * HIDDEN;
    uint16_t *result = malloc(elements * sizeof(*result));
    require(result != NULL, "cannot allocate refined-text copy");
    require(h3_gpu_tensor_read_bf16(dit.refined_text, result, elements),
            "cannot read refined text");
    h3_gpu_tensor_free(dit.refined_text);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s BF16_TRANSFORMER CONVROT_CHECKPOINT\n",
                argv[0]);
        return 2;
    }
    uint16_t *values = malloc((size_t)TEST_TEXT_ROWS * TEST_TEXT_DIM *
                              sizeof(*values));
    require(values != NULL, "cannot allocate synthetic Qwen text");
    for (size_t index = 0; index < (size_t)TEST_TEXT_ROWS * TEST_TEXT_DIM;
         index++) {
        int32_t signed_value = (int32_t)(random_u32() % 2001u) - 1000;
        values[index] = f32_to_bf16((float)signed_value / 4096.0f);
    }
    h3_text_embedding text = {
        .tokens = TEST_TEXT_ROWS,
        .width = TEST_TEXT_DIM,
        .values = values,
    };
    uint16_t *bf16 = refine(argv[1], 0, &text);
    uint16_t *convrot = refine(argv[2], 1, &text);
    size_t bytes = (size_t)TEST_TEXT_ROWS * HIDDEN * sizeof(*bf16);
    require(!memcmp(bf16, convrot, bytes),
            "ConvRot refiner does not reproduce BF16 text conditioning");
    puts("ok: ConvRot refiner matches BF16 text conditioning");
    free(bf16);
    free(convrot);
    free(values);
    return 0;
}
