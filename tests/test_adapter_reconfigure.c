/* Exercise the private DiT session boundary used by inline target refinement.
 * The target stage must rebuild adapter workspace after free_session(), while
 * the ModelTC factor tensors remain resident. */
#define fail h3_adapter_private_fail
#include "../h3_adapter.c"
#undef fail
#include "../h3_dit.c"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_adapter_reconfigure.c: %s\n", message);
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

static void test_session_release_is_idempotent(void) {
    h3_dit dit;
    memset(&dit, 0, sizeof(dit));
    dit.row_maps = calloc(1, sizeof(*dit.row_maps));
    dit.reduced_row_maps = calloc(1, sizeof(*dit.reduced_row_maps));
    dit.final_audio_maps = calloc(1, sizeof(*dit.final_audio_maps));
    dit.final_video_maps = calloc(1, sizeof(*dit.final_video_maps));
    require(dit.row_maps && dit.reduced_row_maps && dit.final_audio_maps &&
            dit.final_video_maps, "allocate session map arrays");
    free_session(&dit);
    require(!dit.row_maps && !dit.reduced_row_maps && !dit.final_audio_maps &&
            !dit.final_video_maps, "free_session clears released map pointers");
    free_session(&dit);
}

static void test_target_refiner_workspace(h3_gpu *gpu) {
    uint16_t one = f32_to_bf16(1.0f);
    uint16_t down_host[128] = {0};
    uint16_t up_host[128] = {0};
    down_host[0] = up_host[0] = one;
    h3_dit dit;
    memset(&dit, 0, sizeof(dit));
    h3_adapter_runtime runtime;
    memset(&runtime, 0, sizeof(runtime));
    runtime.kind = H3_ADAPTER_MODELTC_TURBO;
    runtime.strength = 1.0f;
    runtime.rank = 128;
    dit.gpu = gpu;
    dit.adapter.runtime = &runtime;
    h3_dit_adapter_factor factor = {
        .input = 1,
        .output = 1,
        .down = h3_gpu_tensor_from_bf16(gpu, down_host, 128),
        .up = h3_gpu_tensor_from_bf16(gpu, up_host, 128),
    };
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, &one, 1);
    h3_gpu_tensor *output = h3_gpu_tensor_from_bf16(gpu, &one, 1);
    char error[256] = {0};
    require(factor.down && factor.up && input && output,
            "allocate target refiner tensors");

    /* The first stage owns this workspace.  Releasing its session must not
     * release the runtime factors, and the target needs fresh text-row scratch
     * before its own refine_text dispatch. */
    require(adapter_prepare_scratch(&dit, 1, "first-stage refiner", error,
                                    sizeof(error)), error);
    free_tensor(&dit.adapter_rank);
    require(adapter_prepare_scratch(&dit, 1, "target-stage refiner", error,
                                    sizeof(error)), error);
    require(h3_gpu_begin(gpu), "begin target refiner command");
    require(adapter_apply(&dit, &factor, input, output, 1, -1, 0,
                          error, sizeof(error), "target refiner adapter"), error);
    require(h3_gpu_submit(gpu), "submit target refiner command");
    uint16_t result = 0;
    require(h3_gpu_tensor_read_bf16(output, &result, 1),
            "read target refiner output");
    uint32_t result_bits = (uint32_t)result << 16;
    float result_value;
    memcpy(&result_value, &result_bits, sizeof(result_value));
    require(fabsf(result_value - (1.0f + 8.0f / 128.0f)) <= 0.02f,
            "target refiner uses resident ModelTC adapter factors");
    h3_gpu_tensor_free(factor.down);
    h3_gpu_tensor_free(factor.up);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    free_tensor(&dit.adapter_rank);
}

int main(void) {
    test_session_release_is_idempotent();
    char error[256] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    require(gpu != NULL, error);
    test_target_refiner_workspace(gpu);
    h3_gpu_free(gpu);
    puts("adapter reconfiguration tests passed");
    return 0;
}
