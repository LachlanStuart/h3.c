#include "h3_pdd.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);       \
        failures++;                                                              \
    }                                                                            \
} while (0)

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

static uint16_t reference(const uint16_t *values, unsigned steps,
                          size_t stride, size_t element,
                          const float *plan) {
    float sum = 0.0f;
    for (unsigned step = 0; step < steps; step++)
        sum = fmaf(bf16_to_f32(f32_to_bf16(plan[step])),
                   bf16_to_f32(values[(size_t)step * stride + element]), sum);
    return f32_to_bf16(sum);
}

static void test_independent_modality_heads(void) {
    enum { STEPS = 4, VIDEO_OUT = 3, AUDIO_OUT = 2, HIDDEN = 5 };
    uint16_t video_w[STEPS * VIDEO_OUT * HIDDEN];
    uint16_t video_b[STEPS * VIDEO_OUT];
    uint16_t audio_w[STEPS * AUDIO_OUT * HIDDEN];
    uint16_t audio_b[STEPS * AUDIO_OUT];
    uint16_t fused_video_w[VIDEO_OUT * HIDDEN], fused_video_b[VIDEO_OUT];
    uint16_t fused_audio_w[AUDIO_OUT * HIDDEN], fused_audio_b[AUDIO_OUT];
    const float video_plan[STEPS] = {0.0f, 0.125f, 0.375f, 0.5f};
    const float audio_plan[STEPS] = {0.5f, 0.375f, 0.125f, 0.0f};
    for (unsigned step = 0; step < STEPS; step++) {
        for (unsigned index = 0; index < VIDEO_OUT * HIDDEN; index++)
            video_w[step * VIDEO_OUT * HIDDEN + index] = f32_to_bf16(
                (float)(step + 1) + (float)index * 0.125f);
        for (unsigned index = 0; index < VIDEO_OUT; index++)
            video_b[step * VIDEO_OUT + index] = f32_to_bf16(
                (float)step * 0.25f - (float)index * 0.125f);
        for (unsigned index = 0; index < AUDIO_OUT * HIDDEN; index++)
            audio_w[step * AUDIO_OUT * HIDDEN + index] = f32_to_bf16(
                (float)(4 - step) - (float)index * 0.0625f);
        for (unsigned index = 0; index < AUDIO_OUT; index++)
            audio_b[step * AUDIO_OUT + index] = f32_to_bf16(
                (float)(step + index) * 0.0625f);
    }
    char error[256];
    CHECK(h3_pdd_head_fuse_bf16(video_w, video_b, STEPS, VIDEO_OUT, HIDDEN,
                                 video_plan, STEPS, fused_video_w,
                                 fused_video_b, error, sizeof(error)));
    CHECK(h3_pdd_head_fuse_bf16(audio_w, audio_b, STEPS, AUDIO_OUT, HIDDEN,
                                 audio_plan, STEPS, fused_audio_w,
                                 fused_audio_b, error, sizeof(error)));
    for (unsigned index = 0; index < VIDEO_OUT * HIDDEN; index++)
        CHECK(fused_video_w[index] == reference(video_w, STEPS,
              VIDEO_OUT * HIDDEN, index, video_plan));
    for (unsigned index = 0; index < VIDEO_OUT; index++)
        CHECK(fused_video_b[index] == reference(video_b, STEPS, VIDEO_OUT,
                                                 index, video_plan));
    for (unsigned index = 0; index < AUDIO_OUT * HIDDEN; index++)
        CHECK(fused_audio_w[index] == reference(audio_w, STEPS,
              AUDIO_OUT * HIDDEN, index, audio_plan));
    for (unsigned index = 0; index < AUDIO_OUT; index++)
        CHECK(fused_audio_b[index] == reference(audio_b, STEPS, AUDIO_OUT,
                                                 index, audio_plan));
    CHECK(fused_video_w[1] != fused_audio_w[1]);
}

static void test_rejections(void) {
    uint16_t weights[4] = {0}, biases[2] = {0}, fused_weights[2], fused_biases;
    const float incomplete[2] = {0.25f, 0.25f};
    char error[256];
    CHECK(!h3_pdd_head_fuse_bf16(weights, biases, 2, 1, 2, incomplete, 2,
                                  fused_weights, &fused_biases, error,
                                  sizeof(error)));
    CHECK(strstr(error, "summing to one") != NULL);
    CHECK(!h3_pdd_require_full_bf16_base(1, error, sizeof(error)));
    CHECK(strstr(error, "ConvRot") != NULL);
    CHECK(h3_pdd_require_full_bf16_base(0, error, sizeof(error)));
}

int main(void) {
    test_independent_modality_heads();
    test_rejections();
    return failures ? 1 : 0;
}
