#include "h3_adapter.h"
#include "h3_host.h"
#include "h3_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);      \
        failures++;                                                              \
    }                                                                            \
} while (0)

static int close_enough(float left, float right) {
    return fabsf(left - right) <= 1e-6f * fmaxf(1.0f, fabsf(right));
}

static void test_profile_contracts(void) {
    h3_adapter_profile profile = H3_ADAPTER_PROFILE_NONE;
    CHECK(h3_adapter_profile_parse("modeltc-fl2va-768-4", &profile));
    CHECK(profile == H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_4);
    CHECK(!h3_adapter_profile_parse("modeltc-768", &profile));
    CHECK(!strcmp(h3_adapter_profile_name(profile), "modeltc-fl2va-768-4"));

    h3_params params = H3_PARAMS_DEFAULT;
    params.adapter_path = "adapter.safetensors";
    params.adapter_profile = profile;
    params.steps = 6;
    char error[256];
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    CHECK(params.adapter_kind == H3_ADAPTER_MODELTC_TURBO);
    CHECK(params.sampler == H3_SAMPLER_EULER);
    CHECK(params.scheduler == H3_SCHEDULER_SIMPLE);
    CHECK(params.steps == 6);
    CHECK(h3_adapter_profile_default_steps(profile) == 4);
    CHECK(close_enough(params.video_shift, 6.0f));
    CHECK(close_enough(params.audio_shift, 3.0f));
    params.adapter_strength = 0.0f;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));

    params = (h3_params)H3_PARAMS_DEFAULT;
    params.adapter_path = "pdd.safetensors";
    params.adapter_profile = H3_ADAPTER_PROFILE_PAI_FL2VA_8;
    params.adapter_strength = 0.75f;
    CHECK(!h3_adapter_profile_apply(&params, error, sizeof(error)));
    CHECK(strstr(error, "strength 1.0") != NULL);
    params.adapter_strength = 1.0f;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    CHECK(params.adapter_kind == H3_ADAPTER_ALIBABA_PAI_PDD);
    CHECK(params.steps == 8 && params.scheduler == H3_SCHEDULER_SIMPLE);

    CHECK(h3_adapter_profile_parse("modeltc-ref2va-768-8", &profile));
    CHECK(profile == H3_ADAPTER_PROFILE_MODELTC_REF2VA_768_8);
    CHECK(h3_adapter_profile_default_steps(profile) == 8);
    params = (h3_params)H3_PARAMS_DEFAULT;
    params.adapter_path = "ref.safetensors";
    params.adapter_profile = profile;
    params.steps = 3;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    CHECK(params.steps == 3 && close_enough(params.video_shift, 12.0f));

    params = (h3_params)H3_PARAMS_DEFAULT;
    params.adapter_path = "unexpected.safetensors";
    CHECK(!h3_adapter_profile_apply(&params, error, sizeof(error)));
}

static void test_runtime_schema_rejections(void) {
    char path[] = "/private/tmp/h3_adapter_schema_XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    if (descriptor < 0) return;
    /* A syntactically valid safetensors file with an unexpected inventory.
     * The strict tensor count check prevents unknown LoRA targets from being
     * accepted and silently skipped. */
    const char json[] =
        "{\"unexpected.lora_A.default.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[1,1],\"data_offsets\":[0,2]}}";
    uint64_t bytes = (uint64_t)strlen(json);
    uint16_t data = 0;
    CHECK(write(descriptor, &bytes, sizeof(bytes)) == (ssize_t)sizeof(bytes));
    CHECK(write(descriptor, json, strlen(json)) == (ssize_t)strlen(json));
    CHECK(write(descriptor, &data, sizeof(data)) == (ssize_t)sizeof(data));
    close(descriptor);
    char error[256];
    h3_adapter_runtime *adapter = h3_adapter_runtime_open(path,
        H3_ADAPTER_MODELTC_TURBO, 1.0f, error, sizeof(error));
    CHECK(adapter == NULL);
    CHECK(strstr(error, "schema mismatch") != NULL);
    h3_adapter_runtime_free(adapter);
    unlink(path);
}

static void test_published_modeltc_file(void) {
    const char *path = getenv("H3_TEST_MODELTC_ADAPTER");
    if (!path || !*path) return;
    char error[256];
    h3_adapter_runtime *adapter = h3_adapter_runtime_open(path,
        H3_ADAPTER_MODELTC_TURBO, 1.0f, error, sizeof(error));
    CHECK(adapter != NULL);
    if (adapter) {
        CHECK(h3_adapter_runtime_rank(adapter) == 128);
        CHECK(h3_adapter_runtime_tensor(adapter,
            "transformer_blocks.0.attn.to_q.lora_A.default.weight") != NULL);
    }
    h3_adapter_runtime_free(adapter);
}

static void test_published_pai_file(void) {
    const char *path = getenv("H3_TEST_PAI_ADAPTER");
    if (!path || !*path) return;
    char error[256];
    h3_adapter_runtime *adapter = h3_adapter_runtime_open(path,
        H3_ADAPTER_ALIBABA_PAI_PDD, 1.0f, error, sizeof(error));
    CHECK(adapter != NULL);
    if (adapter) {
        CHECK(h3_adapter_runtime_rank(adapter) == 64);
        CHECK(h3_adapter_runtime_tensor(adapter, "proj_out.weight") != NULL);
    }
    h3_adapter_runtime_free(adapter);
}

static void expect_public_rejection(h3_params *params, const char *needle) {
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    if (!ctx) return;
    CHECK(h3_generate(ctx, "adapter validation", params) == NULL);
    CHECK(strstr(h3_last_error(ctx), needle) != NULL);
    free(ctx);
}

static void test_public_parameter_validation(void) {
    char error[256];
    h3_params params = H3_PARAMS_DEFAULT;
    params.adapter_path = "adapter.safetensors";
    expect_public_rejection(&params, "named profile");

    params = (h3_params)H3_PARAMS_DEFAULT;
    params.adapter_path = "adapter.safetensors";
    params.adapter_profile = H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_4;
    params.steps = 3;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    CHECK(params.steps == 3);
    params.scheduler = H3_SCHEDULER_BETA;
    expect_public_rejection(&params, "must match");
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    params.core_reuse = 2;
    expect_public_rejection(&params, "reuse 1");
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    params.core_reuse = 1;
    params.adapter_profile = H3_ADAPTER_PROFILE_PAI_FL2VA_8;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    params.refine_video_path = "source.mp4";
    expect_public_rejection(&params, "only ModelTC");

    params = (h3_params)H3_PARAMS_DEFAULT;
    params.video_shift = 0.0f;
    expect_public_rejection(&params, "shifts must be finite and positive");
    params.video_shift = NAN;
    expect_public_rejection(&params, "shifts must be finite and positive");

    params = (h3_params)H3_PARAMS_DEFAULT;
    params.adapter_path = "adapter.safetensors";
    params.adapter_profile = H3_ADAPTER_PROFILE_PAI_REF2VA_8;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    expect_public_rejection(&params, "require at least one reference");
    params = (h3_params)H3_PARAMS_DEFAULT;
    params.adapter_path = "adapter.safetensors";
    params.adapter_profile = H3_ADAPTER_PROFILE_PAI_FL2VA_8;
    CHECK(h3_adapter_profile_apply(&params, error, sizeof(error)));
    params.reference_count = 1;
    expect_public_rejection(&params, "cannot be used with references");
}

static void test_inline_adapter_gates(void) {
    char error[256];
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    if (!ctx) return;
    h3_production_params production = {
        .working = H3_PARAMS_DEFAULT,
        .target = H3_PARAMS_DEFAULT
    };
    production.working.adapter_path = "adapter.safetensors";
    production.working.adapter_profile = H3_ADAPTER_PROFILE_PAI_FL2VA_8;
    CHECK(h3_adapter_profile_apply(&production.working, error, sizeof(error)));
    production.target = production.working;
    CHECK(h3_generate_inline_production(ctx, "adapter validation", &production) == NULL);
    CHECK(strstr(h3_last_error(ctx), "not PAI PDD") != NULL);

    production.working = (h3_params)H3_PARAMS_DEFAULT;
    production.working.adapter_path = "adapter.safetensors";
    production.working.adapter_profile = H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_8;
    production.working.steps = 8;
    CHECK(h3_adapter_profile_apply(&production.working, error, sizeof(error)));
    production.target = production.working;
    CHECK(h3_generate_inline_production(ctx, "adapter validation", &production) == NULL);
    CHECK(strstr(h3_last_error(ctx), "requires upscaler") != NULL);
    free(ctx);
}

static void test_shifted_grids(void) {
    h3_sigma_schedule schedule;
    CHECK(h3_serving_schedule_build_shifted(4, 6.0f, 3.0f, &schedule));
    CHECK(schedule.steps == 4);
    CHECK(close_enough(schedule.video[1], 18.0f / 19.0f));
    CHECK(close_enough(schedule.audio[1], 9.0f / 10.0f));
    CHECK(schedule.video[4] == 0.0f && schedule.audio[4] == 0.0f);
    CHECK(h3_beta_schedule_build_shifted(4, 6.0f, 3.0f, &schedule));
    CHECK(schedule.steps == 4);
    CHECK(schedule.video[0] > schedule.video[1]);
    CHECK(schedule.audio[0] > schedule.audio[1]);
    CHECK(!h3_serving_schedule_build_shifted(4, 0.0f, 3.0f, &schedule));
    CHECK(!h3_beta_schedule_build_shifted(4, NAN, 3.0f, &schedule));
}

static void test_pdd_plan(void) {
    float video[32], audio[32];
    char error[256];
    CHECK(h3_adapter_pdd_plan(12.0f, 32, 4, 3, video, 32,
                              error, sizeof(error)));
    CHECK(h3_adapter_pdd_plan(3.0f, 32, 4, 3, audio, 32,
                              error, sizeof(error)));
    float video_sum = 0.0f, audio_sum = 0.0f;
    for (unsigned index = 0; index < 32; index++) {
        if (index < 12 || index >= 16) {
            CHECK(video[index] == 0.0f && audio[index] == 0.0f);
        } else {
            CHECK(video[index] > 0.0f && audio[index] > 0.0f);
            video_sum += video[index];
            audio_sum += audio[index];
        }
    }
    CHECK(close_enough(video_sum, 1.0f));
    CHECK(close_enough(audio_sum, 1.0f));
    CHECK(fabsf(video[12] - audio[12]) > 1e-4f);
    CHECK(!h3_adapter_pdd_plan(12.0f, 31, 4, 0, video, 32,
                               error, sizeof(error)));
}

int main(void) {
    test_profile_contracts();
    test_shifted_grids();
    test_pdd_plan();
    test_runtime_schema_rejections();
    test_published_modeltc_file();
    test_published_pai_file();
    test_public_parameter_validation();
    test_inline_adapter_gates();
    return failures ? 1 : 0;
}
