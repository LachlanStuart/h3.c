#include "h3_adapter.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_adapter_runtime {
    h3_st_header header;
    h3_adapter_kind kind;
    float strength;
    unsigned rank;
};

/* LightX2V's official MiniMax-H3 Turbo inference applies its documented
 * lora_alpha=8 over the rank-128 factors. The safetensors container metadata
 * describes export provenance (and varies across releases), so this is part
 * of the named ModelTC serving profile rather than an inferred file field. */
enum { H3_MODELTC_PROFILE_LORA_ALPHA = 8 };

typedef struct {
    const char *name;
    h3_adapter_profile profile;
    h3_adapter_kind kind;
    int steps;
    float video_shift;
    float audio_shift;
} h3_adapter_profile_info;

static const h3_adapter_profile_info profiles[] = {
    {"modeltc-fl2va-544-4", H3_ADAPTER_PROFILE_MODELTC_FL2VA_544_4,
     H3_ADAPTER_MODELTC_TURBO, 4, 12.0f, 3.0f},
    {"modeltc-fl2va-544-8", H3_ADAPTER_PROFILE_MODELTC_FL2VA_544_8,
     H3_ADAPTER_MODELTC_TURBO, 8, 12.0f, 3.0f},
    {"modeltc-fl2va-768-4", H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_4,
     H3_ADAPTER_MODELTC_TURBO, 4, 6.0f, 3.0f},
    {"modeltc-fl2va-768-8", H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_8,
     H3_ADAPTER_MODELTC_TURBO, 8, 6.0f, 3.0f},
    {"modeltc-ref2va-544-4", H3_ADAPTER_PROFILE_MODELTC_REF2VA_544_4,
     H3_ADAPTER_MODELTC_TURBO, 4, 12.0f, 3.0f},
    {"modeltc-ref2va-768-8", H3_ADAPTER_PROFILE_MODELTC_REF2VA_768_8,
     H3_ADAPTER_MODELTC_TURBO, 8, 12.0f, 3.0f},
    {"pai-fl2va-8", H3_ADAPTER_PROFILE_PAI_FL2VA_8,
     H3_ADAPTER_ALIBABA_PAI_PDD, 8, 12.0f, 3.0f},
    {"pai-ref2va-8", H3_ADAPTER_PROFILE_PAI_REF2VA_8,
     H3_ADAPTER_ALIBABA_PAI_PDD, 8, 12.0f, 3.0f}
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static const h3_adapter_profile_info *profile_info(h3_adapter_profile profile) {
    for (size_t index = 0; index < sizeof(profiles) / sizeof(*profiles); index++)
        if (profiles[index].profile == profile) return &profiles[index];
    return NULL;
}

int h3_adapter_profile_parse(const char *name, h3_adapter_profile *profile) {
    if (!name || !profile) return 0;
    if (!strcmp(name, "none")) {
        *profile = H3_ADAPTER_PROFILE_NONE;
        return 1;
    }
    for (size_t index = 0; index < sizeof(profiles) / sizeof(*profiles); index++) {
        if (!strcmp(name, profiles[index].name)) {
            *profile = profiles[index].profile;
            return 1;
        }
    }
    return 0;
}

const char *h3_adapter_profile_name(h3_adapter_profile profile) {
    if (profile == H3_ADAPTER_PROFILE_NONE) return "none";
    const h3_adapter_profile_info *info = profile_info(profile);
    return info ? info->name : NULL;
}

int h3_adapter_profile_default_steps(h3_adapter_profile profile) {
    const h3_adapter_profile_info *info = profile_info(profile);
    return info ? info->steps : 0;
}

int h3_adapter_profile_is_modeltc(h3_adapter_profile profile) {
    const h3_adapter_profile_info *info = profile_info(profile);
    return info && info->kind == H3_ADAPTER_MODELTC_TURBO;
}

int h3_adapter_profile_apply(h3_params *params, char *error,
                             size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!params) {
        fail(error, error_size, "adapter profile requires parameters");
        return 0;
    }
    if (params->adapter_profile == H3_ADAPTER_PROFILE_NONE) {
        if (params->adapter_path || params->adapter_kind != H3_ADAPTER_NONE) {
            fail(error, error_size, "an adapter path requires an adapter profile");
            return 0;
        }
        return 1;
    }
    const h3_adapter_profile_info *info = profile_info(params->adapter_profile);
    if (!info || !params->adapter_path || !*params->adapter_path ||
        !isfinite(params->adapter_strength) || params->adapter_strength < 0.0f) {
        fail(error, error_size, "invalid adapter profile, path, or strength");
        return 0;
    }
    if (info->kind == H3_ADAPTER_ALIBABA_PAI_PDD &&
        fabsf(params->adapter_strength - 1.0f) > 1e-6f) {
        fail(error, error_size, "PAI PDD requires adapter strength 1.0");
        return 0;
    }
    params->adapter_kind = info->kind;
    params->sampler = H3_SAMPLER_EULER;
    params->scheduler = H3_SCHEDULER_SIMPLE;
    /* ModelTC's published files are named for their nominal 4/8-step
     * profiles, but their LoRA factors do not encode a fixed evaluation
     * count. Preserve an explicit caller choice for ablations and restart
     * suffixes. PAI PDD's heads are schedule-indexed and must stay at 8. */
    if (info->kind == H3_ADAPTER_ALIBABA_PAI_PDD) params->steps = info->steps;
    params->video_shift = info->video_shift;
    params->audio_shift = info->audio_shift;
    return 1;
}

int h3_adapter_pdd_plan(float shift, unsigned fine_steps,
                        unsigned block_size, unsigned coarse_step,
                        float *plan, size_t plan_count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!plan || fine_steps == 0 || block_size == 0 ||
        fine_steps % block_size || plan_count < fine_steps ||
        coarse_step >= fine_steps / block_size || !isfinite(shift) ||
        shift <= 0.0f) {
        fail(error, error_size, "invalid PDD plan arguments");
        return 0;
    }
    for (unsigned index = 0; index < fine_steps; index++) plan[index] = 0.0f;
    unsigned start = coarse_step * block_size;
    double total = 0.0;
    double widths[64];
    if (block_size > sizeof(widths) / sizeof(*widths)) {
        fail(error, error_size, "PDD block is too large");
        return 0;
    }
    for (unsigned local = 0; local < block_size; local++) {
        double sigma0 = 1.0 - (double)(start + local) / fine_steps;
        double sigma1 = 1.0 - (double)(start + local + 1) / fine_steps;
        double shifted0 = shift * sigma0 / (1.0 + (shift - 1.0) * sigma0);
        double shifted1 = shift * sigma1 / (1.0 + (shift - 1.0) * sigma1);
        widths[local] = shifted0 - shifted1;
        total += widths[local];
    }
    if (!(total > 0.0) || !isfinite(total)) {
        fail(error, error_size, "PDD plan has no positive sigma interval");
        return 0;
    }
    for (unsigned local = 0; local < block_size; local++)
        plan[start + local] = (float)(widths[local] / total);
    return 1;
}

static int tensor_shape(const h3_st_header *header, const char *name,
                        uint64_t rows, uint64_t columns, char *error,
                        size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns) {
        fail(error, error_size, "adapter tensor schema mismatch: %s", name);
        return 0;
    }
    return 1;
}

static int vector_shape(const h3_st_header *header, const char *name,
                        uint64_t first, uint64_t second, int matrix,
                        char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    int valid = tensor && tensor->dtype == H3_DTYPE_BF16 &&
        tensor->ndim == (matrix ? 3 : 2) && tensor->shape[0] == 32 &&
        tensor->shape[1] == first && (!matrix || tensor->shape[2] == second);
    if (!valid) {
        fail(error, error_size, "adapter tensor schema mismatch: %s", name);
        return 0;
    }
    return 1;
}

static int expected_lora(const h3_st_header *header, const char *prefix,
                         const char *target, const char *down_suffix,
                         const char *up_suffix, unsigned rank,
                         uint64_t input, uint64_t output, char *error,
                         size_t error_size) {
    char down[192], up[192];
    snprintf(down, sizeof(down), "%s%s%s", prefix, target, down_suffix);
    snprintf(up, sizeof(up), "%s%s%s", prefix, target, up_suffix);
    return tensor_shape(header, down, rank, input, error, error_size) &&
        tensor_shape(header, up, output, rank, error, error_size);
}

static int validate_lora_layers(const h3_st_header *header, unsigned rank,
                                const char *down_suffix, const char *up_suffix,
                                int include_adaln, char *error,
                                size_t error_size) {
    static const struct { const char *name; uint64_t in, out; } targets[] = {
        {"attn.to_q", 5376, 7168}, {"attn.to_k", 5376, 7168},
        {"attn.to_v", 5376, 7168}, {"attn.to_out.0", 7168, 5376},
        {"ff.net.0.proj", 5376, 28672}, {"ff.net.2", 14336, 5376}
    };
    for (unsigned block = 0; block < 50; block++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "transformer_blocks.%u.", block);
        for (size_t target = 0; target < sizeof(targets) / sizeof(*targets); target++)
            if (!expected_lora(header, prefix, targets[target].name,
                               down_suffix, up_suffix, rank,
                               targets[target].in, targets[target].out,
                               error, error_size)) return 0;
        if (include_adaln && !expected_lora(
                header, prefix, "adaln_proj.linear", down_suffix, up_suffix,
                rank, 2688, 96768, error, error_size)) return 0;
    }
    for (unsigned block = 0; block < 2; block++) {
        char prefix[80];
        snprintf(prefix, sizeof(prefix), "token_refiner.refiner_blocks.%u.", block);
        for (size_t target = 0; target < sizeof(targets) / sizeof(*targets); target++)
            if (!expected_lora(header, prefix, targets[target].name,
                               down_suffix, up_suffix, rank,
                               targets[target].in, targets[target].out,
                               error, error_size)) return 0;
    }
    return 1;
}

h3_adapter_runtime *h3_adapter_runtime_open(const char *path,
                                             h3_adapter_kind kind,
                                             float strength, char *error,
                                             size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !isfinite(strength) || strength < 0.0f ||
        (kind != H3_ADAPTER_MODELTC_TURBO &&
         kind != H3_ADAPTER_ALIBABA_PAI_PDD)) {
        fail(error, error_size, "invalid runtime adapter arguments");
        return NULL;
    }
    h3_adapter_runtime *adapter = calloc(1, sizeof(*adapter));
    if (!adapter) {
        fail(error, error_size, "out of memory opening runtime adapter");
        return NULL;
    }
    if (!h3_st_read_header(path, &adapter->header, error, error_size)) {
        free(adapter);
        return NULL;
    }
    adapter->kind = kind;
    adapter->strength = strength;
    if (kind == H3_ADAPTER_MODELTC_TURBO) {
        adapter->rank = 128;
        if (adapter->header.tensor_count != 624) {
            fail(error, error_size, "adapter tensor schema mismatch: unsupported ModelTC inventory");
            goto failed;
        }
        if (!validate_lora_layers(&adapter->header, adapter->rank,
                                  ".lora_A.default.weight",
                                  ".lora_B.default.weight", 0,
                                  error, error_size)) goto failed;
    } else {
        adapter->rank = 64;
        if (adapter->header.tensor_count != 728) {
            fail(error, error_size, "adapter tensor schema mismatch: unsupported PAI inventory");
            goto failed;
        }
        if (fabsf(strength - 1.0f) > 1e-6f ||
            !validate_lora_layers(&adapter->header, adapter->rank,
                                  ".lora_down", ".lora_up", 1,
                                  error, error_size) ||
            !vector_shape(&adapter->header, "proj_out.weight", 96, 5376, 1,
                          error, error_size) ||
            !vector_shape(&adapter->header, "proj_out.bias", 96, 0, 0,
                          error, error_size) ||
            !vector_shape(&adapter->header, "audio_proj_out.weight", 32, 5376,
                          1, error, error_size) ||
            !vector_shape(&adapter->header, "audio_proj_out.bias", 32, 0, 0,
                          error, error_size)) goto failed;
    }
    return adapter;
failed:
    h3_st_free_header(&adapter->header);
    free(adapter);
    return NULL;
}

void h3_adapter_runtime_free(h3_adapter_runtime *adapter) {
    if (!adapter) return;
    h3_st_free_header(&adapter->header);
    free(adapter);
}

unsigned h3_adapter_runtime_rank(const h3_adapter_runtime *adapter) {
    return adapter ? adapter->rank : 0;
}

h3_adapter_kind h3_adapter_runtime_kind(const h3_adapter_runtime *adapter) {
    return adapter ? adapter->kind : H3_ADAPTER_NONE;
}

float h3_adapter_runtime_strength(const h3_adapter_runtime *adapter) {
    return adapter ? adapter->strength : 0.0f;
}

float h3_adapter_runtime_scale(const h3_adapter_runtime *adapter) {
    if (!adapter) return 0.0f;
    if (adapter->kind == H3_ADAPTER_MODELTC_TURBO) {
        if (!adapter->rank) return 0.0f;
        return adapter->strength *
            (float)H3_MODELTC_PROFILE_LORA_ALPHA / (float)adapter->rank;
    }
    return adapter->strength;
}

const h3_st_tensor *h3_adapter_runtime_tensor(
    const h3_adapter_runtime *adapter, const char *name) {
    return adapter ? h3_st_find(&adapter->header, name) : NULL;
}

const h3_st_header *h3_adapter_runtime_header(
    const h3_adapter_runtime *adapter) {
    return adapter ? &adapter->header : NULL;
}

h3_gpu_tensor *h3_adapter_runtime_load_bf16(
    const h3_adapter_runtime *adapter, h3_gpu *gpu, const char *name,
    uint64_t rows, uint64_t columns, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!adapter || !gpu || !name || !tensor_shape(&adapter->header, name,
                                                   rows, columns, error,
                                                   error_size)) return NULL;
    const h3_st_tensor *tensor = h3_st_find(&adapter->header, name);
    h3_gpu_tensor *result = h3_gpu_tensor_load_bf16(
        gpu, adapter->header.path, tensor->file_offset,
        (size_t)(rows * columns));
    if (!result) fail(error, error_size, "cannot load adapter factor %s: %s",
                      name, h3_gpu_error(gpu));
    return result;
}
