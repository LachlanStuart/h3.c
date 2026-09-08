/*
 * Bounded real-checkpoint ConvRot DiT block profiler.
 *
 * This is deliberately a test-only translation unit.  It includes the
 * private h3_dit implementation so that it can prepare a real production
 * session and invoke run_block() without changing the runtime.  By default
 * it loads all 50 layers, then prepares a deterministic block-chain proxy at
 * the 1024x512/243 production geometry (37,680 rows).  --small selects the
 * 512x256/175 proxy (7,246 rows).  The normal (unfenced) block timing keeps
 * the production command chain intact; the optional per-operation profile
 * submits and waits after each operation to expose stage costs.  The
 * --ab-full-k, --ab-head-major, and --ab-optimized modes interleave a
 * disabled reference with the corresponding optimized ConvRot dispatch.
 *
 * Build from projects/h3/code after the normal objects are available.  The
 * small helper in /tmp/h3-convrot-opt/build.sh shows the exact link rule.
 */
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif

#include "../h3_gpu.h"
#include "../h3_adapter.h"
#include "../h3_dit.h"
#include "../h3_host.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    H3_PROFILE_TEXT_TOKENS = 6,
    H3_PROFILE_TEXT_WIDTH = 5120,
    H3_PROFILE_DEFAULT_WIDTH = 1024,
    H3_PROFILE_DEFAULT_HEIGHT = 512,
    H3_PROFILE_DEFAULT_FRAMES = 243,
    H3_PROFILE_SMALL_WIDTH = 512,
    H3_PROFILE_SMALL_HEIGHT = 256,
    H3_PROFILE_SMALL_FRAMES = 175,
    H3_PROFILE_DEFAULT_STEPS = 16,
    H3_PROFILE_MAX_STAGES = 48
};

typedef enum {
    H3_AB_NONE = 0,
    H3_AB_FULL_K,
    H3_AB_HEAD_MAJOR,
    H3_AB_OPTIMIZED
} ab_mode;

typedef struct {
    const char *name;
    unsigned calls;
    double wall_seconds;
    double cpu_seconds;
    double encode_seconds;
    double wait_seconds;
    double gpu_seconds;
    double pass_wall[4];
    double pass_gpu[4];
} profile_stage;

typedef struct {
    h3_gpu *gpu;
    profile_stage stages[H3_PROFILE_MAX_STAGES];
    size_t stage_count;
    unsigned pass;
    int failed;
} profile_context;

static profile_context *active_profile;

static double profile_clock(clockid_t clock_id, int *ok) {
    struct timespec value;
    if (clock_gettime(clock_id, &value) != 0) {
        if (ok) *ok = 0;
        return 0.0;
    }
    if (ok) *ok = 1;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static profile_stage *profile_stage_for(profile_context *context,
                                        const char *name) {
    for (size_t index = 0; index < context->stage_count; index++)
        if (!strcmp(context->stages[index].name, name))
            return &context->stages[index];
    if (context->stage_count >= H3_PROFILE_MAX_STAGES) return NULL;
    profile_stage *stage = &context->stages[context->stage_count++];
    memset(stage, 0, sizeof(*stage));
    stage->name = name;
    return stage;
}

/* A profile fence is a complete submit/wait boundary.  The following begin
 * leaves the command chain ready for the next operation in run_block(). */
static int profile_complete(h3_gpu *gpu, const char *name, int ok,
                            const h3_gpu_stats *before,
                            double wall_start, double cpu_start) {
    if (!active_profile) return ok;
    if (!ok) {
        active_profile->failed = 1;
        return 0;
    }
    if (!h3_gpu_submit(gpu)) {
        active_profile->failed = 1;
        return 0;
    }
    h3_gpu_stats after;
    if (!h3_gpu_get_stats(gpu, &after)) {
        active_profile->failed = 1;
        return 0;
    }
    int wall_ok = 0;
    int cpu_ok = 0;
    double wall_end = profile_clock(CLOCK_MONOTONIC, &wall_ok);
    double cpu_end = profile_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
    if (!wall_ok || !cpu_ok) {
        active_profile->failed = 1;
        return 0;
    }
    profile_stage *stage = profile_stage_for(active_profile, name);
    if (!stage) {
        active_profile->failed = 1;
        return 0;
    }
    stage->calls++;
    double wall = wall_end - wall_start;
    double gpu_time = after.gpu_seconds - before->gpu_seconds;
    stage->wall_seconds += wall;
    stage->cpu_seconds += cpu_end - cpu_start;
    stage->encode_seconds += after.command_encode_seconds -
        before->command_encode_seconds;
    stage->wait_seconds += after.command_wait_seconds -
        before->command_wait_seconds;
    stage->gpu_seconds += gpu_time;
    unsigned pass = active_profile->pass < 4 ? active_profile->pass : 3;
    stage->pass_wall[pass] += wall;
    stage->pass_gpu[pass] += gpu_time;
    if (!h3_gpu_begin(gpu)) {
        active_profile->failed = 1;
        return 0;
    }
    return 1;
}

/* The wrappers are defined before h3_dit.c and selected only while that file
 * is included below.  They pass through unchanged outside the fenced profile. */
#define PROFILE_WRAP(NAME, LABEL, DECL, CALL)                                  \
    static int wrap_##NAME DECL {                                              \
        if (!active_profile) return h3_gpu_##NAME CALL;                        \
        h3_gpu_stats before;                                                    \
        int wall_ok = 0, cpu_ok = 0;                                           \
        double wall_start = profile_clock(CLOCK_MONOTONIC, &wall_ok);          \
        double cpu_start = profile_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);   \
        if (!wall_ok || !cpu_ok || !h3_gpu_get_stats(gpu, &before)) {          \
            active_profile->failed = 1;                                        \
            return 0;                                                           \
        }                                                                       \
        int ok = h3_gpu_##NAME CALL;                                           \
        return profile_complete(gpu, LABEL, ok, &before, wall_start,           \
                                cpu_start);                                     \
    }

static const char *convrot_label(uint32_t input_dim, uint32_t output_dim) {
    if (input_dim == 5376u && output_dim == 21504u) return "convrot_qkv";
    if (input_dim == 7168u && output_dim == 5376u) return "convrot_out";
    if (input_dim == 5376u && output_dim == 28672u) return "convrot_fc1";
    if (input_dim == 14336u && output_dim == 5376u) return "convrot_fc2";
    return "convrot_other";
}

static const char *lora_label(int component, uint32_t input_dim,
                              uint32_t output_dim) {
    if (component == 0) return "lora_q";
    if (component == 1) return "lora_k";
    if (component == 2) return "lora_v";
    if (input_dim == 7168u && output_dim == 5376u) return "lora_out";
    if (input_dim == 5376u && output_dim == 28672u) return "lora_fc1";
    if (input_dim == 14336u && output_dim == 5376u) return "lora_fc2";
    return "lora_other";
}

PROFILE_WRAP(adaln_bf16, "attention_adaln",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *i,
  const h3_gpu_tensor *n, const h3_gpu_tensor *m, const h3_gpu_tensor *r,
  uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift,
  uint32_t scale, float epsilon),
 (gpu, o, i, n, m, r, rows, width, slots, shift, scale, epsilon))
PROFILE_WRAP(linear_convrot_int8_bf16, convrot_label(in_dim, out_dim),
 (h3_gpu *gpu, h3_gpu_tensor *o, h3_gpu_tensor *rot,
  h3_gpu_tensor *qi, h3_gpu_tensor *sc, const h3_gpu_tensor *i,
  const h3_gpu_tensor *w, const h3_gpu_tensor *ws, uint32_t rows,
  uint32_t in_dim, uint32_t out_dim),
 (gpu, o, rot, qi, sc, i, w, ws, rows, in_dim, out_dim))
PROFILE_WRAP(qkv_rope_bf16, "qkv_rope",
 (h3_gpu *gpu, h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
  const h3_gpu_tensor *qkv, const h3_gpu_tensor *qn,
  const h3_gpu_tensor *kn, const h3_gpu_tensor *rc,
  const h3_gpu_tensor *rs, uint32_t rows, uint32_t heads,
  uint32_t head_dim, uint32_t rope_half, float epsilon),
 (gpu, q, k, v, qkv, qn, kn, rc, rs, rows, heads, head_dim, rope_half,
  epsilon))
PROFILE_WRAP(qkv_rope_bf16_for_sdpa, "qkv_rope_sdpa",
 (h3_gpu *gpu, h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
  const h3_gpu_tensor *qkv, const h3_gpu_tensor *qn,
  const h3_gpu_tensor *kn, const h3_gpu_tensor *rc,
  const h3_gpu_tensor *rs, uint32_t rows, uint32_t heads,
  uint32_t head_dim, uint32_t rope_half, float epsilon),
 (gpu, q, k, v, qkv, qn, kn, rc, rs, rows, heads, head_dim, rope_half,
  epsilon))
PROFILE_WRAP(sdpa_bf16, "sdpa",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *q,
  const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint32_t rows,
  uint32_t heads, uint32_t head_dim, float scale),
 (gpu, o, q, k, v, rows, heads, head_dim, scale))
PROFILE_WRAP(gate_adaln_bf16, "gate_adaln",
 (h3_gpu *gpu, h3_gpu_tensor *gr, h3_gpu_tensor *o,
  const h3_gpu_tensor *res, const h3_gpu_tensor *br,
  const h3_gpu_tensor *nw, const h3_gpu_tensor *gm,
  const h3_gpu_tensor *nm, const h3_gpu_tensor *rm, uint32_t rows,
  uint32_t width, uint32_t slots, uint32_t gate, uint32_t shift,
  uint32_t scale, float epsilon),
 (gpu, gr, o, res, br, nw, gm, nm, rm, rows, width, slots, gate, shift,
  scale, epsilon))
PROFILE_WRAP(gate_bf16, "gate",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *res,
  const h3_gpu_tensor *br, const h3_gpu_tensor *m,
  const h3_gpu_tensor *rm, uint32_t rows, uint32_t width, uint32_t slots,
  uint32_t gate),
 (gpu, o, res, br, m, rm, rows, width, slots, gate))
PROFILE_WRAP(swiglu_bf16, "swiglu",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *f,
  uint32_t rows, uint32_t width),
 (gpu, o, f, rows, width))
PROFILE_WRAP(lora_bf16, lora_label(component, in_dim, out_dim),
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *i,
  const h3_gpu_tensor *d, const h3_gpu_tensor *u, float scale,
  uint32_t rows, uint32_t in_dim, uint32_t rank, uint32_t out_dim,
  int component, int grouped),
 (gpu, o, i, d, u, scale, rows, in_dim, rank, out_dim, component, grouped))
PROFILE_WRAP(linear_bf16, "lora_rank_projection",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *i,
  const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint32_t rows,
  uint32_t in_dim, uint32_t out_dim),
 (gpu, o, i, w, b, rows, in_dim, out_dim))
PROFILE_WRAP(linear_add_bf16, "lora_add",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *i,
  const h3_gpu_tensor *w, float scale, uint32_t rows, uint32_t in_dim,
  uint32_t out_dim),
 (gpu, o, i, w, scale, rows, in_dim, out_dim))
PROFILE_WRAP(linear_add_qkv_component_bf16, "lora_qkv_add",
 (h3_gpu *gpu, h3_gpu_tensor *o, const h3_gpu_tensor *i,
  const h3_gpu_tensor *w, float scale, uint32_t rows, uint32_t in_dim,
  uint32_t width, uint32_t component, int grouped),
 (gpu, o, i, w, scale, rows, in_dim, width, component, grouped))

#define h3_gpu_adaln_bf16 wrap_adaln_bf16
#define h3_gpu_linear_convrot_int8_bf16 wrap_linear_convrot_int8_bf16
#define h3_gpu_qkv_rope_bf16 wrap_qkv_rope_bf16
#define h3_gpu_qkv_rope_bf16_for_sdpa wrap_qkv_rope_bf16_for_sdpa
#define h3_gpu_sdpa_bf16 wrap_sdpa_bf16
#define h3_gpu_gate_adaln_bf16 wrap_gate_adaln_bf16
#define h3_gpu_gate_bf16 wrap_gate_bf16
#define h3_gpu_swiglu_bf16 wrap_swiglu_bf16
#define h3_gpu_lora_bf16 wrap_lora_bf16
#define h3_gpu_linear_bf16 wrap_linear_bf16
#define h3_gpu_linear_add_bf16 wrap_linear_add_bf16
#define h3_gpu_linear_add_qkv_component_bf16 wrap_linear_add_qkv_component_bf16
#include "../h3_dit.c"
#undef h3_gpu_adaln_bf16
#undef h3_gpu_linear_convrot_int8_bf16
#undef h3_gpu_qkv_rope_bf16
#undef h3_gpu_qkv_rope_bf16_for_sdpa
#undef h3_gpu_sdpa_bf16
#undef h3_gpu_gate_adaln_bf16
#undef h3_gpu_gate_bf16
#undef h3_gpu_swiglu_bf16
#undef h3_gpu_lora_bf16
#undef h3_gpu_linear_bf16
#undef h3_gpu_linear_add_bf16
#undef h3_gpu_linear_add_qkv_component_bf16
#undef PROFILE_WRAP

static void die(const char *message) {
    fprintf(stderr, "bench_convrot_block: %s\n", message);
    exit(1);
}

static int parse_int(const char *value, const char *label, int *result) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || !end || *end || parsed < 0 ||
        parsed > INT32_MAX) {
        fprintf(stderr, "bench_convrot_block: invalid %s: %s\n", label,
                value);
        return 0;
    }
    *result = (int)parsed;
    return 1;
}

static int parse_float(const char *value, const char *label, float *result) {
    char *end = NULL;
    errno = 0;
    float parsed = strtof(value, &end);
    if (errno || end == value || !end || *end || !isfinite(parsed) ||
        parsed < 0.0f) {
        fprintf(stderr, "bench_convrot_block: invalid %s: %s\n", label,
                value);
        return 0;
    }
    *result = parsed;
    return 1;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int finite_bf16(const uint16_t *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        if ((values[index] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return 0;
    return 1;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s CHECKPOINT [options]\n\n"
            "options:\n"
            "  --adapter PATH          ModelTC runtime LoRA safetensors\n"
            "  --profile NAME          ModelTC adapter profile\n"
            "  --adapter-strength F    LoRA strength (default: 1)\n"
            "  --steps N               schedule evaluations (default: 16)\n"
            "  --step N                fixed forward step (default: 0)\n"
            "  --blocks N              contiguous blocks from block 0 (default: 1)\n"
            "  --small                 proxy geometry (512x256, 175 frames)\n"
            "  --width N               target width (default: 1024)\n"
            "  --height N              target height (default: 512)\n"
            "  --frames N              requested frame count (default: 243)\n"
            "  --direct-lora           disable the cached LoRA MPSGraph path\n"
            "  --ab-full-k             A/B ConvRot full-K dispatch\n"
            "  --ab-head-major         A/B ConvRot head-major SDPA layout\n"
            "  --ab-optimized          A/B both ConvRot optimizations\n"
            "  --no-check              skip full hidden BF16 repeatability reads\n"
            "  --no-profile            skip the fenced per-operation profile\n"
            "  --help                  show this message\n\n"
            "Each run is one cold pass followed by three warm passes.  The\n"
            "unfenced block keeps the production command chain; the stage\n"
            "profile submits and waits after each operation.\n",
            program);
}

static void progress(const char *phase, int completed, int total,
                     void *opaque) {
    (void)opaque;
    if (completed == 0 || completed == total)
        fprintf(stderr, "bench_convrot_block: %s %d/%d\n", phase, completed,
                total);
}

static int set_active_blocks(h3_dit *dit, unsigned active) {
    if (!dit || active > H3_DIT_BLOCKS) return 0;
    memset(dit->block_active, 0, sizeof(dit->block_active));
    for (unsigned index = 0; index < active; index++)
        dit->block_active[index] = 1;
    dit->active_block_count = active;
    dit->core_residual_ready = 0;
    return 1;
}

static int prepare_input(h3_dit *dit, char *error, size_t error_size) {
    if (!dit || dit->video_condition_rows || dit->audio_condition_rows) {
        snprintf(error, error_size,
                 "synthetic T2VA benchmark cannot include condition rows");
        return 0;
    }
    size_t video_count = (size_t)dit->video_rows * 96u;
    size_t audio_count = (size_t)dit->audio_rows * 32u;
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    if (!video || !audio) {
        free(video);
        free(audio);
        snprintf(error, error_size, "out of memory allocating synthetic input");
        return 0;
    }
    for (size_t index = 0; index < video_count; index++)
        video[index] = 0.0005f * (float)((int)(index % 31u) - 15);
    for (size_t index = 0; index < audio_count; index++)
        audio[index] = 0.0007f * (float)((int)(index % 23u) - 11);
    int ok = h3_gpu_tensor_write_f32_range(
        dit->video_input, 0, video, video_count) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input, 0, audio, audio_count);
    free(video);
    free(audio);
    if (!ok)
        snprintf(error, error_size, "cannot write synthetic input: %s",
                 h3_gpu_error(dit->gpu));
    return ok;
}

/* Rebuild only the input packing and leave the post-input hidden tensor ready
 * for run_block().  With no active layers this still exercises production's
 * patch projections and segment packing, then its final heads read but do not
 * modify hidden. */
static int prepare_hidden(h3_dit *dit, int step, char *error,
                          size_t error_size) {
    if (!set_active_blocks(dit, 0)) {
        snprintf(error, error_size, "cannot disable benchmark layers");
        return 0;
    }
    return encode_forward(dit, step, 1, 1, 1, error, error_size);
}

static int invoke_blocks(h3_dit *dit, int step, unsigned blocks, int fenced,
                         profile_context *profile, char *error,
                         size_t error_size) {
    if (!set_active_blocks(dit, blocks) || !h3_gpu_begin(dit->gpu)) {
        snprintf(error, error_size, "cannot begin %u-block command: %s",
                 blocks,
                 h3_gpu_error(dit->gpu));
        return 0;
    }
    int next_adaln = 0;
    int next_quantized = 0;
    active_profile = fenced ? profile : NULL;
    int ok = 1;
    for (unsigned index = 0; index < blocks && ok; index++) {
        int fuse_next = !getenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN") &&
                        index + 1 < blocks;
        ok = run_block(dit, index, step, &dit->blocks[index],
                       next_adaln, next_quantized, fuse_next, index + 1,
                       &next_adaln, &next_quantized, error, error_size);
    }
    active_profile = NULL;
    if (!ok) {
        h3_gpu_abort(dit->gpu);
        return 0;
    }
    if (!h3_gpu_submit(dit->gpu)) {
        snprintf(error, error_size, "cannot submit %u-block command: %s",
                 blocks,
                 h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int read_hidden_check(h3_dit *dit, uint16_t **reference,
                             size_t *reference_count, int *finite,
                             int *byte_equal) {
    size_t count = (size_t)dit->sequence * 5376u;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(dit->hidden, values, count)) {
        free(values);
        return 0;
    }
    int this_finite = finite_bf16(values, count);
    int this_equal = !*reference ||
        !memcmp(*reference, values, count * sizeof(*values));
    if (!*reference) {
        *reference = values;
        *reference_count = count;
    } else {
        free(values);
    }
    *finite = this_finite;
    *byte_equal = this_equal;
    return 1;
}

typedef struct {
    double wall_seconds;
    double cpu_seconds;
    double encode_seconds;
    double wait_seconds;
    double gpu_seconds;
} block_timing;

static uint16_t *read_hidden_values(h3_dit *dit, size_t *count, int *finite) {
    if (!dit || !count || !finite) return NULL;
    size_t value_count = (size_t)dit->sequence * 5376u;
    uint16_t *values = malloc(value_count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(dit->hidden, values, value_count)) {
        free(values);
        return NULL;
    }
    *count = value_count;
    *finite = finite_bf16(values, value_count);
    return values;
}

static const char *pass_name(unsigned pass) {
    return pass == 0 ? "cold" : pass == 1 ? "warm-1" :
           pass == 2 ? "warm-2" : "warm-3";
}

/* Time the production unfenced block prefix, with input packing outside the
 * interval.  h3_gpu_submit() waits for the command buffer, so wall time here
 * includes the complete prefix rather than merely command encoding. */
static int time_unfenced_blocks(h3_dit *dit, int step, unsigned blocks,
                                block_timing *timing, char *error,
                                size_t error_size) {
    if (!prepare_hidden(dit, step, error, error_size)) return 0;
    int wall_ok = 0;
    int cpu_ok = 0;
    double wall_start = profile_clock(CLOCK_MONOTONIC, &wall_ok);
    double cpu_start = profile_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
    h3_gpu_stats before;
    h3_gpu_stats after;
    if (!wall_ok || !cpu_ok || !h3_gpu_get_stats(dit->gpu, &before)) {
        snprintf(error, error_size, "cannot read block start statistics");
        return 0;
    }
    if (!invoke_blocks(dit, step, blocks, 0, NULL, error, error_size))
        return 0;
    double wall_end = profile_clock(CLOCK_MONOTONIC, &wall_ok);
    double cpu_end = profile_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
    if (!wall_ok || !cpu_ok || !h3_gpu_get_stats(dit->gpu, &after)) {
        snprintf(error, error_size, "cannot read block end statistics");
        return 0;
    }
    timing->wall_seconds = wall_end - wall_start;
    timing->cpu_seconds = cpu_end - cpu_start;
    timing->encode_seconds = after.command_encode_seconds -
                             before.command_encode_seconds;
    timing->wait_seconds = after.command_wait_seconds -
                           before.command_wait_seconds;
    timing->gpu_seconds = after.gpu_seconds - before.gpu_seconds;
    return 1;
}

static const char *ab_mode_name(ab_mode mode) {
    switch (mode) {
    case H3_AB_FULL_K: return "full-k";
    case H3_AB_HEAD_MAJOR: return "head-major";
    case H3_AB_OPTIMIZED: return "optimized";
    default: return "none";
    }
}

static int ab_mode_uses_full_k(ab_mode mode) {
    return mode == H3_AB_FULL_K || mode == H3_AB_OPTIMIZED;
}

static int ab_mode_uses_head_major(ab_mode mode) {
    return mode == H3_AB_HEAD_MAJOR || mode == H3_AB_OPTIMIZED;
}

static void ab_clear_environment(ab_mode mode) {
    if (ab_mode_uses_full_k(mode)) unsetenv("H3_DISABLE_CONVROT_FULL_K");
    if (ab_mode_uses_head_major(mode))
        unsetenv("H3_DISABLE_CONVROT_HEAD_MAJOR");
}

static int ab_set_variant(ab_mode mode, int reference, char *error,
                          size_t error_size) {
    if (ab_mode_uses_full_k(mode)) {
        int result = reference ?
            setenv("H3_DISABLE_CONVROT_FULL_K", "1", 1) :
            unsetenv("H3_DISABLE_CONVROT_FULL_K");
        if (result != 0) {
            snprintf(error, error_size,
                     "cannot %s ConvRot full-K dispatch: %s",
                     reference ? "disable" : "enable", strerror(errno));
            return 0;
        }
    }
    if (ab_mode_uses_head_major(mode)) {
        int result = reference ?
            setenv("H3_DISABLE_CONVROT_HEAD_MAJOR", "1", 1) :
            unsetenv("H3_DISABLE_CONVROT_HEAD_MAJOR");
        if (result != 0) {
            snprintf(error, error_size,
                     "cannot %s ConvRot head-major SDPA: %s",
                     reference ? "disable" : "enable", strerror(errno));
            return 0;
        }
    }
    return 1;
}

static void print_block_timing(const char *mode, const char *variant,
                               unsigned pass, const block_timing *timing,
                               int finite, int byte_equal) {
    printf("block ab-%s %-9s %-6s wall=%.6fs process_cpu=%.6fs "
           "encode=%.6fs wait=%.6fs gpu=%.6fs finite=%s byte_equal=%s\n",
           mode, variant, pass_name(pass), timing->wall_seconds,
           timing->cpu_seconds, timing->encode_seconds, timing->wait_seconds,
           timing->gpu_seconds, finite ? "yes" : "no",
           byte_equal ? "yes" : "no");
}

/* Run reference and candidate back-to-back for every pass.  The first pair
 * is cold AB; the warm pairs are AB, BA, AB.  This retains the requested
 * one-cold/three-warm samples while keeping one reversed pair to expose a
 * simple first-in-pair thermal or allocator bias.  The selected environment
 * is intentionally left in the candidate state before the fenced profile. */
static int run_ab(h3_dit *dit, int step, unsigned blocks, ab_mode mode,
                  char *error, size_t error_size) {
    int stable = 1;
    for (unsigned pass = 0; pass < 4; pass++) {
        block_timing reference_timing;
        block_timing candidate_timing;
        uint16_t *reference = NULL;
        uint16_t *candidate = NULL;
        size_t reference_count = 0;
        size_t candidate_count = 0;
        int reference_finite = 0;
        int candidate_finite = 0;
        int reference_first = pass != 2;

        for (unsigned pair_index = 0; pair_index < 2; pair_index++) {
            int reference_call = reference_first ? pair_index == 0 :
                                  pair_index == 1;
            block_timing *timing = reference_call ? &reference_timing :
                                                    &candidate_timing;
            uint16_t **values = reference_call ? &reference : &candidate;
            size_t *value_count = reference_call ? &reference_count :
                                                    &candidate_count;
            int *finite = reference_call ? &reference_finite :
                                           &candidate_finite;
            if (!ab_set_variant(mode, reference_call, error, error_size)) {
                free(reference);
                free(candidate);
                ab_clear_environment(mode);
                return 0;
            }
            if (!time_unfenced_blocks(dit, step, blocks, timing, error,
                                      error_size)) {
                free(reference);
                free(candidate);
                ab_clear_environment(mode);
                return 0;
            }
            *values = read_hidden_values(dit, value_count, finite);
            if (!*values) {
                free(reference);
                free(candidate);
                ab_clear_environment(mode);
                snprintf(error, error_size,
                         "cannot read ConvRot %s hidden BF16 output",
                         reference_call ? "reference" : "candidate");
                return 0;
            }
        }
        int byte_equal = reference_count == candidate_count &&
            !memcmp(reference, candidate,
                    reference_count * sizeof(*reference));
        print_block_timing(ab_mode_name(mode), "reference", pass,
                           &reference_timing, reference_finite, byte_equal);
        print_block_timing(ab_mode_name(mode), "candidate", pass,
                           &candidate_timing, candidate_finite, byte_equal);
        stable = stable && reference_finite && candidate_finite && byte_equal;
        free(reference);
        free(candidate);
    }
    ab_clear_environment(mode);
    printf("result ab-%s stable=%s\n", ab_mode_name(mode),
           stable ? "yes" : "no");
    return stable;
}

static void print_stage_totals(const profile_context *profile,
                               unsigned passes) {
    for (size_t index = 0; index < profile->stage_count; index++) {
        const profile_stage *stage = &profile->stages[index];
        double divisor = stage->calls ? (double)stage->calls : 1.0;
        double warm = passes > 1 ?
            (stage->wall_seconds - stage->pass_wall[0]) /
                (double)(passes - 1) : 0.0;
        double warm_gpu = passes > 1 ?
            (stage->gpu_seconds - stage->pass_gpu[0]) /
                (double)(passes - 1) : 0.0;
        printf("stage %-22s calls=%u cold_wall=%.6fs warm_wall=%.6fs "
               "cold_gpu=%.6fs warm_gpu=%.6fs avg_cpu=%.6fs\n",
               stage->name, stage->calls, stage->pass_wall[0], warm,
               stage->pass_gpu[0], warm_gpu, stage->cpu_seconds / divisor);
    }
    printf("stage_profile passes=%u operations=%zu\n", passes,
           profile->stage_count);
}

static int run_unfenced(h3_dit *dit, int step, unsigned blocks, int checks,
                        unsigned *warm_count, char *error,
                        size_t error_size) {
    uint16_t *reference = NULL;
    size_t reference_count = 0;
    int stable = 1;
    int finite = 0;
    unsigned completed = 0;
    for (unsigned pass = 0; pass < 4; pass++) {
        if (!prepare_hidden(dit, step, error, error_size)) break;
        int wall_ok = 0;
        int cpu_ok = 0;
        double wall_start = profile_clock(CLOCK_MONOTONIC, &wall_ok);
        double cpu_start = profile_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
        h3_gpu_stats before, after;
        h3_gpu_get_stats(dit->gpu, &before);
        int ok = invoke_blocks(dit, step, blocks, 0, NULL, error, error_size);
        double wall_end = profile_clock(CLOCK_MONOTONIC, &wall_ok);
        double cpu_end = profile_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
        h3_gpu_get_stats(dit->gpu, &after);
        if (!ok || !wall_ok || !cpu_ok) break;
        int this_finite = 1;
        int this_equal = 1;
        if (checks && !read_hidden_check(dit, &reference, &reference_count,
                                         &this_finite, &this_equal)) {
            snprintf(error, error_size, "cannot read hidden BF16 check");
            break;
        }
        finite = this_finite;
        stable = stable && this_finite && this_equal;
        printf("block unfenced %-6s wall=%.6fs process_cpu=%.6fs "
               "encode=%.6fs wait=%.6fs gpu=%.6fs finite=%s "
               "byte_equal=%s\n", pass ? pass == 1 ? "warm-1" :
               pass == 2 ? "warm-2" : "warm-3" : "cold",
               wall_end - wall_start, cpu_end - cpu_start,
               after.command_encode_seconds - before.command_encode_seconds,
               after.command_wait_seconds - before.command_wait_seconds,
               after.gpu_seconds - before.gpu_seconds,
               finite ? "yes" : "no", this_equal ? "yes" : "no");
        if (pass) completed++;
    }
    free(reference);
    *warm_count = completed;
    return completed == 3 && stable && finite;
}

static int run_fenced(h3_dit *dit, int step, unsigned blocks, int checks,
                      char *error, size_t error_size) {
    profile_context profile;
    memset(&profile, 0, sizeof(profile));
    profile.gpu = dit->gpu;
    uint16_t *reference = NULL;
    size_t reference_count = 0;
    int stable = 1;
    int finite = 0;
    unsigned completed = 0;
    for (unsigned pass = 0; pass < 4; pass++) {
        if (!prepare_hidden(dit, step, error, error_size)) break;
        profile.pass = pass;
        int ok = invoke_blocks(dit, step, blocks, 1, &profile, error,
                               error_size);
        if (!ok) break;
        int this_finite = 1;
        int this_equal = 1;
        if (checks && !read_hidden_check(dit, &reference, &reference_count,
                                         &this_finite, &this_equal)) {
            snprintf(error, error_size, "cannot read hidden BF16 check");
            break;
        }
        finite = this_finite;
        stable = stable && this_finite && this_equal;
        printf("block fenced   %-6s finite=%s byte_equal=%s\n",
               pass ? pass == 1 ? "warm-1" : pass == 2 ? "warm-2" :
               "warm-3" : "cold", finite ? "yes" : "no",
               this_equal ? "yes" : "no");
        if (pass) completed++;
    }
    free(reference);
    print_stage_totals(&profile, completed + (completed < 3 ? 0u : 1u));
    return completed == 3 && stable && finite && !profile.failed;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage(argv[0]);
        return 0;
    }

    const char *checkpoint = argv[1];
    const char *adapter_path = NULL;
    h3_adapter_profile profile = H3_ADAPTER_PROFILE_NONE;
    float adapter_strength = 1.0f;
    int steps = H3_PROFILE_DEFAULT_STEPS;
    int steps_given = 0;
    int step = 0;
    int blocks = 1;
    int width = H3_PROFILE_DEFAULT_WIDTH;
    int height = H3_PROFILE_DEFAULT_HEIGHT;
    int frames = H3_PROFILE_DEFAULT_FRAMES;
    int checks = 1;
    int direct_lora = 0;
    int profile_enabled = 1;
    ab_mode ab = H3_AB_NONE;

    for (int index = 2; index < argc; index++) {
        const char *argument = argv[index];
        if (!strcmp(argument, "--help") || !strcmp(argument, "-h")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(argument, "--small")) {
            width = H3_PROFILE_SMALL_WIDTH;
            height = H3_PROFILE_SMALL_HEIGHT;
            frames = H3_PROFILE_SMALL_FRAMES;
        } else if (!strcmp(argument, "--adapter")) {
            if (++index >= argc) return 2;
            adapter_path = argv[index];
        } else if (!strcmp(argument, "--profile")) {
            if (++index >= argc || !h3_adapter_profile_parse(argv[index],
                                                               &profile)) {
                fprintf(stderr, "bench_convrot_block: invalid adapter profile\n");
                return 2;
            }
        } else if (!strcmp(argument, "--adapter-strength")) {
            if (++index >= argc || !parse_float(argv[index],
                                                "adapter strength",
                                                &adapter_strength)) return 2;
        } else if (!strcmp(argument, "--steps")) {
            if (++index >= argc || !parse_int(argv[index], "steps", &steps))
                return 2;
            steps_given = 1;
        } else if (!strcmp(argument, "--step")) {
            if (++index >= argc || !parse_int(argv[index], "step", &step))
                return 2;
        } else if (!strcmp(argument, "--blocks")) {
            if (++index >= argc || !parse_int(argv[index], "blocks", &blocks))
                return 2;
        } else if (!strcmp(argument, "--width")) {
            if (++index >= argc || !parse_int(argv[index], "width", &width))
                return 2;
        } else if (!strcmp(argument, "--height")) {
            if (++index >= argc || !parse_int(argv[index], "height", &height))
                return 2;
        } else if (!strcmp(argument, "--frames")) {
            if (++index >= argc || !parse_int(argv[index], "frames", &frames))
                return 2;
        } else if (!strcmp(argument, "--direct-lora")) {
            direct_lora = 1;
        } else if (!strcmp(argument, "--ab-full-k")) {
            if (ab != H3_AB_NONE) {
                fputs("bench_convrot_block: choose only one --ab-* mode\n",
                      stderr);
                return 2;
            }
            ab = H3_AB_FULL_K;
        } else if (!strcmp(argument, "--ab-head-major")) {
            if (ab != H3_AB_NONE) {
                fputs("bench_convrot_block: choose only one --ab-* mode\n",
                      stderr);
                return 2;
            }
            ab = H3_AB_HEAD_MAJOR;
        } else if (!strcmp(argument, "--ab-optimized")) {
            if (ab != H3_AB_NONE) {
                fputs("bench_convrot_block: choose only one --ab-* mode\n",
                      stderr);
                return 2;
            }
            ab = H3_AB_OPTIMIZED;
        } else if (!strcmp(argument, "--no-check")) {
            checks = 0;
        } else if (!strcmp(argument, "--no-profile")) {
            profile_enabled = 0;
        } else {
            fprintf(stderr, "bench_convrot_block: unknown option: %s\n",
                    argument);
            usage(argv[0]);
            return 2;
        }
    }

    if ((adapter_path && profile == H3_ADAPTER_PROFILE_NONE) ||
        (!adapter_path && profile != H3_ADAPTER_PROFILE_NONE)) {
        fputs("bench_convrot_block: --adapter and --profile must be supplied "
              "together\n", stderr);
        return 2;
    }
    if (profile != H3_ADAPTER_PROFILE_NONE &&
        !h3_adapter_profile_is_modeltc(profile)) {
        fputs("bench_convrot_block: only ModelTC profiles are supported\n",
              stderr);
        return 2;
    }
    if (profile == H3_ADAPTER_PROFILE_NONE && adapter_strength != 1.0f) {
        fputs("bench_convrot_block: --adapter-strength requires --adapter "
              "and --profile\n", stderr);
        return 2;
    }
    if (profile != H3_ADAPTER_PROFILE_NONE && !steps_given)
        steps = h3_adapter_profile_default_steps(profile);
    if (steps < 2 || steps > H3_MAX_STEPS || step < 0 || step >= steps ||
        blocks < 1 || (unsigned)blocks > H3_DIT_BLOCKS || width < 32 ||
        height < 32 ||
        frames < 5) {
        fputs("bench_convrot_block: invalid geometry, step, or schedule\n",
              stderr);
        return 2;
    }
    if (direct_lora) setenv("H3_DISABLE_LORA_MPSGRAPH", "1", 1);

    char error[1024] = {0};
    int latent_w = width / H3_VAE_SPATIAL_RATIO;
    int latent_h = height / H3_VAE_SPATIAL_RATIO;
    h3_temporal_shape temporal = h3_temporal(frames);
    h3_layout_spec spec = {
        H3_PROFILE_TEXT_TOKENS, temporal.video_t, latent_h, latent_w,
        temporal.audio_t, temporal.frame_count, NULL, 0, NULL, 0
    };
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) die(error);

    size_t text_count = (size_t)H3_PROFILE_TEXT_TOKENS *
        H3_PROFILE_TEXT_WIDTH;
    uint16_t *text_values = malloc(text_count * sizeof(*text_values));
    if (!text_values) die("out of memory allocating synthetic text");
    for (size_t index = 0; index < text_count; index++) {
        float magnitude = 0.001f * (float)(1 + index % 17u);
        text_values[index] = f32_to_bf16((index & 1u) ? -magnitude : magnitude);
    }
    h3_text_embedding text = {
        .tokens = H3_PROFILE_TEXT_TOKENS,
        .width = H3_PROFILE_TEXT_WIDTH,
        .values = text_values,
        .tags = NULL
    };
    h3_sigma_schedule sigmas;
    h3_params adapter_params = H3_PARAMS_DEFAULT;
    const h3_params *adapter_params_ptr = NULL;
    if (profile != H3_ADAPTER_PROFILE_NONE) {
        adapter_params.adapter_path = adapter_path;
        adapter_params.adapter_profile = profile;
        adapter_params.adapter_strength = adapter_strength;
        adapter_params.steps = steps;
        if (!h3_adapter_profile_apply(&adapter_params, error, sizeof(error)))
            die(error);
        adapter_params_ptr = &adapter_params;
    }
    int schedule_ok = adapter_params_ptr &&
        adapter_params.scheduler == H3_SCHEDULER_SIMPLE ?
        h3_serving_schedule_build_shifted(
            steps, adapter_params.video_shift, adapter_params.audio_shift,
            &sigmas) : h3_beta_schedule_build_shifted(
                steps, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT, &sigmas);
    if (!schedule_ok) die("cannot build sigma schedule");

    printf("config checkpoint=%s geometry=%dx%d/%dframes latent=%dx%dx%d "
           "audio=%d sequence=%zu target=%s steps=%d step=%d checks=%s "
           "lora=%s\n", checkpoint, width, height, temporal.frame_count,
           temporal.video_t, latent_h, latent_w, temporal.audio_t,
           layout.seq_len, layout.seq_len == 37680 ? "37680" :
           layout.seq_len == 7246 ? "7246" : "custom", steps, step,
           checks ? "on" : "off", direct_lora ? "direct" : "mpsgraph");
    printf("config blocks=%d profile=%s\n", blocks,
           profile_enabled ? "fenced" : "off");
    if (ab != H3_AB_NONE)
        printf("config ab-%s=reference(disabled),candidate(enabled),"
               "hidden_compare=forced,order=cold-AB/warm-AB-BA-AB\n",
               ab_mode_name(ab));
    if (profile != H3_ADAPTER_PROFILE_NONE)
        printf("config adapter=%s profile=%s strength=%.7g\n", adapter_path,
               h3_adapter_profile_name(profile), adapter_strength);

    h3_dit *dit = h3_dit_load_t2va(
        checkpoint, "h3_shaders.metal", &text, &layout, &sigmas,
        H3_DIT_BLOCKS, 1, 0, 0, 1.0f,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        adapter_params_ptr, progress, NULL, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "bench_convrot_block: DiT load failed: %s\n", error);
        free(text_values);
        h3_layout_free(&layout);
        return 1;
    }
    printf("loaded path=%s sequence=%u video_rows=%u audio_rows=%u "
           "active_blocks=%u\n", dit->convrot ? "convrot" : "dense",
           dit->sequence, dit->video_rows, dit->audio_rows,
           dit->active_block_count);
    if (!dit->convrot)
        fputs("warning: checkpoint selected the dense run_block path; this "
              "harness is intended for ConvRot checkpoints\n", stderr);

    int ok = prepare_input(dit, error, sizeof(error));
    if (ok) {
        if (ab != H3_AB_NONE) {
            ok = run_ab(dit, step, (unsigned)blocks, ab, error,
                        sizeof(error));
        } else {
            unsigned warm_count = 0;
            ok = run_unfenced(dit, step, (unsigned)blocks, checks,
                              &warm_count, error, sizeof(error));
            printf("result unfenced warm_forwards=%u stable=%s\n", warm_count,
                   ok ? "yes" : "no");
        }
    }
    if (ok && profile_enabled) {
        ok = run_fenced(dit, step, (unsigned)blocks, checks, error,
                        sizeof(error));
        printf("result fenced stable=%s\n", ok ? "yes" : "no");
    }
    if (!ok) fprintf(stderr, "bench_convrot_block: %s\n", error);
    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(text_values);
    return ok ? 0 : 1;
}
