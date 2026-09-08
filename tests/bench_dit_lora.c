/*
 * Native DiT/LoRA timing probe.
 *
 * Compile from projects/h3/code after building libh3.a, for example:
 *
 *   clang -std=c11 -O3 -D_DARWIN_C_SOURCE -I. \
 *     tests/bench_dit_lora.c libh3.a \
 *     -framework Foundation -framework Metal \
 *     -framework MetalPerformanceShaders \
 *     -framework MetalPerformanceShadersGraph -framework Accelerate \
 *     -licucore -lm -o /tmp/bench_dit_lora
 *
 * The checkpoint may be one safetensors file or a directory of shards.  The
 * default fixture is the 512x256, 175-frame T2VA geometry.  --large selects
 * the 1024x512, 243-frame geometry.  This probe deliberately does not set
 * H3_DISABLE_LORA_MPSGRAPH; set that environment variable externally to 1
 * when measuring the direct LoRA path.
 */
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif

#include "h3_adapter.h"
#include "h3_dit.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TEXT_TOKENS = 6,
    TEXT_WIDTH = 5120,
    DEFAULT_FRAMES = 175,
    LARGE_FRAMES = 243,
    DEFAULT_WIDTH = 512,
    DEFAULT_HEIGHT = 256,
    LARGE_WIDTH = 1024,
    LARGE_HEIGHT = 512,
    DEFAULT_STEPS = 20
};

typedef struct {
    double wall_seconds;
    double cpu_seconds;
    h3_gpu_stats before;
    h3_gpu_stats after;
} forward_timing;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s CHECKPOINT [options]\n\n"
            "options:\n"
            "  --adapter PATH          ModelTC runtime LoRA safetensors\n"
            "  --profile NAME          ModelTC adapter profile\n"
            "  --adapter-profile NAME  Alias for --profile\n"
            "  --adapter-strength F    LoRA strength (default: 1)\n"
            "  --steps N               Schedule evaluations (default: 20,\n"
            "                          profile nominal count when adapted)\n"
            "  --step N                Fixed zero-based forward step (default: 0)\n"
            "  --large                 Use 1024x512 and 243 frames\n"
            "  --dump PREFIX           Write PREFIX.video.f32 and\n"
            "                          PREFIX.audio.f32 after the warm forwards\n"
            "  --help                  Show this message\n\n"
            "The default is one cold forward followed by exactly three warm\n"
            "forwards. Set H3_DISABLE_LORA_MPSGRAPH=1 externally to select\n"
            "the direct LoRA implementation; the default selects MPSGraph.\n",
            program);
}

static void die(const char *message) {
    fprintf(stderr, "bench_dit_lora: %s\n", message);
    exit(1);
}

static int parse_nonnegative_int(const char *value, const char *label,
                                 int *result) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || !end || *end || parsed < 0 ||
        parsed > INT32_MAX) {
        fprintf(stderr, "bench_dit_lora: invalid %s: %s\n", label, value);
        return 0;
    }
    *result = (int)parsed;
    return 1;
}

static int parse_nonnegative_float(const char *value, const char *label,
                                   float *result) {
    char *end = NULL;
    errno = 0;
    float parsed = strtof(value, &end);
    if (errno || end == value || !end || *end || !isfinite(parsed) ||
        parsed < 0.0f) {
        fprintf(stderr, "bench_dit_lora: invalid %s: %s\n", label, value);
        return 0;
    }
    *result = parsed;
    return 1;
}

static double clock_seconds(clockid_t clock_id, int *ok) {
    struct timespec value;
    if (clock_gettime(clock_id, &value) != 0) {
        if (ok) *ok = 0;
        return 0.0;
    }
    if (ok) *ok = 1;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int finite_array(const float *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        if (!isfinite(values[index])) return 0;
    return 1;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const uint8_t *values = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void progress(const char *phase, int completed, int total,
                     void *opaque) {
    (void)opaque;
    if (completed == 0 || completed == total)
        fprintf(stderr, "bench_dit_lora: %s %d/%d\n",
                phase, completed, total);
}

static int get_forward_timing(h3_dit *dit, int step,
                              const float *video, const float *audio,
                              float *video_output, float *audio_output,
                              forward_timing *timing,
                              char *error, size_t error_size) {
    if (!timing || !h3_dit_get_gpu_stats(dit, &timing->before)) {
        snprintf(error, error_size, "cannot read initial GPU statistics");
        return 0;
    }
    int wall_ok = 0, cpu_ok = 0;
    double wall_start = clock_seconds(CLOCK_MONOTONIC, &wall_ok);
    double cpu_start = clock_seconds(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
    if (!wall_ok || !cpu_ok) {
        snprintf(error, error_size, "cannot read benchmark clocks");
        return 0;
    }
    int ok = h3_dit_forward(dit, step, video, audio,
                            video_output, audio_output,
                            error, error_size);
    double wall_end = clock_seconds(CLOCK_MONOTONIC, &wall_ok);
    double cpu_end = clock_seconds(CLOCK_PROCESS_CPUTIME_ID, &cpu_ok);
    if (!wall_ok || !cpu_ok) {
        snprintf(error, error_size, "cannot read benchmark clocks after forward");
        return 0;
    }
    if (!ok) return 0;
    if (!h3_dit_get_gpu_stats(dit, &timing->after)) {
        snprintf(error, error_size, "cannot read final GPU statistics");
        return 0;
    }
    timing->wall_seconds = wall_end - wall_start;
    timing->cpu_seconds = cpu_end - cpu_start;
    return 1;
}

static int dump_array(const char *prefix, const char *suffix,
                      const float *values, size_t count,
                      char *error, size_t error_size) {
    size_t prefix_length = strlen(prefix);
    size_t suffix_length = strlen(suffix);
    if (prefix_length > SIZE_MAX - suffix_length - 1) {
        snprintf(error, error_size, "dump path is too long");
        return 0;
    }
    char *path = malloc(prefix_length + suffix_length + 1);
    if (!path) {
        snprintf(error, error_size, "out of memory creating dump path");
        return 0;
    }
    memcpy(path, prefix, prefix_length);
    memcpy(path + prefix_length, suffix, suffix_length + 1);
    FILE *stream = fopen(path, "wb");
    if (!stream) {
        snprintf(error, error_size, "cannot open dump path %s: %s",
                 path, strerror(errno));
        free(path);
        return 0;
    }
    size_t written = fwrite(values, sizeof(*values), count, stream);
    int close_ok = fclose(stream) == 0;
    if (written != count || !close_ok) {
        snprintf(error, error_size, "cannot write dump path %s", path);
        free(path);
        return 0;
    }
    printf("dump %s (%zu F32 values)\n", path, count);
    free(path);
    return 1;
}

static void print_timing(const char *label, const forward_timing *timing,
                         int finite, int byte_equal,
                         size_t video_count, size_t audio_count,
                         const float *video, const float *audio) {
    double encode = timing->after.command_encode_seconds -
        timing->before.command_encode_seconds;
    double wait = timing->after.command_wait_seconds -
        timing->before.command_wait_seconds;
    double gpu = timing->after.gpu_seconds - timing->before.gpu_seconds;
    uint64_t video_hash = hash_bytes(video, video_count * sizeof(*video));
    uint64_t audio_hash = hash_bytes(audio, audio_count * sizeof(*audio));
    printf("forward %-6s wall=%.6fs process_cpu=%.6fs encode=%.6fs wait=%.6fs "
           "gpu=%.6fs finite=%s byte_equal=%s "
           "video_hash=%016" PRIx64 " audio_hash=%016" PRIx64 "\n",
           label, timing->wall_seconds, timing->cpu_seconds,
           encode, wait, gpu, finite ? "yes" : "no",
           byte_equal ? "yes" : "no", video_hash, audio_hash);
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
    const char *dump_prefix = NULL;
    h3_adapter_profile profile = H3_ADAPTER_PROFILE_NONE;
    float adapter_strength = 1.0f;
    int large = 0;
    int steps = DEFAULT_STEPS;
    int steps_given = 0;
    int step = 0;

    for (int index = 2; index < argc; index++) {
        const char *argument = argv[index];
        if (!strcmp(argument, "--help") || !strcmp(argument, "-h")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(argument, "--large")) {
            large = 1;
        } else if (!strcmp(argument, "--adapter") ||
                   !strcmp(argument, "--adapter-path")) {
            if (++index >= argc) {
                fprintf(stderr, "bench_dit_lora: %s requires a path\n",
                        argument);
                return 2;
            }
            adapter_path = argv[index];
        } else if (!strcmp(argument, "--profile") ||
                   !strcmp(argument, "--adapter-profile")) {
            if (++index >= argc) {
                fprintf(stderr, "bench_dit_lora: %s requires a profile\n",
                        argument);
                return 2;
            }
            if (!h3_adapter_profile_parse(argv[index], &profile)) {
                fprintf(stderr, "bench_dit_lora: unknown adapter profile: %s\n",
                        argv[index]);
                return 2;
            }
        } else if (!strcmp(argument, "--adapter-strength")) {
            if (++index >= argc || !parse_nonnegative_float(
                    argv[index], "adapter strength", &adapter_strength))
                return 2;
        } else if (!strcmp(argument, "--steps")) {
            if (++index >= argc || !parse_nonnegative_int(
                    argv[index], "steps", &steps))
                return 2;
            steps_given = 1;
        } else if (!strcmp(argument, "--step")) {
            if (++index >= argc || !parse_nonnegative_int(
                    argv[index], "step", &step))
                return 2;
        } else if (!strcmp(argument, "--dump")) {
            if (++index >= argc || !*argv[index]) {
                fprintf(stderr, "bench_dit_lora: --dump requires a prefix\n");
                return 2;
            }
            dump_prefix = argv[index];
        } else {
            fprintf(stderr, "bench_dit_lora: unknown option: %s\n", argument);
            usage(argv[0]);
            return 2;
        }
    }

    if ((adapter_path && profile == H3_ADAPTER_PROFILE_NONE) ||
        (!adapter_path && profile != H3_ADAPTER_PROFILE_NONE)) {
        fputs("bench_dit_lora: --adapter and --profile must be supplied "
              "together\n", stderr);
        return 2;
    }
    if (profile != H3_ADAPTER_PROFILE_NONE &&
        !h3_adapter_profile_is_modeltc(profile)) {
        fputs("bench_dit_lora: only ModelTC profiles are supported\n", stderr);
        return 2;
    }
    if (profile == H3_ADAPTER_PROFILE_NONE && adapter_strength != 1.0f) {
        fputs("bench_dit_lora: --adapter-strength requires --adapter and "
              "--profile\n", stderr);
        return 2;
    }
    if (profile != H3_ADAPTER_PROFILE_NONE && !steps_given)
        steps = h3_adapter_profile_default_steps(profile);
    if (steps < 2 || steps > H3_MAX_STEPS) {
        fprintf(stderr, "bench_dit_lora: steps must be in [2, %d]\n",
                H3_MAX_STEPS);
        return 2;
    }
    if (step < 0 || step >= steps) {
        fprintf(stderr, "bench_dit_lora: step must be in [0, %d]\n",
                steps - 1);
        return 2;
    }

    h3_params adapter_params = H3_PARAMS_DEFAULT;
    const h3_params *adapter_params_ptr = NULL;
    char error[512] = {0};
    if (profile != H3_ADAPTER_PROFILE_NONE) {
        adapter_params.adapter_path = adapter_path;
        adapter_params.adapter_profile = profile;
        adapter_params.adapter_strength = adapter_strength;
        adapter_params.steps = steps;
        if (!h3_adapter_profile_apply(&adapter_params, error, sizeof(error))) {
            fprintf(stderr, "bench_dit_lora: invalid adapter parameters: %s\n",
                    error);
            return 2;
        }
        adapter_params_ptr = &adapter_params;
    }

    int requested_frames = large ? LARGE_FRAMES : DEFAULT_FRAMES;
    int width = large ? LARGE_WIDTH : DEFAULT_WIDTH;
    int height = large ? LARGE_HEIGHT : DEFAULT_HEIGHT;
    h3_temporal_shape temporal = h3_temporal(requested_frames);
    int latent_w = width / H3_VAE_SPATIAL_RATIO;
    int latent_h = height / H3_VAE_SPATIAL_RATIO;
    h3_layout_spec spec = {
        TEXT_TOKENS, temporal.video_t, latent_h, latent_w,
        temporal.audio_t, temporal.frame_count, NULL, 0, NULL, 0
    };
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "bench_dit_lora: cannot build layout: %s\n", error);
        return 1;
    }

    size_t text_count = (size_t)TEXT_TOKENS * TEXT_WIDTH;
    uint16_t *text_values = malloc(text_count * sizeof(*text_values));
    if (!text_values) die("out of memory allocating synthetic text");
    for (size_t index = 0; index < text_count; index++) {
        float magnitude = 0.001f * (float)(1 + index % 17);
        float value = (index & 1u) ? -magnitude : magnitude;
        text_values[index] = f32_to_bf16(value);
    }
    h3_text_embedding text = {
        .tokens = TEXT_TOKENS,
        .width = TEXT_WIDTH,
        .values = text_values,
        .tags = NULL
    };

    const char *disable_graph = getenv("H3_DISABLE_LORA_MPSGRAPH");
    int graph_enabled = !(disable_graph && *disable_graph &&
                          strcmp(disable_graph, "0"));
    printf("config checkpoint=%s geometry=%dx%d/%dframes latent=%dx%dx%d "
           "audio=%d steps=%d fixed_step=%d blocks=50 lora_mpsgraph=%s\n",
           checkpoint, width, height, temporal.frame_count,
           temporal.video_t, latent_h, latent_w, temporal.audio_t,
           steps, step, graph_enabled ? "on" : "off");
    if (profile != H3_ADAPTER_PROFILE_NONE)
        printf("config adapter=%s profile=%s strength=%.7g\n",
               adapter_path, h3_adapter_profile_name(profile), adapter_strength);

    h3_sigma_schedule sigmas;
    int schedule_ok = adapter_params_ptr &&
        adapter_params.scheduler == H3_SCHEDULER_SIMPLE ?
        h3_serving_schedule_build_shifted(
            steps, adapter_params.video_shift, adapter_params.audio_shift,
            &sigmas) :
        h3_beta_schedule_build_shifted(
            steps, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT, &sigmas);
    if (!schedule_ok) {
        fputs("bench_dit_lora: cannot build DiT sigma schedule\n", stderr);
        free(text_values);
        h3_layout_free(&layout);
        return 1;
    }

    double load_start = clock_seconds(CLOCK_MONOTONIC, NULL);
    h3_dit *dit = h3_dit_load_t2va(
        checkpoint, "h3_shaders.metal", &text, &layout, &sigmas,
        50, 1, 0, 0, 1.0f,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        adapter_params_ptr, progress, NULL, error, sizeof(error));
    double load_seconds = clock_seconds(CLOCK_MONOTONIC, NULL) - load_start;
    if (!dit) {
        fprintf(stderr, "bench_dit_lora: DiT load failed: %s\n", error);
        free(text_values);
        h3_layout_free(&layout);
        return 1;
    }

    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    float *video_output = malloc(video_count * sizeof(*video_output));
    float *audio_output = malloc(audio_count * sizeof(*audio_output));
    float *reference_video = malloc(video_count * sizeof(*reference_video));
    float *reference_audio = malloc(audio_count * sizeof(*reference_audio));
    if (!video || !audio || !video_output || !audio_output ||
        !reference_video || !reference_audio)
        die("out of memory allocating synthetic latent buffers");
    for (size_t index = 0; index < video_count; index++) {
        float magnitude = 0.002f * (float)(1 + index % 29);
        video[index] = (index & 1u) ? -magnitude : magnitude;
    }
    for (size_t index = 0; index < audio_count; index++) {
        float magnitude = 0.003f * (float)(1 + index % 13);
        audio[index] = (index & 1u) ? -magnitude : magnitude;
    }
    if (!finite_array(video, video_count) || !finite_array(audio, audio_count))
        die("synthetic latent values are not finite");

    forward_timing timing;
    int finite = 0;
    int stable = 1;
    if (!get_forward_timing(dit, step, video, audio, video_output,
                            audio_output, &timing, error, sizeof(error))) {
        fprintf(stderr, "bench_dit_lora: cold forward failed: %s\n", error);
        stable = 0;
    } else {
        finite = finite_array(video_output, video_count) &&
            finite_array(audio_output, audio_count);
        if (!finite) stable = 0;
        if (finite) {
            memcpy(reference_video, video_output,
                   video_count * sizeof(*reference_video));
            memcpy(reference_audio, audio_output,
                   audio_count * sizeof(*reference_audio));
        }
        print_timing("cold", &timing, finite, 1,
                     video_count, audio_count, video_output, audio_output);
    }
    int warm_count = 0;
    for (int warm = 0; stable && warm < 3; warm++) {
        if (!get_forward_timing(dit, step, video, audio, video_output,
                                audio_output, &timing, error, sizeof(error))) {
            fprintf(stderr, "bench_dit_lora: warm-%d forward failed: %s\n",
                    warm + 1, error);
            stable = 0;
            break;
        }
        finite = finite_array(video_output, video_count) &&
            finite_array(audio_output, audio_count);
        int byte_equal = finite &&
            !memcmp(reference_video, video_output,
                    video_count * sizeof(*reference_video)) &&
            !memcmp(reference_audio, audio_output,
                    audio_count * sizeof(*reference_audio));
        print_timing(warm == 0 ? "warm-1" : warm == 1 ? "warm-2" : "warm-3",
                     &timing, finite, byte_equal,
                     video_count, audio_count, video_output, audio_output);
        warm_count++;
        if (!finite || !byte_equal) stable = 0;
    }

    if (stable && dump_prefix &&
        (!dump_array(dump_prefix, ".video.f32", video_output, video_count,
                     error, sizeof(error)) ||
         !dump_array(dump_prefix, ".audio.f32", audio_output, audio_count,
                     error, sizeof(error)))) {
        fprintf(stderr, "bench_dit_lora: %s\n", error);
        stable = 0;
    }
    printf("result load=%.3fs warm_forwards=%d finite=%s byte_stable=%s\n",
           load_seconds, warm_count, finite ? "yes" : "no", stable ? "yes" : "no");

    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(text_values);
    free(video);
    free(audio);
    free(video_output);
    free(audio_output);
    free(reference_video);
    free(reference_audio);
    return stable && finite ? 0 : 1;
}
