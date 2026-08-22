#include "h3_video_encoder.h"

#include "h3_weights.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RGB_CHANNELS = 3,
    LATENT_CHANNELS = 24,
    MOMENT_CHANNELS = 48,
    LEVELS = 6,
    BLOCKS = 2,
    GROUPS = 32,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

static const uint32_t level_channels[LEVELS] = {128, 256, 256, 512, 512, 1024};
static const uint32_t space_strides[LEVELS] = {2, 2, 2, 2, 1, 1};
static const uint32_t time_strides[LEVELS] = {1, 2, 2, 1, 1, 1};

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride_t;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t depth_front;
    uint32_t height_before;
    uint32_t height_after;
    uint32_t width_before;
    uint32_t width_after;
} encoder_conv;

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
} encoder_norm;

typedef struct {
    encoder_norm norm1;
    encoder_conv conv1;
    encoder_norm norm2;
    encoder_conv conv2;
    encoder_conv shortcut;
    int has_shortcut;
} encoder_block;

typedef struct {
    encoder_block blocks[BLOCKS];
    encoder_conv downsample;
    int has_downsample;
} encoder_level;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *store;
    int fp16;
    encoder_conv conv_in;
    encoder_level levels[LEVELS];
    encoder_norm norm_out;
    encoder_conv conv_out;
    encoder_conv quant;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
    h3_gpu_tensor *latent_mean_gpu;
    h3_gpu_tensor *latent_std_gpu;
} encoder_context;

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

/* A tile forward is encoded into one Metal command buffer.  Scratch buffers
 * therefore have to remain resident until that command buffer completes; the
 * old per-layer submit made immediate frees safe, but also serialized every
 * convolution on the host. */
typedef struct {
    h3_gpu_tensor **items;
    size_t count;
    size_t capacity;
} tensor_arena;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(encoder_context *encoder, int ok, char *error,
                  size_t error_size, const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(encoder->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void tensor_arena_free(tensor_arena *arena) {
    if (!arena) return;
    for (size_t index = 0; index < arena->count; index++)
        h3_gpu_tensor_free(arena->items[index]);
    free(arena->items);
    memset(arena, 0, sizeof(*arena));
}

static h3_gpu_tensor *tensor_arena_own(tensor_arena *arena,
                                       h3_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    if (arena->count == arena->capacity) {
        size_t next = arena->capacity ? arena->capacity * 2 : 64;
        h3_gpu_tensor **grown = realloc(arena->items,
                                        next * sizeof(*grown));
        if (!grown) {
            h3_gpu_tensor_free(tensor);
            return NULL;
        }
        arena->items = grown;
        arena->capacity = next;
    }
    arena->items[arena->count++] = tensor;
    return tensor;
}

static h3_gpu_tensor *tensor_arena_new_activation(encoder_context *encoder,
                                                   tensor_arena *arena,
                                                   size_t elements) {
    h3_gpu_tensor *tensor = encoder->fp16 ?
        h3_gpu_tensor_new_f16(encoder->gpu, elements) :
        h3_gpu_tensor_new_f32(encoder->gpu, elements);
    return tensor_arena_own(arena, tensor);
}

static int configured_fp16(void) {
    const char *value = getenv("H3_VIDEO_ENCODER_FP16");
    return !value || !*value || strcmp(value, "0");
}

static void free_conv(encoder_conv *conv) {
    free_tensor(&conv->weight);
    free_tensor(&conv->bias);
}

static void free_norm(encoder_norm *norm) {
    free_tensor(&norm->weight);
    free_tensor(&norm->bias);
}

static void cleanup(encoder_context *encoder) {
    if (!encoder) return;
    free_conv(&encoder->conv_in);
    for (int level = 0; level < LEVELS; level++) {
        for (int block = 0; block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            free_norm(&item->norm1);
            free_conv(&item->conv1);
            free_norm(&item->norm2);
            free_conv(&item->conv2);
            free_conv(&item->shortcut);
        }
        free_conv(&encoder->levels[level].downsample);
    }
    free_norm(&encoder->norm_out);
    free_conv(&encoder->conv_out);
    free_conv(&encoder->quant);
    h3_gpu_tensor_free(encoder->latent_mean_gpu);
    h3_gpu_tensor_free(encoder->latent_std_gpu);
    h3_weight_store_free(encoder->store);
    h3_gpu_free(encoder->gpu);
    memset(encoder, 0, sizeof(*encoder));
}

static h3_gpu_tensor *load_f32(encoder_context *encoder, const char *name,
                               int ndim, const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(encoder->store, encoder->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(encoder_context *encoder, const char *name,
                         uint64_t width, char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(encoder, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f5(encoder_context *encoder, const char *name,
                         uint64_t output_channels, uint64_t input_channels,
                         uint64_t kernel, char *error, size_t error_size) {
    uint64_t shape[] = {
        output_channels, input_channels, kernel, kernel, kernel
    };
    return load_f32(encoder, name, 5, shape, error, error_size);
}

static int load_conv(encoder_context *encoder, encoder_conv *conv,
                     const char *prefix, uint32_t input_channels,
                     uint32_t output_channels, uint32_t kernel,
                     uint32_t stride_t, uint32_t stride_h,
                     uint32_t stride_w, uint32_t depth_front,
                     uint32_t height_before, uint32_t height_after,
                     uint32_t width_before, uint32_t width_after,
                     char *error, size_t error_size) {
    char name[192];
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->stride_t = stride_t;
    conv->stride_h = stride_h;
    conv->stride_w = stride_w;
    conv->depth_front = depth_front;
    conv->height_before = height_before;
    conv->height_after = height_after;
    conv->width_before = width_before;
    conv->width_after = width_after;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = f5(encoder, name, output_channels, input_channels, kernel,
                      error, error_size);
    if (!conv->weight) return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    conv->bias = f1(encoder, name, output_channels, error, error_size);
    return conv->bias != NULL;
}

static int load_norm(encoder_context *encoder, encoder_norm *norm,
                     const char *prefix, uint32_t channels,
                     char *error, size_t error_size) {
    char name[192];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    norm->weight = f1(encoder, name, channels, error, error_size);
    if (!norm->weight) return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    norm->bias = f1(encoder, name, channels, error, error_size);
    return norm->bias != NULL;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_normalization(encoder_context *encoder,
                              const char *weight_directory,
                              char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open video VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", encoder->latent_mean,
                               LATENT_CHANNELS, error, error_size) &&
             parse_float_array(json, "latents_std", encoder->latent_std,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        if (encoder->latent_std[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent deviation is invalid");
            return 0;
        }
    if (ok) {
        encoder->latent_mean_gpu = h3_gpu_tensor_from_f32(
            encoder->gpu, encoder->latent_mean, LATENT_CHANNELS);
        encoder->latent_std_gpu = h3_gpu_tensor_from_f32(
            encoder->gpu, encoder->latent_std, LATENT_CHANNELS);
        if (!encoder->latent_mean_gpu || !encoder->latent_std_gpu) {
            fail(error, error_size, "cannot upload video VAE latent normalization");
            return 0;
        }
    }
    return ok;
}

static int load_weights(encoder_context *encoder, char *error,
                        size_t error_size) {
    if (!load_conv(encoder, &encoder->conv_in, "encoder.conv_in", 3, 128, 3,
                   1, 1, 1, 2, 1, 1, 1, 1, error, error_size)) return 0;
    uint32_t previous = 128;
    for (int level = 0; level < LEVELS; level++) {
        uint32_t channels = level_channels[level];
        for (int block = 0; block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            uint32_t input_channels = block ? channels : previous;
            char prefix[192];
            snprintf(prefix, sizeof(prefix), "encoder.down.%d.block.%d", level,
                     block);
            char name[224];
            snprintf(name, sizeof(name), "%s.norm1", prefix);
            if (!load_norm(encoder, &item->norm1, name, input_channels,
                           error, error_size)) return 0;
            snprintf(name, sizeof(name), "%s.conv1", prefix);
            if (!load_conv(encoder, &item->conv1, name, input_channels,
                           channels, 3, 1, 1, 1, 2, 1, 1, 1, 1,
                           error, error_size)) return 0;
            snprintf(name, sizeof(name), "%s.norm2", prefix);
            if (!load_norm(encoder, &item->norm2, name, channels,
                           error, error_size)) return 0;
            snprintf(name, sizeof(name), "%s.conv2", prefix);
            if (!load_conv(encoder, &item->conv2, name, channels, channels, 3,
                           1, 1, 1, 2, 1, 1, 1, 1,
                           error, error_size)) return 0;
            if (input_channels != channels) {
                snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
                if (!load_conv(encoder, &item->shortcut, name, input_channels,
                               channels, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                               error, error_size)) return 0;
                item->has_shortcut = 1;
            }
        }
        if (space_strides[level] * time_strides[level] > 1) {
            char name[192];
            snprintf(name, sizeof(name), "encoder.down.%d.downsample.conv", level);
            uint32_t spatial_tail = space_strides[level] == 2 ? 1 : 0;
            if (!load_conv(encoder, &encoder->levels[level].downsample, name,
                           channels, channels, 3, time_strides[level],
                           space_strides[level], space_strides[level], 2, 0,
                           spatial_tail, 0, spatial_tail, error,
                           error_size)) return 0;
            encoder->levels[level].has_downsample = 1;
        }
        previous = channels;
    }
    if (!load_norm(encoder, &encoder->norm_out, "encoder.norm_out", 1024,
                   error, error_size) ||
        !load_conv(encoder, &encoder->conv_out, "encoder.conv_out", 1024,
                   MOMENT_CHANNELS, 3, 1, 1, 1, 2, 1, 1, 1, 1,
                   error, error_size) ||
        !load_conv(encoder, &encoder->quant, "quant_conv", MOMENT_CHANNELS,
                   MOMENT_CHANNELS, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                   error, error_size)) return 0;
    return 1;
}

static int convert_tensor_fp16(encoder_context *encoder,
                               h3_gpu_tensor **tensor,
                               tensor_arena *f32_weights, char *error,
                               size_t error_size) {
    if (!tensor || !*tensor) return 1;
    h3_gpu_tensor *f32 = *tensor;
    size_t elements = h3_gpu_tensor_elements(f32);
    *tensor = NULL;
    if (elements > UINT32_MAX || !tensor_arena_own(f32_weights, f32)) {
        fail(error, error_size, "cannot retain F32 VideoVAE weight");
        return 0;
    }
    h3_gpu_tensor *fp16 = h3_gpu_tensor_new_f16(encoder->gpu, elements);
    if (!fp16) {
        fail(error, error_size, "cannot allocate FP16 VideoVAE weight");
        return 0;
    }
    *tensor = fp16;
    return gpu_op(encoder, h3_gpu_cast_f32_to_f16(
        encoder->gpu, fp16, f32, (uint32_t)elements), error, error_size,
        "convert VideoVAE weight to FP16");
}

static int convert_conv_fp16(encoder_context *encoder, encoder_conv *conv,
                             tensor_arena *f32_weights, char *error,
                             size_t error_size) {
    return convert_tensor_fp16(encoder, &conv->weight, f32_weights,
                               error, error_size) &&
           convert_tensor_fp16(encoder, &conv->bias, f32_weights,
                               error, error_size);
}

static int convert_norm_fp16(encoder_context *encoder, encoder_norm *norm,
                             tensor_arena *f32_weights, char *error,
                             size_t error_size) {
    return convert_tensor_fp16(encoder, &norm->weight, f32_weights,
                               error, error_size) &&
           convert_tensor_fp16(encoder, &norm->bias, f32_weights,
                               error, error_size);
}

static int prepare_fp16_weights(encoder_context *encoder, char *error,
                                size_t error_size) {
    tensor_arena f32_weights = {0};
    int ok = gpu_op(encoder, h3_gpu_begin(encoder->gpu), error, error_size,
                    "begin VideoVAE FP16 weight conversion") &&
             convert_conv_fp16(encoder, &encoder->conv_in, &f32_weights,
                               error, error_size);
    for (int level = 0; ok && level < LEVELS; level++) {
        for (int block = 0; ok && block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            ok = convert_norm_fp16(encoder, &item->norm1, &f32_weights,
                                   error, error_size) &&
                 convert_conv_fp16(encoder, &item->conv1, &f32_weights,
                                   error, error_size) &&
                 convert_norm_fp16(encoder, &item->norm2, &f32_weights,
                                   error, error_size) &&
                 convert_conv_fp16(encoder, &item->conv2, &f32_weights,
                                   error, error_size) &&
                 (!item->has_shortcut || convert_conv_fp16(
                     encoder, &item->shortcut, &f32_weights,
                     error, error_size));
        }
        if (ok && encoder->levels[level].has_downsample)
            ok = convert_conv_fp16(encoder,
                    &encoder->levels[level].downsample, &f32_weights,
                    error, error_size);
    }
    if (ok)
        ok = convert_norm_fp16(encoder, &encoder->norm_out, &f32_weights,
                               error, error_size) &&
             convert_conv_fp16(encoder, &encoder->conv_out, &f32_weights,
                               error, error_size) &&
             convert_conv_fp16(encoder, &encoder->quant, &f32_weights,
                               error, error_size);
    int submitted = h3_gpu_submit(encoder->gpu);
    if (ok && !submitted)
        ok = gpu_op(encoder, 0, error, error_size,
                    "submit VideoVAE FP16 weight conversion");
    tensor_arena_free(&f32_weights);
    return ok && submitted;
}

static size_t tensor_elements(uint32_t depth, uint32_t height, uint32_t width,
                              uint32_t channels) {
    return (size_t)depth * height * width * channels;
}

static int conv_op(encoder_context *encoder, h3_gpu_tensor *output,
                   const h3_gpu_tensor *input, const encoder_conv *conv,
                   uint32_t depth, uint32_t height, uint32_t width,
                   h3_gpu_tensor *padded, char *error, size_t error_size) {
    const h3_gpu_tensor *source = input;
    uint32_t source_depth = depth, source_height = height, source_width = width;
    if (padded) {
        int padded_ok = encoder->fp16 ? h3_gpu_vae_encoder_pad_f16(
            encoder->gpu, padded, input, 1, depth, height, width,
            conv->input_channels, conv->depth_front, conv->height_before,
            conv->height_after, conv->width_before, conv->width_after) :
            h3_gpu_vae_encoder_pad_f32(
            encoder->gpu, padded, input, 1, depth, height, width,
            conv->input_channels, conv->depth_front, conv->height_before,
            conv->height_after, conv->width_before, conv->width_after);
        if (!gpu_op(encoder, padded_ok,
                error, error_size, "visual encoder causal padding")) return 0;
        source = padded;
        source_depth += conv->depth_front;
        source_height += conv->height_before + conv->height_after;
        source_width += conv->width_before + conv->width_after;
    }
    int conv_ok = encoder->fp16 ? h3_gpu_conv3d_f16(
            encoder->gpu, output, source, conv->weight, conv->bias, 1,
            source_depth, source_height, source_width, conv->input_channels,
            conv->output_channels, conv->kernel, conv->kernel, conv->kernel,
            conv->stride_t, conv->stride_h, conv->stride_w) :
        h3_gpu_conv3d_f32(
            encoder->gpu, output, source, conv->weight, conv->bias, 1,
            source_depth, source_height, source_width, conv->input_channels,
            conv->output_channels, conv->kernel, conv->kernel, conv->kernel,
            conv->stride_t, conv->stride_h, conv->stride_w);
    return gpu_op(encoder, conv_ok, error, error_size,
        "visual encoder Conv3d");
}

static h3_gpu_tensor *run_conv(encoder_context *encoder,
                               tensor_arena *arena,
                               h3_gpu_tensor *input,
                               const encoder_conv *conv,
                               uint32_t depth, uint32_t height, uint32_t width,
                               uint32_t *output_depth, uint32_t *output_height,
                               uint32_t *output_width, char *error,
                               size_t error_size) {
    uint32_t padded_d = depth + conv->depth_front;
    uint32_t padded_h = height + conv->height_before + conv->height_after;
    uint32_t padded_w = width + conv->width_before + conv->width_after;
    *output_depth = (padded_d - conv->kernel) / conv->stride_t + 1;
    *output_height = (padded_h - conv->kernel) / conv->stride_h + 1;
    *output_width = (padded_w - conv->kernel) / conv->stride_w + 1;
    h3_gpu_tensor *padded = NULL;
    if (conv->depth_front || conv->height_before || conv->height_after ||
        conv->width_before || conv->width_after)
        padded = tensor_arena_new_activation(
            encoder, arena,
            tensor_elements(padded_d, padded_h, padded_w,
                            conv->input_channels));
    h3_gpu_tensor *output = tensor_arena_new_activation(
        encoder, arena,
        tensor_elements(*output_depth, *output_height,
                        *output_width, conv->output_channels));
    int ok = output && (!((conv->depth_front || conv->height_before ||
                           conv->height_after || conv->width_before ||
                           conv->width_after)) || padded);
    if (!ok) {
        fail(error, error_size, "cannot allocate visual encoder convolution");
    } else
        ok = conv_op(encoder, output, input, conv, depth, height, width, padded,
                     error, error_size);
    if (!ok) return NULL;
    return output;
}

static h3_gpu_tensor *run_block(encoder_context *encoder,
                                tensor_arena *arena,
                                h3_gpu_tensor *input,
                                const encoder_block *block, uint32_t depth,
                                uint32_t height, uint32_t width,
                                uint32_t input_channels,
                                uint32_t output_channels, char *error,
                                size_t error_size) {
    size_t input_count = tensor_elements(depth, height, width, input_channels);
    size_t output_count = tensor_elements(depth, height, width, output_channels);
    size_t pad1_count = tensor_elements(depth + 2, height + 2, width + 2,
                                        input_channels);
    size_t pad2_count = tensor_elements(depth + 2, height + 2, width + 2,
                                        output_channels);
    h3_gpu_tensor *norm1 = tensor_arena_new_activation(encoder, arena,
                                                 input_count);
    h3_gpu_tensor *pad1 = tensor_arena_new_activation(encoder, arena, pad1_count);
    h3_gpu_tensor *hidden = tensor_arena_new_activation(encoder, arena,
                                                  output_count);
    h3_gpu_tensor *norm2 = tensor_arena_new_activation(encoder, arena,
                                                 output_count);
    h3_gpu_tensor *pad2 = tensor_arena_new_activation(encoder, arena, pad2_count);
    h3_gpu_tensor *output = tensor_arena_new_activation(encoder, arena,
                                                  output_count);
    h3_gpu_tensor *shortcut = block->has_shortcut ?
        tensor_arena_new_activation(encoder, arena, output_count) : NULL;
    int ok = norm1 && pad1 && hidden && norm2 && pad2 && output &&
             (!block->has_shortcut || shortcut);
    if (!ok) {
        fail(error, error_size, "cannot allocate visual encoder residual block");
        goto done;
    }
    int norm_ok = encoder->fp16 ? h3_gpu_vae_encoder_group_norm_silu_f16(
        encoder->gpu, norm1, input, block->norm1.weight, block->norm1.bias,
        1, depth, height, width, input_channels, GROUPS, 1e-6f) :
        h3_gpu_vae_encoder_group_norm_silu_f32(
        encoder->gpu, norm1, input, block->norm1.weight, block->norm1.bias,
        1, depth, height, width, input_channels, GROUPS, 1e-6f);
    ok = gpu_op(encoder, norm_ok,
             error, error_size, "visual encoder group norm 1") &&
         conv_op(encoder, hidden, norm1, &block->conv1, depth, height, width,
                 pad1, error, error_size);
    if (ok) {
        norm_ok = encoder->fp16 ? h3_gpu_vae_encoder_group_norm_silu_f16(
            encoder->gpu, norm2, hidden, block->norm2.weight, block->norm2.bias,
            1, depth, height, width, output_channels, GROUPS, 1e-6f) :
            h3_gpu_vae_encoder_group_norm_silu_f32(
            encoder->gpu, norm2, hidden, block->norm2.weight, block->norm2.bias,
            1, depth, height, width, output_channels, GROUPS, 1e-6f);
        ok = gpu_op(encoder, norm_ok, error, error_size,
                    "visual encoder group norm 2");
    }
    if (ok) ok =
         conv_op(encoder, output, norm2, &block->conv2, depth, height, width,
                 pad2, error, error_size);
    const h3_gpu_tensor *residual = input;
    if (ok && block->has_shortcut) {
        ok = conv_op(encoder, shortcut, input, &block->shortcut, depth, height,
                     width, NULL, error, error_size);
        residual = shortcut;
    }
    if (ok) {
        int add_ok = encoder->fp16 ? h3_gpu_add_f16(
            encoder->gpu, output, residual, output, (uint32_t)output_count) :
            h3_gpu_add_scaled_f32(encoder->gpu, output, residual, output,
                                  1.0f, 1.0f, (uint32_t)output_count);
        ok = gpu_op(encoder, add_ok, error, error_size,
                    "visual encoder residual add");
    }
done:
    if (!ok) return NULL;
    return output;
}

static h3_gpu_tensor *encode_tile_quant(encoder_context *encoder,
                                        const float *pixels, int frames,
                                        int height, int width,
                                        int *latent_time, char *error,
                                        size_t error_size) {
    size_t pixel_count = (size_t)frames * height * width * RGB_CHANNELS;
    float *normalized = encoder->fp16 ? NULL :
        malloc(pixel_count * sizeof(*normalized));
    if (!encoder->fp16 && !normalized) {
        fail(error, error_size, "out of memory normalizing visual anchor");
        return NULL;
    }
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    if (!encoder->fp16) {
        size_t destination = 0;
        for (int time = 0; time < frames; time++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    for (int channel = 0; channel < RGB_CHANNELS; channel++) {
                        size_t source = (((size_t)channel * frames + time) *
                                         height + y) * width + x;
                        normalized[destination++] =
                            (pixels[source] - mean[channel]) /
                            deviation[channel];
                    }
    }
    tensor_arena arena = {0};
    h3_gpu_tensor *uploaded = tensor_arena_own(
        &arena, h3_gpu_tensor_from_f32(
            encoder->gpu, encoder->fp16 ? pixels : normalized, pixel_count));
    h3_gpu_tensor *hidden = encoder->fp16 ? tensor_arena_new_activation(
        encoder, &arena, pixel_count) : uploaded;
    free(normalized);
    if (!uploaded || !hidden) {
        fail(error, error_size, "cannot allocate visual encoder pixels");
        tensor_arena_free(&arena);
        return NULL;
    }
    int began = gpu_op(encoder, h3_gpu_begin(encoder->gpu), error, error_size,
                       "begin visual encoder tile");
    if (!began) {
        tensor_arena_free(&arena);
        return NULL;
    }
    h3_gpu_tensor *quant = NULL;
    int ok = 1;
    if (encoder->fp16)
        ok = gpu_op(encoder, h3_gpu_vae_encoder_normalize_pixels_f16(
            encoder->gpu, hidden, uploaded, 1, (uint32_t)frames,
            (uint32_t)height, (uint32_t)width), error, error_size,
            "normalize visual encoder pixels to FP16");
    uint32_t depth = (uint32_t)frames, h = (uint32_t)height, w = (uint32_t)width;
    uint32_t next_d, next_h, next_w;
    h3_gpu_tensor *next = ok ? run_conv(
        encoder, &arena, hidden, &encoder->conv_in,
                                   depth, h, w, &next_d, &next_h, &next_w,
                                   error, error_size) : NULL;
    hidden = next;
    if (!hidden) goto done;
    depth = next_d; h = next_h; w = next_w;
    uint32_t previous = 128;
    for (int level = 0; level < LEVELS && hidden; level++) {
        uint32_t channels = level_channels[level];
        for (int block = 0; block < BLOCKS && hidden; block++) {
            uint32_t input_channels = block ? channels : previous;
            next = run_block(encoder, &arena, hidden,
                             &encoder->levels[level].blocks[block], depth, h, w,
                             input_channels, channels, error, error_size);
            hidden = next;
        }
        if (hidden && encoder->levels[level].has_downsample) {
            next = run_conv(encoder, &arena, hidden,
                            &encoder->levels[level].downsample, depth, h, w,
                            &next_d, &next_h, &next_w, error, error_size);
            hidden = next;
            depth = next_d; h = next_h; w = next_w;
        }
        previous = channels;
    }
    if (!hidden) goto done;

    size_t hidden_count = tensor_elements(depth, h, w, 1024);
    size_t padded_count = tensor_elements(depth + 2, h + 2, w + 2, 1024);
    size_t moment_count = tensor_elements(depth, h, w, MOMENT_CHANNELS);
    h3_gpu_tensor *norm = tensor_arena_new_activation(encoder, &arena,
                                                hidden_count);
    h3_gpu_tensor *padded = tensor_arena_new_activation(encoder, &arena,
                                                  padded_count);
    h3_gpu_tensor *moments = tensor_arena_new_activation(encoder, &arena,
                                                   moment_count);
    quant = encoder->fp16 ? h3_gpu_tensor_new_f16(encoder->gpu, moment_count) :
                            h3_gpu_tensor_new_f32(encoder->gpu, moment_count);
    ok = norm && padded && moments && quant;
    if (!ok) {
        fail(error, error_size, "cannot allocate visual encoder output");
    } else {
        int norm_ok = encoder->fp16 ? h3_gpu_vae_encoder_group_norm_silu_f16(
            encoder->gpu, norm, hidden, encoder->norm_out.weight,
            encoder->norm_out.bias, 1, depth, h, w, 1024, GROUPS, 1e-6f) :
            h3_gpu_vae_encoder_group_norm_silu_f32(
            encoder->gpu, norm, hidden, encoder->norm_out.weight,
            encoder->norm_out.bias, 1, depth, h, w, 1024, GROUPS, 1e-6f);
        ok = gpu_op(encoder, norm_ok,
                 error, error_size, "visual encoder output norm") &&
             conv_op(encoder, moments, norm, &encoder->conv_out, depth, h, w,
                     padded, error, error_size) &&
             conv_op(encoder, quant, moments, &encoder->quant, depth, h, w,
                     NULL, error, error_size);
    }
done:
    if (!gpu_op(encoder, h3_gpu_submit(encoder->gpu), error, error_size,
                "submit visual encoder tile")) ok = 0;
    tensor_arena_free(&arena);
    if (!ok) {
        h3_gpu_tensor_free(quant);
        return NULL;
    }
    *latent_time = (int)depth;
    return quant;
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_axis_build(int extent, tile_axis *axis, char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < SPATIAL_RATIO || extent % SPATIAL_RATIO) {
        fail(error, error_size,
             "visual encoder extent must be a multiple of %d", SPATIAL_RATIO);
        return 0;
    }
    if (extent <= TILE_PIXELS) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing encoder tiles");
            return 0;
        }
        return 1;
    }
    int count = (extent + TILE_PIXELS - 1) / TILE_PIXELS;
    while (TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing encoder tiles");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + TILE_PIXELS -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = TILE_PIXELS;
    return 1;
}

static float *extract_pixel_tile(const float *pixels, int frames,
                                 int full_h, int full_w, int start_y,
                                 int start_x, int tile_h, int tile_w,
                                 char *error, size_t error_size) {
    size_t count = (size_t)RGB_CHANNELS * frames * tile_h * tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting visual anchor tile");
        return NULL;
    }
    for (int channel = 0; channel < RGB_CHANNELS; channel++)
        for (int time = 0; time < frames; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * frames + time) * full_h +
                                  start_y + y) * full_w + start_x;
                size_t destination = (((size_t)channel * frames + time) *
                                       tile_h + y) * tile_w;
                memcpy(tile + destination, pixels + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

/* Ref2VA's encode_temporal pads the source timeline once, then slices fixed
 * clips. Constructing an individual clip here is equivalent and avoids a
 * host-side padded full-video allocation; this is the one permitted RGB
 * boundary before its upload to the encoder. */
static float *extract_pixel_tile_ref2va_clip(const float *pixels, int frames,
                                             int full_h, int full_w,
                                             int clip_start, int start_y,
                                             int start_x, int tile_h,
                                             int tile_w, char *error,
                                             size_t error_size) {
    enum { CLIP_FRAMES = 17 };
    size_t count = (size_t)RGB_CHANNELS * CLIP_FRAMES * tile_h * tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting Ref2VA temporal clip");
        return NULL;
    }
    for (int channel = 0; channel < RGB_CHANNELS; channel++)
        for (int time = 0; time < CLIP_FRAMES; time++) {
            int source_time = clip_start + time;
            if (source_time >= frames) source_time = frames - 1;
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * frames + source_time) *
                                 full_h + start_y + y) * full_w + start_x;
                size_t destination = (((size_t)channel * CLIP_FRAMES + time) *
                                      tile_h + y) * tile_w;
                memcpy(tile + destination, pixels + source,
                       (size_t)tile_w * sizeof(*tile));
            }
        }
    return tile;
}

int h3_ref2va_video_encode_plan_build(int frames,
                                      h3_ref2va_video_encode_plan *plan) {
    enum { CLIP_FRAMES = 17, TOKENS_PER_CLIP = 5, TOKEN_DROP = 3 };
    if (!plan || frames < 1 || frames > INT_MAX - (CLIP_FRAMES - 1)) return 0;
    int chunks = (frames + CLIP_FRAMES - 1) / CLIP_FRAMES;
    if (chunks > (INT_MAX - TOKEN_DROP) / TOKENS_PER_CLIP) return 0;
    int encoded_tokens = chunks * TOKENS_PER_CLIP;
    plan->chunks = chunks;
    plan->padded_frames = chunks * CLIP_FRAMES - frames;
    plan->encoded_tokens = encoded_tokens;
    plan->latent_tokens = encoded_tokens - TOKEN_DROP;
    return plan->latent_tokens > 0;
}

static int stitch_gpu_tiles(encoder_context *encoder,
                            h3_gpu_tensor *destination,
                            h3_gpu_tensor *const *tiles,
                            const tile_axis *y_axis,
                            const tile_axis *x_axis,
                            uint32_t time_offset, uint32_t chunk_time,
                            uint32_t full_time, uint32_t full_height,
                            uint32_t full_width, char *error,
                            size_t error_size) {
    uint32_t tile_h = (uint32_t)(y_axis->length / SPATIAL_RATIO);
    uint32_t tile_w = (uint32_t)(x_axis->length / SPATIAL_RATIO);
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            int above = tile_y ? index - x_axis->count : index;
            int left = tile_x ? index - 1 : index;
            uint32_t overlap_y = tile_y ?
                (uint32_t)(y_axis->overlaps[tile_y - 1] / SPATIAL_RATIO) : 0;
            uint32_t overlap_x = tile_x ?
                (uint32_t)(x_axis->overlaps[tile_x - 1] / SPATIAL_RATIO) : 0;
            uint32_t keep_h = tile_h - (uint32_t)(
                tile_y + 1 < y_axis->count ?
                y_axis->overlaps[tile_y] / SPATIAL_RATIO : 0);
            uint32_t keep_w = tile_w - (uint32_t)(
                tile_x + 1 < x_axis->count ?
                x_axis->overlaps[tile_x] / SPATIAL_RATIO : 0);
            int ok = encoder->fp16 ? h3_gpu_vae_encoder_stitch_latent_f16(
                encoder->gpu, destination, tiles[index],
                tile_y ? tiles[above] : NULL,
                tile_x ? tiles[left] : NULL, encoder->latent_mean_gpu,
                encoder->latent_std_gpu, time_offset, chunk_time, full_time,
                full_height, full_width, tile_h, tile_w,
                (uint32_t)(y_axis->starts[tile_y] / SPATIAL_RATIO),
                (uint32_t)(x_axis->starts[tile_x] / SPATIAL_RATIO),
                overlap_y, overlap_x, keep_h, keep_w) :
                h3_gpu_vae_encoder_stitch_latent_f32(
                encoder->gpu, destination, tiles[index],
                tile_y ? tiles[above] : NULL,
                tile_x ? tiles[left] : NULL, encoder->latent_mean_gpu,
                encoder->latent_std_gpu, time_offset, chunk_time, full_time,
                full_height, full_width, tile_h, tile_w,
                (uint32_t)(y_axis->starts[tile_y] / SPATIAL_RATIO),
                (uint32_t)(x_axis->starts[tile_x] / SPATIAL_RATIO),
                overlap_y, overlap_x, keep_h, keep_w);
            if (!gpu_op(encoder, ok, error, error_size,
                        "visual encoder latent stitch")) return 0;
        }
    return 1;
}

int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !*weight_directory || !shader_source_path ||
        !*shader_source_path || !pixels || !output || frames < 1 || height < 32 ||
        width < 32 || height % SPATIAL_RATIO || width % SPATIAL_RATIO) {
        fail(error, error_size, "invalid visual encoder arguments");
        return 0;
    }
    tile_axis y_axis = {0}, x_axis = {0};
    int ok = tile_axis_build(height, &y_axis, error, error_size) &&
             tile_axis_build(width, &x_axis, error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis);
        tile_axis_free(&x_axis);
        return 0;
    }
    encoder_context encoder = {0};
    encoder.fp16 = configured_fp16();
    encoder.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (encoder.gpu)
        h3_gpu_profile_set_label(encoder.gpu, "video VAE encoder");
    if (encoder.gpu)
        encoder.store = h3_weight_store_open(weight_directory, error, error_size);
    ok = encoder.gpu && encoder.store &&
         load_normalization(&encoder, weight_directory, error, error_size) &&
         load_weights(&encoder, error, error_size) &&
         (!encoder.fp16 || prepare_fp16_weights(&encoder, error, error_size));
    int tile_count = y_axis.count * x_axis.count;
    h3_gpu_tensor **tiles = ok ? calloc((size_t)tile_count, sizeof(*tiles)) : NULL;
    h3_gpu_tensor *final = NULL;
    if (ok && !tiles) {
        fail(error, error_size, "out of memory allocating visual encoder tiles");
        ok = 0;
    }
    int latent_time = 0;
    for (int y = 0, completed = 0; ok && y < y_axis.count; y++)
        for (int x = 0; ok && x < x_axis.count; x++, completed++) {
            float *tile = extract_pixel_tile(
                pixels, frames, height, width, y_axis.starts[y], x_axis.starts[x],
                y_axis.length, x_axis.length, error, error_size);
            int current_time = 0;
            tiles[completed] = tile ? encode_tile_quant(
                &encoder, tile, frames, y_axis.length, x_axis.length,
                &current_time, error, error_size) : NULL;
            free(tile);
            ok = tiles[completed] != NULL &&
                 (!latent_time || latent_time == current_time);
            if (ok) latent_time = current_time;
            if (ok && progress) progress(completed + 1, tile_count, progress_opaque);
        }
    int latent_h = height / SPATIAL_RATIO;
    int latent_w = width / SPATIAL_RATIO;
    size_t latent_count = (size_t)LATENT_CHANNELS * (size_t)latent_time *
                          (size_t)latent_h * (size_t)latent_w;
    if (ok && latent_count > UINT32_MAX) {
        fail(error, error_size, "visual encoder latent is too large");
        ok = 0;
    }
    if (ok) final = h3_gpu_tensor_new_f32(encoder.gpu, latent_count);
    if (ok && !final) {
        fail(error, error_size, "cannot allocate visual encoder latent");
        ok = 0;
    }
    if (ok) ok = gpu_op(&encoder, h3_gpu_begin(encoder.gpu), error,
                        error_size, "begin visual encoder stitch");
    if (ok) ok = stitch_gpu_tiles(
        &encoder, final, tiles, &y_axis, &x_axis, 0, (uint32_t)latent_time,
        (uint32_t)latent_time, (uint32_t)latent_h, (uint32_t)latent_w,
        error, error_size);
    if (ok) ok = gpu_op(&encoder, h3_gpu_submit(encoder.gpu), error,
                        error_size, "submit visual encoder stitch");
    if (ok) {
        output->values = malloc(latent_count * sizeof(*output->values));
        ok = output->values && h3_gpu_tensor_read_f32(
            final, output->values, latent_count) &&
            h3_gpu_get_stats(encoder.gpu, &output->gpu_stats);
        if (!ok) fail(error, error_size, "cannot read visual encoder latent");
        if (ok && (output->gpu_stats.host_tensor_reads != 1 ||
                   output->gpu_stats.host_tensor_writes != 0 ||
                   output->gpu_stats.blit_copies != 0)) {
            fail(error, error_size,
                 "visual encoder made an unexpected tensor transfer");
            ok = 0;
        }
        output->time = latent_time;
        output->height = latent_h;
        output->width = latent_w;
    }
    if (tiles) for (int index = 0; index < tile_count; index++)
        h3_gpu_tensor_free(tiles[index]);
    free(tiles);
    h3_gpu_tensor_free(final);
    cleanup(&encoder);
    tile_axis_free(&y_axis);
    tile_axis_free(&x_axis);
    if (!ok) h3_video_latent_free(output);
    return ok;
}

int h3_video_vae_encode_ref2va_temporal(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !*weight_directory || !shader_source_path ||
        !*shader_source_path || !pixels || !output || frames < 1 || height < 32 ||
        width < 32 || height % SPATIAL_RATIO || width % SPATIAL_RATIO) {
        fail(error, error_size, "invalid Ref2VA temporal encoder arguments");
        return 0;
    }
    h3_ref2va_video_encode_plan plan;
    tile_axis y_axis = {0}, x_axis = {0};
    encoder_context encoder = {0};
    encoder.fp16 = configured_fp16();
    h3_gpu_tensor *staged = NULL, *final = NULL;
    h3_gpu_tensor **tiles = NULL;
    int ok = h3_ref2va_video_encode_plan_build(frames, &plan) &&
             tile_axis_build(height, &y_axis, error, error_size) &&
             tile_axis_build(width, &x_axis, error, error_size);
    if (!ok) {
        fail(error, error_size, "invalid Ref2VA temporal encode plan");
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    int latent_h = height / SPATIAL_RATIO;
    int latent_w = width / SPATIAL_RATIO;
    size_t staged_elements = (size_t)LATENT_CHANNELS * plan.encoded_tokens *
                             latent_h * latent_w;
    size_t final_elements = (size_t)LATENT_CHANNELS * plan.latent_tokens *
                            latent_h * latent_w;
    if (staged_elements > UINT32_MAX || final_elements > UINT32_MAX) {
        fail(error, error_size, "Ref2VA temporal latent is too large");
        goto done;
    }
    encoder.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (encoder.gpu)
        h3_gpu_profile_set_label(encoder.gpu, "Ref2VA temporal video encoder");
    if (encoder.gpu)
        encoder.store = h3_weight_store_open(weight_directory, error, error_size);
    ok = encoder.gpu && encoder.store &&
         load_normalization(&encoder, weight_directory, error, error_size) &&
         load_weights(&encoder, error, error_size) &&
         (!encoder.fp16 || prepare_fp16_weights(&encoder, error, error_size));
    int tile_count = y_axis.count * x_axis.count;
    if (ok) staged = h3_gpu_tensor_new_f32(encoder.gpu, staged_elements);
    if (ok) final = h3_gpu_tensor_new_f32(encoder.gpu, final_elements);
    if (ok) tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (ok && (!staged || !final || !tiles)) {
        fail(error, error_size, "out of memory allocating Ref2VA temporal latents");
        ok = 0;
    }
    for (int chunk = 0, completed = 0; ok && chunk < plan.chunks; chunk++) {
        for (int tile_y = 0; ok && tile_y < y_axis.count; tile_y++)
            for (int tile_x = 0; ok && tile_x < x_axis.count; tile_x++, completed++) {
                int index = tile_y * x_axis.count + tile_x;
                int current_time = 0;
                float *tile = extract_pixel_tile_ref2va_clip(
                    pixels, frames, height, width, chunk * 17,
                    y_axis.starts[tile_y], x_axis.starts[tile_x],
                    y_axis.length, x_axis.length, error, error_size);
                if (tile) tiles[index] = encode_tile_quant(
                    &encoder, tile, 17, y_axis.length, x_axis.length,
                    &current_time, error, error_size);
                free(tile);
                ok = tiles[index] != NULL && current_time == 5;
                if (!ok && !error[0])
                    fail(error, error_size,
                         "Ref2VA temporal clip did not encode to five tokens");
                if (ok && progress)
                    progress(completed + 1, plan.chunks * tile_count,
                             progress_opaque);
            }
        if (ok) ok = gpu_op(&encoder, h3_gpu_begin(encoder.gpu), error,
                            error_size, "begin Ref2VA temporal stitch");
        if (ok) ok = stitch_gpu_tiles(
            &encoder, staged, tiles, &y_axis, &x_axis,
            (uint32_t)(chunk * 5), 5, (uint32_t)plan.encoded_tokens,
            (uint32_t)latent_h, (uint32_t)latent_w, error, error_size);
        if (ok) ok = gpu_op(&encoder, h3_gpu_submit(encoder.gpu), error,
                            error_size, "submit Ref2VA temporal stitch");
        for (int index = 0; index < tile_count; index++) {
            h3_gpu_tensor_free(tiles[index]);
            tiles[index] = NULL;
        }
    }
    if (ok) ok = gpu_op(&encoder, h3_gpu_begin(encoder.gpu), error,
                        error_size, "begin Ref2VA temporal token drop") &&
        gpu_op(&encoder, h3_gpu_vae_encoder_temporal_take_f32(
            encoder.gpu, final, staged, (uint32_t)plan.encoded_tokens,
            (uint32_t)plan.latent_tokens, (uint32_t)latent_h,
            (uint32_t)latent_w), error, error_size,
            "Ref2VA temporal token drop") &&
        gpu_op(&encoder, h3_gpu_submit(encoder.gpu), error, error_size,
            "submit Ref2VA temporal token drop");
    if (ok) {
        output->values = malloc(final_elements * sizeof(*output->values));
        ok = output->values && h3_gpu_tensor_read_f32(
            final, output->values, final_elements) &&
            h3_gpu_get_stats(encoder.gpu, &output->gpu_stats);
        if (!ok) fail(error, error_size, "cannot read Ref2VA temporal latent");
        if (ok && (output->gpu_stats.host_tensor_reads != 1 ||
                   output->gpu_stats.host_tensor_writes != 0)) {
            fail(error, error_size,
                 "Ref2VA temporal encoder made an unexpected host tensor transfer");
            ok = 0;
        }
        output->time = plan.latent_tokens;
        output->height = latent_h;
        output->width = latent_w;
    }

done:
    if (tiles) for (int index = 0; index < y_axis.count * x_axis.count; index++)
        h3_gpu_tensor_free(tiles[index]);
    free(tiles);
    h3_gpu_tensor_free(staged);
    h3_gpu_tensor_free(final);
    cleanup(&encoder);
    tile_axis_free(&y_axis);
    tile_axis_free(&x_axis);
    if (!ok) h3_video_latent_free(output);
    return ok;
}

void h3_video_latent_free(h3_video_latent *latent) {
    if (!latent) return;
    free(latent->values);
    memset(latent, 0, sizeof(*latent));
}
