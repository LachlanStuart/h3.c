/* Public API for the h3-metal MiniMax-H3 inference engine. */
#ifndef H3_H
#define H3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VERSION "0.1.0-dev"
#define H3_DEFAULT_WIDTH 864
#define H3_DEFAULT_HEIGHT 480
#define H3_DEFAULT_FRAMES 56
#define H3_DEFAULT_STEPS 20
#define H3_DEFAULT_DIT_LAYERS 50
#define H3_MIN_DIT_LAYERS 35
#define H3_REFERENCE_AUDIO_RATE 32000
#define H3_REFERENCE_AUDIO_MAX_SECONDS 45
#define H3_REFERENCE_AUDIO_MAX_SAMPLES \
    (H3_REFERENCE_AUDIO_RATE * H3_REFERENCE_AUDIO_MAX_SECONDS)

typedef struct h3_ctx h3_ctx;
typedef struct h3_result h3_result;
typedef struct h3_joint_latents h3_joint_latents;

typedef struct {
    size_t embedding_entries;
    size_t embedding_bytes;
    int prepared_dit;
    int video_decoder;
} h3_cache_info;

typedef enum {
    H3_REFERENCE_IMAGE = 1,
    H3_REFERENCE_VIDEO = 2,
    H3_REFERENCE_AUDIO = 3,
    H3_REFERENCE_VIDEO_AUDIO = 4
} h3_reference_kind;

typedef struct {
    h3_reference_kind kind;
    const char *path;
    const char *audio_path;
    int include_embedded_audio;
} h3_reference;

typedef enum {
    H3_REFERENCE_IMAGE_MATCH = 0,
    H3_REFERENCE_IMAGE_MAX = 1
} h3_reference_image_size;

typedef enum {
    H3_SAMPLER_RES = 0,
    H3_SAMPLER_EULER = 1
} h3_sampler;

typedef enum {
    H3_SCHEDULER_SIMPLE = 0,
    H3_SCHEDULER_BETA = 1
} h3_scheduler;

typedef enum {
    H3_ADAPTER_NONE = 0,
    H3_ADAPTER_MODELTC_TURBO = 1,
    H3_ADAPTER_ALIBABA_PAI_PDD = 2
} h3_adapter_kind;

typedef enum {
    H3_ADAPTER_PROFILE_NONE = 0,
    H3_ADAPTER_PROFILE_MODELTC_FL2VA_544_4,
    H3_ADAPTER_PROFILE_MODELTC_FL2VA_544_8,
    H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_4,
    H3_ADAPTER_PROFILE_MODELTC_FL2VA_768_8,
    H3_ADAPTER_PROFILE_MODELTC_REF2VA_544_4,
    H3_ADAPTER_PROFILE_MODELTC_REF2VA_768_8,
    H3_ADAPTER_PROFILE_PAI_FL2VA_8,
    H3_ADAPTER_PROFILE_PAI_REF2VA_8
} h3_adapter_profile;

/* H.264 is the portable delivery format. FFV1 is an RGB lossless diagnostic
 * format and must be written to a Matroska (.mkv) container. */
typedef enum {
    H3_VIDEO_CODEC_H264 = 0,
    H3_VIDEO_CODEC_FFV1 = 1
} h3_video_codec;

/* These names map to libx264's documented preset values. They are an enum
 * rather than caller-supplied FFmpeg arguments so process spawning remains
 * fixed and safe. */
typedef enum {
    H3_VIDEO_PRESET_ULTRAFAST = 0,
    H3_VIDEO_PRESET_SUPERFAST,
    H3_VIDEO_PRESET_VERYFAST,
    H3_VIDEO_PRESET_FASTER,
    H3_VIDEO_PRESET_FAST,
    H3_VIDEO_PRESET_MEDIUM,
    H3_VIDEO_PRESET_SLOW,
    H3_VIDEO_PRESET_SLOWER,
    H3_VIDEO_PRESET_VERYSLOW
} h3_video_preset;

typedef struct {
    int width;
    int height;
    int stride;
    const uint8_t *rgb;
    int frame_index;
    int frame_count;
    /* Non-negative only for an intermediate denoising preview. */
    int denoise_step;
    int denoise_steps;
} h3_frame;

typedef int (*h3_frame_callback)(const h3_frame *frame, void *opaque);
typedef int (*h3_progress_callback)(const char *phase, int completed, int total,
                                    void *opaque);

typedef struct {
    int width;
    int height;
    int frames;
    int steps;
    h3_sampler sampler;
    h3_scheduler scheduler;
    /* Runtime-only adapter: its deltas are never written to a checkpoint. */
    const char *adapter_path;
    h3_adapter_kind adapter_kind;
    h3_adapter_profile adapter_profile;
    float adapter_strength;
    /* Independent shifted sigma grids; released base defaults are 12/3. */
    float video_shift;
    float audio_shift;
    uint64_t seed;
    /* Optional exact DiT safetensors file. The remaining tokenizer, text
     * encoder, and VAEs continue to come from model_dir. */
    const char *dit_checkpoint;
    const char *output_path;
    /* Write only the generated 32 kHz stereo waveform as a standalone WAV.
     * Joint DiT denoising and AudioVAE decoding still run; VideoVAE decode,
     * RGB delivery, and video muxing are skipped. */
    int audio_only;
    /* Optionally emit a second FFV1/Matroska diagnostic from these exact
     * generated RGB and PCM buffers. Must differ from output_path. */
    const char *lossless_output_path;
    /* Optional normalized F32 [24,T,H,W] latent written after denoising and
     * before VideoVAE decode. The stable file format is documented by
     * h3_latent_io.h. */
    const char *latent_output_path;
    /* Experimental RepLDM/Restart second pass. The input MP4 is decoded,
     * re-encoded by the VAEs, video-only forward noised, then refined over a
     * suffix of a normal serving schedule. */
    const char *refine_video_path;
    /* Optional clean normalized high-resolution video latent. When present,
     * restart refinement skips RGB decode/resize/VideoVAE encode entirely;
     * refine_video_path still supplies the clean frozen audio and mux source. */
    const char *refine_latent_path;
    /* Internal staged-production handoff.  `latent_only` transfers the
     * post-denoise joint tensors to latent_result; inline_refine supplies the
     * clean upscaled video plus frozen audio without an intermediate file. */
    int latent_only;
    h3_joint_latents *latent_result;
    const h3_joint_latents *inline_refine;
    /* Retain the existing same-checkpoint DiT and rebuild only its
     * geometry-dependent session.  Used by h3_generate_inline_production. */
    int reuse_prepared_dit;
    int restart_steps;
    int restart_schedule_steps;
    int freeze_audio;
    h3_video_codec video_codec;
    h3_video_preset video_preset;
    /* libx264 CRF in [0, 51]. Ignored for FFV1. */
    int video_crf;
    /* Structural boundary inputs. They may accompany ordered references when
     * dit_checkpoint selects a Hybrid model that supports both paths. */
    const char *first_frame;
    const char *last_frame;
    const h3_reference *references;
    size_t reference_count;
    h3_reference_image_size reference_image_size;
    /* Evaluate one of every N denoiser steps. 1 is the close-reference path,
     * 2 is the validated fast path, and 3 is the aggressive fast path. */
    int denoise_reuse;
    /* Number of gate-ranked DiT residual blocks to retain. 50 is exact,
     * 45 is the validated fast setting, and 40 is more aggressive. */
    int dit_layers;
    /* Recompute the transformer core every N denoiser steps while refreshing
     * the timestep head each step. 1 is exact, 4 fast, and 6 aggressive. */
    int core_reuse;
    /* Pair adjacent horizontal video tokens through middle DiT blocks while
     * preserving their full-resolution residual. Early noisy evaluations use
     * a deeper reduced interval. This is a validated aggressive speed mode. */
    int token_reduction;
    /* Use one int8 activation scale per FC2 row and the M5 full-K kernel.
     * Faster, but more numerically aggressive than grouped int8. */
    int use_int8_row_fc2;
    /* Restore the released spatial RoPE grid at 256x256. The default applies
     * a visually validated half-scale grid only at that native canvas. */
    int use_reference_rope;
    /* Keep only two original BF16 DiT blocks in memory and overlap reading the
     * next block from the checkpoint with execution of the current block. */
    int ssd_streaming;
    /* Optional lower internal model canvas. Both must be zero (exact output
     * canvas) or valid same-aspect dimensions no larger than width/height. */
    int render_width;
    int render_height;
    /* Force the portable close-reference BF16/MPS MLP implementation instead
     * of the fastest validated native MLP supported by the current GPU. */
    int use_slower_bf16_mlp;
    /* Force the portable close-reference BF16 QKV projection. */
    int use_slower_bf16_qkv;
    /* Force the portable BF16 attention-output projection. */
    int use_slower_bf16_attention_output;
    /* Use FP32 cooperative accumulators in M5 BF16 TensorOps, then round the
     * projection result back to BF16 at the normal output boundary. */
    int use_fp32_bf16_accumulator;
    /* Materialize row-major BF16 after SDPA before int8 quantization. */
    int use_slower_row_major_attention_output;
    /* Keep int8 projection-input quantization as standalone kernels. */
    int use_slower_unfused_int8_inputs;
    /* Keep Q/K norm and RoPE as a separate kernel after int8 QKV. */
    int use_slower_unfused_qkv_rope;
    /* Force scalar BF16 loads in the fused Q/K RMS reducer. */
    int use_slower_scalar_qkv_rms;
    /* Reread int8 dequantization scales from device memory per output. */
    int use_slower_uncached_int8_scales;
    /* Use the generic runtime-bound FC1 TensorOps K loop. */
    int use_slower_dynamic_fc1_k;
    /* Force the original 256-thread FC2 grouped activation quantizer. */
    int use_slower_grouped_quantizer;
    /* Decode and deliver one representative frame after every Euler step. */
    int preview_denoise;
    h3_frame_callback on_frame;
    h3_progress_callback on_progress;
    void *callback_opaque;
} h3_params;

/* Internal/public ABI handoff for a staged generation.  Values are normalized
 * host F32 tensors in native H3 layout.  A producer transfers ownership to the
 * caller only when `latent_only` is selected; use h3_joint_latents_free. */
struct h3_joint_latents {
    float *video;
    float *audio;
    int video_time;
    int video_height;
    int video_width;
    int audio_time;
};

typedef struct {
    h3_params working;
    h3_params target;
    const char *upscaler_python;
    const char *upscaler_script;
    const char *upscaler_source;
    const char *upscaler_checkpoint;
    float upscale;
} h3_production_params;

#define H3_PARAMS_DEFAULT { \
    .width = H3_DEFAULT_WIDTH, .height = H3_DEFAULT_HEIGHT, \
    .frames = H3_DEFAULT_FRAMES, .steps = H3_DEFAULT_STEPS, \
    .sampler = H3_SAMPLER_EULER, .scheduler = H3_SCHEDULER_BETA, \
    .adapter_kind = H3_ADAPTER_NONE, .adapter_profile = H3_ADAPTER_PROFILE_NONE, \
    .adapter_strength = 1.0f, .video_shift = 12.0f, .audio_shift = 3.0f, \
    .seed = UINT64_C(42), \
    .video_codec = H3_VIDEO_CODEC_H264, \
    .video_preset = H3_VIDEO_PRESET_SLOW, .video_crf = 18, \
    .reference_image_size = H3_REFERENCE_IMAGE_MATCH, \
    .denoise_reuse = 1, .dit_layers = H3_DEFAULT_DIT_LAYERS, \
    .core_reuse = 1 \
}

typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;
    int metal4;
    int unified_memory;
} h3_device_info;

typedef struct {
    uint64_t bytes;
    uint64_t tensor_bytes;
    size_t files;
    size_t tensors;
} h3_component_info;

typedef struct {
    h3_component_info text_encoder;
    h3_component_info fl2va_transformer;
    h3_component_info ref2va_transformer;
    h3_component_info video_vae;
    h3_component_info audio_vae;
} h3_model_info;

struct h3_result {
    int width;
    int height;
    int frames;
    int fps;
    int sample_rate;
    uint64_t seed;
};

/* Load model metadata and initialize the Metal device. Weights remain unmapped. */
h3_ctx *h3_load_dir(const char *model_dir);
void h3_free(h3_ctx *ctx);

const char *h3_last_error(const h3_ctx *ctx);
const h3_device_info *h3_device(const h3_ctx *ctx);
const h3_model_info *h3_model(const h3_ctx *ctx);

/* Interactive-session reuse. Disabled by default so one-shot callers retain
 * the original phase-by-phase memory lifetime. */
void h3_cache_set_enabled(h3_ctx *ctx, int enabled);
void h3_cache_clear(h3_ctx *ctx);
void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info);

/* Generate media, delivering decoded frames incrementally through on_frame. */
h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params);
h3_result *h3_generate_inline_production(
    h3_ctx *ctx, const char *prompt, const h3_production_params *params);
void h3_joint_latents_free(h3_joint_latents *latents);
void h3_result_free(h3_result *result);

#ifdef __cplusplus
}
#endif
#endif
