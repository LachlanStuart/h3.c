#include "h3_ffmpeg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_av_mux.c: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    enum { WIDTH = 32, HEIGHT = 32, FRAMES = 8, SAMPLES = 64000 };
    const char *path = argc > 1 ? argv[1] : "/tmp/h3-av-mux-test.mp4";
    const char *lossless_path = "/tmp/h3-av-mux-lossless-test.mkv";
    const char *source_audio_path = "/tmp/h3-av-source-audio-test.mp4";
    const char *short_source_path = "/tmp/h3-av-short-source-test.mp4";
    const char *long_video_path = "/tmp/h3-av-long-video-test.mp4";
    h3_ffmpeg_video_plan plan;
    if (strcmp(h3_video_preset_name(H3_VIDEO_PRESET_SLOW), "slow") ||
        strcmp(h3_video_preset_name(H3_VIDEO_PRESET_FAST), "fast") ||
        h3_video_preset_name((h3_video_preset)-1) ||
        !h3_video_settings_valid(H3_VIDEO_CODEC_H264,
                                 H3_VIDEO_PRESET_SLOW, 18) ||
        !h3_video_settings_valid(H3_VIDEO_CODEC_FFV1,
                                 H3_VIDEO_PRESET_VERYSLOW, 51) ||
        h3_video_settings_valid(H3_VIDEO_CODEC_H264,
                                H3_VIDEO_PRESET_SLOW, 52))
        die("video settings did not validate default and override values");
    if (!h3_ffmpeg_video_plan_build(H3_VIDEO_CODEC_H264,
                                    H3_VIDEO_PRESET_SLOW, 18, &plan) ||
        strcmp(plan.video_codec, "libx264") || strcmp(plan.preset, "slow") ||
        strcmp(plan.crf, "18") || strcmp(plan.pixel_format, "yuv420p") ||
        strcmp(plan.audio_codec, "aac") || plan.container)
        die("default H.264 FFmpeg argument plan is incorrect");
    if (!h3_ffmpeg_video_plan_build(H3_VIDEO_CODEC_H264,
                                    H3_VIDEO_PRESET_FAST, 27, &plan) ||
        strcmp(plan.preset, "fast") || strcmp(plan.crf, "27"))
        die("overridden H.264 FFmpeg argument plan is incorrect");
    if (!h3_ffmpeg_video_plan_build(H3_VIDEO_CODEC_FFV1,
                                    H3_VIDEO_PRESET_FAST, 27, &plan) ||
        strcmp(plan.video_codec, "ffv1") || plan.preset || plan.crf[0] ||
        strcmp(plan.pixel_format, "rgb24") ||
        strcmp(plan.audio_codec, "pcm_f32le") ||
        strcmp(plan.container, "matroska"))
        die("lossless FFmpeg argument plan is incorrect");
    uint8_t *rgb = malloc(WIDTH * HEIGHT * 3 * FRAMES);
    float *pcm = malloc(2 * SAMPLES * sizeof(*pcm));
    if (!rgb || !pcm) die("out of memory creating mux fixture");
    for (int frame = 0; frame < FRAMES; frame++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t pixel = ((size_t)frame * HEIGHT * WIDTH +
                                (size_t)y * WIDTH + (size_t)x) * 3;
                rgb[pixel] = (uint8_t)(x * 8);
                rgb[pixel + 1] = (uint8_t)(y * 8);
                rgb[pixel + 2] = (uint8_t)(frame * 32);
            }
    for (int channel = 0; channel < 2; channel++)
        for (int sample = 0; sample < SAMPLES; sample++)
            pcm[(size_t)channel * SAMPLES + (size_t)sample] =
                0.05f * sinf(2.0f * 3.14159265358979323846f *
                             (float)(220 + channel * 110) * (float)sample /
                             32000.0f);
    char error[512];
    if (h3_ffmpeg_write_av_rgb24_f32(
            "/tmp/h3-lossless-wrong-container.mp4", rgb, FRAMES,
            WIDTH, HEIGHT, 24, pcm, SAMPLES, 2, 32000,
            H3_VIDEO_CODEC_FFV1, H3_VIDEO_PRESET_SLOW, 18,
            error, sizeof(error)))
        die("FFV1 accepted a non-Matroska output path");
    if (!h3_ffmpeg_write_av_rgb24_f32(path, rgb, FRAMES, WIDTH, HEIGHT, 24,
                                      pcm, SAMPLES, 2, 32000,
                                      H3_VIDEO_CODEC_H264,
                                      H3_VIDEO_PRESET_SLOW, 18,
                                      error, sizeof(error))) die(error);
    struct stat status;
    if (stat(path, &status) != 0 || status.st_size < 1000)
        die("FFmpeg did not create a nonempty A/V container");
    int probed_width = 0, probed_height = 0;
    if (!h3_ffprobe_visual_size(path, &probed_width, &probed_height,
                                error, sizeof(error))) die(error);
    if (probed_width != WIDTH || probed_height != HEIGHT)
        die("FFprobe returned unexpected visual dimensions");
    float *decoded = NULL;
    if (!h3_ffmpeg_read_image_f32(path, WIDTH, HEIGHT,
                                  H3_IMAGE_FIT_STRETCH, &decoded,
                                  error, sizeof(error))) die(error);
    double red = 0.0, green = 0.0, blue = 0.0;
    for (size_t pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
        if (!isfinite(decoded[pixel]) || decoded[pixel] < 0.0f ||
            decoded[pixel] > 1.0f ||
            !isfinite(decoded[WIDTH * HEIGHT + pixel]) ||
            !isfinite(decoded[2 * WIDTH * HEIGHT + pixel]))
            die("decoded FFmpeg image contains invalid pixels");
        red += decoded[pixel];
        green += decoded[WIDTH * HEIGHT + pixel];
        blue += decoded[2 * WIDTH * HEIGHT + pixel];
    }
    if (red < 100.0 || green < 100.0 || blue > 80.0)
        die("decoded FFmpeg image has unexpected channel-major content");
    free(decoded);
    decoded = NULL;
    if (!h3_ffmpeg_read_image_f32(path, WIDTH, HEIGHT * 2,
                                  H3_IMAGE_FIT_COVER, &decoded,
                                  error, sizeof(error))) die(error);
    free(decoded);
    decoded = NULL;
    int decoded_frames = 0;
    if (!h3_ffmpeg_read_video_f32(path, WIDTH, HEIGHT, FRAMES,
                                  &decoded, &decoded_frames,
                                  error, sizeof(error))) die(error);
    if (decoded_frames != 5)
        die("FFmpeg video input did not align to 5+17k frames");
    size_t decoded_values = (size_t)3 * decoded_frames * WIDTH * HEIGHT;
    for (size_t index = 0; index < decoded_values; index++) {
        if (!isfinite(decoded[index]) || decoded[index] < 0.0f ||
            decoded[index] > 1.0f)
            die("decoded FFmpeg video contains invalid pixels");
    }
    free(decoded);
    decoded = NULL;
    if (!h3_ffmpeg_read_video_f32_bicubic(path, WIDTH, HEIGHT, FRAMES,
                                           &decoded, &decoded_frames,
                                           error, sizeof(error))) die(error);
    if (decoded_frames != 5)
        die("bicubic FFmpeg video input did not align to 5+17k frames");
    for (size_t index = 0; index < decoded_values; index++) {
        if (!isfinite(decoded[index]) || decoded[index] < 0.0f ||
            decoded[index] > 1.0f)
            die("bicubic decoded FFmpeg video contains invalid pixels");
    }
    free(decoded);
    decoded = NULL;
    float *decoded_pcm = NULL;
    int decoded_samples = 0;
    if (!h3_ffmpeg_read_audio_f32(path, SAMPLES, 1,
                                  &decoded_pcm, &decoded_samples,
                                  error, sizeof(error))) die(error);
    if (decoded_samples != SAMPLES)
        die("FFmpeg audio input returned an unexpected sample count");
    double left_energy = 0.0, right_energy = 0.0;
    for (int sample = 0; sample < decoded_samples; sample++) {
        float left = decoded_pcm[sample];
        float right = decoded_pcm[decoded_samples + sample];
        if (!isfinite(left) || !isfinite(right))
            die("decoded FFmpeg audio contains non-finite PCM");
        left_energy += (double)left * left;
        right_energy += (double)right * right;
    }
    if (left_energy < 1.0 || right_energy < 1.0)
        die("decoded FFmpeg audio has no stereo signal");
    free(decoded_pcm);
    decoded_pcm = NULL;

    /* Restart refinement uses bicubic RGB input and must preserve this
     * original source soundtrack rather than encode the frozen model audio. */
    if (!h3_ffmpeg_write_rgb24_with_source_audio(
            source_audio_path, rgb, FRAMES, WIDTH, HEIGHT, 24, path,
            H3_VIDEO_PRESET_SLOW, 18, error, sizeof(error))) die(error);
    if (!h3_ffmpeg_read_audio_f32(source_audio_path, 8000, 1,
                                  &decoded_pcm, &decoded_samples,
                                  error, sizeof(error))) die(error);
    if (decoded_samples != 8000)
        die("source-audio mux did not provide the bounded soundtrack");
    left_energy = 0.0;
    right_energy = 0.0;
    for (int sample = 0; sample < decoded_samples; sample++) {
        left_energy += (double)decoded_pcm[sample] * decoded_pcm[sample];
        right_energy += (double)decoded_pcm[decoded_samples + sample] *
                        decoded_pcm[decoded_samples + sample];
    }
    if (left_energy < 0.01 || right_energy < 0.01)
        die("source-audio mux has no copied stereo signal");
    free(decoded_pcm);
    decoded_pcm = NULL;

    /* A 56-frame H3 target is 2.333 seconds while its 93-row clean AudioVAE
     * grid is 2.325 seconds.  Model that mismatch with a much shorter source
     * soundtrack: stream-copy muxing must retain all 56 video frames. */
    if (!h3_ffmpeg_write_av_rgb24_f32(
            short_source_path, rgb, FRAMES, WIDTH, HEIGHT, 24,
            pcm, 8000, 2, 32000, H3_VIDEO_CODEC_H264,
            H3_VIDEO_PRESET_SLOW, 18, error, sizeof(error))) die(error);
    enum { LONG_FRAMES = 56 };
    uint8_t *long_rgb = malloc((size_t)WIDTH * HEIGHT * 3 * LONG_FRAMES);
    if (!long_rgb) die("out of memory creating long video fixture");
    for (int frame = 0; frame < LONG_FRAMES; frame++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t pixel = ((size_t)frame * HEIGHT * WIDTH +
                                (size_t)y * WIDTH + (size_t)x) * 3;
                long_rgb[pixel] = (uint8_t)(frame * 3);
                long_rgb[pixel + 1] = (uint8_t)(x * 8);
                long_rgb[pixel + 2] = (uint8_t)(y * 8);
            }
    if (!h3_ffmpeg_write_rgb24_with_source_audio(
            long_video_path, long_rgb, LONG_FRAMES, WIDTH, HEIGHT, 24,
            short_source_path, H3_VIDEO_PRESET_SLOW, 18,
            error, sizeof(error))) die(error);
    free(long_rgb);
    decoded = NULL;
    if (!h3_ffmpeg_read_video_f32(long_video_path, WIDTH, HEIGHT,
                                  LONG_FRAMES, &decoded, &decoded_frames,
                                  error, sizeof(error))) die(error);
    if (decoded_frames != LONG_FRAMES)
        die("short source audio truncated the restart output video");
    free(decoded);
    decoded = NULL;

    /* The diagnostic path gets the exact same raw RGB and F32 PCM buffers,
     * but its FFV1/Matroska encoding must preserve those bytes. */
    if (!h3_ffmpeg_write_av_rgb24_f32(
            lossless_path, rgb, FRAMES, WIDTH, HEIGHT, 24,
            pcm, SAMPLES, 2, 32000, H3_VIDEO_CODEC_FFV1,
            H3_VIDEO_PRESET_FAST, 37, error, sizeof(error))) die(error);
    decoded = NULL;
    if (!h3_ffmpeg_read_video_f32(lossless_path, WIDTH, HEIGHT, FRAMES,
                                  &decoded, &decoded_frames,
                                  error, sizeof(error))) die(error);
    if (decoded_frames != 5)
        die("lossless FFmpeg video input did not align to 5+17k frames");
    for (int frame = 0; frame < decoded_frames; frame++)
        for (int channel = 0; channel < 3; channel++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++) {
                    size_t pixel = (size_t)y * WIDTH + (size_t)x;
                    size_t input = ((size_t)frame * HEIGHT * WIDTH + pixel) * 3 +
                                   (size_t)channel;
                    size_t output = ((size_t)channel * decoded_frames +
                                     (size_t)frame) * WIDTH * HEIGHT + pixel;
                    float expected = (float)rgb[input] / 255.0f;
                    if (fabsf(decoded[output] - expected) > 1e-7f)
                        die("lossless FFV1 did not preserve RGB pixels");
                }
    free(decoded);
    decoded = NULL;
    if (!h3_ffmpeg_read_audio_f32(lossless_path, SAMPLES, 1,
                                  &decoded_pcm, &decoded_samples,
                                  error, sizeof(error))) die(error);
    if (decoded_samples != SAMPLES ||
        memcmp(decoded_pcm, pcm, (size_t)2 * SAMPLES * sizeof(*pcm)))
        die("lossless Matroska did not preserve F32 PCM");
    free(decoded_pcm);
    printf("ok: concurrent FFmpeg video/PCM pipes created %s (%lld bytes)\n",
           path, (long long)status.st_size);
    free(rgb);
    free(pcm);
    return 0;
}
