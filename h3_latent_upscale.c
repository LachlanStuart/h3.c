#include "h3_latent_upscale.h"

#include <errno.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#include <spawn.h>

extern char **environ;

#define H3_LATENT_CHANNELS 24u
#define H3_LATENT_HEADER_BYTES 20u

static void fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static void put_u32_le(unsigned char *destination, uint32_t value) {
    destination[0] = (unsigned char)value;
    destination[1] = (unsigned char)(value >> 8);
    destination[2] = (unsigned char)(value >> 16);
    destination[3] = (unsigned char)(value >> 24);
}

static uint32_t get_u32_le(const unsigned char *source) {
    return (uint32_t)source[0] | ((uint32_t)source[1] << 8) |
           ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static int elements_for(int time, int height, int width, size_t *elements) {
    if (time < 2 || height < 1 || width < 1) return 0;
    size_t result = H3_LATENT_CHANNELS;
    const int dims[] = {time, height, width};
    for (size_t index = 0; index < sizeof(dims) / sizeof(dims[0]); index++) {
        if ((size_t)dims[index] > SIZE_MAX / result) return 0;
        result *= (size_t)dims[index];
    }
    if (result > SIZE_MAX / sizeof(float)) return 0;
    *elements = result;
    return 1;
}

static int write_all(int fd, const void *bytes, size_t count) {
    const unsigned char *cursor = bytes;
    while (count) {
        ssize_t written = write(fd, cursor, count);
        if (written > 0) { cursor += (size_t)written; count -= (size_t)written; }
        else if (written < 0 && errno == EINTR) continue;
        else return 0;
    }
    return 1;
}

static int read_all(int fd, void *bytes, size_t count) {
    unsigned char *cursor = bytes;
    while (count) {
        ssize_t read_count = read(fd, cursor, count);
        if (read_count > 0) { cursor += (size_t)read_count; count -= (size_t)read_count; }
        else if (read_count < 0 && errno == EINTR) continue;
        else return 0;
    }
    return 1;
}

int h3_latent_upscale_pipe(const char *python, const char *script,
                           const char *source, const char *checkpoint,
                           const float *input, int input_time,
                           int input_height, int input_width,
                           float **output, int *output_time,
                           int *output_height, int *output_width,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) *output = NULL;
    if (!script || !*script || !source || !*source || !checkpoint || !*checkpoint ||
        !input || !output || !output_time || !output_height || !output_width) {
        fail(error, error_size, "invalid latent upscaler arguments");
        return 0;
    }
    size_t input_elements;
    if (!elements_for(input_time, input_height, input_width, &input_elements)) {
        fail(error, error_size, "invalid input latent geometry");
        return 0;
    }
    for (size_t index = 0; index < input_elements; index++) {
        if (!isfinite(input[index])) {
            fail(error, error_size, "input latent contains non-finite values");
            return 0;
        }
    }
    int in_pipe[2] = {-1, -1}, out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) || pipe(out_pipe)) {
        fail(error, error_size, "cannot create latent upscaler pipes");
        if (in_pipe[0] >= 0) close(in_pipe[0]); if (in_pipe[1] >= 0) close(in_pipe[1]);
        if (out_pipe[0] >= 0) close(out_pipe[0]); if (out_pipe[1] >= 0) close(out_pipe[1]);
        return 0;
    }
    char *arguments[10];
    int argument = 0;
    if (python && *python) arguments[argument++] = (char *)python;
    arguments[argument++] = (char *)script;
    arguments[argument++] = "--stream";
    arguments[argument++] = "--source";
    arguments[argument++] = (char *)source;
    arguments[argument++] = "--checkpoint";
    arguments[argument++] = (char *)checkpoint;
    arguments[argument] = NULL;
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions)) {
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);
        fail(error, error_size, "cannot configure latent upscaler pipes");
        return 0;
    }
    int action_ok = !posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO) &&
        !posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO) &&
        !posix_spawn_file_actions_addclose(&actions, in_pipe[0]) &&
        !posix_spawn_file_actions_addclose(&actions, in_pipe[1]) &&
        !posix_spawn_file_actions_addclose(&actions, out_pipe[0]) &&
        !posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
    if (!action_ok) {
        posix_spawn_file_actions_destroy(&actions);
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);
        fail(error, error_size, "cannot configure latent upscaler pipes");
        return 0;
    }
    pid_t child = 0;
    int spawn = posix_spawnp(&child, arguments[0], &actions, NULL,
                             arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(in_pipe[0]); close(out_pipe[1]);
    if (spawn) {
        close(in_pipe[1]); close(out_pipe[0]);
        fail(error, error_size, "cannot launch latent upscaler sidecar");
        return 0;
    }
    unsigned char header[H3_LATENT_HEADER_BYTES] = {'H','3','L','A','T','F','3','2'};
    put_u32_le(header + 8, (uint32_t)input_time);
    put_u32_le(header + 12, (uint32_t)input_height);
    put_u32_le(header + 16, (uint32_t)input_width);
    /* A malformed/import-failed child may exit before consuming stdin.  Treat
     * EPIPE as a normal bridge error instead of terminating the H3 process. */
    struct sigaction ignored, previous;
    memset(&ignored, 0, sizeof(ignored));
    ignored.sa_handler = SIG_IGN;
    int signal_ok = !sigaction(SIGPIPE, &ignored, &previous);
    int ok = signal_ok && write_all(in_pipe[1], header, sizeof(header)) &&
             write_all(in_pipe[1], input, input_elements * sizeof(*input));
    close(in_pipe[1]);
    if (signal_ok) sigaction(SIGPIPE, &previous, NULL);
    unsigned char output_header[H3_LATENT_HEADER_BYTES];
    if (ok) ok = read_all(out_pipe[0], output_header, sizeof(output_header));
    size_t output_elements = 0;
    uint32_t time = 0, height = 0, width = 0;
    if (ok && !memcmp(output_header, "H3LATF32", 8)) {
        time = get_u32_le(output_header + 8);
        height = get_u32_le(output_header + 12);
        width = get_u32_le(output_header + 16);
        ok = time == (uint32_t)input_time &&
             height == (uint32_t)input_height * 2u &&
             width == (uint32_t)input_width * 2u &&
             elements_for((int)time, (int)height, (int)width, &output_elements);
    } else ok = 0;
    float *values = ok ? malloc(output_elements * sizeof(*values)) : NULL;
    if (ok && !values) ok = 0;
    if (ok) ok = read_all(out_pipe[0], values, output_elements * sizeof(*values));
    if (ok) for (size_t index = 0; index < output_elements; index++)
        if (!isfinite(values[index])) { ok = 0; break; }
    close(out_pipe[0]);
    int status = 0;
    pid_t waited;
    do waited = waitpid(child, &status, 0); while (waited < 0 && errno == EINTR);
    if (!ok || waited != child || !WIFEXITED(status) || WEXITSTATUS(status)) {
        free(values);
        fail(error, error_size, "latent upscaler sidecar failed or returned malformed data");
        return 0;
    }
    *output = values;
    *output_time = (int)time; *output_height = (int)height; *output_width = (int)width;
    return 1;
}
