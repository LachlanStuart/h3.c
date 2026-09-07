#include "h3_latent_io.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define H3_LATENT_CHANNELS 24u
#define H3_LATENT_HEADER_BYTES 20u

static void fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static void fail_path(char *error, size_t error_size,
                      const char *action, const char *path) {
    if (error && error_size)
        snprintf(error, error_size, "cannot %s latent %s: %s",
                 action, path, strerror(errno));
}

static void put_u32_le(unsigned char *destination, uint32_t value) {
    destination[0] = (unsigned char)(value & 0xffu);
    destination[1] = (unsigned char)((value >> 8) & 0xffu);
    destination[2] = (unsigned char)((value >> 16) & 0xffu);
    destination[3] = (unsigned char)((value >> 24) & 0xffu);
}

static uint32_t get_u32_le(const unsigned char *source) {
    return (uint32_t)source[0] |
           (uint32_t)source[1] << 8 |
           (uint32_t)source[2] << 16 |
           (uint32_t)source[3] << 24;
}

static int dimensions_valid(uint32_t time, uint32_t height, uint32_t width,
                            size_t *elements) {
    if (time < 2 || !height || !width ||
        time > INT32_MAX || height > INT32_MAX || width > INT32_MAX)
        return 0;
    size_t count = H3_LATENT_CHANNELS;
    const uint32_t dimensions[3] = {time, height, width};
    for (size_t index = 0; index < 3; index++) {
        if (count > SIZE_MAX / dimensions[index]) return 0;
        count *= dimensions[index];
    }
    if (count > SIZE_MAX / sizeof(float)) return 0;
    *elements = count;
    return 1;
}

static int host_is_little_endian(void) {
    const uint16_t value = 1;
    return *(const unsigned char *)&value == 1;
}

int h3_video_latent_file_write(const char *path,
                               const float *values,
                               int time, int height, int width,
                               char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    size_t elements = 0;
    if (!path || !*path || !values || time < 0 || height < 0 || width < 0 ||
        !dimensions_valid((uint32_t)time, (uint32_t)height, (uint32_t)width,
                          &elements) || !host_is_little_endian()) {
        fail(error, error_size, "invalid H3 latent write arguments");
        return 0;
    }
    for (size_t index = 0; index < elements; index++) {
        if (!isfinite(values[index])) {
            fail(error, error_size, "refusing to write non-finite H3 latent");
            return 0;
        }
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        fail_path(error, error_size, "create", path);
        return 0;
    }
    unsigned char header[H3_LATENT_HEADER_BYTES] = {
        'H', '3', 'L', 'A', 'T', 'F', '3', '2'
    };
    put_u32_le(header + 8, (uint32_t)time);
    put_u32_le(header + 12, (uint32_t)height);
    put_u32_le(header + 16, (uint32_t)width);
    int ok = fwrite(header, 1, sizeof(header), file) == sizeof(header) &&
        fwrite(values, sizeof(*values), elements, file) == elements;
    if (fclose(file) != 0) ok = 0;
    if (!ok) {
        fail_path(error, error_size, "write", path);
        return 0;
    }
    return 1;
}

int h3_video_latent_file_read(const char *path,
                              h3_video_latent *latent,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (latent) memset(latent, 0, sizeof(*latent));
    if (!path || !*path || !latent || !host_is_little_endian()) {
        fail(error, error_size, "invalid H3 latent read arguments");
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail_path(error, error_size, "open", path);
        return 0;
    }
    struct stat status;
    unsigned char header[H3_LATENT_HEADER_BYTES];
    if (fstat(fileno(file), &status) != 0 ||
        fread(header, 1, sizeof(header), file) != sizeof(header)) {
        fail_path(error, error_size, "read", path);
        fclose(file);
        return 0;
    }
    uint32_t time = get_u32_le(header + 8);
    uint32_t height = get_u32_le(header + 12);
    uint32_t width = get_u32_le(header + 16);
    size_t elements = 0;
    if (memcmp(header, "H3LATF32", 8) ||
        !dimensions_valid(time, height, width, &elements) ||
        status.st_size < 0 ||
        (uint64_t)status.st_size != H3_LATENT_HEADER_BYTES +
            (uint64_t)elements * sizeof(float)) {
        fail(error, error_size, "malformed H3 latent file");
        fclose(file);
        return 0;
    }
    float *values = malloc(elements * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading H3 latent");
        fclose(file);
        return 0;
    }
    size_t read_elements = fread(values, sizeof(*values), elements, file);
    int close_status = fclose(file);
    int ok = read_elements == elements && close_status == 0;
    if (!ok) {
        free(values);
        fail_path(error, error_size, "read", path);
        return 0;
    }
    for (size_t index = 0; index < elements; index++) {
        if (!isfinite(values[index])) {
            free(values);
            fail(error, error_size, "H3 latent contains non-finite values");
            return 0;
        }
    }
    latent->time = (int)time;
    latent->height = (int)height;
    latent->width = (int)width;
    latent->values = values;
    return 1;
}
