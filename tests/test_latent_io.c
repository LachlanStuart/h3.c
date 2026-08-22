#include "h3_latent_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *message) {
    fprintf(stderr, "latent io test: %s\n", message);
    exit(1);
}

int main(void) {
    char path[] = "/tmp/h3-latent-io-XXXXXX";
    int descriptor = mkstemp(path);
    if (descriptor < 0 || close(descriptor) != 0) die("cannot create fixture");

    enum { TIME = 7, HEIGHT = 2, WIDTH = 3 };
    size_t elements = (size_t)24 * TIME * HEIGHT * WIDTH;
    float *values = malloc(elements * sizeof(*values));
    if (!values) die("out of memory");
    for (size_t index = 0; index < elements; index++)
        values[index] = (float)index / 127.0f - 2.0f;

    char error[512];
    if (!h3_video_latent_file_write(path, values, TIME, HEIGHT, WIDTH,
                                    error, sizeof(error))) die(error);
    h3_video_latent latent;
    if (!h3_video_latent_file_read(path, &latent, error, sizeof(error)))
        die(error);
    if (latent.time != TIME || latent.height != HEIGHT ||
        latent.width != WIDTH ||
        memcmp(latent.values, values, elements * sizeof(*values)))
        die("round trip changed latent geometry or values");
    h3_video_latent_free(&latent);

    values[13] = NAN;
    if (h3_video_latent_file_write(path, values, TIME, HEIGHT, WIDTH,
                                   error, sizeof(error)))
        die("accepted a non-finite latent");
    values[13] = 0.0f;

    FILE *file = fopen(path, "wb");
    if (!file || fwrite("H3LATF32", 1, 8, file) != 8 || fclose(file) != 0)
        die("cannot create truncated fixture");
    if (h3_video_latent_file_read(path, &latent, error, sizeof(error)))
        die("accepted a truncated latent");

    unlink(path);
    free(values);
    puts("latent io: ok");
    return 0;
}
