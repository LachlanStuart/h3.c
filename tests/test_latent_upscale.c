#include "h3_latent_upscale.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_script(int descriptor, const char *source) {
    assert(!ftruncate(descriptor, 0));
    assert(lseek(descriptor, 0, SEEK_SET) == 0);
    size_t length = strlen(source);
    assert(write(descriptor, source, length) == (ssize_t)length);
    assert(lseek(descriptor, 0, SEEK_SET) == 0);
}

static int invoke(const char *script, float **output, int *time,
                  int *height, int *width) {
    float input[24 * 2 * 2 * 2];
    for (size_t index = 0; index < sizeof(input) / sizeof(input[0]); index++)
        input[index] = (float)index / 17.0f;
    char error[256];
    return h3_latent_upscale_pipe("/usr/bin/python3", script, "source.py",
                                  "checkpoint.safetensors", input, 2, 2, 2,
                                  output, time, height, width, error, sizeof(error));
}

/* The fake child covers framing and all bridge failure paths without Torch/MPS. */
int main(void) {
    char path[] = "/tmp/h3-upscale-XXXXXX";
    int descriptor = mkstemp(path);
    assert(descriptor >= 0);
    static const char good[] =
        "import struct,sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(struct.pack('<8sIII',b'H3LATF32',2,4,4))\n"
        "sys.stdout.buffer.write(struct.pack('<%df'%(24*2*4*4),*([1.0]*(24*2*4*4))))\n";
    write_script(descriptor, good);
    float *output = NULL;
    int time = 0, height = 0, width = 0;
    assert(invoke(path, &output, &time, &height, &width));
    assert(time == 2 && height == 4 && width == 4);
    assert(output[0] == 1.0f && output[24 * 2 * 4 * 4 - 1] == 1.0f);
    free(output);

    static const char early_exit[] = "import sys;sys.exit(2)\n";
    write_script(descriptor, early_exit);
    output = NULL;
    assert(!invoke(path, &output, &time, &height, &width));
    assert(!output);

    static const char malformed[] =
        "import struct,sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(struct.pack('<8sIII',b'H3LATF32',2,3,4))\n";
    write_script(descriptor, malformed);
    assert(!invoke(path, &output, &time, &height, &width));

    static const char nonfinite[] =
        "import struct,sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(struct.pack('<8sIII',b'H3LATF32',2,4,4))\n"
        "sys.stdout.buffer.write(struct.pack('<%df'%(24*2*4*4),*([float('nan')]*(24*2*4*4))))\n";
    write_script(descriptor, nonfinite);
    assert(!invoke(path, &output, &time, &height, &width));
    assert(!output);
    assert(!close(descriptor));
    assert(!unlink(path));
    return 0;
}
