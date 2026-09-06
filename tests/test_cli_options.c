#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

static int failures;

static void expect_command(const char *arguments, int expected_status,
                           const char *expected_text) {
    char command[2048];
    int length = snprintf(command, sizeof(command),
        "./h3 %s 2>&1", arguments);
    if (length < 0 || (size_t)length >= sizeof(command)) {
        fprintf(stderr, "test command is too long\n");
        failures++;
        return;
    }
    FILE *pipe = popen(command, "r");
    if (!pipe) {
        perror("popen");
        failures++;
        return;
    }
    char output[8192] = {0};
    size_t used = 0;
    while (used + 1 < sizeof(output)) {
        size_t count = fread(output + used, 1, sizeof(output) - used - 1,
                             pipe);
        used += count;
        if (!count) break;
    }
    int status = pclose(pipe);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != expected_status ||
        !strstr(output, expected_text)) {
        fprintf(stderr, "unexpected CLI result for: %s\n", arguments);
        fprintf(stderr, "status=%d output=%s\n", status, output);
        failures++;
    }
}

int main(void) {
    /* These all stop during CLI validation, before a model or Metal is used. */
    expect_command("--help", 0, "--inline-production");
    expect_command("-d /not-a-model -p test --adapter fake.safetensors "
                   "--adapter-profile modeltc-fl2va-544-4 --scheduler beta",
                   2, "conflicts with an explicitly supplied");
    expect_command("-d /not-a-model -p test --adapter-strength 0", 2,
                   "require --adapter-profile");
    expect_command("-d /not-a-model -p test --target-width 1024", 2,
                   "require --inline-production");
    expect_command("-d /not-a-model -p test --inline-production --audio-only",
                   2, "cannot be combined with --audio-only");
    expect_command("-d /not-a-model -p test --inline-production "
                   "--target-width 1024 --target-height 576 "
                   "--upscaler-script up.py --upscaler-source source "
                   "--upscaler-checkpoint checkpoint --refine-video input.mp4",
                   2, "--refine-video");
    return failures ? 1 : 0;
}
