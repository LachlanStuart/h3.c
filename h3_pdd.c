#include "h3_pdd.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* IEEE round-to-nearest, ties-to-even, matching Metal's BF16 cast. */
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int h3_pdd_require_full_bf16_base(int convrot, char *error,
                                  size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!convrot) return 1;
    fail(error, error_size,
         "Alibaba PAI PDD requires a full BF16 MiniMax-H3 checkpoint; "
         "ConvRot is unsupported");
    return 0;
}

int h3_pdd_head_fuse_bf16(const uint16_t *weights, const uint16_t *biases,
                          unsigned steps, uint32_t output_dim,
                          uint32_t input_dim, const float *plan,
                          size_t plan_count, uint16_t *fused_weights,
                          uint16_t *fused_biases, char *error,
                          size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !biases || !plan || !fused_weights || !fused_biases ||
        !steps || !output_dim || !input_dim || plan_count < steps) {
        fail(error, error_size, "invalid PDD final-head fusion arguments");
        return 0;
    }
    if ((size_t)output_dim > SIZE_MAX / input_dim) {
        fail(error, error_size, "PDD final-head dimensions overflow");
        return 0;
    }
    size_t head_weights = (size_t)output_dim * input_dim;
    if ((size_t)steps > SIZE_MAX / head_weights) {
        fail(error, error_size, "PDD final-head dimensions overflow");
        return 0;
    }

    double plan_sum = 0.0;
    for (unsigned step = 0; step < steps; step++) {
        if (!isfinite(plan[step]) || plan[step] < 0.0f) {
            fail(error, error_size, "PDD final-head plan is not nonnegative");
            return 0;
        }
        plan_sum += plan[step];
    }
    if (!isfinite(plan_sum) || fabs(plan_sum - 1.0) > 1e-4) {
        fail(error, error_size,
             "PDD final-head plan must have coefficients summing to one");
        return 0;
    }

    for (size_t element = 0; element < head_weights; element++) {
        float sum = 0.0f;
        for (unsigned step = 0; step < steps; step++) {
            float coefficient = bf16_to_f32(f32_to_bf16(plan[step]));
            sum = fmaf(coefficient,
                       bf16_to_f32(weights[(size_t)step * head_weights +
                                           element]), sum);
        }
        fused_weights[element] = f32_to_bf16(sum);
    }
    for (uint32_t output = 0; output < output_dim; output++) {
        float sum = 0.0f;
        for (unsigned step = 0; step < steps; step++) {
            float coefficient = bf16_to_f32(f32_to_bf16(plan[step]));
            sum = fmaf(coefficient,
                       bf16_to_f32(biases[(size_t)step * output_dim + output]),
                       sum);
        }
        fused_biases[output] = f32_to_bf16(sum);
    }
    return 1;
}
