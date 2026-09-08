/* Private GPU seam: isolate rotation/quantization from the following GEMM.
 * Build this translation unit alone with the usual Objective-C frameworks.
 * Usage: h3_convrot_kernels_bench [rows=37680] [--gemm|--sdpa]
 * Projection and attention comparisons use one cold and three interleaved
 * warm calls per variant. The default measures rotation/quantization alone. */
#include "../h3_gpu.m"

static void check(int ok, h3_gpu *gpu) {
    if (!ok) {
        fprintf(stderr, "ConvRot quantizer probe: %s\n", h3_gpu_error(gpu));
        exit(1);
    }
}

static void measure_width(h3_gpu *gpu, uint32_t rows, uint32_t columns) {
    size_t count = (size_t)rows * columns;
    size_t padded = (rows + 127u) & ~127u;
    uint16_t *input = malloc(count * sizeof(*input));
    check(input != NULL, gpu);
    uint32_t rng = 12345;
    for (size_t i = 0; i < count; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float value = ((float)(rng >> 8) / 8388608.0f - 1.0f) * 3.0f;
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        input[i] = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    }
    h3_gpu_tensor *x = h3_gpu_tensor_from_bf16(gpu, input, count);
    h3_gpu_tensor *rotated = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(gpu, padded * columns);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, padded);
    check(x && rotated && quantized && scales, gpu);
    free(input);
    size_t rotation_bytes = count * sizeof(uint16_t);
    size_t quantized_bytes = padded * columns;
    size_t scale_bytes = padded * sizeof(float);
    void *reference_rotation = malloc(rotation_bytes);
    void *reference_quantized = malloc(quantized_bytes);
    void *reference_scales = malloc(scale_bytes);
    check(reference_rotation && reference_quantized && reference_scales, gpu);
    for (unsigned i = 0; i < 4; i++) {
        h3_gpu_stats before, after;
        check(h3_gpu_get_stats(gpu, &before), gpu);
        double started = h3_gpu_now();
        check(h3_gpu_begin(gpu), gpu);
        check(h3_gpu_convrot_quantize_bf16_int8_rows(
            gpu, rotated, quantized, scales, x, rows, columns), gpu);
        check(h3_gpu_submit(gpu), gpu);
        double elapsed = h3_gpu_now() - started;
        check(h3_gpu_get_stats(gpu, &after), gpu);
        void *rotation_data = TENSOR(rotated).buffer.contents;
        void *quantized_data = TENSOR(quantized).buffer.contents;
        void *scale_data = TENSOR(scales).buffer.contents;
        if (i == 0) {
            memcpy(reference_rotation, rotation_data, rotation_bytes);
            memcpy(reference_quantized, quantized_data, quantized_bytes);
            memcpy(reference_scales, scale_data, scale_bytes);
        } else if (memcmp(reference_rotation, rotation_data, rotation_bytes) ||
                   memcmp(reference_quantized, quantized_data, quantized_bytes) ||
                   memcmp(reference_scales, scale_data, scale_bytes)) {
            fprintf(stderr, "ConvRot output bytes changed: rows=%u width=%u\n",
                    rows, columns);
            const unsigned char *want[] = {reference_rotation, reference_quantized, reference_scales};
            const unsigned char *got[] = {rotation_data, quantized_data, scale_data};
            size_t sizes[] = {rotation_bytes, quantized_bytes, scale_bytes};
            const char *labels[] = {"rotation", "quantized", "scales"};
            for (unsigned part = 0; part < 3; part++) {
                size_t differences = 0, first = 0;
                for (size_t j = 0; j < sizes[part]; j++) if (want[part][j] != got[part][j]) {
                    if (!differences) first = j;
                    differences++;
                }
                fprintf(stderr, "%s bytes changed=%zu first=%zu reference=%u actual=%u\n",
                        labels[part], differences, first, want[part][first], got[part][first]);
            }
            exit(1);
        }
        printf("quant rows=%u width=%u variant=%s phase=%s wall_ms=%.4f gpu_ms=%.4f exact=yes\n",
               rows, columns, "current",
               i == 0 ? "cold" : "warm", elapsed * 1000,
               (after.gpu_seconds - before.gpu_seconds) * 1000);
    }
    free(reference_rotation); free(reference_quantized); free(reference_scales);
    h3_gpu_tensor_free(x); h3_gpu_tensor_free(rotated);
    h3_gpu_tensor_free(quantized); h3_gpu_tensor_free(scales);
}

static void measure_gemm(h3_gpu *gpu, uint32_t rows, uint32_t columns,
                          uint32_t outputs, NSString *kernel) {
    size_t padded = (rows + 127u) & ~127u;
    size_t x_count = padded * columns, w_count = (size_t)outputs * columns;
    size_t y_count = (size_t)rows * outputs;
    int8_t *x_values = malloc(x_count), *w_values = malloc(w_count);
    float *x_scales = malloc(padded * sizeof(float));
    float *w_scales = malloc(outputs * sizeof(float));
    uint16_t *reference = malloc(y_count * sizeof(uint16_t));
    check(x_values && w_values && x_scales && w_scales && reference, gpu);
    uint32_t rng = 8765;
    for (size_t i = 0; i < x_count; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        x_values[i] = (int8_t)(rng & 255);
    }
    for (size_t i = 0; i < w_count; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        w_values[i] = (int8_t)(rng & 255);
    }
    for (size_t i = 0; i < padded; i++) x_scales[i] = 0.001f + (float)(i % 23) * 0.0001f;
    for (size_t i = 0; i < outputs; i++) w_scales[i] = 0.002f + (float)(i % 31) * 0.0001f;
    h3_gpu_tensor *x = h3_gpu_tensor_from_i8(gpu, x_values, x_count);
    h3_gpu_tensor *w = h3_gpu_tensor_from_i8(gpu, w_values, w_count);
    h3_gpu_tensor *xs = h3_gpu_tensor_from_f32(gpu, x_scales, padded);
    h3_gpu_tensor *ws = h3_gpu_tensor_from_f32(gpu, w_scales, outputs);
    h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, y_count);
    check(x && w && xs && ws && y, gpu);
    free(x_values); free(w_values); free(x_scales); free(w_scales);
    const int pattern[] = {0, 1, 0, 1, 1, 0, 0, 1};
    for (unsigned i = 0; i < 8; i++) {
        h3_gpu_stats before, after;
        check(h3_gpu_get_stats(gpu, &before), gpu);
        double started = h3_gpu_now();
        check(h3_gpu_begin(gpu), gpu);
        if (!pattern[i]) {
            setenv("H3_DISABLE_CONVROT_FULL_K", "1", 1);
            check(h3_gpu_linear_int8_quantized_bf16(gpu, y, x, xs, w, ws,
                                                   rows, columns, outputs), gpu);
            unsetenv("H3_DISABLE_CONVROT_FULL_K");
        } else {
            H3GPU *object = GPU(gpu);
            id<MTLComputeCommandEncoder> encoder = [object.command computeCommandEncoder];
            [encoder setComputePipelineState:h3_gpu_pipeline(object,
                kernel)];
            [encoder setBuffer:TENSOR(x).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(w).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(xs).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(ws).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(y).buffer offset:0 atIndex:4];
            linear_args args = {rows, columns, outputs, 0, 0};
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(padded / 128 * (outputs / 128), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
        check(h3_gpu_submit(gpu), gpu);
        double elapsed = h3_gpu_now() - started;
        check(h3_gpu_get_stats(gpu, &after), gpu);
        if (!i) memcpy(reference, TENSOR(y).buffer.contents, y_count * sizeof(uint16_t));
        else if (memcmp(reference, TENSOR(y).buffer.contents, y_count * sizeof(uint16_t))) {
            fprintf(stderr, "ConvRot FC2 GEMM output bytes changed\n"); exit(1);
        }
        printf("gemm rows=%u k=%u n=%u variant=%s phase=%s wall_ms=%.4f gpu_ms=%.4f exact=yes\n",
               rows, columns, outputs, pattern[i] ? "full-k" : "current", i < 2 ? "cold" : "warm",
               elapsed * 1000, (after.gpu_seconds - before.gpu_seconds) * 1000);
    }
    free(reference);
    h3_gpu_tensor_free(x); h3_gpu_tensor_free(w); h3_gpu_tensor_free(xs);
    h3_gpu_tensor_free(ws); h3_gpu_tensor_free(y);
}

static void measure_sdpa(h3_gpu *gpu, uint32_t rows) {
    const uint32_t heads = 56, width = 128;
    size_t count = (size_t)rows * heads * width;
    h3_gpu_tensor *row[3], *head[3];
    uint16_t *values = malloc(count * sizeof(*values));
    uint16_t *transposed = malloc(count * sizeof(*transposed));
    uint16_t *reference = malloc(count * sizeof(*reference));
    check(values && transposed && reference, gpu);
    uint32_t rng = 97531;
    for (unsigned part = 0; part < 3; part++) {
        for (size_t i = 0; i < count; i++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            float value = ((float)(rng >> 8) / 8388608.0f - 1.0f);
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            values[i] = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        }
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t h = 0; h < heads; h++)
                memcpy(transposed + ((size_t)h * rows + r) * width,
                       values + ((size_t)r * heads + h) * width,
                       width * sizeof(*values));
        row[part] = h3_gpu_tensor_from_bf16(gpu, values, count);
        head[part] = h3_gpu_tensor_from_bf16(gpu, transposed, count);
        check(row[part] && head[part], gpu);
    }
    free(values); free(transposed);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, count);
    check(output != NULL, gpu);
    const int pattern[] = {0, 1, 0, 1, 1, 0, 0, 1};
    for (unsigned i = 0; i < 8; i++) {
        h3_gpu_tensor **inputs = pattern[i] ? head : row;
        check(h3_gpu_begin(gpu), gpu);
        GPU(gpu).headMajorSDPAInputs = pattern[i] != 0;
        double start = h3_gpu_now();
        check(h3_gpu_sdpa_bf16(gpu, output, inputs[0], inputs[1], inputs[2],
                               rows, heads, width, 1.0f / sqrtf(128.0f)), gpu);
        check(h3_gpu_submit(gpu), gpu);
        double elapsed = h3_gpu_now() - start;
        if (!i) memcpy(reference, TENSOR(output).buffer.contents, count * sizeof(*reference));
        size_t differences = 0;
        uint16_t *actual = TENSOR(output).buffer.contents;
        double squared_error = 0, squared_reference = 0;
        for (size_t j = 0; j < count; j++) {
            differences += actual[j] != reference[j];
            uint32_t a_bits = (uint32_t)actual[j] << 16;
            uint32_t b_bits = (uint32_t)reference[j] << 16;
            float a, b;
            memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4);
            check(isfinite(a), gpu);
            squared_error += ((double)a-b)*((double)a-b);
            squared_reference += (double)b*b;
        }
        printf("sdpa rows=%u variant=%s phase=%s wall_ms=%.4f changed=%zu relative_l2=%.9g\n",
               rows, pattern[i] ? "head-major" : "row-major", i < 2 ? "cold" : "warm",
               elapsed * 1000, differences, sqrt(squared_error / squared_reference));
    }
    free(reference);
    for (unsigned part = 0; part < 3; part++) {
        h3_gpu_tensor_free(row[part]); h3_gpu_tensor_free(head[part]);
    }
    h3_gpu_tensor_free(output);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    char *end = NULL;
    unsigned long parsed = argc > 1 ? strtoul(argv[1], &end, 10) : 37680;
    if (argc > 3 || !parsed || parsed > 100000 || (end && *end) ||
        (argc == 3 && strcmp(argv[2], "--gemm") && strcmp(argv[2], "--sdpa"))) return 2;
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    if (argc == 3 && !strcmp(argv[2], "--sdpa")) measure_sdpa(gpu, (uint32_t)parsed);
    else if (argc == 3) {
        measure_gemm(gpu, (uint32_t)parsed, 14336, 5376,
                     @"h3_linear_int8_nax_r128_full_k14336");
        measure_gemm(gpu, (uint32_t)parsed, 5376, 21504,
                     @"h3_linear_int8_nax_r128_full_k5376_n21504");
        measure_gemm(gpu, (uint32_t)parsed, 7168, 5376,
                     @"h3_linear_int8_nax_r128_full_k7168_n5376");
        measure_gemm(gpu, (uint32_t)parsed, 5376, 28672,
                     @"h3_linear_int8_nax_r128_full_k5376_n28672");
    }
    else {
        const uint32_t widths[] = {5376, 7168, 14336};
        for (size_t i = 0; i < sizeof(widths)/sizeof(*widths); i++)
            measure_width(gpu, (uint32_t)parsed, widths[i]);
    }
    h3_gpu_free(gpu);
    return 0;
}
