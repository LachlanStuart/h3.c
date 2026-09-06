#ifndef H3_PDD_H
#define H3_PDD_H

#include <stddef.h>
#include <stdint.h>

/* Published Alibaba PAI PDD checkpoint schema.  The final heads are complete
 * per-interval linear heads, not low-rank deltas. */
enum {
    H3_PDD_HEAD_STEPS = 32u,
    H3_PDD_VIDEO_OUTPUTS = 96u,
    H3_PDD_AUDIO_OUTPUTS = 32u,
    H3_PDD_HEAD_HIDDEN = 5376u
};

#define H3_PDD_VIDEO_WEIGHT_NAME "proj_out.weight"
#define H3_PDD_VIDEO_BIAS_NAME "proj_out.bias"
#define H3_PDD_AUDIO_WEIGHT_NAME "audio_proj_out.weight"
#define H3_PDD_AUDIO_BIAS_NAME "audio_proj_out.bias"

/* PDD final heads are presently defined only for the released full-BF16
 * transformer.  ConvRot replaces the backbone's dense paths and its runtime
 * adapter support has not established an equivalent PDD execution contract. */
int h3_pdd_require_full_bf16_base(int convrot, char *error,
                                  size_t error_size);

/* Fuse a block's complete PDD heads into one BF16 linear head. `weights` has
 * row-major shape [steps, output_dim, input_dim], `biases` has
 * [steps, output_dim], and `plan` has `steps` independently-normalized PDD
 * coefficients. The caller owns the output buffers, which may be the DiT's
 * existing BF16 final-head buffers. The published implementation casts its
 * plan to the BF16 head dtype; this helper does the same, accumulates in F32,
 * then rounds once to BF16, avoiding repeated re-quantization while a PDD
 * block is armed. */
int h3_pdd_head_fuse_bf16(const uint16_t *weights, const uint16_t *biases,
                          unsigned steps, uint32_t output_dim,
                          uint32_t input_dim, const float *plan,
                          size_t plan_count, uint16_t *fused_weights,
                          uint16_t *fused_biases, char *error,
                          size_t error_size);

#endif
