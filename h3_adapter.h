#ifndef H3_ADAPTER_H
#define H3_ADAPTER_H

#include "h3.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>

/* Parse only the stable user-facing profile names. */
int h3_adapter_profile_parse(const char *name, h3_adapter_profile *profile);
const char *h3_adapter_profile_name(h3_adapter_profile profile);

/* Set the trained sampler/scheduler/shift contract. ModelTC Turbo profiles
 * deliberately leave `steps` untouched so callers can evaluate the adapter
 * at a chosen count; their names retain the published nominal count for
 * provenance. PAI PDD still pins its published count. */
int h3_adapter_profile_apply(h3_params *params, char *error,
                             size_t error_size);

/* The published nominal count is a CLI default. ModelTC users may explicitly
 * choose another count, while PAI PDD may not. */
int h3_adapter_profile_default_steps(h3_adapter_profile profile);
int h3_adapter_profile_is_modeltc(h3_adapter_profile profile);

/* Published PDD plan: the N fine intervals are grouped into blocks of L.
 * The result has N entries, zero outside `coarse_step * L .. + L`, and sums
 * to one. It is independent for the video and audio shifted grids. */
int h3_adapter_pdd_plan(float shift, unsigned fine_steps,
                        unsigned block_size, unsigned coarse_step,
                        float *plan, size_t plan_count,
                        char *error, size_t error_size);

typedef struct h3_adapter_runtime h3_adapter_runtime;

/* Parse and validate the full published factor/key schema without allocating
 * a second model copy.  The returned header remains available to native DiT
 * loading code for direct factor uploads. */
h3_adapter_runtime *h3_adapter_runtime_open(const char *path,
                                             h3_adapter_kind kind,
                                             float strength,
                                             char *error,
                                             size_t error_size);
void h3_adapter_runtime_free(h3_adapter_runtime *adapter);
unsigned h3_adapter_runtime_rank(const h3_adapter_runtime *adapter);
h3_adapter_kind h3_adapter_runtime_kind(const h3_adapter_runtime *adapter);
float h3_adapter_runtime_strength(const h3_adapter_runtime *adapter);
const h3_st_tensor *h3_adapter_runtime_tensor(
    const h3_adapter_runtime *adapter, const char *name);
const h3_st_header *h3_adapter_runtime_header(
    const h3_adapter_runtime *adapter);
/* Direct shared-Metal upload of a prevalidated BF16 factor. */
h3_gpu_tensor *h3_adapter_runtime_load_bf16(
    const h3_adapter_runtime *adapter, h3_gpu *gpu, const char *name,
    uint64_t rows, uint64_t columns, char *error, size_t error_size);

#endif
