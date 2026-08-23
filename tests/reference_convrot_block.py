#!/usr/bin/env python3
"""Deterministic block-0 reference for Comfy TensorWiseINT8 ConvRot H3."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14336
ROWS = 128


def stats(value):
    value = value.float()
    return {
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
        "rms": value.square().mean().sqrt().item(),
    }


def regular_hadamard(device):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1],
         [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.bfloat16, device=device)
    result = h4
    while result.shape[0] < 256:
        result = torch.kron(result, h4)
    return result / 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--dump-intermediates", action="store_true",
                        help="write BF16 stage tensors for native parity checks")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    hadamard = regular_hadamard(device)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as store:
        def tensor(name, dtype=None):
            value = store.get_tensor(name).to(device)
            return value.to(dtype) if dtype is not None else value

        table = tensor("adaln_t_table", torch.float32)
        t_emb = table[512]
        modulation = (t_emb @ tensor(
            "blocks.0.adaln_proj.linear.weight", torch.float32).T + tensor(
            "blocks.0.adaln_proj.linear.bias", torch.float32)).reshape(3, 6, HIDDEN)
        norm1 = tensor("blocks.0.norm1.weight", torch.bfloat16)
        norm2 = tensor("blocks.0.norm2.weight", torch.bfloat16)
        q_norm = tensor("blocks.0.attn.q_norm.weight", torch.bfloat16)
        k_norm = tensor("blocks.0.attn.k_norm.weight", torch.bfloat16)

        weights = {}
        scales = {}
        for stem in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            weights[stem] = tensor(f"blocks.0.{stem}.weight", torch.int8)
            scales[stem] = tensor(
                f"blocks.0.{stem}.weight_scale", torch.float32).reshape(1, -1)

    def linear(x, stem):
        grouped = x.reshape(-1, x.shape[-1] // 256, 256)
        rotated = torch.matmul(grouped, hadamard).reshape_as(x)
        maximum = rotated.abs().amax(dim=-1, keepdim=True)
        scale = (maximum.float() / 127.0).clamp(min=1e-30)
        quantized = torch.round(rotated / scale.to(rotated.dtype)).clamp(
            -128, 127).to(torch.int8)
        if hasattr(torch, "int8_mm"):
            accumulated = torch.int8_mm(quantized, weights[stem].T.contiguous())
        else:
            accumulated = torch._int_mm(quantized, weights[stem].T.contiguous())
        return (accumulated.float() * (scale * scales[stem])).to(torch.bfloat16)

    def rms(x, weight):
        return F.rms_norm(x, (x.shape[-1],), weight, eps=1e-5)

    def adaln(x, weight, shift, scale):
        value = rms(x, weight)
        value.mul_(1 + scale.to(torch.bfloat16)).add_(shift.to(torch.bfloat16))
        return value

    host = torch.arange(ROWS * HIDDEN, dtype=torch.int64).reshape(ROWS, HIDDEN)
    x = (((host % 4001).float() - 2000) / 997).to(torch.bfloat16).to(device)
    input_bytes = x.cpu().contiguous().view(torch.uint16).numpy().tobytes()
    (output_dir / "input.bf16").write_bytes(input_bytes)
    def dump_bf16(name, value):
        if not args.dump_intermediates:
            return
        (output_dir / f"{name}.bf16").write_bytes(
            value.to(torch.bfloat16).cpu().contiguous().view(torch.uint16)
            .numpy().tobytes())

    report = {"input": stats(x), "modulation": stats(modulation)}

    h = adaln(x, norm1, modulation[0, 0], modulation[0, 1])
    report["attention_adaln"] = stats(h)
    dump_bf16("attention_adaln", h)
    qkv = linear(h, "attn.qkv_proj")
    report["qkv"] = stats(qkv)
    dump_bf16("qkv", qkv)
    q, k, v = qkv.split(INNER, dim=-1)
    q = rms(q.reshape(ROWS, HEADS, HEAD_DIM), q_norm)
    k = rms(k.reshape(ROWS, HEADS, HEAD_DIM), k_norm)
    v = v.reshape(ROWS, HEADS, HEAD_DIM)
    dump_bf16("query", q)
    dump_bf16("key", k)
    dump_bf16("value", v)
    attention = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        scale=HEAD_DIM ** -0.5)
    attention = attention.squeeze(0).transpose(0, 1).reshape(ROWS, INNER)
    dump_bf16("attention_heads", attention)
    qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores_f32 = torch.matmul(qh.float(), kh.float().transpose(-1, -2))
    scores_f32 *= HEAD_DIM ** -0.5
    manual_f32 = torch.matmul(torch.softmax(scores_f32, dim=-1), vh.float())
    dump_bf16("attention_heads_manual_f32", manual_f32.transpose(0, 1)
              .reshape(ROWS, INNER))
    probabilities_bf16 = torch.softmax(scores_f32, dim=-1).to(torch.bfloat16)
    manual_probability_bf16 = torch.matmul(probabilities_bf16.float(),
                                           vh.float())
    dump_bf16("attention_heads_probability_bf16", manual_probability_bf16
              .transpose(0, 1).reshape(ROWS, INNER))
    manual_score_f16 = torch.matmul(torch.softmax(scores_f32.to(torch.float16)
                                                  .float(), dim=-1), vh.float())
    dump_bf16("attention_heads_score_f16", manual_score_f16.transpose(0, 1)
              .reshape(ROWS, INNER))
    maximum_f32 = scores_f32.amax(-1, keepdim=True)
    probabilities_exp2 = torch.exp2(
        (scores_f32 - maximum_f32) * 1.4426950408889634)
    manual_exp2 = torch.matmul(probabilities_exp2 / probabilities_exp2.sum(
        -1, keepdim=True), vh.float())
    dump_bf16("attention_heads_exp2_f32", manual_exp2.transpose(0, 1)
              .reshape(ROWS, INNER))
    # FlashAttention-2 computes QK in FP32 then uses exp2 with the combined
    # float scale * log2(e).  Preserve that multiplication order separately:
    # the usual ``scores_f32`` path has already rounded after scale.
    scores_raw_f32 = torch.matmul(qh.float(), kh.float().transpose(-1, -2))
    flash_scale_log2 = (HEAD_DIM ** -0.5) * 1.4426950408889634
    probabilities_flash_exp2 = torch.exp2(
        scores_raw_f32 * flash_scale_log2 -
        (scores_raw_f32 * flash_scale_log2).amax(-1, keepdim=True))
    manual_flash_exp2 = torch.matmul(
        probabilities_flash_exp2 / probabilities_flash_exp2.sum(-1, keepdim=True),
        vh.float())
    dump_bf16("attention_heads_flashscale_exp2", manual_flash_exp2
              .transpose(0, 1).reshape(ROWS, INNER))
    q_flash_scaled = qh.float() * flash_scale_log2
    scores_qflash_scaled = torch.matmul(q_flash_scaled,
                                        kh.float().transpose(-1, -2))
    probabilities_qflash_scaled = torch.exp2(
        scores_qflash_scaled - scores_qflash_scaled.amax(-1, keepdim=True))
    manual_qflash_scaled = torch.matmul(
        probabilities_qflash_scaled /
        probabilities_qflash_scaled.sum(-1, keepdim=True), vh.float())
    dump_bf16("attention_heads_qflashscale_exp2", manual_qflash_scaled
              .transpose(0, 1).reshape(ROWS, INNER))

    def chunked_scores(chunk_size, pairwise):
        partials = [torch.matmul(qh.float()[..., start:start + chunk_size],
                                 kh.float()[..., start:start + chunk_size]
                                 .transpose(-1, -2))
                    for start in range(0, HEAD_DIM, chunk_size)]
        while pairwise and len(partials) > 1:
            partials = [left + right for left, right in
                        zip(partials[::2], partials[1::2])]
        if pairwise:
            return partials[0]
        total = partials[0]
        for partial in partials[1:]:
            total = total + partial
        return total

    for chunk_size in (16, 32, 64):
        for pairwise in (False, True):
            scores = chunked_scores(chunk_size, pairwise) * HEAD_DIM ** -0.5
            manual = torch.matmul(torch.softmax(scores, dim=-1), vh.float())
            name = f"attention_heads_qk{chunk_size}_{'pair' if pairwise else 'seq'}"
            dump_bf16(name, manual.transpose(0, 1).reshape(ROWS, INNER))

    # Keep the scores and softmax fixed while changing only the PV dot-product
    # accumulation tree.  This distinguishes Tensor Core PV accumulation from
    # QK/softmax effects in CUDA Flash SDPA.
    probabilities_f32 = torch.softmax(scores_f32, dim=-1)

    def chunked_values(chunk_size, pairwise):
        partials = [torch.matmul(probabilities_f32[..., start:start + chunk_size],
                                 vh.float()[..., start:start + chunk_size, :])
                    for start in range(0, ROWS, chunk_size)]
        if pairwise:
            while len(partials) > 1:
                partials = [left + right if index + 1 < len(partials) else left
                            for index, (left, right) in enumerate(
                                zip(partials[::2], partials[1::2] + [None]))]
            return partials[0]
        total = partials[0]
        for partial in partials[1:]:
            total = total + partial
        return total

    for chunk_size in (16, 32, 64):
        for pairwise in (False, True):
            manual = chunked_values(chunk_size, pairwise)
            name = f"attention_heads_pv{chunk_size}_{'pair' if pairwise else 'seq'}"
            dump_bf16(name, manual.transpose(0, 1).reshape(ROWS, INNER))
    scores_prescale_f32 = torch.matmul(
        qh.float() * (HEAD_DIM ** -0.5), kh.float().transpose(-1, -2))
    manual_prescale_f32 = torch.matmul(
        torch.softmax(scores_prescale_f32, dim=-1), vh.float())
    dump_bf16("attention_heads_prescale_f32", manual_prescale_f32
              .transpose(0, 1).reshape(ROWS, INNER))
    scores_bf16 = torch.matmul(qh, kh.transpose(-1, -2)) * HEAD_DIM ** -0.5
    manual_bf16 = torch.matmul(torch.softmax(scores_bf16, dim=-1), vh)
    dump_bf16("attention_heads_manual_bf16", manual_bf16.transpose(0, 1)
              .reshape(ROWS, INNER))
    scores_prescale_bf16 = torch.matmul(
        qh * (HEAD_DIM ** -0.5), kh.transpose(-1, -2))
    manual_prescale_bf16 = torch.matmul(
        torch.softmax(scores_prescale_bf16, dim=-1), vh)
    dump_bf16("attention_heads_prescale_bf16", manual_prescale_bf16
              .transpose(0, 1).reshape(ROWS, INNER))

    def online_attention(block_size):
        qf, kf, vf = qh.float(), kh.float(), vh.float()
        maximum = torch.full((*qf.shape[:-1], 1), -float("inf"),
                             device=device)
        normalizer = torch.zeros_like(maximum)
        accumulator = torch.zeros_like(qf)
        for start in range(0, ROWS, block_size):
            key_chunk = kf[:, start:start + block_size]
            value_chunk = vf[:, start:start + block_size]
            scores = torch.matmul(qf, key_chunk.transpose(-1, -2))
            scores *= HEAD_DIM ** -0.5
            next_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            probabilities = torch.exp(scores - next_maximum)
            accumulator = accumulator * torch.exp(maximum - next_maximum)
            accumulator += torch.matmul(probabilities, value_chunk)
            normalizer = normalizer * torch.exp(maximum - next_maximum)
            normalizer += probabilities.sum(-1, keepdim=True)
            maximum = next_maximum
        return accumulator / normalizer

    for block_size in (16, 32, 64, 128):
        online = online_attention(block_size)
        dump_bf16(f"attention_heads_online_{block_size}", online
                  .transpose(0, 1).reshape(ROWS, INNER))
    attention = linear(attention, "attn.out_proj")
    report["attention_output"] = stats(attention)
    dump_bf16("attention_output", attention)
    after_attention = torch.addcmul(
        x, attention, modulation[0, 2].to(torch.bfloat16))
    report["attention_residual"] = stats(after_attention)
    dump_bf16("attention_residual", after_attention)

    h = adaln(after_attention, norm2, modulation[0, 3], modulation[0, 4])
    report["mlp_adaln"] = stats(h)
    dump_bf16("mlp_adaln", h)
    fc1 = linear(h, "mlp.fc1")
    dump_bf16("fc1", fc1)
    gate, up = fc1.split(FFN, dim=-1)
    activated = F.silu(gate) * up
    report["swiglu"] = stats(activated)
    dump_bf16("swiglu", activated)
    mlp = linear(activated, "mlp.fc2")
    report["mlp_output"] = stats(mlp)
    dump_bf16("mlp_output", mlp)
    final = torch.addcmul(
        after_attention, mlp, modulation[0, 5].to(torch.bfloat16))
    report["output"] = stats(final)
    dump_bf16("output", final)
    (output_dir / "output.bf16").write_bytes(
        final.cpu().contiguous().view(torch.uint16).numpy().tobytes())
    (output_dir / "stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
