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
    report = {"input": stats(x), "modulation": stats(modulation)}

    h = adaln(x, norm1, modulation[0, 0], modulation[0, 1])
    report["attention_adaln"] = stats(h)
    qkv = linear(h, "attn.qkv_proj")
    report["qkv"] = stats(qkv)
    q, k, v = qkv.split(INNER, dim=-1)
    q = rms(q.reshape(ROWS, HEADS, HEAD_DIM), q_norm)
    k = rms(k.reshape(ROWS, HEADS, HEAD_DIM), k_norm)
    v = v.reshape(ROWS, HEADS, HEAD_DIM)
    attention = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        scale=HEAD_DIM ** -0.5)
    attention = attention.squeeze(0).transpose(0, 1).reshape(ROWS, INNER)
    attention = linear(attention, "attn.out_proj")
    report["attention_output"] = stats(attention)
    after_attention = torch.addcmul(
        x, attention, modulation[0, 2].to(torch.bfloat16))
    report["attention_residual"] = stats(after_attention)

    h = adaln(after_attention, norm2, modulation[0, 3], modulation[0, 4])
    report["mlp_adaln"] = stats(h)
    fc1 = linear(h, "mlp.fc1")
    gate, up = fc1.split(FFN, dim=-1)
    activated = F.silu(gate) * up
    report["swiglu"] = stats(activated)
    mlp = linear(activated, "mlp.fc2")
    report["mlp_output"] = stats(mlp)
    final = torch.addcmul(
        after_attention, mlp, modulation[0, 5].to(torch.bfloat16))
    report["output"] = stats(final)
    (output_dir / "output.bf16").write_bytes(
        final.cpu().contiguous().view(torch.uint16).numpy().tobytes())
    (output_dir / "stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
