#!/usr/bin/env python3
"""Fold a MiniMax-H3 LoRA into the bf16 checkpoint by in-place byte patching.

h3.c has no LoRA runtime, and does not need one: a LoRA is W' = W + scale*(B@A),
which can be baked into the checkpoint once, offline. This tool clones the
original shards (copy-on-write where the filesystem supports it) and rewrites
only the byte ranges of the targeted tensors, so headers, tensor order, and
alignment stay byte-identical to the originals.

Folding the 4-step Turbo distillation adapter this way enables 5-6 step
sampling at zero runtime cost. Measured on an M5 Max (960x544, identical
prompt/seed): 39 frames 87s -> 65s and a 5s clip 8.8min -> 6.2min versus the
--steps 20 --reuse 2 --layers 45 preset, at comparable visual quality.

Requires only numpy. Adapter tensors must be named <target>.lora_A.weight /
<target>.lora_B.weight over the checkpoint's own key space, as
larryvrh/MiniMax-H3-Turbo-Lora's packaged .safetensors files are.

Usage:
  python3 tools/fold_turbo_lora.py \
      --checkpoint MiniMax-H3/FL2VA/transformer \
      --lora minimax_h3_turbo_v4_step600_ema.safetensors \
      --out MiniMax-H3-turbo/FL2VA/transformer
Then point h3 at a model directory whose transformer is the folded tree and
sample with --steps 5 or 6 (4 shows motion smear; do not combine with --reuse).
"""
import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MAX_HEADER_BYTES = 100 * 1024 * 1024


def read_header(path):
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise SystemExit(f"{path}: truncated safetensors prefix")
        (hlen,) = struct.unpack("<Q", prefix)
        file_size = path.stat().st_size
        if hlen > MAX_HEADER_BYTES or hlen > file_size - 8:
            raise SystemExit(f"{path}: invalid safetensors header size {hlen}")
        payload = f.read(hlen)
        if len(payload) != hlen:
            raise SystemExit(f"{path}: truncated safetensors header")
        header = json.loads(payload)
    header.pop("__metadata__", None)
    return hlen, header


def validate_tensor(path, key, info, base):
    try:
        shape = info["shape"]
        start, end = info["data_offsets"]
        dtype = info["dtype"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{path}: malformed metadata for {key}: {exc}") from exc
    if (not isinstance(shape, list) or
            any(not isinstance(size, int) or size < 0 for size in shape) or
            not isinstance(start, int) or not isinstance(end, int) or
            start < 0 or end < start):
        raise SystemExit(f"{path}: invalid shape or offsets for {key}")
    item_bytes = {"BF16": 2, "F32": 4}.get(dtype)
    if item_bytes is None:
        raise SystemExit(f"{path}: unsupported dtype {dtype} for {key}")
    elements = math.prod(shape)
    if end - start != elements * item_bytes or base + end > path.stat().st_size:
        raise SystemExit(f"{path}: byte range does not match {key} metadata")
    return shape


def load_tensor(path, info, base):
    start, end = info["data_offsets"]
    dtype = info["dtype"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).copy()
    raise SystemExit(f"unsupported dtype {dtype} for {path}")


def f32_to_bf16_bytes(x):
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16  # round to nearest even
    return rounded.astype("<u2").tobytes()


def clone(src, dst):
    copy_program = Path("/bin/cp")
    if (sys.platform != "darwin" or not copy_program.is_file() or
            subprocess.run([str(copy_program), "-c", str(src), str(dst)],
                           capture_output=True).returncode != 0):
        shutil.copy2(src, dst)


def preflight(checkpoint, lora_path, lora_hdr, lora_base, pairs):
    shards = sorted(checkpoint.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {checkpoint}")
    targets = {}
    for shard in shards:
        hlen, header = read_header(shard)
        base = 8 + hlen
        for key in header.keys() & pairs.keys():
            if key in targets:
                raise SystemExit(f"{key}: target appears in multiple shards")
            validate_tensor(shard, key, header[key], base)
            targets[key] = header[key]
    missing = pairs.keys() - targets.keys()
    if missing:
        raise SystemExit("adapter targets missing from checkpoint: " +
                         ", ".join(sorted(missing)))
    for target, (a_key, b_key) in pairs.items():
        if b_key not in lora_hdr:
            raise SystemExit(f"{target}: missing adapter tensor {b_key}")
        a_shape = validate_tensor(
            lora_path, a_key, lora_hdr[a_key], lora_base)
        b_shape = validate_tensor(
            lora_path, b_key, lora_hdr[b_key], lora_base)
        target_shape = targets[target]["shape"]
        if (len(a_shape) != 2 or len(b_shape) != 2 or
                len(target_shape) != 2 or
                b_shape[1] != a_shape[0] or
                [b_shape[0], a_shape[1]] != target_shape):
            raise SystemExit(
                f"{target}: incompatible B{b_shape} @ A{a_shape} for "
                f"checkpoint shape {target_shape}")
        if targets[target]["dtype"] != "BF16":
            raise SystemExit(
                f"{target}: expected BF16, got {targets[target]['dtype']}")
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="directory of original bf16 transformer shards")
    ap.add_argument("--lora", required=True, type=Path,
                    help="LoRA .safetensors with <key>.lora_A/.lora_B pairs")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for the folded shards")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    if not math.isfinite(args.scale):
        raise SystemExit("--scale must be finite")
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite existing --out: {args.out}")

    lhl, lora_hdr = read_header(args.lora)
    lora_base = 8 + lhl
    pairs = {}
    for key in lora_hdr:
        if key.endswith(".lora_A.weight"):
            target = key[: -len(".lora_A.weight")] + ".weight"
            pairs[target] = (key, key[: -len(".lora_A.weight")] + ".lora_B.weight")
    if not pairs:
        raise SystemExit("no .lora_A/.lora_B pairs found in the adapter")
    print(f"{len(pairs)} adapter pairs")

    shards = preflight(
        args.checkpoint, args.lora, lora_hdr, lora_base, pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.out.name}.tmp-", dir=args.out.parent))
    try:
        for extra in ("config.json", "model.safetensors.index.json"):
            src = args.checkpoint / extra
            if src.exists():
                shutil.copy2(src, staging / extra)

        patched = 0
        for shard in shards:
            dst = staging / shard.name
            clone(shard, dst)
            hlen, hdr = read_header(dst)
            base = 8 + hlen
            todo = [k for k in hdr if k in pairs]
            if not todo:
                continue
            with open(dst, "r+b") as f:
                for key in todo:
                    info = hdr[key]
                    a_key, b_key = pairs[key]
                    A = load_tensor(args.lora, lora_hdr[a_key], lora_base)
                    B = load_tensor(args.lora, lora_hdr[b_key], lora_base)
                    W = load_tensor(dst, info, base)
                    folded = W + args.scale * (
                        B.astype(np.float32) @ A.astype(np.float32))
                    # Check the algebra before committing this tensor's bytes.
                    x = np.random.default_rng(0).standard_normal(
                        W.shape[1]).astype(np.float32)
                    direct = folded @ x
                    composed = W @ x + args.scale * (B @ (A @ x))
                    rel = (np.max(np.abs(direct - composed)) /
                           (np.max(np.abs(composed)) + 1e-9))
                    if rel > 1e-4:
                        raise SystemExit(
                            f"{key}: parity check failed (rel={rel:.2e})")
                    buf = f32_to_bf16_bytes(folded)
                    start, end = info["data_offsets"]
                    if len(buf) != end - start:
                        raise SystemExit(f"{key}: byte count mismatch")
                    f.seek(base + start)
                    f.write(buf)
                    patched += 1
            print(f"{shard.name}: cumulative {patched}/{len(pairs)}")

        if patched != len(pairs):
            raise SystemExit(
                f"only {patched} of {len(pairs)} adapter pairs matched "
                "checkpoint tensors - wrong checkpoint or adapter?")
        if args.out.exists() or args.out.is_symlink():
            raise SystemExit(f"refusing to overwrite existing --out: {args.out}")
        staging.rename(args.out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"done: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
