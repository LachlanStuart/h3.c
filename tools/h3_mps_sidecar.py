#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "einops>=0.8,<1",
#   "numpy>=2,<3",
#   "psutil>=7,<8",
#   "safetensors>=0.6,<1",
#   "torch>=2.8,<3",
#   "typing-extensions>=4.15,<5",
# ]
# ///
"""Run the published MiniMax-H3 3D latent upscaler on Apple MPS.

The script deliberately imports the publisher's implementation, loads the
actual checkpoint through safetensors, and disables PyTorch MPS CPU fallback.
It exchanges normalized H3 video latents through h3_latent_io.h's H3LATF32
format.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import struct
import sys
import threading
import time
import types

# This must be set before importing torch. Unsupported MPS operations fail.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open

MAGIC = b"H3LATF32"
CHANNELS = 24


def read_latent_stream(file):
    header = file.read(20)
    if len(header) != 20:
        raise ValueError("truncated H3 latent header")
    magic, time_tokens, height, width = struct.unpack("<8sIII", header)
    if magic != MAGIC or time_tokens < 2 or height < 1 or width < 1:
        raise ValueError("malformed H3 latent header")
    expected = CHANNELS * time_tokens * height * width
    bytes_value = file.read(expected * 4)
    if len(bytes_value) != expected * 4 or file.read(1):
        raise ValueError("latent payload size does not match its header")
    values = np.frombuffer(bytes_value, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("input H3 latent contains non-finite values")
    return torch.from_numpy(values.copy()).reshape(
        1, CHANNELS, time_tokens, height, width
    )


def read_latent(path):
    with open(path, "rb") as file:
        return read_latent_stream(file)


def write_latent_stream(file, tensor):
    output = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    batch, channels, time_tokens, height, width = output.shape
    if batch != 1 or channels != CHANNELS:
        raise ValueError(f"expected [1,24,T,H,W], got {list(output.shape)}")
    values = output.numpy()
    if not np.isfinite(values).all():
        raise ValueError("refusing to write non-finite H3 latent")
    file.write(struct.pack("<8sIII", MAGIC, time_tokens, height, width))
    file.write(values.astype("<f4", copy=False).tobytes())
    file.flush()


def write_latent(path, tensor):
    with open(path, "wb") as file:
        write_latent_stream(file, tensor)


class MemorySampler:
    def __init__(self):
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_rss = 0
        self.peak_mps_current = 0
        self.peak_mps_driver = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.peak_mps_current = max(
            self.peak_mps_current, torch.mps.current_allocated_memory()
        )
        self.peak_mps_driver = max(
            self.peak_mps_driver, torch.mps.driver_allocated_memory()
        )

    def _run(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(0.005)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self._sample()


def import_published_upscaler(source_path, models_dir):
    # The publisher's node imports ComfyUI's folder_paths at module load. This
    # tiny stub supplies only the model-folder interface; no ComfyUI code runs.
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.models_dir = models_dir
    folder_paths.add_model_folder_path = lambda name, path: (
        folder_paths.folder_names_and_paths.setdefault(name, ([path], set()))
    )
    folder_paths.get_folder_paths = (
        lambda name: folder_paths.folder_names_and_paths[name][0]
    )
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location(
        "published_h3_upscaler_3d", source_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import published upscaler source {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_stats(tensor):
    values = tensor.float()
    return {
        "finite": int(torch.isfinite(values).sum().item()),
        "elements": values.numel(),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gib(value):
    return value / (1024**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="publisher's minimax_h3_latent_upscaler_3d.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--metrics")
    parser.add_argument("--stream", action="store_true",
                        help="read H3LATF32 from stdin and write it to stdout")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    if args.stream:
        if args.input or args.output or args.metrics:
            parser.error("--stream cannot be combined with file input/output/metrics")
    elif not args.input or not args.output or not args.metrics:
        parser.error("--input, --output, and --metrics are required without --stream")

    if not 1.0 < args.scale <= 4.0:
        raise ValueError("proof sidecar requires an upscale in (1, 4]")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not built and available")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS CPU fallback was not disabled")

    checkpoint_hash = sha256_file(args.checkpoint)
    if args.expected_sha256 and checkpoint_hash != args.expected_sha256.lower():
        raise ValueError("checkpoint SHA-256 does not match expected value")

    with safe_open(args.checkpoint, framework="pt", device="cpu") as file:
        keys = list(file.keys())
        metadata = file.metadata()
        dtype_counts = {}
        for key in keys:
            dtype_name = str(file.get_slice(key).get_dtype())
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    if not keys or set(dtype_counts) != {"BF16"}:
        raise ValueError(f"checkpoint is not entirely BF16: {dtype_counts}")

    module = import_published_upscaler(
        args.source, os.path.dirname(args.checkpoint)
    )
    input_cpu = read_latent_stream(sys.stdin.buffer) if args.stream else read_latent(args.input)
    input_shape = list(input_cpu.shape)
    input_metrics = tensor_stats(input_cpu)
    if input_metrics["finite"] != input_metrics["elements"] or input_metrics["std"] == 0:
        raise ValueError("input latent is non-finite or constant")

    device = torch.device("mps")
    torch.mps.empty_cache()
    started = time.perf_counter()
    with MemorySampler() as memory:
        safetensors_started = time.perf_counter()
        raw_state = module._load_raw_sd(args.checkpoint)
        safetensors_seconds = time.perf_counter() - safetensors_started
        state = module._extract_upscaler_sd(raw_state)
        config = module._detect_arch(state)

        construct_started = time.perf_counter()
        model = module.LatentResizer3D(
            in_channels=config["in_channels"],
            in_blocks=config["in_blocks"],
            out_blocks=config["out_blocks"],
            channels=config["channels"],
            dropout=config["dropout"],
            attn=config["attn"],
            temporal_every=config["temporal_every"],
            temporal_kernel=config["temporal_kernel"],
        )
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        construct_seconds = time.perf_counter() - construct_started

        transfer_started = time.perf_counter()
        model = model.to(device=device, dtype=torch.bfloat16)
        input_mps = input_cpu.to(device=device, dtype=torch.bfloat16)
        mean, std = module._make_norm_tensors(device, torch.bfloat16)
        torch.mps.synchronize()
        transfer_seconds = time.perf_counter() - transfer_started
        del raw_state, state
        gc.collect()

        target = (
            input_shape[2],
            round(input_shape[3] * args.scale),
            round(input_shape[4] * args.scale),
        )
        inference_started = time.perf_counter()
        with torch.inference_mode():
            normalized = (input_mps - mean) / std
            output_mps = model(
                normalized, scale=args.scale, target_size=target
            )
            output_mps = output_mps * std + mean
        torch.mps.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        expected_shape = [1, CHANNELS, *target]
        if list(output_mps.shape) != expected_shape:
            raise RuntimeError(
                f"upscaler returned {list(output_mps.shape)}, expected {expected_shape}"
            )
        if output_mps.device.type != "mps" or output_mps.dtype != torch.bfloat16:
            raise RuntimeError("upscaler output left MPS or BF16")

        readback_started = time.perf_counter()
        output_cpu = output_mps.to(device="cpu", dtype=torch.float32)
        torch.mps.synchronize()
        readback_seconds = time.perf_counter() - readback_started
        output_metrics = tensor_stats(output_cpu)
        if (output_metrics["finite"] != output_metrics["elements"] or
                output_metrics["std"] == 0):
            raise RuntimeError("upscaler returned non-finite or constant output")

        # This is a rejection test, not an alternate execution path. The
        # published network itself deliberately has one trained trilinear
        # feature-resize stage; the final latent must differ materially from a
        # plain trilinear resize of the input latent.
        simple = F.interpolate(
            input_cpu, size=target, mode="trilinear", align_corners=False
        )
        learned_delta_rmse = float(
            torch.mean((output_cpu - simple) ** 2).sqrt().item()
        )
        if learned_delta_rmse <= 1e-4:
            raise RuntimeError("learned output collapsed to trilinear substitution")
        if args.stream:
            write_latent_stream(sys.stdout.buffer, output_cpu)
        else:
            write_latent(args.output, output_cpu)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    input_transfer_bytes = input_mps.numel() * input_mps.element_size()
    output_transfer_bytes = output_mps.numel() * output_mps.element_size()
    metrics = {
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "device": str(device),
        "model_parameter_devices": sorted(
            {str(parameter.device) for parameter in model.parameters()}
        ),
        "model_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        ),
        "output_device_before_readback": str(output_mps.device),
        "output_dtype_before_readback": str(output_mps.dtype),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": os.path.getsize(args.checkpoint),
        "safetensors_tensor_count": len(keys),
        "safetensors_dtypes": dtype_counts,
        "safetensors_metadata": metadata,
        "architecture": config,
        "parameters": parameters,
        "bf16_parameter_bytes_host_to_mps": parameter_bytes,
        "bf16_input_bytes_host_to_mps": input_transfer_bytes,
        "bf16_output_bytes_mps_to_host": output_transfer_bytes,
        "host_to_mps_bytes_total": parameter_bytes + input_transfer_bytes,
        "mps_to_host_bytes_total": output_transfer_bytes,
        "intermediate_host_transfers": 0,
        "input_shape": input_shape,
        "input_stats": input_metrics,
        "output_shape": list(output_cpu.shape),
        "output_stats": output_metrics,
        "learned_vs_plain_trilinear_rmse": learned_delta_rmse,
        "safetensors_load_seconds": safetensors_seconds,
        "construct_and_state_load_seconds": construct_seconds,
        "mps_transfer_seconds": transfer_seconds,
        "mps_inference_seconds": inference_seconds,
        "mps_readback_seconds": readback_seconds,
        "total_seconds": time.perf_counter() - started,
        "sampled_peak_process_rss_gib": gib(memory.peak_rss),
        "sampled_peak_mps_current_gib": gib(memory.peak_mps_current),
        "sampled_peak_mps_driver_gib": gib(memory.peak_mps_driver),
        "output": "stdout" if args.stream else os.path.abspath(args.output),
    }
    if args.stream:
        print(json.dumps(metrics, sort_keys=True), file=sys.stderr)
    else:
        with open(args.metrics, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, sort_keys=True)
            file.write("\n")
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
