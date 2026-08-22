#!/usr/bin/env python3
"""Numerically compare learned-upscale-only and restart-refined H3 outputs."""

import argparse
import hashlib
import json
import struct
import subprocess

import numpy as np


def read_latent(path):
    with open(path, "rb") as file:
        magic, time_tokens, height, width = struct.unpack("<8sIII", file.read(20))
        if magic != b"H3LATF32":
            raise ValueError(f"bad latent magic: {path}")
        values = np.fromfile(file, dtype="<f4")
    expected = 24 * time_tokens * height * width
    if values.size != expected:
        raise ValueError(f"bad latent size: {path}")
    return values.reshape(1, 24, time_tokens, height, width)


def probe(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=index,codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration,size,bit_rate", "-of", "json", path,
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def video_rgb(path, frames, height, width):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    values = np.frombuffer(raw, dtype=np.uint8)
    expected = frames * height * width * 3
    if values.size != expected:
        raise ValueError(f"decoded {values.size} RGB bytes, expected {expected}: {path}")
    return values.reshape(frames, height, width, 3)


def audio_md5(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0",
         "-c", "copy", "-f", "md5", "-"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip().removeprefix("MD5=")


def array_stats(values):
    data = values.astype(np.float64)
    return {
        "elements": int(data.size),
        "finite": int(np.isfinite(data).sum()),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)),
    }


def video_stats(values):
    data = values.astype(np.float32)
    differences = np.sqrt(np.mean(np.diff(data, axis=0) ** 2, axis=(1, 2, 3)))
    hashes = {hashlib.md5(frame.tobytes()).hexdigest() for frame in values}
    result = array_stats(values)
    result.update({
        "unique_frames": len(hashes),
        "adjacent_frame_rmse_min": float(differences.min()),
        "adjacent_frame_rmse_mean": float(differences.mean()),
        "adjacent_frame_rmse_max": float(differences.max()),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upscale-latent", required=True)
    parser.add_argument("--refined-latent", required=True)
    parser.add_argument("--upscale-video", required=True)
    parser.add_argument("--refined-video", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    upscale_latent = read_latent(args.upscale_latent)
    refined_latent = read_latent(args.refined_latent)
    if upscale_latent.shape != refined_latent.shape:
        raise ValueError("latent shapes differ")
    latent_delta = refined_latent.astype(np.float64) - upscale_latent

    upscale_probe = probe(args.upscale_video)
    refined_probe = probe(args.refined_video)
    upscale_stream = upscale_probe["streams"][0]
    refined_stream = refined_probe["streams"][0]
    geometry = (
        int(upscale_stream["nb_frames"]),
        int(upscale_stream["height"]),
        int(upscale_stream["width"]),
    )
    if geometry != (
        int(refined_stream["nb_frames"]),
        int(refined_stream["height"]),
        int(refined_stream["width"]),
    ):
        raise ValueError("video shapes differ")
    upscale_rgb = video_rgb(args.upscale_video, *geometry)
    refined_rgb = video_rgb(args.refined_video, *geometry)
    video_delta = refined_rgb.astype(np.float32) - upscale_rgb.astype(np.float32)

    audio = {
        "source": audio_md5(args.source_video),
        "upscale_only": audio_md5(args.upscale_video),
        "refined": audio_md5(args.refined_video),
    }
    metrics = {
        "latent_shape": list(upscale_latent.shape),
        "upscale_latent": array_stats(upscale_latent),
        "refined_latent": array_stats(refined_latent),
        "refined_minus_upscale_latent_rmse": float(
            np.sqrt(np.mean(latent_delta ** 2))
        ),
        "upscale_video_probe": upscale_probe,
        "refined_video_probe": refined_probe,
        "upscale_video_rgb": video_stats(upscale_rgb),
        "refined_video_rgb": video_stats(refined_rgb),
        "refined_minus_upscale_rgb8_rmse": float(
            np.sqrt(np.mean(video_delta ** 2))
        ),
        "audio_packet_md5": audio,
        "audio_packet_md5_all_equal": len(set(audio.values())) == 1,
    }
    with open(args.metrics, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
