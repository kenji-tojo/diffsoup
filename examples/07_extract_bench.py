# examples/07_extract_bench.py
# Extract results from a web benchmark JSON into human-readable outputs.
#
# Produces (inside <output_dir>/):
#   summary.json      — metadata + FPS statistics (avg, median, min, max)
#   summary.csv       — per-pose FPS, timings, camera matrices
#   scene.json        — reproducible scene configuration (if present)
#   images/           — per-pose screenshots decoded from base64
#
# Usage:
#   python examples/07_extract_bench.py \
#       --input benchmark_diffsoup_30poses.json
#
#   python examples/07_extract_bench.py \
#       --input benchmark_diffsoup_30poses.json \
#       --output_dir results/bench_extract
#
# Dependencies:
#   (standard library only)

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, median


# ── Helpers ──────────────────────────────────────────────────────────


def parse_data_url(data_url: str) -> tuple[str | None, str | None]:
    """Parse a data URI, returning (mime_type, base64_payload)."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None, None
    try:
        header, payload = data_url.split(",", 1)
    except ValueError:
        return None, None
    mime = header[5:].split(";", 1)[0] if header.startswith("data:") else None
    if ";base64" not in header:
        return mime, None
    return mime, payload


def decode_screenshot(data_url: str) -> bytes | None:
    """Decode a PNG data-URL to raw bytes, or return None."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        return None
    mime, payload = parse_data_url(data_url)
    if mime != "image/png" or payload is None:
        return None
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        return base64.b64decode(payload)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract results from a web benchmark JSON.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the benchmark JSON exported by benchmark.html.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory.  Default: same directory as the input file.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[error] File not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    images_dir = out_dir / "images"
    os.makedirs(images_dir, exist_ok=True)

    # ── Load benchmark JSON ──────────────────────────────────────────

    print(f"[load] {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    results = bench.get("results")
    if not isinstance(results, list) or not results:
        print("[error] JSON doesn't contain a non-empty 'results' list.",
              file=sys.stderr)
        sys.exit(2)

    print(f"[info] {len(results)} poses")

    # ── Scene configuration ──────────────────────────────────────────

    scene_info = bench.get("scene") or bench.get("metaScene")
    if isinstance(scene_info, dict):
        scene_path = out_dir / "scene.json"
        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_info, f, indent=2)
        print(f"[save] {scene_path}")

    # ── Per-pose extraction ──────────────────────────────────────────

    rows = []
    fps_values = []
    num_images = 0

    for r in results:
        pose = r.get("poseIndex")
        fps = r.get("fps")
        sec = r.get("seconds")
        cam_pos = r.get("cameraPosition")
        cam_tgt = r.get("cameraTarget")
        view = r.get("viewMatrix")
        proj = r.get("projMatrix")

        if isinstance(fps, (int, float)):
            fps_values.append(float(fps))

        img_rel = ""
        png_bytes = decode_screenshot(r.get("screenshotPngDataUrl", ""))
        if png_bytes is not None:
            idx = int(pose) if pose is not None else len(rows)
            img_name = f"pose_{idx:03d}.png"
            img_rel = os.path.join("images", img_name)
            with open(out_dir / img_rel, "wb") as pf:
                pf.write(png_bytes)
            num_images += 1

        rows.append({
            "poseIndex": pose,
            "fps": fps,
            "seconds": sec,
            "image": img_rel,
            "cameraPosition": json.dumps(cam_pos) if cam_pos is not None else "",
            "cameraTarget": json.dumps(cam_tgt) if cam_tgt is not None else "",
            "viewMatrix": json.dumps(view) if view is not None else "",
            "projMatrix": json.dumps(proj) if proj is not None else "",
        })

    print(f"[save] {images_dir}  ({num_images} png)")

    # ── FPS statistics ───────────────────────────────────────────────

    stats = {}
    if fps_values:
        fps_sorted = sorted(fps_values)
        stats = {
            "count": len(fps_values),
            "avg_fps": mean(fps_values),
            "median_fps": median(fps_values),
            "min_fps": fps_sorted[0],
            "max_fps": fps_sorted[-1],
        }
        print(f"[stat] count={stats['count']}  "
              f"avg={stats['avg_fps']:.2f}  "
              f"median={stats['median_fps']:.2f}  "
              f"min={stats['min_fps']:.2f}  "
              f"max={stats['max_fps']:.2f}")

    # ── Write summary JSON ───────────────────────────────────────────

    summary = {
        "meta": {
            "timestamp": bench.get("timestamp"),
            "methodTag": bench.get("methodTag"),
            "userAgent": bench.get("userAgent"),
            "devicePixelRatio": bench.get("devicePixelRatio"),
            "viewport": bench.get("viewport"),
            "renderer": bench.get("renderer"),
            "sceneBounds": bench.get("sceneBounds"),
            "measureFrames": bench.get("measureFrames"),
            "numPoses": bench.get("numPoses"),
            "gpuSurrogate": bench.get("gpuSurrogate"),
            "scene": scene_info if isinstance(scene_info, dict) else None,
        },
        "stats": stats,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")

    # ── Write summary CSV ────────────────────────────────────────────

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "poseIndex", "fps", "seconds", "image",
        "cameraPosition", "cameraTarget", "viewMatrix", "projMatrix",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[save] {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────

    total_kb = sum(
        os.path.getsize(out_dir / name)
        for name in os.listdir(out_dir)
        if os.path.isfile(out_dir / name)
    ) / 1024
    print(f"\n[done] Extracted {len(rows)} poses → {out_dir}/  ({total_kb:.0f} KB)")


if __name__ == "__main__":
    main()
