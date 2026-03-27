# DiffSoup
### Direct Differentiable Rasterization of Triangle Soup for Extreme Radiance Field Simplification

[Kenji Tojo](https://kenji-tojo.github.io/), [Bernd Bickel](https://berndbickel.com/about-me), [Nobuyuki Umetani](https://cgenglab.github.io/en/authors/admin/)

**CVPR 2026**

[Project Page (TBD)](https://github.com/kenji-tojo/diffsoup) | [Paper (TBD)](https://github.com/kenji-tojo/diffsoup) | [Video (TBD)](https://github.com/kenji-tojo/diffsoup)

## Tested Environment

- Ubuntu 22.04 LTS
- Python 3.10
- CUDA 12.4
- NVIDIA RTX 4090

## Installation

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/kenji-tojo/diffsoup.git
cd diffsoup
python3 -m venv venv
source venv/bin/activate
```

Install PyTorch with CUDA 12.4 from the [official website](https://pytorch.org/get-started/locally/):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Build and install the `diffsoup` module:

```bash
pip3 install -v .
```

Install the remaining dependencies:

```bash
pip3 install -r requirements.txt
```

## Getting Started

### Training

The example scripts in `examples/` cover the main training scenarios:

```bash
# MipNeRF-360 scenes (COLMAP-based, e.g. kitchen, garden, bicycle)
python3 examples/01_mip360.py --scene_root ./datasets/360_v2/kitchen

# NeRF-Synthetic scenes (Blender, e.g. lego, chair, hotdog)
python3 examples/02_synthetic.py --scene lego

# Random initialisation (no MobileNeRF mesh required)
python3 examples/03_random_init.py --scene lego
```

Each script saves a checkpoint (`final_params.pt`), rendered images, and test metrics to its output directory (e.g. `results/01_mip360/kitchen/`).

### Interactive Viewer

View a trained checkpoint with the native OpenGL viewer. The viewer has no CUDA dependency and can be installed on any machine (e.g. a laptop) with OpenGL support:

```bash
pip3 install -v viewer/

python3 examples/04_view_results.py --ckpt results/01_mip360/kitchen/final_params.pt
```

Controls: left-drag to orbit, right-drag to pan, scroll to zoom. The world up direction is auto-detected from the checkpoint (`--up X Y Z` to override).

### FPS Benchmark

Measure rendering throughput across all training and test views:

```bash
# MipNeRF-360
python3 examples/05_benchmark_fps.py \
    --ckpt results/01_mip360/kitchen/final_params.pt \
    --scene_root ./datasets/360_v2/kitchen

# NeRF-Synthetic
python3 examples/05_benchmark_fps.py \
    --ckpt results/02_synthetic/lego/final_params.pt \
    --scene_root ./datasets/nerf_synthetic/lego
```

Results (per-frame timings, mean FPS) are saved to `benchmark_output/` beside the checkpoint.

### Web Viewer

A browser-based viewer is included in `web/`. It runs on any device with WebGL 2 support, including phones.

**Step 1: Export assets**

```bash
# Export one or more checkpoints
python3 examples/06_export_web.py \
    --ckpt results/01_mip360/kitchen/final_params.pt

python3 examples/06_export_web.py \
    --ckpt results/02_synthetic/lego/final_params.pt
```

This writes web-ready files (mesh PLY, LUT PNGs, MLP JSON, metadata) to `web/data/<scene>/` and updates `web/data/models.json`.

**Step 2: Start a local server**

```bash
cd web
python3 -m http.server 8080 --bind 0.0.0.0
```

**Step 3: Open in a browser**

- Desktop: [http://localhost:8080](http://localhost:8080)
- Phone (same network): `http://<your-lan-ip>:8080`

Use the dropdown in the top-left corner to switch between exported scenes. To find your LAN IP, run `hostname -I` on Linux or `ifconfig | grep inet` on macOS.

### Mobile Benchmark

`web/benchmark.html` loads all 14 models onto a grid and runs a deterministic FPS benchmark on a phone using a Fibonacci-lattice camera trajectory. It requires the pre-exported scene data in `web/data/ours_mobile_results/`.

**Step 1: Start a local server on your machine**

```bash
cd web
python3 -m http.server 8080 --bind 0.0.0.0
```

**Step 2: Open on your phone**

Navigate to `http://<your-lan-ip>:8080/benchmark.html` on a phone connected to the same network. To find your LAN IP, run `hostname -I` on Linux or `ifconfig | grep inet` on macOS.

For reproducibility, hold the phone in landscape orientation (width > height) during the benchmark.

![Mobile benchmark screenshot](picks/mobile_pose_002.png)

Press **Start** to begin. Results (per-pose FPS, camera matrices, screenshots) are exported as a JSON file when the run finishes. To extract human-readable outputs from the JSON:

```bash
python3 examples/07_extract_bench.py \
    --input benchmark_diffsoup_30poses.json
```

This produces `summary.json` (FPS statistics), `summary.csv` (per-pose data), and decoded screenshots in `images/`.

## Citation

```bibtex
@inproceedings{tojo2026diffsoup,
  title     = {DiffSoup: Direct Differentiable Rasterization of Triangle Soup for Extreme Radiance Field Simplification},
  author    = {Tojo, Kenji and Bickel, Bernd and Umetani, Nobuyuki},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Contact

If you encounter any issues (e.g. missing files), please feel free to contact the first author [Kenji Tojo](https://kenji-tojo.github.io/). Questions are also welcome!
