# DiffSoup
### Direct Differentiable Rasterization of Triangle Soup for Extreme Radiance Field Simplification

[Kenji Tojo](https://kenji-tojo.github.io/), [Bernd Bickel](https://berndbickel.com/about-me), [Nobuyuki Umetani](https://cgenglab.github.io/en/authors/admin/)

**CVPR 2026**

[Project Page](TBD) | [Paper](TBD) | [Video](TBD)

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

You can now run the example scripts, e.g.:

```bash
python3 examples/01_mip360.py
```

## Citation

```bibtex
@inproceedings{tojo2026diffsoup,
  title     = {DiffSoup: Direct Differentiable Rasterization of Triangle Soup for Extreme Radiance Field Simplification},
  author    = {Tojo, Kenji and Bickel, Bernd and Umetani, Nobuyuki},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
