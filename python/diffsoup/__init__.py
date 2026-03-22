# python/diffsoup/__init__.py
"""DiffSoup: differentiable triangle-soup rendering and optimisation."""

from __future__ import annotations

import torch
from ._core import __version__
from . import _core
from . import optimize

from .rasterize import (
    edge_grad,
    opacity_aux_loss,
    encode_view_dir_sh2,
    encode_view_dir_freq,
    count_triangle_ids,
)

from .multires import (
    ColorMLP,
    feats_at_level,
    build_multires_triangle_color,
    rasterize_multires_triangle_alpha,
    multires_triangle_color,
    accumulate_to_level,
)

from .remesh import (
    split_triangle_soup,
    split_triangle_soup_until,
    split_triangle_soup_clip,
    split_triangle_soup_clip_until,
    expand_by_index,
)

from .point3d import (
    nn_spacing,
    triangle_soup_from_points,
    remove_unreferenced_vertices_from_soup,
)
