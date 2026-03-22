#pragma once

namespace diffsoup {
namespace cuda {

int compute_triangle_rects(
    int H, int W, int B,
    int V, const float* pos,     // [B * V][4]
    int T, const int* tri,       // [T][3]
    int* triangle_rects,         // [B * T][4]: h0, h_len, w0, w_len
    int* frag_prefix_sum         // [B * T]
);

void compute_fragments(
    int H, int W,
    int V, const float* pos,
    int T, const int* tri,
    int num_tris,
    int num_frags,
    const int* frag_prefix_sum,
    const int* triangle_rects,
    int* frag_pix,
    float* frag_attrs
);

void depth_test(
    int B, int H, int W,
    int num_frags,
    const int* frag_pix,       // [num_frags, 3]
    const float* frag_attrs,   // [num_frags, 4]
    const float* frag_alpha,   // [num_frags]
    const float* alpha_thresh, // [num_frags]
    float* rast_out            // [B, H, W, 4]
);

int filter_valid_fragments(
    int num_frags,
    const int* frag_pix,         // [num_frags, 3]
    const float* frag_attrs,     // [num_frags, 4]
    int* frag_pix_out,           // [num_frags, 3]
    float* frag_attrs_out        // [num_frags, 4]
);

void backward_edge_grad(
    int B, int H, int W, int C,
    const float* __restrict__ color,         // [B, H, W, C]
    const float* __restrict__ grad_color,    // [B, H, W, C]
    const float* __restrict__ rast,          // [B, H, W, 4]
    int V,
    const float* __restrict__ pos,           // [B, V, 4]
    float* __restrict__ grad_pos,            // [B, V, 4]
    const int* __restrict__ tri              // [T, 3]
);

void backward_radiance_field_loss(
    int B, int H, int W, int C,
    const float* __restrict__ color,         // [B, H, W, C]
    const float* __restrict__ target,        // [B, H, W, C]
    const float* __restrict__ rast,          // [B, H, W, 4]
    int num_frags,
    const int* __restrict__ frag_pix,        // [num_frags, 3]
    const float* __restrict__ frag_attrs,    // [num_frags, 4]
    const float* __restrict__ frag_alpha,    // [num_frags]
    float* __restrict__ grad_frag_alpha      // [num_frags]
);

void frag_alpha_mobilenerf(
    int num_frags,
    const float* frag_attrs, // [num_frags, 4]
    const float* uvs,        // [V, 2]
    const int* tri,          // [T, 3]
    int texH,
    int texW,
    const float* feat0,      // [texH, texW, 4]
    float* frag_alpha        // [num_frags,]
);

void lookup_feats_mobilenerf(
    int B, int H, int W,
    float* rast,         // [B, H, W, 4]
    const float* uvs,    // [V, 2]
    const int* tri,      // [T, 3]
    int texH,
    int texW,
    const float* feat0,  // [texH, texW, 4]
    const float* feat1,  // [texH, texW, 4]
    float* image         // [H, W, 8]
);

} // namespace cuda
} // namespace diffsoup