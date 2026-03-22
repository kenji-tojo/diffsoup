#pragma once

#include <cstdint>

namespace diffsoup {

namespace {
inline uint32_t feats_at_level(uint32_t level) {
    return (level == 0u) ? 3u : (((1u << (level - 1u)) + 1u) * ((1u << level) + 1u));
}
inline uint32_t total_feats_from_levels(
    const uint32_t min_level,
    const uint32_t max_level
) {
    uint32_t S = 0;
    for (uint32_t level = min_level; level <= max_level; ++level) {
        S += feats_at_level(level);
    }
    return S;
}
} // namespace

namespace cuda {

void multires_triangle_alpha(
    int num_frags,
    const float* __restrict__ frag_attrs,   // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const float* alpha_src,                 // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ frag_alpha          // [num_frags]
);

void backward_multires_triangle_alpha(
    int num_frags,
    const float* __restrict__ frag_attrs,      // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    float* grad_alpha_src,                      // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_frag_alpha   // [num_frags]
);

void multires_triangle_color(
    int B, int H, int W,
    const float* __restrict__ rast,          // [B, H, W, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    const float* features,                   // [B, S, feature_dim]
    float* out                               // [B, H, W, feature_dim]
);

void backward_multires_triangle_color(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    float* grad_features,                // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_out   // [B, H, W, feature_dim]
);

void accumulate_to_level_forward(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,                // original “max” used for layout/stride
    const uint32_t target_level,             // ≤ max_level
    const uint32_t feature_dim,
    const float* __restrict__ features,      // [T, Σ_{l=min..concat} S_l, C]
    float* __restrict__ f_target             // [T, feats_at_level(target_level), C]
);

void accumulate_to_level_backward(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,                // original “max” used for layout/stride
    const uint32_t target_level,             // ≤ max_level
    const uint32_t feature_dim,
    float* __restrict__ grad_features,       // [T, Σ_{l=min..concat} S_l, C]  (zero before call)
    const float* __restrict__ grad_f_target  // [T, feats_at_level(target_level), C]
);

} // namespace cuda
} // namespace diffsoup