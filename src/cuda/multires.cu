#include "multires.cuh"

#include <math.h>

#include "cuda_common.cuh"

namespace diffsoup {
namespace cuda {

__device__ void multires_triangle_interp_d(
    float b0, float b1,            // barycentric coordinates
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    const float* features,         // [S, feature_dim], where S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* out                     // [feature_dim]
) {
    uint32_t offset = 0;

    // For each subdivision level from min_level to max_level
    for (uint32_t level = min_level; level <= max_level; ++level) {
        const uint32_t res = 1 << level;
        const float res_f = static_cast<float>(res);

        float b0_level = b0 * res_f;
        float b1_level = b1 * res_f;

        const uint32_t x = MIN(static_cast<uint32_t>(floorf(b0_level)), res - 1);
        const uint32_t y = MIN(static_cast<uint32_t>(floorf(b1_level)), res - 1 - x);  // x + y <= res - 1

        b0_level = b0_level - static_cast<float>(x);
        b1_level = b1_level - static_cast<float>(y);

        const bool flip = b0_level + b1_level > 1.f;
        const uint32_t flip_u = static_cast<uint32_t>(flip);
        const float flip_f = static_cast<float>(flip);

        const uint32_t x0 = x + 1;
        const uint32_t y0 = y;
        const uint32_t x1 = x;
        const uint32_t y1 = y + 1;
        const uint32_t x2 = x + flip_u;
        const uint32_t y2 = MIN(y + flip_u, res - x2);  // x2 + y2 <= res

        uint32_t index[3];
        index[0] = (x0 + y0) * (x0 + y0 + 1) / 2 + y0;
        index[1] = (x1 + y1) * (x1 + y1 + 1) / 2 + y1;
        index[2] = (x2 + y2) * (x2 + y2 + 1) / 2 + y2;

        float weight[3];
        weight[0] = (1.f - flip_f) * b0_level + flip_f * (1.f - b1_level);
        weight[1] = (1.f - flip_f) * b1_level + flip_f * (1.f - b0_level);
        weight[2] = 1.f - weight[0] - weight[1];

        const float* level_features = features + offset * feature_dim;

        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            #pragma unroll 4
            for (uint32_t j = 0; j < feature_dim; ++j) {
                out[j] += weight[i] * level_features[index[i] * feature_dim + j];
            }
        }

        if (level == 0) offset += 3;
        else offset += ((1 << (level - 1)) + 1) * ((1 << level) + 1);
    }
}

__device__ void backward_multires_triangle_interp_d(
    float b0, float b1,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    float* grad_features,          // [S, feature_dim], where S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* grad_out          // [feature_dim]
) {
    uint32_t offset = 0;

    // For each subdivision level from min_level to max_level
    for (uint32_t level = min_level; level <= max_level; ++level) {
        const uint32_t res = 1 << level;
        const float res_f = static_cast<float>(res);

        float b0_level = b0 * res_f;
        float b1_level = b1 * res_f;

        const uint32_t x = MIN(static_cast<uint32_t>(floorf(b0_level)), res - 1);
        const uint32_t y = MIN(static_cast<uint32_t>(floorf(b1_level)), res - 1 - x);  // x + y <= res - 1

        b0_level = b0_level - static_cast<float>(x);
        b1_level = b1_level - static_cast<float>(y);

        const bool flip = b0_level + b1_level > 1.f;
        const uint32_t flip_u = static_cast<uint32_t>(flip);
        const float flip_f = static_cast<float>(flip);

        const uint32_t x0 = x + 1;
        const uint32_t y0 = y;
        const uint32_t x1 = x;
        const uint32_t y1 = y + 1;
        const uint32_t x2 = x + flip_u;
        const uint32_t y2 = MIN(y + flip_u, res - x2);  // x2 + y2 <= res

        uint32_t index[3];
        index[0] = (x0 + y0) * (x0 + y0 + 1) / 2 + y0;
        index[1] = (x1 + y1) * (x1 + y1 + 1) / 2 + y1;
        index[2] = (x2 + y2) * (x2 + y2 + 1) / 2 + y2;

        float weight[3];
        weight[0] = (1.f - flip_f) * b0_level + flip_f * (1.f - b1_level);
        weight[1] = (1.f - flip_f) * b1_level + flip_f * (1.f - b0_level);
        weight[2] = 1.f - weight[0] - weight[1];

        float* level_grad_features = grad_features + offset * feature_dim;

        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            #pragma unroll 4
            for (uint32_t j = 0; j < feature_dim; ++j) {
                atomicAdd(&level_grad_features[index[i] * feature_dim + j], grad_out[j] * weight[i]);
            }
        }

        if (level == 0) offset += 3;
        else offset += ((1 << (level - 1)) + 1) * ((1 << level) + 1);
    }
}

__global__ void multires_triangle_alpha_kernel(
    int num_frags,
    const float* __restrict__ frag_attrs,   // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    const float* alpha_src,                 // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ frag_alpha          // [num_frags]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int triangle_id = static_cast<int>(frag_attrs[idx * 4 + 3]) - 1;
    if (triangle_id < 0) return;

    const float b0 = frag_attrs[idx * 4 + 0];
    const float b1 = frag_attrs[idx * 4 + 1];

    float alpha = 0.f;

    multires_triangle_interp_d(
        b0, b1, min_level, max_level, /*feature_dim=*/1,
        &alpha_src[triangle_id * S], &alpha
    );

    frag_alpha[idx] = alpha;
}

void multires_triangle_alpha(
    int num_frags,
    const float* __restrict__ frag_attrs,   // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const float* alpha_src,                 // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ frag_alpha          // [num_frags]
) {
    if (num_frags == 0) return;

    const uint32_t S = total_feats_from_levels(min_level, max_level);

    multires_triangle_alpha_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS>>>(
        num_frags, frag_attrs, min_level, max_level, S,
        alpha_src, frag_alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void backward_multires_triangle_alpha_kernel(
    int num_frags,
    const float* __restrict__ frag_attrs,        // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    float* grad_alpha_src,                       // [T, S]
    const float* __restrict__ grad_frag_alpha    // [num_frags]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int triangle_id = static_cast<int>(frag_attrs[idx * 4 + 3]) - 1;
    if (triangle_id < 0) return;

    const float b0 = frag_attrs[idx * 4 + 0];
    const float b1 = frag_attrs[idx * 4 + 1];
    const float grad_alpha = grad_frag_alpha[idx];

    backward_multires_triangle_interp_d(
        b0, b1, min_level, max_level, /*feature_dim=*/1,
        &grad_alpha_src[triangle_id * S],
        &grad_alpha
    );
}

void backward_multires_triangle_alpha(
    int num_frags,
    const float* __restrict__ frag_attrs,      // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    float* grad_alpha_src,                      // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_frag_alpha   // [num_frags]
) {
    if (num_frags == 0) return;

    const uint32_t S = total_feats_from_levels(min_level, max_level);

    backward_multires_triangle_alpha_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS>>>(
        num_frags, frag_attrs, min_level, max_level, S,
        grad_alpha_src, grad_frag_alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void multires_triangle_color_kernel(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    const uint32_t feature_dim,
    const float* features,               // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ out              // [B, H, W, feature_dim]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const int tri_id = static_cast<int>(rast[idx * 4 + 3]) - 1;
    if (tri_id < 0) return;

    const float b0 = rast[idx * 4 + 0];
    const float b1 = rast[idx * 4 + 1];

    multires_triangle_interp_d(
        b0, b1, min_level, max_level, feature_dim,
        &features[tri_id * S * feature_dim], &out[idx * feature_dim]
    );
}

void multires_triangle_color(
    int B, int H, int W,
    const float* __restrict__ rast,          // [B, H, W, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    const float* features,                   // [B, S, feature_dim]
    float* out                               // [B, H, W, feature_dim]
) {
    const uint32_t S = total_feats_from_levels(min_level, max_level);

    multires_triangle_color_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS>>>(
        B, H, W, rast, min_level, max_level,
        S, feature_dim, features, out
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void backward_multires_triangle_color_kernel(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    const uint32_t feature_dim,
    float* grad_features,                // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_out   // [B, H, W, feature_dim]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const int tri_id = static_cast<int>(rast[idx * 4 + 3]) - 1;
    if (tri_id < 0) return;

    const float b0 = rast[idx * 4 + 0];
    const float b1 = rast[idx * 4 + 1];

    backward_multires_triangle_interp_d(
        b0, b1, min_level, max_level, feature_dim,
        &grad_features[tri_id * S * feature_dim],
        &grad_out[idx * feature_dim]
    );
}

void backward_multires_triangle_color(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    float* grad_features,                // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_out   // [B, H, W, feature_dim]
) {
    const uint32_t S = total_feats_from_levels(min_level, max_level);

    backward_multires_triangle_color_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS>>>(
        B, H, W, rast, min_level, max_level,
        S, feature_dim, grad_features, grad_out
    );

    CUDA_CHECK(cudaGetLastError());
}

// inverse of: idx = T_n + y, where n = x+y, T_n = n(n+1)/2, 0<=y<=n, x = n-y
inline __device__ void index_to_xy_on_level(uint32_t L, uint32_t idx, uint32_t& x, uint32_t& y) {
    // solve n from triangular number: T_n <= idx < T_{n+1}
    // n = floor((sqrt(8*idx+1)-1)/2)
    const float fi   = static_cast<float>(idx);
    const uint32_t n = static_cast<uint32_t>(floorf((sqrtf(8.f * fi + 1.f) - 1.f) * 0.5f));
    const uint32_t Tn = (n * (n + 1u)) >> 1; // n(n+1)/2
    y = idx - Tn;
    x = n - y;

    // (x,y) are coordinates on the level-L vertex lattice with constraint x+y<=2^L
    // nothing else to do here.
}

// --- forward: accumulate all levels (min..target) -> target lattice --------

__global__ void accumulate_to_level_forward_kernel(
    int T,                                   // triangle count
    const uint32_t min_level,                // included
    const uint32_t max_level,                // included
    const uint32_t target_level,             // included; target_level >= min_level
    const uint32_t S_stride_total,           // Σ_{l=min..concat_level} S_l (per-triangle stride)
    const uint32_t S_T,                      // S at level L=target_level
    const uint32_t feature_dim,
    const float* __restrict__ features,      // [T, S_stride_total, C]
    float* __restrict__ f_target             // [T, S_T, C]
) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t N   = static_cast<uint64_t>(T) * static_cast<uint64_t>(S_T);
    if (tid >= N) return;

    const uint32_t t = static_cast<uint32_t>(tid / S_T);
    const uint32_t k = static_cast<uint32_t>(tid % S_T);

    // decode (x,y) at target level
    uint32_t x, y;
    index_to_xy_on_level(target_level, k, x, y);

    const float Rf = static_cast<float>(1u << target_level);
    const float b0 = static_cast<float>(x) / Rf;
    const float b1 = static_cast<float>(y) / Rf;

    // output pointer (zero then accumulate)
    float* out = &f_target[(static_cast<uint64_t>(t) * S_T + k) * feature_dim];
    #pragma unroll 4
    for (uint32_t c = 0; c < feature_dim; ++c) out[c] = 0.f;

    // evaluate existing multires function exactly at this target-vertex
    const float* tri_feats = &features[(static_cast<uint64_t>(t) * S_stride_total) * feature_dim];
    multires_triangle_interp_d(
        b0, b1,
        /*min_level=*/min_level, /*max_level=*/max_level,
        feature_dim,
        tri_feats,   // [Σ_{l=min..concat} S_l, C] but only min..target are read
        out          // [C]
    );
}

void accumulate_to_level_forward(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,                // original “max” used for layout/stride
    const uint32_t target_level,             // ≤ max_level
    const uint32_t feature_dim,
    const float* __restrict__ features,      // [T, Σ_{l=min..concat} S_l, C]
    float* __restrict__ f_target             // [T, feats_at_level(target_level), C]
) {
    if (T == 0) return;
    const uint32_t S_stride_total = total_feats_from_levels(min_level, max_level);
    const uint32_t S_T            = feats_at_level(target_level);

    const uint64_t N = static_cast<uint64_t>(T) * static_cast<uint64_t>(S_T);
    accumulate_to_level_forward_kernel<<<CUDA_BLOCKS(N), CUDA_THREADS>>>(
        T, min_level, max_level, target_level, S_stride_total, S_T, feature_dim, features, f_target
    );
    CUDA_CHECK(cudaGetLastError());
}

// --- backward: grad wrt multires levels from grad at target lattice --------

__global__ void accumulate_to_level_backward_kernel(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t target_level,
    const uint32_t S_stride_total,           // Σ_{l=min..concat} S_l
    const uint32_t S_T,                      // feats_at_level(target_level)
    const uint32_t feature_dim,
    float* __restrict__ grad_features,       // [T, Σ_{l=min..concat} S_l, C]  (zeroed by caller)
    const float* __restrict__ grad_f_target  // [T, S_T, C]
) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t N   = static_cast<uint64_t>(T) * static_cast<uint64_t>(S_T);
    if (tid >= N) return;

    const uint32_t t = static_cast<uint32_t>(tid / S_T);
    const uint32_t k = static_cast<uint32_t>(tid % S_T);

    // decode (x,y) at target level
    uint32_t x, y;
    index_to_xy_on_level(target_level, k, x, y);

    const float Rf = static_cast<float>(1u << target_level);
    const float b0 = static_cast<float>(x) / Rf;
    const float b1 = static_cast<float>(y) / Rf;

    float* tri_grad_feats = &grad_features[(static_cast<uint64_t>(t) * S_stride_total) * feature_dim];
    const float* grad_out = &grad_f_target[(static_cast<uint64_t>(t) * S_T + k) * feature_dim];

    // scatter-add into grad_features using your existing backward device fn
    backward_multires_triangle_interp_d(
        b0, b1,
        /*min_level=*/min_level, /*max_level=*/max_level,
        feature_dim,
        tri_grad_feats,   // atomicAdd inside helper
        grad_out
    );
}

void accumulate_to_level_backward(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,                // original “max” used for layout/stride
    const uint32_t target_level,             // ≤ max_level
    const uint32_t feature_dim,
    float* __restrict__ grad_features,       // [T, Σ_{l=min..concat} S_l, C]  (zero before call)
    const float* __restrict__ grad_f_target  // [T, feats_at_level(target_level), C]
) {
    if (T == 0) return;
    const uint32_t S_stride_total = total_feats_from_levels(min_level, max_level);
    const uint32_t S_T            = feats_at_level(target_level);

    const uint64_t N = static_cast<uint64_t>(T) * static_cast<uint64_t>(S_T);
    accumulate_to_level_backward_kernel<<<CUDA_BLOCKS(N), CUDA_THREADS>>>(
        T, min_level, max_level, target_level, S_stride_total, S_T, feature_dim, grad_features, grad_f_target
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace diffsoup