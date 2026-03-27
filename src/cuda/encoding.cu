#include "encoding.cuh"

#include "cuda_common.cuh"

namespace diffsoup {
namespace cuda {

namespace {
__device__ inline void ndc_to_world(
    const float ndc[4],      // NDC position (x, y, z, w)
    const float inv_mvp[16], // Inverse MVP matrix (row-major)
    float world_pos[3])      // Output world position
{
    // Multiply mvp_inv * ndc
    float clip[4];
    clip[0] = inv_mvp[0] * ndc[0] + inv_mvp[1] * ndc[1] + inv_mvp[2] * ndc[2] + inv_mvp[3] * ndc[3];
    clip[1] = inv_mvp[4] * ndc[0] + inv_mvp[5] * ndc[1] + inv_mvp[6] * ndc[2] + inv_mvp[7] * ndc[3];
    clip[2] = inv_mvp[8] * ndc[0] + inv_mvp[9] * ndc[1] + inv_mvp[10] * ndc[2] + inv_mvp[11] * ndc[3];
    clip[3] = inv_mvp[12] * ndc[0] + inv_mvp[13] * ndc[1] + inv_mvp[14] * ndc[2] + inv_mvp[15] * ndc[3];

    // Perspective divide
    float inv_w = 1.0f / clip[3];
    world_pos[0] = clip[0] * inv_w;
    world_pos[1] = clip[1] * inv_w;
    world_pos[2] = clip[2] * inv_w;
}

__device__ inline void compute_view_direction(
    float ndc_x,             // NDC x coordinate [-1, 1]
    float ndc_y,             // NDC y coordinate [-1, 1]
    const float inv_mvp[16], // Inverse MVP matrix (row-major)
    float view_dir[3])       // Output normalized view direction (from surface to camera)
{
    // Create two points along the viewing ray in NDC space
    float ndc_near[4] = {ndc_x, ndc_y, -1.0f, 1.0f};
    float ndc_far[4] = {ndc_x, ndc_y, 1.0f, 1.0f};

    // Transform both points to world space
    float world_near[3], world_far[3];
    ndc_to_world(ndc_near, inv_mvp, world_near);
    ndc_to_world(ndc_far, inv_mvp, world_far);

    // View direction is from far to near (pointing back to camera)
    float dx = world_near[0] - world_far[0];
    float dy = world_near[1] - world_far[1];
    float dz = world_near[2] - world_far[2];

    // Normalize
    float inv_length = rsqrtf(dx * dx + dy * dy + dz * dz);
    view_dir[0] = dx * inv_length;
    view_dir[1] = dy * inv_length;
    view_dir[2] = dz * inv_length;
}

// Constants for SH evaluation
__constant__ float SH_C0 = 0.28209479177387814f; // 1 / (2 * sqrt(pi))
__constant__ float SH_C1 = 0.4886025119029199f;  // sqrt(3) / (2 * sqrt(pi))
__constant__ float SH_C2[] = {
    1.0925484305920792f,  // sqrt(15) / (2 * sqrt(pi))
    -1.0925484305920792f, // -sqrt(15) / (2 * sqrt(pi))
    0.31539156525252005f, // sqrt(5) / (4 * sqrt(pi))
    -1.0925484305920792f, // -sqrt(15) / (2 * sqrt(pi))
    0.5462742152960396f   // sqrt(15) / (4 * sqrt(pi))
};

// Evaluate spherical harmonics up to degree 2
__device__ inline void eval_sh2(const float x, const float y, const float z, float* sh) {
    // Degree 0
    sh[0] = SH_C0;

    // Degree 1
    sh[1] = -SH_C1 * y;
    sh[2] = SH_C1 * z;
    sh[3] = -SH_C1 * x;

    // Degree 2
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;

    sh[4] = SH_C2[0] * xy;
    sh[5] = SH_C2[1] * yz;
    sh[6] = SH_C2[2] * (2.0f * zz - xx - yy);
    sh[7] = SH_C2[3] * xz;
    sh[8] = SH_C2[4] * (xx - yy);
}

} // namespace

__global__ void encode_view_dir_sh2_kernel(
    int B, int H, int W,
    const float* __restrict__ rast, // [B, H, W, 4]
    const float* inv_mvp,           // [B, 4, 4]
    float* __restrict__ encoding    // [B, H, W, 9]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const unsigned int batch = idx / (H * W);
    const unsigned int y = (idx % (H * W)) / W;
    const unsigned int x = (idx % (H * W)) % W;

    const int tri_id = static_cast<int>(rast[idx * 4 + 3]) - 1;
    if (tri_id < 0) return;

    const float ndc_x = -1.f + 2.f * (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
    const float ndc_y = -1.f + 2.f * (static_cast<float>(y) + 0.5f) / static_cast<float>(H);

    float view_dir[3];
    compute_view_direction(ndc_x, ndc_y, &inv_mvp[batch * 16], view_dir);

    float basis[9];
    eval_sh2(view_dir[0], view_dir[1], view_dir[2], basis);

    #pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        encoding[idx * 9 + i] = basis[i];
    }
}

void encode_view_dir_sh2(
    int B, int H, int W,
    const float* __restrict__ rast, // [B, H, W, 4]
    const float* inv_mvp,           // [B, 4, 4]
    float* __restrict__ encoding    // [B, H, W, 9]
) {
    encode_view_dir_sh2_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS>>>(
        B, H, W, rast, inv_mvp, encoding
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace diffsoup