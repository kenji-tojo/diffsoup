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

__device__ void perturb_direction_vmf(
    float* direction,      // Direction to perturb (modified in-place)
    float kappa,          // Concentration parameter
    float sample0,        // Uniform(0,1) for w
    float sample1         // Uniform(0,1) for theta
) {
    // First, normalize the input direction and save it as the mean
    float mu_x = direction[0];
    float mu_y = direction[1];
    float mu_z = direction[2];

    float norm = sqrtf(mu_x * mu_x + mu_y * mu_y + mu_z * mu_z);
    mu_x /= norm;
    mu_y /= norm;
    mu_z /= norm;

    // Step 1: Transform sample0 to w using inverse CDF
    float exp_neg_kappa = expf(-kappa);
    float two_sinh_kappa = 2.0f * sinhf(kappa);
    float w = logf(exp_neg_kappa + two_sinh_kappa * sample0) / kappa;

    // Convert sample1 to angle theta
    float theta = sample1 * 2.0f * 3.14159265359f;

    // Step 2: Build orthonormal basis with mu
    // Find vector orthogonal to mu
    float v1_x, v1_y, v1_z;
    if (fabsf(mu_x) < 0.9f) {
        v1_x = 1.0f; v1_y = 0.0f; v1_z = 0.0f;
    } else {
        v1_x = 0.0f; v1_y = 1.0f; v1_z = 0.0f;
    }

    // Gram-Schmidt
    float dot = v1_x * mu_x + v1_y * mu_y + v1_z * mu_z;
    v1_x -= dot * mu_x;
    v1_y -= dot * mu_y;
    v1_z -= dot * mu_z;

    // Normalize
    float v1_norm = sqrtf(v1_x * v1_x + v1_y * v1_y + v1_z * v1_z);
    float inv_v1_norm = 1.0f / v1_norm;
    v1_x *= inv_v1_norm;
    v1_y *= inv_v1_norm;
    v1_z *= inv_v1_norm;

    // Cross product for second orthogonal vector
    float v2_x = mu_y * v1_z - mu_z * v1_y;
    float v2_y = mu_z * v1_x - mu_x * v1_z;  
    float v2_z = mu_x * v1_y - mu_y * v1_x;

    // Generate vMF sample and write directly to direction
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    float sqrt_1_minus_w2 = sqrtf(fmaxf(0.0f, 1.0f - w * w));

    direction[0] = w * mu_x + sqrt_1_minus_w2 * (cos_theta * v1_x + sin_theta * v2_x);
    direction[1] = w * mu_y + sqrt_1_minus_w2 * (cos_theta * v1_y + sin_theta * v2_y);
    direction[2] = w * mu_z + sqrt_1_minus_w2 * (cos_theta * v1_z + sin_theta * v2_z);
}
} // namespace

__global__ void encode_view_dir_freq_kernel(
    int B, int H, int W,
    const float* __restrict__ rast, // [B, H, W, 4]
    const float* inv_mvp,           // [B, 4, 4]
    float freq,
    float* __restrict__ encoding,   // [B, H, W, 9]
    float vmf_kappa,
    const float* __restrict__ vmf_samples // [B, H, W, 2]
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

    if (vmf_kappa > 0.f && vmf_samples != nullptr) {
        perturb_direction_vmf(view_dir, vmf_kappa, vmf_samples[idx * 2 + 0], vmf_samples[idx * 2 + 1]);
    }

    const float vx = view_dir[0];
    const float vy = view_dir[1];
    const float vz = view_dir[2];

    encoding[idx * 9 + 0] = vx;
    encoding[idx * 9 + 1] = vy;
    encoding[idx * 9 + 2] = vz;
    encoding[idx * 9 + 3] = sin(pi<float>() * 2.0f * freq * vx);
    encoding[idx * 9 + 4] = sin(pi<float>() * 2.0f * freq * vy);
    encoding[idx * 9 + 5] = sin(pi<float>() * 2.0f * freq * vz);
    encoding[idx * 9 + 6] = cos(pi<float>() * 2.0f * freq * vx);
    encoding[idx * 9 + 7] = cos(pi<float>() * 2.0f * freq * vy);
    encoding[idx * 9 + 8] = cos(pi<float>() * 2.0f * freq * vz);
}

void encode_view_dir_freq(
    int B, int H, int W,
    const float* __restrict__ rast, // [B, H, W, 4]
    const float* inv_mvp,           // [B, 4, 4]
    float freq,
    float* __restrict__ encoding,   // [B, H, W, 9]
    float vmf_kappa,
    const float* __restrict__ vmf_samples // [B, H, W, 2]
) {
    encode_view_dir_freq_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS>>>(
        B, H, W, rast, inv_mvp, freq, encoding, vmf_kappa, vmf_samples
    );
    CUDA_CHECK(cudaGetLastError());
}

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