#include "rasterize.cuh"
#include "cuda_common.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace diffsoup {
namespace cuda {

__device__ __forceinline__ int outcode4(float x, float y, float z, float w) {
    int c = 0;
    c |= (x < -w) ? 1 : 0;  c |= (x >  w) ? 2 : 0;
    c |= (y < -w) ? 4 : 0;  c |= (y >  w) ? 8 : 0;
    c |= (z < -w) ? 16: 0;  c |= (z >  w) ? 32: 0;
    return c;
}

__device__ __forceinline__ void accum_ndc4(
    float x, float y, float z, float w,
    float& xmin, float& ymin, float& zmin,
    float& xmax, float& ymax, float& zmax)
{
    const float invw = 1.f / w;
    float nx = x * invw;
    float ny = y * invw;
    float nz = z * invw;

    // Guard nans/infs: clamp toward boundary to stay conservative
    if (!isfinite(nx)) nx = (nx > 0.f ? 1.f : -1.f);
    if (!isfinite(ny)) ny = (ny > 0.f ? 1.f : -1.f);
    if (!isfinite(nz)) nz = (nz > 0.f ? 1.f : -1.f);

    // Clamp to clip cube
    nx = fminf(1.f, fmaxf(-1.f, nx));
    ny = fminf(1.f, fmaxf(-1.f, ny));
    nz = fminf(1.f, fmaxf(-1.f, nz));

    xmin = fminf(xmin, nx);  xmax = fmaxf(xmax, nx);
    ymin = fminf(ymin, ny);  ymax = fmaxf(ymax, ny);
    zmin = fminf(zmin, nz);  zmax = fmaxf(zmax, nz);
}

__global__ void compute_triangle_rects_kernel(
    int H, int W, int B,
    int V, const float* pos,      // [B * V][4]
    int T, const int* tri,        // [T][3]
    int* triangle_rects,          // [B * T][4]: h0, h_len, w0, w_len
    int* frag_counts              // [B * T]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    const unsigned int b = idx / T;
    const unsigned int t = idx % T;

    const int t0 = tri[t * 3 + 0];
    const int t1 = tri[t * 3 + 1];
    const int t2 = tri[t * 3 + 2];

    // Fetch clip-space vertices (x,y,z,w)
    const float p0x = pos[(b * V + t0) * 4 + 0];
    const float p0y = pos[(b * V + t0) * 4 + 1];
    const float p0z = pos[(b * V + t0) * 4 + 2];
    const float p0w = pos[(b * V + t0) * 4 + 3];

    const float p1x = pos[(b * V + t1) * 4 + 0];
    const float p1y = pos[(b * V + t1) * 4 + 1];
    const float p1z = pos[(b * V + t1) * 4 + 2];
    const float p1w = pos[(b * V + t1) * 4 + 3];

    const float p2x = pos[(b * V + t2) * 4 + 0];
    const float p2y = pos[(b * V + t2) * 4 + 1];
    const float p2z = pos[(b * V + t2) * 4 + 2];
    const float p2w = pos[(b * V + t2) * 4 + 3];

    // Require all w>0 (hardware-like)
    if (!(p0w > 0.f && p1w > 0.f && p2w > 0.f)) {
        triangle_rects[idx*4+0]=0; triangle_rects[idx*4+1]=0;
        triangle_rects[idx*4+2]=0; triangle_rects[idx*4+3]=0;
        frag_counts[idx]=0; return;
    }

    // Homogeneous trivial reject against all 6 planes
    const int c0 = outcode4(p0x,p0y,p0z,p0w);
    const int c1 = outcode4(p1x,p1y,p1z,p1w);
    const int c2 = outcode4(p2x,p2y,p2z,p2w);
    if ((c0 & c1 & c2) != 0) {
        triangle_rects[idx*4+0]=0; triangle_rects[idx*4+1]=0;
        triangle_rects[idx*4+2]=0; triangle_rects[idx*4+3]=0;
        frag_counts[idx]=0; return;
    }

    // Project to NDC and clamp to [-1,1]
    float xmin =  1.f, ymin =  1.f, zmin =  1.f;
    float xmax = -1.f, ymax = -1.f, zmax = -1.f;

    accum_ndc4(p0x,p0y,p0z,p0w, xmin,ymin,zmin, xmax,ymax,zmax);
    accum_ndc4(p1x,p1y,p1z,p1w, xmin,ymin,zmin, xmax,ymax,zmax);
    accum_ndc4(p2x,p2y,p2z,p2w, xmin,ymin,zmin, xmax,ymax,zmax);

    // Z reject in NDC
    if (zmax < -1.f || zmin > 1.f) {
        triangle_rects[idx*4+0]=0; triangle_rects[idx*4+1]=0;
        triangle_rects[idx*4+2]=0; triangle_rects[idx*4+3]=0;
        frag_counts[idx]=0; return;
    }

    // NDC -> pixel bbox (exclusive max)
    const float fH = static_cast<float>(H);
    const float fW = static_cast<float>(W);

    float x0f = 0.5f * (xmin + 1.f) * fW;
    float x1f = 0.5f * (xmax + 1.f) * fW;
    float y0f = 0.5f * (ymin + 1.f) * fH;
    float y1f = 0.5f * (ymax + 1.f) * fH;

    if (x0f > x1f) { float tmp=x0f; x0f=x1f; x1f=tmp; }
    if (y0f > y1f) { float tmp=y0f; y0f=y1f; y1f=tmp; }

    int w0 = MAX(static_cast<int>(floorf(x0f)), 0);
    int w1 = MIN(static_cast<int>(floorf(x1f)), W - 1) + 1;
    int h0 = MAX(static_cast<int>(floorf(y0f)), 0);
    int h1 = MIN(static_cast<int>(floorf(y1f)), H - 1) + 1;

    const int w_len = MAX(w1 - w0, 0);
    const int h_len = MAX(h1 - h0, 0);

    if (w_len == 0 || h_len == 0) {
        triangle_rects[idx*4+0]=0; triangle_rects[idx*4+1]=0;
        triangle_rects[idx*4+2]=0; triangle_rects[idx*4+3]=0;
        frag_counts[idx]=0; return;
    }

    triangle_rects[idx*4+0]=h0; triangle_rects[idx*4+1]=h_len;
    triangle_rects[idx*4+2]=w0; triangle_rects[idx*4+3]=w_len;
    frag_counts[idx]=h_len * w_len;
}

int compute_triangle_rects(
    int H, int W, int B,
    int V, const float* pos,     // [B * V][4]
    int T, const int* tri,       // [T][3]
    int* triangle_rects,         // [B * T][4]: h0, h_len, w0, w_len
    int* frag_prefix_sum         // [B * T]
) {
    const int total = B * T;
    if (total == 0) return 0;

    int* frag_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&frag_counts, sizeof(int) * total));

    compute_triangle_rects_kernel<<<CUDA_BLOCKS(total), CUDA_THREADS>>>(
        H, W, B, V, pos, T, tri, triangle_rects, frag_counts
    );

    // Use Thrust for prefix sum on raw CUDA memory
    thrust::inclusive_scan(
        thrust::device_pointer_cast(frag_counts),
        thrust::device_pointer_cast(frag_counts + total),
        thrust::device_pointer_cast(frag_prefix_sum)
    );

    int num_frags = 0;
    CUDA_CHECK(cudaMemcpy(&num_frags, frag_prefix_sum + (total - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(frag_counts));
    CUDA_CHECK(cudaGetLastError());
    return num_frags;
}

__forceinline__ __device__ bool intersect_triangle(
    float ndc_y, float ndc_x, // pixel coordinates in NDC
    const float* pos,         // [V, 4]
    const int tri[3],         // triangle indices
    float &b0, float &b1,     // barycentric coordinates
    float &out_z,             // clip-space z coordinate of the fragment
    float &out_w              // clip-space w coordinate of the fragment
) {
    const int i0 = tri[0];
    const int i1 = tri[1];
    const int i2 = tri[2];

    const float p0x = pos[i0 * 4 + 0], p0y = pos[i0 * 4 + 1], p0z = pos[i0 * 4 + 2], p0w = pos[i0 * 4 + 3];
    const float p1x = pos[i1 * 4 + 0], p1y = pos[i1 * 4 + 1], p1z = pos[i1 * 4 + 2], p1w = pos[i1 * 4 + 3];
    const float p2x = pos[i2 * 4 + 0], p2y = pos[i2 * 4 + 1], p2z = pos[i2 * 4 + 2], p2w = pos[i2 * 4 + 3];

    const float q0x = p0x - ndc_x * p0w, q0y = p0y - ndc_y * p0w;
    const float q1x = p1x - ndc_x * p1w, q1y = p1y - ndc_y * p1w;
    const float q2x = p2x - ndc_x * p2w, q2y = p2y - ndc_y * p2w;

    const float A0 = q1x * q2y - q1y * q2x;
    const float A1 = q2x * q0y - q2y * q0x;
    const float A2 = q0x * q1y - q0y * q1x;
    const float A = A0 + A1 + A2;

    if (A == 0.f) return false;

    const float invA = 1.f / A;
    b0 = A0 * invA;
    b1 = A1 * invA;
    const float b2 = 1.0f - b0 - b1;

    out_z = b0 * p0z + b1 * p1z + b2 * p2z;
    out_w = b0 * p0w + b1 * p1w + b2 * p2w;

    return b0 >= 0.f && b1 >= 0.f && b0 + b1 <= 1.f;
}

__global__ void compute_fragments_kernel(
    int H, int W,
    int V, const float* pos,
    int T, const int* tri,
    int num_tris,                // == B * T
    int num_frags,
    const int* frag_prefix_sum,  // [T]
    const int* triangle_rects,   // [T, 4]
    int* frag_pix,               // [num_frags, 3]
    float* frag_attrs            // [num_frags, 4]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    // Binary search to find triangle index
    int lo = 0, hi = num_tris;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (frag_prefix_sum[mid] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    const int tri_idx = lo;
    const int batch_idx = tri_idx / T;
    const int batch_tri = tri_idx % T;

    const int* rect = &triangle_rects[tri_idx * 4];
    const int h0 = rect[0];
    const int h_len = rect[1];
    const int w0 = rect[2];
    const int w_len = rect[3];
    if (w_len <= 0 || h_len <= 0) return;

    const int local_idx = static_cast<int>(idx) - (tri_idx == 0 ? 0 : frag_prefix_sum[tri_idx - 1]);
    const int y = h0 + local_idx / w_len;
    const int x = w0 + local_idx % w_len;

    if (x < 0 || x >= W || y < 0 || y >= H) return;

    const float ndc_y = -1.f + 2.f * (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
    const float ndc_x = -1.f + 2.f * (static_cast<float>(x) + 0.5f) / static_cast<float>(W);

    float b0, b1;
    float clip_z;
    float clip_w;

    const bool intersect = intersect_triangle(
        ndc_y, ndc_x,
        &pos[batch_idx * V * 4],
        &tri[batch_tri * 3],
        b0, b1, clip_z, clip_w
    );

    if (!intersect) return;

    const float depth = clip_z / clip_w;

    if (!(depth >= -1.f && depth <= 1.f)) return;

    frag_pix[idx * 3 + 0] = batch_idx;
    frag_pix[idx * 3 + 1] = y;
    frag_pix[idx * 3 + 2] = x;
    frag_attrs[idx * 4 + 0] = b0;
    frag_attrs[idx * 4 + 1] = b1;
    frag_attrs[idx * 4 + 2] = depth;
    frag_attrs[idx * 4 + 3] = static_cast<float>(batch_tri + 1);
}

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
) {
    if (num_frags == 0) return;

    compute_fragments_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS>>>(
        H, W,
        V, pos,
        T, tri,
        num_tris,
        num_frags,
        frag_prefix_sum,
        triangle_rects,
        frag_pix,
        frag_attrs
    );

    CUDA_CHECK(cudaGetLastError());
}

// --- Packing Helpers ---
__device__ __forceinline__ long long pack_depth_and_index(float zw, int index) {
    int zw_bits;
    memcpy(&zw_bits, &zw, sizeof(float)); // reinterpret float as int bits
    return (static_cast<long long>(zw_bits) << 32) | static_cast<unsigned int>(index);
}

__device__ __forceinline__ int unpack_index(long long packed) {
    return static_cast<int>(packed & 0xFFFFFFFFLL);
}

// --- Depth Test Kernel ---
__global__ void depth_test_kernel(
    int H, int W,
    int num_frags,
    const int* __restrict__ frag_pix,        // [num_frags, 3]
    const float* __restrict__ frag_attrs,    // [num_frags, 4]
    const float* __restrict__ frag_alpha,    // [num_frags]
    const float* __restrict__ alpha_thresh,  // [num_frags]
    long long* frag_index                    // [B, H, W], initialized to LLONG_MAX
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int batch = frag_pix[idx * 3 + 0];
    if (batch < 0) return;

    const int y = frag_pix[idx * 3 + 1];
    const int x = frag_pix[idx * 3 + 2];
    const int pixel = batch * H * W + y * W + x;

    if (frag_alpha[idx] < alpha_thresh[idx]) return;

    const float zw = frag_attrs[idx * 4 + 2];
    if (zw <= -1.f || !isfinite(zw)) return;

    const long long packed = pack_depth_and_index(zw + 2.f, static_cast<int>(idx));
    atomicMin(&frag_index[pixel], packed);
}

__global__ void gather_depth_test_kernel(
    int B, int H, int W,
    const long long* __restrict__ frag_index,  // [B, H, W]
    const float* __restrict__ frag_attrs,      // [num_frags, 4]
    float* __restrict__ rast_out               // [B, H, W, 4]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const long long packed = frag_index[idx];
    const int i_frag = unpack_index(packed);

    if (i_frag < 0) return;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rast_out[idx * 4 + i] = frag_attrs[i_frag * 4 + i];
    }
}

__global__ void fill_ll_max(long long* arr, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] = LLONG_MAX;
}

void depth_test(
    int B, int H, int W,
    int num_frags,
    const int* frag_pix,       // [num_frags, 3]
    const float* frag_attrs,   // [num_frags, 4]
    const float* frag_alpha,   // [num_frags]
    const float* alpha_thresh, // [num_frags]
    float* rast_out            // [B, H, W, 4]
) {
    if (num_frags == 0) return;

    const int total = B * H * W;

    long long* frag_index = nullptr;
    CUDA_CHECK(cudaMalloc(&frag_index, sizeof(long long) * total));
    fill_ll_max<<<CUDA_BLOCKS(total), CUDA_THREADS>>>(frag_index, total);

    depth_test_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS>>>(
        H, W, num_frags, frag_pix, frag_attrs,
        frag_alpha, alpha_thresh, frag_index
    );

    gather_depth_test_kernel<<<CUDA_BLOCKS(total), CUDA_THREADS>>>(
        B, H, W, frag_index, frag_attrs, rast_out
    );

    CUDA_CHECK(cudaFree(frag_index));
    CUDA_CHECK(cudaGetLastError()); // catch kernel errors
}

__global__ void filter_valid_fragments_kernel(
    int num_frags,
    const int* __restrict__ frag_pix,        // [num_frags, 3]
    const float* __restrict__ frag_attrs,    // [num_frags, 4]
    int* __restrict__ frag_pix_out,          // [num_frags, 3] (preallocated)
    float* __restrict__ frag_attrs_out,      // [num_frags, 4]
    int* __restrict__ global_counter         // [1]
) {
    extern __shared__ int shared_scan[]; // shared memory for scan and block count
    int* valid_flags = shared_scan;
    int* block_base  = shared_scan + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Step 1: Each thread checks if valid
    int is_valid = 0;
    if (idx < num_frags) {
        int frag_val = frag_pix[3 * idx + 0];
        is_valid = (frag_val >= 0);
    }
    valid_flags[tid] = is_valid;
    __syncthreads();

    // Step 2: Inclusive scan (Hillis-Steele)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset)
            temp = valid_flags[tid - offset];
        __syncthreads();
        valid_flags[tid] += temp;
        __syncthreads();
    }

    // Step 3: First thread in block gets total count
    if (tid == blockDim.x - 1) {
        int total = valid_flags[tid];
        block_base[0] = atomicAdd(global_counter, total);
    }
    __syncthreads();

    // Step 4: Write to output if valid
    if (idx < num_frags && is_valid) {
        int local_idx = (tid > 0) ? valid_flags[tid - 1] : 0;
        int output_idx = block_base[0] + local_idx;

        // Copy frag_pix (3 ints)
        for (int i = 0; i < 3; ++i)
            frag_pix_out[3 * output_idx + i] = frag_pix[3 * idx + i];

        // Copy frag_attrs (4 floats)
        for (int i = 0; i < 4; ++i)
            frag_attrs_out[4 * output_idx + i] = frag_attrs[4 * idx + i];
    }
}

int filter_valid_fragments(
    int num_frags,
    const int* frag_pix,         // [num_frags, 3]
    const float* frag_attrs,     // [num_frags, 4]
    int* frag_pix_out,           // [num_frags, 3]
    float* frag_attrs_out        // [num_frags, 4]
) {
    int valid_count = 0;
    if (num_frags == 0) return valid_count;

    int* global_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&global_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(global_counter, 0, sizeof(int)));

    const size_t shared_mem = sizeof(int) * (CUDA_THREADS + 1);

    filter_valid_fragments_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS, shared_mem>>>(
        num_frags,
        frag_pix,
        frag_attrs,
        frag_pix_out,
        frag_attrs_out,
        global_counter
    );

    CUDA_CHECK(cudaMemcpy(&valid_count, global_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(global_counter));
    CUDA_CHECK(cudaGetLastError()); // catch kernel errors
    return valid_count;
}

// CUDA device function: tests intersection between segments (1–2) and (3–4)
// Writes the intersection point to (ix, iy) if they intersect
// Returns true for all intersection types (crossing, touching, overlapping)
// Arguments: x0,y0,x1,y1,x2,y2,x3,y3, (float* ix, float* iy), eps (default = 1e-7f)
__device__ __forceinline__
bool segment_intersect2D(
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3,
    float& ix, float& iy,
    float eps = 1e-7f)
{
    // Direction vectors
    const float r_x = x1 - x0;
    const float r_y = y1 - y0;
    const float s_x = x3 - x2;
    const float s_y = y3 - y2;

    const float denom = r_x * s_y - r_y * s_x;
    const float qp_x  = x2 - x0;
    const float qp_y  = y2 - y0;

    // Parallel or collinear
    if (fabsf(denom) <= eps) {
        const float cross_qp_r = qp_x * r_y - qp_y * r_x;
        if (fabsf(cross_qp_r) > eps) return false;  // parallel but not collinear

        const float rdotr = r_x * r_x + r_y * r_y;
        if (rdotr < eps) return false; // degenerate (zero-length)

        // Project segment 2 endpoints onto segment 1
        const float t0 = ((x2 - x0) * r_x + (y2 - y0) * r_y) / rdotr;
        const float t1 = ((x3 - x0) * r_x + (y3 - y0) * r_y) / rdotr;
        const float tmin = fminf(t0, t1), tmax = fmaxf(t0, t1);

        if (tmax < -eps || tmin > 1.f + eps) return false; // disjoint collinear

        // Overlapping; midpoint of overlap
        const float t = fmaxf(0.f, fminf(1.f, 0.5f * (fmaxf(0.f, tmin) + fminf(1.f, tmax))));
        ix = x0 + t * r_x;
        iy = y0 + t * r_y;
        return true;
    }

    // Non-parallel: compute intersection parameters
    const float t = (qp_x * s_y - qp_y * s_x) / denom;
    const float u = (qp_x * r_y - qp_y * r_x) / denom;

    if (t < -eps || t > 1.f + eps || u < -eps || u > 1.f + eps)
        return false; // intersection point outside segment bounds

    ix = x0 + t * r_x;
    iy = y0 + t * r_y;
    return true;
}

__device__ __forceinline__
bool find_edge_intersection(
    const float* pos, // [V, 4]
    const int tri[3],
    float x0, float y0,
    float x1, float y1,
    int &i0, int &i1,
    float &s0, float &s1
) {
    const int v0 = tri[0], v1 = tri[1], v2 = tri[2];

    const float p0x = pos[v0*4+0], p0y = pos[v0*4+1], p0w = pos[v0*4+3];
    const float p1x = pos[v1*4+0], p1y = pos[v1*4+1], p1w = pos[v1*4+3];
    const float p2x = pos[v2*4+0], p2y = pos[v2*4+1], p2w = pos[v2*4+3];

    const float r0x = p0x / p0w, r0y = p0y / p0w;
    const float r1x = p1x / p1w, r1y = p1y / p1w;
    const float r2x = p2x / p2w, r2y = p2y / p2w;

    float ix, iy;

    // Determine which edge is hit (same as before)
    float Ax, Ay, Aw, Bx, By, Bw;
    if (segment_intersect2D(x0, y0, x1, y1, r0x, r0y, r1x, r1y, ix, iy)) {
        i0 = 0; i1 = 1;
        Ax = p0x; Ay = p0y; Aw = p0w;
        Bx = p1x; By = p1y; Bw = p1w;
    } else if (segment_intersect2D(x0, y0, x1, y1, r1x, r1y, r2x, r2y, ix, iy)) {
        i0 = 1; i1 = 2;
        Ax = p1x; Ay = p1y; Aw = p1w;
        Bx = p2x; By = p2y; Bw = p2w;
    } else if (segment_intersect2D(x0, y0, x1, y1, r2x, r2y, r0x, r0y, ix, iy)) {
        i0 = 2; i1 = 0;
        Ax = p2x; Ay = p2y; Aw = p2w;
        Bx = p0x; By = p0y; Bw = p0w;
    } else {
        return false;
    }

    // Perspective-correct edge parameter t using both x & y (branch-light)
    const float dx = Bx - Ax;
    const float dy = By - Ay;
    const float dw = Bw - Aw;

    // a + t b = 0 in "warped" residual space where ndc = (ix, iy)
    const float ax = Ax - ix * Aw;
    const float ay = Ay - iy * Aw;
    const float bx = dx - ix * dw;
    const float by = dy - iy * dw;

    const float denom = bx*bx + by*by;
    const float t = (denom != 0.0f) ? -(ax*bx + ay*by) / denom : 0.5f; // homogeneous midpoint if degenerate

    // Edge weights (perspective-correct along the edge)
    s0 = 1.0f - t;
    s1 = t;

    return true;
}

// --- helpers (header-scope or above the kernel) -----------------------------

__device__ __forceinline__ bool tri_normal_ndc( // normal in NDC xyz, normalized
    const float* pos_clip, const int tri[3], float n[3])
{
    // Load clip and divide by w -> NDC
    const int v0=tri[0], v1=tri[1], v2=tri[2];
    const float x0=pos_clip[v0*4+0], y0=pos_clip[v0*4+1], z0=pos_clip[v0*4+2], w0=pos_clip[v0*4+3];
    const float x1=pos_clip[v1*4+0], y1=pos_clip[v1*4+1], z1=pos_clip[v1*4+2], w1=pos_clip[v1*4+3];
    const float x2=pos_clip[v2*4+0], y2=pos_clip[v2*4+1], z2=pos_clip[v2*4+2], w2=pos_clip[v2*4+3];
    if (fabsf(w0)<1e-30f || fabsf(w1)<1e-30f || fabsf(w2)<1e-30f) return false;

    const float X0=x0/w0, Y0=y0/w0, Z0=z0/w0;
    const float X1=x1/w1, Y1=y1/w1, Z1=z1/w1;
    const float X2=x2/w2, Y2=y2/w2, Z2=z2/w2;

    // Normal = cross(e1, e2) in NDC
    const float e1x=X1-X0, e1y=Y1-Y0, e1z=Z1-Z0;
    const float e2x=X2-X0, e2y=Y2-Y0, e2z=Z2-Z0;
    float nx = e1y*e2z - e1z*e2y;
    float ny = e1z*e2x - e1x*e2z;
    float nz = e1x*e2y - e1y*e2x;
    const float l2 = nx*nx + ny*ny + nz*nz;
    if (!(l2 > 0.f)) return false;
    const float invl = rsqrtf(l2);
    n[0]=nx*invl; n[1]=ny*invl; n[2]=nz*invl;
    return true;
}

// DRTK-style 2D mapping (EdgeGrad), returns db/dp in 2D plane
__device__ __forceinline__ void db_dp_2d(const float nvar_x, const float nvar_y,
                                         const float nfix_x, const float nfix_y,
                                         float &dbx, float &dby)
{
    // Normalize both
    float nvx=nvar_x, nvy=nvar_y, nfx=nfix_x, nfy=nfix_y;
    float il = rsqrtf(max(1e-30f, nvx*nvx+nvy*nvy)); nvx*=il; nvy*=il;
    il = rsqrtf(max(1e-30f, nfx*nfx+nfy*nfy));       nfx*=il; nfy*=il;

    // b = (-n_fix.y, n_fix.x)
    const float bx = -nfy, by = nfx;
    const float denom = bx*nvx + by*nvy;            // dot(b, n_vary)
    const float s = (fabsf(denom) > 1e-12f) ? (bx/denom) : 0.f;
    dbx = s * nvx;
    dby = s * nvy;
}

__global__ void edge_grad_kernel(
    int B, int H, int W, int C,
    const float* __restrict__ color,         // [B, H, W, C]
    const float* __restrict__ grad_color,    // [B, H, W, C]
    const float* __restrict__ rast,          // [B, H, W, 4]
    int V,
    const float* __restrict__ pos,           // [B, V, 4]
    float* __restrict__ grad_pos,            // [B, V, 4]
    const int* __restrict__ tri              // [T, 3]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const unsigned int batch = idx / (H * W);
    const unsigned int y0 = (idx % (H * W)) / W;
    const unsigned int x0 = (idx % (H * W)) % W;

    const float ndc_x0 = -1.f + 2.f * (static_cast<float>(x0) + 0.5f) / static_cast<float>(W);
    const float ndc_y0 = -1.f + 2.f * (static_cast<float>(y0) + 0.5f) / static_cast<float>(H);

    const int pix0 = batch * H * W + y0 * W + x0;
    const float depth0 = rast[pix0 * 4 + 2];
    const int it0 = static_cast<int>(rast[pix0 * 4 + 3]) - 1;

    if (it0 < 0) return;

    int verts0[3];
    verts0[0] = tri[it0 * 3 + 0];
    verts0[1] = tri[it0 * 3 + 1];
    verts0[2] = tri[it0 * 3 + 2];

    const float* batch_pos = &pos[batch * V * 4];

    const float* color0 = &color[pix0 * C];
    const float* grad_color0 = &grad_color[pix0 * C];

    float grad_pos_verts[12]{0.f};

    const int dy[4] = {0,  0, -1, 1};
    const int dx[4] = {1, -1,  0, 0};

    for (int i_edge = 0; i_edge < 4; ++i_edge) {
        const int y1 = y0 + dy[i_edge];
        const int x1 = x0 + dx[i_edge];

        if (y1 < 0 || y1 >= H || x1 < 0 || x1 >= W) continue;

        const int pix1 = batch * H * W + y1 * W + x1;
        const float depth1 = rast[pix1 * 4 + 2];
        const int it1 = static_cast<int>(rast[pix1 * 4 + 3]) - 1;

        const float* color1 = &color[pix1 * C];
        const float* grad_color1 = &grad_color[pix1 * C];

        float adjoint = 0.f;
        #pragma unroll 4
        for (int i = 0; i < C; ++i) {
            adjoint += 0.5f * (grad_color0[i] + grad_color1[i]) * (color0[i] - color1[i]);
        }

        if (adjoint == 0.f) continue;

        const float grad_x = adjoint * static_cast<float>(dx[i_edge]) * static_cast<float>(W) / 2.f;
        const float grad_y = adjoint * static_cast<float>(dy[i_edge]) * static_cast<float>(H) / 2.f;

        const float ndc_x1 = -1.f + 2.f * (static_cast<float>(x1) + 0.5f) / static_cast<float>(W);
        const float ndc_y1 = -1.f + 2.f * (static_cast<float>(y1) + 0.5f) / static_cast<float>(H);

        const float ndc_mid_x = 0.5f * (ndc_x0 + ndc_x1);
        const float ndc_mid_y = 0.5f * (ndc_y0 + ndc_y1);

        if (it1 == it0) {
            // not an edge
            float b0, b1, clip_z, clip_w;
            if (intersect_triangle(ndc_mid_y, ndc_mid_x, batch_pos, verts0, b0, b1, clip_z, clip_w)) {
                const float b2 = 1.f - b0 - b1;
                const float clip_x = ndc_mid_x * clip_w;
                const float clip_y = ndc_mid_y * clip_w;

                const float inv_w = 1.f / clip_w;
                const float inv_w2 = inv_w * inv_w;
                const float g_sum_w = grad_x * clip_x + grad_y * clip_y;

                grad_pos_verts[0 * 4 + 0] += b0 * grad_x * inv_w;
                grad_pos_verts[1 * 4 + 0] += b1 * grad_x * inv_w;
                grad_pos_verts[2 * 4 + 0] += b2 * grad_x * inv_w;

                grad_pos_verts[0 * 4 + 1] += b0 * grad_y * inv_w;
                grad_pos_verts[1 * 4 + 1] += b1 * grad_y * inv_w;
                grad_pos_verts[2 * 4 + 1] += b2 * grad_y * inv_w;

                grad_pos_verts[0 * 4 + 3] += -b0 * g_sum_w * inv_w2;
                grad_pos_verts[1 * 4 + 3] += -b1 * g_sum_w * inv_w2;
                grad_pos_verts[2 * 4 + 3] += -b2 * g_sum_w * inv_w2;
            }
            continue;
        }

        float clip_z1_it0, clip_w1_it0;
        bool intersect1_it0;
        {
            float b0, b1;
            intersect1_it0 = intersect_triangle(
                ndc_y1, ndc_x1, batch_pos, verts0,
                b0, b1, clip_z1_it0, clip_w1_it0
            );
        }

        if (intersect1_it0) {
            // this is an *implicit* edge (triangle intersection or opacity edge)
            const float depth1_it0 = clip_z1_it0 / clip_w1_it0;

            if (it1 >= 0 && depth1 < depth1_it0) {
                float clip_z0_it1, clip_w0_it1;
                bool intersect0_it1;
                {
                    float b0, b1;
                    intersect0_it1 = intersect_triangle(
                        ndc_y0, ndc_x0, batch_pos, &tri[it1 * 3],
                        b0, b1, clip_z0_it1, clip_w0_it1
                    );
                }

                if (intersect0_it1) {
                    const float depth0_it1 = clip_z0_it1 / clip_w0_it1;

                    if (depth0 < depth0_it1) {
                        // triangle intersection
                        float n_var[3], n_fix[3];
                        if (tri_normal_ndc(batch_pos, verts0, n_var) && tri_normal_ndc(batch_pos, &tri[it1*3], n_fix)) {
                            // Decide orientation of the neighbor pair:
                            const bool horiz_pair = (dx[i_edge] != 0);  // neighbor is left/right
                            float gx_ndc=0.f, gy_ndc=0.f, gz_ndc=0.f;

                            if (horiz_pair) {
                                // Use XZ plane, apply grad_x
                                float dbx, dbz;
                                db_dp_2d(
                                    /*n_vary*/ n_var[0], n_var[2],
                                    /*n_fix */ n_fix[0], n_fix[2],
                                    dbx, dbz
                                );
                                gx_ndc = grad_x * dbx;
                                gz_ndc = grad_x * dbz;
                                // gy_ndc stays 0
                            } else {
                                // Vertical neighbor: use YZ plane, apply grad_y
                                float dby, dbz;
                                db_dp_2d(
                                    /*n_vary*/ n_var[1], n_var[2],
                                    /*n_fix */ n_fix[1], n_fix[2],
                                    dby, dbz
                                );
                                gy_ndc = grad_y * dby;
                                gz_ndc = grad_y * dbz;
                                // gx_ndc stays 0
                            }

                            float b0, b1, clip_z, clip_w;
                            if (intersect_triangle(ndc_mid_y, ndc_mid_x, batch_pos, verts0, b0, b1, clip_z, clip_w)) {
                                const float b2 = 1.f - b0 - b1;
                                const float clip_x = ndc_mid_x * clip_w;
                                const float clip_y = ndc_mid_y * clip_w;

                                const float inv_w  = 1.f / clip_w;
                                const float inv_w2 = inv_w * inv_w;
                                const float g_sum_w = gx_ndc * clip_x + gy_ndc * clip_y + gz_ndc * clip_z;

                                grad_pos_verts[0*4+0] += b0 * gx_ndc * inv_w;
                                grad_pos_verts[1*4+0] += b1 * gx_ndc * inv_w;
                                grad_pos_verts[2*4+0] += b2 * gx_ndc * inv_w;

                                grad_pos_verts[0*4+1] += b0 * gy_ndc * inv_w;
                                grad_pos_verts[1*4+1] += b1 * gy_ndc * inv_w;
                                grad_pos_verts[2*4+1] += b2 * gy_ndc * inv_w;

                                grad_pos_verts[0*4+2] += b0 * gz_ndc * inv_w;
                                grad_pos_verts[1*4+2] += b1 * gz_ndc * inv_w;
                                grad_pos_verts[2*4+2] += b2 * gz_ndc * inv_w;

                                grad_pos_verts[0*4+3] += -b0 * g_sum_w * inv_w2;
                                grad_pos_verts[1*4+3] += -b1 * g_sum_w * inv_w2;
                                grad_pos_verts[2*4+3] += -b2 * g_sum_w * inv_w2;
                            }
                        }
                    }
                }
            }
            else if (it1 < 0 || depth0 < depth1) {
                // active opacity edge
                float b0, b1, clip_z, clip_w;
                if (intersect_triangle(ndc_mid_y, ndc_mid_x, batch_pos, verts0, b0, b1, clip_z, clip_w)) {
                    const float b2 = 1.f - b0 - b1;
                    const float clip_x = ndc_mid_x * clip_w;
                    const float clip_y = ndc_mid_y * clip_w;

                    const float inv_w = 1.f / clip_w;
                    const float inv_w2 = inv_w * inv_w;
                    const float g_sum_w = grad_x * clip_x + grad_y * clip_y;

                    grad_pos_verts[0 * 4 + 0] += b0 * grad_x * inv_w;
                    grad_pos_verts[1 * 4 + 0] += b1 * grad_x * inv_w;
                    grad_pos_verts[2 * 4 + 0] += b2 * grad_x * inv_w;

                    grad_pos_verts[0 * 4 + 1] += b0 * grad_y * inv_w;
                    grad_pos_verts[1 * 4 + 1] += b1 * grad_y * inv_w;
                    grad_pos_verts[2 * 4 + 1] += b2 * grad_y * inv_w;

                    grad_pos_verts[0 * 4 + 3] += -b0 * g_sum_w * inv_w2;
                    grad_pos_verts[1 * 4 + 3] += -b1 * g_sum_w * inv_w2;
                    grad_pos_verts[2 * 4 + 3] += -b2 * g_sum_w * inv_w2;
                }
            }
        }
        else {
            // this is a *geometric* edge
            if (it1 < 0 || depth0 < depth1) {
                // active geometric edge
                int i0, i1;
                float s0, s1;

                if (find_edge_intersection(batch_pos, verts0, ndc_x0, ndc_y0, ndc_x1, ndc_y1, i0, i1, s0, s1)) {
                    const float clip_x = s0 * batch_pos[verts0[i0] * 4 + 0] + s1 * batch_pos[verts0[i1] * 4 + 0];
                    const float clip_y = s0 * batch_pos[verts0[i0] * 4 + 1] + s1 * batch_pos[verts0[i1] * 4 + 1];
                    const float clip_w = s0 * batch_pos[verts0[i0] * 4 + 3] + s1 * batch_pos[verts0[i1] * 4 + 3];

                    const float inv_w = 1.f / clip_w;
                    const float inv_w2 = inv_w * inv_w;
                    const float g_sum_w = grad_x * clip_x + grad_y * clip_y;

                    grad_pos_verts[i0 * 4 + 0] += s0 * grad_x * inv_w;
                    grad_pos_verts[i1 * 4 + 0] += s1 * grad_x * inv_w;
                    grad_pos_verts[i0 * 4 + 1] += s0 * grad_y * inv_w;
                    grad_pos_verts[i1 * 4 + 1] += s1 * grad_y * inv_w;
                    grad_pos_verts[i0 * 4 + 3] += -s0 * g_sum_w * inv_w2;
                    grad_pos_verts[i1 * 4 + 3] += -s1 * g_sum_w * inv_w2;
                }
            }
        }
    }

    float* batch_grad_pos = &grad_pos[batch * V * 4];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            atomicAdd(&batch_grad_pos[verts0[i] * 4 + j], grad_pos_verts[i * 4 + j]);
        }
    }
}

void backward_edge_grad(
    int B, int H, int W, int C,
    const float* __restrict__ color,         // [B, H, W, C]
    const float* __restrict__ grad_color,    // [B, H, W, C]
    const float* __restrict__ rast,          // [B, H, W, 4]
    int V,
    const float* __restrict__ pos,           // [B, V, 4]
    float* __restrict__ grad_pos,            // [B, V, 4]
    const int* __restrict__ tri              // [T, 3]
) {
    edge_grad_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS>>>(
        B, H, W, C, color, grad_color, rast,
        V, pos, grad_pos, tri
    );

    CUDA_CHECK(cudaGetLastError()); // catch kernel errors
}

__global__ void backward_radiance_field_loss_kernel(
    int B, int H, int W, int C,
    const float* __restrict__ color,         // [B, H, W, C]
    const float* __restrict__ target,        // [B, H, W, C]
    const float* __restrict__ rast,          // [B, H, W, 4]
    int num_frags,
    const int* __restrict__ frag_pix,        // [num_frags, 3]
    const float* __restrict__ frag_attrs,    // [num_frags, 4]
    const float* __restrict__ frag_alpha,    // [num_frags]
    float* __restrict__ grad_frag_alpha      // [num_frags]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int batch = frag_pix[idx * 3 + 0];
    const int y = frag_pix[idx * 3 + 1];
    const int x = frag_pix[idx * 3 + 2];
    const int pixel = batch * H * W + y * W + x;

    if (batch < 0) return;

    const float depth_frag = frag_attrs[idx * 4 + 2];
    const int triangle_frag = static_cast<int>(frag_attrs[idx * 4 + 3]) - 1;

    const float depth_pixel = rast[pixel * 4 + 2];
    const int triangle_pixel = static_cast<int>(rast[pixel * 4 + 3]) - 1;

    if (triangle_pixel >= 0 && depth_frag > depth_pixel) return;

    const float* color_pixel = &color[pixel * C];
    const float* target_pixel = &target[pixel * C];

    float loss = 0.f;

    #pragma unroll 4
    for (int i = 0; i < C; ++i) {
        loss += abs(color_pixel[i] - target_pixel[i]);
    }

    constexpr float eps = 1e-5f;
    const float alpha = frag_alpha[idx];

    if (triangle_pixel >= 0 && triangle_frag == triangle_pixel) {
        grad_frag_alpha[idx] += loss / MAX(alpha, eps);
    }
    else {
        grad_frag_alpha[idx] -= loss / MAX(1.f - alpha, eps);
    }
}

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
) {
    if (num_frags == 0) return;

    backward_radiance_field_loss_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS>>>(
        B, H, W, C, color, target, rast, num_frags, frag_pix, frag_attrs,
        frag_alpha, grad_frag_alpha
    );

    CUDA_CHECK(cudaGetLastError()); // catch kernel errors
}

__global__ void frag_alpha_mobilenerf_kernel(
    int num_frags,
    const float* frag_attrs, // [num_frags, 4]
    const float* uvs,        // [V, 2]
    const int* tri,          // [T, 3]
    int texH,
    int texW,
    const float* feat0,      // [texH, texW, 4]
    float* frag_alpha        // [num_frags,]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int tri_id = static_cast<int>(frag_attrs[idx * 4 + 3]) - 1;
    if (tri_id < 0) return;

    const float b0 = frag_attrs[idx * 4 + 0];
    const float b1 = frag_attrs[idx * 4 + 1];
    const float b2 = 1.f - b0 - b1;

    const int i0 = tri[tri_id * 3 + 0];
    const int i1 = tri[tri_id * 3 + 1];
    const int i2 = tri[tri_id * 3 + 2];

    // ----- barycentric UV interpolation -----
    const float u = b0 * uvs[i0 * 2 + 0]
                  + b1 * uvs[i1 * 2 + 0]
                  + b2 * uvs[i2 * 2 + 0];

    const float v = b0 * uvs[i0 * 2 + 1]
                  + b1 * uvs[i1 * 2 + 1]
                  + b2 * uvs[i2 * 2 + 1];

    // ----- nearest neighbor texel indices -----
    // assuming u,v in [0,1], left→right, top→bottom (or whatever your UVs are)
    float fx = u * (texW - 1);
    float fy = v * (texH - 1);

    int ix = static_cast<int>(floorf(fx + 0.5f));  // round to nearest
    int iy = static_cast<int>(floorf(fy + 0.5f));

    // clamp to valid range
    ix = max(0, min(texW - 1, ix));
    iy = max(0, min(texH - 1, iy));

    // ----- flat index into feat0[y, x, c] -----
    const int tex_idx = (iy * texW + ix) * 4;  // 4 = RGBA

    // assuming alpha is channel 3
    const float alpha = feat0[tex_idx + 0];
    frag_alpha[idx] = alpha;
}

void frag_alpha_mobilenerf(
    int num_frags,
    const float* frag_attrs, // [num_frags, 4]
    const float* uvs,        // [V, 2]
    const int* tri,          // [T, 3]
    int texH,
    int texW,
    const float* feat0,      // [texH, texW, 4]
    float* frag_alpha        // [num_frags,]
) {
    if (num_frags == 0) return;

    frag_alpha_mobilenerf_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS>>>(
        num_frags, frag_attrs, uvs, tri, texH, texW,
        feat0, frag_alpha
    );

    CUDA_CHECK(cudaGetLastError()); // catch kernel errors
}

__global__ void lookup_feats_mobilenerf_kernel(
    int B, int H, int W,
    float* rast,         // [B, H, W, 4]
    const float* uvs,    // [V, 2]
    const int* tri,      // [T, 3]
    int texH,
    int texW,
    const float* feat0,  // [texH, texW, 4]
    const float* feat1,  // [texH, texW, 4]
    float* image         // [H, W, 8]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const int tri_id = static_cast<int>(rast[idx * 4 + 3]) - 1;
    if (tri_id < 0) return;

    const float b0 = rast[idx * 4 + 0];
    const float b1 = rast[idx * 4 + 1];
    const float b2 = 1.f - b0 - b1;

    const int i0 = tri[tri_id * 3 + 0];
    const int i1 = tri[tri_id * 3 + 1];
    const int i2 = tri[tri_id * 3 + 2];

    // ----- barycentric UV interpolation -----
    const float u = b0 * uvs[i0 * 2 + 0]
                  + b1 * uvs[i1 * 2 + 0]
                  + b2 * uvs[i2 * 2 + 0];

    const float v = b0 * uvs[i0 * 2 + 1]
                  + b1 * uvs[i1 * 2 + 1]
                  + b2 * uvs[i2 * 2 + 1];

    // ----- nearest neighbor texel indices -----
    // assuming u,v in [0,1], left→right, top→bottom (or whatever your UVs are)
    float fx = u * (texW - 1);
    float fy = v * (texH - 1);

    int ix = static_cast<int>(floorf(fx + 0.5f));  // round to nearest
    int iy = static_cast<int>(floorf(fy + 0.5f));

    // clamp to valid range
    ix = max(0, min(texW - 1, ix));
    iy = max(0, min(texH - 1, iy));

    // ----- flat index into feat0[y, x, c] -----
    const int tex_idx = (iy * texW + ix) * 4;  // 4 = RGBA

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        image[idx * 8 + 0 + i] = feat0[tex_idx + i];
        image[idx * 8 + 4 + i] = feat1[tex_idx + i];
    }
}

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
) {
    const int total = B * H * W;

    lookup_feats_mobilenerf_kernel<<<CUDA_BLOCKS(total), CUDA_THREADS>>>(
        B, H, W, rast, uvs, tri, texH, texW,
        feat0, feat1, image
    );

    CUDA_CHECK(cudaGetLastError()); // catch kernel errors
}

} // namespace cuda
} // namespace diffsoup