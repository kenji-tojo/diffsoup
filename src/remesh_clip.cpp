#include "remesh_clip.h"

namespace diffsoup {

// Inside test in NDC cube [-1,1]^3
static inline bool inside_ndc_3D(float x, float y, float z) {
    return (x >= -1.0f && x <= 1.0f) &&
           (y >= -1.0f && y <= 1.0f) &&
           (z >= -1.0f && z <= 1.0f);
}

// Safe division to NDC; returns (ndc_val, bad_w flag)
static inline std::pair<float,bool> to_ndc(float x, float w) {
    if (w > 0.0f) return std::make_pair(x / w, false);
    return std::make_pair(0.0f, true);
}

static inline void hom_transform(
    const float mat[16],
    const float in[3],
    float out[4]
) {
    out[0] = mat[0]  * in[0] + mat[1]  * in[1] + mat[2]  * in[2] + mat[3];
    out[1] = mat[4]  * in[0] + mat[5]  * in[1] + mat[6]  * in[2] + mat[7];
    out[2] = mat[8]  * in[0] + mat[9]  * in[1] + mat[10] * in[2] + mat[11];
    out[3] = mat[12] * in[0] + mat[13] * in[1] + mat[14] * in[2] + mat[15];
}

// Length in “image-height units”:
// - Convert both endpoints to NDC (x/w, y/w, z/w).
// - Otherwise, dx is scaled by aspect (W/H) so metric matches image height.
inline float TriangleSoupSplitterClip::screenLen2Between(int a, int b, float aspectWH) const {
    const float* Aworld = p(a);
    const float* Bworld = p(b);

    float A[4], B[4];
    hom_transform(mvp4x4, Aworld, A);
    hom_transform(mvp4x4, Bworld, B);

    std::pair<float,bool> Ax_p = to_ndc(A[0], A[3]);
    std::pair<float,bool> Ay_p = to_ndc(A[1], A[3]);
    std::pair<float,bool> Az_p = to_ndc(A[2], A[3]);

    std::pair<float,bool> Bx_p = to_ndc(B[0], B[3]);
    std::pair<float,bool> By_p = to_ndc(B[1], B[3]);
    std::pair<float,bool> Bz_p = to_ndc(B[2], B[3]);

    const bool A_wbad = (Ax_p.second || Ay_p.second || Az_p.second);
    const bool B_wbad = (Bx_p.second || By_p.second || Bz_p.second);

    const float Ax = Ax_p.first, Ay = Ay_p.first, Az = Az_p.first;
    const float Bx = Bx_p.first, By = By_p.first, Bz = Bz_p.first;

    const bool Ainside = !A_wbad && inside_ndc_3D(Ax, Ay, Az);
    const bool Binside = !B_wbad && inside_ndc_3D(Bx, By, Bz);

    // If either of the endpoints is outside the NDC cube, ignore the edge
    if (!Ainside || !Binside) return 0.0f;

    // Distance in "image-height units": scale x by aspect (W/H)
    const float dx = (Ax - Bx) * aspectWH;
    const float dy = (Ay - By);
    return dx * dx + dy * dy;
}

void TriangleSoupSplitterClip::enqueueTriangleEdges(
    int t, std::priority_queue<EdgeRef>& pq, float aspectWH) const
{
    int i0, i1, i2;
    triIndices(t, i0, i1, i2);
    int g = triGen[t];

    const float len2_01 = screenLen2Between(i0, i1, aspectWH);
    const float len2_12 = screenLen2Between(i1, i2, aspectWH);
    const float len2_20 = screenLen2Between(i2, i0, aspectWH);

    if (len2_01 > 0) pq.push(EdgeRef{t, 0, len2_01, g});
    if (len2_12 > 0) pq.push(EdgeRef{t, 1, len2_12, g});
    if (len2_20 > 0) pq.push(EdgeRef{t, 2, len2_20, g});
}

int TriangleSoupSplitterClip::splitTriangleEdge(int t, int e) {
    int i0, i1, i2;
    triIndices(t, i0, i1, i2);

    int a, b, c;
    if (e == 0) { a = i0; b = i1; c = i2; }
    else if (e == 1) { a = i1; b = i2; c = i0; }
    else             { a = i2; b = i0; c = i1; }

    const float* Aworld = p(a);
    const float* Bworld = p(b);

    float A[4], B[4];
    hom_transform(mvp4x4, Aworld, A);
    hom_transform(mvp4x4, Bworld, B);

    // Compute NDC for endpoints
    std::pair<float,bool> Ax_p = to_ndc(A[0], A[3]);
    std::pair<float,bool> Ay_p = to_ndc(A[1], A[3]);
    std::pair<float,bool> Az_p = to_ndc(A[2], A[3]);

    std::pair<float,bool> Bx_p = to_ndc(B[0], B[3]);
    std::pair<float,bool> By_p = to_ndc(B[1], B[3]);
    std::pair<float,bool> Bz_p = to_ndc(B[2], B[3]);

    const float Ax = Ax_p.first, Ay = Ay_p.first, Az = Az_p.first;
    const float Bx = Bx_p.first, By = By_p.first, Bz = Bz_p.first;

    // NDC midpoint -> halves the screen-space distance
    const float ndc_x = 0.5f * (Ax + Bx);
    const float ndc_y = 0.5f * (Ay + By);

    // Perspective-correct edge parameter t using both x & y (branch-light)
    const float dx = B[0] - A[0];
    const float dy = B[1] - A[1];
    const float dw = B[3] - A[3];

    // a + t b = 0 in "warped" residual space where ndc = (ix, iy)
    const float ax = A[0] - ndc_x * A[3];
    const float ay = A[1] - ndc_y * A[3];
    const float bx = dx - ndc_x * dw;
    const float by = dy - ndc_y * dw;

    const float denom = bx*bx + by*by;
    const float tB = (denom != 0.0f) ? -(ax*bx + ay*by) / denom : 0.5f; // homogeneous midpoint if degenerate
    const float tA = 1.0f - tB;

    float Cworld[3];
    Cworld[0] = tA * Aworld[0] + tB * Bworld[0];
    Cworld[1] = tA * Aworld[1] + tB * Bworld[1];
    Cworld[2] = tA * Aworld[2] + tB * Bworld[2];

    int mA = addVertex(Cworld[0], Cworld[1], Cworld[2]);
    int mB = addVertex(Cworld[0], Cworld[1], Cworld[2]);

    int cA = copyVertex(c);
    int cB = c;

    setTri(t, a, mA, cA);

    int newTriIdx = static_cast<int>(triangles.size() / 3);
    triangles.push_back(mB);
    triangles.push_back(b);
    triangles.push_back(cB);

    // Origin propagation / bookkeeping
    const int origin = triOrigin[t];
    triOrigin[t] = origin;
    triOrigin.push_back(origin);

    if (static_cast<int>(faceMapping.size()) < newTriIdx + 1)
        faceMapping.resize(newTriIdx + 1);
    if (static_cast<int>(sameAsOriginal.size()) < newTriIdx + 1)
        sameAsOriginal.resize(newTriIdx + 1, 0);

    faceMapping[t] = origin;
    faceMapping[newTriIdx] = origin;

    sameAsOriginal[t] = 0;
    sameAsOriginal[newTriIdx] = 0;

    triGen[t] += 1;
    triGen.push_back(triGen[t]);

    return newTriIdx;
}

TriangleSoupSplitterClip::TriangleSoupSplitterClip(
    const float* mvp,
    const float* verts,
    const int* tris,
    int nv, int nt,
    const int* valid_tris)
    : originalNumTriangles(nt)
{
    // now 4 floats per vertex
    vertices.assign(verts, verts + nv * 3);
    triangles.assign(tris, tris + nt * 3);
    valid_triangles.assign(valid_tris, valid_tris + nt);

    for (int i = 0; i < 16; ++i) mvp4x4[i] = mvp[i];

    triGen.assign(nt, 0);
    triOrigin.resize(nt);
    for (int i = 0; i < nt; ++i) triOrigin[i] = i;

    faceMapping.resize(nt);
    for (int i = 0; i < nt; ++i) faceMapping[i] = triOrigin[i];

    sameAsOriginal.resize(nt, 1);
}

void TriangleSoupSplitterClip::splitLongEdges(int numSplits, float tau_ratio, float aspectWH) {
    const float tau = 2.0f * tau_ratio;

    if (numSplits == 0) return;

    // In unbounded mode we require tau>0 (ratio of image height)
    if (numSplits < 0 && !(tau > 0.0f)) {
        return;
    }

    if (numSplits > 0) {
        // 2 new verts per split * 4 floats
        vertices.reserve(vertices.size() + static_cast<size_t>(numSplits) * 8);
        triangles.reserve(triangles.size() + static_cast<size_t>(numSplits) * 3);
        triGen.reserve(triGen.size() + static_cast<size_t>(numSplits));
        triOrigin.reserve(triOrigin.size() + static_cast<size_t>(numSplits));
        faceMapping.reserve(faceMapping.size() + static_cast<size_t>(numSplits));
        sameAsOriginal.reserve(sameAsOriginal.size() + static_cast<size_t>(numSplits));
    }

    std::priority_queue<EdgeRef> pq;

    const int T = getNumTriangles();
    for (int t = 0; t < T; ++t) {
        if (valid_triangles[t] > 0) {
            enqueueTriangleEdges(t, pq, aspectWH);
        }
    }

    // tau is a ratio of image height; we compare squared values in same units.
    const float tau2 = (tau <= 0.0f) ? -1.0f : (tau * tau);

    int splits = 0;
    while ((numSplits < 0 || splits < numSplits) && !pq.empty()) {
        EdgeRef top = pq.top(); pq.pop();
        if (top.tri < 0 || top.tri >= getNumTriangles()) continue;
        if (top.gen != triGen[top.tri]) continue;

        // If current "longest" is below threshold, we’re done.
        if (tau2 >= 0.0f && top.len2 < tau2) break;

        int newTri = splitTriangleEdge(top.tri, top.e);
        enqueueTriangleEdges(top.tri, pq, aspectWH);
        enqueueTriangleEdges(newTri, pq, aspectWH);
        ++splits;
    }
}

int TriangleSoupSplitterClip::getNumVertices() const { return static_cast<int>(vertices.size() / 3); }
int TriangleSoupSplitterClip::getNumTriangles() const { return static_cast<int>(triangles.size() / 3); }
int TriangleSoupSplitterClip::getOriginalNumTriangles() const { return originalNumTriangles; }

void TriangleSoupSplitterClip::exportToFlatArrays(float* outVerts, int* outFaces) const {
    std::memcpy(outVerts, vertices.data(), vertices.size() * sizeof(float));
    std::memcpy(outFaces, triangles.data(), triangles.size() * sizeof(int));
}

void TriangleSoupSplitterClip::getFaceMapping(int* outMapping) const {
    const int T = getNumTriangles();
    std::memcpy(outMapping, faceMapping.data(), static_cast<size_t>(T) * sizeof(int));
}

void TriangleSoupSplitterClip::getSameAsOriginal(int* outFlags) const {
    const int T = getNumTriangles();
    for (int t = 0; t < T; ++t) outFlags[t] = sameAsOriginal[t] ? 1 : 0;
}

// ---- small inline helpers ----
inline void TriangleSoupSplitterClip::triIndices(int t, int& i0, int& i1, int& i2) const {
    const int b = 3 * t;
    i0 = triangles[b + 0]; i1 = triangles[b + 1]; i2 = triangles[b + 2];
}
inline void TriangleSoupSplitterClip::setTri(int t, int i0, int i1, int i2) {
    const int b = 3 * t;
    triangles[b + 0] = i0; triangles[b + 1] = i1; triangles[b + 2] = i2;
}
inline int TriangleSoupSplitterClip::addVertex(float x, float y, float z) {
    int idx = getNumVertices();
    vertices.push_back(x); vertices.push_back(y); vertices.push_back(z);
    return idx;
}
inline int TriangleSoupSplitterClip::copyVertex(int src) {
    const float* s = p(src);
    return addVertex(s[0], s[1], s[2]);
}

} // namespace diffsoup