#include "remesh.h"

namespace diffsoup {

void TriangleSoupSplitter::enqueueTriangleEdges(int t, std::priority_queue<EdgeRef>& pq) const {
    int i0, i1, i2;
    triIndices(t, i0, i1, i2);
    int g = triGen[t];

    pq.push(EdgeRef{t, 0, lengthSquaredBetween(i0, i1), g});
    pq.push(EdgeRef{t, 1, lengthSquaredBetween(i1, i2), g});
    pq.push(EdgeRef{t, 2, lengthSquaredBetween(i2, i0), g});
}

int TriangleSoupSplitter::splitTriangleEdge(int t, int e) {
    int i0, i1, i2;
    triIndices(t, i0, i1, i2);

    int a, b, c;
    if (e == 0) { a = i0; b = i1; c = i2; }
    else if (e == 1) { a = i1; b = i2; c = i0; }
    else { a = i2; b = i0; c = i1; }

    const float* A = p(a);
    const float* B = p(b);
    float mx = 0.5f * (A[0] + B[0]);
    float my = 0.5f * (A[1] + B[1]);
    float mz = 0.5f * (A[2] + B[2]);

    int mA = addVertex(mx, my, mz);
    int mB = addVertex(mx, my, mz);

    int cA = copyVertex(c);
    int cB = c;

    setTri(t, a, mA, cA);

    int newTriIdx = static_cast<int>(triangles.size() / 3);
    triangles.push_back(mB);
    triangles.push_back(b);
    triangles.push_back(cB);

    // Origin propagation
    const int origin = triOrigin[t];
    triOrigin[t] = origin;
    triOrigin.push_back(origin);

    // Ensure mapping arrays sized to cover newTriIdx
    if (static_cast<int>(faceMapping.size()) < newTriIdx + 1) {
        faceMapping.resize(newTriIdx + 1);
    }
    if (static_cast<int>(sameAsOriginal.size()) < newTriIdx + 1) {
        sameAsOriginal.resize(newTriIdx + 1, 0);
    }

    // Update face→origin mapping for both children
    faceMapping[t] = origin;
    faceMapping[newTriIdx] = origin;

    // Mark modification flags
    sameAsOriginal[t] = 0;
    sameAsOriginal[newTriIdx] = 0;

    // gens update — child inherits the parent's updated generation
    triGen[t] += 1;
    triGen.push_back(triGen[t]);

    return newTriIdx;
}

TriangleSoupSplitter::TriangleSoupSplitter(const float* verts, const int* tris, int nv, int nt)
    : originalNumTriangles(nt)
{
    vertices.assign(verts, verts + nv * 3);
    triangles.assign(tris, tris + nt * 3);

    triGen.assign(nt, 0);
    triOrigin.resize(nt);
    for (int i = 0; i < nt; ++i) triOrigin[i] = i;

    faceMapping.resize(nt);
    for (int i = 0; i < nt; ++i) faceMapping[i] = triOrigin[i];

    // Initially, all current triangles are identical to originals
    sameAsOriginal.resize(nt, 1);
}

void TriangleSoupSplitter::splitLongEdges(int numSplits, float tau) {
    if (numSplits == 0) return;

    // Prevent infinite loop: unbounded mode requires a positive tau
    if (numSplits < 0 && !(tau > 0.0f)) {
        return; // or assert/throw
    }

    // Only pre-reserve when we have a finite, sensible cap.
    if (numSplits > 0) {
        // 3 new vertices per split -> 9 floats
        vertices.reserve(vertices.size() + static_cast<size_t>(numSplits) * 9);
        // 1 new triangle per split -> 3 ints
        triangles.reserve(triangles.size() + static_cast<size_t>(numSplits) * 3);
        triGen.reserve(triGen.size() + static_cast<size_t>(numSplits));
        triOrigin.reserve(triOrigin.size() + static_cast<size_t>(numSplits));
        faceMapping.reserve(faceMapping.size() + static_cast<size_t>(numSplits));
        sameAsOriginal.reserve(sameAsOriginal.size() + static_cast<size_t>(numSplits));
    }
    // In unbounded mode (numSplits < 0) we skip reserves to avoid huge allocations.
    // Reserves are only a perf hint; removing them entirely is also fine.

    std::priority_queue<EdgeRef> pq;

    const int T = getNumTriangles();
    for (int t = 0; t < T; ++t) enqueueTriangleEdges(t, pq);

    // Guarded tau^2: negative means "ignore threshold"
    const float tau2 = (tau <= 0.0f) ? -1.0f : tau * tau;

    int splits = 0;
    while ((numSplits < 0 || splits < numSplits) && !pq.empty()) {
        EdgeRef top = pq.top(); pq.pop();
        if (top.tri < 0 || top.tri >= getNumTriangles()) continue;
        if (top.gen != triGen[top.tri]) continue;

        // If the current longest edge is below threshold, all others are too.
        if (tau2 >= 0.0f && top.len2 < tau2) break;

        int newTri = splitTriangleEdge(top.tri, top.e);
        enqueueTriangleEdges(top.tri, pq);
        enqueueTriangleEdges(newTri, pq);
        ++splits;
    }
}

int TriangleSoupSplitter::getNumVertices() const {
    return static_cast<int>(vertices.size() / 3);
}

int TriangleSoupSplitter::getNumTriangles() const {
    return static_cast<int>(triangles.size() / 3);
}

int TriangleSoupSplitter::getOriginalNumTriangles() const {
    return originalNumTriangles;
}

void TriangleSoupSplitter::exportToFlatArrays(float* outVerts, int* outFaces) const {
    std::memcpy(outVerts, vertices.data(), vertices.size() * sizeof(float));
    std::memcpy(outFaces, triangles.data(), triangles.size() * sizeof(int));
}

void TriangleSoupSplitter::getFaceMapping(int* outMapping) const {
    const int T = getNumTriangles();
    std::memcpy(outMapping, faceMapping.data(), static_cast<size_t>(T) * sizeof(int));
}

void TriangleSoupSplitter::getSameAsOriginal(int* outFlags) const {
    const int T = getNumTriangles();
    for (int t = 0; t < T; ++t) outFlags[t] = sameAsOriginal[t] ? 1 : 0;
}

} // namespace diffsoup