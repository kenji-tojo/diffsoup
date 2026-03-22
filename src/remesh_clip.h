#pragma once

#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <utility>

namespace diffsoup {

class TriangleSoupSplitterClip {
private:
    // xyz per vertex
    std::vector<float> vertices;   // flat xyz
    std::vector<int>   triangles;  // flat i0,i1,i2
    std::vector<int>   valid_triangles;
    int originalNumTriangles;

    float mvp4x4[16];

    std::vector<int>            faceMapping;
    std::vector<unsigned char>  sameAsOriginal;
    std::vector<int>            triGen;
    std::vector<int>            triOrigin;

    struct EdgeRef {
        int tri, e;
        float len2;
        int gen;
        bool operator<(const EdgeRef& o) const { return len2 < o.len2; }
    };

    inline const float* p(int vi) const { return &vertices[3 * vi]; }

    // New: screen-space length^2 with aspect ratio and clip rules.
    inline float screenLen2Between(int a, int b, float aspectWH) const;

    inline void triIndices(int t, int& i0, int& i1, int& i2) const;
    inline void setTri(int t, int i0, int i1, int i2);
    inline int  addVertex(float x, float y, float z);
    inline int  copyVertex(int src);

    void enqueueTriangleEdges(int t, std::priority_queue<EdgeRef>& pq, float aspectWH) const;
    int  splitTriangleEdge(int t, int e);

public:
    TriangleSoupSplitterClip(
        const float* mvp,
        const float* verts,
        const int* tris,
        int nv, int nt,
        const int* valid_tris);

    // Keep convenience API; add aspect
    void splitLongEdges(int numSplits) { splitLongEdges(numSplits, 0.0f, 1.0f); }
    void splitLongEdges(int numSplits, float tau_ratio /* ratio vs image height */, float aspectWH);

    // Split until all edges <= tau (ratio vs image height)
    void splitLongEdgesUntil(float tau_ratio, int hardCap = -1, float aspectWH = 1.0f) {
        splitLongEdges(hardCap, tau_ratio, aspectWH);
    }

    int  getNumVertices() const;
    int  getNumTriangles() const;
    int  getOriginalNumTriangles() const;

    void exportToFlatArrays(float* outVerts, int* outFaces) const;
    void getFaceMapping(int* outMapping) const;
    void getSameAsOriginal(int* outFlags) const;
};

} // namespace diffsoup