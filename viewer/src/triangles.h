#pragma once

#include <cstdint>

namespace viewer {

class Triangles {
public:
    Triangles(int V, const float* verts,
              int F, const int* faces)
        : V_(V), verts_(verts), F_(F), faces_(faces) {}

    int V_;
    const float* verts_;
    int F_;
    const int* faces_;
};

} // namespace viewer