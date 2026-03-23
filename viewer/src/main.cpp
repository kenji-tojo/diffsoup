#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <stack>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "viewer.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m)
{
    m.attr("__version__") = "0.1.0";

    m.def("launch_viewer_with_mlp", [] (
        nb::ndarray<float,   nb::numpy, nb::shape<-1, 3>,     nb::c_contig> verts, // [V, 3], float32
        nb::ndarray<int,     nb::numpy, nb::shape<-1, 3>,     nb::c_contig> faces, // [F, 3], int32
        nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> face_color_lut0, // [H, W, 4], uint8
        nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> face_color_lut1, // [H, W, 4], uint8
        // ---- MLP weights (all float32) ----
        nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W1,  // [16,16]
        nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b1,  // [16]
        nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W2,  // [16,16]
        nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b2,  // [16]
        nb::ndarray<float, nb::numpy, nb::shape<3,16>,  nb::c_contig> W3,  // [3,16]
        nb::ndarray<float, nb::numpy, nb::shape<3>,     nb::c_contig> b3,  // [3]
        float enc_freq,
        const char* output_dir
    ) {
        const int V = static_cast<int>(verts.shape(0));
        const int F = static_cast<int>(faces.shape(0));

        // Note: your C++ variable names previously called these W,H, but the arrays are [H,W,4].
        const int H = static_cast<int>(face_color_lut0.shape(0));
        const int W = static_cast<int>(face_color_lut0.shape(1));

        // Basic sanity (dtype constraints are handled by nanobind template args)
        if (face_color_lut1.shape(0) != H || face_color_lut1.shape(1) != W) {
            throw std::runtime_error("LUT0 and LUT1 sizes must match");
        }

        float cen[3]{0};
        for (int i = 0; i < V; ++i) {
            float w = 1.0f / static_cast<float>(V);
            cen[0] += verts(i, 0) * w;
            cen[1] += verts(i, 1) * w;
            cen[2] += verts(i, 2) * w;
        }

        viewer::Viewer app;
        app.set_triangles(V, verts.data(), F, faces.data());
        app.set_triangle_color_lut(0, W, H, face_color_lut0.data());
        app.set_triangle_color_lut(1, W, H, face_color_lut1.data());
        app.set_camera_center(cen);
        app.set_mlp_weights(
            W1.data(), b1.data(),
            W2.data(), b2.data(),
            W3.data(), b3.data(),
            enc_freq
        );
        app.set_output_dir(output_dir);
        app.launch(1200, 1200);
    });
    
    m.def("benchmark_viewer_with_mlp",
        [] (
            // geometry
            nb::ndarray<float,   nb::numpy, nb::shape<-1, 3>,     nb::c_contig> verts, // [V,3]
            nb::ndarray<int,     nb::numpy, nb::shape<-1, 3>,     nb::c_contig> faces, // [F,3]

            // per-triangle LUTs
            nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> face_color_lut0, // [H,W,4]
            nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> face_color_lut1, // [H,W,4]

            // MLP weights
            nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W1,
            nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b1,
            nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W2,
            nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b2,
            nb::ndarray<float, nb::numpy, nb::shape<3,16>,  nb::c_contig> W3,
            nb::ndarray<float, nb::numpy, nb::shape<3>,     nb::c_contig> b3,
            float enc_freq,

            // benchmark-specific
            nb::ndarray<float, nb::numpy, nb::shape<-1, 4, 4>, nb::c_contig> mvps, // [B,4,4]
            int width,
            int height,
            int warmup_frames,
            int save_every,
            const char* output_dir
        )
        {
            const int V = static_cast<int>(verts.shape(0));
            const int F = static_cast<int>(faces.shape(0));
            const int H = static_cast<int>(face_color_lut0.shape(0));
            const int W = static_cast<int>(face_color_lut0.shape(1));
            const int B = static_cast<int>(mvps.shape(0));

            if (face_color_lut1.shape(0) != H || face_color_lut1.shape(1) != W) {
                throw std::runtime_error("LUT0 and LUT1 sizes must match");
            }

            // camera center = vertex mean (same as interactive path)
            float cen[3]{0.f, 0.f, 0.f};
            for (int i = 0; i < V; ++i) {
                float wv = 1.0f / float(V);
                cen[0] += verts(i, 0) * wv;
                cen[1] += verts(i, 1) * wv;
                cen[2] += verts(i, 2) * wv;
            }

            viewer::Viewer app;
            app.set_output_dir(output_dir);
            app.set_triangles(V, verts.data(), F, faces.data());
            app.set_triangle_color_lut(0, W, H, face_color_lut0.data());
            app.set_triangle_color_lut(1, W, H, face_color_lut1.data());
            app.set_camera_center(cen);
            app.set_mlp_weights(
                W1.data(), b1.data(),
                W2.data(), b2.data(),
                W3.data(), b3.data(),
                enc_freq
            );

            // mvps is contiguous [B,4,4] float32 → pass pointer
            app.launch_benchmark(
                width,
                height,
                B,
                mvps.data(),
                warmup_frames,
                save_every
            );
        });
}