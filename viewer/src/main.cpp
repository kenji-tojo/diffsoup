// main.cpp — nanobind Python bindings for the viewer.

#include <cstdio>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "viewer.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.attr("__version__") = "0.1.0";

    m.def("launch_viewer_with_mlp", [](
        nb::ndarray<float,   nb::numpy, nb::shape<-1, 3>,     nb::c_contig> verts,
        nb::ndarray<int,     nb::numpy, nb::shape<-1, 3>,     nb::c_contig> faces,
        nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> lut0,
        nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> lut1,
        nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W1,
        nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b1,
        nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W2,
        nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b2,
        nb::ndarray<float, nb::numpy, nb::shape<3,16>,  nb::c_contig> W3,
        nb::ndarray<float, nb::numpy, nb::shape<3>,     nb::c_contig> b3,
        const char* output_dir,
        const char* up_axis_str
    ) {
        const int V = int(verts.shape(0));
        const int F = int(faces.shape(0));
        const int H = int(lut0.shape(0));
        const int W = int(lut0.shape(1));

        if (int(lut1.shape(0)) != H || int(lut1.shape(1)) != W)
            throw std::runtime_error("LUT0 and LUT1 sizes must match");

        // Parse up-axis string.
        viewer::UpAxis up = viewer::UpAxis::NEG_Z;
        if (!viewer::parse_up_axis(up_axis_str, up))
            throw std::runtime_error(
                std::string("Unknown up_axis: '") + up_axis_str +
                "'. Expected one of: pos_y, neg_y, pos_z, neg_z, +y, -y, +z, -z");

        // Compute mesh centroid for the camera target.
        float cen[3]{};
        const float inv_V = 1.f / float(V);
        for (int i = 0; i < V; ++i) {
            cen[0] += verts(i, 0) * inv_V;
            cen[1] += verts(i, 1) * inv_V;
            cen[2] += verts(i, 2) * inv_V;
        }

        viewer::Viewer app;
        app.set_mesh(V, verts.data(), F, faces.data());
        app.set_triangle_color_lut(0, W, H, lut0.data());
        app.set_triangle_color_lut(1, W, H, lut1.data());
        app.set_up_axis(up);
        app.set_camera_target(cen);
        app.set_mlp_weights(W1.data(), b1.data(),
                            W2.data(), b2.data(),
                            W3.data(), b3.data());
        app.set_output_dir(output_dir);
        app.launch(1200, 1200);
    });

    m.def("benchmark_viewer_with_mlp", [](
        nb::ndarray<float,   nb::numpy, nb::shape<-1, 3>,     nb::c_contig> verts,
        nb::ndarray<int,     nb::numpy, nb::shape<-1, 3>,     nb::c_contig> faces,
        nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> lut0,
        nb::ndarray<uint8_t, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig> lut1,
        nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W1,
        nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b1,
        nb::ndarray<float, nb::numpy, nb::shape<16,16>, nb::c_contig> W2,
        nb::ndarray<float, nb::numpy, nb::shape<16>,    nb::c_contig> b2,
        nb::ndarray<float, nb::numpy, nb::shape<3,16>,  nb::c_contig> W3,
        nb::ndarray<float, nb::numpy, nb::shape<3>,     nb::c_contig> b3,
        nb::ndarray<float, nb::numpy, nb::shape<-1, 4, 4>, nb::c_contig> mvps,
        int width, int height, int warmup_frames, int save_every,
        const char* output_dir,
        const char* up_axis_str
    ) {
        const int V = int(verts.shape(0));
        const int F = int(faces.shape(0));
        const int H = int(lut0.shape(0));
        const int W = int(lut0.shape(1));
        const int B = int(mvps.shape(0));

        if (int(lut1.shape(0)) != H || int(lut1.shape(1)) != W)
            throw std::runtime_error("LUT0 and LUT1 sizes must match");

        viewer::UpAxis up = viewer::UpAxis::NEG_Z;
        if (!viewer::parse_up_axis(up_axis_str, up))
            throw std::runtime_error(
                std::string("Unknown up_axis: '") + up_axis_str +
                "'. Expected one of: pos_y, neg_y, pos_z, neg_z, +y, -y, +z, -z");

        float cen[3]{};
        const float inv_V = 1.f / float(V);
        for (int i = 0; i < V; ++i) {
            cen[0] += verts(i, 0) * inv_V;
            cen[1] += verts(i, 1) * inv_V;
            cen[2] += verts(i, 2) * inv_V;
        }

        viewer::Viewer app;
        app.set_output_dir(output_dir);
        app.set_mesh(V, verts.data(), F, faces.data());
        app.set_triangle_color_lut(0, W, H, lut0.data());
        app.set_triangle_color_lut(1, W, H, lut1.data());
        app.set_up_axis(up);
        app.set_camera_target(cen);
        app.set_mlp_weights(W1.data(), b1.data(),
                            W2.data(), b2.data(),
                            W3.data(), b3.data());
        app.launch_benchmark(width, height, B, mvps.data(),
                             warmup_frames, save_every);
    });
}
