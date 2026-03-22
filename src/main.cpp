// src/main.cpp

#include <cstdio>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <stack>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cuda/rasterize.cuh"
#include "cuda/multires.cuh"
#include "cuda/encoding.cuh"
#include "remesh.h"
#include "remesh_clip.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace ds = diffsoup;

NB_MODULE(_core, m)
{
    m.attr("__version__") = "0.1.0";

    m.def("compute_triangle_rects", [](
        int H, int W,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,               // [B, V, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,     nb::c_contig> tri,               // [T, 3]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 4>,     nb::c_contig> triangle_rects,    // [B * T, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1>,        nb::c_contig> frag_prefix_sum    // [B * T]
    ) -> int {
        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int T = static_cast<int>(tri.shape(0));

        return ds::cuda::compute_triangle_rects(
            H, W, B,
            V, pos.data(),
            T, tri.data(),
            triangle_rects.data(),
            frag_prefix_sum.data()
        );
    }, "Compute screen-space bounding rectangles and fragment prefix sums for each triangle");

    m.def("compute_fragments", [](
        int H, int W,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>,     nb::c_contig> pos,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,         nb::c_contig> tri,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1>,            nb::c_contig> frag_prefix_sum,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 4>,         nb::c_contig> triangle_rects,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,         nb::c_contig> frag_pix,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>,         nb::c_contig> frag_attrs
    ) {
        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int T = static_cast<int>(tri.shape(0));
        const int num_tris = B * T;
        const int num_frags = static_cast<int>(frag_pix.shape(0));

        ds::cuda::compute_fragments(
            H, W,
            V, pos.data(),
            T, tri.data(),
            num_tris,
            num_frags,
            frag_prefix_sum.data(),
            triangle_rects.data(),
            frag_pix.data(),
            frag_attrs.data()
        );
    }, "Compute rasterization fragments with barycentric coordinates and triangle ID");

    m.def("depth_test", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 3>,  nb::c_contig> frag_pix,          // [num_frags, 3]
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, 4>,  nb::c_contig> frag_attrs,        // [num_frags, 4]
        nb::ndarray<float,   nb::pytorch, nb::shape<-1>,     nb::c_contig> frag_alpha,        // [num_frags]
        nb::ndarray<float,   nb::pytorch, nb::shape<-1>,     nb::c_contig> alpha_thresh,      // [num_frags]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast_out     // [B, H, W, 4]
    ) {
        const int num_frags = static_cast<int>(frag_pix.shape(0));
        const int B = static_cast<int>(rast_out.shape(0));
        const int H = static_cast<int>(rast_out.shape(1));
        const int W = static_cast<int>(rast_out.shape(2));

        ds::cuda::depth_test(
            B, H, W, num_frags, frag_pix.data(), frag_attrs.data(),
            frag_alpha.data(), alpha_thresh.data(), rast_out.data()
        );
    }, "Rasterize visible fragments via z-buffer depth test");

    m.def("filter_valid_fragments", [](
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix_out,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs_out
    ) -> int {
        const int num_frags = static_cast<int>(frag_pix.shape(0));

        const int num_valid_frags = ds::cuda::filter_valid_fragments(
            num_frags, frag_pix.data(), frag_attrs.data(),
            frag_pix_out.data(), frag_attrs_out.data()
        );

        return num_valid_frags;
    }, "Filter valid fragments where frag_pix[:, 0] >= 0 using CUDA");

    m.def("backward_edge_grad", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> color,       // [B, H, W, C]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> grad_color,  // [B, H, W, C]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,        // [B, H, W, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,              // [B, V, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> grad_pos,         // [B, V, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> tri                   // [T, 3]
    ) {
        const int B = static_cast<int>(color.shape(0));
        const int H = static_cast<int>(color.shape(1));
        const int W = static_cast<int>(color.shape(2));
        const int C = static_cast<int>(color.shape(3));
        const int V = static_cast<int>(pos.shape(1));

        return ds::cuda::backward_edge_grad(
            B, H, W, C, color.data(), grad_color.data(), rast.data(),
            V, pos.data(), grad_pos.data(), tri.data()
        );
    }, "TODO: description");

    m.def("encode_view_dir_sh2", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast,    // [B, H, W, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4, 4>,      nb::c_contig> inv_mvp, // [B, 4, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 9>, nb::c_contig> encoding // [B, H, W, 9]
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));

        ds::cuda::encode_view_dir_sh2(
            B, H, W, rast.data(), inv_mvp.data(), encoding.data()
        );
    }, "TODO: description");

    m.def("encode_view_dir_freq", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast,     // [B, H, W, 4]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4, 4>,      nb::c_contig> inv_mvp,  // [B, 4, 4]
        float freq,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 9>, nb::c_contig> encoding, // [B, H, W, 9]
        float vmf_kappa,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 2>, nb::c_contig> vmf_samples // [B, H, W, 2]
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));

        ds::cuda::encode_view_dir_freq(
            B, H, W, rast.data(), inv_mvp.data(),
            freq, encoding.data(),
            vmf_kappa, vmf_samples.size() > 0 ? vmf_samples.data() : nullptr
        );
    }, "TODO: description");

    m.def("backward_radiance_field_loss", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> color,            // [B, H, W, C]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> target,           // [B, H, W, C]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,             // [B, H, W, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,          nb::c_contig> frag_pix,         // [num_frags, 3]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>,          nb::c_contig> frag_attrs,       // [num_frags, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1>,             nb::c_contig> frag_alpha,       // [num_frags]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1>,             nb::c_contig> grad_frag_alpha   // [num_frags]
    ) {
        const int B = static_cast<int>(color.shape(0));
        const int H = static_cast<int>(color.shape(1));
        const int W = static_cast<int>(color.shape(2));
        const int C = static_cast<int>(color.shape(3));
        const int num_frags = static_cast<int>(frag_pix.shape(0));

        return ds::cuda::backward_radiance_field_loss(
            B, H, W, C, color.data(), target.data(), rast.data(),
            num_frags, frag_pix.data(), frag_attrs.data(),
            frag_alpha.data(), grad_frag_alpha.data()
        );
    }, "Evaluate the gradient of radiance field loss.");

    m.def("multires_triangle_alpha", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>,  nb::c_contig> frag_attrs,   // [num_frags, 4]
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig> alpha_src,     // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
        nb::ndarray<float, nb::pytorch, nb::shape<-1>,     nb::c_contig> frag_alpha     // [num_frags]
    ) {
        const int num_frags = static_cast<int>(frag_attrs.shape(0));

        const uint32_t S = ds::total_feats_from_levels(min_level, max_level);
        if (alpha_src.shape(1) != S) {
            throw std::runtime_error("Invalid feature size.");
        }

        ds::cuda::multires_triangle_alpha(
            num_frags, frag_attrs.data(), min_level, max_level,
            alpha_src.data(), frag_alpha.data()
        );
    }, "TODO: description");

    m.def("backward_multires_triangle_alpha", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>,  nb::c_contig> frag_attrs,        // [num_frags, 4]
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig> grad_alpha_src,    // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
        nb::ndarray<float, nb::pytorch, nb::shape<-1>,     nb::c_contig> grad_frag_alpha    // [num_frags]
    ) {
        const int num_frags = static_cast<int>(frag_attrs.shape(0));

        const uint32_t S = ds::total_feats_from_levels(min_level, max_level);
        if (grad_alpha_src.shape(1) != S) {
            throw std::runtime_error("Invalid feature size.");
        }

        ds::cuda::backward_multires_triangle_alpha(
            num_frags, frag_attrs.data(), min_level, max_level,
            grad_alpha_src.data(), grad_frag_alpha.data()
        );
    }, "TODO: description");

    m.def("multires_triangle_color", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,        // [B, H, W, 4]
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>,     nb::c_contig> features,    // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> out          // [B, H, W, feature_dim]
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));
        const int feature_dim = static_cast<int>(features.shape(2));

        const uint32_t S = ds::total_feats_from_levels(min_level, max_level);
        if (features.shape(1) != S) {
            throw std::runtime_error("Invalid feature size.");
        }

        ds::cuda::multires_triangle_color(
            B, H, W, rast.data(), min_level, max_level, feature_dim,
            features.data(), out.data()
        );
    }, "TODO: description");

    m.def("backward_multires_triangle_color", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,             // [B, H, W, 4]
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>,     nb::c_contig> grad_features,    // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> grad_out          // [B, H, W, feature_dim]
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));
        const int feature_dim = static_cast<int>(grad_features.shape(2));

        const uint32_t S = ds::total_feats_from_levels(min_level, max_level);
        if (grad_features.shape(1) != S) {
            throw std::runtime_error("Invalid feature size.");
        }

        ds::cuda::backward_multires_triangle_color(
            B, H, W, rast.data(), min_level, max_level, feature_dim,
            grad_features.data(), grad_out.data()
        );
    }, "TODO: description");

    m.def("accumulate_to_level_forward", [](
        int min_level, int max_level, int target_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> feat_all,  // [T, Σ S_l, C]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> feat_max   // [T, S_L, C]
    ) {
        const int T = static_cast<int>(feat_all.shape(0));
        const int feat_dim = static_cast<int>(feat_all.shape(2));

        const uint32_t S = ds::total_feats_from_levels(min_level, max_level);
        if (feat_all.shape(1) != S) {
            throw std::runtime_error("Invalid feature size.");
        }

        const uint32_t S_L = ds::feats_at_level(target_level);
        if (feat_max.shape(1) != S_L) {
            throw std::runtime_error("Invalid feature size.");
        }

        ds::cuda::accumulate_to_level_forward(
            T, min_level, max_level, target_level, feat_dim,
            feat_all.data(), feat_max.data()
        );
    }, "TODO: description");

    m.def("accumulate_to_level_backward", [](
        int min_level, int max_level, int target_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> grad_feat_all,  // [T, Σ S_l, C]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> grad_feat_max   // [T, S_L, C]
    ) {
        const int T = static_cast<int>(grad_feat_all.shape(0));
        const int feat_dim = static_cast<int>(grad_feat_all.shape(2));

        const uint32_t S = ds::total_feats_from_levels(min_level, max_level);
        if (grad_feat_all.shape(1) != S) {
            throw std::runtime_error("Invalid feature size.");
        }

        const uint32_t S_L = ds::feats_at_level(target_level);
        if (grad_feat_max.shape(1) != S_L) {
            throw std::runtime_error("Invalid feature size.");
        }

        ds::cuda::accumulate_to_level_backward(
            T, min_level, max_level, target_level, feat_dim,
            grad_feat_all.data(), grad_feat_max.data()
        );
    }, "TODO: description");

    m.def("split_triangle_soup", [](
        nb::ndarray<float, nb::numpy, nb::shape<-1, 3>, nb::c_contig> verts,
        nb::ndarray<int,   nb::numpy, nb::shape<-1, 3>, nb::c_contig> faces,
        int numSplits,
        float tau
    ) -> nb::tuple {
        const int N = (int) verts.shape(0);
        const int M = (int) faces.shape(0);

        diffsoup::TriangleSoupSplitter splitter(verts.data(), faces.data(), N, M);
        splitter.splitLongEdges(numSplits, tau);

        const int newN = splitter.getNumVertices();
        const int newM = splitter.getNumTriangles();

        // ---- allocate (owned by Python through capsule) ----

        float *verts_ptr = new float[newN * 3];
        int   *faces_ptr = new int[newM * 3];
        int   *map_ptr   = new int[newM];
        int   *flag_ptr  = new int[newM];

        splitter.exportToFlatArrays(verts_ptr, faces_ptr);
        splitter.getFaceMapping(map_ptr);
        splitter.getSameAsOriginal(flag_ptr);

        nb::capsule verts_owner(verts_ptr, [](void *p) noexcept { delete[] (float*) p; });
        nb::capsule faces_owner(faces_ptr, [](void *p) noexcept { delete[] (int*) p; });
        nb::capsule map_owner(map_ptr, [](void *p) noexcept { delete[] (int*) p; });
        nb::capsule flag_owner(flag_ptr, [](void *p) noexcept { delete[] (int*) p; });

        nb::ndarray<float, nb::numpy, nb::shape<-1,3>, nb::c_contig> outVerts(verts_ptr, { (size_t)newN, (size_t)3 }, verts_owner);
        nb::ndarray<int, nb::numpy, nb::shape<-1,3>, nb::c_contig> outFaces(faces_ptr, { (size_t)newM, (size_t)3 }, faces_owner);
        nb::ndarray<int, nb::numpy, nb::shape<-1>, nb::c_contig> faceMapping(map_ptr, { (size_t)newM }, map_owner);
        nb::ndarray<int, nb::numpy, nb::shape<-1>, nb::c_contig> faceFlags(flag_ptr, { (size_t)newM }, flag_owner);

        return nb::make_tuple(outVerts, outFaces, faceMapping, faceFlags);
    }, nb::rv_policy::take_ownership,
       "Split triangle soup by longest edges");

    m.def("split_triangle_soup_clip", [](
        nb::ndarray<float, nb::numpy, nb::shape<4, 4>,  nb::c_contig> mvp,
        nb::ndarray<float, nb::numpy, nb::shape<-1, 3>, nb::c_contig> verts,
        nb::ndarray<int,   nb::numpy, nb::shape<-1, 3>, nb::c_contig> faces,
        nb::ndarray<int,   nb::numpy, nb::shape<-1>,    nb::c_contig> valid_faces,
        int numSplits, float tau_ratio, float aspectWH
    ) -> nb::tuple {
        const int N = (int) verts.shape(0);
        const int M = (int) faces.shape(0);

        diffsoup::TriangleSoupSplitterClip splitter(
            mvp.data(), verts.data(), faces.data(), N, M,
            valid_faces.data()
            );

        splitter.splitLongEdges(numSplits, tau_ratio, aspectWH);

        const int newN = splitter.getNumVertices();
        const int newM = splitter.getNumTriangles();

        // ---- allocate (owned by Python through capsule) ----
        float *verts_ptr = new float[newN * 3];
        int   *faces_ptr = new int[newM * 3];
        int   *map_ptr   = new int[newM];
        int   *flag_ptr  = new int[newM];

        splitter.exportToFlatArrays(verts_ptr, faces_ptr);
        splitter.getFaceMapping(map_ptr);
        splitter.getSameAsOriginal(flag_ptr);

        nb::capsule verts_owner(verts_ptr, [](void *p) noexcept { delete[] (float*) p; });
        nb::capsule faces_owner(faces_ptr, [](void *p) noexcept { delete[] (int*) p; });
        nb::capsule map_owner  (map_ptr,   [](void *p) noexcept { delete[] (int*) p; });
        nb::capsule flag_owner (flag_ptr,  [](void *p) noexcept { delete[] (int*) p; });

        nb::ndarray<float, nb::numpy, nb::shape<-1, 3>, nb::c_contig> outVerts(verts_ptr, { (size_t)newN, (size_t)3 }, verts_owner);
        nb::ndarray<int, nb::numpy, nb::shape<-1, 3>, nb::c_contig> outFaces(faces_ptr, { (size_t)newM, (size_t)3 }, faces_owner);
        nb::ndarray<int, nb::numpy, nb::shape<-1>, nb::c_contig> faceMapping(map_ptr, { (size_t)newM }, map_owner);
        nb::ndarray<int, nb::numpy, nb::shape<-1>, nb::c_contig> faceFlags(flag_ptr, { (size_t)newM }, flag_owner);

        return nb::make_tuple(outVerts, outFaces, faceMapping, faceFlags);
    }, nb::rv_policy::take_ownership,
       "Split triangle soup by longest edges measured in screen space (NDC); "
       "verts are clip-space xyzw, tau is a ratio of image height, dx scaled by W/H.");

    m.def("frag_alpha_mobilenerf", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs, // [num_frags, 4]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> uvs,
        nb::ndarray<int,   nb::pytorch, nb::shape<-1, 3>, nb::c_contig> tri,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> feat0,
        nb::ndarray<float, nb::pytorch, nb::shape<-1>,    nb::c_contig> frag_alpha  // [num_frags,]
    ) {
        const int num_frags = static_cast<int>(frag_attrs.shape(0));
        const int texH = static_cast<int>(feat0.shape(0));
        const int texW = static_cast<int>(feat0.shape(1));

        ds::cuda::frag_alpha_mobilenerf(
            num_frags, frag_attrs.data(), uvs.data(), tri.data(),
            texH, texW, feat0.data(), frag_alpha.data()
        );
    }, "TODO: description");

    m.def("lookup_feats_mobilenerf", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast, // [B, H, W, 4]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> uvs,
        nb::ndarray<int,   nb::pytorch, nb::shape<-1, 3>, nb::c_contig> tri,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> feat0,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> feat1,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 8>, nb::c_contig> image // [B, H, W, 4]
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));
        const int texH = static_cast<int>(feat0.shape(0));
        const int texW = static_cast<int>(feat0.shape(1));

        ds::cuda::lookup_feats_mobilenerf(
            B, H, W, rast.data(), uvs.data(), tri.data(),
            texH, texW, feat0.data(), feat1.data(), image.data()
        );
    }, "TODO: description");
}
