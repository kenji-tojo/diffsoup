// src/main.cpp
// Nanobind module definition for the DiffSoup CUDA extension (_core).

#include <cstdio>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>

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

    // ── Rasterisation ───────────────────────────────────────────────

    m.def("compute_triangle_rects", [](
        int H, int W,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,     nb::c_contig> tri,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 4>,     nb::c_contig> triangle_rects,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1>,        nb::c_contig> frag_prefix_sum
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
    }, "Compute screen-space bounding rectangles and fragment prefix sums for each triangle.");

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
    }, "Rasterise triangles into per-pixel fragments with barycentric coordinates.");

    m.def("depth_test", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 3>,  nb::c_contig> frag_pix,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, 4>,  nb::c_contig> frag_attrs,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1>,     nb::c_contig> frag_alpha,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1>,     nb::c_contig> alpha_thresh,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast_out
    ) {
        const int num_frags = static_cast<int>(frag_pix.shape(0));
        const int B = static_cast<int>(rast_out.shape(0));
        const int H = static_cast<int>(rast_out.shape(1));
        const int W = static_cast<int>(rast_out.shape(2));

        ds::cuda::depth_test(
            B, H, W, num_frags, frag_pix.data(), frag_attrs.data(),
            frag_alpha.data(), alpha_thresh.data(), rast_out.data()
        );
    }, "Resolve fragment visibility via z-buffer depth test.");

    m.def("filter_valid_fragments", [](
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix_out,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs_out
    ) -> int {
        const int num_frags = static_cast<int>(frag_pix.shape(0));

        return ds::cuda::filter_valid_fragments(
            num_frags, frag_pix.data(), frag_attrs.data(),
            frag_pix_out.data(), frag_attrs_out.data()
        );
    }, "Compact fragment buffers by removing invalid (off-screen) entries.");

    // ── Edge gradients ──────────────────────────────────────────────

    m.def("backward_edge_grad", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> color,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> grad_color,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> grad_pos,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> tri
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
    }, "Backward pass for silhouette / edge gradients into vertex positions.");

    // ── Stochastic opacity masking (auxiliary loss) ─────────────────

    m.def("backward_opacity_aux_loss", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> color,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> target,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,          nb::c_contig> frag_pix,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>,          nb::c_contig> frag_attrs,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1>,             nb::c_contig> frag_alpha,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1>,             nb::c_contig> grad_frag_alpha
    ) {
        const int B = static_cast<int>(color.shape(0));
        const int H = static_cast<int>(color.shape(1));
        const int W = static_cast<int>(color.shape(2));
        const int C = static_cast<int>(color.shape(3));
        const int num_frags = static_cast<int>(frag_pix.shape(0));

        return ds::cuda::backward_opacity_aux_loss(
            B, H, W, C, color.data(), target.data(), rast.data(),
            num_frags, frag_pix.data(), frag_attrs.data(),
            frag_alpha.data(), grad_frag_alpha.data()
        );
    }, "Analytic gradient of the stochastic opacity masking auxiliary objective.");

    // ── View-direction encoding ─────────────────────────────────────

    m.def("encode_view_dir_sh2", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4, 4>,      nb::c_contig> inv_mvp,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 9>, nb::c_contig> encoding
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));

        ds::cuda::encode_view_dir_sh2(
            B, H, W, rast.data(), inv_mvp.data(), encoding.data()
        );
    }, "Evaluate order-2 spherical-harmonic basis on per-pixel view directions.");

    m.def("encode_view_dir_freq", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4, 4>,      nb::c_contig> inv_mvp,
        float freq,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 9>, nb::c_contig> encoding,
        float vmf_kappa,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 2>, nb::c_contig> vmf_samples
    ) {
        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));

        ds::cuda::encode_view_dir_freq(
            B, H, W, rast.data(), inv_mvp.data(),
            freq, encoding.data(),
            vmf_kappa, vmf_samples.size() > 0 ? vmf_samples.data() : nullptr
        );
    }, "Sinusoidal (frequency) encoding of per-pixel view directions.");

    // ── Multi-resolution triangle features ──────────────────────────

    m.def("multires_triangle_alpha", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>,  nb::c_contig> frag_attrs,
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig> alpha_src,
        nb::ndarray<float, nb::pytorch, nb::shape<-1>,     nb::c_contig> frag_alpha
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
    }, "Interpolate per-fragment opacity from multi-resolution triangle features.");

    m.def("backward_multires_triangle_alpha", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>,  nb::c_contig> frag_attrs,
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig> grad_alpha_src,
        nb::ndarray<float, nb::pytorch, nb::shape<-1>,     nb::c_contig> grad_frag_alpha
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
    }, "Backward pass for multires_triangle_alpha.");

    m.def("multires_triangle_color", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>,     nb::c_contig> features,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> out
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
    }, "Interpolate per-pixel colour from multi-resolution triangle features.");

    m.def("backward_multires_triangle_color", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,
        int min_level, int max_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>,     nb::c_contig> grad_features,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> grad_out
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
    }, "Backward pass for multires_triangle_color.");

    // ── Cross-level accumulation ────────────────────────────────────

    m.def("accumulate_to_level_forward", [](
        int min_level, int max_level, int target_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> feat_all,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> feat_max
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
    }, "Accumulate multi-resolution features down to a single target level (forward).");

    m.def("accumulate_to_level_backward", [](
        int min_level, int max_level, int target_level,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> grad_feat_all,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> grad_feat_max
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
    }, "Backward pass for accumulate_to_level.");

    // ── Remeshing ───────────────────────────────────────────────────

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
       "Split a triangle soup by repeatedly bisecting the longest edges in world space.");

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
       "Split a triangle soup by longest edges measured in screen space (NDC); "
       "tau_ratio is in image-height units, x-axis scaled by W/H.");
}
