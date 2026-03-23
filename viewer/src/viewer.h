// viewer.h — OpenGL mesh viewer with per-triangle LUT + MLP compositing.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "camera.h"
#include "options.h"

namespace viewer {

class Viewer {
public:
    Viewer();
    ~Viewer();

    /// Interactive viewer (opens a window, blocks until closed).
    void launch(int width, int height);

    /// Headless benchmark: renders each of `B` supplied MVPs, logs timings.
    void launch_benchmark(int width, int height,
                          int B, const float* mvps,
                          int warmup_frames, int save_every);

    // ---- Data upload (call before launch / launch_benchmark) ---------------

    void set_output_dir(const std::string& dir);

    /// Upload mesh geometry.  `verts` is V×3 float, `faces` is F×3 int32.
    void set_mesh(int V, const float* verts, int F, const int* faces);

    /// Upload per-triangle RGBA LUTs (width × height, RGBA8).
    /// `idx` 0 → buffer A, `idx` 1 → buffer B.
    void set_triangle_color_lut(int idx, int width, int height,
                                const unsigned char* rgba);

    /// Upload trained MLP weights (all row-major float32):
    ///   W1 [16×16], b1 [16], W2 [16×16], b2 [16], W3 [3×16], b3 [3].
    void set_mlp_weights(const float* W1, const float* b1,
                         const float* W2, const float* b2,
                         const float* W3, const float* b3);

    /// Set the orbit-camera target to the mesh centroid (or any point).
    void set_camera_target(const float target[3]);

    // ---- GLFW callbacks (forwarded from free functions) ---------------------

    void on_mouse_button(GLFWwindow* win, int button, int action, int mods);
    void on_cursor_pos  (GLFWwindow* win, double x, double y);
    void on_scroll      (GLFWwindow* win, double xoff, double yoff);
    void on_framebuffer_size(GLFWwindow* win, int w, int h);

private:
    // ---- Initialization helpers -------------------------------------------
    void init_gl();
    void resize_fbo(int width, int height);
    void upload_mesh_to_gpu();
    void upload_pending_luts();
    void upload_pending_mlp();

    // ---- Rendering --------------------------------------------------------
    void render(const glm::mat4& mvp);
    void draw_gui();

    // ---- State ------------------------------------------------------------
    std::string            m_output_dir;
    std::unique_ptr<OrbitCamera> m_camera;
    Options                m_options;

    GLFWwindow* m_window = nullptr;
    bool        m_gl_ready = false;

    // G-buffer FBO (2 color attachments + depth)
    GLuint m_fbo            = 0;
    GLuint m_tex_color[2]   = {};  // A, B
    GLuint m_tex_depth      = 0;

    // Geometry pass
    GLuint m_geom_vao       = 0;
    GLuint m_geom_vbo_pos   = 0;
    GLuint m_geom_vbo_tid   = 0;
    GLuint m_geom_prog      = 0;
    GLint  m_loc_mvp        = -1;
    GLint  m_loc_tri_tex0   = -1;
    GLint  m_loc_tri_tex1   = -1;
    GLint  m_loc_tri_tex_sz = -1;
    GLint  m_loc_level      = -1;

    // Composite (MLP) pass
    GLuint m_post_prog      = 0;
    GLuint m_post_vao       = 0;
    GLint  m_post_loc_texA  = -1;
    GLint  m_post_loc_texB  = -1;
    GLint  m_post_loc_inv_mvp = -1;

    // UBOs for MLP weight blocks (std140)
    GLuint m_ubo_W1 = 0, m_ubo_B1 = 0;
    GLuint m_ubo_W2 = 0, m_ubo_B2 = 0;
    GLuint m_ubo_W3 = 0, m_ubo_B3 = 0;

    // Host-side mesh staging
    std::vector<float>    m_pos_host;
    std::vector<uint32_t> m_tid_host;
    GLsizei m_vertex_count   = 0;
    int     m_face_count     = 0;
    bool    m_mesh_dirty     = false;

    // Per-triangle LUT textures
    GLuint m_lut_tex[2]      = {};
    int    m_lut_w = 0, m_lut_h = 0;
    std::vector<unsigned char> m_lut_staging[2];
    bool   m_lut_pending[2]  = {};
    bool   m_lut_ready[2]    = {};
    int    m_level           = 5;

    // MLP weight staging
    std::vector<float> m_W1, m_b1, m_W2, m_b2, m_W3, m_b3;
    bool m_mlp_pending = false;
};

}  // namespace viewer

// Convenience accessor used by GLFW free-function callbacks.
#define GET_VIEWER(win) static_cast<viewer::Viewer*>(glfwGetWindowUserPointer(win))
