// viewer.h — OpenGL mesh viewer with per-triangle LUT + MLP compositing.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "camera.h"

namespace viewer {

struct Options {
    bool  render_geometry = true;
    float background[3]   = {1.f, 1.f, 1.f};
};

class Viewer {
public:
    Viewer();
    ~Viewer();

    void launch(int width, int height);
    void launch_benchmark(int width, int height,
                          int B, const float* mvps,
                          int warmup_frames, int save_every);

    // ---- Data upload (call before launch) ----------------------------------

    void set_output_dir(const std::string& dir);
    void set_mesh(int V, const float* verts, int F, const int* faces);
    void set_triangle_color_lut(int idx, int width, int height,
                                const unsigned char* rgba);
    void set_mlp_weights(const float* W1, const float* b1,
                         const float* W2, const float* b2,
                         const float* W3, const float* b3);
    void set_camera_target(const float target[3]);
    void set_world_up(const float up[3]);

    // ---- GLFW callbacks ----------------------------------------------------

    void on_mouse_button(GLFWwindow* win, int button, int action, int mods);
    void on_cursor_pos  (GLFWwindow* win, double x, double y);
    void on_scroll      (GLFWwindow* win, double xoff, double yoff);
    void on_framebuffer_size(GLFWwindow* win, int w, int h);

private:
    void init_gl();
    void resize_fbo(int width, int height);
    void upload_mesh_to_gpu();
    void upload_pending_luts();
    void upload_pending_mlp();
    void render(const glm::mat4& mvp);
    void draw_gui();

    std::string            m_output_dir;
    std::unique_ptr<OrbitCamera> m_camera;
    Options                m_options;

    GLFWwindow* m_window = nullptr;
    bool        m_gl_ready = false;

    GLuint m_fbo            = 0;
    GLuint m_tex_color[2]   = {};
    GLuint m_tex_depth      = 0;

    GLuint m_geom_vao       = 0;
    GLuint m_geom_vbo_pos   = 0;
    GLuint m_geom_vbo_tid   = 0;
    GLuint m_geom_prog      = 0;
    GLint  m_loc_mvp        = -1;
    GLint  m_loc_tri_tex0   = -1;
    GLint  m_loc_tri_tex1   = -1;
    GLint  m_loc_tri_tex_sz = -1;
    GLint  m_loc_level      = -1;

    GLuint m_post_prog      = 0;
    GLuint m_post_vao       = 0;
    GLint  m_post_loc_texA  = -1;
    GLint  m_post_loc_texB  = -1;
    GLint  m_post_loc_inv_mvp = -1;

    GLuint m_ubo_W1 = 0, m_ubo_B1 = 0;
    GLuint m_ubo_W2 = 0, m_ubo_B2 = 0;
    GLuint m_ubo_W3 = 0, m_ubo_B3 = 0;

    std::vector<float>    m_pos_host;
    std::vector<uint32_t> m_tid_host;
    GLsizei m_vertex_count   = 0;
    int     m_face_count     = 0;
    bool    m_mesh_dirty     = false;

    GLuint m_lut_tex[2]      = {};
    int    m_lut_w = 0, m_lut_h = 0;
    std::vector<unsigned char> m_lut_staging[2];
    bool   m_lut_pending[2]  = {};
    bool   m_lut_ready[2]    = {};
    int    m_level           = 5;

    std::vector<float> m_W1, m_b1, m_W2, m_b2, m_W3, m_b3;
    bool m_mlp_pending = false;
};

}  // namespace viewer

#define GET_VIEWER(win) static_cast<viewer::Viewer*>(glfwGetWindowUserPointer(win))
