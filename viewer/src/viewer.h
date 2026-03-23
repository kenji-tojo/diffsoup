#pragma once

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <cstdint>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "camera.h"
#include "triangles.h"
#include "options.h"

namespace viewer {

class Viewer {
public:
    Viewer();
    ~Viewer();

    void launch(int width, int height);
    void set_output_dir(std::string output_dir);
    void set_triangles(int V, const float* verts, int F, const int* faces);

    // Upload per-triangle RGBA LUTs (W x H, RGBA8). idx=0 fills A; idx=1 fills B.
    void set_triangle_color_lut(int idx, int width, int height, const unsigned char* rgba);

    // Upload trained MLP weights:
    // W1[16*16] row-major (out x in), b1[16],
    // W2[16*16], b2[16], W3[3*16], b3[3].
    void set_mlp_weights(
        const float* W1, const float* b1,
        const float* W2, const float* b2,
        const float* W3, const float* b3,
        float enc_freq
    );

    void set_camera_center(float cen[3]);

    // Input callbacks
    void _mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
    void _cursor_pos_callback(GLFWwindow* window, double x, double y);
    void _scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    void _window_size_callback(GLFWwindow *window, int width, int height);

    void launch_benchmark(
        int width,
        int height,
        int B,
        const float* mvps,   // B * 16 floats, column-major
        int warmup_frames,
        int save_every
    );

private:
    std::string m_output_dir;

    std::unique_ptr<Camera>    m_camera;
    std::unique_ptr<Triangles> m_triangles;
    Options m_options;

    GLFWwindow *m_window = nullptr;

    // First pass FBO with 2 color attachments (textures) + depth (texture)
    GLuint m_FBO{0};
    GLuint m_tex_color[2]{0,0}; // A (0), B (1)
    GLuint m_tex_depth{0};

    // Geometry pass
    GLuint m_vao = 0;
    GLuint m_vbo_pos = 0;
    GLuint m_vbo_tid = 0;
    GLuint m_prog = 0;
    GLint  m_loc_uMVP = -1;
    GLint  m_loc_uTriTex0 = -1;
    GLint  m_loc_uTriTex1 = -1;
    GLint  m_loc_uTriTexSize = -1;
    GLint  m_loc_uLevel = -1;

    // Composite pass (A,B → screen via MLP)
    GLuint m_postProg{0};
    GLuint m_postVAO{0};
    GLint  m_postLoc_texA{-1};
    GLint  m_postLoc_texB{-1};

    // UBOs for weights (std140): block-structured 16x16 → 4x4 mat4 tiles
    // We use 1D arrays in GLSL: W1[16], B1[4]; W2[16], B2[4]; W3[4], B3.
    GLuint m_uboW1{0}, m_uboB1{0};
    GLuint m_uboW2{0}, m_uboB2{0};
    GLuint m_uboW3{0}, m_uboB3{0};

    GLsizei m_vertex_count = 0;

    // Host geometry staging
    std::vector<float>    m_pos_dup_host;
    std::vector<uint32_t> m_tid_dup_host;
    bool m_need_gl_upload = false;

    // Per-triangle color LUTs (two textures, same WxH)
    GLuint m_triTex[2]{0,0};
    int    m_triTexW = 0;
    int    m_triTexH = 0;
    bool   m_triTexReady[2]{false,false};   // uploaded to GL
    int    m_level = 5;

    GLint  m_postLoc_uInvMVP{-1};
    GLint  m_postLoc_uEncFreq{-1};

    // Staging for MLP weights (store until GL context exists)
    std::vector<float> m_W1_staging, m_b1_staging;
    std::vector<float> m_W2_staging, m_b2_staging;
    std::vector<float> m_W3_staging, m_b3_staging;
    bool m_mlp_weights_pending = false;
    float  m_enc_freq = 16.0f;

    // Staging for uploads before GL context
    std::vector<unsigned char> m_triTexStaging[2]; // W*H*4 bytes
    bool   m_triTexPending[2]{false,false};        // upload at first render

    bool m_started = false;

    bool     m_benchmark_mode = false;
    glm::mat4 m_benchmark_mvp;

    void draw_gui();
    void start();
    void resize(int width, int height);
    void render();
    void ensure_gl_upload();

    void ensure_default_luts();
    void ensure_luts_uploaded();

    void upload_default_mlp_weights_as_ubos();
    void upload_mlp_weights_from_staging();
    void ensure_ubo_alloc();
};

} // namespace viewer

#ifndef GET_VIEWER
#define GET_VIEWER(window) (static_cast<viewer::Viewer*>(glfwGetWindowUserPointer(window)))
#endif

// Global callbacks
void glfw_update_title(GLFWwindow* window);
void glfw_error_callback(int error, const char* description);
void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y);
void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void glfw_window_size_callback(GLFWwindow* window, int width, int height);
