// viewer.cpp — OpenGL mesh viewer implementation.

#include "viewer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write a binary PPM (P6) screenshot from RGBA pixels.
static bool write_ppm(const char* path, int w, int h,
                      const unsigned char* rgba) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << ' ' << h << "\n255\n";
    for (int i = 0; i < w * h; ++i)
        f.write(reinterpret_cast<const char*>(rgba + i * 4), 3);
    return f.good();
}

static void check_gl(const char* call) {
    GLenum e;
    while ((e = glGetError()) != GL_NO_ERROR)
        std::cerr << "[GL] 0x" << std::hex << e << std::dec
                  << " after " << call << '\n';
}
#define GL(x) do { x; check_gl(#x); } while (0)

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    GL(glShaderSource(s, 1, &src, nullptr));
    GL(glCompileShader(s));
    GLint ok = GL_FALSE;
    GL(glGetShaderiv(s, GL_COMPILE_STATUS, &ok));
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(std::max(1, len), '\0');
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    GL(glAttachShader(p, vs));
    GL(glAttachShader(p, fs));
    GL(glLinkProgram(p));
    GLint ok = GL_FALSE;
    GL(glGetProgramiv(p, GL_LINK_STATUS, &ok));
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(std::max(1, len), '\0');
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return p;
}

// ---------------------------------------------------------------------------
// GLFW bootstrap
// ---------------------------------------------------------------------------

static void glfw_error_cb(int /*error*/, const char* desc) {
    std::cerr << desc << '\n';
}

static GLFWwindow* create_window(int w, int h, const char* title) {
    glfwSetErrorCallback(glfw_error_cb);
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        std::exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* win = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if (!win) {
        glfwTerminate();
        std::cerr << "glfwCreateWindow failed\n";
        std::exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(win);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialise OpenGL loader\n";
        std::exit(EXIT_FAILURE);
    }

    glfwSwapInterval(0);

    GL(glClearDepth(1.0));
    GL(glDepthFunc(GL_LESS));
    GL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, false);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;

    return win;
}

// FPS title display
static void update_title(GLFWwindow* win) {
    static double prev = 0.0;
    static int    frames = 0;
    ++frames;
    double now = glfwGetTime();
    if (now - prev > 0.5) {
        int w, h;
        glfwGetWindowSize(win, &w, &h);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Viewer  %dx%d  FPS: %.1f",
                      w, h, frames / (now - prev));
        glfwSetWindowTitle(win, buf);
        prev = now;
        frames = 0;
    }
}

// ---------------------------------------------------------------------------
// UBO binding points
// ---------------------------------------------------------------------------
static constexpr GLuint kBind_W1 = 0, kBind_B1 = 1;
static constexpr GLuint kBind_W2 = 2, kBind_B2 = 3;
static constexpr GLuint kBind_W3 = 4, kBind_B3 = 5;

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

static const char* kVS_Geom = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in uint aTriID;

flat out uint vTriID;
out vec3 vBary;

uniform mat4 uMVP;

void main() {
    vTriID = aTriID;
    int corner = gl_VertexID % 3;
    vBary = (corner == 0) ? vec3(1,0,0) :
            (corner == 1) ? vec3(0,1,0) :
                            vec3(0,0,1);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* kFS_Geom = R"(
#version 330 core
flat in uint vTriID;
in vec3 vBary;

layout(location=0) out vec4 FragA;
layout(location=1) out vec4 FragB;

uniform ivec2     uTriTexSize;
uniform sampler2D uTriTex0;
uniform sampler2D uTriTex1;
uniform int       uLevel;

ivec2 idx_to_coord(int idx, int texW) {
    return ivec2(idx % texW, idx / texW);
}
int level_size(int L) {
    if (L == 0) return 3;
    int a = (1 << (L - 1)) + 1;
    int b = (1 << L) + 1;
    return a * b;
}

void main() {
    int texW = uTriTexSize.x;
    int texH = uTriTexSize.y;
    if (texW * texH <= 0) { FragA = FragB = vec4(1,0,1,1); return; }

    int S    = level_size(uLevel);
    int base = int(vTriID) * S;

    float b0 = vBary.x, b1 = vBary.y;
    int   res   = 1 << uLevel;
    float res_f = float(res);

    float b0l = b0 * res_f;
    float b1l = b1 * res_f;
    int x = clamp(int(floor(b0l)), 0, res - 1);
    int y = clamp(int(floor(b1l)), 0, (res - 1) - x);
    b0l -= float(x);
    b1l -= float(y);

    bool  flip   = (b0l + b1l) > 1.0;
    int   flip_u = flip ? 1 : 0;
    float flip_f = flip ? 1.0 : 0.0;

    int x0 = x + 1,       y0 = y;
    int x1 = x,           y1 = y + 1;
    int x2 = x + flip_u,  y2 = min(y + flip_u, res - x2);

    int idx0 = (x0+y0)*(x0+y0+1)/2 + y0;
    int idx1 = (x1+y1)*(x1+y1+1)/2 + y1;
    int idx2 = (x2+y2)*(x2+y2+1)/2 + y2;

    float w0 = mix(b0l, 1.0 - b1l, flip_f);
    float w1 = mix(b1l, 1.0 - b0l, flip_f);
    float w2 = 1.0 - w0 - w1;

    ivec2 c0 = idx_to_coord(base + idx0, texW);
    ivec2 c1 = idx_to_coord(base + idx1, texW);
    ivec2 c2 = idx_to_coord(base + idx2, texW);

    vec4 a  = texelFetch(uTriTex0, c0, 0)*w0 + texelFetch(uTriTex0, c1, 0)*w1 + texelFetch(uTriTex0, c2, 0)*w2;
    vec4 b  = texelFetch(uTriTex1, c0, 0)*w0 + texelFetch(uTriTex1, c1, 0)*w1 + texelFetch(uTriTex1, c2, 0)*w2;

    if (b.a < 0.5) discard;

    FragA = a;
    FragB = vec4(b.rgb, 1.0);
}
)";

static const char* kVS_Post = R"(
#version 410 core
const vec2 verts[3] = vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
out vec2 vUV;
void main() {
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
    vec2 uv = 0.5 * (gl_Position.xy + 1.0);
    vUV = uv;
}
)";

static const char* kFS_Post = R"(
#version 410 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D texA;
uniform sampler2D texB;
uniform mat4 uInvMVP;

layout(std140) uniform W1Block { mat4 W1[16]; };
layout(std140) uniform B1Block { vec4 B1[4];  };
layout(std140) uniform W2Block { mat4 W2[16]; };
layout(std140) uniform B2Block { vec4 B2[4];  };
layout(std140) uniform W3Block { mat4 W3[4];  };
layout(std140) uniform B3Block { vec4 B3;     };

vec4  relu4(vec4 x) { return max(x, 0.0); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

vec3 ndc_to_world(vec4 ndc) {
    vec4 clip = uInvMVP * ndc;
    float w = clip.w;
    if (abs(w) < 1e-20) w = 1e-20;
    return clip.xyz / w;
}

// Spherical-harmonics basis up to degree 2 (9 coefficients).
const float SH_C0   =  0.28209479177387814;
const float SH_C1   =  0.4886025119029199;
const float SH_C2_0 =  1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 =  0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 =  0.5462742152960396;

void eval_sh2(vec3 d, out float sh[9]) {
    sh[0] = SH_C0;
    sh[1] = -SH_C1 * d.y;
    sh[2] =  SH_C1 * d.z;
    sh[3] = -SH_C1 * d.x;
    float xx = d.x*d.x, yy = d.y*d.y, zz = d.z*d.z;
    sh[4] = SH_C2_0 * d.x * d.y;
    sh[5] = SH_C2_1 * d.y * d.z;
    sh[6] = SH_C2_2 * (2.0*zz - xx - yy);
    sh[7] = SH_C2_3 * d.x * d.z;
    sh[8] = SH_C2_4 * (xx - yy);
}

void main() {
    vec4 A = texture(texA, vUV);
    vec4 B = texture(texB, vUV);

    if (B.a < 0.5) { FragColor = vec4(A.rgb, 1.0); return; }

    // Per-pixel view direction via inverse MVP.
    float ndc_x = vUV.x *  2.0 - 1.0;
    float ndc_y = vUV.y *  2.0 - 1.0;
    vec3 w_near = ndc_to_world(vec4(ndc_x, ndc_y, -1.0, 1.0));
    vec3 w_far  = ndc_to_world(vec4(ndc_x, ndc_y,  1.0, 1.0));
    vec3 v = normalize(w_near - w_far);

    // SH2 view-direction encoding.
    float sh[9];
    eval_sh2(v, sh);

    // 16-D MLP input: A.rgba (4) + B.rgb (3) + SH2 (9) = 16.
    vec4 x0 = vec4(A.r, A.g, A.b, A.a);
    vec4 x1 = vec4(B.r, B.g, B.b, sh[0]);
    vec4 x2 = vec4(sh[1], sh[2], sh[3], sh[4]);
    vec4 x3 = vec4(sh[5], sh[6], sh[7], sh[8]);

    // Layer 1: 16->16, ReLU
    vec4 y0 = relu4(W1[ 0]*x0 + W1[ 1]*x1 + W1[ 2]*x2 + W1[ 3]*x3 + B1[0]);
    vec4 y1 = relu4(W1[ 4]*x0 + W1[ 5]*x1 + W1[ 6]*x2 + W1[ 7]*x3 + B1[1]);
    vec4 y2 = relu4(W1[ 8]*x0 + W1[ 9]*x1 + W1[10]*x2 + W1[11]*x3 + B1[2]);
    vec4 y3 = relu4(W1[12]*x0 + W1[13]*x1 + W1[14]*x2 + W1[15]*x3 + B1[3]);

    // Layer 2: 16->16, ReLU
    vec4 z0 = relu4(W2[ 0]*y0 + W2[ 1]*y1 + W2[ 2]*y2 + W2[ 3]*y3 + B2[0]);
    vec4 z1 = relu4(W2[ 4]*y0 + W2[ 5]*y1 + W2[ 6]*y2 + W2[ 7]*y3 + B2[1]);
    vec4 z2 = relu4(W2[ 8]*y0 + W2[ 9]*y1 + W2[10]*y2 + W2[11]*y3 + B2[2]);
    vec4 z3 = relu4(W2[12]*y0 + W2[13]*y1 + W2[14]*y2 + W2[15]*y3 + B2[3]);

    // Layer 3: 16->3, sigmoid
    vec4 acc = W3[0]*z0 + W3[1]*z1 + W3[2]*z2 + W3[3]*z3 + B3;
    vec3 mlp = vec3(sigmoid(acc.x), sigmoid(acc.y), sigmoid(acc.z));

    // Residual blend: (1-a)*albedo + a*mlp, where a = A.a.
    FragColor = vec4(mix(A.rgb, mlp, A.a), 1.0);
}
)";

// ---------------------------------------------------------------------------
// Viewer implementation
// ---------------------------------------------------------------------------
namespace viewer {

Viewer::Viewer()
    : m_camera(std::make_unique<OrbitCamera>()) {}

Viewer::~Viewer() {
    if (!m_window) return;

    auto del_buf = [](GLuint& id) { if (id) { glDeleteBuffers(1, &id); id = 0; } };
    auto del_tex = [](GLuint& id) { if (id) { glDeleteTextures(1, &id); id = 0; } };
    auto del_vao = [](GLuint& id) { if (id) { glDeleteVertexArrays(1, &id); id = 0; } };
    auto del_fbo = [](GLuint& id) { if (id) { glDeleteFramebuffers(1, &id); id = 0; } };
    auto del_prg = [](GLuint& id) { if (id) { glDeleteProgram(id); id = 0; } };

    del_prg(m_post_prog); del_vao(m_post_vao);
    del_prg(m_geom_prog); del_buf(m_geom_vbo_pos); del_buf(m_geom_vbo_tid); del_vao(m_geom_vao);
    del_tex(m_lut_tex[0]); del_tex(m_lut_tex[1]);
    del_tex(m_tex_color[0]); del_tex(m_tex_color[1]); del_tex(m_tex_depth);
    del_fbo(m_fbo);
    del_buf(m_ubo_W1); del_buf(m_ubo_B1);
    del_buf(m_ubo_W2); del_buf(m_ubo_B2);
    del_buf(m_ubo_W3); del_buf(m_ubo_B3);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

// ---------------------------------------------------------------------------
// Data upload (pre-launch)
// ---------------------------------------------------------------------------

void Viewer::set_output_dir(const std::string& dir) {
    m_output_dir = dir;
    if (!m_output_dir.empty() && m_output_dir.back() != '/')
        m_output_dir.push_back('/');
}

void Viewer::set_mesh(int V, const float* verts, int F, const int* faces) {
    m_pos_host.clear();
    m_tid_host.clear();
    m_vertex_count = 0;
    m_face_count   = 0;

    if (V <= 0 || F <= 0 || !verts || !faces) return;

    m_pos_host.reserve(size_t(F) * 9);
    m_tid_host.reserve(size_t(F) * 3);

    for (int f = 0; f < F; ++f) {
        const int i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
        if (i0 < 0 || i0 >= V || i1 < 0 || i1 >= V || i2 < 0 || i2 >= V) continue;
        for (int vi : {i0, i1, i2}) {
            m_pos_host.push_back(verts[vi*3]);
            m_pos_host.push_back(verts[vi*3+1]);
            m_pos_host.push_back(verts[vi*3+2]);
            m_tid_host.push_back(uint32_t(f));
        }
    }
    m_vertex_count = GLsizei(m_pos_host.size() / 3);
    m_face_count   = F;
    m_mesh_dirty   = true;
}

void Viewer::set_triangle_color_lut(int idx, int width, int height,
                                    const unsigned char* rgba) {
    if (idx != 0 && idx != 1) return;
    m_lut_w = width;
    m_lut_h = height;
    m_lut_staging[idx].assign(rgba, rgba + size_t(width) * height * 4);
    m_lut_pending[idx] = true;
    if (m_gl_ready) upload_pending_luts();
}

void Viewer::set_mlp_weights(const float* W1, const float* b1,
                             const float* W2, const float* b2,
                             const float* W3, const float* b3) {
    auto assign = [](std::vector<float>& dst, const float* src, size_t n) {
        if (src) dst.assign(src, src + n); else dst.clear();
    };
    assign(m_W1, W1, 16*16);  assign(m_b1, b1, 16);
    assign(m_W2, W2, 16*16);  assign(m_b2, b2, 16);
    assign(m_W3, W3, 3*16);   assign(m_b3, b3, 3);
    m_mlp_pending = true;
    if (m_gl_ready) upload_pending_mlp();
}

void Viewer::set_camera_target(const float target[3]) {
    m_camera->target = glm::vec3(target[0], target[1], target[2]);
    m_camera->update();
}

void Viewer::set_world_up(const float up[3]) {
    m_camera->world_up = glm::normalize(glm::vec3(up[0], up[1], up[2]));
    m_camera->update();
}

// ---------------------------------------------------------------------------
// GL initialisation
// ---------------------------------------------------------------------------

static void ensure_ubo(GLuint& ubo, GLsizeiptr bytes, GLuint binding) {
    if (ubo) return;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, binding, ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Viewer::init_gl() {
    if (m_gl_ready) return;

    // Geometry pass
    GL(glGenVertexArrays(1, &m_geom_vao));
    GL(glGenBuffers(1, &m_geom_vbo_pos));
    GL(glGenBuffers(1, &m_geom_vbo_tid));

    {
        GLuint vs = compile_shader(GL_VERTEX_SHADER,   kVS_Geom);
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFS_Geom);
        m_geom_prog = link_program(vs, fs);
        glDeleteShader(vs); glDeleteShader(fs);
    }
    m_loc_mvp        = glGetUniformLocation(m_geom_prog, "uMVP");
    m_loc_tri_tex0   = glGetUniformLocation(m_geom_prog, "uTriTex0");
    m_loc_tri_tex1   = glGetUniformLocation(m_geom_prog, "uTriTex1");
    m_loc_tri_tex_sz = glGetUniformLocation(m_geom_prog, "uTriTexSize");
    m_loc_level      = glGetUniformLocation(m_geom_prog, "uLevel");

    // Post / composite pass
    {
        GLuint vs = compile_shader(GL_VERTEX_SHADER,   kVS_Post);
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFS_Post);
        m_post_prog = link_program(vs, fs);
        glDeleteShader(vs); glDeleteShader(fs);
    }
    m_post_loc_texA    = glGetUniformLocation(m_post_prog, "texA");
    m_post_loc_texB    = glGetUniformLocation(m_post_prog, "texB");
    m_post_loc_inv_mvp = glGetUniformLocation(m_post_prog, "uInvMVP");

    auto bind_block = [&](const char* name, GLuint binding) {
        GLuint idx = glGetUniformBlockIndex(m_post_prog, name);
        if (idx != GL_INVALID_INDEX)
            glUniformBlockBinding(m_post_prog, idx, binding);
    };
    bind_block("W1Block", kBind_W1); bind_block("B1Block", kBind_B1);
    bind_block("W2Block", kBind_W2); bind_block("B2Block", kBind_B2);
    bind_block("W3Block", kBind_W3); bind_block("B3Block", kBind_B3);

    GL(glGenVertexArrays(1, &m_post_vao));

    // UBOs (zero-initialised)
    ensure_ubo(m_ubo_W1, sizeof(float)*16*16, kBind_W1);
    ensure_ubo(m_ubo_B1, sizeof(float)*16,    kBind_B1);
    ensure_ubo(m_ubo_W2, sizeof(float)*16*16, kBind_W2);
    ensure_ubo(m_ubo_B2, sizeof(float)*16,    kBind_B2);
    ensure_ubo(m_ubo_W3, sizeof(float)*4*16,  kBind_W3);
    ensure_ubo(m_ubo_B3, sizeof(float)*4,     kBind_B3);

    m_gl_ready = true;

    // Flush any data that was staged before GL context existed.
    upload_mesh_to_gpu();
    upload_pending_luts();
    upload_pending_mlp();
}

// ---------------------------------------------------------------------------
// GPU uploads
// ---------------------------------------------------------------------------

void Viewer::upload_mesh_to_gpu() {
    if (!m_mesh_dirty || !m_gl_ready) return;
    if (m_pos_host.empty()) { m_mesh_dirty = false; return; }

    GL(glBindVertexArray(m_geom_vao));

    GL(glBindBuffer(GL_ARRAY_BUFFER, m_geom_vbo_pos));
    GL(glBufferData(GL_ARRAY_BUFFER,
                    m_pos_host.size() * sizeof(float),
                    m_pos_host.data(), GL_STATIC_DRAW));
    GL(glEnableVertexAttribArray(0));
    GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));

    GL(glBindBuffer(GL_ARRAY_BUFFER, m_geom_vbo_tid));
    GL(glBufferData(GL_ARRAY_BUFFER,
                    m_tid_host.size() * sizeof(uint32_t),
                    m_tid_host.data(), GL_STATIC_DRAW));
    GL(glEnableVertexAttribArray(1));
    GL(glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, 0, nullptr));

    GL(glBindVertexArray(0));
    m_mesh_dirty = false;
}

static void upload_lut_tex(GLuint& tex, int W, int H,
                           const std::vector<unsigned char>& rgba) {
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Viewer::upload_pending_luts() {
    if (!m_gl_ready) return;
    for (int i = 0; i < 2; ++i) {
        if (!m_lut_pending[i]) continue;
        if (m_lut_w <= 0 || m_lut_h <= 0) continue;
        if (int(m_lut_staging[i].size()) != m_lut_w * m_lut_h * 4) continue;
        upload_lut_tex(m_lut_tex[i], m_lut_w, m_lut_h, m_lut_staging[i]);
        m_lut_ready[i]   = true;
        m_lut_pending[i] = false;
        m_lut_staging[i] = {};  // free staging memory
    }
}

/// Tile a row-major weight matrix into column-major mat4 blocks for std140.
static void tile_weights(float* dst, const float* src,
                         int out_dim, int in_dim) {
    std::memset(dst, 0, sizeof(float) * 16 * 16);
    for (int tr = 0; tr < 4; ++tr)
        for (int tc = 0; tc < 4; ++tc)
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) {
                    int gr = tr * 4 + r, gc = tc * 4 + c;
                    if (gr < out_dim && gc < in_dim)
                        dst[(tr*4+tc)*16 + c*4 + r] = src[gr * in_dim + gc];
                }
}

void Viewer::upload_pending_mlp() {
    if (!m_mlp_pending || !m_gl_ready) return;

    auto upload_buf = [](GLuint ubo, const void* data, GLsizeiptr bytes) {
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, bytes, data);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    };

    float tiles[16*16];

    if (!m_W1.empty()) { tile_weights(tiles, m_W1.data(), 16, 16); upload_buf(m_ubo_W1, tiles, sizeof(tiles)); }
    if (!m_b1.empty()) { float b[16]={}; std::copy_n(m_b1.data(), 16, b); upload_buf(m_ubo_B1, b, sizeof(b)); }
    if (!m_W2.empty()) { tile_weights(tiles, m_W2.data(), 16, 16); upload_buf(m_ubo_W2, tiles, sizeof(tiles)); }
    if (!m_b2.empty()) { float b[16]={}; std::copy_n(m_b2.data(), 16, b); upload_buf(m_ubo_B2, b, sizeof(b)); }

    if (!m_W3.empty()) {
        float t3[4*16] = {};
        for (int tc = 0; tc < 4; ++tc)
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 3; ++r) {
                    int gc = tc * 4 + c;
                    if (gc < 16)
                        t3[tc*16 + c*4 + r] = m_W3[r * 16 + gc];
                }
        upload_buf(m_ubo_W3, t3, sizeof(t3));
    }
    if (!m_b3.empty()) {
        float b[4] = { m_b3[0], m_b3[1], m_b3[2], 0.f };
        upload_buf(m_ubo_B3, b, sizeof(b));
    }

    m_mlp_pending = false;
}

// ---------------------------------------------------------------------------
// FBO resize
// ---------------------------------------------------------------------------

void Viewer::resize_fbo(int width, int height) {
    if (width == m_camera->width && height == m_camera->height) return;
    m_camera->width  = width;
    m_camera->height = height;

    if (!m_fbo) glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    for (int i = 0; i < 2; ++i) {
        if (!m_tex_color[i]) glGenTextures(1, &m_tex_color[i]);
        glBindTexture(GL_TEXTURE_2D, m_tex_color[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                               GL_TEXTURE_2D, m_tex_color[i], 0);
    }

    if (!m_tex_depth) glGenTextures(1, &m_tex_depth);
    glBindTexture(GL_TEXTURE_2D, m_tex_depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0,
                 GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, m_tex_depth, 0);

    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, bufs);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[viewer] FBO incomplete\n";
        std::exit(EXIT_FAILURE);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void Viewer::render(const glm::mat4& mvp) {
    const int w = m_camera->width, h = m_camera->height;

    // Pass 0 — rasterise mesh into two colour attachments.
    GL(glBindFramebuffer(GL_FRAMEBUFFER, m_fbo));
    GL(glViewport(0, 0, w, h));

    const float* bg = m_options.background;
    const GLfloat clear_a[] = { bg[0], bg[1], bg[2], 1.f };
    const GLfloat clear_b[] = { bg[0], bg[1], bg[2], 0.f };
    GL(glClearBufferfv(GL_COLOR, 0, clear_a));
    GL(glClearBufferfv(GL_COLOR, 1, clear_b));
    GL(glClear(GL_DEPTH_BUFFER_BIT));

    if (m_vertex_count > 0 && m_geom_vao && m_options.render_geometry) {
        GL(glEnable(GL_DEPTH_TEST));
        GL(glDisable(GL_BLEND));
        GL(glDisable(GL_CULL_FACE));

        GL(glUseProgram(m_geom_prog));
        GL(glUniformMatrix4fv(m_loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp)));
        GL(glUniform1i(m_loc_tri_tex0, 0));
        GL(glUniform1i(m_loc_tri_tex1, 1));
        GL(glUniform2i(m_loc_tri_tex_sz, m_lut_w, m_lut_h));
        GL(glUniform1i(m_loc_level, m_level));

        GL(glActiveTexture(GL_TEXTURE0)); GL(glBindTexture(GL_TEXTURE_2D, m_lut_tex[0]));
        GL(glActiveTexture(GL_TEXTURE1)); GL(glBindTexture(GL_TEXTURE_2D, m_lut_tex[1]));

        GL(glBindVertexArray(m_geom_vao));
        GL(glDrawArrays(GL_TRIANGLES, 0, m_vertex_count));
        GL(glBindVertexArray(0));
        GL(glUseProgram(0));
    }

    // Pass 1 — composite to default framebuffer via MLP.
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL(glViewport(0, 0, w, h));
    GL(glDisable(GL_DEPTH_TEST));

    GL(glUseProgram(m_post_prog));
    GL(glUniform1i(m_post_loc_texA, 0));
    GL(glUniform1i(m_post_loc_texB, 1));

    GL(glActiveTexture(GL_TEXTURE0)); GL(glBindTexture(GL_TEXTURE_2D, m_tex_color[0]));
    GL(glActiveTexture(GL_TEXTURE1)); GL(glBindTexture(GL_TEXTURE_2D, m_tex_color[1]));

    glm::mat4 inv_mvp = glm::inverse(mvp);
    GL(glUniformMatrix4fv(m_post_loc_inv_mvp, 1, GL_FALSE, glm::value_ptr(inv_mvp)));

    GL(glBindVertexArray(m_post_vao));
    GL(glDrawArrays(GL_TRIANGLES, 0, 3));
    GL(glBindVertexArray(0));
    GL(glUseProgram(0));
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

void Viewer::draw_gui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(280, 260), ImGuiCond_Once);
    ImGui::Begin("Settings");

    ImGui::Text("Resolution: %dx%d", m_camera->width, m_camera->height);
    ImGui::Text("Faces: %d", m_face_count);
    ImGui::ColorEdit3("Background", m_options.background);

    ImGui::SliderFloat("FOV", &m_camera->fov_y_deg, 10.f, 120.f);
    ImGui::SliderFloat("Near clip", &m_camera->near_clip, 0.01f, 50.f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Far clip", &m_camera->far_clip, 1.f, 1000.f, "%.1f", ImGuiSliderFlags_Logarithmic);

    if (!m_output_dir.empty() && ImGui::Button("Save screenshot")) {
        int w = m_camera->width, h = m_camera->height;
        std::vector<unsigned char> px(4 * w * h), flipped(4 * w * h);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        for (int row = 0; row < h; ++row)
            std::memcpy(&flipped[row * w * 4],
                        &px[(h - 1 - row) * w * 4], size_t(w) * 4);
        write_ppm((m_output_dir + "screenshot.ppm").c_str(), w, h, flipped.data());
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ---------------------------------------------------------------------------
// Input callbacks
// ---------------------------------------------------------------------------

void Viewer::on_mouse_button(GLFWwindow* win, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(win, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(m_window, &x, &y);
        bool pan = (button == GLFW_MOUSE_BUTTON_RIGHT) ||
                   (button == GLFW_MOUSE_BUTTON_MIDDLE) ||
                   (mods & GLFW_MOD_SHIFT);
        m_camera->begin_drag(float(x), float(y), pan);
    } else if (action == GLFW_RELEASE) {
        m_camera->end_drag();
    }
}

void Viewer::on_cursor_pos(GLFWwindow* win, double x, double y) {
    ImGui_ImplGlfw_CursorPosCallback(win, x, y);
    if (ImGui::GetIO().WantCaptureMouse) return;
    m_camera->drag_update(float(x), float(y));
}

void Viewer::on_scroll(GLFWwindow* win, double xoff, double yoff) {
    ImGui_ImplGlfw_ScrollCallback(win, xoff, yoff);
    if (ImGui::GetIO().WantCaptureMouse) return;
    m_camera->scroll(float(yoff));
}

void Viewer::on_framebuffer_size(GLFWwindow* /*win*/, int w, int h) {
    resize_fbo(w, h);
}

// ---------------------------------------------------------------------------
// Interactive launch
// ---------------------------------------------------------------------------

void Viewer::launch(int w, int h) {
    if (m_window) return;
    m_window = create_window(w, h, "Viewer");
    init_gl();

    int fw, fh;
    glfwGetFramebufferSize(m_window, &fw, &fh);
    resize_fbo(fw, fh);

    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, [](GLFWwindow* w, int b, int a, int m) {
        GET_VIEWER(w)->on_mouse_button(w, b, a, m);
    });
    glfwSetCursorPosCallback(m_window, [](GLFWwindow* w, double x, double y) {
        GET_VIEWER(w)->on_cursor_pos(w, x, y);
    });
    glfwSetScrollCallback(m_window, [](GLFWwindow* w, double x, double y) {
        GET_VIEWER(w)->on_scroll(w, x, y);
    });
    glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* w, int fw, int fh) {
        GET_VIEWER(w)->on_framebuffer_size(w, fw, fh);
    });

    while (!glfwWindowShouldClose(m_window)) {
        update_title(m_window);
        m_camera->update();
        render(m_camera->mvp());
        draw_gui();
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

void Viewer::launch_benchmark(int width, int height,
                              int B, const float* mvps,
                              int warmup_frames, int save_every) {
    m_window = create_window(width, height, "Benchmark");
    init_gl();
    resize_fbo(width, height);

    // Off-screen FBO for readback.
    GLuint final_fbo = 0, final_tex = 0;
    GL(glGenFramebuffers(1, &final_fbo));
    GL(glBindFramebuffer(GL_FRAMEBUFFER, final_fbo));

    GL(glGenTextures(1, &final_tex));
    GL(glBindTexture(GL_TEXTURE_2D, final_tex));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, final_tex, 0));
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[viewer] benchmark FBO incomplete\n";
        std::exit(EXIT_FAILURE);
    }
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    auto render_one = [&](const glm::mat4& mvp) {
        GL(glBindFramebuffer(GL_FRAMEBUFFER, m_fbo));
        GL(glViewport(0, 0, width, height));

        const float* bg = m_options.background;
        const GLfloat c0[] = {bg[0],bg[1],bg[2],1.f}, c1[] = {bg[0],bg[1],bg[2],0.f};
        GL(glClearBufferfv(GL_COLOR, 0, c0));
        GL(glClearBufferfv(GL_COLOR, 1, c1));
        GL(glClear(GL_DEPTH_BUFFER_BIT));

        if (m_vertex_count > 0 && m_geom_vao && m_options.render_geometry) {
            GL(glEnable(GL_DEPTH_TEST));
            GL(glDisable(GL_BLEND));
            GL(glDisable(GL_CULL_FACE));
            GL(glUseProgram(m_geom_prog));
            GL(glUniformMatrix4fv(m_loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp)));
            GL(glUniform1i(m_loc_tri_tex0, 0));
            GL(glUniform1i(m_loc_tri_tex1, 1));
            GL(glUniform2i(m_loc_tri_tex_sz, m_lut_w, m_lut_h));
            GL(glUniform1i(m_loc_level, m_level));
            GL(glActiveTexture(GL_TEXTURE0)); GL(glBindTexture(GL_TEXTURE_2D, m_lut_tex[0]));
            GL(glActiveTexture(GL_TEXTURE1)); GL(glBindTexture(GL_TEXTURE_2D, m_lut_tex[1]));
            GL(glBindVertexArray(m_geom_vao));
            GL(glDrawArrays(GL_TRIANGLES, 0, m_vertex_count));
            GL(glBindVertexArray(0));
            GL(glUseProgram(0));
        }

        GL(glBindFramebuffer(GL_FRAMEBUFFER, final_fbo));
        GL(glViewport(0, 0, width, height));
        GL(glDisable(GL_DEPTH_TEST));
        GL(glUseProgram(m_post_prog));
        GL(glUniform1i(m_post_loc_texA, 0));
        GL(glUniform1i(m_post_loc_texB, 1));
        GL(glActiveTexture(GL_TEXTURE0)); GL(glBindTexture(GL_TEXTURE_2D, m_tex_color[0]));
        GL(glActiveTexture(GL_TEXTURE1)); GL(glBindTexture(GL_TEXTURE_2D, m_tex_color[1]));
        glm::mat4 inv = glm::inverse(mvp);
        GL(glUniformMatrix4fv(m_post_loc_inv_mvp, 1, GL_FALSE, glm::value_ptr(inv)));
        GL(glBindVertexArray(m_post_vao));
        GL(glDrawArrays(GL_TRIANGLES, 0, 3));
        GL(glBindVertexArray(0));
        GL(glUseProgram(0));

        GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, final_fbo));
        GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        GL(glBlitFramebuffer(0,0,width,height, 0,0,width,height,
                             GL_COLOR_BUFFER_BIT, GL_NEAREST));
        GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    };

    glm::mat4 mvp_buf;
    for (int i = 0; i < warmup_frames; ++i) {
        mvp_buf = glm::make_mat4(mvps);
        render_one(mvp_buf);
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }

    constexpr int kRepeat = 100;
    std::vector<double> times_ms;
    times_ms.reserve(B);

    for (int i = 0; i < B; ++i) {
        mvp_buf = glm::make_mat4(mvps + i * 16);

        GL(glFinish());
        double t0 = glfwGetTime();
        for (int r = 0; r < kRepeat; ++r)
            render_one(mvp_buf);
        GL(glFinish());
        double t1 = glfwGetTime();

        glfwSwapBuffers(m_window);
        glfwPollEvents();
        times_ms.push_back((t1 - t0) * 1000.0 / kRepeat);

        if (save_every > 0 && (i % save_every == 0) && !m_output_dir.empty()) {
            std::vector<unsigned char> px(4*width*height), flipped(4*width*height);
            GL(glBindFramebuffer(GL_FRAMEBUFFER, final_fbo));
            GL(glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, px.data()));
            GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
            for (int row = 0; row < height; ++row)
                std::memcpy(&flipped[row*width*4],
                            &px[(height-1-row)*width*4], size_t(width)*4);
            char path[512];
            std::snprintf(path, sizeof(path), "%sscreenshots/benchmark_%05d.ppm",
                          m_output_dir.c_str(), i);
            write_ppm(path, width, height, flipped.data());
        }
    }

    if (!m_output_dir.empty()) {
        {
            std::ofstream f(m_output_dir + "benchmark_frames.txt");
            for (size_t i = 0; i < times_ms.size(); ++i)
                f << i << ' ' << times_ms[i] << '\n';
        }
        double sum = 0, mn = times_ms[0], mx = times_ms[0];
        for (double t : times_ms) { sum += t; mn = std::min(mn,t); mx = std::max(mx,t); }
        double mean = sum / double(times_ms.size());
        std::ofstream f(m_output_dir + "benchmark_summary.txt");
        f << "frames: " << B << '\n'
          << "mean_ms: " << mean << '\n'
          << "min_ms:  " << mn << '\n'
          << "max_ms:  " << mx << '\n'
          << "fps:     " << 1000.0 / mean << '\n';
    }

    GL(glDeleteTextures(1, &final_tex));
    GL(glDeleteFramebuffers(1, &final_fbo));
    glfwDestroyWindow(m_window);
    glfwTerminate();
    m_window = nullptr;
}

}  // namespace viewer
