#include "viewer.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "glm/gtc/type_ptr.hpp"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

// Minimal screenshot writer – no external dependency.
// Writes a binary PPM (P6) file from RGBA pixels, dropping the alpha channel.
static bool write_ppm(const char* path, int w, int h, const unsigned char* rgba) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << ' ' << h << "\n255\n";
    for (int i = 0; i < w * h; ++i)
        f.write(reinterpret_cast<const char*>(rgba + i * 4), 3);
    return f.good();
}

#include "camera.h"
#include "triangles.h"
#include "options.h"

// ---- logging helpers ----
static inline void __gl_check_error(const char* what) {
    GLenum e;
    while ((e = glGetError()) != GL_NO_ERROR) {
        std::cerr << "[GL ERROR] 0x" << std::hex << e << std::dec << " after " << what << std::endl;
    }
}
#define GL_CALL(x) do { x; __gl_check_error(#x); } while(0)

// ---------- tiny GL helpers ----------
static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    GL_CALL(glShaderSource(s, 1, &src, nullptr));
    GL_CALL(glCompileShader(s));
    GLint ok = GL_FALSE;
    GL_CALL(glGetShaderiv(s, GL_COMPILE_STATUS, &ok));
    if (!ok) {
        GLint len = 0;
        GL_CALL(glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len));
        std::vector<GLchar> log(std::max(1, len));
        GL_CALL(glGetShaderInfoLog(s, len, nullptr, log.data()));
        std::cerr << "Shader compile error:\n" << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    GL_CALL(glAttachShader(p, vs));
    GL_CALL(glAttachShader(p, fs));
    GL_CALL(glLinkProgram(p));
    GLint ok = GL_FALSE;
    GL_CALL(glGetProgramiv(p, GL_LINK_STATUS, &ok));
    if (!ok) {
        GLint len = 0;
        GL_CALL(glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len));
        std::vector<GLchar> log(std::max(1, len));
        GL_CALL(glGetProgramInfoLog(p, len, nullptr, log.data()));
        std::cerr << "Program link error:\n" << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return p;
}

// ---------- GLFW init ----------
namespace {
static double __viewer_stamp_prev = 0.0;
static int    __viewer_frame_count = 0;
}

void glfw_update_title(GLFWwindow* window) {
    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - __viewer_stamp_prev;
    if (elapsed > 0.5) {
        __viewer_stamp_prev = stamp_curr;
        const double fps = (double)__viewer_frame_count / elapsed;

        // --- get window size ---
        int w, h;
        glfwGetWindowSize(window, &w, &h);

        char tmp[128];
        std::snprintf(
            tmp, sizeof(tmp),
            "Viewer - %dx%d - FPS: %.2f",
            w, h, fps
        );

        glfwSetWindowTitle(window, tmp);
        __viewer_frame_count = 0;
    }
    __viewer_frame_count++;
}

void glfw_error_callback(int error, const char* description) {
    (void)error;
    std::cerr << description << std::endl;
}

static GLFWwindow* glfw_init(const int width, const int height) {
    glfwSetErrorCallback(::glfw_error_callback);

    if (!glfwInit()) {
        std::cerr << "glfwInit failed" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(width, height, "Viewer", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        std::cerr << "glfwCreateWindow failed" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGL((GLADloadfunc) glfwGetProcAddress)) {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glfwSwapInterval(0);

    GL_CALL(glClearDepth(1.0));
    GL_CALL(glDepthFunc(GL_LESS));
    GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    GL_CALL(glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

// ---------- UBO binding points ----------
static constexpr GLuint kBind_W1 = 0;
static constexpr GLuint kBind_B1 = 1;
static constexpr GLuint kBind_W2 = 2;
static constexpr GLuint kBind_B2 = 3;
static constexpr GLuint kBind_W3 = 4;
static constexpr GLuint kBind_B3 = 5;

// ========== viewer implementation ==========
namespace viewer {

using std::uint32_t;

Viewer::Viewer()
    : m_camera(std::make_unique<Camera>()) {}

Viewer::~Viewer() {
    if (m_window != nullptr) {
        if (m_postProg) glDeleteProgram(m_postProg);
        if (m_postVAO)  glDeleteVertexArrays(1, &m_postVAO);

        if (m_prog)     glDeleteProgram(m_prog);
        if (m_vbo_pos)  glDeleteBuffers(1, &m_vbo_pos);
        if (m_vbo_tid)  glDeleteBuffers(1, &m_vbo_tid);
        if (m_vao)      glDeleteVertexArrays(1, &m_vao);

        if (m_triTex[0]) glDeleteTextures(1, &m_triTex[0]);
        if (m_triTex[1]) glDeleteTextures(1, &m_triTex[1]);

        if (m_tex_color[0]) glDeleteTextures(1, &m_tex_color[0]);
        if (m_tex_color[1]) glDeleteTextures(1, &m_tex_color[1]);
        if (m_tex_depth)    glDeleteTextures(1, &m_tex_depth);
        if (m_FBO)          glDeleteFramebuffers(1, &m_FBO);

        if (m_uboW1) glDeleteBuffers(1, &m_uboW1);
        if (m_uboB1) glDeleteBuffers(1, &m_uboB1);
        if (m_uboW2) glDeleteBuffers(1, &m_uboW2);
        if (m_uboB2) glDeleteBuffers(1, &m_uboB2);
        if (m_uboW3) glDeleteBuffers(1, &m_uboW3);
        if (m_uboB3) glDeleteBuffers(1, &m_uboB3);

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
}

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
    vBary = (corner == 0) ? vec3(1.0, 0.0, 0.0) :
            (corner == 1) ? vec3(0.0, 1.0, 0.0) :
                            vec3(0.0, 0.0, 1.0);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* kFS_Geom = R"(
#version 330 core
flat in uint vTriID;
in vec3 vBary;

layout(location=0) out vec4 FragA; // COLOR0
layout(location=1) out vec4 FragB; // COLOR1

uniform ivec2     uTriTexSize;   // shared W,H for both LUTs
uniform sampler2D uTriTex0;      // LUT #0
uniform sampler2D uTriTex1;      // LUT #1
uniform int       uLevel;        // single level to sample (e.g., 5)

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
    int cap  = texW * texH;
    if (cap <= 0) {
        FragA = vec4(1,0,1,1);
        FragB = FragA;
        return;
    }

    int S     = level_size(uLevel);
    int base  = int(vTriID) * S;

    float b0 = vBary.x;
    float b1 = vBary.y;

    int   res   = 1 << uLevel;
    float res_f = float(res);

    float b0l = b0 * res_f;
    float b1l = b1 * res_f;

    int x = int(floor(b0l));
    int y = int(floor(b1l));
    x = clamp(x, 0, res - 1);
    y = clamp(y, 0, (res - 1) - x);

    b0l -= float(x);
    b1l -= float(y);

    bool  flip   = (b0l + b1l) > 1.0;
    int   flip_u = flip ? 1 : 0;
    float flip_f = flip ? 1.0 : 0.0;

    int x0 = x + 1;
    int y0 = y;
    int x1 = x;
    int y1 = y + 1;
    int x2 = x + flip_u;
    int y2 = min(y + flip_u, res - x2);

    int idx0 = (x0 + y0) * (x0 + y0 + 1) / 2 + y0;
    int idx1 = (x1 + y1) * (x1 + y1 + 1) / 2 + y1;
    int idx2 = (x2 + y2) * (x2 + y2 + 1) / 2 + y2;

    float w0 = mix(b0l, 1.0 - b1l, flip_f);
    float w1 = mix(b1l, 1.0 - b0l, flip_f);
    float w2 = 1.0 - w0 - w1;

    int i0 = base + idx0;
    int i1 = base + idx1;
    int i2 = base + idx2;

    ivec2 c0 = idx_to_coord(i0, texW);
    ivec2 c1 = idx_to_coord(i1, texW);
    ivec2 c2 = idx_to_coord(i2, texW);

    vec4 a0 = texelFetch(uTriTex0, c0, 0);
    vec4 a1 = texelFetch(uTriTex0, c1, 0);
    vec4 a2 = texelFetch(uTriTex0, c2, 0);

    vec4 b0s = texelFetch(uTriTex1, c0, 0);
    vec4 b1s = texelFetch(uTriTex1, c1, 0);
    vec4 b2s = texelFetch(uTriTex1, c2, 0);

    vec4 interpA = a0 * w0 + a1 * w1 + a2 * w2;
    vec4 interpB = b0s * w0 + b1s * w1 + b2s * w2;

    if (interpB.a < 0.5) discard; // exact same semantics

    FragA = interpA;
    FragB = vec4(vec3(interpB.rgb), 1.0); // write opaque on FG
}
)";

static const char* kVS_Post = R"(
#version 410 core
const vec2 verts[3] = vec2[3]( vec2(-1,-1), vec2(3,-1), vec2(-1,3) );
out vec2 vUV;
void main() {
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
    vec2 uv = 0.5 * (gl_Position.xy + vec2(1.0));
    vUV = vec2(uv.x, 1.0 - uv.y);
}
)";

// UBO blocks (std140). We index 1D arrays W1[16], B1[4] etc.
// W tiles are mat4 blocks for a 4x4 tiling of the 16x16 matrix.
// Note: W3 is 3x16 -> represented as one block-row of 4 mat4s; only xyz rows used.
static const char* kFS_Post = R"(
#version 410 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D texA;
uniform sampler2D texB;

// NEW: inverse MVP (column-major, same as GLSL mats)
uniform mat4 uInvMVP;
// NEW: encoding frequency
uniform float uEncFreq;

layout(std140) uniform W1Block { mat4 W1[16]; };
layout(std140) uniform B1Block { vec4 B1[4];   };
layout(std140) uniform W2Block { mat4 W2[16]; };
layout(std140) uniform B2Block { vec4 B2[4];   };
layout(std140) uniform W3Block { mat4 W3[4];   };
layout(std140) uniform B3Block { vec4 B3;      };

vec4 relu4(vec4 x){ return max(x, 0.0); }
float sigmoid(float x){ return 1.0/(1.0+exp(-x)); }

// --- helper: NDC->world multiply; GLSL mats are column-major ---
vec3 ndcToWorld(vec4 ndc, mat4 invMVP){
    vec4 clip = invMVP * ndc;
    float inv_w = (abs(clip.w) > 1e-20) ? (1.0 / clip.w) : 0.0;
    return clip.xyz * inv_w;
}

// ---- SH2 constants ----
const float SH_C0    = 0.28209479177387814;  // 1 / (2 * sqrt(pi))
const float SH_C1    = 0.4886025119029199;   // sqrt(3) / (2 * sqrt(pi))
const float SH_C2_0  = 1.0925484305920792;   // sqrt(15) / (2 * sqrt(pi))
const float SH_C2_1  = -1.0925484305920792;  // -sqrt(15) / (2 * sqrt(pi))
const float SH_C2_2  = 0.31539156525252005;  // sqrt(5) / (4 * sqrt(pi))
const float SH_C2_3  = -1.0925484305920792;  // -sqrt(15) / (2 * sqrt(pi))
const float SH_C2_4  = 0.5462742152960396;   // sqrt(15) / (4 * sqrt(pi))

// Evaluate SH basis up to degree 2.
// Order:
//  0: Y_0^0
//  1: Y_1^-1
//  2: Y_1^0
//  3: Y_1^1
//  4..8: degree 2 terms
void eval_sh2(in vec3 d, out float sh[9]) {
    float x = d.x;
    float y = d.y;
    float z = d.z;

    // Degree 0
    sh[0] = SH_C0;

    // Degree 1
    sh[1] = -SH_C1 * y;
    sh[2] =  SH_C1 * z;
    sh[3] = -SH_C1 * x;

    // Degree 2
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;

    sh[4] = SH_C2_0 * xy;
    sh[5] = SH_C2_1 * yz;
    sh[6] = SH_C2_2 * (2.0 * zz - xx - yy);
    sh[7] = SH_C2_3 * xz;
    sh[8] = SH_C2_4 * (xx - yy);
}

void main() {
    vec4 A = texture(texA, vUV);
    vec4 B = texture(texB, vUV);

    // background: if not foreground, pass through A
    if (B.a < 0.5) {
        FragColor = vec4(A.rgb, 1.0);
        return;
    }

    // ---- compute per-pixel view dir (pixel center) ----
    // Our vUV is [0,1]^2 with a Y flip done in VS.
    // Map to NDC:
    float ndc_x = -1.0 + 2.0 * vUV.x;
    float ndc_y = -1.0 + 2.0 * vUV.y;

    // Two points on the ray in NDC
    vec3 world_near = ndcToWorld(vec4(ndc_x, ndc_y, -1.0, 1.0), uInvMVP);
    vec3 world_far  = ndcToWorld(vec4(ndc_x, ndc_y,  1.0, 1.0), uInvMVP);

    // View direction from far -> near (points toward camera)
    vec3 v = normalize(world_near - world_far);

    // ---- SH2 encoding of view dir ----
    float sh[9];
    eval_sh2(v, sh);

    // ---- pack 16-D input ----
    // idx 0..3 : A.rgba
    // idx 4..6 : B.rgb
    // idx 7..15: [vx, vy, vz, sin(w*vx), sin(w*vy), sin(w*vz), cos(w*vx), cos(w*vy), cos(w*vz)]
    // Packed into four vec4s (x0..x3)
    vec4 x0 = vec4(A.r, A.g, A.b, A.a);
    vec4 x1 = vec4(B.r, B.g, B.b, sh[0]);          // dims 4,5,6,7
    vec4 x2 = vec4(sh[1], sh[2], sh[3], sh[4]);    // dims 8,9,10,11
    vec4 x3 = vec4(sh[5], sh[6], sh[7], sh[8]);    // dims 12,13,14,15

    // ---- 16->16 ReLU ----
    vec4 y0 = W1[ 0]*x0 + W1[ 1]*x1 + W1[ 2]*x2 + W1[ 3]*x3 + B1[0];
    vec4 y1 = W1[ 4]*x0 + W1[ 5]*x1 + W1[ 6]*x2 + W1[ 7]*x3 + B1[1];
    vec4 y2 = W1[ 8]*x0 + W1[ 9]*x1 + W1[10]*x2 + W1[11]*x3 + B1[2];
    vec4 y3 = W1[12]*x0 + W1[13]*x1 + W1[14]*x2 + W1[15]*x3 + B1[3];
    y0 = relu4(y0); y1 = relu4(y1); y2 = relu4(y2); y3 = relu4(y3);

    // ---- 16->16 ReLU ----
    vec4 z0 = W2[ 0]*y0 + W2[ 1]*y1 + W2[ 2]*y2 + W2[ 3]*y3 + B2[0];
    vec4 z1 = W2[ 4]*y0 + W2[ 5]*y1 + W2[ 6]*y2 + W2[ 7]*y3 + B2[1];
    vec4 z2 = W2[ 8]*y0 + W2[ 9]*y1 + W2[10]*y2 + W2[11]*y3 + B2[2];
    vec4 z3 = W2[12]*y0 + W2[13]*y1 + W2[14]*y2 + W2[15]*y3 + B2[3];
    z0 = relu4(z0); z1 = relu4(z1); z2 = relu4(z2); z3 = relu4(z3);

    // ---- 16->3 sigmoid ----
    vec4 acc = W3[0]*z0 + W3[1]*z1 + W3[2]*z2 + W3[3]*z3 + B3; // xyz used
    vec3 mlp = vec3(sigmoid(acc.x), sigmoid(acc.y), sigmoid(acc.z));

    // ---- residual blend: (1-residual)*albedo + residual*mlp ----
    float residual = A.a;
    vec3 albedo = A.rgb;
    vec3 out3 = mix(albedo, mlp, residual);

    FragColor = vec4(out3, 1.0);
}
)";

void Viewer::ensure_ubo_alloc() {
    auto ensure = [](GLuint &ubo, GLsizeiptr size, GLuint binding){
        if (!ubo) {
            glGenBuffers(1, &ubo);
            glBindBuffer(GL_UNIFORM_BUFFER, ubo);
            glBufferData(GL_UNIFORM_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_UNIFORM_BUFFER, binding, ubo);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
        }
    };
    
    ensure(m_uboW1, sizeof(float)*16*16, kBind_W1);
    ensure(m_uboB1, sizeof(float)*4*4,   kBind_B1);
    ensure(m_uboW2, sizeof(float)*16*16, kBind_W2);
    ensure(m_uboB2, sizeof(float)*4*4,   kBind_B2);
    ensure(m_uboW3, sizeof(float)*4*16,  kBind_W3);
    ensure(m_uboB3, sizeof(float)*4,     kBind_B3);
}

void Viewer::upload_default_mlp_weights_as_ubos() {
    ensure_ubo_alloc();
    
    // Upload zeros directly WITHOUT calling set_mlp_weights()
    // This prevents overwriting the staged weights from the user
    float zeros_256[16*16] = {0};  // for W1, W2
    float zeros_16[16] = {0};       // for B1, B2  
    float zeros_48[3*16] = {0};     // for W3
    float zeros_4[4] = {0};         // for B3
    
    // Upload W1
    glBindBuffer(GL_UNIFORM_BUFFER, m_uboW1);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(zeros_256), zeros_256);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Upload B1
    glBindBuffer(GL_UNIFORM_BUFFER, m_uboB1);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(zeros_16), zeros_16);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Upload W2
    glBindBuffer(GL_UNIFORM_BUFFER, m_uboW2);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(zeros_256), zeros_256);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Upload B2
    glBindBuffer(GL_UNIFORM_BUFFER, m_uboB2);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(zeros_16), zeros_16);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Upload W3 (64 floats for 4 mat4s)
    float zeros_64[64] = {0};
    glBindBuffer(GL_UNIFORM_BUFFER, m_uboW3);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(zeros_64), zeros_64);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Upload B3
    glBindBuffer(GL_UNIFORM_BUFFER, m_uboB3);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(zeros_4), zeros_4);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Viewer::upload_mlp_weights_from_staging() {
    if (!m_mlp_weights_pending) return;
    
    ensure_ubo_alloc();
    
    // Helper lambdas for uploading
    auto upload_W_tiled = [](GLuint ubo, const float* W, int out_dim, int in_dim) {
        if (!W) return;
        
        float tiles[16 * 16] = {0};
        
        // Weights are row-major: W[row][col] = W[row*in_dim + col]
        // We need column-major mat4 blocks for GLSL
        
        for (int tile_row = 0; tile_row < 4; ++tile_row) {
            for (int tile_col = 0; tile_col < 4; ++tile_col) {
                int tile_idx = tile_row * 4 + tile_col;
                
                // Fill this mat4 in column-major order
                for (int col = 0; col < 4; ++col) {
                    for (int row = 0; row < 4; ++row) {
                        int global_row = tile_row * 4 + row;
                        int global_col = tile_col * 4 + col;
                        
                        if (global_row < out_dim && global_col < in_dim) {
                            // Read from row-major, write to column-major
                            tiles[tile_idx * 16 + col * 4 + row] = W[global_row * in_dim + global_col];
                        }
                    }
                }
            }
        }
        
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(tiles), tiles);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    };
    
    auto upload_B_grouped4 = [](GLuint ubo, const float* B, int dim) {
        if (!B) return;
        float b[16] = {0};
        for (int i = 0; i < dim && i < 16; ++i) {
            b[i] = B[i];
        }
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(b), b);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    };
    
    auto upload_W3 = [](GLuint ubo, const float* W3) {
        if (!W3) return;
        float tiles[4 * 16] = {0};
        
        // W3 is 3x16 row-major, we need 4 mat4s but only use first 3 rows
        for (int tile_col = 0; tile_col < 4; ++tile_col) {
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 3; ++row) {  // Only 3 output dims
                    int global_col = tile_col * 4 + col;
                    if (global_col < 16) {
                        // Read from row-major, write to column-major
                        tiles[tile_col * 16 + col * 4 + row] = W3[row * 16 + global_col];
                    }
                }
            }
        }
        
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(tiles), tiles);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    };
    
    // Now upload from staging
    if (!m_W1_staging.empty()) upload_W_tiled(m_uboW1, m_W1_staging.data(), 16, 16);
    if (!m_b1_staging.empty()) upload_B_grouped4(m_uboB1, m_b1_staging.data(), 16);
    if (!m_W2_staging.empty()) upload_W_tiled(m_uboW2, m_W2_staging.data(), 16, 16);
    if (!m_b2_staging.empty()) upload_B_grouped4(m_uboB2, m_b2_staging.data(), 16);
    if (!m_W3_staging.empty()) upload_W3(m_uboW3, m_W3_staging.data());
    if (!m_b3_staging.empty()) {
        float B3v[4] = { m_b3_staging[0], m_b3_staging[1], m_b3_staging[2], 0.0f };
        glBindBuffer(GL_UNIFORM_BUFFER, m_uboB3);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(B3v), B3v);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
    
    m_mlp_weights_pending = false;
}

void Viewer::set_mlp_weights(
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    float enc_freq
) {
    m_enc_freq = enc_freq;
    
    // Store weights for later upload
    if (W1) m_W1_staging.assign(W1, W1 + 16*16);
    if (b1) m_b1_staging.assign(b1, b1 + 16);
    if (W2) m_W2_staging.assign(W2, W2 + 16*16);
    if (b2) m_b2_staging.assign(b2, b2 + 16);
    if (W3) m_W3_staging.assign(W3, W3 + 3*16);
    if (b3) m_b3_staging.assign(b3, b3 + 3);
    
    m_mlp_weights_pending = true;
    
    // If GL is already initialized, upload immediately
    if (m_started && m_window) {
        upload_mlp_weights_from_staging();
    }
}

void Viewer::set_camera_center(float cen[3]) {
    m_camera->origin[0] = cen[0];
    m_camera->origin[1] = cen[1];
    m_camera->origin[2] = cen[2];
    m_camera->_update();
}

void Viewer::start() {
    if (m_started || !m_window) return;

    // Geometry VAO/VBOs
    GL_CALL(glGenVertexArrays(1, &m_vao));
    GL_CALL(glGenBuffers(1, &m_vbo_pos));
    GL_CALL(glGenBuffers(1, &m_vbo_tid));

    // First pass program
    GLuint vs = compile_shader(GL_VERTEX_SHADER,   kVS_Geom);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFS_Geom);
    m_prog = link_program(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    m_loc_uMVP        = glGetUniformLocation(m_prog, "uMVP");
    m_loc_uTriTex0    = glGetUniformLocation(m_prog, "uTriTex0");
    m_loc_uTriTex1    = glGetUniformLocation(m_prog, "uTriTex1");
    m_loc_uTriTexSize = glGetUniformLocation(m_prog, "uTriTexSize");
    m_loc_uLevel      = glGetUniformLocation(m_prog, "uLevel");

    // Post/composite program
    GLuint pvs = compile_shader(GL_VERTEX_SHADER, kVS_Post);
    GLuint pfs = compile_shader(GL_FRAGMENT_SHADER, kFS_Post);
    m_postProg = link_program(pvs, pfs);
    glDeleteShader(pvs); glDeleteShader(pfs);
    m_postLoc_texA = glGetUniformLocation(m_postProg, "texA");
    m_postLoc_texB = glGetUniformLocation(m_postProg, "texB");
    m_postLoc_uInvMVP  = glGetUniformLocation(m_postProg, "uInvMVP");
    m_postLoc_uEncFreq = glGetUniformLocation(m_postProg, "uEncFreq");
    auto bind_block = [&](const char* name, GLuint binding) {
        GLuint idx = glGetUniformBlockIndex(m_postProg, name);
        if (idx != GL_INVALID_INDEX) {
            glUniformBlockBinding(m_postProg, idx, binding);
        }
    };
    bind_block("W1Block", 0);
    bind_block("B1Block", 1);
    bind_block("W2Block", 2);
    bind_block("B2Block", 3);
    bind_block("W3Block", 4);
    bind_block("B3Block", 5);

    // Post VAO
    GL_CALL(glGenVertexArrays(1, &m_postVAO));

    // Allocate UBOs + upload defaults
    upload_default_mlp_weights_as_ubos();
    upload_mlp_weights_from_staging();

    m_started = true;
}

static void create_or_upload_lut(GLuint& tex, int W, int H, const std::vector<unsigned char>& rgba) {
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Viewer::ensure_luts_uploaded() {
    if (!m_started || !m_window) return;
    for (int i = 0; i < 2; ++i) {
        if (!m_triTexPending[i]) continue;
        if (m_triTexW <= 0 || m_triTexH <= 0) continue;
        if ((int)m_triTexStaging[i].size() != m_triTexW * m_triTexH * 4) continue;

        create_or_upload_lut(m_triTex[i], m_triTexW, m_triTexH, m_triTexStaging[i]);
        m_triTexReady[i]   = true;
        m_triTexPending[i] = false;
        m_triTexStaging[i].clear();
        m_triTexStaging[i].shrink_to_fit();
    }
}

void Viewer::ensure_default_luts() {
    auto make_default = [](int W, int H) {
        std::vector<unsigned char> pix(W*H*4);
        for (int i = 0; i < W*H; ++i) {
            float t = (W*H > 1) ? float(i) / float(W*H - 1) : 0.f;
            pix[4*i+0] = (unsigned char)(255.0f * t);
            pix[4*i+1] = (unsigned char)(255.0f * (1.0f - t));
            pix[4*i+2] = 128;
            pix[4*i+3] = 255;
        }
        return pix;
    };

    const int W = (m_triTexW > 0 ? m_triTexW : 4);
    const int H = (m_triTexH > 0 ? m_triTexH : 4);

    for (int i = 0; i < 2; ++i) {
        if (m_triTexReady[i] || m_triTexPending[i]) continue;
        auto pix = make_default(W, H);
        set_triangle_color_lut(i, W, H, pix.data());
    }
}

void Viewer::ensure_gl_upload() {
    if (!m_need_gl_upload || !m_started) return;
    if (m_pos_dup_host.empty() || m_tid_dup_host.empty()) {
        m_need_gl_upload = false;
        return;
    }

    GL_CALL(glBindVertexArray(m_vao));

    GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, m_vbo_pos));
    GL_CALL(glBufferData(GL_ARRAY_BUFFER,
                         m_pos_dup_host.size() * sizeof(float),
                         m_pos_dup_host.data(),
                         GL_STATIC_DRAW));
    GL_CALL(glEnableVertexAttribArray(0));
    GL_CALL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (void*)0));

    GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, m_vbo_tid));
    GL_CALL(glBufferData(GL_ARRAY_BUFFER,
                         m_tid_dup_host.size() * sizeof(uint32_t),
                         m_tid_dup_host.data(),
                         GL_STATIC_DRAW));
    GL_CALL(glEnableVertexAttribArray(1));
    GL_CALL(glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, sizeof(uint32_t), (void*)0));

    GL_CALL(glBindVertexArray(0));

    m_vertex_count = (GLsizei)(m_pos_dup_host.size() / 3);
    m_need_gl_upload = false;
}

void Viewer::resize(int width, int height) {
    if (width == m_camera->width && height == m_camera->height) return;

    start();
    m_camera->width  = width;
    m_camera->height = height;

    if (!m_FBO) glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    for (int i = 0; i < 2; ++i) {
        if (!m_tex_color[i]) glGenTextures(1, &m_tex_color[i]);
        glBindTexture(GL_TEXTURE_2D, m_tex_color[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, m_tex_color[i], 0);
    }

    if (!m_tex_depth) glGenTextures(1, &m_tex_depth);
    glBindTexture(GL_TEXTURE_2D, m_tex_depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0,
                 GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_tex_depth, 0);

    GLenum draw_buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, draw_buffers);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[viewer] FBO incomplete: 0x" << std::hex << status << std::dec << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    ensure_gl_upload();
}

void Viewer::_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto &cam = *m_camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS) {
        const bool SHIFT = mods & GLFW_MOD_SHIFT;
        cam.begin_drag((float)x, (float)y,
                       SHIFT || button == GLFW_MOUSE_BUTTON_MIDDLE,
                       button == GLFW_MOUSE_BUTTON_RIGHT ||
                       (button == GLFW_MOUSE_BUTTON_MIDDLE && SHIFT));
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void Viewer::_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    (void)window;
    m_camera->drag_update((float)x, (float)y);
}

void Viewer::_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto &cam = *m_camera;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

void Viewer::_window_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    GL_CALL(glViewport(0, 0, width, height));
    resize(width, height);
}

void Viewer::render() {
    start();
    ensure_gl_upload();
    ensure_luts_uploaded();
    ensure_default_luts();

    // ---- pass 0: raster into COLOR0/1 ----
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, m_camera->width, m_camera->height);

    const GLfloat clear0[] = {
        m_options.background_brightness,
        m_options.background_brightness,
        m_options.background_brightness,
        1.0f
    };
    const GLfloat clear1[] = {
        m_options.background_brightness,
        m_options.background_brightness,
        m_options.background_brightness,
        0.0f
    };
    glClearBufferfv(GL_COLOR, 0, clear0);
    glClearBufferfv(GL_COLOR, 1, clear1);
    glClear(GL_DEPTH_BUFFER_BIT);

    // --------------------------------------------------
    // MVP SELECTION LOGIC
    // --------------------------------------------------
    glm::mat4 mvp;
    if (!m_benchmark_mode) {
        // Interactive viewer path (camera-driven)
        m_camera->_update();
        mvp = m_camera->K * m_camera->w2c;
    } else {
        // Benchmark path (external MVP supplied)
        mvp = m_benchmark_mvp;
    }
    // --------------------------------------------------

    if (m_triangles && m_vertex_count > 0 && m_vao && m_options.render_geometry) {
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        glUseProgram(m_prog);
        glUniformMatrix4fv(m_loc_uMVP, 1, GL_FALSE, glm::value_ptr(mvp));

        glUniform1i(m_loc_uTriTex0, 0);
        glUniform1i(m_loc_uTriTex1, 1);
        glUniform2i(m_loc_uTriTexSize, m_triTexW, m_triTexH);
        glUniform1i(m_loc_uLevel, m_level);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_triTex[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_triTex[1]);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLES, 0, m_vertex_count);
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
    }

    // ---- pass 1: composite to screen via UBO-backed MLP ----
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_camera->width, m_camera->height);

    glDisable(GL_DEPTH_TEST);
    glUseProgram(m_postProg);

    glUniform1i(m_postLoc_texA, 0);
    glUniform1i(m_postLoc_texB, 1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_tex_color[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_tex_color[1]);

    // NOTE:
    // invMVP must correspond to the SAME MVP used in geometry pass
    glm::mat4 invMVP = glm::inverse(mvp);
    glUniformMatrix4fv(m_postLoc_uInvMVP, 1, GL_FALSE, glm::value_ptr(invMVP));
    glUniform1f(m_postLoc_uEncFreq, m_enc_freq);

    glBindVertexArray(m_postVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    // ---- UI (disabled during benchmark) ----
    if (!m_benchmark_mode) {
        // draw_gui();
    }
}

void Viewer::launch(int nw, int nh) {
    if (m_window != nullptr) return;

    m_window = ::glfw_init(nw, nh);
    start();

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    resize(width, height);

    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, ::glfw_mouse_button_callback);
    glfwSetCursorPosCallback(m_window, ::glfw_cursor_pos_callback);
    glfwSetScrollCallback(m_window, ::glfw_scroll_callback);
    glfwSetFramebufferSizeCallback(m_window, ::glfw_window_size_callback);

    while (!glfwWindowShouldClose(m_window)) {
        glEnable(GL_DEPTH_TEST);
        ::glfw_update_title(m_window);

        render();

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void Viewer::set_output_dir(std::string output_dir) {
    if (output_dir.empty()) return;
    if (output_dir.back() != '/') output_dir.push_back('/');
    m_output_dir = std::move(output_dir);
}

void Viewer::set_triangles(int V, const float* verts, int F, const int* faces) {
    if (V <= 0 || F <= 0 || !verts || !faces) {
        m_vertex_count = 0;
        m_pos_dup_host.clear();
        m_tid_dup_host.clear();
        m_need_gl_upload = false;
        return;
    }

    m_triangles = std::make_unique<Triangles>(V, verts, F, faces);

    m_pos_dup_host.clear();
    m_tid_dup_host.clear();
    m_pos_dup_host.reserve(size_t(F) * 3 * 3);
    m_tid_dup_host.reserve(size_t(F) * 3);

    for (int f = 0; f < F; ++f) {
        const int i0 = faces[3*f + 0];
        const int i1 = faces[3*f + 1];
        const int i2 = faces[3*f + 2];
        if (i0 < 0 || i0 >= V || i1 < 0 || i1 >= V || i2 < 0 || i2 >= V) continue;

        const float* p0 = &verts[i0*3];
        const float* p1 = &verts[i1*3];
        const float* p2 = &verts[i2*3];

        m_pos_dup_host.insert(m_pos_dup_host.end(), {
            p0[0], p0[1], p0[2],
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2]
        });
        m_tid_dup_host.push_back(uint32_t(f));
        m_tid_dup_host.push_back(uint32_t(f));
        m_tid_dup_host.push_back(uint32_t(f));
    }

    m_vertex_count = (GLsizei)(m_pos_dup_host.size() / 3);
    m_need_gl_upload = true;
}

void Viewer::set_triangle_color_lut(int idx, int width, int height, const unsigned char* rgba) {
    if (idx != 0 && idx != 1) return;
    m_triTexW = width;
    m_triTexH = height;
    m_triTexStaging[idx].assign(rgba, rgba + (size_t)width * height * 4);
    m_triTexPending[idx] = true;

    if (m_started && m_window) ensure_luts_uploaded();
}

void Viewer::draw_gui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(300., 400.f), ImGuiCond_Once);
    ImGui::Begin("GUI");

    ImGui::Text("Window: %dx%d", m_camera->width, m_camera->height);

    if (!m_output_dir.empty()) {
        if (ImGui::Button("save screenshot (ppm)")) {
            const auto output_path = m_output_dir + "screenshot.ppm";
            std::cout << "saving screenshot in " << output_path << std::endl;

            int width = m_camera->width, height = m_camera->height;
            std::vector<unsigned char> window_pixels(4 * width * height);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, window_pixels.data());

            std::vector<unsigned char> flipped_pixels(4 * width * height);
            for (int row=0; row<height; ++row) {
                std::memcpy(&flipped_pixels[row * width * 4],
                            &window_pixels[(height - row - 1) * width * 4], 4 * width);
            }

            write_ppm(output_path.c_str(), width, height, flipped_pixels.data());
        }
    }

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Background Brightness", &m_options.background_brightness, 0.0f, 1.0f);
        ImGui::Checkbox("render_geometry", &m_options.render_geometry);
        ImGui::Text("Tri LUT: %s (%dx%d)",
            (m_triTexReady[0] && m_triTexReady[1]) ? "both set" :
            (m_triTexReady[0] || m_triTexReady[1]) ? "one set" : "default",
            m_triTexW, m_triTexH);
        if (m_triangles) {
            ImGui::Text("Faces: %d", m_triangles->F_);
        }
        static const char* levels[] = { "0", "1", "2", "3", "4", "5" };
        ImGui::Combo("Texture level (DEBUG)", &m_level, levels, IM_ARRAYSIZE(levels));
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Viewer::launch_benchmark(
    int width,
    int height,
    int B,
    const float* mvps,
    int warmup_frames,
    int save_every
) {
    m_window = ::glfw_init(width, height);
    start();
    resize(width, height);

    glfwSetWindowUserPointer(m_window, this);

    std::vector<double> times_ms;
    times_ms.reserve(B);

    // Enable benchmark mode (disables camera MVP path)
    m_benchmark_mode = true;

    // Make sure uploads are ready
    ensure_gl_upload();
    ensure_luts_uploaded();
    ensure_default_luts();
    upload_mlp_weights_from_staging();

    // Resolve output directory prefix
    std::string out_dir = m_output_dir;
    if (!out_dir.empty() && out_dir.back() != '/')
        out_dir.push_back('/');

    // ------------------------------------------------------------
    // Create an OFFSCREEN final-RGB framebuffer at (width,height)
    // ------------------------------------------------------------
    GLuint finalFBO = 0;
    GLuint finalTex = 0;

    GL_CALL(glGenFramebuffers(1, &finalFBO));
    GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, finalFBO));

    GL_CALL(glGenTextures(1, &finalTex));
    GL_CALL(glBindTexture(GL_TEXTURE_2D, finalTex));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

    GL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, finalTex, 0));

    {
        GLenum drawBuf = GL_COLOR_ATTACHMENT0;
        GL_CALL(glDrawBuffers(1, &drawBuf));
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[viewer] finalFBO incomplete: 0x"
                  << std::hex << status << std::dec << std::endl;
        std::exit(EXIT_FAILURE);
    }

    GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    // ------------------------------------------------------------
    // Helper: render one frame using provided MVP
    // ------------------------------------------------------------
    auto render_one = [&](const glm::mat4& mvp) {
        // PASS 0: geometry -> m_FBO
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, m_FBO));
        GL_CALL(glViewport(0, 0, width, height));

        const GLfloat clear0[] = {
            m_options.background_brightness,
            m_options.background_brightness,
            m_options.background_brightness,
            1.0f
        };
        const GLfloat clear1[] = {
            m_options.background_brightness,
            m_options.background_brightness,
            m_options.background_brightness,
            0.0f
        };

        GL_CALL(glClearBufferfv(GL_COLOR, 0, clear0));
        GL_CALL(glClearBufferfv(GL_COLOR, 1, clear1));
        GL_CALL(glClear(GL_DEPTH_BUFFER_BIT));

        if (m_triangles && m_vertex_count > 0 && m_vao && m_options.render_geometry) {
            GL_CALL(glEnable(GL_DEPTH_TEST));
            GL_CALL(glDisable(GL_BLEND));
            GL_CALL(glDisable(GL_CULL_FACE));

            GL_CALL(glUseProgram(m_prog));
            GL_CALL(glUniformMatrix4fv(m_loc_uMVP, 1, GL_FALSE, glm::value_ptr(mvp)));

            GL_CALL(glUniform1i(m_loc_uTriTex0, 0));
            GL_CALL(glUniform1i(m_loc_uTriTex1, 1));
            GL_CALL(glUniform2i(m_loc_uTriTexSize, m_triTexW, m_triTexH));
            GL_CALL(glUniform1i(m_loc_uLevel, m_level));

            GL_CALL(glActiveTexture(GL_TEXTURE0));
            GL_CALL(glBindTexture(GL_TEXTURE_2D, m_triTex[0]));
            GL_CALL(glActiveTexture(GL_TEXTURE1));
            GL_CALL(glBindTexture(GL_TEXTURE_2D, m_triTex[1]));

            GL_CALL(glBindVertexArray(m_vao));
            GL_CALL(glDrawArrays(GL_TRIANGLES, 0, m_vertex_count));
            GL_CALL(glBindVertexArray(0));

            GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
            GL_CALL(glUseProgram(0));
        }

        // PASS 1: post/MLP -> finalFBO
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, finalFBO));
        GL_CALL(glViewport(0, 0, width, height));

        const GLfloat clearFinal[] = {0.f, 0.f, 0.f, 1.f};
        GL_CALL(glClearBufferfv(GL_COLOR, 0, clearFinal));

        GL_CALL(glDisable(GL_DEPTH_TEST));
        GL_CALL(glUseProgram(m_postProg));

        GL_CALL(glUniform1i(m_postLoc_texA, 0));
        GL_CALL(glUniform1i(m_postLoc_texB, 1));

        GL_CALL(glActiveTexture(GL_TEXTURE0));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, m_tex_color[0]));
        GL_CALL(glActiveTexture(GL_TEXTURE1));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, m_tex_color[1]));

        glm::mat4 invMVP = glm::inverse(mvp);
        GL_CALL(glUniformMatrix4fv(m_postLoc_uInvMVP, 1, GL_FALSE, glm::value_ptr(invMVP)));
        GL_CALL(glUniform1f(m_postLoc_uEncFreq, m_enc_freq));

        GL_CALL(glBindVertexArray(m_postVAO));
        GL_CALL(glDrawArrays(GL_TRIANGLES, 0, 3));
        GL_CALL(glBindVertexArray(0));

        GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
        GL_CALL(glUseProgram(0));

        // Blit to default framebuffer for swap
        GL_CALL(glBindFramebuffer(GL_READ_FRAMEBUFFER, finalFBO));
        GL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
        GL_CALL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        GL_CALL(glBlitFramebuffer(
            0, 0, width, height,
            0, 0, width, height,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST
        ));
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    };

    // ---- warmup (NOT timed) ----
    for (int i = 0; i < warmup_frames; ++i) {
        m_benchmark_mvp = glm::make_mat4(mvps);
        render_one(m_benchmark_mvp);
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }

    // ---- timed frames ----
    for (int i = 0; i < B; ++i) {
        m_benchmark_mvp = glm::make_mat4(mvps + i * 16);

        constexpr int kRepeat = 100;

        GL_CALL(glFinish());
        double t0 = glfwGetTime();

        for (int r = 0; r < kRepeat; ++r) {
            render_one(m_benchmark_mvp);
        }

        GL_CALL(glFinish());
        double t1 = glfwGetTime();

        glfwSwapBuffers(m_window);
        glfwPollEvents();

        double ms = ((t1 - t0) * 1000.0) / double(kRepeat);
        times_ms.push_back(ms);

        // ---- screenshot ----
        if (save_every > 0 && (i % save_every == 0) && !out_dir.empty()) {
            std::vector<unsigned char> pixels(4 * width * height);
            std::vector<unsigned char> flipped(4 * width * height);

            GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, finalFBO));
            GL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
            GL_CALL(glPixelStorei(GL_PACK_ALIGNMENT, 1));
            GL_CALL(glReadPixels(
                0, 0, width, height,
                GL_RGBA, GL_UNSIGNED_BYTE,
                pixels.data()));
            GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

            for (int row = 0; row < height; ++row) {
                std::memcpy(
                    &flipped[row * width * 4],
                    &pixels[(height - 1 - row) * width * 4],
                    (size_t)width * 4
                );
            }

            char path[512];
            std::snprintf(
                path, sizeof(path),
                "%sscreenshots/benchmark_%05d.ppm",
                out_dir.c_str(), i
            );
            write_ppm(path, width, height, flipped.data());
        }
    }

    m_benchmark_mode = false;

    // ------------------------------------------------------------
    // Write logs
    // ------------------------------------------------------------
    if (!out_dir.empty()) {
        // Per-frame log
        {
            std::ofstream f(out_dir + "benchmark_frames.txt");
            for (size_t i = 0; i < times_ms.size(); ++i)
                f << i << " " << times_ms[i] << "\n";
        }

        // Summary log
        double sum = 0.0;
        double mn = times_ms[0], mx = times_ms[0];
        for (double t : times_ms) {
            sum += t;
            mn = std::min(mn, t);
            mx = std::max(mx, t);
        }
        double mean_ms = sum / times_ms.size();
        double fps = 1000.0 / mean_ms;

        std::ofstream f(out_dir + "benchmark_summary.txt");
        f << "frames: " << B << "\n";
        f << "mean_ms: " << mean_ms << "\n";
        f << "min_ms:  " << mn << "\n";
        f << "max_ms:  " << mx << "\n";
        f << "fps:     " << fps << "\n";
    }

    // Cleanup
    if (finalTex) GL_CALL(glDeleteTextures(1, &finalTex));
    if (finalFBO) GL_CALL(glDeleteFramebuffers(1, &finalFBO));

    glfwDestroyWindow(m_window);
    glfwTerminate();
    m_window = nullptr;
}

} // namespace viewer

// ---- global callbacks ----
void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    GET_VIEWER(window)->_mouse_button_callback(window, button, action, mods);
}
void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_VIEWER(window)->_cursor_pos_callback(window, x, y);
}
void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    GET_VIEWER(window)->_scroll_callback(window, xoffset, yoffset);
}
void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    GET_VIEWER(window)->_window_size_callback(window, width, height);
}
