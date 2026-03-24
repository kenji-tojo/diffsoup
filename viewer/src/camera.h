// camera.h — Orbit camera with configurable up-axis.

#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace viewer {

/// World-space up direction.  Choose based on your dataset convention:
///   POS_Z  — NeRF-synthetic (Blender-exported, after OBJ swizzle)
///   NEG_Y  — COLMAP / MipNeRF-360 scenes
///   POS_Y  — some OpenGL / game-engine data
///   NEG_Z  — rare, but supported
enum class UpAxis { POS_Y, NEG_Y, POS_Z, NEG_Z };

/// A simple orbit camera that rotates around a target point.
///
/// Controls:
///   Left-drag   → orbit  (rotate yaw / pitch)
///   Right-drag  → pan    (shift target in screen plane)
///   Scroll      → dolly  (change distance to target)
class OrbitCamera {
public:
    explicit OrbitCamera(int width = 800, int height = 800,
                         UpAxis up = UpAxis::NEG_Y);

    // ---- Interaction -------------------------------------------------------

    void begin_drag(float x, float y, bool pan);
    void drag_update(float x, float y);
    void end_drag();
    void scroll(float delta);

    // ---- Matrices ----------------------------------------------------------

    void update();

    const glm::mat4& mvp()  const { return m_mvp; }
    const glm::mat4& view() const { return m_view; }
    const glm::mat4& proj() const { return m_proj; }

    /// World-space eye position (recomputed on update()).
    const glm::vec3& eye() const { return m_eye; }

    // ---- Tunable state (public for GUI sliders) ----------------------------

    glm::vec3 target{0.f};
    float distance  = 5.f;
    float yaw       = 0.f;       ///< Horizontal angle (radians)
    float pitch     = -0.35f;    ///< Vertical angle (radians); >0 = above target
    float fov_y_deg = 40.f;
    float near_clip = 0.01f;
    float far_clip  = 100.f;
    UpAxis up_axis;

    int width, height;

private:
    glm::mat4 m_view{1.f};
    glm::mat4 m_proj{1.f};
    glm::mat4 m_mvp{1.f};
    glm::vec3 m_eye{0.f};

    // Drag bookkeeping
    bool  m_dragging   = false;
    bool  m_panning    = false;
    float m_drag_x     = 0.f;
    float m_drag_y     = 0.f;
    float m_drag_yaw   = 0.f;
    float m_drag_pitch = 0.f;
    glm::vec3 m_drag_target{0.f};

    static constexpr float kOrbitSpeed = 0.005f;
    static constexpr float kPanSpeed   = 0.003f;
    static constexpr float kScrollStep = 0.1f;
};

/// Parse a string like "pos_y", "neg_z", "+Y", "-Z", "y", "z" into UpAxis.
/// Returns false if the string is not recognised.
bool parse_up_axis(const char* str, UpAxis& out);

/// Human-readable label (e.g. "+Y", "−Z").
const char* up_axis_label(UpAxis a);

}  // namespace viewer
