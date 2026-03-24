// camera.h — Orbit camera with configurable up direction.

#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace viewer {

/// Orbit camera.  Pass any unit vector as world_up.
///
/// Controls:
///   Left-drag   → orbit  (rotate yaw / pitch)
///   Right-drag  → pan    (shift target in screen plane)
///   Scroll      → dolly  (change distance to target)
class OrbitCamera {
public:
    explicit OrbitCamera(int width = 800, int height = 800,
                         glm::vec3 up = {0.f, 0.f, 1.f});

    void begin_drag(float x, float y, bool pan);
    void drag_update(float x, float y);
    void end_drag();
    void scroll(float delta);
    void update();

    const glm::mat4& mvp()  const { return m_mvp; }
    const glm::mat4& view() const { return m_view; }
    const glm::mat4& proj() const { return m_proj; }
    const glm::vec3& eye()  const { return m_eye; }

    // ---- Tunable state (public for GUI) ------------------------------------

    glm::vec3 target{0.f};
    glm::vec3 world_up;       ///< Unit vector pointing "up" in world space.
    float distance  = 5.f;
    float yaw       = 0.f;
    float pitch     = -0.35f;
    float fov_y_deg = 40.f;
    float near_clip = 2.0f;
    float far_clip  = 100.0f;

    int width, height;

private:
    glm::mat4 m_view{1.f};
    glm::mat4 m_proj{1.f};
    glm::mat4 m_mvp{1.f};
    glm::vec3 m_eye{0.f};

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

}  // namespace viewer
