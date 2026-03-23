// camera.h — Orbit camera with standard game-engine-style controls.

#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace viewer {

/// A simple orbit camera that rotates around a target point.
///
/// Controls:
///   Left-drag   → orbit  (rotate yaw / pitch)
///   Right-drag  → pan    (shift target in screen plane)
///   Scroll      → dolly  (change distance to target)
class OrbitCamera {
public:
    explicit OrbitCamera(int width = 800, int height = 800);

    // ---- Interaction -------------------------------------------------------

    /// Call on mouse-button press.  `pan` = true for right-drag panning.
    void begin_drag(float x, float y, bool pan);

    /// Call every cursor-move while dragging.
    void drag_update(float x, float y);

    /// Call on mouse-button release.
    void end_drag();

    /// Dolly in / out by the signed scroll amount.
    void scroll(float delta);

    // ---- Matrices ----------------------------------------------------------

    /// Recalculates view and projection matrices from current parameters.
    void update();

    /// The combined projection × view matrix (ready for uniforms).
    const glm::mat4& mvp()  const { return m_mvp; }
    const glm::mat4& view() const { return m_view; }
    const glm::mat4& proj() const { return m_proj; }

    // ---- Tunable state (public for GUI sliders etc.) -----------------------

    glm::vec3 target{0.f};   ///< World-space orbit center
    float distance  = 5.f;   ///< Distance from target
    float yaw       = 0.f;   ///< Horizontal angle (radians)
    float pitch     = 0.f;   ///< Vertical angle (radians), clamped to ±89°
    float fov_y_deg = 40.f;  ///< Vertical field of view (degrees)
    float near_clip = 0.01f;
    float far_clip  = 100.f;

    int width, height;

private:
    glm::mat4 m_view{1.f};
    glm::mat4 m_proj{1.f};
    glm::mat4 m_mvp{1.f};

    // Drag bookkeeping
    bool  m_dragging   = false;
    bool  m_panning    = false;
    float m_drag_x     = 0.f;
    float m_drag_y     = 0.f;
    float m_drag_yaw   = 0.f;
    float m_drag_pitch = 0.f;
    glm::vec3 m_drag_target{0.f};

    // Sensitivity
    static constexpr float kOrbitSpeed = 0.005f;
    static constexpr float kPanSpeed   = 0.003f;
    static constexpr float kScrollStep = 0.1f;
};

}  // namespace viewer
