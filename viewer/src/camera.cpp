// camera.cpp — Orbit camera implementation.

#include "camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace viewer {

// ---------------------------------------------------------------------------
// Orbit helpers
// ---------------------------------------------------------------------------

/// Build an orthonormal horizontal basis from an arbitrary up vector.
/// Returns (right, forward) where up × right = forward.
/// The sign convention is chosen so that lookAt + this basis give correct
/// trackball drag behaviour for any up direction.
static void horizontal_basis(const glm::vec3& up,
                             glm::vec3& right, glm::vec3& fwd) {
    // Pick a seed axis that isn't parallel to up.
    glm::vec3 seed = (std::abs(up.x) < 0.9f) ? glm::vec3(1,0,0)
                                               : glm::vec3(0,1,0);
    // cross(seed, up) gives a right vector whose orbit tangent is opposite
    // to lookAt's screen-right — exactly what trackball dragging needs.
    right = glm::normalize(glm::cross(seed, up));
    fwd   = glm::cross(up, right);
}

/// Eye position relative to target.
static glm::vec3 orbit_offset(const glm::vec3& up,
                               float yaw, float pitch, float dist) {
    glm::vec3 right, fwd;
    horizontal_basis(up, right, fwd);

    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cy = std::cos(yaw),   sy = std::sin(yaw);

    return dist * (cp * (cy * fwd + sy * right) + sp * up);
}

// ---------------------------------------------------------------------------
// OrbitCamera
// ---------------------------------------------------------------------------

OrbitCamera::OrbitCamera(int width, int height, glm::vec3 up)
    : world_up(glm::normalize(up)), width(width), height(height) {
    update();
}

void OrbitCamera::begin_drag(float x, float y, bool pan) {
    m_dragging    = true;
    m_panning     = pan;
    m_drag_x      = x;
    m_drag_y      = y;
    m_drag_yaw    = yaw;
    m_drag_pitch  = pitch;
    m_drag_target = target;
}

void OrbitCamera::drag_update(float x, float y) {
    if (!m_dragging) return;

    const float dx = x - m_drag_x;
    const float dy = y - m_drag_y;

    if (m_panning) {
        const float speed = kPanSpeed * distance;
        const glm::vec3 right{m_view[0][0], m_view[1][0], m_view[2][0]};
        const glm::vec3 up   {m_view[0][1], m_view[1][1], m_view[2][1]};
        target = m_drag_target + right * dx * speed - up * dy * speed;
    } else {
        yaw   = m_drag_yaw   + dx * kOrbitSpeed;
        pitch = m_drag_pitch + dy * kOrbitSpeed;
        constexpr float kLimit = glm::radians(89.f);
        pitch = std::clamp(pitch, -kLimit, kLimit);
    }

    update();
}

void OrbitCamera::end_drag() {
    m_dragging = false;
}

void OrbitCamera::scroll(float delta) {
    distance *= (delta > 0.f) ? (1.f - kScrollStep) : (1.f + kScrollStep);
    distance  = std::max(distance, 0.01f);
    update();
}

void OrbitCamera::update() {
    m_eye  = target + orbit_offset(world_up, yaw, pitch, distance);
    m_view = glm::lookAt(m_eye, target, world_up);

    const float aspect = (height > 0) ? float(width) / float(height) : 1.f;
    m_proj = glm::perspective(glm::radians(fov_y_deg), aspect, near_clip, far_clip);
    m_mvp  = m_proj * m_view;
}

}  // namespace viewer
