// camera.cpp — Orbit camera implementation.

#include "camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace viewer {

OrbitCamera::OrbitCamera(int width, int height)
    : width(width), height(height) {
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
        // Pan in the camera's local right / up directions.
        const float speed = kPanSpeed * distance;
        const glm::vec3 right{m_view[0][0], m_view[1][0], m_view[2][0]};
        const glm::vec3 up   {m_view[0][1], m_view[1][1], m_view[2][1]};
        target = m_drag_target - right * dx * speed + up * dy * speed;
    } else {
        // Orbit.
        yaw   = m_drag_yaw   + dx * kOrbitSpeed;
        pitch = m_drag_pitch - dy * kOrbitSpeed;

        // Clamp pitch to avoid gimbal flip.
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
    // Model convention: -Z is up.
    // pitch > 0 → eye moves in -Z direction (above the model).
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cy = std::cos(yaw),   sy = std::sin(yaw);

    const glm::vec3 eye{
        target.x + distance * cp * cy,
        target.y + distance * cp * sy,
        target.z - distance * sp,
    };

    // Build view matrix manually (avoids lookAt handedness flip with -Z up).
    const glm::vec3 world_up{0.f, 0.f, -1.f};
    glm::vec3 fwd   = glm::normalize(target - eye);
    glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));
    glm::vec3 up    = glm::cross(right, fwd);

    m_view = glm::mat4(1.f);
    m_view[0][0] =  right.x; m_view[1][0] =  right.y; m_view[2][0] =  right.z;
    m_view[0][1] =  up.x;    m_view[1][1] =  up.y;    m_view[2][1] =  up.z;
    m_view[0][2] = -fwd.x;   m_view[1][2] = -fwd.y;   m_view[2][2] = -fwd.z;
    m_view[3][0] = -glm::dot(right, eye);
    m_view[3][1] = -glm::dot(up, eye);
    m_view[3][2] =  glm::dot(fwd, eye);

    const float aspect = (height > 0) ? float(width) / float(height) : 1.f;
    m_proj = glm::perspective(glm::radians(fov_y_deg), aspect, near_clip, far_clip);
    m_mvp  = m_proj * m_view;
}

}  // namespace viewer
