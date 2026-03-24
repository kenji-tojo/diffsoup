// camera.cpp — Orbit camera implementation.

#include "camera.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>

namespace viewer {

// ---------------------------------------------------------------------------
// Up-axis helpers
// ---------------------------------------------------------------------------

static glm::vec3 up_vector(UpAxis a) {
    switch (a) {
        case UpAxis::POS_Y: return { 0.f,  1.f,  0.f};
        case UpAxis::NEG_Y: return { 0.f, -1.f,  0.f};
        case UpAxis::POS_Z: return { 0.f,  0.f,  1.f};
        case UpAxis::NEG_Z: return { 0.f,  0.f, -1.f};
    }
    return {0.f, 1.f, 0.f};
}

/// Compute eye offset from target in world space.
/// yaw rotates horizontally (around the up axis).
/// pitch lifts the eye above (+) or below (−) the horizontal plane.
static glm::vec3 orbit_offset(UpAxis a, float yaw, float pitch, float dist) {
    const float cp = std::cos(pitch), sp = std::sin(pitch);

    // For positive up-axes the cross product in lookAt flips the screen-right
    // direction relative to the yaw rotation.  Negate yaw for those so that
    // "drag right → scene rotates right" holds for every up-axis.
    const float y = (a == UpAxis::POS_Y || a == UpAxis::POS_Z) ? -yaw : yaw;
    const float cy = std::cos(y), sy = std::sin(y);

    switch (a) {
        case UpAxis::POS_Y:
            return dist * glm::vec3(cp * sy, sp, cp * cy);
        case UpAxis::NEG_Y:
            return dist * glm::vec3(cp * sy, -sp, cp * cy);
        case UpAxis::POS_Z:
            return dist * glm::vec3(cp * cy, cp * sy, sp);
        case UpAxis::NEG_Z:
            return dist * glm::vec3(cp * cy, cp * sy, -sp);
    }
    return {};
}

bool parse_up_axis(const char* s, UpAxis& out) {
    if (!s) return false;
    // Skip leading whitespace / sign noise
    auto eq = [](const char* a, const char* b) {
        while (*a && *b) {
            char ca = *a >= 'A' && *a <= 'Z' ? (*a + 32) : *a;
            char cb = *b >= 'A' && *b <= 'Z' ? (*b + 32) : *b;
            if (ca != cb) return false;
            ++a; ++b;
        }
        return *a == 0 && *b == 0;
    };
    if (eq(s, "pos_y") || eq(s, "+y") || eq(s, "y"))  { out = UpAxis::POS_Y; return true; }
    if (eq(s, "neg_y") || eq(s, "-y"))                 { out = UpAxis::NEG_Y; return true; }
    if (eq(s, "pos_z") || eq(s, "+z") || eq(s, "z"))  { out = UpAxis::POS_Z; return true; }
    if (eq(s, "neg_z") || eq(s, "-z"))                 { out = UpAxis::NEG_Z; return true; }
    return false;
}

const char* up_axis_label(UpAxis a) {
    switch (a) {
        case UpAxis::POS_Y: return "+Y";
        case UpAxis::NEG_Y: return "-Y";
        case UpAxis::POS_Z: return "+Z";
        case UpAxis::NEG_Z: return "-Z";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// OrbitCamera
// ---------------------------------------------------------------------------

OrbitCamera::OrbitCamera(int width, int height, UpAxis up)
    : up_axis(up), width(width), height(height) {
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
        // Pan along camera-local right and up.
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
    m_eye  = target + orbit_offset(up_axis, yaw, pitch, distance);
    m_view = glm::lookAt(m_eye, target, up_vector(up_axis));

    const float aspect = (height > 0) ? float(width) / float(height) : 1.f;
    m_proj = glm::perspective(glm::radians(fov_y_deg), aspect, near_clip, far_clip);
    m_mvp  = m_proj * m_view;
}

}  // namespace viewer
