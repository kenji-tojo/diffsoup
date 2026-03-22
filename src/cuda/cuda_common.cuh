/**
 * @file cuda_common.cuh
 * @brief Essential CUDA utilities and definitions
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <cuda_runtime.h>

#define CUDA_THREADS 256
#define CUDA_BLOCKS(num) (((num) + CUDA_THREADS - 1) / CUDA_THREADS)

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef CLAMP
#define CLAMP(x, lo, hi) MAX(lo, MIN(hi, x))
#endif

// Mathematical constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Math constants
template<typename T>
__device__ __host__ constexpr T pi() {
    return static_cast<T>(M_PI);
}

template<typename T>
__device__ __host__ constexpr T two_pi() { return 2 * pi<T>(); }

template<typename T>
__device__ __host__ constexpr T inv_pi() { return 1 / pi<T>(); }

// Utility functions  
template<typename T>
__device__ __host__ constexpr T clamp(T x, T lo, T hi) { 
    return fmax(lo, fmin(hi, x)); 
}

template<typename T>
__device__ __host__ constexpr T lerp(T a, T b, T t) { 
    return a + t * (b - a); 
}

// Memory helpers
template<typename T>
T* malloc_gpu(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
void free_gpu(T* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

namespace {

// Forward: sigmoid(x) = 1 / (1 + exp(-x))
__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Backward: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
__device__ inline float sigmoid_grad(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

} // namespace
