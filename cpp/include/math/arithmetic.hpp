#pragma once

#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <iostream>
#include <atomic>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef KVTENSOR_USE_ONEMKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>
#endif

#include <dnnl.hpp>
#include <dnnl_version.h>

// SIMD intrinsics for F16C (fast float16 conversion)
#ifdef __F16C__
#include <immintrin.h>
#define KVTENSOR_USE_F16C 1
#elif defined(__AVX2__) || defined(__AVX__)
// Try to use F16C if available (runtime check needed)
#include <immintrin.h>
#define KVTENSOR_USE_F16C 1
#else
#define KVTENSOR_USE_F16C 0
#endif

// Include DType definition (needed for convert_buffer_to_float32)
#include "kvtensor/types.hpp"

namespace kvtensor {
namespace math {

// Dispatch backend
enum class Backend { Reference, OneDNN };

inline Backend& backend_ref() {
    static Backend backend = Backend::OneDNN;
    return backend;
}

inline void set_backend(Backend backend) {
    backend_ref() = backend;
}

inline Backend get_backend() {
    return backend_ref();
}

inline void require_onednn(const char* op_name) {
    if (get_backend() != Backend::OneDNN) {
        throw std::runtime_error(std::string("oneDNN required for op: ") + op_name);
    }
}

// Arithmetic operations wrapper
// This is a placeholder implementation that can be replaced with optimized libraries
// (BLAS, ACL, MKL, etc.)

enum class Transpose { No, Yes };

template <typename T>
constexpr bool is_supported_onednn() {
    return std::is_same<T, float>::value;
}

inline dnnl::engine& onednn_engine() {
    static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
}

inline dnnl::stream& onednn_stream() {
    static dnnl::stream stream(onednn_engine());
    return stream;
}

inline dnnl::reduction::primitive_desc onednn_reduction_pd(
    dnnl::algorithm alg,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
) {
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
    return dnnl::reduction::primitive_desc(
        onednn_engine(),
        alg,
        src_md,
        dst_md,
        0.0f,
        0.0f,
        dnnl::primitive_attr(),
        false
    );
#else
    auto desc = dnnl::reduction::desc(alg, src_md, dst_md, 0.0f, 0.0f);
    return dnnl::reduction::primitive_desc(desc, onednn_engine());
#endif
}

inline dnnl::softmax_forward::primitive_desc onednn_softmax_pd(
    const dnnl::memory::desc& md,
    int axis
) {
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
    return dnnl::softmax_forward::primitive_desc(
        onednn_engine(),
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::softmax_accurate,
        md,
        md,
        axis,
        dnnl::primitive_attr(),
        false
    );
#else
    auto desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference, md, axis);
    return dnnl::softmax_forward::primitive_desc(desc, onednn_engine());
#endif
}

inline void onednn_matmul_impl(
    const void* A,
    const void* B,
    void* C,
    int64_t m,
    int64_t k,
    int64_t n,
    const dnnl::memory::dims& a_strides,
    const dnnl::memory::dims& b_strides,
    const dnnl::memory::dims& c_strides,
    dnnl::memory::data_type a_dt,
    dnnl::memory::data_type b_dt,
    dnnl::memory::data_type c_dt,
    float output_scale = 1.0f
) {
    auto& engine = onednn_engine();
    auto& stream = onednn_stream();

    dnnl::memory::dims a_dims = {m, k};
    dnnl::memory::dims b_dims = {k, n};
    dnnl::memory::dims c_dims = {m, n};

    auto a_md = dnnl::memory::desc(a_dims, a_dt, a_strides);
    auto b_md = dnnl::memory::desc(b_dims, b_dt, b_strides);
    auto c_md = dnnl::memory::desc(c_dims, c_dt, c_strides);

#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
    auto pd = dnnl::matmul::primitive_desc(
        engine,
        a_md,
        b_md,
        c_md,
        dnnl::primitive_attr(),
        false
    );
#else
    auto desc = dnnl::matmul::desc(a_md, b_md, c_md);
    auto pd = dnnl::matmul::primitive_desc(desc, engine);
#endif

    dnnl::matmul(pd).execute(stream, {
        {DNNL_ARG_SRC, dnnl::memory(a_md, engine, const_cast<void*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::memory(b_md, engine, const_cast<void*>(B))},
        {DNNL_ARG_DST, dnnl::memory(c_md, engine, C)}
    });
    stream.wait();

    if (output_scale != 1.0f && c_dt == dnnl::memory::data_type::f32) {
        auto* out = static_cast<float*>(C);
        int64_t row_stride = n;
        if (c_strides.size() >= 1) {
            row_stride = static_cast<int64_t>(c_strides[0]);
        }
        for (int64_t i = 0; i < m; ++i) {
            float* row = out + i * row_stride;
            for (int64_t j = 0; j < n; ++j) {
                row[j] *= output_scale;
            }
        }
    }
}

inline void onednn_eltwise_impl(
    const float* A,
    int64_t size,
    float* B,
    dnnl::algorithm alg,
    float alpha = 0.0f,
    float beta = 0.0f
) {
    auto& engine = onednn_engine();
    auto& stream = onednn_stream();
    dnnl::memory::dims dims = {size};
    auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
    auto pd = dnnl::eltwise_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        alg,
        md,
        md,
        alpha,
        beta,
        dnnl::primitive_attr(),
        false
    );
#else
    auto desc = dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_inference,
        alg,
        md,
        alpha,
        beta
    );
    auto pd = dnnl::eltwise_forward::primitive_desc(desc, engine);
#endif
    auto src_mem = dnnl::memory(md, engine, const_cast<float*>(A));
    auto dst_mem = dnnl::memory(md, engine, B);
    dnnl::eltwise_forward(pd).execute(stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    stream.wait();
}

// BFloat16 helpers (bfloat16 stored in uint16_t)
// BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
// It's essentially the top 16 bits of a float32
inline float bf16_to_float(uint16_t bf) {
    // BFloat16 to float32: pad lower 16 mantissa bits with zeros
    uint32_t f = (static_cast<uint32_t>(bf) << 16);
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

// Fast SIMD-optimized float to bfloat16 conversion
// BFloat16 is just the top 16 bits of float32, so we can use SIMD efficiently
#ifdef __AVX2__
inline void f32_to_bf16_avx_simd(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    
    // Process 8 floats per loop using AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        // Extract top 16 bits: shift right by 16, then pack to 16-bit
        __m256i shifted = _mm256_srli_epi32(_mm256_castps_si256(x), 16);
        // Pack 32-bit integers to 16-bit
        __m128i lo = _mm256_extracti128_si256(shifted, 0);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
    }
    
    // Tail: scalar fallback
    for (; i < n; ++i) {
        uint32_t f;
        std::memcpy(&f, &src[i], sizeof(float));
        dst[i] = static_cast<uint16_t>(f >> 16);
    }
}
#else
// Scalar version for non-AVX2 systems
inline void f32_to_bf16_scalar_loop(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t f;
        std::memcpy(&f, &src[i], sizeof(float));
        dst[i] = static_cast<uint16_t>(f >> 16);
    }
}
#endif

// Single element conversion
inline uint16_t float_to_bf16(float in) {
    uint32_t f;
    std::memcpy(&f, &in, sizeof(float));
    return static_cast<uint16_t>(f >> 16);
}

template <typename T>
inline float to_float_scalar(T v) {
    return static_cast<float>(v);
}

template <>
inline float to_float_scalar<uint16_t>(uint16_t v) {
    return bf16_to_float(v);
}

template <typename T>
inline T from_float_scalar(float v) {
    return static_cast<T>(v);
}

template <>
inline uint16_t from_float_scalar<uint16_t>(float v) {
    return float_to_bf16(v);
}

template <typename T>
inline void to_float_array(const T* in, int64_t size, float* out) {
    for (int64_t i = 0; i < size; ++i) {
        out[i] = to_float_scalar(in[i]);
    }
}

template <typename T>
inline void from_float_array(const float* in, int64_t size, T* out) {
    for (int64_t i = 0; i < size; ++i) {
        out[i] = from_float_scalar<T>(in[i]);
    }
}

// Matrix multiplication: C = A @ B
// A: (m, k), B: (k, n), C: (m, n)
template<typename T>
void matmul(
    const T* A, int64_t m, int64_t k,
    const T* B, int64_t n,
    T* C
) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("matmul");
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            std::cout << "[Matmul] float32 -> oneDNN matmul" << std::endl;
        }
        dnnl::memory::dims a_strides = {k, 1};
        dnnl::memory::dims b_strides = {n, 1};
        dnnl::memory::dims c_strides = {n, 1};
        onednn_matmul_impl(
            A, B, C,
            m, k, n,
            a_strides, b_strides, c_strides,
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::f32
        );
        return;
    } else {
        std::vector<float> Af(static_cast<size_t>(m * k));
        std::vector<float> Bf(static_cast<size_t>(k * n));
        std::vector<float> Cf(static_cast<size_t>(m * n));
        to_float_array(A, m * k, Af.data());
        to_float_array(B, k * n, Bf.data());
        matmul<float>(Af.data(), m, k, Bf.data(), n, Cf.data());
        from_float_array(Cf.data(), m * n, C);
    }
}

// Matrix multiplication with strided B (row-major)
// A: (m, k) contiguous, B: (k, n) with row stride ldb, C: (m, n) contiguous
template<typename T>
void matmul_strided(
    const T* A, int64_t m, int64_t k,
    const T* B, int64_t n, int64_t ldb,
    T* C
) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("matmul_strided");
        dnnl::memory::dims a_strides = {k, 1};
        dnnl::memory::dims b_strides = {ldb, 1};
        dnnl::memory::dims c_strides = {n, 1};
        onednn_matmul_impl(
            A, B, C,
            m, k, n,
            a_strides, b_strides, c_strides,
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::f32
        );
        return;
    } else {
        std::vector<float> Af(static_cast<size_t>(m * k));
        std::vector<float> Bf(static_cast<size_t>(k * n));
        std::vector<float> Cf(static_cast<size_t>(m * n));
        to_float_array(A, m * k, Af.data());
        for (int64_t l = 0; l < k; ++l) {
            for (int64_t j = 0; j < n; ++j) {
                Bf[l * n + j] = to_float_scalar(B[l * ldb + j]);
            }
        }
        matmul<float>(Af.data(), m, k, Bf.data(), n, Cf.data());
        from_float_array(Cf.data(), m * n, C);
    }
}

// Matrix multiplication with transpose flags
// A: (m, k) or (k, m) if trans_a, B: (k, n) or (n, k) if trans_b
template<typename T>
void matmul_ex(
    const T* A, int64_t m, int64_t k,
    const T* B, int64_t n,
    T* C,
    Transpose trans_a,
    Transpose trans_b
) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("matmul_ex");
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            std::cout << "[Matmul] float32_ex -> oneDNN matmul" << std::endl;
        }
        dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
            ? dnnl::memory::dims{1, m}
            : dnnl::memory::dims{k, 1};
        dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
            ? dnnl::memory::dims{1, k}
            : dnnl::memory::dims{n, 1};
        dnnl::memory::dims c_strides = {n, 1};
        onednn_matmul_impl(
            A, B, C,
            m, k, n,
            a_strides, b_strides, c_strides,
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::f32
        );
        return;
    } else {
        std::vector<float> Af(static_cast<size_t>(m * k));
        std::vector<float> Bf(static_cast<size_t>(k * n));
        std::vector<float> Cf(static_cast<size_t>(m * n));
        to_float_array(A, m * k, Af.data());
        to_float_array(B, k * n, Bf.data());
        matmul_ex<float>(Af.data(), m, k, Bf.data(), n, Cf.data(), trans_a, trans_b);
        from_float_array(Cf.data(), m * n, C);
    }
}

// MKL type definitions for low-bit precision
#ifdef KVTENSOR_USE_ONEMKL
#ifndef MKL_BF16
typedef uint16_t MKL_BF16;
#endif
#ifndef MKL_INT32
typedef int32_t MKL_INT32;
#endif
#endif

// INT8 quantization/dequantization helpers
// Quantization: q = round(f / scale) + zero_point
// Dequantization: f = (q - zero_point) * scale
// For symmetric quantization (zero_point = 0): q = round(f / scale), f = q * scale

// Compute quantization parameters from float32 array
// Returns scale for symmetric quantization (zero_point = 0)
inline float compute_quantization_scale(const float* data, size_t n, float min_val = -128.0f, float max_val = 127.0f) {
    if (n == 0) return 1.0f;
    
    float data_min = data[0];
    float data_max = data[0];
    for (size_t i = 1; i < n; ++i) {
        data_min = std::min(data_min, data[i]);
        data_max = std::max(data_max, data[i]);
    }
    
    // Scale to fit in [min_val, max_val] range
    float range = data_max - data_min;
    if (range == 0.0f) return 1.0f;
    
    float scale = range / (max_val - min_val);
    return scale;
}

// Convert float32 to int8 with quantization (symmetric, zero_point = 0)
inline int8_t float_to_int8(float f, float scale) {
    float quantized = f / scale;
    quantized = std::round(quantized);
    quantized = std::max(-128.0f, std::min(127.0f, quantized));
    return static_cast<int8_t>(quantized);
}

// Convert int32 to float32 with dequantization
inline float int32_to_float(int32_t q, float scale) {
    return static_cast<float>(q) * scale;
}

// BFloat16 GEMM: C = A @ B (A and B are bfloat16, C is float32)
inline void matmul_bf16bf16f32(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    float* C
) {
    require_onednn("matmul_bf16bf16f32");
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::cout << "[Matmul] bf16 -> oneDNN matmul" << std::endl;
    }
    dnnl::memory::dims a_strides = {k, 1};
    dnnl::memory::dims b_strides = {n, 1};
    dnnl::memory::dims c_strides = {n, 1};
    onednn_matmul_impl(
        A, B, C,
        m, k, n,
        a_strides, b_strides, c_strides,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::f32
    );
    return;
}

// BFloat16 GEMM: C = A @ B (A and B are bfloat16, C is bfloat16)
inline void matmul_bf16bf16bf16(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    uint16_t* C
) {
    require_onednn("matmul_bf16bf16bf16");
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::cout << "[Matmul] bf16 -> oneDNN matmul (bf16 output)" << std::endl;
    }
    dnnl::memory::dims a_strides = {k, 1};
    dnnl::memory::dims b_strides = {n, 1};
    dnnl::memory::dims c_strides = {n, 1};
    onednn_matmul_impl(
        A, B, C,
        m, k, n,
        a_strides, b_strides, c_strides,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16
    );
    return;
}

// BFloat16 GEMM with transpose support
inline void matmul_ex_bf16bf16f32(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    float* C,
    Transpose trans_a,
    Transpose trans_b
) {
    require_onednn("matmul_ex_bf16bf16f32");
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::cout << "[Matmul] bf16_ex -> oneDNN matmul" << std::endl;
    }
    dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
        ? dnnl::memory::dims{1, m}
        : dnnl::memory::dims{k, 1};
    dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
        ? dnnl::memory::dims{1, k}
        : dnnl::memory::dims{n, 1};
    dnnl::memory::dims c_strides = {n, 1};
    onednn_matmul_impl(
        A, B, C,
        m, k, n,
        a_strides, b_strides, c_strides,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::f32
    );
    return;
}

// BFloat16 GEMM with transpose support (bf16 output)
inline void matmul_ex_bf16bf16bf16(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    uint16_t* C,
    Transpose trans_a,
    Transpose trans_b
) {
    require_onednn("matmul_ex_bf16bf16bf16");
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::cout << "[Matmul] bf16_ex -> oneDNN matmul (bf16 output)" << std::endl;
    }
    dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
        ? dnnl::memory::dims{1, m}
        : dnnl::memory::dims{k, 1};
    dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
        ? dnnl::memory::dims{1, k}
        : dnnl::memory::dims{n, 1};
    dnnl::memory::dims c_strides = {n, 1};
    onednn_matmul_impl(
        A, B, C,
        m, k, n,
        a_strides, b_strides, c_strides,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16
    );
    return;
}

// INT8 GEMM: C = A @ B (A is int8, B is uint8, C is int32, then dequantized to float32)
// Uses cblas_gemm_s8u8s32 with symmetric quantization (zero_point = 0)
// For row-major layout: A is int8, B is uint8
inline void matmul_int8_int8_f32(
    const int8_t* A, int64_t m, int64_t k,
    const uint8_t* B, int64_t n,
    float* C,
    float scale_a,
    float scale_b,
    float scale_c = 1.0f
) {
    require_onednn("matmul_int8_int8_f32");
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::cout << "[Matmul] int8 -> oneDNN matmul" << std::endl;
    }
    float combined_scale = (scale_a * scale_b) / scale_c;
    dnnl::memory::dims a_strides = {k, 1};
    dnnl::memory::dims b_strides = {n, 1};
    dnnl::memory::dims c_strides = {n, 1};
    onednn_matmul_impl(
        A, B, C,
        m, k, n,
        a_strides, b_strides, c_strides,
        dnnl::memory::data_type::s8,
        dnnl::memory::data_type::u8,
        dnnl::memory::data_type::f32,
        combined_scale
    );
    return;
}

// INT8 GEMM with transpose support
inline void matmul_ex_int8_int8_f32(
    const int8_t* A, int64_t m, int64_t k,
    const uint8_t* B, int64_t n,
    float* C,
    Transpose trans_a,
    Transpose trans_b,
    float scale_a,
    float scale_b,
    float scale_c
) {
    require_onednn("matmul_ex_int8_int8_f32");
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::cout << "[Matmul] int8_ex -> oneDNN matmul" << std::endl;
    }
    float combined_scale = (scale_a * scale_b) / scale_c;
    dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
        ? dnnl::memory::dims{1, m}
        : dnnl::memory::dims{k, 1};
    dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
        ? dnnl::memory::dims{1, k}
        : dnnl::memory::dims{n, 1};
    dnnl::memory::dims c_strides = {n, 1};
    onednn_matmul_impl(
        A, B, C,
        m, k, n,
        a_strides, b_strides, c_strides,
        dnnl::memory::data_type::s8,
        dnnl::memory::data_type::u8,
        dnnl::memory::data_type::f32,
        combined_scale
    );
    return;
}

// Conversion functions with OpenMP parallelization

// Convert float32 array to bfloat16 array (SIMD-optimized with OpenMP parallelization)
inline void convert_float32_to_bf16(
    const float* src,
    size_t num_elements,
    uint16_t* dst
) {
#ifdef __AVX2__
    // Use SIMD-optimized conversion with OpenMP parallelization
    #ifdef _OPENMP
    // Skip parallelization for small arrays to avoid overhead
    // Threshold: if array fits in 2 chunks, use single-threaded SIMD
    const size_t chunk_size = 256;
    const size_t num_chunks = (num_elements + chunk_size - 1) / chunk_size;
    
    if (num_chunks <= 1) {
        // Small array: use single-threaded SIMD (no OpenMP overhead)
        f32_to_bf16_avx_simd(src, dst, num_elements);
    } else {
        // Large array: parallelize across chunks
        #pragma omp parallel
        {
            const int num_threads = omp_get_num_threads();
            const int thread_id = omp_get_thread_num();
            const size_t chunks_per_thread = (num_chunks + num_threads - 1) / num_threads;
            const size_t start_chunk = thread_id * chunks_per_thread;
            const size_t end_chunk = std::min(start_chunk + chunks_per_thread, num_chunks);
            
            // Each thread processes its chunks using SIMD
            for (size_t chunk = start_chunk; chunk < end_chunk; ++chunk) {
                const size_t start = chunk * chunk_size;
                const size_t end = std::min(start + chunk_size, num_elements);
                if (end > start) {
                    f32_to_bf16_avx_simd(src + start, dst + start, end - start);
                }
            }
        }
    }
    #else
    // Single-threaded SIMD conversion
    f32_to_bf16_avx_simd(src, dst, num_elements);
    #endif
#else
    // Fallback: scalar conversion with OpenMP
    #ifdef _OPENMP
    // Skip parallelization for very small arrays
    if (num_elements > 16384) {  // Threshold: only parallelize if > 16K elements
        #pragma omp parallel for
        for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = float_to_bf16(src[i]);
        }
    } else {
        for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = float_to_bf16(src[i]);
        }
    }
    #else
    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
    #endif
#endif
}

// Convert float32 array to int8 array with quantization (parallelized with OpenMP)
inline void convert_float32_to_int8(
    const float* src,
    size_t num_elements,
    int8_t* dst,
    float scale
) {
    if (scale == 0.0f) {
        std::memset(dst, 0, num_elements * sizeof(int8_t));
        return;
    }
#if defined(__AVX2__)
    const size_t simd_width = 16;
    const size_t simd_end = (num_elements / simd_width) * simd_width;
    const float inv_scale = 1.0f / scale;
    const __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
    const __m256 min_val = _mm256_set1_ps(-128.0f);
    const __m256 max_val = _mm256_set1_ps(127.0f);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 v0 = _mm256_loadu_ps(src + i);
        __m256 v1 = _mm256_loadu_ps(src + i + 8);
        v0 = _mm256_mul_ps(v0, inv_scale_vec);
        v1 = _mm256_mul_ps(v1, inv_scale_vec);
        v0 = _mm256_min_ps(max_val, _mm256_max_ps(min_val, v0));
        v1 = _mm256_min_ps(max_val, _mm256_max_ps(min_val, v1));

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m128i i0_lo = _mm256_castsi256_si128(i0);
        __m128i i0_hi = _mm256_extracti128_si256(i0, 1);
        __m128i i1_lo = _mm256_castsi256_si128(i1);
        __m128i i1_hi = _mm256_extracti128_si256(i1, 1);
        __m128i pack0 = _mm_packs_epi32(i0_lo, i0_hi);
        __m128i pack1 = _mm_packs_epi32(i1_lo, i1_hi);
        __m128i pack8 = _mm_packs_epi16(pack0, pack1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), pack8);
    }
    for (size_t i = simd_end; i < num_elements; ++i) {
        dst[i] = float_to_int8(src[i], scale);
    }
#else
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = float_to_int8(src[i], scale);
    }
#endif
}

// Convert float32 array to uint8 array with quantization (for matrix B in row-major)
// For cblas_gemm_s8u8s32 with row-major, B must be uint8
inline void convert_float32_to_uint8(
    const float* src,
    size_t num_elements,
    uint8_t* dst,
    float scale
) {
    if (scale == 0.0f) {
        std::memset(dst, 128, num_elements * sizeof(uint8_t));
        return;
    }
#if defined(__AVX2__)
    const size_t simd_width = 16;
    const size_t simd_end = (num_elements / simd_width) * simd_width;
    const float inv_scale = 1.0f / scale;
    const __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
    const __m256 min_val = _mm256_set1_ps(-128.0f);
    const __m256 max_val = _mm256_set1_ps(127.0f);
    const __m128i offset = _mm_set1_epi8(static_cast<char>(0x80));
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 v0 = _mm256_loadu_ps(src + i);
        __m256 v1 = _mm256_loadu_ps(src + i + 8);
        v0 = _mm256_mul_ps(v0, inv_scale_vec);
        v1 = _mm256_mul_ps(v1, inv_scale_vec);
        v0 = _mm256_min_ps(max_val, _mm256_max_ps(min_val, v0));
        v1 = _mm256_min_ps(max_val, _mm256_max_ps(min_val, v1));

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m128i i0_lo = _mm256_castsi256_si128(i0);
        __m128i i0_hi = _mm256_extracti128_si256(i0, 1);
        __m128i i1_lo = _mm256_castsi256_si128(i1);
        __m128i i1_hi = _mm256_extracti128_si256(i1, 1);
        __m128i pack0 = _mm_packs_epi32(i0_lo, i0_hi);
        __m128i pack1 = _mm_packs_epi32(i1_lo, i1_hi);
        __m128i pack8 = _mm_packs_epi16(pack0, pack1);
        pack8 = _mm_add_epi8(pack8, offset);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), pack8);
    }
    for (size_t i = simd_end; i < num_elements; ++i) {
        int8_t quantized = float_to_int8(src[i], scale);
        dst[i] = static_cast<uint8_t>(static_cast<int32_t>(quantized) + 128);
    }
#else
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < num_elements; ++i) {
        int8_t quantized = float_to_int8(src[i], scale);
        dst[i] = static_cast<uint8_t>(static_cast<int32_t>(quantized) + 128);
    }
#endif
}

// Convert int8 array to uint8 array by adding 128 (for matrix B in row-major)
inline void convert_int8_to_uint8(
    const int8_t* src,
    size_t num_elements,
    uint8_t* dst
) {
#if defined(__AVX2__)
    const size_t simd_width = 32;
    const size_t simd_end = (num_elements / simd_width) * simd_width;
    const __m256i offset = _mm256_set1_epi8(static_cast<char>(0x80));
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        v = _mm256_add_epi8(v, offset);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (size_t i = simd_end; i < num_elements; ++i) {
        dst[i] = static_cast<uint8_t>(static_cast<int32_t>(src[i]) + 128);
    }
#else
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = static_cast<uint8_t>(static_cast<int32_t>(src[i]) + 128);
    }
#endif
}

// Convert buffer to float32 based on dtype (for non-GEMM operations, parallelized with OpenMP)
inline void convert_buffer_to_float32(
    const uint8_t* src,
    kvtensor::DType src_dtype,
    size_t num_elements,
    float* dst
) {
    if (src_dtype == kvtensor::DType::FLOAT32) {
        std::memcpy(dst, src, num_elements * sizeof(float));
    } else if (src_dtype == kvtensor::DType::BFLOAT16) {
        const uint16_t* bf16_ptr = reinterpret_cast<const uint16_t*>(src);
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = bf16_to_float(bf16_ptr[i]);
        }
    } else if (src_dtype == kvtensor::DType::INT8) {
        const int8_t* int8_ptr = reinterpret_cast<const int8_t*>(src);
        // For INT8, we need a scale for dequantization
        // Use a default scale of 1.0 for now (should be stored with the data)
        // In practice, the scale should be retrieved from metadata
        float default_scale = 1.0f;
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = int32_to_float(static_cast<int32_t>(int8_ptr[i]), default_scale);
        }
    }
}

// Matrix transpose: B = A^T
// A: (m, n), B: (n, m)
template<typename T>
void transpose(const T* A, int64_t m, int64_t n, T* B) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("transpose");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        dnnl::memory::dims dims = {m, n};
        auto src_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::dims{n, 1});
        auto dst_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::dims{1, m});
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(A));
        auto dst_mem = dnnl::memory(dst_md, engine, B);
        dnnl::reorder(src_mem, dst_mem).execute(stream, src_mem, dst_mem);
        stream.wait();
        return;
    } else {
        std::vector<float> Af(static_cast<size_t>(m * n));
        std::vector<float> Bf(static_cast<size_t>(m * n));
        to_float_array(A, m * n, Af.data());
        transpose<float>(Af.data(), m, n, Bf.data());
        from_float_array(Bf.data(), m * n, B);
    }
}

// Element-wise addition: C = A + B
template<typename T>
void add(const T* A, const T* B, int64_t size, T* C) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("add");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        dnnl::memory::dims dims = {size};
        auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
        std::vector<float> scales = {1.0f, 1.0f};
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
        dnnl::sum::primitive_desc pd(engine, md, scales, {md, md}, dnnl::primitive_attr(), false);
#else
        dnnl::sum::primitive_desc pd(md, scales, {md, md}, engine);
#endif
        auto mem_a = dnnl::memory(md, engine, const_cast<float*>(A));
        auto mem_b = dnnl::memory(md, engine, const_cast<float*>(B));
        auto mem_c = dnnl::memory(md, engine, C);
        dnnl::sum(pd).execute(stream, {
            {DNNL_ARG_MULTIPLE_SRC, mem_a},
            {DNNL_ARG_MULTIPLE_SRC + 1, mem_b},
            {DNNL_ARG_DST, mem_c}
        });
        stream.wait();
        return;
    } else {
        std::vector<float> Af(size), Bf(size), Cf(size);
        to_float_array(A, size, Af.data());
        to_float_array(B, size, Bf.data());
        add<float>(Af.data(), Bf.data(), size, Cf.data());
        from_float_array(Cf.data(), size, C);
    }
}

// Element-wise multiplication: C = A * B
template<typename T>
void multiply(const T* A, const T* B, int64_t size, T* C) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("multiply");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        dnnl::memory::dims dims = {size};
        auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
        auto pd = dnnl::binary::primitive_desc(
            engine,
            dnnl::algorithm::binary_mul,
            md,
            md,
            md,
            dnnl::primitive_attr(),
            false
        );
#else
        auto desc = dnnl::binary::desc(dnnl::algorithm::binary_mul, md, md, md);
        auto pd = dnnl::binary::primitive_desc(desc, engine);
#endif
        auto mem_a = dnnl::memory(md, engine, const_cast<float*>(A));
        auto mem_b = dnnl::memory(md, engine, const_cast<float*>(B));
        auto mem_c = dnnl::memory(md, engine, C);
        dnnl::binary(pd).execute(stream, {
            {DNNL_ARG_SRC_0, mem_a},
            {DNNL_ARG_SRC_1, mem_b},
            {DNNL_ARG_DST, mem_c}
        });
        stream.wait();
        return;
    } else {
        std::vector<float> Af(size), Bf(size), Cf(size);
        to_float_array(A, size, Af.data());
        to_float_array(B, size, Bf.data());
        multiply<float>(Af.data(), Bf.data(), size, Cf.data());
        from_float_array(Cf.data(), size, C);
    }
}

// Element-wise division: C = A / B
template<typename T>
void divide(const T* A, const T* B, int64_t size, T* C) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("divide");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        dnnl::memory::dims dims = {size};
        auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
        auto pd = dnnl::binary::primitive_desc(
            engine,
            dnnl::algorithm::binary_div,
            md,
            md,
            md,
            dnnl::primitive_attr(),
            false
        );
#else
        auto desc = dnnl::binary::desc(dnnl::algorithm::binary_div, md, md, md);
        auto pd = dnnl::binary::primitive_desc(desc, engine);
#endif
        auto mem_a = dnnl::memory(md, engine, const_cast<float*>(A));
        auto mem_b = dnnl::memory(md, engine, const_cast<float*>(B));
        auto mem_c = dnnl::memory(md, engine, C);
        dnnl::binary(pd).execute(stream, {
            {DNNL_ARG_SRC_0, mem_a},
            {DNNL_ARG_SRC_1, mem_b},
            {DNNL_ARG_DST, mem_c}
        });
        stream.wait();
        return;
    } else {
        std::vector<float> Af(size), Bf(size), Cf(size);
        to_float_array(A, size, Af.data());
        to_float_array(B, size, Bf.data());
        divide<float>(Af.data(), Bf.data(), size, Cf.data());
        from_float_array(Cf.data(), size, C);
    }
}

// Scalar multiplication: B = A * scalar
template<typename T>
void scale(const T* A, T scalar, int64_t size, T* B) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("scale");
        onednn_eltwise_impl(A, size, B, dnnl::algorithm::eltwise_linear, scalar, 0.0f);
        return;
    } else {
        std::vector<float> Af(size), Bf(size);
        to_float_array(A, size, Af.data());
        float scalar_f = to_float_scalar(scalar);
        scale<float>(Af.data(), scalar_f, size, Bf.data());
        from_float_array(Bf.data(), size, B);
    }
}

// Sum along axis (for 2D matrices)
// axis=0: sum over rows, axis=1: sum over columns
template<typename T>
void sum(const T* A, int64_t rows, int64_t cols, int axis, T* result) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("sum");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        auto src_md = dnnl::memory::desc(
            {rows, cols},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );
        dnnl::memory::dims dst_dims = (axis == 0)
            ? dnnl::memory::dims{1, cols}
            : dnnl::memory::dims{rows, 1};
        auto dst_md = dnnl::memory::desc(
            dst_dims,
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );
        auto pd = onednn_reduction_pd(dnnl::algorithm::reduction_sum, src_md, dst_md);
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(A));
        auto dst_mem = dnnl::memory(dst_md, engine, result);
        dnnl::reduction(pd).execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();
        return;
    } else {
        std::vector<float> Af(rows * cols);
        std::vector<float> Rf((axis == 0) ? cols : rows);
        to_float_array(A, rows * cols, Af.data());
        sum<float>(Af.data(), rows, cols, axis, Rf.data());
        from_float_array(Rf.data(), (axis == 0) ? cols : rows, result);
        return;
    }
}

// Mean along axis
template<typename T>
void mean(const T* A, int64_t rows, int64_t cols, int axis, T* result) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("mean");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        auto src_md = dnnl::memory::desc(
            {rows, cols},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );
        dnnl::memory::dims dst_dims = (axis == 0)
            ? dnnl::memory::dims{1, cols}
            : dnnl::memory::dims{rows, 1};
        auto dst_md = dnnl::memory::desc(
            dst_dims,
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );
        auto pd = onednn_reduction_pd(dnnl::algorithm::reduction_mean, src_md, dst_md);
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(A));
        auto dst_mem = dnnl::memory(dst_md, engine, result);
        dnnl::reduction(pd).execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();
        return;
    } else {
        std::vector<float> Af(rows * cols);
        std::vector<float> Rf((axis == 0) ? cols : rows);
        to_float_array(A, rows * cols, Af.data());
        mean<float>(Af.data(), rows, cols, axis, Rf.data());
        from_float_array(Rf.data(), (axis == 0) ? cols : rows, result);
        return;
    }
}

// Max along axis
template<typename T>
void max(const T* A, int64_t rows, int64_t cols, int axis, T* result) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("max");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();
        auto src_md = dnnl::memory::desc(
            {rows, cols},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );
        dnnl::memory::dims dst_dims = (axis == 0)
            ? dnnl::memory::dims{1, cols}
            : dnnl::memory::dims{rows, 1};
        auto dst_md = dnnl::memory::desc(
            dst_dims,
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );
        auto pd = onednn_reduction_pd(dnnl::algorithm::reduction_max, src_md, dst_md);
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(A));
        auto dst_mem = dnnl::memory(dst_md, engine, result);
        dnnl::reduction(pd).execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();
        return;
    }
    std::vector<float> Af(rows * cols);
    std::vector<float> Rf((axis == 0) ? cols : rows);
    to_float_array(A, rows * cols, Af.data());
    max<float>(Af.data(), rows, cols, axis, Rf.data());
    from_float_array(Rf.data(), (axis == 0) ? cols : rows, result);
}

// Exponential: B = exp(A)
template<typename T>
void exp(const T* A, int64_t size, T* B) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("exp");
        onednn_eltwise_impl(A, size, B, dnnl::algorithm::eltwise_exp);
        return;
    } else {
        std::vector<float> Af(size), Bf(size);
        to_float_array(A, size, Af.data());
        exp<float>(Af.data(), size, Bf.data());
        from_float_array(Bf.data(), size, B);
    }
}

// Sigmoid: B = sigmoid(A) = 1 / (1 + exp(-A))
template<typename T>
void sigmoid(const T* A, int64_t size, T* B) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("sigmoid");
        onednn_eltwise_impl(A, size, B, dnnl::algorithm::eltwise_logistic);
        return;
    } else {
        std::vector<float> Af(size), Bf(size);
        to_float_array(A, size, Af.data());
        sigmoid<float>(Af.data(), size, Bf.data());
        from_float_array(Bf.data(), size, B);
    }
}

// SiLU: B = SiLU(A) = A * sigmoid(A)
template<typename T>
void silu(const T* A, int64_t size, T* B) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("silu");
        onednn_eltwise_impl(A, size, B, dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
        return;
    } else {
        std::vector<float> Af(size), Bf(size);
        to_float_array(A, size, Af.data());
        silu<float>(Af.data(), size, Bf.data());
        from_float_array(Bf.data(), size, B);
    }
}

// GELU (approx): B = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
template<typename T>
void gelu(const T* A, int64_t size, T* B) {
    for (int64_t i = 0; i < size; ++i) {
        float x = static_cast<float>(A[i]);
        float x3 = x * x * x;
        float t = 0.7978845608f * (x + 0.044715f * x3);
        float y = 0.5f * x * (1.0f + std::tanh(t));
        B[i] = static_cast<T>(y);
    }
}

template<>
inline void gelu<float>(const float* A, int64_t size, float* B) {
    for (int64_t i = 0; i < size; ++i) {
        float x = A[i];
        float x3 = x * x * x;
        float t = 0.7978845608f * (x + 0.044715f * x3);
        B[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
}

template<typename T>
void gelu(const std::vector<T>& A, std::vector<T>& B) {
    if (B.size() != A.size()) {
        B.resize(A.size());
    }
    gelu(A.data(), static_cast<int64_t>(A.size()), B.data());
}

// Square root: B = sqrt(A)
template<typename T>
void sqrt(const T* A, int64_t size, T* B) {
    if constexpr (std::is_same<T, float>::value) {
        require_onednn("sqrt");
        onednn_eltwise_impl(A, size, B, dnnl::algorithm::eltwise_sqrt);
        return;
    } else {
        std::vector<float> Af(size), Bf(size);
        to_float_array(A, size, Af.data());
        sqrt<float>(Af.data(), size, Bf.data());
        from_float_array(Bf.data(), size, B);
    }
}

// Softmax along axis
// axis=0: softmax over rows, axis=1: softmax over columns
template<typename T>
void softmax(const T* A, int64_t rows, int64_t cols, int axis, T* B) {
    if constexpr (!std::is_same<T, float>::value) {
        std::vector<float> Af(rows * cols);
        std::vector<float> Bf(rows * cols);
        to_float_array(A, rows * cols, Af.data());
        softmax<float>(Af.data(), rows, cols, axis, Bf.data());
        from_float_array(Bf.data(), rows * cols, B);
        return;
    }

    require_onednn("softmax");
    auto& engine = onednn_engine();
    auto& stream = onednn_stream();
    auto md = dnnl::memory::desc(
        {rows, cols},
        dnnl::memory::data_type::f32,
        dnnl::memory::format_tag::ab
    );
    auto pd = onednn_softmax_pd(md, axis);
    auto src_mem = dnnl::memory(md, engine, const_cast<float*>(A));
    auto dst_mem = dnnl::memory(md, engine, B);
    dnnl::softmax_forward(pd).execute(stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    stream.wait();
    return;
}

// Concatenate arrays along axis
// For simplicity, this handles 2D concatenation
template<typename T>
void concatenate(
    const std::vector<const T*>& arrays,
    const std::vector<std::pair<int64_t, int64_t>>& shapes,
    int axis,
    T* result
) {
    if (arrays.empty()) return;

    if constexpr (std::is_same<T, float>::value) {
        require_onednn("concatenate");
        auto& engine = onednn_engine();
        auto& stream = onednn_stream();

        int64_t result_rows = shapes[0].first;
        int64_t result_cols = shapes[0].second;
        if (axis == 0) {
            result_rows = 0;
            for (const auto& shape : shapes) {
                result_rows += shape.first;
            }
        } else if (axis == 1) {
            result_cols = 0;
            for (const auto& shape : shapes) {
                result_cols += shape.second;
            }
        }

        auto dst_md = dnnl::memory::desc(
            {result_rows, result_cols},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab
        );

        std::vector<dnnl::memory::desc> src_mds;
        src_mds.reserve(arrays.size());
        for (size_t i = 0; i < arrays.size(); ++i) {
            src_mds.emplace_back(
                dnnl::memory::desc(
                    {shapes[i].first, shapes[i].second},
                    dnnl::memory::data_type::f32,
                    dnnl::memory::format_tag::ab
                )
            );
        }

        auto pd = dnnl::concat::primitive_desc(dst_md, axis, src_mds, engine);
        dnnl::concat prim(pd);

        std::unordered_map<int, dnnl::memory> args;
        for (size_t i = 0; i < arrays.size(); ++i) {
            args[DNNL_ARG_MULTIPLE_SRC + static_cast<int>(i)] =
                dnnl::memory(src_mds[i], engine, const_cast<float*>(arrays[i]));
        }
        args[DNNL_ARG_DST] = dnnl::memory(dst_md, engine, result);
        prim.execute(stream, args);
        stream.wait();
        return;
    }
    
    if (axis == 0) {
        // Concatenate along rows
        int64_t result_cols = shapes[0].second;
        int64_t row_offset = 0;
        for (size_t arr_idx = 0; arr_idx < arrays.size(); ++arr_idx) {
            int64_t arr_rows = shapes[arr_idx].first;
            int64_t arr_cols = shapes[arr_idx].second;
            const T* arr = arrays[arr_idx];
            
            for (int64_t i = 0; i < arr_rows; ++i) {
                std::memcpy(
                    result + (row_offset + i) * result_cols,
                    arr + i * arr_cols,
                    arr_cols * sizeof(T)
                );
            }
            row_offset += arr_rows;
        }
    } else if (axis == 1) {
        // Concatenate along columns
        int64_t result_rows = shapes[0].first;
        int64_t result_cols = 0;
        for (const auto& shape : shapes) {
            result_cols += shape.second;
        }
        int64_t col_offset = 0;
        for (size_t arr_idx = 0; arr_idx < arrays.size(); ++arr_idx) {
            int64_t arr_rows = shapes[arr_idx].first;
            int64_t arr_cols = shapes[arr_idx].second;
            const T* arr = arrays[arr_idx];
            
            for (int64_t i = 0; i < arr_rows; ++i) {
                std::memcpy(
                    result + i * result_cols + col_offset,
                    arr + i * arr_cols,
                    arr_cols * sizeof(T)
                );
            }
            col_offset += arr_cols;
        }
    }
}

} // namespace math
} // namespace kvtensor
