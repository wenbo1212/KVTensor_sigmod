#pragma once

#include "math/arithmetic.hpp"

namespace kvtensor {
namespace math {

// BF16 GEMM with output row stride (C is float32 with ldc row stride).
inline void matmul_bf16bf16f32_out_strided(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    float* C, int64_t ldc
) {
    require_onednn("matmul_bf16bf16f32_out_strided");
    dnnl::memory::dims a_strides = {k, 1};
    dnnl::memory::dims b_strides = {n, 1};
    dnnl::memory::dims c_strides = {ldc, 1};
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

// BF16 GEMM with output row stride (C is bf16 with ldc row stride).
inline void matmul_bf16bf16bf16_out_strided(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    uint16_t* C, int64_t ldc
) {
    require_onednn("matmul_bf16bf16bf16_out_strided");
    dnnl::memory::dims a_strides = {k, 1};
    dnnl::memory::dims b_strides = {n, 1};
    dnnl::memory::dims c_strides = {ldc, 1};
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

// BF16 GEMM with transpose and output row stride.
inline void matmul_ex_bf16bf16f32_out_strided(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    float* C, int64_t ldc,
    Transpose trans_a,
    Transpose trans_b
) {
    require_onednn("matmul_ex_bf16bf16f32_out_strided");
    dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
        ? dnnl::memory::dims{1, m}
        : dnnl::memory::dims{k, 1};
    dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
        ? dnnl::memory::dims{1, k}
        : dnnl::memory::dims{n, 1};
    dnnl::memory::dims c_strides = {ldc, 1};
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

// BF16 GEMM with transpose and output row stride (bf16 output).
inline void matmul_ex_bf16bf16bf16_out_strided(
    const uint16_t* A, int64_t m, int64_t k,
    const uint16_t* B, int64_t n,
    uint16_t* C, int64_t ldc,
    Transpose trans_a,
    Transpose trans_b
) {
    require_onednn("matmul_ex_bf16bf16bf16_out_strided");
    dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
        ? dnnl::memory::dims{1, m}
        : dnnl::memory::dims{k, 1};
    dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
        ? dnnl::memory::dims{1, k}
        : dnnl::memory::dims{n, 1};
    dnnl::memory::dims c_strides = {ldc, 1};
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

// INT8 GEMM with output row stride (C is float32 with ldc row stride).
inline void matmul_int8_int8_f32_out_strided(
    const int8_t* A, int64_t m, int64_t k,
    const uint8_t* B, int64_t n,
    float* C, int64_t ldc,
    float scale_a,
    float scale_b,
    float scale_c = 1.0f
) {
    require_onednn("matmul_int8_int8_f32_out_strided");
    float combined_scale = (scale_a * scale_b) / scale_c;
    dnnl::memory::dims a_strides = {k, 1};
    dnnl::memory::dims b_strides = {n, 1};
    dnnl::memory::dims c_strides = {ldc, 1};
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

// INT8 GEMM with transpose and output row stride.
inline void matmul_ex_int8_int8_f32_out_strided(
    const int8_t* A, int64_t m, int64_t k,
    const uint8_t* B, int64_t n,
    float* C, int64_t ldc,
    Transpose trans_a,
    Transpose trans_b,
    float scale_a,
    float scale_b,
    float scale_c
) {
    require_onednn("matmul_ex_int8_int8_f32_out_strided");
    float combined_scale = (scale_a * scale_b) / scale_c;
    dnnl::memory::dims a_strides = (trans_a == Transpose::Yes)
        ? dnnl::memory::dims{1, m}
        : dnnl::memory::dims{k, 1};
    dnnl::memory::dims b_strides = (trans_b == Transpose::Yes)
        ? dnnl::memory::dims{1, k}
        : dnnl::memory::dims{n, 1};
    dnnl::memory::dims c_strides = {ldc, 1};
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

} // namespace math
} // namespace kvtensor
