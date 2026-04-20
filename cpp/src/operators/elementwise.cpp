#include "kvtensor/operators.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <dnnl.hpp>
#include <dnnl_version.h>
#include <cstring>
#include <stdexcept>

namespace kvtensor {

// Helper functions
template<typename T>
const float* get_float32_ptr(const std::vector<uint8_t>& data) {
    return reinterpret_cast<const float*>(data.data());
}

template<typename T>
float* get_float32_ptr_mut(std::vector<uint8_t>& data) {
    return reinterpret_cast<float*>(data.data());
}

// RowColumnElementwiseAdd operator
class RowColumnElementwiseAdd : public Operator {
public:
    RowColumnElementwiseAdd(
        const std::string& lhs_id,
        const std::string& rhs_id,
        const std::string& result_id,
        float alpha = 1.0f,
        float beta = 1.0f,
        bool store_in_memory = true
    ) : lhs_id_(lhs_id),
        rhs_id_(rhs_id),
        result_id_(result_id),
        alpha_(alpha),
        beta_(beta),
        store_in_memory_(store_in_memory) {
        name_ = "rowcolumn_elementwise_add";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve inputs
        auto lhs = ctx.resolve_in_memory(lhs_id_);
        auto rhs = ctx.resolve_in_memory(rhs_id_);
        
        if (!lhs) {
            auto block = ctx.resolve_block_matrix(lhs_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                lhs = ctx.store_in_memory(lhs_id_ + "_dense", block->shape(), block->dtype(), std::move(dense));
            } else {
                throw std::runtime_error("LHS matrix not found: " + lhs_id_);
            }
        }
        
        if (!rhs) {
            auto block = ctx.resolve_block_matrix(rhs_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                rhs = ctx.store_in_memory(rhs_id_ + "_dense", block->shape(), block->dtype(), std::move(dense));
            } else {
                throw std::runtime_error("RHS matrix not found: " + rhs_id_);
            }
        }
        
        auto [lhs_rows, lhs_cols] = lhs->shape();
        auto [rhs_rows, rhs_cols] = rhs->shape();
        
        if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
            throw std::runtime_error("Shape mismatch for elementwise add");
        }
        
        // Get dtypes
        DType lhs_dtype = lhs->dtype();
        DType rhs_dtype = rhs->dtype();
        
        int64_t size = lhs_rows * lhs_cols;
        bool use_bf16 = (lhs_dtype == DType::BFLOAT16 && rhs_dtype == DType::BFLOAT16);
        std::vector<float> lhs_f32;
        std::vector<float> rhs_f32;
        const void* lhs_ptr = nullptr;
        const void* rhs_ptr = nullptr;

        if (use_bf16) {
            lhs_ptr = lhs->data().data();
            rhs_ptr = rhs->data().data();
        } else {
            lhs_f32.resize(size);
            rhs_f32.resize(size);
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_buffer_to_float32(lhs->data().data(), lhs_dtype, size, lhs_f32.data());
                math::convert_buffer_to_float32(rhs->data().data(), rhs_dtype, size, rhs_f32.data());
            }
            lhs_ptr = lhs_f32.data();
            rhs_ptr = rhs_f32.data();
        }
        
        // Allocate result
        size_t elem_size = use_bf16 ? sizeof(uint16_t) : sizeof(float);
        size_t result_size = static_cast<size_t>(lhs_rows * lhs_cols) * elem_size;
        std::vector<uint8_t> result_data(result_size);
        void* result_ptr = result_data.data();
        
        // Compute: result = alpha * lhs + beta * rhs
        math::require_onednn("rowcolumn_elementwise_add");
        auto& engine = math::onednn_engine();
        auto& stream = math::onednn_stream();
        dnnl::memory::dims dims = {size};
        auto md = dnnl::memory::desc(
            dims,
            use_bf16 ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::x
        );
        std::vector<float> scales = {alpha_, beta_};
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
        dnnl::sum::primitive_desc pd(engine, md, scales, {md, md}, dnnl::primitive_attr(), false);
#else
        dnnl::sum::primitive_desc pd(md, scales, {md, md}, engine);
#endif
        auto mem_a = dnnl::memory(md, engine, const_cast<void*>(lhs_ptr));
        auto mem_b = dnnl::memory(md, engine, const_cast<void*>(rhs_ptr));
        auto mem_c = dnnl::memory(md, engine, result_ptr);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            dnnl::sum(pd).execute(stream, {
                {DNNL_ARG_MULTIPLE_SRC, mem_a},
                {DNNL_ARG_MULTIPLE_SRC + 1, mem_b},
                {DNNL_ARG_DST, mem_c}
            });
            stream.wait();
        }
        
        auto result = ctx.store_in_memory(
            result_id_,
            lhs->shape(),
            use_bf16 ? DType::BFLOAT16 : DType::FLOAT32,
            std::move(result_data)
        );
        
        return result;
    }

private:
    std::string lhs_id_;
    std::string rhs_id_;
    std::string result_id_;
    float alpha_;
    float beta_;
    bool store_in_memory_;
};

// RowColumnElementwiseMultiply operator
class RowColumnElementwiseMultiply : public Operator {
public:
    RowColumnElementwiseMultiply(
        const std::string& lhs_id,
        const std::string& rhs_id,
        const std::string& result_id,
        bool store_in_memory = true
    ) : lhs_id_(lhs_id),
        rhs_id_(rhs_id),
        result_id_(result_id),
        store_in_memory_(store_in_memory) {
        name_ = "rowcolumn_elementwise_multiply";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve inputs
        auto lhs = ctx.resolve_in_memory(lhs_id_);
        auto rhs = ctx.resolve_in_memory(rhs_id_);
        
        if (!lhs) {
            auto block = ctx.resolve_block_matrix(lhs_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                lhs = ctx.store_in_memory(lhs_id_ + "_dense", block->shape(), block->dtype(), std::move(dense));
            } else {
                throw std::runtime_error("LHS matrix not found: " + lhs_id_);
            }
        }
        
        if (!rhs) {
            auto block = ctx.resolve_block_matrix(rhs_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                rhs = ctx.store_in_memory(rhs_id_ + "_dense", block->shape(), block->dtype(), std::move(dense));
            } else {
                throw std::runtime_error("RHS matrix not found: " + rhs_id_);
            }
        }
        
        auto [lhs_rows, lhs_cols] = lhs->shape();
        auto [rhs_rows, rhs_cols] = rhs->shape();
        
        if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
            throw std::runtime_error("Shape mismatch for elementwise multiply");
        }
        
        // Get dtypes
        DType lhs_dtype = lhs->dtype();
        DType rhs_dtype = rhs->dtype();
        
        int64_t size = lhs_rows * lhs_cols;
        bool use_bf16 = (lhs_dtype == DType::BFLOAT16 && rhs_dtype == DType::BFLOAT16);
        std::vector<float> lhs_f32;
        std::vector<float> rhs_f32;
        const void* lhs_ptr = nullptr;
        const void* rhs_ptr = nullptr;

        if (use_bf16) {
            lhs_ptr = lhs->data().data();
            rhs_ptr = rhs->data().data();
        } else {
            lhs_f32.resize(size);
            rhs_f32.resize(size);
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_buffer_to_float32(lhs->data().data(), lhs_dtype, size, lhs_f32.data());
                math::convert_buffer_to_float32(rhs->data().data(), rhs_dtype, size, rhs_f32.data());
            }
            lhs_ptr = lhs_f32.data();
            rhs_ptr = rhs_f32.data();
        }
        
        // Allocate result
        size_t elem_size = use_bf16 ? sizeof(uint16_t) : sizeof(float);
        size_t result_size = static_cast<size_t>(lhs_rows * lhs_cols) * elem_size;
        std::vector<uint8_t> result_data(result_size);
        void* result_ptr = result_data.data();
        
        // Compute: result = lhs * rhs
        math::require_onednn("rowcolumn_elementwise_multiply");
        auto& engine = math::onednn_engine();
        auto& stream = math::onednn_stream();
        dnnl::memory::dims dims = {size};
        auto md = dnnl::memory::desc(
            dims,
            use_bf16 ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::x
        );
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
        auto mem_a = dnnl::memory(md, engine, const_cast<void*>(lhs_ptr));
        auto mem_b = dnnl::memory(md, engine, const_cast<void*>(rhs_ptr));
        auto mem_c = dnnl::memory(md, engine, result_ptr);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            dnnl::binary(pd).execute(stream, {
                {DNNL_ARG_SRC_0, mem_a},
                {DNNL_ARG_SRC_1, mem_b},
                {DNNL_ARG_DST, mem_c}
            });
            stream.wait();
        }
        
        auto result = ctx.store_in_memory(
            result_id_,
            lhs->shape(),
            use_bf16 ? DType::BFLOAT16 : DType::FLOAT32,
            std::move(result_data)
        );
        
        return result;
    }

private:
    std::string lhs_id_;
    std::string rhs_id_;
    std::string result_id_;
    bool store_in_memory_;
};

// Factory functions
std::unique_ptr<Operator> create_row_column_elementwise_add(
    const std::string& lhs_id,
    const std::string& rhs_id,
    const std::string& result_id,
    float alpha,
    float beta,
    bool store_in_memory
) {
    return std::make_unique<RowColumnElementwiseAdd>(lhs_id, rhs_id, result_id, alpha, beta, store_in_memory);
}

std::unique_ptr<Operator> create_row_column_elementwise_multiply(
    const std::string& lhs_id,
    const std::string& rhs_id,
    const std::string& result_id,
    bool store_in_memory
) {
    return std::make_unique<RowColumnElementwiseMultiply>(lhs_id, rhs_id, result_id, store_in_memory);
}

} // namespace kvtensor
