#pragma once

#include "kvtensor/types.hpp"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace kvtensor {

struct ProfileMatrixRead {
    std::string matrix_id;
    Shape matrix_shape{0, 0};
    DType dtype = DType::FLOAT32;
    SplitMode split_mode = SplitMode::ROW;
    int64_t chunk_size = 0;
    uint64_t bytes_read = 0;
    uint64_t chunk_reads = 0;
};

struct ProfileGemmBucket {
    std::string operator_class;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    uint64_t calls = 0;
    uint64_t flops = 0;
};

struct ProfileStats {
    std::atomic<uint64_t> compute_ns{0};
    std::atomic<uint64_t> other_compute_ns{0};
    std::atomic<uint64_t> kv_read_ns{0};
    std::atomic<uint64_t> decompress_ns{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> gemm_flops{0};
    mutable std::mutex detail_mutex;
    std::unordered_map<std::string, ProfileMatrixRead> matrix_reads;
    std::unordered_map<std::string, ProfileGemmBucket> gemm_buckets;
};

enum class ProfileKind {
    Compute,
    OtherCompute,
    KVRead,
    Decompress
};

void set_active_profile(ProfileStats* stats);
ProfileStats* active_profile();
void reset_profile(ProfileStats* stats);
void add_profile_bytes(uint64_t bytes);
void add_profile_gemm_flops(uint64_t flops);
void add_profile_gemm(
    const std::string& operator_class,
    int64_t m,
    int64_t k,
    int64_t n
);
void add_profile_kv_read_ns(uint64_t ns);
void add_profile_matrix_read(
    const std::string& matrix_id,
    const Shape& matrix_shape,
    DType dtype,
    SplitMode split_mode,
    int64_t chunk_size,
    uint64_t bytes
);

class ProfileScope {
public:
    explicit ProfileScope(ProfileKind kind);
    ~ProfileScope();

private:
    ProfileKind kind_;
    uint64_t start_ns_;
};

} // namespace kvtensor
