#include "kvtensor/profile.hpp"
#include <chrono>
#include <sstream>

namespace kvtensor {
namespace {
std::atomic<ProfileStats*> g_active_profile{nullptr};

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}
} // namespace

void set_active_profile(ProfileStats* stats) {
    g_active_profile.store(stats, std::memory_order_release);
}

ProfileStats* active_profile() {
    return g_active_profile.load(std::memory_order_acquire);
}

void reset_profile(ProfileStats* stats) {
    if (!stats) {
        return;
    }
    stats->compute_ns.store(0, std::memory_order_relaxed);
    stats->other_compute_ns.store(0, std::memory_order_relaxed);
    stats->kv_read_ns.store(0, std::memory_order_relaxed);
    stats->decompress_ns.store(0, std::memory_order_relaxed);
    stats->bytes_read.store(0, std::memory_order_relaxed);
    stats->gemm_flops.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(stats->detail_mutex);
    stats->matrix_reads.clear();
    stats->gemm_buckets.clear();
}

void add_profile_bytes(uint64_t bytes) {
    if (auto* stats = g_active_profile.load(std::memory_order_acquire)) {
        stats->bytes_read.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void add_profile_gemm_flops(uint64_t flops) {
    if (auto* stats = g_active_profile.load(std::memory_order_acquire)) {
        stats->gemm_flops.fetch_add(flops, std::memory_order_relaxed);
    }
}

void add_profile_gemm(
    const std::string& operator_class,
    int64_t m,
    int64_t k,
    int64_t n
) {
    if (m <= 0 || k <= 0 || n <= 0) {
        return;
    }
    auto* stats = g_active_profile.load(std::memory_order_acquire);
    if (!stats) {
        return;
    }
    const uint64_t flops = static_cast<uint64_t>(2ULL) *
        static_cast<uint64_t>(m) *
        static_cast<uint64_t>(k) *
        static_cast<uint64_t>(n);
    stats->gemm_flops.fetch_add(flops, std::memory_order_relaxed);

    std::ostringstream key_builder;
    key_builder << operator_class << "|" << m << "|" << k << "|" << n;
    const std::string key = key_builder.str();

    std::lock_guard<std::mutex> lock(stats->detail_mutex);
    auto& bucket = stats->gemm_buckets[key];
    if (bucket.calls == 0) {
        bucket.operator_class = operator_class;
        bucket.m = m;
        bucket.k = k;
        bucket.n = n;
    }
    bucket.calls += 1;
    bucket.flops += flops;
}

void add_profile_kv_read_ns(uint64_t ns) {
    if (auto* stats = g_active_profile.load(std::memory_order_acquire)) {
        stats->kv_read_ns.fetch_add(ns, std::memory_order_relaxed);
    }
}

void add_profile_matrix_read(
    const std::string& matrix_id,
    const Shape& matrix_shape,
    DType dtype,
    SplitMode split_mode,
    int64_t chunk_size,
    uint64_t bytes
) {
    auto* stats = g_active_profile.load(std::memory_order_acquire);
    if (!stats || matrix_id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(stats->detail_mutex);
    auto& entry = stats->matrix_reads[matrix_id];
    if (entry.chunk_reads == 0) {
        entry.matrix_id = matrix_id;
        entry.matrix_shape = matrix_shape;
        entry.dtype = dtype;
        entry.split_mode = split_mode;
        entry.chunk_size = chunk_size;
    }
    entry.bytes_read += bytes;
    entry.chunk_reads += 1;
}

ProfileScope::ProfileScope(ProfileKind kind)
    : kind_(kind), start_ns_(now_ns()) {}

ProfileScope::~ProfileScope() {
    ProfileStats* stats = g_active_profile.load(std::memory_order_acquire);
    if (!stats) {
        return;
    }
    uint64_t elapsed = now_ns() - start_ns_;
    switch (kind_) {
        case ProfileKind::Compute:
            stats->compute_ns.fetch_add(elapsed, std::memory_order_relaxed);
            break;
        case ProfileKind::OtherCompute:
            stats->other_compute_ns.fetch_add(elapsed, std::memory_order_relaxed);
            break;
        case ProfileKind::KVRead:
            stats->kv_read_ns.fetch_add(elapsed, std::memory_order_relaxed);
            break;
        case ProfileKind::Decompress:
            stats->decompress_ns.fetch_add(elapsed, std::memory_order_relaxed);
            break;
    }
}

} // namespace kvtensor
