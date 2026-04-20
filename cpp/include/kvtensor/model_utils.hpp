#pragma once

#include "kvtensor/bufferpool.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kvtensor {

struct PrefetchConfig {
    size_t arena_size_mb = 512;
    size_t prefetch_window = 128;
    bool ring = false;
    bool simulate_prefetch = false;
    uint64_t simulate_get_latency_ms = 0;
    std::string graph_path;
    size_t graph_max_nodes = 1000000;
};

struct PreloadOptions {
    bool log = true;
};

// Read a matrix from storage into a dense buffer (bypassing bufferpool).
std::vector<uint8_t> read_matrix_from_storage(
    const std::shared_ptr<BlockMatrix>& block_matrix,
    SimpleDBStorage* storage
);

// Collect matrices that exist in DB and are not preloaded.
std::unordered_map<std::string, std::shared_ptr<BlockMatrix>> collect_db_matrices(
    OperatorContext& ctx,
    const std::vector<std::string>& matrix_ids,
    const std::unordered_set<std::string>& preloaded_ids
);

// Build prefetch sequence in the same order as matrix_ids.
std::vector<ChunkKey> build_prefetch_sequence(
    const std::vector<std::string>& matrix_ids,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
);

// Start or restart prefetch with a shared bufferpool.
void start_or_restart_prefetch(
    OperatorContext& ctx,
    SimpleDBStorage* storage,
    const PrefetchConfig& config,
    const std::vector<std::string>& matrix_ids,
    const std::unordered_set<std::string>& preloaded_ids
);

// Preload matrices into memory.
void preload_matrices(
    const std::unordered_set<std::string>& preload_ids,
    OperatorContext& ctx,
    const PreloadOptions& options = {}
);

} // namespace kvtensor
