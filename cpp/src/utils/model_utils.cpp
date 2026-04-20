#include "kvtensor/model_utils.hpp"
#include "kvtensor/prefetch_graph.hpp"
#include "kvtensor/prefetch_graph_provider.hpp"
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace kvtensor {

namespace {
std::string resolve_graph_path(SimpleDBStorage* storage, const std::string& graph_path) {
    if (!graph_path.empty()) {
        return graph_path;
    }
    if (!storage) {
        return "";
    }
    std::filesystem::path candidate = std::filesystem::path(storage->path()) / "prefetch_graph.txt";
    if (std::filesystem::exists(candidate)) {
        return candidate.string();
    }
    return "";
}
} // namespace

static size_t dtype_size_bytes(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::BFLOAT16:
        case DType::FLOAT16: return 2;
        case DType::INT8: return 1;
        default: return 4;
    }
}

std::vector<uint8_t> read_matrix_from_storage(
    const std::shared_ptr<BlockMatrix>& block_matrix,
    SimpleDBStorage* storage
) {
    int64_t rows = std::get<0>(block_matrix->shape());
    int64_t cols = std::get<1>(block_matrix->shape());
    DType dtype = block_matrix->dtype();
    size_t element_size = dtype_size_bytes(dtype);
    size_t total_size = static_cast<size_t>(rows * cols) * element_size;
    std::vector<uint8_t> result(total_size, 0);

    if (block_matrix->split_mode() == SplitMode::ROW) {
        int64_t num_chunks = block_matrix->num_row_chunks();
        for (int64_t i = 0; i < num_chunks; ++i) {
            auto chunk_data_opt = storage->get_row_chunk(block_matrix->matrix_id(), i);
            if (!chunk_data_opt.has_value()) {
                throw std::runtime_error("Failed to read row chunk " + std::to_string(i) +
                                         " for matrix " + block_matrix->matrix_id());
            }
            const auto& chunk_data = chunk_data_opt.value();
            auto [chunk_rows, chunk_cols] = block_matrix->row_chunk_shape(i);

            int64_t row_offset = i * block_matrix->chunk_size();
            for (int64_t r = 0; r < chunk_rows; ++r) {
                size_t src_offset = static_cast<size_t>(r * chunk_cols) * element_size;
                size_t dst_offset = static_cast<size_t>((row_offset + r) * cols) * element_size;
                std::memcpy(
                    result.data() + dst_offset,
                    chunk_data.data() + src_offset,
                    static_cast<size_t>(chunk_cols) * element_size
                );
            }
        }
    } else if (block_matrix->split_mode() == SplitMode::COLUMN) {
        int64_t num_chunks = block_matrix->num_col_chunks();
        for (int64_t j = 0; j < num_chunks; ++j) {
            auto chunk_data_opt = storage->get_col_chunk(block_matrix->matrix_id(), j);
            if (!chunk_data_opt.has_value()) {
                throw std::runtime_error("Failed to read col chunk " + std::to_string(j) +
                                         " for matrix " + block_matrix->matrix_id());
            }
            const auto& chunk_data = chunk_data_opt.value();
            auto [chunk_rows, chunk_cols] = block_matrix->col_chunk_shape(j);

            int64_t col_offset = j * block_matrix->chunk_size();
            for (int64_t c = 0; c < chunk_cols; ++c) {
                for (int64_t r = 0; r < chunk_rows; ++r) {
                    size_t src_offset = static_cast<size_t>(r * chunk_cols + c) * element_size;
                    size_t dst_offset = static_cast<size_t>(r * cols + col_offset + c) * element_size;
                    std::memcpy(
                        result.data() + dst_offset,
                        chunk_data.data() + src_offset,
                        element_size
                    );
                }
            }
        }
    }

    return result;
}

std::unordered_map<std::string, std::shared_ptr<BlockMatrix>> collect_db_matrices(
    OperatorContext& ctx,
    const std::vector<std::string>& matrix_ids,
    const std::unordered_set<std::string>& preloaded_ids
) {
    std::unordered_map<std::string, std::shared_ptr<BlockMatrix>> matrices;
    for (const auto& matrix_id : matrix_ids) {
        if (preloaded_ids.find(matrix_id) != preloaded_ids.end()) {
            continue;
        }
        try {
            auto matrix = ctx.resolve_block_matrix(matrix_id);
            if (matrix) {
                matrices[matrix_id] = matrix;
            }
        } catch (const std::exception&) {
            // Matrix missing or in-memory only; skip.
        }
    }
    return matrices;
}

std::vector<ChunkKey> build_prefetch_sequence(
    const std::vector<std::string>& matrix_ids,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
) {
    std::vector<ChunkKey> sequence;
    for (const auto& matrix_id : matrix_ids) {
        auto it = matrices.find(matrix_id);
        if (it == matrices.end()) {
            continue;
        }
        const auto& matrix = it->second;
        if (matrix->split_mode() == SplitMode::COLUMN) {
            for (int64_t j = 0; j < matrix->num_col_chunks(); ++j) {
                sequence.emplace_back(matrix_id, j, SplitMode::COLUMN);
            }
        } else if (matrix->split_mode() == SplitMode::ROW) {
            for (int64_t i = 0; i < matrix->num_row_chunks(); ++i) {
                sequence.emplace_back(matrix_id, i, SplitMode::ROW);
            }
        }
    }
    return sequence;
}

void start_or_restart_prefetch(
    OperatorContext& ctx,
    SimpleDBStorage* storage,
    const PrefetchConfig& config,
    const std::vector<std::string>& matrix_ids,
    const std::unordered_set<std::string>& preloaded_ids
) {
    std::string graph_path = resolve_graph_path(storage, config.graph_path);
    ctx.set_simulated_get_latency_ms(config.simulate_get_latency_ms);
    ctx.set_simulate_get_chunks(config.simulate_prefetch);

    if (!storage || (matrix_ids.empty() && graph_path.empty())) {
        if (ctx.buffer_pool() != nullptr) {
            ctx.stop_weight_prefetch();
            ctx.set_buffer_pool(nullptr);
        }
        return;
    }

    PrefetchGraph graph;
    std::vector<std::string> effective_ids = matrix_ids;
    if (!graph_path.empty()) {
        std::string error;
        if (!graph.load_from_file(graph_path, &error)) {
            throw std::runtime_error(error);
        }
        effective_ids.clear();
        effective_ids.reserve(graph.nodes().size());
        for (const auto& [node_id, _] : graph.nodes()) {
            effective_ids.push_back(node_id);
        }
    }

    auto matrices = collect_db_matrices(ctx, effective_ids, preloaded_ids);
    if (matrices.empty()) {
        if (ctx.buffer_pool() != nullptr) {
            ctx.stop_weight_prefetch();
            ctx.set_buffer_pool(nullptr);
        }
        return;
    }

    if (ctx.buffer_pool() == nullptr) {
        auto buffer_pool = std::make_unique<BufferPool>(
            config.arena_size_mb,
            storage,
            config.prefetch_window
        );
        ctx.set_buffer_pool(std::move(buffer_pool));
    }

    ctx.buffer_pool()->set_prefetch_ring(config.ring);
    ctx.buffer_pool()->set_prefetch_simulation(config.simulate_prefetch, config.simulate_get_latency_ms);

    if (!graph_path.empty()) {
        auto provider = create_graph_prefetch_provider(graph, matrices, config.graph_max_nodes);
        ctx.buffer_pool()->start_prefetch_provider(std::move(provider), matrices);
        return;
    }

    auto sequence = build_prefetch_sequence(effective_ids, matrices);
    if (sequence.empty()) {
        if (ctx.buffer_pool() != nullptr) {
            ctx.stop_weight_prefetch();
            ctx.set_buffer_pool(nullptr);
        }
        return;
    }

    ctx.start_sequence_prefetch(sequence, matrices);
}

void preload_matrices(
    const std::unordered_set<std::string>& preload_ids,
    OperatorContext& ctx,
    const PreloadOptions& options
) {
    if (preload_ids.empty()) {
        return;
    }

    if (options.log) {
        std::cout << "Preloading " << preload_ids.size() << " matrices into memory..." << std::endl;
    }

    SimpleDBStorage* storage = ctx.storage();
    if (!storage) {
        if (options.log) {
            std::cerr << "Error: Storage not available for preloading" << std::endl;
        }
        return;
    }

    for (const auto& matrix_id : preload_ids) {
        try {
            auto existing = ctx.get_in_memory(matrix_id);
            if (existing) {
                if (options.log) {
                    std::cout << "  Matrix " << matrix_id << " already in memory, skipping" << std::endl;
                }
                continue;
            }

            auto norm_weight = ctx.get_norm_weight(matrix_id);
            if (norm_weight) {
                if (options.log) {
                    std::cout << "  Matrix " << matrix_id
                              << " already available as resident norm weight, skipping" << std::endl;
                }
                continue;
            }

            auto block_matrix = ctx.resolve_block_matrix(matrix_id);
            if (!block_matrix) {
                if (options.log) {
                    std::cerr << "  Warning: Matrix " << matrix_id
                              << " not found in database, skipping" << std::endl;
                }
                continue;
            }

            auto dense_data = read_matrix_from_storage(block_matrix, storage);
            DType dtype = block_matrix->dtype();
            ctx.store_in_memory(matrix_id, block_matrix->shape(), dtype, std::move(dense_data));

            (void)dtype;
        } catch (const std::exception& e) {
            if (options.log) {
                std::cerr << "  Error preloading " << matrix_id << ": " << e.what() << std::endl;
            }
        }
    }

    if (options.log) {
        std::cout << "Preloading complete" << std::endl;
    }
}

} // namespace kvtensor
