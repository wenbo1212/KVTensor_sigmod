#include "kvtensor/prefetch_graph_provider.hpp"
#include <iostream>

namespace kvtensor {
namespace {

struct MatrixChunkInfo {
    SplitMode split_mode = SplitMode::COLUMN;
    int64_t num_chunks = 0;
};

class GraphPrefetchSequenceProvider : public PrefetchSequenceProvider {
public:
    GraphPrefetchSequenceProvider(const PrefetchGraph& graph,
                                  const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices,
                                  size_t max_nodes)
        : graph_(graph),
          max_nodes_(max_nodes) {
        for (const auto& [matrix_id, matrix] : matrices) {
            if (!matrix) {
                continue;
            }
            MatrixChunkInfo info;
            info.split_mode = matrix->split_mode();
            if (info.split_mode == SplitMode::COLUMN) {
                info.num_chunks = matrix->num_col_chunks();
            } else {
                info.num_chunks = matrix->num_row_chunks();
            }
            chunk_info_[matrix_id] = info;
        }
        reset();
    }

    bool next(ChunkKey* out_key) override {
        if (!out_key || finished_) {
            return false;
        }
        while (true) {
            if (!current_matrix_.empty() && current_chunk_idx_ < current_chunk_count_) {
                *out_key = ChunkKey(current_matrix_, current_chunk_idx_, current_split_mode_, current_step_);
                ++current_chunk_idx_;
                ++position_;
                return true;
            }

            std::string next_matrix;
            int64_t next_step = 0;
            if (!next_node(&next_matrix, &next_step)) {
                finished_ = true;
                return false;
            }
            if (nodes_emitted_ >= max_nodes_) {
                std::cerr << "[PrefetchGraph] Reached max_nodes=" << max_nodes_
                          << ", stopping traversal" << std::endl;
                finished_ = true;
                return false;
            }
            ++nodes_emitted_;

            auto info_it = chunk_info_.find(next_matrix);
            if (info_it == chunk_info_.end()) {
                std::cerr << "[PrefetchGraph] Unknown matrix id: " << next_matrix << std::endl;
                continue;
            }
            current_matrix_ = next_matrix;
            current_step_ = next_step;
            current_chunk_idx_ = 0;
            current_chunk_count_ = info_it->second.num_chunks;
            current_split_mode_ = info_it->second.split_mode;
            if (current_chunk_count_ <= 0) {
                continue;
            }
        }
    }

    void reset() override {
        runtime_links_.clear();
        runtime_links_.reserve(graph_.links().size());
        for (const auto& link : graph_.links()) {
            RuntimeLink rt;
            rt.unlimited = (link.step == 0);
            rt.remaining = link.step;
            runtime_links_.push_back(rt);
        }
        current_node_.clear();
        started_ = false;
        finished_ = false;
        nodes_emitted_ = 0;
        current_matrix_.clear();
        current_chunk_idx_ = 0;
        current_chunk_count_ = 0;
        current_split_mode_ = SplitMode::COLUMN;
        current_step_ = 0;
        position_ = 0;
        loop_start_.clear();
        current_loop_step_ = 0;
    }

    size_t position() const override {
        return position_;
    }

    size_t length_hint() const override {
        return 0;
    }

private:
    struct RuntimeLink {
        bool unlimited = false;
        int64_t remaining = 0;
    };

    bool next_node(std::string* out_id, int64_t* out_step) {
        if (!out_id || !out_step) {
            return false;
        }
        if (!started_) {
            current_node_ = graph_.start_id();
            if (current_node_.empty()) {
                return false;
            }
            started_ = true;
            *out_id = current_node_;
            *out_step = 0;
            return true;
        }
        auto node_it = graph_.nodes().find(current_node_);
        if (node_it == graph_.nodes().end()) {
            return false;
        }
        const auto& node = node_it->second;
        for (size_t link_idx : node.outgoing) {
            const auto& link = graph_.links()[link_idx];
            auto& runtime = runtime_links_[link_idx];
            if (runtime.unlimited || runtime.remaining > 0) {
                if (!runtime.unlimited && runtime.remaining > 0) {
                    runtime.remaining -= 1;
                }
                if (link.step > 0 && loop_start_.empty()) {
                    loop_start_ = link.from;
                }
                if (link.step > 0 && !loop_start_.empty() && link.to == loop_start_) {
                    ++current_loop_step_;
                }
                if (link.step > 0) {
                    *out_step = current_loop_step_;
                } else {
                    *out_step = 0;
                }
                current_node_ = link.to;
                *out_id = current_node_;
                return true;
            }
        }
        return false;
    }

    PrefetchGraph graph_;
    std::unordered_map<std::string, MatrixChunkInfo> chunk_info_;
    size_t max_nodes_ = 0;

    std::vector<RuntimeLink> runtime_links_;
    std::string current_node_;
    bool started_ = false;
    bool finished_ = false;
    size_t nodes_emitted_ = 0;

    std::string current_matrix_;
    int64_t current_chunk_idx_ = 0;
    int64_t current_chunk_count_ = 0;
    SplitMode current_split_mode_ = SplitMode::COLUMN;
    int64_t current_step_ = 0;
    size_t position_ = 0;

    std::string loop_start_;
    int64_t current_loop_step_ = 0;
};

} // namespace

std::unique_ptr<PrefetchSequenceProvider> create_graph_prefetch_provider(
    const PrefetchGraph& graph,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices,
    size_t max_nodes
) {
    return std::make_unique<GraphPrefetchSequenceProvider>(graph, matrices, max_nodes);
}

} // namespace kvtensor
