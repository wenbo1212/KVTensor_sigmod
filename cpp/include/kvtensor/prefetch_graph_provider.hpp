#pragma once

#include "kvtensor/bufferpool.hpp"
#include "kvtensor/prefetch_graph.hpp"
#include "kvtensor/matrix.hpp"
#include <memory>
#include <unordered_map>

namespace kvtensor {

std::unique_ptr<PrefetchSequenceProvider> create_graph_prefetch_provider(
    const PrefetchGraph& graph,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices,
    size_t max_nodes = 1000000
);

} // namespace kvtensor
