#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvtensor {

class PrefetchGraph {
public:
    struct Link {
        std::string from;
        std::string to;
        int64_t step = 0;
    };

    struct Node {
        std::string id;
        std::vector<size_t> outgoing;
    };

    bool load_from_file(const std::string& path, std::string* error = nullptr);

    std::vector<std::string> build_sequence(size_t max_nodes = 100000,
                                            std::string* error = nullptr) const;

    const std::string& start_id() const { return start_id_; }
    const std::unordered_map<std::string, Node>& nodes() const { return nodes_; }
    const std::vector<Link>& links() const { return links_; }

private:
    std::unordered_map<std::string, Node> nodes_;
    std::vector<Link> links_;
    std::string start_id_;
};

} // namespace kvtensor
