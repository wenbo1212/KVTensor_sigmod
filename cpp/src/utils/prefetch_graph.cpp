#include "kvtensor/prefetch_graph.hpp"
#include <fstream>
#include <sstream>
#include <cctype>

namespace kvtensor {
namespace {

std::string trim(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

} // namespace

bool PrefetchGraph::load_from_file(const std::string& path, std::string* error) {
    nodes_.clear();
    links_.clear();
    start_id_.clear();

    std::ifstream in(path);
    if (!in.is_open()) {
        if (error) {
            *error = "Failed to open prefetch graph file: " + path;
        }
        return false;
    }

    auto ensure_node = [&](const std::string& id) -> Node& {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            Node node;
            node.id = id;
            auto result = nodes_.emplace(id, std::move(node));
            return result.first->second;
        }
        return it->second;
    };

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        std::istringstream iss(trimmed);
        std::string keyword;
        iss >> keyword;
        if (keyword == "node") {
            std::string id;
            iss >> id;
            if (id.empty()) {
                if (error) {
                    *error = "Line " + std::to_string(line_no) + ": node missing id";
                }
                return false;
            }
            ensure_node(id);
        } else if (keyword == "start") {
            std::string id;
            iss >> id;
            if (id.empty()) {
                if (error) {
                    *error = "Line " + std::to_string(line_no) + ": start missing id";
                }
                return false;
            }
            start_id_ = id;
            ensure_node(id);
        } else if (keyword == "link") {
            std::string from;
            std::string to;
            int64_t step = 0;
            if (!(iss >> from >> to >> step)) {
                if (error) {
                    *error = "Line " + std::to_string(line_no) + ": link requires <from> <to> <step>";
                }
                return false;
            }
            if (step < 0) {
                if (error) {
                    *error = "Line " + std::to_string(line_no) + ": link step must be >= 0";
                }
                return false;
            }
            ensure_node(from);
            ensure_node(to);
            size_t link_idx = links_.size();
            links_.push_back(Link{from, to, step});
            nodes_[from].outgoing.push_back(link_idx);
        } else {
            if (error) {
                *error = "Line " + std::to_string(line_no) + ": unknown directive '" + keyword + "'";
            }
            return false;
        }
    }

    if (start_id_.empty()) {
        if (error) {
            *error = "Prefetch graph missing start node";
        }
        return false;
    }
    return true;
}

std::vector<std::string> PrefetchGraph::build_sequence(size_t max_nodes, std::string* error) const {
    std::vector<std::string> sequence;
    if (start_id_.empty()) {
        if (error) {
            *error = "Prefetch graph has no start node";
        }
        return sequence;
    }
    auto start_it = nodes_.find(start_id_);
    if (start_it == nodes_.end()) {
        if (error) {
            *error = "Start node not found in graph: " + start_id_;
        }
        return sequence;
    }

    struct RuntimeLink {
        bool unlimited = false;
        int64_t remaining = 0;
    };
    std::vector<RuntimeLink> runtime_links;
    runtime_links.reserve(links_.size());
    for (const auto& link : links_) {
        RuntimeLink rt;
        rt.unlimited = (link.step == 0);
        rt.remaining = link.step;
        runtime_links.push_back(rt);
    }

    std::string current = start_id_;
    sequence.push_back(current);

    while (sequence.size() < max_nodes) {
        auto node_it = nodes_.find(current);
        if (node_it == nodes_.end()) {
            break;
        }
        const Node& node = node_it->second;
        bool advanced = false;
        for (size_t link_idx : node.outgoing) {
            const auto& link = links_[link_idx];
            auto& runtime = runtime_links[link_idx];
            if (runtime.unlimited || runtime.remaining > 0) {
                if (!runtime.unlimited && runtime.remaining > 0) {
                    runtime.remaining -= 1;
                }
                current = link.to;
                sequence.push_back(current);
                advanced = true;
                break;
            }
        }
        if (!advanced) {
            break;
        }
    }

    if (sequence.size() >= max_nodes) {
        if (error) {
            *error = "Prefetch graph exceeded max_nodes (possible infinite loop)";
        }
        sequence.clear();
    }
    return sequence;
}

} // namespace kvtensor
