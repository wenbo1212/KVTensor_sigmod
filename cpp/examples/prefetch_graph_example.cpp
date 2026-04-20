#include "kvtensor/prefetch_graph.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

struct Args {
    std::string graph_path;
    size_t max_nodes = 100000;

    static Args parse(int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--graph" && i + 1 < argc) {
                args.graph_path = argv[++i];
            } else if (arg == "--max-nodes" && i + 1 < argc) {
                args.max_nodes = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--help" || arg == "-h") {
                print_help();
                std::exit(0);
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_help();
                std::exit(1);
            }
        }
        if (args.graph_path.empty()) {
            std::cerr << "Missing --graph path\n";
            print_help();
            std::exit(1);
        }
        return args;
    }

    static void print_help() {
        std::cout << "Usage: prefetch_graph_example --graph PATH [--max-nodes N]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --graph PATH     Prefetch graph definition file\n";
        std::cout << "  --max-nodes N    Maximum nodes to traverse (default: 100000)\n";
        std::cout << "  --help, -h       Show this help message\n";
    }
};

int main(int argc, char** argv) {
    Args args = Args::parse(argc, argv);

    kvtensor::PrefetchGraph graph;
    std::string error;
    if (!graph.load_from_file(args.graph_path, &error)) {
        std::cerr << "Error: " << error << std::endl;
        return 1;
    }

    auto sequence = graph.build_sequence(args.max_nodes, &error);
    if (sequence.empty()) {
        std::cerr << "Error: " << error << std::endl;
        return 1;
    }

    for (size_t i = 0; i < sequence.size(); ++i) {
        std::cout << i << " " << sequence[i] << "\n";
    }

    return 0;
}
