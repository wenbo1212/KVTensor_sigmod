#include "kvtensor/context.hpp"
#include "kvtensor/model_utils.hpp"
#include "kvtensor/storage.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace kvtensor;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <db_path> <output_dir> [matrix_id...]\n"
                  << "If matrix_id list is omitted, all matrices from metadata.jsonl are exported.\n";
        return 1;
    }

    std::string db_path = argv[1];
    std::filesystem::path out_dir = argv[2];

    try {
        SimpleDBStorage storage(db_path);
        MatrixRegistry registry(&storage);

        std::vector<std::string> ids;
        if (argc > 3) {
            for (int i = 3; i < argc; ++i) {
                ids.emplace_back(argv[i]);
            }
        } else {
            ids = registry.list_matrix_ids();
        }

        if (ids.empty()) {
            std::cerr << "No matrices found to export.\n";
            return 1;
        }

        std::filesystem::create_directories(out_dir);

        for (const auto& id : ids) {
            try {
                auto matrix = registry.get_matrix(id);
                auto data = read_matrix_from_storage(matrix, &storage); // dense, row-major
                std::filesystem::path bin_path = out_dir / (id + ".bin");
                std::ofstream out(bin_path, std::ios::binary | std::ios::trunc);
                if (!out) {
                    throw std::runtime_error("Failed to open output file: " + bin_path.string());
                }
                out.write(reinterpret_cast<const char*>(data.data()),
                          static_cast<std::streamsize>(data.size()));
                std::cout << "Exported " << id << " -> " << bin_path << " (" << data.size() / 1e6 << " MB)\n";
            } catch (const std::exception& e) {
                std::cerr << "Skip " << id << " (error: " << e.what() << ")\n";
                continue;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
