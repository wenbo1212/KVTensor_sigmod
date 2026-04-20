#include "kvtensor/simpledb.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <leveldb/cache.h>
#include <leveldb/db.h>
#include <leveldb/iterator.h>
#include <memory>
#include <string>
#include <vector>

namespace {

struct BenchConfig {
    std::string simpledb_path;
    std::string leveldb_path;
    std::string prefix = "transformer.0";
    size_t limit = 100;
    std::string mode = "both";   // both|simpledb|leveldb
    std::string method = "both"; // both|iter|get
};

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto read_string = [&](std::string& target) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            target = argv[++i];
        };
        auto read_size = [&](size_t& target) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            target = static_cast<size_t>(std::stoull(argv[++i]));
        };
        if (arg == "--simpledb-path") {
            read_string(cfg.simpledb_path);
        } else if (arg == "--leveldb-path") {
            read_string(cfg.leveldb_path);
        } else if (arg == "--prefix") {
            read_string(cfg.prefix);
        } else if (arg == "--limit") {
            read_size(cfg.limit);
        } else if (arg == "--mode") {
            read_string(cfg.mode);
        } else if (arg == "--method") {
            read_string(cfg.method);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --simpledb-path PATH\n"
                      << "  --leveldb-path PATH\n"
                      << "  --prefix PREFIX\n"
                      << "  --limit N\n"
                      << "  --mode both|simpledb|leveldb\n"
                      << "  --method both|iter|get\n";
            std::exit(0);
        }
    }
    return cfg;
}

struct BenchResult {
    double ms = 0.0;
    size_t bytes = 0;
    size_t count = 0;
};

BenchResult bench_simpledb_iter(kvtensor::SimpleKVStore& db, const std::string& prefix, size_t limit) {
    BenchResult result;
    auto start = std::chrono::steady_clock::now();
    size_t count = 0;
    db.iterate_prefix(prefix, [&](const std::string&, const uint8_t*, size_t len) {
        result.bytes += len;
        count++;
        if (count >= limit) {
            return false;
        }
        return true;
    });
    auto end = std::chrono::steady_clock::now();
    result.count = count;
    result.ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

BenchResult bench_simpledb_get(kvtensor::SimpleKVStore& db, const std::vector<std::string>& keys) {
    BenchResult result;
    auto start = std::chrono::steady_clock::now();
    std::vector<uint8_t> value;
    for (const auto& key : keys) {
        if (db.get(key, &value)) {
            result.bytes += value.size();
            result.count++;
        }
    }
    auto end = std::chrono::steady_clock::now();
    result.ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

std::unique_ptr<leveldb::DB> open_leveldb(const std::string& path) {
    leveldb::Options options;
    options.create_if_missing = false;
    options.block_cache = leveldb::NewLRUCache(256 * 1024 * 1024);
    leveldb::DB* db_ptr = nullptr;
    leveldb::Status status = leveldb::DB::Open(options, path, &db_ptr);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open LevelDB: " + status.ToString());
    }
    return std::unique_ptr<leveldb::DB>(db_ptr);
}

std::vector<std::string> collect_leveldb_keys(leveldb::DB* db, const std::string& prefix, size_t limit) {
    std::vector<std::string> keys;
    leveldb::ReadOptions read_opts;
    std::unique_ptr<leveldb::Iterator> it(db->NewIterator(read_opts));
    for (it->Seek(prefix); it->Valid() && keys.size() < limit; it->Next()) {
        if (!it->key().starts_with(prefix)) {
            break;
        }
        keys.emplace_back(it->key().ToString());
    }
    return keys;
}

BenchResult bench_leveldb_iter(leveldb::DB* db, const std::string& prefix, size_t limit) {
    BenchResult result;
    leveldb::ReadOptions read_opts;
    std::unique_ptr<leveldb::Iterator> it(db->NewIterator(read_opts));
    auto start = std::chrono::steady_clock::now();
    for (it->Seek(prefix); it->Valid() && result.count < limit; it->Next()) {
        if (!it->key().starts_with(prefix)) {
            break;
        }
        result.bytes += it->value().size();
        result.count++;
    }
    auto end = std::chrono::steady_clock::now();
    result.ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

BenchResult bench_leveldb_get(leveldb::DB* db, const std::vector<std::string>& keys) {
    BenchResult result;
    leveldb::ReadOptions read_opts;
    auto start = std::chrono::steady_clock::now();
    for (const auto& key : keys) {
        std::string value;
        leveldb::Status s = db->Get(read_opts, key, &value);
        if (s.ok()) {
            result.bytes += value.size();
            result.count++;
        }
    }
    auto end = std::chrono::steady_clock::now();
    result.ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

void print_result(const std::string& label, const BenchResult& result) {
    double mb = result.bytes / (1024.0 * 1024.0);
    double mbps = result.ms > 0.0 ? (mb * 1000.0 / result.ms) : 0.0;
    std::cout << label << ": " << result.count << " chunks, "
              << mb << " MB, " << result.ms << " ms, "
              << mbps << " MB/s" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        BenchConfig cfg = parse_args(argc, argv);

        if (cfg.mode == "both" || cfg.mode == "simpledb") {
            if (cfg.simpledb_path.empty()) {
                throw std::runtime_error("simpledb path required");
            }
            kvtensor::SimpleKVStore db = kvtensor::SimpleKVStore::OpenReadOnly(cfg.simpledb_path);
            std::cout << "=== SimpleKVStore Benchmark ===" << std::endl;
            if (cfg.method == "both" || cfg.method == "iter") {
                auto result = bench_simpledb_iter(db, cfg.prefix, cfg.limit);
                print_result("SimpleDB iterate", result);
            }
            if (cfg.method == "both" || cfg.method == "get") {
                std::vector<std::string> keys;
                db.iterate_prefix(cfg.prefix, [&](const std::string& key, const uint8_t*, size_t) {
                    keys.push_back(key);
                    return keys.size() < cfg.limit;
                });
                auto result = bench_simpledb_get(db, keys);
                print_result("SimpleDB get", result);
            }
        }

        if (cfg.mode == "both" || cfg.mode == "leveldb") {
            if (cfg.leveldb_path.empty()) {
                throw std::runtime_error("leveldb path required");
            }
            auto db = open_leveldb(cfg.leveldb_path);
            std::cout << "=== LevelDB Benchmark ===" << std::endl;
            if (cfg.method == "both" || cfg.method == "iter") {
                auto result = bench_leveldb_iter(db.get(), cfg.prefix, cfg.limit);
                print_result("LevelDB iterate", result);
            }
            if (cfg.method == "both" || cfg.method == "get") {
                auto keys = collect_leveldb_keys(db.get(), cfg.prefix, cfg.limit);
                auto result = bench_leveldb_get(db.get(), keys);
                print_result("LevelDB get", result);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
