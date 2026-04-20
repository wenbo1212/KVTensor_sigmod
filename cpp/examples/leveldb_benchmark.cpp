/**
 * LevelDB Cursor Read Benchmark
 * 
 * Measures the time for each cursor read operation to diagnose I/O performance issues.
 * Uses the existing weights database.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <memory>
#include <leveldb/db.h>
#include <leveldb/iterator.h>
#include <leveldb/cache.h>

struct ReadStats {
    double time_us;
    size_t value_size;
    std::string key;
};

void print_histogram(const std::vector<double>& times_us) {
    if (times_us.empty()) return;
    
    // Define buckets (in microseconds)
    std::vector<std::pair<double, int>> buckets = {
        {100, 0},      // < 0.1ms
        {1000, 0},     // < 1ms
        {5000, 0},     // < 5ms
        {10000, 0},    // < 10ms
        {20000, 0},    // < 20ms
        {50000, 0},    // < 50ms
        {100000, 0},   // < 100ms
        {500000, 0},   // < 500ms
        {1000000, 0},  // < 1s
        {1e9, 0}       // >= 1s
    };
    
    for (double t : times_us) {
        for (auto& [threshold, count] : buckets) {
            if (t < threshold) {
                count++;
                break;
            }
        }
    }
    
    std::cout << "\nLatency Histogram:" << std::endl;
    std::cout << "  < 0.1ms:  " << buckets[0].second << std::endl;
    std::cout << "  < 1ms:    " << buckets[1].second << std::endl;
    std::cout << "  < 5ms:    " << buckets[2].second << std::endl;
    std::cout << "  < 10ms:   " << buckets[3].second << std::endl;
    std::cout << "  < 20ms:   " << buckets[4].second << std::endl;
    std::cout << "  < 50ms:   " << buckets[5].second << std::endl;
    std::cout << "  < 100ms:  " << buckets[6].second << std::endl;
    std::cout << "  < 500ms:  " << buckets[7].second << std::endl;
    std::cout << "  < 1s:     " << buckets[8].second << std::endl;
    std::cout << "  >= 1s:    " << buckets[9].second << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <leveldb_path> [--prefix <prefix>] [--limit <n>] [--verbose] [--method <cursor|get|both>]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./weights_single_db --prefix transformer.0 --limit 100 --method both" << std::endl;
        std::cerr << "Methods:" << std::endl;
        std::cerr << "  cursor: Use iterator-based sequential access (current approach)" << std::endl;
        std::cerr << "  get:    Use db.Get() for each key (random access)" << std::endl;
        std::cerr << "  both:   Compare both methods (default)" << std::endl;
        return 1;
    }
    
    std::string db_path = argv[1];
    std::string prefix = "";
    int limit = -1;
    bool verbose = false;
    std::string method = "both";
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (arg == "--limit" && i + 1 < argc) {
            limit = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--method" && i + 1 < argc) {
            method = argv[++i];
        }
    }
    
    std::cout << "=== LevelDB Read Benchmark: Cursor vs Get() ===" << std::endl;
    std::cout << "Database: " << db_path << std::endl;
    std::cout << "Method: " << method << std::endl;
    if (!prefix.empty()) {
        std::cout << "Prefix filter: " << prefix << std::endl;
    }
    if (limit > 0) {
        std::cout << "Limit: " << limit << " reads" << std::endl;
    }
    std::cout << std::endl;
    
    // Open database with read optimizations
    leveldb::Options options;
    options.create_if_missing = false;
    
    // 256MB cache (same as runtime)
    std::unique_ptr<leveldb::Cache> cache(leveldb::NewLRUCache(256 * 1024 * 1024));
    options.block_cache = cache.get();
    
    leveldb::DB* db_ptr = nullptr;
    leveldb::Status status = leveldb::DB::Open(options, db_path, &db_ptr);
    
    if (!status.ok()) {
        std::cerr << "Failed to open database: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::unique_ptr<leveldb::DB> db(db_ptr);
    
    // First, collect all keys (for Get() method)
    std::vector<std::string> keys;
    {
        leveldb::ReadOptions read_opts;
        read_opts.fill_cache = true;
        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(read_opts));
        
        if (!prefix.empty()) {
            it->Seek(prefix);
        } else {
            it->SeekToFirst();
        }
        
        while (it->Valid()) {
            std::string key = it->key().ToString();
            if (!prefix.empty() && key.find(prefix) != 0) {
                break;
            }
            keys.push_back(key);
            if (limit > 0 && static_cast<int>(keys.size()) >= limit) {
                break;
            }
            it->Next();
        }
    }
    
    if (keys.empty()) {
        std::cout << "No entries found!" << std::endl;
        return 0;
    }
    
    std::cout << "Collected " << keys.size() << " keys for benchmarking" << std::endl;
    std::cout << std::endl;
    
    // Benchmark function for cursor-based access
    auto benchmark_cursor = [&](bool measure_copy = false) -> std::pair<std::vector<ReadStats>, double> {
        std::vector<ReadStats> stats;
        size_t total_bytes = 0;
        
        leveldb::ReadOptions read_opts;
        read_opts.fill_cache = true;
        std::unique_ptr<leveldb::Iterator> it(db->NewIterator(read_opts));
        
        if (!prefix.empty()) {
            it->Seek(prefix);
        } else {
            it->SeekToFirst();
        }
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        int count = 0;
        while (it->Valid() && count < static_cast<int>(keys.size())) {
            std::string key = it->key().ToString();
            if (!prefix.empty() && key.find(prefix) != 0) {
                break;
            }
            
            // Measure value access (zero-copy pointer access)
            auto start = std::chrono::high_resolution_clock::now();
            leveldb::Slice value = it->value();
            size_t value_size = value.size();
            const char* value_ptr = value.data();
            
            // Access data to force read
            volatile char dummy = value_ptr[0];
            (void)dummy;
            
            if (measure_copy) {
                // Measure memcpy time
                std::vector<uint8_t> copy_buffer(value_size);
                auto copy_start = std::chrono::high_resolution_clock::now();
                std::memcpy(copy_buffer.data(), value_ptr, value_size);
                auto copy_end = std::chrono::high_resolution_clock::now();
                double copy_time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count() / 1000.0;
                stats.push_back({copy_time_us, value_size, key});
            } else {
                auto end = std::chrono::high_resolution_clock::now();
                double time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
                stats.push_back({time_us, value_size, key});
            }
            
            total_bytes += value_size;
            count++;
            
            if (limit > 0 && count >= limit) {
                break;
            }
            
            it->Next();
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        
        return {stats, total_time_ms};
    };
    
    // Benchmark function for Get()-based access
    auto benchmark_get = [&](bool measure_copy = false) -> std::pair<std::vector<ReadStats>, double> {
        std::vector<ReadStats> stats;
        size_t total_bytes = 0;
        
        leveldb::ReadOptions read_opts;
        read_opts.fill_cache = true;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        for (const auto& key : keys) {
            // Measure Get() call
            auto start = std::chrono::high_resolution_clock::now();
            std::string value;
            leveldb::Status s = db->Get(read_opts, key, &value);
            
            if (!s.ok()) {
                continue;
            }
            
            size_t value_size = value.size();
            const char* value_ptr = value.data();
            
            // Access data to force read
            volatile char dummy = value_ptr[0];
            (void)dummy;
            
            if (measure_copy) {
                // Measure memcpy time (though Get() already copied)
                std::vector<uint8_t> copy_buffer(value_size);
                auto copy_start = std::chrono::high_resolution_clock::now();
                std::memcpy(copy_buffer.data(), value_ptr, value_size);
                auto copy_end = std::chrono::high_resolution_clock::now();
                double copy_time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count() / 1000.0;
                stats.push_back({copy_time_us, value_size, key});
            } else {
                auto end = std::chrono::high_resolution_clock::now();
                double time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
                stats.push_back({time_us, value_size, key});
            }
            
            total_bytes += value_size;
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        
        return {stats, total_time_ms};
    };
    
    // Print statistics helper
    auto print_stats = [&](const std::string& method_name, const std::vector<ReadStats>& stats, double total_time_ms) {
        if (stats.empty()) return;
        
        std::vector<double> times;
        size_t total_bytes = 0;
        for (const auto& s : stats) {
            times.push_back(s.time_us);
            total_bytes += s.value_size;
        }
        std::sort(times.begin(), times.end());
        
        double min_time = times.front();
        double max_time = times.back();
        double median_time = times[times.size() / 2];
        double p95_time = times[static_cast<size_t>(times.size() * 0.95)];
        double p99_time = times[static_cast<size_t>(times.size() * 0.99)];
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        double throughput_mbs = (total_bytes / 1024.0 / 1024.0) / (total_time_ms / 1000.0);
        
        std::cout << "\n=== " << method_name << " Results ===" << std::endl;
        std::cout << "Total reads: " << stats.size() << std::endl;
        std::cout << "Total bytes: " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Total time:  " << total_time_ms << " ms" << std::endl;
        std::cout << "Throughput:  " << std::fixed << std::setprecision(2) << throughput_mbs << " MB/s" << std::endl;
        
        std::cout << "\nLatency:" << std::endl;
        std::cout << "  Min:    " << std::setprecision(2) << (min_time / 1000.0) << " ms" << std::endl;
        std::cout << "  Avg:    " << (avg_time / 1000.0) << " ms" << std::endl;
        std::cout << "  Median: " << (median_time / 1000.0) << " ms" << std::endl;
        std::cout << "  P95:    " << (p95_time / 1000.0) << " ms" << std::endl;
        std::cout << "  P99:    " << (p99_time / 1000.0) << " ms" << std::endl;
        std::cout << "  Max:    " << (max_time / 1000.0) << " ms" << std::endl;
    };
    
    // Run benchmarks
    if (method == "cursor" || method == "both") {
        std::cout << "=== Benchmarking Cursor-Based Access (Zero-Copy Pointer) ===" << std::endl;
        auto [cursor_stats, cursor_time] = benchmark_cursor(false);
        print_stats("Cursor (Zero-Copy)", cursor_stats, cursor_time);
        
        std::cout << "\n=== Benchmarking Cursor + Memcpy (Current Approach) ===" << std::endl;
        auto [cursor_copy_stats, cursor_copy_time] = benchmark_cursor(true);
        print_stats("Cursor + Memcpy", cursor_copy_stats, cursor_copy_time);
    }
    
    if (method == "get" || method == "both") {
        std::cout << "\n=== Benchmarking Get()-Based Access ===" << std::endl;
        auto [get_stats, get_time] = benchmark_get(false);
        print_stats("Get()", get_stats, get_time);
    }
    
    // Comparison
    if (method == "both") {
        auto [cursor_stats, cursor_time] = benchmark_cursor(false);
        auto [get_stats, get_time] = benchmark_get(false);
        
        std::cout << "\n=== Comparison ===" << std::endl;
        std::cout << "Cursor time: " << cursor_time << " ms" << std::endl;
        std::cout << "Get() time:  " << get_time << " ms" << std::endl;
        if (cursor_time > 0) {
            std::cout << "Speedup:     " << std::fixed << std::setprecision(2) 
                      << (get_time / cursor_time) << "x " 
                      << (get_time > cursor_time ? "(cursor faster)" : "(get faster)") << std::endl;
        }
        
        // Note about zero-copy
        std::cout << "\n=== Zero-Copy Analysis ===" << std::endl;
        std::cout << "Cursor: Provides zero-copy access via iterator.value().data() pointer" << std::endl;
        std::cout << "        BUT: Pointer becomes invalid when iterator advances or is destroyed" << std::endl;
        std::cout << "Get():   Always copies data into std::string (no zero-copy option)" << std::endl;
        std::cout << "        Data is safe to use after Get() returns" << std::endl;
    }
    
    return 0;
}
