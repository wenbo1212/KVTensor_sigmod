#pragma once

#include "kvtensor/types.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Profiling functions - call to see bufferpool performance stats
void print_bufferpool_profile();
void reset_bufferpool_profile();

namespace kvtensor {

// Forward declarations
class BlockMatrix;
class SimpleDBStorage;

// ChunkKey: Immutable key identifying a chunk in the buffer pool
struct ChunkKey {
    std::string matrix_id;
    int64_t chunk_idx;
    SplitMode split_mode;
    int64_t step;
    
    ChunkKey(const std::string& id, int64_t idx, SplitMode mode, int64_t step_val = 0)
        : matrix_id(id), chunk_idx(idx), split_mode(mode), step(step_val) {}
    
    bool operator==(const ChunkKey& other) const {
        return matrix_id == other.matrix_id &&
               chunk_idx == other.chunk_idx &&
               split_mode == other.split_mode &&
               step == other.step;
    }
    
    std::string to_string() const {
        return matrix_id + ":" + split_mode_to_string(split_mode) + ":" +
               std::to_string(chunk_idx) + ":step=" + std::to_string(step);
    }
};

// Hash function for ChunkKey
struct ChunkKeyHash {
    std::size_t operator()(const ChunkKey& key) const {
        std::size_t h1 = std::hash<std::string>{}(key.matrix_id);
        std::size_t h2 = std::hash<int64_t>{}(key.chunk_idx);
        std::size_t h3 = std::hash<int>{}(static_cast<int>(key.split_mode));
        std::size_t h4 = std::hash<int64_t>{}(key.step);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

class PrefetchSequenceProvider {
public:
    virtual ~PrefetchSequenceProvider() = default;
    virtual bool next(ChunkKey* out_key) = 0;
    virtual void reset() = 0;
    virtual size_t position() const = 0;
    virtual size_t length_hint() const = 0;
};

// ChunkView: Lightweight view into chunk data
struct ChunkView {
    const uint8_t* data;
    size_t size;
    Shape shape;
    DType dtype;
    
    ChunkView(const uint8_t* d, size_t s, Shape sh, DType dt)
        : data(d), size(s), shape(sh), dtype(dt) {}
};

struct PinnedChunk {
    const uint8_t* data = nullptr;
    size_t size = 0;
    Shape shape = {0, 0};
    DType dtype = DType::FLOAT32;

    PinnedChunk() = default;
    PinnedChunk(const uint8_t* d,
                size_t s,
                Shape sh,
                DType dt,
                ChunkKey key,
                size_t slot_idx,
                class BufferPool* pool,
                std::shared_ptr<void> owned = nullptr);
    ~PinnedChunk();

    PinnedChunk(const PinnedChunk&) = delete;
    PinnedChunk& operator=(const PinnedChunk&) = delete;
    PinnedChunk(PinnedChunk&& other) noexcept;
    PinnedChunk& operator=(PinnedChunk&& other) noexcept;

    void release();
    bool valid() const { return data != nullptr; }

private:
    ChunkKey key_{"", 0, SplitMode::ROW, 0};
    size_t slot_idx_ = 0;
    class BufferPool* pool_ = nullptr;
    bool active_ = false;
    std::shared_ptr<void> owned_;
};

// BufferPool: Pointer pool cache with direct Get() loading and background prefetching
class BufferPool {
public:
    BufferPool(
        size_t max_memory_mb,
        SimpleDBStorage* storage,
        size_t prefetch_window = 32
    );
    
    ~BufferPool();
    
    // Non-copyable, non-moveable
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;
    
    // Get chunk from pool (RAII pinned)
    PinnedChunk get_chunk(
        const std::string& matrix_id,
        int64_t chunk_idx,
        const BlockMatrix& matrix,
        int64_t step = 0,
        double timeout_seconds = 30.0
    );

    // Try to get without blocking; returns nullopt if not yet available
    std::optional<PinnedChunk> try_get_chunk(const ChunkKey& key);

    PinnedChunk get_chunk_pinned(
        const std::string& matrix_id,
        int64_t chunk_idx,
        const BlockMatrix& matrix,
        int64_t step = 0,
        double timeout_seconds = 30.0
    );
    
    // Preload initial chunks synchronously
    void preload_initial_chunks(
        const std::vector<ChunkKey>& sequence,
        const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
    );
    
    // Start background prefetch thread
    void start_sequence_prefetch(
        const std::vector<ChunkKey>& sequence,
        const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
    );

    // Start background prefetch thread using a sequence provider
    void start_prefetch_provider(
        std::unique_ptr<PrefetchSequenceProvider> provider,
        const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
    );
    
    // Stop prefetch thread and clear state
    void stop_prefetch();
    
    // Sync consumption position (called by operators when they consume a chunk)
    void sync_consumption_position(const ChunkKey& chunk_key);

    // Enable/disable ring prefetch (wrap to start when reaching end)
    void set_prefetch_ring(bool enabled);

    // Configure simulated prefetch (skip DB reads and inject latency)
    void set_prefetch_simulation(bool enabled, uint64_t get_latency_ms);
    
    // Clear all chunks from buffer pool
    void clear();
    
    // Get statistics
    struct Stats {
        size_t cached_chunks;
        size_t slot_capacity;
        size_t memory_used_bytes;
        size_t memory_total_bytes;
        bool prefetch_active;
        size_t consumption_position;
        size_t prefetch_position;
        size_t sequence_length;
    };
    
    Stats get_stats() const;
    
    // Get profiling statistics
    struct ProfileStats {
        uint64_t get_chunk_calls;
        uint64_t cache_hits;
        uint64_t cache_misses;
        uint64_t total_wait_time_ns;
        uint64_t max_wait_time_ns;
        uint64_t evict_count;
        uint64_t prefetch_get_calls;
        uint64_t prefetch_get_time_ns;
    };
    
    ProfileStats get_profile_stats() const;
    
    // Get prefetch sequence (for debugging)
    std::vector<ChunkKey> get_prefetch_sequence() const;
    
    // Get list of currently cached chunks (for debugging)
    std::vector<ChunkKey> get_cached_chunks() const;

private:
    friend struct PinnedChunk;
    void unpin_chunk(const ChunkKey& chunk_key, size_t slot_idx);

    // LRU entry for O(1) eviction
    struct LRUEntry {
        ChunkKey key;
        size_t chunk_size;
        
        LRUEntry(const ChunkKey& k, size_t sz) : key(k), chunk_size(sz) {}
    };

    
    // Prefetch worker thread
    void prefetch_worker();
    bool next_prefetch_key(ChunkKey* out_key, size_t* out_pos = nullptr);

    void prepare_simulated_chunks();

    
    // Clear buffer pool (must be called with lock held)
    void clear_buffer_pool();
    
    // Clear prefetch state (must be called with lock held)
    void clear_prefetch_state();

    enum class SlotState { Empty, Filling, Ready };
    struct Slot {
        AlignedString buffer;
        size_t used = 0;
        Shape shape{0, 0};
        DType dtype = DType::FLOAT32;
        ChunkKey key{"", 0, SplitMode::ROW};
        std::atomic<uint32_t> refs{0};
        std::atomic<SlotState> state{SlotState::Empty};
        std::mutex mtx;
        std::condition_variable cv;
        size_t seq_pos = 0;

        Slot() = default;
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
        Slot(Slot&& other) noexcept
            : buffer(std::move(other.buffer)),
              used(other.used),
              shape(other.shape),
              dtype(other.dtype),
              key(std::move(other.key)),
              refs(other.refs.load(std::memory_order_relaxed)),
              state(other.state.load(std::memory_order_relaxed)),
              seq_pos(other.seq_pos) {}
        Slot& operator=(Slot&& other) noexcept {
            if (this != &other) {
                buffer = std::move(other.buffer);
                used = other.used;
                shape = other.shape;
                dtype = other.dtype;
                key = std::move(other.key);
                refs.store(other.refs.load(std::memory_order_relaxed), std::memory_order_relaxed);
                state.store(other.state.load(std::memory_order_relaxed), std::memory_order_relaxed);
                seq_pos = other.seq_pos;
            }
            return *this;
        }
    };
    std::optional<std::pair<size_t, Slot*>> pin_ready_slot(const ChunkKey& key);

    struct SimulatedChunkShape {
        int64_t rows;
        int64_t cols;
        DType dtype;

        bool operator==(const SimulatedChunkShape& other) const {
            return rows == other.rows && cols == other.cols && dtype == other.dtype;
        }
    };

    struct SimulatedChunkShapeHash {
        std::size_t operator()(const SimulatedChunkShape& key) const {
            std::size_t h1 = std::hash<int64_t>{}(key.rows);
            std::size_t h2 = std::hash<int64_t>{}(key.cols);
            std::size_t h3 = std::hash<int>{}(static_cast<int>(key.dtype));
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    // Pointer pool structure
    std::vector<Slot> slots_;
    mutable std::unordered_map<ChunkKey, size_t, ChunkKeyHash> slot_index_;
    mutable std::mutex slot_index_mutex_;
    std::atomic<size_t> total_memory_used_{0};  // Thread-safe memory tracking
    size_t max_memory_;  // Maximum memory limit in bytes

    // Thread safety and coordination
    std::atomic<bool> prefetch_active_;
    std::condition_variable wait_cv_;
    std::mutex wait_mutex_;
    std::condition_variable memory_cv_;
    std::mutex memory_mutex_;
    
    // Prefetch state
    std::thread prefetch_thread_;
    std::vector<ChunkKey> prefetch_sequence_;
    std::unordered_map<ChunkKey, size_t, ChunkKeyHash> prefetch_index_;  // ChunkKey -> position in sequence
    std::unique_ptr<PrefetchSequenceProvider> sequence_provider_;
    size_t sequence_length_ = 0;
    std::unordered_map<std::string, std::shared_ptr<BlockMatrix>> matrix_registry_;
    size_t prefetch_window_;
    std::atomic<size_t> consumption_position_;
    std::atomic<size_t> prefetch_position_;  // Tracks how far prefetch has progressed
    std::atomic<bool> prefetch_ring_{false};

    // Storage reference
    SimpleDBStorage* storage_;

    // Simulation config
    std::atomic<bool> simulate_prefetch_{false};
    std::atomic<uint64_t> simulate_get_latency_ms_{0};
    std::unordered_map<SimulatedChunkShape, AlignedString, SimulatedChunkShapeHash> simulated_chunk_pool_;

};

} // namespace kvtensor
