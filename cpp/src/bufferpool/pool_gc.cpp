#include "kvtensor/bufferpool.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <utility>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <sstream>
#include <random>
#include <cstring>

// Profiling counters
static std::atomic<uint64_t> g_get_chunk_calls{0};
static std::atomic<uint64_t> g_cache_hits{0};
static std::atomic<uint64_t> g_cache_misses{0};
static std::atomic<uint64_t> g_wait_time_ns{0};
static std::atomic<uint64_t> g_max_wait_ns{0};
static std::atomic<uint64_t> g_evict_count{0};
static std::atomic<uint64_t> g_prefetch_get_calls{0};
static std::atomic<uint64_t> g_prefetch_get_time_ns{0};
static constexpr uint64_t kSlowWaitNs = 50ULL * 1000ULL * 1000ULL;
static constexpr uint64_t kSlowGetNs = 50ULL * 1000ULL * 1000ULL;

void print_bufferpool_profile() {
    uint64_t calls = g_get_chunk_calls.load();
    uint64_t hits = g_cache_hits.load();
    uint64_t misses = g_cache_misses.load();
    double hit_rate = (calls > 0) ? (100.0 * hits / calls) : 0;
    double wait_ms = g_wait_time_ns.load() / 1000000.0;
    double max_wait_ms = g_max_wait_ns.load() / 1000000.0;

    uint64_t prefetch_gets = g_prefetch_get_calls.load();
    double avg_get_ms = (prefetch_gets > 0) ? (g_prefetch_get_time_ns.load() / 1000000.0) / prefetch_gets : 0;

    std::cout << "\n=== BufferPool Profile ===" << std::endl;
    std::cout << "get_chunk: " << calls << " calls" << std::endl;
    std::cout << "  cache hits:   " << hits << " (" << std::fixed << std::setprecision(1) << hit_rate << "%)" << std::endl;
    std::cout << "  cache misses: " << misses << " (waited for prefetch)" << std::endl;
    std::cout << "  total wait:   " << std::setprecision(2) << wait_ms << " ms" << std::endl;
    std::cout << "  max wait:     " << max_wait_ms << " ms (single call)" << std::endl;
    if (misses > 0) {
        double avg_wait_ms = wait_ms / misses;
        std::cout << "  avg wait:     " << std::setprecision(2) << avg_wait_ms << " ms per miss" << std::endl;
    }
    std::cout << "evictions: " << g_evict_count.load() << " chunks" << std::endl;
    std::cout << "prefetch (Get operations):" << std::endl;
    std::cout << "  total Get() calls: " << prefetch_gets << std::endl;
    if (prefetch_gets > 0) {
        std::cout << "  avg Get() time:  " << std::setprecision(3) << avg_get_ms << " ms per chunk" << std::endl;
        std::cout << "  total Get time:  " << std::setprecision(2) << (g_prefetch_get_time_ns.load() / 1000000.0) << " ms" << std::endl;
    }
    std::cout << "==========================\n" << std::endl;
}

void reset_bufferpool_profile() {
    g_get_chunk_calls.store(0);
    g_cache_hits.store(0);
    g_cache_misses.store(0);
    g_wait_time_ns.store(0);
    g_max_wait_ns.store(0);
    g_evict_count.store(0);
    g_prefetch_get_calls.store(0);
    g_prefetch_get_time_ns.store(0);
}

namespace kvtensor {

namespace {

static std::string format_chunk_key(const std::string& matrix_id, SplitMode split_mode, int64_t chunk_idx) {
    std::ostringstream oss;
    oss << matrix_id << ":" << (split_mode == SplitMode::COLUMN ? "col" : "row") << ":"
        << std::setfill('0') << std::setw(6) << chunk_idx;
    return oss.str();
}

static size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::BFLOAT16:
        case DType::FLOAT16: return 2;
        case DType::INT8: return 1;
        default: return 4;
    }
}

static void fill_random_buffer(AlignedString* buffer, DType dtype, std::mt19937& rng) {
    if (!buffer || buffer->empty()) {
        return;
    }

    char* raw = buffer->data();
    size_t bytes = buffer->size();

    switch (dtype) {
        case DType::FLOAT32: {
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            size_t count = bytes / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
                float value = dist(rng);
                std::memcpy(raw + i * sizeof(float), &value, sizeof(float));
            }
            break;
        }
        case DType::FLOAT16: {
            std::uniform_int_distribution<int> sign_dist(0, 1);
            std::uniform_int_distribution<int> exp_dist(1, 30);  // avoid 0/31 (denorm/inf)
            std::uniform_int_distribution<int> mant_dist(0, 1023);
            size_t count = bytes / sizeof(uint16_t);
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits = static_cast<uint16_t>(
                    (sign_dist(rng) << 15) |
                    (exp_dist(rng) << 10) |
                    mant_dist(rng)
                );
                std::memcpy(raw + i * sizeof(uint16_t), &bits, sizeof(uint16_t));
            }
            break;
        }
        case DType::BFLOAT16: {
            std::uniform_int_distribution<int> sign_dist(0, 1);
            std::uniform_int_distribution<int> exp_dist(1, 254);  // avoid 0/255 (denorm/inf)
            std::uniform_int_distribution<int> mant_dist(0, 127);
            size_t count = bytes / sizeof(uint16_t);
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits = static_cast<uint16_t>(
                    (sign_dist(rng) << 15) |
                    (exp_dist(rng) << 7) |
                    mant_dist(rng)
                );
                std::memcpy(raw + i * sizeof(uint16_t), &bits, sizeof(uint16_t));
            }
            break;
        }
        case DType::INT8: {
            std::uniform_int_distribution<int> dist(-64, 64);
            for (size_t i = 0; i < bytes; ++i) {
                raw[i] = static_cast<char>(dist(rng));
            }
            break;
        }
        default:
            break;
    }
}

class VectorPrefetchSequenceProvider : public PrefetchSequenceProvider {
public:
    explicit VectorPrefetchSequenceProvider(std::vector<ChunkKey> sequence)
        : sequence_(std::move(sequence)) {}

    bool next(ChunkKey* out_key) override {
        if (!out_key || pos_ >= sequence_.size()) {
            return false;
        }
        *out_key = sequence_[pos_];
        ++pos_;
        return true;
    }

    void reset() override {
        pos_ = 0;
    }

    size_t position() const override {
        return pos_;
    }

    size_t length_hint() const override {
        return sequence_.size();
    }

private:
    std::vector<ChunkKey> sequence_;
    size_t pos_ = 0;
};

} // namespace

BufferPool::BufferPool(
    size_t max_memory_mb,
    SimpleDBStorage* storage,
    size_t prefetch_window
) : max_memory_(max_memory_mb * 1024 * 1024),
    prefetch_active_(false),
    prefetch_window_(prefetch_window),
    consumption_position_(0),
    prefetch_position_(0),
    storage_(storage) {

    if (max_memory_mb == 0 || storage == nullptr) {
        throw std::runtime_error("BufferPool: invalid initialization parameters");
    }

    total_memory_used_.store(0);
}

BufferPool::~BufferPool() {
    stop_prefetch();
}

PinnedChunk::PinnedChunk(const uint8_t* d,
                         size_t s,
                         Shape sh,
                         DType dt,
                         ChunkKey key,
                         size_t slot_idx,
                         BufferPool* pool,
                         std::shared_ptr<void> owned)
    : data(d),
      size(s),
      shape(sh),
      dtype(dt),
      key_(std::move(key)),
      slot_idx_(slot_idx),
      pool_(pool),
      active_(true),
      owned_(std::move(owned)) {}

PinnedChunk::~PinnedChunk() {
    release();
}

PinnedChunk::PinnedChunk(PinnedChunk&& other) noexcept
    : data(other.data),
      size(other.size),
      shape(other.shape),
      dtype(other.dtype),
      key_(std::move(other.key_)),
      slot_idx_(other.slot_idx_),
      pool_(other.pool_),
      active_(other.active_),
      owned_(std::move(other.owned_)) {
    other.data = nullptr;
    other.size = 0;
    other.pool_ = nullptr;
    other.active_ = false;
}

PinnedChunk& PinnedChunk::operator=(PinnedChunk&& other) noexcept {
    if (this != &other) {
        release();
        data = other.data;
        size = other.size;
        shape = other.shape;
        dtype = other.dtype;
        key_ = std::move(other.key_);
        slot_idx_ = other.slot_idx_;
        pool_ = other.pool_;
        active_ = other.active_;
        owned_ = std::move(other.owned_);
        other.data = nullptr;
        other.size = 0;
        other.pool_ = nullptr;
        other.active_ = false;
    }
    return *this;
}

void PinnedChunk::release() {
    if (active_ && pool_) {
        pool_->unpin_chunk(key_, slot_idx_);
    }
    active_ = false;
    pool_ = nullptr;
    owned_.reset();
    data = nullptr;
    size = 0;
}

void BufferPool::unpin_chunk(const ChunkKey& chunk_key, size_t slot_idx) {
    if (slot_idx >= slots_.size()) {
        return;
    }
    Slot& slot = slots_[slot_idx];
    std::unique_lock<std::mutex> slot_lock(slot.mtx);
    uint32_t prev_refs = slot.refs.fetch_sub(1, std::memory_order_acq_rel);
    if (prev_refs <= 1 && slot.state.load(std::memory_order_acquire) == SlotState::Ready) {
        // Last holder released this slot
        slot.state.store(SlotState::Empty, std::memory_order_release);
        total_memory_used_.fetch_sub(slot.used, std::memory_order_relaxed);
        slot.used = 0;
        {
            std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
            slot_index_.erase(chunk_key);
        }
        slot.cv.notify_all();
        memory_cv_.notify_all();
        wait_cv_.notify_all();
    }
}

std::optional<std::pair<size_t, BufferPool::Slot*>> BufferPool::pin_ready_slot(const ChunkKey& key) {
    std::optional<size_t> idx_opt;
    {
        std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
        auto it = slot_index_.find(key);
        if (it != slot_index_.end()) {
            idx_opt = it->second;
        }
    }
    if (!idx_opt) {
        return std::nullopt;
    }
    Slot& slot = slots_[*idx_opt];
    if (slot.state.load(std::memory_order_acquire) != SlotState::Ready) {
        return std::nullopt;
    }
    slot.refs.fetch_add(1, std::memory_order_relaxed);
    return std::make_pair(*idx_opt, &slot);
}

PinnedChunk BufferPool::get_chunk(
    const std::string& matrix_id,
    int64_t chunk_idx,
    const BlockMatrix& matrix,
    int64_t step,
    double timeout_seconds
) {
    g_get_chunk_calls.fetch_add(1);
    ChunkKey chunk_key(matrix_id.empty() ? matrix.matrix_id() : matrix_id,
                       chunk_idx, matrix.split_mode(), step);

    auto start = std::chrono::high_resolution_clock::now();
    auto last_wait = start;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
    bool counted_miss = false;

    while (true) {
        if (auto access = pin_ready_slot(chunk_key)) {
            g_cache_hits.fetch_add(1);
            auto it = prefetch_index_.find(chunk_key);
            if (it != prefetch_index_.end()) {
                consumption_position_.store(it->second, std::memory_order_relaxed);
            } else {
                consumption_position_.fetch_add(1, std::memory_order_relaxed);
            }

            // Guardrail: verify slot metadata matches requested chunk before handing out.
            auto [rows, cols] = (chunk_key.split_mode == SplitMode::COLUMN)
                ? matrix.col_chunk_shape(chunk_idx)
                : matrix.row_chunk_shape(chunk_idx);
            size_t expected_bytes = rows * cols * dtype_size(matrix.dtype());
            Slot* slot = access->second;
            if (slot->used != expected_bytes || slot->buffer.size() != expected_bytes || slot->dtype != matrix.dtype()) {
                // Undo the pin and surface a clear error instead of crashing later in GEMM.
                unpin_chunk(chunk_key, access->first);
                std::ostringstream oss;
                oss << "BufferPool: chunk metadata mismatch for " << chunk_key.to_string()
                    << " expected_bytes=" << expected_bytes
                    << " slot.used=" << slot->used
                    << " buffer.size()=" << slot->buffer.size()
                    << " dtype=" << dtype_to_string(matrix.dtype());
                throw std::runtime_error(oss.str());
            }
            const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(slot->buffer.data());
            if (data_ptr == nullptr) {
                unpin_chunk(chunk_key, access->first);
                throw std::runtime_error("BufferPool: null data pointer for " + chunk_key.to_string());
            }
            return PinnedChunk(
                data_ptr,
                slot->used,
                slot->shape,
                slot->dtype,
                chunk_key,
                access->first,
                this
            );
        }

        if (!counted_miss) {
            g_cache_misses.fetch_add(1);
            counted_miss = true;
        }

        if (!prefetch_active_.load(std::memory_order_acquire)) {
            throw std::runtime_error("BufferPool: Prefetch inactive, chunk not available for " +
                                     chunk_key.to_string());
        }

        std::unique_lock<std::mutex> wait_lock(wait_mutex_);
        if (wait_cv_.wait_until(wait_lock, deadline) == std::cv_status::timeout) {
            bool in_sequence = false;
            if (!prefetch_index_.empty()) {
                in_sequence = (prefetch_index_.find(chunk_key) != prefetch_index_.end());
            } else {
                in_sequence = (matrix_registry_.find(chunk_key.matrix_id) != matrix_registry_.end());
            }
            std::ostringstream oss;
            oss << "BufferPool: Chunk not loaded after timeout for " << chunk_key.to_string()
                << " in_prefetch_sequence=" << (in_sequence ? "true" : "false")
                << " prefetch_active=" << (prefetch_active_.load(std::memory_order_relaxed) ? "true" : "false")
                << " prefetch_pos=" << prefetch_position_.load(std::memory_order_relaxed)
                << " consumption_pos=" << consumption_position_.load(std::memory_order_relaxed);
            throw std::runtime_error(oss.str());
        }

        auto now = std::chrono::high_resolution_clock::now();
        uint64_t wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_wait).count();
        last_wait = now;
        g_wait_time_ns.fetch_add(wait_ns);
        uint64_t current_max = g_max_wait_ns.load(std::memory_order_relaxed);
        if (wait_ns > current_max) {
            g_max_wait_ns.store(wait_ns, std::memory_order_relaxed);
        }
        if (wait_ns > kSlowWaitNs) {
            std::cerr << "[BufferPool] Slow wait: " << (wait_ns / 1000000.0) << " ms for "
                      << chunk_key.to_string() << std::endl;
        }
    }
}

std::optional<PinnedChunk> BufferPool::try_get_chunk(const ChunkKey& key) {
    if (!prefetch_active_.load(std::memory_order_acquire)) {
        return std::nullopt;
    }

    auto access = pin_ready_slot(key);
    if (!access.has_value()) {
        return std::nullopt;
    }
    g_cache_hits.fetch_add(1);

    Slot* slot = access->second;
    // Validate that the cached slot still matches the requested key and metadata.
    if (!(slot->key == key)) {
        unpin_chunk(key, access->first);
        std::ostringstream oss;
        oss << "BufferPool: slot key mismatch for " << key.to_string()
            << " cached=" << slot->key.to_string();
        throw std::runtime_error(oss.str());
    }
    // shape/dtype/size validation
    auto [rows, cols] = slot->shape;
    size_t expected_bytes = static_cast<size_t>(rows * cols) * dtype_size(slot->dtype);
    if (slot->used != expected_bytes || slot->buffer.size() != expected_bytes) {
        unpin_chunk(key, access->first);
        std::ostringstream oss;
        oss << "BufferPool: cached chunk size mismatch for " << key.to_string()
            << " expected_bytes=" << expected_bytes
            << " slot.used=" << slot->used
            << " buffer.size()=" << slot->buffer.size()
            << " dtype=" << dtype_to_string(slot->dtype);
        throw std::runtime_error(oss.str());
    }
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(slot->buffer.data());
    if (data_ptr == nullptr) {
        unpin_chunk(key, access->first);
        throw std::runtime_error("BufferPool: null data pointer for " + key.to_string());
    }

    return PinnedChunk(
        data_ptr,
        slot->used,
        slot->shape,
        slot->dtype,
        key,
        access->first,
        this
    );
}

PinnedChunk BufferPool::get_chunk_pinned(
    const std::string& matrix_id,
    int64_t chunk_idx,
    const BlockMatrix& matrix,
    int64_t step,
    double timeout_seconds
) {
    g_get_chunk_calls.fetch_add(1);
    ChunkKey chunk_key(matrix_id.empty() ? matrix.matrix_id() : matrix_id,
                       chunk_idx, matrix.split_mode(), step);

    auto start = std::chrono::high_resolution_clock::now();
    auto last_wait = start;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
    bool counted_miss = false;

    while (true) {
        if (auto access = pin_ready_slot(chunk_key)) {
            g_cache_hits.fetch_add(1);
            auto it = prefetch_index_.find(chunk_key);
            if (it != prefetch_index_.end()) {
                consumption_position_.store(it->second, std::memory_order_relaxed);
            } else {
                consumption_position_.fetch_add(1, std::memory_order_relaxed);
            }

            auto [rows, cols] = (chunk_key.split_mode == SplitMode::COLUMN)
                ? matrix.col_chunk_shape(chunk_idx)
                : matrix.row_chunk_shape(chunk_idx);
            size_t expected_bytes = rows * cols * dtype_size(matrix.dtype());
            Slot* slot = access->second;
            if (slot->used != expected_bytes || slot->buffer.size() != expected_bytes || slot->dtype != matrix.dtype()) {
                unpin_chunk(chunk_key, access->first);
                std::ostringstream oss;
                oss << "BufferPool: chunk metadata mismatch for " << chunk_key.to_string()
                    << " expected_bytes=" << expected_bytes
                    << " slot.used=" << slot->used
                    << " buffer.size()=" << slot->buffer.size()
                    << " dtype=" << dtype_to_string(matrix.dtype());
                throw std::runtime_error(oss.str());
            }
            const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(slot->buffer.data());
            if (data_ptr == nullptr) {
                unpin_chunk(chunk_key, access->first);
                throw std::runtime_error("BufferPool: null data pointer for " + chunk_key.to_string());
            }
            return PinnedChunk(
                data_ptr,
                slot->used,
                slot->shape,
                slot->dtype,
                chunk_key,
                access->first,
                this
            );
        }

        if (!counted_miss) {
            g_cache_misses.fetch_add(1);
            counted_miss = true;
        }

        if (!prefetch_active_.load(std::memory_order_acquire)) {
            throw std::runtime_error("BufferPool: Prefetch inactive, chunk not available for " +
                                     chunk_key.to_string());
        }

        std::unique_lock<std::mutex> wait_lock(wait_mutex_);
        if (wait_cv_.wait_until(wait_lock, deadline) == std::cv_status::timeout) {
            bool in_sequence = false;
            if (!prefetch_index_.empty()) {
                in_sequence = (prefetch_index_.find(chunk_key) != prefetch_index_.end());
            } else {
                in_sequence = (matrix_registry_.find(chunk_key.matrix_id) != matrix_registry_.end());
            }
            std::ostringstream oss;
            oss << "BufferPool: Chunk not loaded after timeout for " << chunk_key.to_string()
                << " in_prefetch_sequence=" << (in_sequence ? "true" : "false")
                << " prefetch_active=" << (prefetch_active_.load(std::memory_order_relaxed) ? "true" : "false")
                << " prefetch_pos=" << prefetch_position_.load(std::memory_order_relaxed)
                << " consumption_pos=" << consumption_position_.load(std::memory_order_relaxed);
            throw std::runtime_error(oss.str());
        }

        auto now = std::chrono::high_resolution_clock::now();
        uint64_t wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_wait).count();
        last_wait = now;
        g_wait_time_ns.fetch_add(wait_ns);
        uint64_t current_max = g_max_wait_ns.load(std::memory_order_relaxed);
        if (wait_ns > current_max) {
            g_max_wait_ns.store(wait_ns, std::memory_order_relaxed);
        }
        if (wait_ns > kSlowWaitNs) {
            std::cerr << "[BufferPool] Slow wait: " << (wait_ns / 1000000.0) << " ms for "
                      << chunk_key.to_string() << std::endl;
        }
    }
}

void BufferPool::start_sequence_prefetch(
    const std::vector<ChunkKey>& sequence,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
) {
    stop_prefetch();

    if (sequence.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
        slot_index_.clear();
    }
    prefetch_sequence_ = sequence;
    matrix_registry_ = matrices;
    prefetch_index_.clear();
    prefetch_index_.reserve(sequence.size());
    for (size_t i = 0; i < sequence.size(); ++i) {
        prefetch_index_[sequence[i]] = i;
    }
    sequence_provider_ = std::make_unique<VectorPrefetchSequenceProvider>(sequence);
    sequence_length_ = sequence_provider_->length_hint();
    prepare_simulated_chunks();

    // Derive slot count from memory budget and first chunk size (approximate).
    size_t slot_count = 1;
    if (!prefetch_sequence_.empty()) {
        const ChunkKey& first = prefetch_sequence_.front();
        auto matrix_it = matrices.find(first.matrix_id);
        if (matrix_it == matrices.end()) {
            throw std::runtime_error("BufferPool: matrix not found for prefetch initialization");
        }
        auto matrix = matrix_it->second;
        Shape shape = (first.split_mode == SplitMode::COLUMN)
            ? matrix->col_chunk_shape(first.chunk_idx)
            : matrix->row_chunk_shape(first.chunk_idx);
        auto [rows, cols] = shape;
        size_t expected_bytes = rows * cols * dtype_size(matrix->dtype());
        if (expected_bytes == 0) {
            expected_bytes = 1;
        }
        size_t max_slots_by_mem = std::max<size_t>(1, max_memory_ / expected_bytes);
        slot_count = std::min<size_t>(sequence.size(), max_slots_by_mem);
    }
    slots_.clear();
    slots_.resize(slot_count);
    total_memory_used_.store(0);
    consumption_position_.store(0);
    prefetch_position_.store(0);

    prefetch_active_.store(true, std::memory_order_release);
    prefetch_thread_ = std::thread(&BufferPool::prefetch_worker, this);
}

void BufferPool::start_prefetch_provider(
    std::unique_ptr<PrefetchSequenceProvider> provider,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
) {
    stop_prefetch();
    if (!provider) {
        return;
    }

    {
        std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
        slot_index_.clear();
    }
    prefetch_sequence_.clear();
    prefetch_index_.clear();
    matrix_registry_ = matrices;
    sequence_provider_ = std::move(provider);
    sequence_length_ = sequence_provider_->length_hint();
    prepare_simulated_chunks();

    ChunkKey first_key("", 0, SplitMode::COLUMN);
    if (!sequence_provider_->next(&first_key)) {
        sequence_provider_.reset();
        sequence_length_ = 0;
        return;
    }
    sequence_provider_->reset();

    // Derive slot count from memory budget and first chunk size (approximate).
    size_t slot_count = 1;
    auto matrix_it = matrices.find(first_key.matrix_id);
    if (matrix_it == matrices.end()) {
        throw std::runtime_error("BufferPool: matrix not found for prefetch initialization");
    }
    auto matrix = matrix_it->second;
    Shape shape = (first_key.split_mode == SplitMode::COLUMN)
        ? matrix->col_chunk_shape(first_key.chunk_idx)
        : matrix->row_chunk_shape(first_key.chunk_idx);
    auto [rows, cols] = shape;
    size_t expected_bytes = rows * cols * dtype_size(matrix->dtype());
    if (expected_bytes == 0) {
        expected_bytes = 1;
    }
    size_t max_slots_by_mem = std::max<size_t>(1, max_memory_ / expected_bytes);
    if (sequence_length_ > 0) {
        slot_count = std::min<size_t>(sequence_length_, max_slots_by_mem);
    } else {
        slot_count = max_slots_by_mem;
    }

    slots_.clear();
    slots_.resize(slot_count);
    total_memory_used_.store(0);
    consumption_position_.store(0);
    prefetch_position_.store(0);

    prefetch_active_.store(true, std::memory_order_release);
    prefetch_thread_ = std::thread(&BufferPool::prefetch_worker, this);
}

void BufferPool::stop_prefetch() {
    prefetch_active_.store(false, std::memory_order_release);
    wait_cv_.notify_all();
    memory_cv_.notify_all();
    for (auto& slot : slots_) {
        slot.cv.notify_all();
    }

    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }

    clear_prefetch_state();
    clear_buffer_pool();
}

void BufferPool::sync_consumption_position(const ChunkKey& chunk_key) {
    auto it = prefetch_index_.find(chunk_key);
    if (it != prefetch_index_.end()) {
        consumption_position_.store(it->second, std::memory_order_relaxed);
        return;
    }
    consumption_position_.fetch_add(1, std::memory_order_relaxed);
}

void BufferPool::set_prefetch_ring(bool enabled) {
    prefetch_ring_.store(enabled, std::memory_order_relaxed);
    wait_cv_.notify_all();
}

void BufferPool::set_prefetch_simulation(bool enabled, uint64_t get_latency_ms) {
    simulate_prefetch_.store(enabled, std::memory_order_relaxed);
    simulate_get_latency_ms_.store(get_latency_ms, std::memory_order_relaxed);
    if (!enabled && !prefetch_active_.load(std::memory_order_acquire)) {
        simulated_chunk_pool_.clear();
    }
}

void BufferPool::clear() {
    clear_buffer_pool();
}

BufferPool::Stats BufferPool::get_stats() const {
    size_t cached = 0;
    for (const auto& slot : slots_) {
        if (slot.state.load(std::memory_order_relaxed) == SlotState::Ready) {
            ++cached;
        }
    }
    return Stats{
        .cached_chunks = cached,
        .slot_capacity = slots_.size(),
        .memory_used_bytes = total_memory_used_.load(std::memory_order_relaxed),
        .memory_total_bytes = max_memory_,
        .prefetch_active = prefetch_active_.load(std::memory_order_relaxed),
        .consumption_position = consumption_position_.load(std::memory_order_relaxed),
        .prefetch_position = prefetch_position_.load(std::memory_order_relaxed),
        .sequence_length = sequence_length_
    };
}

BufferPool::ProfileStats BufferPool::get_profile_stats() const {
    return ProfileStats{
        .get_chunk_calls = g_get_chunk_calls.load(),
        .cache_hits = g_cache_hits.load(),
        .cache_misses = g_cache_misses.load(),
        .total_wait_time_ns = g_wait_time_ns.load(),
        .max_wait_time_ns = g_max_wait_ns.load(),
        .evict_count = g_evict_count.load(),
        .prefetch_get_calls = g_prefetch_get_calls.load(),
        .prefetch_get_time_ns = g_prefetch_get_time_ns.load()
    };
}

std::vector<ChunkKey> BufferPool::get_prefetch_sequence() const {
    return prefetch_sequence_;
}

std::vector<ChunkKey> BufferPool::get_cached_chunks() const {
    std::vector<ChunkKey> cached;
    std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
    cached.reserve(slot_index_.size());
    for (const auto& [key, _] : slot_index_) {
        cached.push_back(key);
    }
    return cached;
}

bool BufferPool::next_prefetch_key(ChunkKey* out_key, size_t* out_pos) {
    if (!sequence_provider_ || !out_key) {
        return false;
    }
    while (true) {
        if (sequence_provider_->next(out_key)) {
            size_t pos = prefetch_position_.fetch_add(1, std::memory_order_relaxed);
            if (out_pos) {
                *out_pos = pos;
            }
            return true;
        }
        if (prefetch_ring_.load(std::memory_order_relaxed)) {
            sequence_provider_->reset();
            continue;
        }
        std::cout << "[PrefetchNext] end of sequence" << std::endl;
        return false;
    }
}

void BufferPool::prefetch_worker() {
    struct PrefetchedEntry {
        ChunkKey key;
        std::string key_str;
        AlignedString buffer;
        Shape shape{0, 0};
        DType dtype{DType::FLOAT32};
        size_t expected_bytes{0};
        size_t seq_pos{0};
    };

    auto still_active = [&]() { return prefetch_active_.load(std::memory_order_acquire); };

    while (still_active()) {
        bool simulate = simulate_prefetch_.load(std::memory_order_relaxed);
        uint64_t simulate_ms = simulate_get_latency_ms_.load(std::memory_order_relaxed);
        size_t batch_size = prefetch_window_ > 0 ? prefetch_window_ : 1;
        std::vector<PrefetchedEntry> batch;
        batch.reserve(batch_size);

        // Phase 1: collect keys/metadata for the batch (no DB I/O here).
        for (size_t i = 0; i < batch_size && still_active(); ++i) {
            ChunkKey chunk_key("", 0, SplitMode::COLUMN);
            size_t seq_pos = 0;
            if (!next_prefetch_key(&chunk_key, &seq_pos)) {
                break;
            }
            auto matrix_it = matrix_registry_.find(chunk_key.matrix_id);
            if (matrix_it == matrix_registry_.end()) {
                std::cerr << "[BufferPool] Missing matrix for prefetch: "
                          << chunk_key.to_string()
                          << " (registry size=" << matrix_registry_.size() << "), skipping"
                          << std::endl;
                continue;
            }
            auto matrix = matrix_it->second;
            Shape shape = (chunk_key.split_mode == SplitMode::COLUMN)
                ? matrix->col_chunk_shape(chunk_key.chunk_idx)
                : matrix->row_chunk_shape(chunk_key.chunk_idx);
            auto [rows, cols] = shape;
            size_t expected_bytes = rows * cols * dtype_size(matrix->dtype());
        size_t memory_limit = max_memory_;

            if (expected_bytes > memory_limit && memory_limit > 0) {
                std::ostringstream oss;
                oss << "BufferPool: chunk " << chunk_key.to_string()
                    << " size " << expected_bytes << " exceeds memory limit " << memory_limit;
                throw std::runtime_error(oss.str());
            }

            std::string key = format_chunk_key(chunk_key.matrix_id, chunk_key.split_mode, chunk_key.chunk_idx);
            batch.push_back(PrefetchedEntry{
                chunk_key,
                std::move(key),
                {},
                shape,
                matrix->dtype(),
                expected_bytes,
                seq_pos
            });
        }

        if (!still_active() || batch.empty()) {
            break;
        }

        // Phase 2: fetch values from DB sequentially using the collected keys.
        for (auto& entry : batch) {
            auto get_start = std::chrono::high_resolution_clock::now();
            bool ok = true;
            if (simulate) {
                if (simulate_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(simulate_ms));
                }
                auto [rows, cols] = entry.shape;
                SimulatedChunkShape shape_key{rows, cols, entry.dtype};
                auto it = simulated_chunk_pool_.find(shape_key);
                if (it == simulated_chunk_pool_.end()) {
                    std::ostringstream oss;
                    oss << "BufferPool: simulated chunk shape missing for "
                        << rows << "x" << cols << " dtype=" << dtype_to_string(entry.dtype);
                    throw std::runtime_error(oss.str());
                }
                entry.buffer = it->second;
            } else {
                ok = storage_->get_value_into(entry.key_str, &entry.buffer);
            }
            auto get_end = std::chrono::high_resolution_clock::now();
            uint64_t get_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(get_end - get_start).count();
            g_prefetch_get_time_ns.fetch_add(get_ns);
            g_prefetch_get_calls.fetch_add(1);

            if (!ok) {
                throw std::runtime_error("BufferPool: Failed to get chunk from database");
            }
            if (get_ns > kSlowGetNs) {
                std::cerr << "[BufferPool] Slow Get(): " << (get_ns / 1000000.0) << " ms for "
                          << entry.key_str << std::endl;
            }
            if (entry.buffer.size() != entry.expected_bytes) {
                std::ostringstream oss;
                oss << "[BufferPool] Value size mismatch for chunk " << entry.key.to_string()
                    << " expected " << entry.expected_bytes << " bytes, got " << entry.buffer.size();
                throw std::runtime_error(oss.str());
            }
        }

        // Phase 3: place prefetched values into slots (ring buffer).
        for (auto& entry : batch) {
        size_t memory_limit = max_memory_;
            {
                std::unique_lock<std::mutex> mem_lock(memory_mutex_);
                memory_cv_.wait(mem_lock, [&]() {
                    return !still_active() ||
                           (total_memory_used_.load(std::memory_order_relaxed) + entry.expected_bytes <= memory_limit);
                });
            }
            if (!still_active()) {
                break;
            }

            size_t slot_idx = entry.seq_pos % slots_.size();
            Slot& slot = slots_[slot_idx];
            {
                std::unique_lock<std::mutex> slot_lock(slot.mtx);
                slot.cv.wait(slot_lock, [&]() {
                    return !still_active() ||
                           (slot.refs.load(std::memory_order_relaxed) == 0 &&
                            slot.state.load(std::memory_order_acquire) == SlotState::Empty);
                });
                if (!still_active()) {
                    break;
                }
                slot.state.store(SlotState::Filling, std::memory_order_release);
            }

            slot.buffer = std::move(entry.buffer);
            slot.used = slot.buffer.size();
            slot.shape = entry.shape;
            slot.dtype = entry.dtype;
            slot.key = entry.key;
            slot.seq_pos = entry.seq_pos;
            slot.refs.store(0, std::memory_order_relaxed);

            total_memory_used_.fetch_add(slot.used, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
                slot_index_[entry.key] = slot_idx;
            }

            slot.state.store(SlotState::Ready, std::memory_order_release);
            slot.cv.notify_all();
            wait_cv_.notify_all();
        }
    }

    // Mark prefetch as inactive when the worker finishes naturally.
    prefetch_active_.store(false, std::memory_order_release);
    wait_cv_.notify_all();
    memory_cv_.notify_all();
    for (auto& slot : slots_) {
        slot.cv.notify_all();
    }
}

void BufferPool::clear_buffer_pool() {
    for (auto& slot : slots_) {
        slot.buffer.clear();
        slot.buffer.shrink_to_fit();
        slot.used = 0;
        slot.refs.store(0, std::memory_order_relaxed);
        slot.state.store(SlotState::Empty, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> idx_lock(slot_index_mutex_);
        slot_index_.clear();
    }
    total_memory_used_.store(0, std::memory_order_relaxed);
}

void BufferPool::clear_prefetch_state() {
    prefetch_sequence_.clear();
    prefetch_index_.clear();
    sequence_provider_.reset();
    sequence_length_ = 0;
    matrix_registry_.clear();
    consumption_position_.store(0, std::memory_order_relaxed);
    prefetch_position_.store(0, std::memory_order_relaxed);
    simulated_chunk_pool_.clear();
}

void BufferPool::preload_initial_chunks(
    const std::vector<ChunkKey>&,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>&
) {
    // Prefetch handles loading; no-op.
}

void BufferPool::prepare_simulated_chunks() {
    simulated_chunk_pool_.clear();
    if (!simulate_prefetch_.load(std::memory_order_relaxed)) {
        return;
    }
    std::mt19937 rng(0xC001D00Du);
    if (!prefetch_sequence_.empty()) {
        for (const auto& chunk_key : prefetch_sequence_) {
            auto matrix_it = matrix_registry_.find(chunk_key.matrix_id);
            if (matrix_it == matrix_registry_.end()) {
                throw std::runtime_error("BufferPool: matrix not found for simulation");
            }
            auto matrix = matrix_it->second;
            Shape shape = (chunk_key.split_mode == SplitMode::COLUMN)
                ? matrix->col_chunk_shape(chunk_key.chunk_idx)
                : matrix->row_chunk_shape(chunk_key.chunk_idx);
            auto [rows, cols] = shape;
            SimulatedChunkShape shape_key{rows, cols, matrix->dtype()};
            if (simulated_chunk_pool_.find(shape_key) != simulated_chunk_pool_.end()) {
                continue;
            }
            size_t expected_bytes = rows * cols * dtype_size(matrix->dtype());
            AlignedString buffer;
            buffer.resize(expected_bytes);
            fill_random_buffer(&buffer, matrix->dtype(), rng);
            simulated_chunk_pool_.emplace(shape_key, std::move(buffer));
        }
        return;
    }

    for (const auto& [matrix_id, matrix] : matrix_registry_) {
        if (!matrix) {
            continue;
        }
        if (matrix->split_mode() == SplitMode::COLUMN) {
            for (int64_t j = 0; j < matrix->num_col_chunks(); ++j) {
                Shape shape = matrix->col_chunk_shape(j);
                auto [rows, cols] = shape;
                SimulatedChunkShape shape_key{rows, cols, matrix->dtype()};
                if (simulated_chunk_pool_.find(shape_key) != simulated_chunk_pool_.end()) {
                    continue;
                }
                size_t expected_bytes = rows * cols * dtype_size(matrix->dtype());
                AlignedString buffer;
                buffer.resize(expected_bytes);
                fill_random_buffer(&buffer, matrix->dtype(), rng);
                simulated_chunk_pool_.emplace(shape_key, std::move(buffer));
            }
        } else if (matrix->split_mode() == SplitMode::ROW) {
            for (int64_t i = 0; i < matrix->num_row_chunks(); ++i) {
                Shape shape = matrix->row_chunk_shape(i);
                auto [rows, cols] = shape;
                SimulatedChunkShape shape_key{rows, cols, matrix->dtype()};
                if (simulated_chunk_pool_.find(shape_key) != simulated_chunk_pool_.end()) {
                    continue;
                }
                size_t expected_bytes = rows * cols * dtype_size(matrix->dtype());
                AlignedString buffer;
                buffer.resize(expected_bytes);
                fill_random_buffer(&buffer, matrix->dtype(), rng);
                simulated_chunk_pool_.emplace(shape_key, std::move(buffer));
            }
        }
    }
}

} // namespace kvtensor
