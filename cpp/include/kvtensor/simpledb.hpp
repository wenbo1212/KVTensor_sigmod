#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include <cstdlib>
#include <new>

namespace kvtensor {

template <typename T, std::size_t Alignment = 4096>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template <class U> constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }
};

template <class T, class U, std::size_t A>
constexpr bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }
template <class T, class U, std::size_t A>
constexpr bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }

using AlignedString = std::basic_string<char, std::char_traits<char>, AlignedAllocator<char, 4096>>;

struct SimpleKVOptions {
    bool create_if_missing = false;
    bool truncate = false;
    bool read_only = false;
};

class SimpleKVStore {
public:
    static SimpleKVStore Create(const std::string& path);
    static SimpleKVStore OpenReadOnly(const std::string& path);

    explicit SimpleKVStore(const std::string& path, SimpleKVOptions options);
    ~SimpleKVStore();

    SimpleKVStore(const SimpleKVStore&) = delete;
    SimpleKVStore& operator=(const SimpleKVStore&) = delete;
    SimpleKVStore(SimpleKVStore&& other) noexcept;
    SimpleKVStore& operator=(SimpleKVStore&& other) noexcept;

    bool put(const std::string& key, const uint8_t* data, size_t len);
    bool get(const std::string& key, std::vector<uint8_t>* out) const;
    bool get_into(const std::string& key, AlignedString& out) const;

    using IterateCallback = std::function<bool(const std::string& key, const uint8_t* data, size_t len)>;
    bool iterate_prefix(const std::string& prefix, const IterateCallback& cb) const;

    void flush_index();
    size_t entry_count() const { return entries_.size(); }

private:
    struct IndexEntry {
        std::string key;
        uint64_t offset = 0;
        uint32_t value_len = 0;
    };

    void load_index();
    void rebuild_index_from_data();
    void add_index_entry(const std::string& key, uint64_t offset, uint32_t value_len);
    void update_prefix_range(const std::string& prefix, size_t index);

    std::string data_path() const;
    std::string index_path() const;

    std::string path_;
    int data_fd_ = -1;
    bool read_only_ = false;
    bool dirty_index_ = false;
    uint64_t data_size_ = 0;

    std::vector<IndexEntry> entries_;
    std::unordered_map<std::string, size_t> key_index_;
    std::unordered_map<std::string, std::pair<size_t, size_t>> prefix_ranges_;

    static constexpr uint32_t kAlignment = 4096;
};

} // namespace kvtensor
