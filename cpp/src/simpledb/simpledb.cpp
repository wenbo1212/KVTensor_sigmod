#include "kvtensor/simpledb.hpp"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <memory>
#include <cstdlib>

#include <fcntl.h>
#include <unistd.h>

namespace kvtensor {

namespace {

constexpr uint32_t kRecordMagic = 0x31564B53;  // "SKV1"
constexpr uint32_t kIndexMagic = 0x31494B53;   // "SKI1"

struct IndexHeader {
    uint32_t magic = kIndexMagic;
    uint32_t version = 1;
    uint32_t alignment = 4096;
    uint64_t count = 0;
};

static bool write_all(int fd, const void* data, size_t len) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t written = ::write(fd, ptr, remaining);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        remaining -= static_cast<size_t>(written);
        ptr += written;
    }
    return true;
}

static bool read_all(int fd, uint64_t offset, void* data, size_t len) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(data);
    size_t remaining = len;
    uint64_t current = offset;
    while (remaining > 0) {
        ssize_t read_bytes = ::pread(fd, ptr, remaining, static_cast<off_t>(current));
        if (read_bytes < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (read_bytes == 0) {
            return false;
        }
        remaining -= static_cast<size_t>(read_bytes);
        ptr += read_bytes;
        current += static_cast<uint64_t>(read_bytes);
    }
    return true;
}

static std::string key_prefix(const std::string& key) {
    size_t pos = key.rfind(':');
    if (pos == std::string::npos) {
        return key;
    }
    return key.substr(0, pos);
}

static uint64_t align_up(uint64_t value, uint32_t align) {
    return (value + align - 1) / align * align;
}

} // namespace

SimpleKVStore SimpleKVStore::Create(const std::string& path) {
    SimpleKVOptions options;
    options.create_if_missing = true;
    options.truncate = true;
    options.read_only = false;
    return SimpleKVStore(path, options);
}

SimpleKVStore SimpleKVStore::OpenReadOnly(const std::string& path) {
    SimpleKVOptions options;
    options.create_if_missing = false;
    options.truncate = false;
    options.read_only = true;
    return SimpleKVStore(path, options);
}

SimpleKVStore::SimpleKVStore(const std::string& path, SimpleKVOptions options)
    : path_(path), read_only_(options.read_only) {
    std::filesystem::path dir(path_);
    if (options.create_if_missing) {
        std::filesystem::create_directories(dir);
    }

    int flags = options.read_only ? O_RDONLY : O_RDWR;
#ifdef O_DIRECT
    if (options.read_only) {
        flags |= O_DIRECT;
    }
#endif
    if (!options.read_only) {
        flags |= O_CREAT;
    }
    if (options.truncate) {
        flags |= O_TRUNC;
    }

    data_fd_ = ::open(data_path().c_str(), flags, 0644);
    if (data_fd_ < 0) {
        throw std::runtime_error("SimpleKVStore: failed to open data file");
    }

    off_t end_pos = ::lseek(data_fd_, 0, SEEK_END);
    if (end_pos < 0) {
        throw std::runtime_error("SimpleKVStore: failed to seek data file");
    }
    data_size_ = static_cast<uint64_t>(end_pos);

    if (!options.truncate) {
        load_index();
    }
}

SimpleKVStore::~SimpleKVStore() {
    if (!read_only_ && dirty_index_) {
        flush_index();
    }
    if (data_fd_ >= 0) {
        ::close(data_fd_);
        data_fd_ = -1;
    }
}

SimpleKVStore::SimpleKVStore(SimpleKVStore&& other) noexcept
    : path_(std::move(other.path_)),
      data_fd_(other.data_fd_),
      read_only_(other.read_only_),
      dirty_index_(other.dirty_index_),
      data_size_(other.data_size_),
      entries_(std::move(other.entries_)),
      key_index_(std::move(other.key_index_)),
      prefix_ranges_(std::move(other.prefix_ranges_)) {
    other.data_fd_ = -1;
}

SimpleKVStore& SimpleKVStore::operator=(SimpleKVStore&& other) noexcept {
    if (this != &other) {
        if (data_fd_ >= 0) {
            ::close(data_fd_);
        }
        path_ = std::move(other.path_);
        data_fd_ = other.data_fd_;
        read_only_ = other.read_only_;
        dirty_index_ = other.dirty_index_;
        data_size_ = other.data_size_;
        entries_ = std::move(other.entries_);
        key_index_ = std::move(other.key_index_);
        prefix_ranges_ = std::move(other.prefix_ranges_);
        other.data_fd_ = -1;
    }
    return *this;
}

std::string SimpleKVStore::data_path() const {
    return (std::filesystem::path(path_) / "data.kv").string();
}

std::string SimpleKVStore::index_path() const {
    return (std::filesystem::path(path_) / "index.kv").string();
}

bool SimpleKVStore::put(const std::string& key, const uint8_t* data, size_t len) {
    if (read_only_) {
        return false;
    }

    uint64_t aligned_offset = align_up(data_size_, kAlignment);
    if (aligned_offset > data_size_) {
        uint64_t pad = aligned_offset - data_size_;
        std::vector<uint8_t> zeros(static_cast<size_t>(pad), 0);
        if (!write_all(data_fd_, zeros.data(), zeros.size())) {
            return false;
        }
        data_size_ = aligned_offset;
    }

    if (!write_all(data_fd_, data, len)) {
        return false;
    }

    data_size_ += len;
    add_index_entry(key, aligned_offset, static_cast<uint32_t>(len));
    dirty_index_ = true;
    return true;
}

bool SimpleKVStore::get(const std::string& key, std::vector<uint8_t>* out) const {
    auto it = key_index_.find(key);
    if (it == key_index_.end()) {
        return false;
    }
    const IndexEntry& entry = entries_[it->second];
    uint64_t value_offset = entry.offset;
    size_t aligned_len = static_cast<size_t>(align_up(entry.value_len, kAlignment));
    void* aligned_buf = nullptr;
    if (posix_memalign(&aligned_buf, kAlignment, aligned_len) != 0 || aligned_buf == nullptr) {
        return false;
    }
    ssize_t n = ::pread(data_fd_, aligned_buf, aligned_len, static_cast<off_t>(value_offset));
    bool ok = (n == static_cast<ssize_t>(aligned_len));
    if (ok) {
        out->resize(entry.value_len);
        std::memcpy(out->data(), aligned_buf, entry.value_len);
    }
    free(aligned_buf);
    return ok;
}

bool SimpleKVStore::get_into(const std::string& key, AlignedString& out) const {
    auto it = key_index_.find(key);
    if (it == key_index_.end()) {
        return false;
    }
    const IndexEntry& entry = entries_[it->second];
    // Hint kernel for sequential access within the current prefix window.
    auto range_it = prefix_ranges_.find(key_prefix(key));
    if (range_it != prefix_ranges_.end()) {
        size_t start_idx = range_it->second.first;
        size_t end_idx = range_it->second.second;
        if (end_idx > start_idx) {
            const auto& last_entry = entries_[end_idx - 1];
            uint64_t window_start = entries_[start_idx].offset;
            uint64_t window_end = last_entry.offset + align_up(last_entry.value_len, kAlignment);
            uint64_t advise_len = window_end - window_start;
#ifdef POSIX_FADV_WILLNEED
            ::posix_fadvise(data_fd_, static_cast<off_t>(window_start),
                            static_cast<off_t>(advise_len), POSIX_FADV_WILLNEED);
#endif
        }
    }
    size_t aligned_len = static_cast<size_t>(align_up(entry.value_len, kAlignment));
    out.resize(aligned_len);
    if (entry.value_len == 0) {
        out.clear();
        return true;
    }

    char* buf = out.data();
    ssize_t n = ::pread(data_fd_, buf, aligned_len, static_cast<off_t>(entry.offset));
    bool ok = (n == static_cast<ssize_t>(aligned_len));
    out.resize(entry.value_len);
    return ok;
}

bool SimpleKVStore::iterate_prefix(const std::string& prefix, const IterateCallback& cb) const {
    auto range_it = prefix_ranges_.find(prefix);
    size_t start = 0;
    size_t end = entries_.size();
    if (range_it != prefix_ranges_.end()) {
        start = range_it->second.first;
        end = range_it->second.second;
    }
    std::vector<uint8_t> buffer;
    for (size_t i = start; i < end; ++i) {
        const IndexEntry& entry = entries_[i];
        if (entry.key.rfind(prefix, 0) != 0) {
            continue;
        }
        size_t aligned_len = static_cast<size_t>(align_up(entry.value_len, kAlignment));
        void* aligned_buf = nullptr;
        if (posix_memalign(&aligned_buf, kAlignment, aligned_len) != 0 || aligned_buf == nullptr) {
            return false;
        }
        ssize_t n = ::pread(data_fd_, aligned_buf, aligned_len, static_cast<off_t>(entry.offset));
        if (n != static_cast<ssize_t>(aligned_len)) {
            free(aligned_buf);
            return false;
        }
        buffer.resize(entry.value_len);
        std::memcpy(buffer.data(), aligned_buf, entry.value_len);
        free(aligned_buf);
        if (!cb(entry.key, buffer.data(), buffer.size())) {
            break;
        }
    }
    return true;
}

void SimpleKVStore::flush_index() {
    std::ofstream out(index_path(), std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("SimpleKVStore: failed to write index");
    }

    IndexHeader header;
    header.count = entries_.size();
    header.alignment = kAlignment;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const auto& entry : entries_) {
        uint32_t key_len = static_cast<uint32_t>(entry.key.size());
        out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        out.write(entry.key.data(), entry.key.size());
        out.write(reinterpret_cast<const char*>(&entry.offset), sizeof(entry.offset));
        out.write(reinterpret_cast<const char*>(&entry.value_len), sizeof(entry.value_len));
    }
    dirty_index_ = false;
}

void SimpleKVStore::load_index() {
    std::ifstream in(index_path(), std::ios::binary);
    if (!in.good()) {
        throw std::runtime_error("SimpleKVStore: missing index file");
    }

    IndexHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kIndexMagic) {
        throw std::runtime_error("SimpleKVStore: invalid index header");
    }
    if (header.alignment != kAlignment) {
        throw std::runtime_error("SimpleKVStore: unsupported alignment in index");
    }

    entries_.clear();
    key_index_.clear();
    prefix_ranges_.clear();
    entries_.reserve(static_cast<size_t>(header.count));

    for (uint64_t i = 0; i < header.count; ++i) {
        uint32_t key_len = 0;
        in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        if (!in) {
            break;
        }
        std::string key(key_len, '\0');
        in.read(key.data(), key_len);
        uint64_t offset = 0;
        uint32_t value_len = 0;
        in.read(reinterpret_cast<char*>(&offset), sizeof(offset));
        in.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
        if (!in) {
            break;
        }
        add_index_entry(key, offset, value_len);
    }
}

void SimpleKVStore::rebuild_index_from_data() {
    throw std::runtime_error("SimpleKVStore: rebuild_index_from_data unsupported in O_DIRECT layout");
}

void SimpleKVStore::add_index_entry(const std::string& key, uint64_t offset, uint32_t value_len) {
    IndexEntry entry;
    entry.key = key;
    entry.offset = offset;
    entry.value_len = value_len;
    size_t index = entries_.size();
    entries_.push_back(std::move(entry));
    key_index_[key] = index;
    update_prefix_range(key_prefix(key), index);
}

void SimpleKVStore::update_prefix_range(const std::string& prefix, size_t index) {
    auto it = prefix_ranges_.find(prefix);
    if (it == prefix_ranges_.end()) {
        prefix_ranges_[prefix] = {index, index + 1};
    } else {
        it->second.second = index + 1;
    }
}

} // namespace kvtensor
