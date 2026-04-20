#include "kvtensor/storage.hpp"
#include "kvtensor/simpledb.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace kvtensor {

SimpleDBStorage::SimpleDBStorage(const std::string& path, bool read_only)
    : path_(path) {
    SimpleKVOptions options;
    options.create_if_missing = !read_only;
    options.truncate = false;
    options.read_only = read_only;
    db_ = std::make_unique<SimpleKVStore>(path_, options);
}

SimpleDBStorage::~SimpleDBStorage() = default;

SimpleDBStorage::SimpleDBStorage(SimpleDBStorage&& other) noexcept
    : path_(std::move(other.path_)),
      db_(std::move(other.db_)) {}

SimpleDBStorage& SimpleDBStorage::operator=(SimpleDBStorage&& other) noexcept {
    if (this != &other) {
        path_ = std::move(other.path_);
        db_ = std::move(other.db_);
    }
    return *this;
}

std::string SimpleDBStorage::format_row_chunk_key(const std::string& matrix_id, int64_t chunk_idx) const {
    std::ostringstream oss;
    oss << matrix_id << ":row:" << std::setfill('0') << std::setw(6) << chunk_idx;
    return oss.str();
}

std::string SimpleDBStorage::format_col_chunk_key(const std::string& matrix_id, int64_t chunk_idx) const {
    std::ostringstream oss;
    oss << matrix_id << ":col:" << std::setfill('0') << std::setw(6) << chunk_idx;
    return oss.str();
}

std::optional<std::vector<uint8_t>> SimpleDBStorage::get_row_chunk(
    const std::string& matrix_id,
    int64_t chunk_idx
) const {
    std::string key = format_row_chunk_key(matrix_id, chunk_idx);
    std::vector<uint8_t> value;
    if (!db_->get(key, &value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::vector<uint8_t>> SimpleDBStorage::get_col_chunk(
    const std::string& matrix_id,
    int64_t chunk_idx
) const {
    std::string key = format_col_chunk_key(matrix_id, chunk_idx);
    std::vector<uint8_t> value;
    if (!db_->get(key, &value)) {
        return std::nullopt;
    }
    return value;
}

void SimpleDBStorage::put_row_chunk(
    const std::string& matrix_id,
    int64_t chunk_idx,
    const uint8_t* data,
    size_t data_size
) {
    std::string key = format_row_chunk_key(matrix_id, chunk_idx);
    if (!db_->put(key, data, data_size)) {
        throw std::runtime_error("SimpleDB write error (row chunk)");
    }
}

void SimpleDBStorage::put_col_chunk(
    const std::string& matrix_id,
    int64_t chunk_idx,
    const uint8_t* data,
    size_t data_size
) {
    std::string key = format_col_chunk_key(matrix_id, chunk_idx);
    if (!db_->put(key, data, data_size)) {
        throw std::runtime_error("SimpleDB write error (col chunk)");
    }
}

bool SimpleDBStorage::get_value(const std::string& key, AlignedString* out) const {
    return get_value_into(key, out);
}

bool SimpleDBStorage::get_value_into(const std::string& key, AlignedString* out) const {
    if (out == nullptr) {
        return false;
    }
    return db_->get_into(key, *out);
}

std::unique_ptr<SimpleDBCursor> SimpleDBStorage::get_row_chunk_cursor(
    const std::string& matrix_id
) const {
    std::string prefix = matrix_id + ":row:";
    std::vector<std::string> keys;
    db_->iterate_prefix(prefix, [&](const std::string& key, const uint8_t*, size_t) {
        keys.push_back(key);
        return true;
    });
    std::sort(keys.begin(), keys.end());
    return std::make_unique<SimpleDBCursor>(std::move(keys), const_cast<SimpleDBStorage*>(this));
}

std::unique_ptr<SimpleDBCursor> SimpleDBStorage::get_col_chunk_cursor(
    const std::string& matrix_id
) const {
    std::string prefix = matrix_id + ":col:";
    std::vector<std::string> keys;
    db_->iterate_prefix(prefix, [&](const std::string& key, const uint8_t*, size_t) {
        keys.push_back(key);
        return true;
    });
    std::sort(keys.begin(), keys.end());
    return std::make_unique<SimpleDBCursor>(std::move(keys), const_cast<SimpleDBStorage*>(this));
}

std::unique_ptr<SimpleDBTransaction> SimpleDBStorage::begin(bool write) const {
    return std::make_unique<SimpleDBTransaction>(const_cast<SimpleDBStorage*>(this), write);
}

SimpleDBTransaction::SimpleDBTransaction(SimpleDBStorage* storage, bool write)
    : storage_(storage), write_(write) {}

SimpleDBTransaction::~SimpleDBTransaction() {
    if (write_ && (!pending_writes_.empty() || !pending_deletes_.empty())) {
        commit();
    }
}

std::optional<std::vector<uint8_t>> SimpleDBTransaction::get(const std::string& key) const {
    for (const auto& entry : pending_writes_) {
        if (entry.first == key) {
            return entry.second;
        }
    }
    std::vector<uint8_t> value;
    if (!storage_->db_->get(key, &value)) {
        return std::nullopt;
    }
    return value;
}

void SimpleDBTransaction::put(const std::string& key, const uint8_t* data, size_t data_size) {
    if (!write_) {
        throw std::runtime_error("SimpleDBTransaction: write attempted on read-only transaction");
    }
    pending_writes_.emplace_back(key, std::vector<uint8_t>(data, data + data_size));
}

void SimpleDBTransaction::del(const std::string& key) {
    if (!write_) {
        throw std::runtime_error("SimpleDBTransaction: delete attempted on read-only transaction");
    }
    pending_deletes_.push_back(key);
}

std::unique_ptr<SimpleDBCursor> SimpleDBTransaction::iterator(const std::string& prefix) const {
    std::vector<std::string> keys;
    storage_->db_->iterate_prefix(prefix, [&](const std::string& key, const uint8_t*, size_t) {
        keys.push_back(key);
        return true;
    });
    std::sort(keys.begin(), keys.end());
    return std::make_unique<SimpleDBCursor>(std::move(keys), storage_);
}

void SimpleDBTransaction::commit() {
    for (const auto& entry : pending_writes_) {
        if (!storage_->db_->put(entry.first, entry.second.data(), entry.second.size())) {
            throw std::runtime_error("SimpleDBTransaction: write failed");
        }
    }
    if (!pending_deletes_.empty()) {
        throw std::runtime_error("SimpleDBTransaction: deletes not supported");
    }
    pending_writes_.clear();
    pending_deletes_.clear();
    storage_->db_->flush_index();
}

void SimpleDBTransaction::abort() {
    pending_writes_.clear();
    pending_deletes_.clear();
}

SimpleDBCursor::SimpleDBCursor(std::vector<std::string> keys, SimpleDBStorage* storage)
    : keys_(std::move(keys)), storage_(storage) {
    load_value();
}

bool SimpleDBCursor::Valid() const {
    return index_ < keys_.size();
}

void SimpleDBCursor::Next() {
    if (index_ < keys_.size()) {
        ++index_;
    }
    load_value();
}

const std::string& SimpleDBCursor::key() const {
    static const std::string kEmpty;
    if (!Valid()) {
        return kEmpty;
    }
    return keys_[index_];
}

const std::string& SimpleDBCursor::value() const {
    return value_;
}

void SimpleDBCursor::load_value() {
    value_.clear();
    if (!Valid()) {
        return;
    }
    std::vector<uint8_t> buf;
    if (!storage_->db_->get(keys_[index_], &buf)) {
        return;
    }
    value_.assign(reinterpret_cast<const char*>(buf.data()), buf.size());
}

} // namespace kvtensor
