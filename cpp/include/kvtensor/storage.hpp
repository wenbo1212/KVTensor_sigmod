#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "kvtensor/simpledb.hpp"
#include "types.hpp"

namespace kvtensor {

// Forward declarations
class SimpleDBTransaction;
class SimpleDBCursor;
class SimpleKVStore;

// SimpleKVStore-backed storage wrapper
class SimpleDBStorage {
public:
    explicit SimpleDBStorage(const std::string& path, bool read_only = false);
    ~SimpleDBStorage();

    // Non-copyable
    SimpleDBStorage(const SimpleDBStorage&) = delete;
    SimpleDBStorage& operator=(const SimpleDBStorage&) = delete;

    // Moveable
    SimpleDBStorage(SimpleDBStorage&&) noexcept;
    SimpleDBStorage& operator=(SimpleDBStorage&&) noexcept;

    // Get row chunk
    std::optional<std::vector<uint8_t>> get_row_chunk(
        const std::string& matrix_id,
        int64_t chunk_idx
    ) const;

    // Get column chunk
    std::optional<std::vector<uint8_t>> get_col_chunk(
        const std::string& matrix_id,
        int64_t chunk_idx
    ) const;

    // Put row chunk
    void put_row_chunk(
        const std::string& matrix_id,
        int64_t chunk_idx,
        const uint8_t* data,
        size_t data_size
    );

    // Put column chunk
    void put_col_chunk(
        const std::string& matrix_id,
        int64_t chunk_idx,
        const uint8_t* data,
        size_t data_size
    );

    // Get raw value by key
    bool get_value(const std::string& key, AlignedString* out) const;
    bool get_value_into(const std::string& key, AlignedString* out) const;

    const std::string& path() const { return path_; }

    // Get cursor for row chunks (for sequential iteration)
    std::unique_ptr<SimpleDBCursor> get_row_chunk_cursor(
        const std::string& matrix_id
    ) const;

    // Get cursor for column chunks (for sequential iteration)
    std::unique_ptr<SimpleDBCursor> get_col_chunk_cursor(
        const std::string& matrix_id
    ) const;


    // Begin transaction (for batch operations)
    std::unique_ptr<SimpleDBTransaction> begin(bool write = false) const;

private:
    friend class SimpleDBTransaction;
    friend class SimpleDBCursor;

    std::string format_row_chunk_key(const std::string& matrix_id, int64_t chunk_idx) const;
    std::string format_col_chunk_key(const std::string& matrix_id, int64_t chunk_idx) const;

    std::string path_;
    std::unique_ptr<class SimpleKVStore> db_;
};

// Transaction-like wrapper for batch operations
class SimpleDBTransaction {
public:
    explicit SimpleDBTransaction(SimpleDBStorage* storage, bool write = false);
    ~SimpleDBTransaction();

    // Non-copyable, non-moveable
    SimpleDBTransaction(const SimpleDBTransaction&) = delete;
    SimpleDBTransaction& operator=(const SimpleDBTransaction&) = delete;
    SimpleDBTransaction(SimpleDBTransaction&&) = delete;
    SimpleDBTransaction& operator=(SimpleDBTransaction&&) = delete;

    // Get value
    std::optional<std::vector<uint8_t>> get(const std::string& key) const;

    // Put value
    void put(const std::string& key, const uint8_t* data, size_t data_size);

    // Delete key
    void del(const std::string& key);

    // Get iterator with prefix
    std::unique_ptr<SimpleDBCursor> iterator(const std::string& prefix = "") const;

    // Commit transaction
    void commit();

    // Abort transaction (no-op for LevelDB, but kept for API compatibility)
    void abort();

private:
    SimpleDBStorage* storage_;
    bool write_;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> pending_writes_;
    std::vector<std::string> pending_deletes_;
};

class SimpleDBCursor {
public:
    SimpleDBCursor(std::vector<std::string> keys, SimpleDBStorage* storage);

    bool Valid() const;
    void Next();
    const std::string& key() const;
    const std::string& value() const;

private:
    void load_value();

    std::vector<std::string> keys_;
    SimpleDBStorage* storage_;
    size_t index_ = 0;
    std::string value_;
};

} // namespace kvtensor
