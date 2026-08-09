#include "h40/embedding_lookup.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace h40 {

RowRange resolve_embedding_row(const EmbeddingTable& table, std::uint32_t token_id) {
    if (table.rows == 0 || table.row_bytes == 0) {
        throw std::invalid_argument("embedding table must have non-zero rows and row_bytes");
    }
    if (token_id >= table.rows) {
        throw std::out_of_range("embedding token id is outside table rows");
    }
    const std::uint64_t row_offset = static_cast<std::uint64_t>(token_id) * table.row_bytes;
    if (row_offset > std::numeric_limits<std::uint64_t>::max() - table.base_offset) {
        throw std::overflow_error("embedding row offset overflow");
    }
    return {table.file_id, table.base_offset + row_offset, table.row_bytes};
}

EmbeddingRowCache::EmbeddingRowCache(std::size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes) {}

std::span<const std::byte> EmbeddingRowCache::get(std::uint32_t token_id) const noexcept {
    for (const auto& entry : entries_) {
        if (entry.token_id == token_id) {
            return entry.data;
        }
    }
    return {};
}

void EmbeddingRowCache::put(std::uint32_t token_id, std::span<const std::byte> row) {
    if (row.size() > capacity_bytes_) {
        return;
    }
    for (auto& entry : entries_) {
        if (entry.token_id == token_id) {
            used_bytes_ -= entry.data.size();
            entry.data.assign(row.begin(), row.end());
            entry.stamp = ++clock_;
            used_bytes_ += entry.data.size();
            return;
        }
    }
    while (used_bytes_ + row.size() > capacity_bytes_ && !entries_.empty()) {
        const auto victim = std::min_element(entries_.begin(), entries_.end(), [](const Entry& lhs, const Entry& rhs) {
            return lhs.stamp < rhs.stamp;
        });
        used_bytes_ -= victim->data.size();
        entries_.erase(victim);
    }
    entries_.push_back({token_id, std::vector<std::byte>(row.begin(), row.end()), ++clock_});
    used_bytes_ += row.size();
}

void EmbeddingRowCache::clear() noexcept {
    entries_.clear();
    used_bytes_ = 0;
}

} // namespace h40
