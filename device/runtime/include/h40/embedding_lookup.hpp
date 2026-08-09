#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace h40 {

struct EmbeddingTable {
    std::uint32_t file_id{};
    std::uint64_t base_offset{};
    std::uint32_t rows{};
    std::uint32_t row_bytes{};
};

struct RowRange {
    std::uint32_t file_id{};
    std::uint64_t offset{};
    std::uint32_t length{};
};

[[nodiscard]] RowRange resolve_embedding_row(const EmbeddingTable& table, std::uint32_t token_id);

class EmbeddingRowCache {
public:
    explicit EmbeddingRowCache(std::size_t capacity_bytes);

    [[nodiscard]] std::span<const std::byte> get(std::uint32_t token_id) const noexcept;
    void put(std::uint32_t token_id, std::span<const std::byte> row);
    void clear() noexcept;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry {
        std::uint32_t token_id{};
        std::vector<std::byte> data;
        std::uint64_t stamp{};
    };

    std::size_t capacity_bytes_{};
    std::size_t used_bytes_{};
    std::uint64_t clock_{};
    std::vector<Entry> entries_;
};

} // namespace h40
