#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace h40 {

struct H40mTensorRecord {
    std::string name;
    std::string file;
    std::uint64_t offset{};
    std::uint64_t length{};
    std::string dtype;
    std::vector<std::size_t> shape;
};

class H40mTensorCatalog {
public:
    static H40mTensorCatalog load_tsv(const std::filesystem::path& path);

    [[nodiscard]] std::optional<H40mTensorRecord> find(std::string_view name) const;
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

private:
    std::vector<H40mTensorRecord> records_;
};

class FileTensorReader {
public:
    explicit FileTensorReader(std::filesystem::path root) : root_(std::move(root)) {}

    void read(const H40mTensorRecord& record, std::span<std::byte> out) const;
    void read_bf16_row(const H40mTensorRecord& record, std::size_t row, std::span<float> out) const;
    void read_bf16_vector(const H40mTensorRecord& record, std::span<float> out) const;
    void bf16_matvec(
        const H40mTensorRecord& record,
        std::span<const float> input,
        std::span<float> output,
        std::span<std::uint16_t> row_buffer) const;
    void bf16_matvec_rows(
        const H40mTensorRecord& record,
        std::size_t row_begin,
        std::span<const float> input,
        std::span<float> output,
        std::span<std::uint16_t> row_buffer) const;

private:
    std::filesystem::path root_;
};

}  // namespace h40
