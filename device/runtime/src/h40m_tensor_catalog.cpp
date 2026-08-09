#include "h40/h40m_tensor_catalog.hpp"

#include "h40/gptoss_expert.hpp"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace h40 {

namespace {

std::uint64_t parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) throw std::invalid_argument("invalid uint64 in tensor catalog");
    return value;
}

std::vector<std::size_t> parse_shape(std::string_view text) {
    std::vector<std::size_t> shape;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto pos = text.find('x', start);
        const auto part = text.substr(start, pos == std::string_view::npos ? text.size() - start : pos - start);
        if (!part.empty()) shape.push_back(static_cast<std::size_t>(parse_u64(part)));
        if (pos == std::string_view::npos) break;
        start = pos + 1;
    }
    return shape;
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= line.size()) {
        const auto pos = line.find('\t', start);
        parts.push_back(line.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return parts;
}

}  // namespace

H40mTensorCatalog H40mTensorCatalog::load_tsv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open H40M tensor catalog");
    std::string line;
    std::getline(in, line);
    if (line != "name\tfile\toffset\tlength\tdtype\tshape") {
        throw std::runtime_error("unexpected H40M tensor catalog header");
    }

    H40mTensorCatalog catalog;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto parts = split_tab(line);
        if (parts.size() != 6) throw std::runtime_error("malformed H40M tensor catalog row");
        H40mTensorRecord record;
        record.name = parts[0];
        record.file = parts[1];
        record.offset = parse_u64(parts[2]);
        record.length = parse_u64(parts[3]);
        record.dtype = parts[4];
        record.shape = parse_shape(parts[5]);
        catalog.records_.push_back(std::move(record));
    }
    return catalog;
}

std::optional<H40mTensorRecord> H40mTensorCatalog::find(std::string_view name) const {
    for (const auto& record : records_) {
        if (record.name == name) return record;
    }
    return std::nullopt;
}

void FileTensorReader::read(const H40mTensorRecord& record, std::span<std::byte> out) const {
    if (out.size() != record.length) throw std::invalid_argument("tensor read output size mismatch");
    std::ifstream in(root_ / record.file, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open tensor shard");
    in.seekg(static_cast<std::streamoff>(record.offset), std::ios::beg);
    if (!in) throw std::runtime_error("failed to seek tensor shard");
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (in.gcount() != static_cast<std::streamsize>(out.size())) throw std::runtime_error("short tensor read");
}

void FileTensorReader::read_bf16_row(const H40mTensorRecord& record, std::size_t row, std::span<float> out) const {
    if (record.dtype != "BF16" || record.shape.size() != 2) {
        throw std::invalid_argument("BF16 row reads require a 2D BF16 tensor");
    }
    const auto rows = record.shape[0];
    const auto cols = record.shape[1];
    if (row >= rows || out.size() != cols) throw std::invalid_argument("BF16 row shape mismatch");
    const auto row_bytes = cols * sizeof(std::uint16_t);
    if (record.length != rows * row_bytes) throw std::invalid_argument("BF16 tensor byte size mismatch");

    std::vector<std::uint16_t> bf16(cols);
    std::ifstream in(root_ / record.file, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open tensor shard");
    const auto offset = record.offset + row * row_bytes;
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) throw std::runtime_error("failed to seek BF16 row");
    in.read(reinterpret_cast<char*>(bf16.data()), static_cast<std::streamsize>(row_bytes));
    if (in.gcount() != static_cast<std::streamsize>(row_bytes)) throw std::runtime_error("short BF16 row read");
    for (std::size_t i = 0; i < cols; ++i) out[i] = bf16_to_float(bf16[i]);
}

void FileTensorReader::read_bf16_vector(const H40mTensorRecord& record, std::span<float> out) const {
    if (record.dtype != "BF16" || record.shape.size() != 1) {
        throw std::invalid_argument("BF16 vector reads require a 1D BF16 tensor");
    }
    if (out.size() != record.shape[0] || record.length != out.size() * sizeof(std::uint16_t)) {
        throw std::invalid_argument("BF16 vector shape mismatch");
    }
    std::vector<std::byte> bytes(record.length);
    read(record, bytes);
    const auto* values = reinterpret_cast<const std::uint16_t*>(bytes.data());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = bf16_to_float(values[i]);
}

void FileTensorReader::bf16_matvec(
    const H40mTensorRecord& record,
    std::span<const float> input,
    std::span<float> output,
    std::span<std::uint16_t> row_buffer) const {
    if (record.dtype != "BF16" || record.shape.size() != 2) {
        throw std::invalid_argument("BF16 matvec requires a 2D BF16 tensor");
    }
    const auto rows = record.shape[0];
    const auto cols = record.shape[1];
    if (input.size() != cols || output.size() != rows || row_buffer.size() < cols) {
        throw std::invalid_argument("BF16 matvec shape mismatch");
    }
    const auto row_bytes = cols * sizeof(std::uint16_t);
    if (record.length != rows * row_bytes) throw std::invalid_argument("BF16 matrix byte size mismatch");

    std::ifstream in(root_ / record.file, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open tensor shard");
    in.seekg(static_cast<std::streamoff>(record.offset), std::ios::beg);
    if (!in) throw std::runtime_error("failed to seek BF16 matrix");
    for (std::size_t row = 0; row < rows; ++row) {
        in.read(reinterpret_cast<char*>(row_buffer.data()), static_cast<std::streamsize>(row_bytes));
        if (in.gcount() != static_cast<std::streamsize>(row_bytes)) throw std::runtime_error("short BF16 matrix row read");
        float acc = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            acc += input[col] * bf16_to_float(row_buffer[col]);
        }
        output[row] = acc;
    }
}

void FileTensorReader::bf16_matvec_rows(
    const H40mTensorRecord& record,
    std::size_t row_begin,
    std::span<const float> input,
    std::span<float> output,
    std::span<std::uint16_t> row_buffer) const {
    if (record.dtype != "BF16" || record.shape.size() != 2) {
        throw std::invalid_argument("BF16 row-range matvec requires a 2D BF16 tensor");
    }
    const auto rows = record.shape[0];
    const auto cols = record.shape[1];
    if (input.size() != cols || row_buffer.size() < cols) {
        throw std::invalid_argument("BF16 row-range matvec shape mismatch");
    }
    if (row_begin > rows || output.size() > rows - row_begin) {
        throw std::out_of_range("BF16 row-range matvec rows outside tensor");
    }
    const auto row_bytes = cols * sizeof(std::uint16_t);
    if (record.length != rows * row_bytes) throw std::invalid_argument("BF16 matrix byte size mismatch");

    std::ifstream in(root_ / record.file, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open tensor shard");
    const auto offset = record.offset + row_begin * row_bytes;
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) throw std::runtime_error("failed to seek BF16 row range");
    for (std::size_t row = 0; row < output.size(); ++row) {
        in.read(reinterpret_cast<char*>(row_buffer.data()), static_cast<std::streamsize>(row_bytes));
        if (in.gcount() != static_cast<std::streamsize>(row_bytes)) throw std::runtime_error("short BF16 row-range read");
        float acc = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) acc += input[col] * bf16_to_float(row_buffer[col]);
        output[row] = acc;
    }
}

}  // namespace h40
