#include "h40/gptoss_expert.hpp"
#include "h40/h40m_tensor_catalog.hpp"
#include "h40/parallel_bf16.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kRows = 17;
constexpr std::size_t kCols = 32;

std::filesystem::path write_matrix() {
    const auto path = std::filesystem::temp_directory_path() / "h40_parallel_bf16_test.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t col = 0; col < kCols; ++col) {
            const float value = static_cast<float>((row + 1) * (col % 7 + 1)) / 64.0F;
            const auto bf16 = h40::float_to_bf16(value);
            out.write(reinterpret_cast<const char*>(&bf16), sizeof(bf16));
        }
    }
    return path;
}

void assert_close(std::span<const float> lhs, std::span<const float> rhs) {
    assert(lhs.size() == rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        assert(std::abs(lhs[i] - rhs[i]) <= 1.0e-6F);
    }
}

}  // namespace

int main() {
    const auto path = write_matrix();
    h40::FileTensorReader reader(path.parent_path());
    h40::H40mTensorRecord record{
        "test.weight",
        path.filename().string(),
        0,
        kRows * kCols * sizeof(std::uint16_t),
        "BF16",
        {kRows, kCols},
    };
    std::array<float, kCols> input{};
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i + 1) / 32.0F;

    std::vector<float> serial(kRows);
    std::array<std::uint16_t, kCols> row_buffer{};
    reader.bf16_matvec(record, input, serial, row_buffer);
    h40::ParallelBf16Matvec parallel(reader, 8, kCols);
    for (const std::size_t workers : {1U, 2U, 4U, 6U, 8U}) {
        std::vector<float> output(kRows);
        parallel.matvec(record, input, output, workers);
        assert_close(serial, output);

        std::vector<float> range(9);
        parallel.matvec_rows(record, 3, input, range, workers);
        assert_close(std::span<const float>(serial).subspan(3, range.size()), range);
    }
    std::filesystem::remove(path);
    return 0;
}
