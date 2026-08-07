#pragma once

#include "h40/tensor_provider.hpp"

#include <atomic>
#include <filesystem>

namespace h40 {

class FlashTensorProvider final : public TensorProvider {
public:
    explicit FlashTensorProvider(const std::filesystem::path& path);
    ~FlashTensorProvider() override;

    FlashTensorProvider(const FlashTensorProvider&) = delete;
    FlashTensorProvider& operator=(const FlashTensorProvider&) = delete;

    void read(const TensorSlice& slice, std::span<std::byte> out) override;
    [[nodiscard]] ReadStats stats() const noexcept override;
    [[nodiscard]] std::string name() const override;

private:
    int fd_{-1};
    std::filesystem::path path_;
    std::atomic<std::uint64_t> ops_{0};
    std::atomic<std::uint64_t> bytes_{0};
    std::atomic<std::uint64_t> ns_{0};
};

} // namespace h40
