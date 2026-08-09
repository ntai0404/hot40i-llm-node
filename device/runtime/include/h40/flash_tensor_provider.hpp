#pragma once

#include "h40/tensor_provider.hpp"

#include <atomic>
#include <functional>
#include <filesystem>
#include <mutex>

namespace h40 {

struct FlashReadTrace {
    TensorSlice slice{};
    std::uint64_t nanoseconds{};
};

class FlashTensorProvider final : public TensorProvider {
public:
    using TraceSink = std::function<void(const FlashReadTrace&)>;

    explicit FlashTensorProvider(const std::filesystem::path& path);
    ~FlashTensorProvider() override;

    FlashTensorProvider(const FlashTensorProvider&) = delete;
    FlashTensorProvider& operator=(const FlashTensorProvider&) = delete;

    void read(const TensorSlice& slice, std::span<std::byte> out) override;
    [[nodiscard]] ReadStats stats() const noexcept override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    void set_trace_sink(TraceSink sink);

private:
    int fd_{-1};
    std::filesystem::path path_;
    std::uint64_t file_size_{0};
    std::atomic<std::uint64_t> ops_{0};
    std::atomic<std::uint64_t> bytes_{0};
    std::atomic<std::uint64_t> ns_{0};
    mutable std::mutex io_mutex_;
    mutable std::mutex trace_mutex_;
    TraceSink trace_sink_;
};

} // namespace h40
