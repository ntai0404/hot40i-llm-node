#pragma once

#include "h40/expert_loader.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <thread>

namespace h40 {

struct ExpertReadPipelineStats {
    std::uint64_t submitted{};
    std::uint64_t completed{};
    std::uint64_t bytes{};
    std::uint64_t read_nanoseconds{};
    std::uint64_t wait_nanoseconds{};
};

struct PrefetchedExpert {
    ExpertKey key{};
    std::span<const std::byte> bytes;
    std::uint64_t read_nanoseconds{};
    std::uint64_t wait_nanoseconds{};
};

class ExpertReadPipeline {
public:
    ExpertReadPipeline(const ExpertLoader& loader, std::span<std::byte> staging_buffer);
    ~ExpertReadPipeline();

    ExpertReadPipeline(const ExpertReadPipeline&) = delete;
    ExpertReadPipeline& operator=(const ExpertReadPipeline&) = delete;

    void submit(ExpertKey key, bool verify_checksum = false);
    [[nodiscard]] PrefetchedExpert wait();
    [[nodiscard]] bool busy() const noexcept;
    [[nodiscard]] ExpertReadPipelineStats stats() const noexcept;

private:
    void worker_loop();

    const ExpertLoader& loader_;
    std::span<std::byte> staging_buffer_;
    mutable std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable result_ready_;
    std::thread worker_;
    ExpertKey key_{};
    std::size_t result_bytes_{};
    std::uint64_t result_read_nanoseconds_{};
    std::exception_ptr error_;
    ExpertReadPipelineStats stats_{};
    bool verify_checksum_{};
    bool job_available_{};
    bool running_{};
    bool result_available_{};
    bool stopping_{};
};

}  // namespace h40
