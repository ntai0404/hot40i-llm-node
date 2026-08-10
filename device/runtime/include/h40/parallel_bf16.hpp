#pragma once

#include "h40/h40m_tensor_catalog.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace h40 {

class ParallelBf16Matvec {
public:
    ParallelBf16Matvec(
        const FileTensorReader& reader,
        std::size_t max_workers,
        std::size_t max_columns);
    ~ParallelBf16Matvec();

    ParallelBf16Matvec(const ParallelBf16Matvec&) = delete;
    ParallelBf16Matvec& operator=(const ParallelBf16Matvec&) = delete;

    void matvec(
        const H40mTensorRecord& record,
        std::span<const float> input,
        std::span<float> output,
        std::size_t workers);
    void matvec_rows(
        const H40mTensorRecord& record,
        std::size_t row_begin,
        std::span<const float> input,
        std::span<float> output,
        std::size_t workers);

    [[nodiscard]] std::size_t max_workers() const noexcept { return max_workers_; }
    [[nodiscard]] std::size_t max_columns() const noexcept { return max_columns_; }

private:
    void worker_loop(std::size_t worker_index);
    void run_partition(std::size_t worker_index);

    const FileTensorReader& reader_;
    std::size_t max_workers_{};
    std::size_t max_columns_{};
    std::vector<std::uint16_t> row_buffers_;
    std::vector<std::thread> workers_;
    std::mutex call_mutex_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    const H40mTensorRecord* record_{};
    const float* input_{};
    float* output_{};
    std::size_t input_size_{};
    std::size_t output_size_{};
    std::size_t row_begin_{};
    std::size_t active_workers_{};
    std::size_t remaining_workers_{};
    std::uint64_t generation_{};
    std::exception_ptr error_;
    bool stopping_{};
};

}  // namespace h40
