#include "h40/parallel_bf16.hpp"

#include <algorithm>
#include <stdexcept>

namespace h40 {

ParallelBf16Matvec::ParallelBf16Matvec(
    const FileTensorReader& reader,
    std::size_t max_workers,
    std::size_t max_columns)
    : reader_(reader),
      max_workers_(max_workers),
      max_columns_(max_columns),
      row_buffers_(max_workers * max_columns) {
    if (max_workers_ == 0) throw std::invalid_argument("BF16 worker count must be non-zero");
    if (max_columns_ == 0) throw std::invalid_argument("BF16 row-buffer width must be non-zero");
    workers_.reserve(max_workers_ - 1);
    for (std::size_t worker = 1; worker < max_workers_; ++worker) {
        workers_.emplace_back(&ParallelBf16Matvec::worker_loop, this, worker);
    }
}

ParallelBf16Matvec::~ParallelBf16Matvec() {
    std::scoped_lock call_lock(call_mutex_);
    {
        std::scoped_lock lock(mutex_);
        stopping_ = true;
    }
    work_ready_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void ParallelBf16Matvec::matvec(
    const H40mTensorRecord& record,
    std::span<const float> input,
    std::span<float> output,
    std::size_t workers) {
    if (record.shape.size() != 2 || output.size() != record.shape[0]) {
        throw std::invalid_argument("parallel BF16 full matvec output shape mismatch");
    }
    matvec_rows(record, 0, input, output, workers);
}

void ParallelBf16Matvec::matvec_rows(
    const H40mTensorRecord& record,
    std::size_t row_begin,
    std::span<const float> input,
    std::span<float> output,
    std::size_t workers) {
    if (record.dtype != "BF16" || record.shape.size() != 2) {
        throw std::invalid_argument("parallel BF16 matvec requires a 2D BF16 tensor");
    }
    if (workers == 0 || workers > max_workers_) {
        throw std::invalid_argument("parallel BF16 worker count is outside configured bounds");
    }
    if (input.size() != record.shape[1] || input.size() > max_columns_) {
        throw std::invalid_argument("parallel BF16 matvec input shape exceeds configured row buffer");
    }
    if (row_begin > record.shape[0] || output.size() > record.shape[0] - row_begin) {
        throw std::out_of_range("parallel BF16 matvec row range is outside tensor");
    }
    if (output.empty()) return;
    workers = std::min(workers, output.size());

    std::scoped_lock call_lock(call_mutex_);
    if (workers == 1) {
        reader_.bf16_matvec_rows(
            record,
            row_begin,
            input,
            output,
            std::span<std::uint16_t>(row_buffers_).first(max_columns_));
        return;
    }

    {
        std::scoped_lock lock(mutex_);
        record_ = &record;
        input_ = input.data();
        input_size_ = input.size();
        output_ = output.data();
        output_size_ = output.size();
        row_begin_ = row_begin;
        active_workers_ = workers;
        remaining_workers_ = workers - 1;
        error_ = nullptr;
        ++generation_;
    }
    work_ready_.notify_all();

    try {
        run_partition(0);
    } catch (...) {
        std::scoped_lock lock(mutex_);
        if (!error_) error_ = std::current_exception();
    }

    std::unique_lock lock(mutex_);
    work_done_.wait(lock, [&] { return remaining_workers_ == 0; });
    const auto error = error_;
    lock.unlock();
    if (error) std::rethrow_exception(error);
}

void ParallelBf16Matvec::worker_loop(std::size_t worker_index) {
    std::uint64_t seen_generation = 0;
    while (true) {
        {
            std::unique_lock lock(mutex_);
            work_ready_.wait(lock, [&] { return stopping_ || generation_ != seen_generation; });
            if (stopping_) return;
            seen_generation = generation_;
            if (worker_index >= active_workers_) continue;
        }

        try {
            run_partition(worker_index);
        } catch (...) {
            std::scoped_lock lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }

        {
            std::scoped_lock lock(mutex_);
            if (remaining_workers_ == 0) {
                if (!error_) error_ = std::make_exception_ptr(std::logic_error("BF16 worker accounting underflow"));
            } else {
                --remaining_workers_;
            }
            if (remaining_workers_ == 0) work_done_.notify_one();
        }
    }
}

void ParallelBf16Matvec::run_partition(std::size_t worker_index) {
    const auto begin = output_size_ * worker_index / active_workers_;
    const auto end = output_size_ * (worker_index + 1) / active_workers_;
    const auto columns = record_->shape[1];
    auto row_buffer = std::span<std::uint16_t>(row_buffers_)
                          .subspan(worker_index * max_columns_, columns);
    reader_.bf16_matvec_rows(
        *record_,
        row_begin_ + begin,
        std::span<const float>(input_, input_size_),
        std::span<float>(output_ + begin, end - begin),
        row_buffer);
}

}  // namespace h40
