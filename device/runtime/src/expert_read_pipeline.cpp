#include "h40/expert_read_pipeline.hpp"

#include <chrono>
#include <stdexcept>

namespace h40 {

ExpertReadPipeline::ExpertReadPipeline(
    const ExpertLoader& loader,
    std::span<std::byte> staging_buffer)
    : loader_(loader), staging_buffer_(staging_buffer) {
    if (staging_buffer_.empty()) throw std::invalid_argument("expert prefetch staging buffer must be non-empty");
    worker_ = std::thread(&ExpertReadPipeline::worker_loop, this);
}

ExpertReadPipeline::~ExpertReadPipeline() {
    {
        std::scoped_lock lock(mutex_);
        stopping_ = true;
    }
    work_ready_.notify_one();
    if (worker_.joinable()) worker_.join();
}

void ExpertReadPipeline::submit(ExpertKey key, bool verify_checksum) {
    const auto record = loader_.index().find_record(key);
    if (!record.has_value()) throw std::out_of_range("expert key missing from model index");
    if (record->slice.length > staging_buffer_.size()) {
        throw std::invalid_argument("expert prefetch staging buffer is too small");
    }
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) throw std::logic_error("expert read pipeline is stopping");
        if (job_available_ || running_ || result_available_) {
            throw std::logic_error("expert read pipeline already has outstanding work");
        }
        key_ = key;
        verify_checksum_ = verify_checksum;
        error_ = nullptr;
        job_available_ = true;
        ++stats_.submitted;
    }
    work_ready_.notify_one();
}

PrefetchedExpert ExpertReadPipeline::wait() {
    const auto wait_start = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex_);
    if (!job_available_ && !running_ && !result_available_) {
        throw std::logic_error("expert read pipeline has no outstanding work");
    }
    result_ready_.wait(lock, [&] { return result_available_; });
    const auto wait_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_start)
            .count());
    stats_.wait_nanoseconds += wait_ns;
    const auto key = key_;
    const auto bytes = result_bytes_;
    const auto read_ns = result_read_nanoseconds_;
    const auto error = error_;
    result_available_ = false;
    error_ = nullptr;
    lock.unlock();
    if (error) std::rethrow_exception(error);
    return {key, staging_buffer_.first(bytes), read_ns, wait_ns};
}

bool ExpertReadPipeline::busy() const noexcept {
    std::scoped_lock lock(mutex_);
    return job_available_ || running_ || result_available_;
}

ExpertReadPipelineStats ExpertReadPipeline::stats() const noexcept {
    std::scoped_lock lock(mutex_);
    return stats_;
}

void ExpertReadPipeline::worker_loop() {
    while (true) {
        ExpertKey key;
        bool verify_checksum = false;
        {
            std::unique_lock lock(mutex_);
            work_ready_.wait(lock, [&] { return stopping_ || job_available_; });
            if (stopping_ && !job_available_) return;
            key = key_;
            verify_checksum = verify_checksum_;
            job_available_ = false;
            running_ = true;
        }

        std::size_t bytes = 0;
        std::exception_ptr error;
        const auto read_start = std::chrono::steady_clock::now();
        try {
            bytes = loader_.load(key, staging_buffer_, verify_checksum).bytes.size();
        } catch (...) {
            error = std::current_exception();
        }
        const auto read_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - read_start)
                .count());

        {
            std::scoped_lock lock(mutex_);
            result_bytes_ = bytes;
            result_read_nanoseconds_ = read_ns;
            error_ = error;
            running_ = false;
            result_available_ = true;
            if (!error) {
                ++stats_.completed;
                stats_.bytes += bytes;
                stats_.read_nanoseconds += read_ns;
            }
        }
        result_ready_.notify_one();
    }
}

}  // namespace h40
