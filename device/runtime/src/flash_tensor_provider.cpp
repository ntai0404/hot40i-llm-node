#include "h40/flash_tensor_provider.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef _WIN32
#define H40_OPEN _open
#define H40_CLOSE _close
#define H40_READ _read
#define H40_SEEK _lseeki64
#define H40_RDONLY _O_RDONLY
#define H40_BINARY _O_BINARY
using h40_file_offset_t = long long;
#else
#define H40_OPEN ::open
#define H40_CLOSE ::close
#define H40_READ ::read
#define H40_PREAD ::pread
#define H40_SEEK ::lseek
#define H40_RDONLY O_RDONLY
#define H40_BINARY 0
using h40_file_offset_t = off_t;
#endif

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

namespace h40 {

FlashTensorProvider::FlashTensorProvider(const std::filesystem::path& path) : path_(path) {
    const auto path_string = path.string();
    fd_ = H40_OPEN(path_string.c_str(), H40_RDONLY | H40_BINARY | O_CLOEXEC);
    if (fd_ < 0) {
        throw std::system_error(errno, std::generic_category(), "open " + path.string());
    }
    const auto size = std::filesystem::file_size(path);
    if (size > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("file too large for FlashTensorProvider");
    }
    file_size_ = static_cast<std::uint64_t>(size);
}

FlashTensorProvider::~FlashTensorProvider() {
    if (fd_ >= 0) {
        H40_CLOSE(fd_);
    }
}

namespace {

void validate_range(const TensorSlice& slice, std::size_t out_size, std::uint64_t file_size) {
    if (slice.length != out_size) {
        throw std::invalid_argument("TensorSlice length must match output buffer size");
    }
    if (slice.offset > file_size || slice.length > file_size - slice.offset) {
        throw std::out_of_range("TensorSlice range exceeds flash tensor file size");
    }
    const auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<h40_file_offset_t>::max());
    if (slice.offset > max_offset || slice.length > max_offset - slice.offset) {
        throw std::out_of_range("TensorSlice range exceeds platform file offset range");
    }
}

}  // namespace

void FlashTensorProvider::read(const TensorSlice& slice, std::span<std::byte> out) {
    validate_range(slice, out.size(), file_size_);

    const auto start = std::chrono::steady_clock::now();
    std::size_t done = 0;
    while (done < out.size()) {
#ifdef _WIN32
        std::scoped_lock lock(io_mutex_);
        if (H40_SEEK(fd_, static_cast<h40_file_offset_t>(slice.offset + done), SEEK_SET) < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "seek " + path_.string());
        }
        const auto remaining = out.size() - done;
        const auto count = static_cast<unsigned int>(
            std::min<std::size_t>(remaining, std::numeric_limits<unsigned int>::max()));
        const auto rc = H40_READ(fd_, out.data() + done, count);
        if (rc < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "read " + path_.string());
        }
#else
        const auto remaining = out.size() - done;
        const auto count = std::min<std::size_t>(remaining, std::numeric_limits<ssize_t>::max());
        const auto offset = static_cast<h40_file_offset_t>(slice.offset + done);
        const auto rc = H40_PREAD(fd_, out.data() + done, count, offset);
        if (rc < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread " + path_.string());
        }
#endif
        if (rc == 0) {
            throw std::runtime_error("unexpected EOF while reading tensor slice");
        }
        done += static_cast<std::size_t>(rc);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    ops_.fetch_add(1, std::memory_order_relaxed);
    bytes_.fetch_add(done, std::memory_order_relaxed);
    ns_.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order_relaxed);
    TraceSink sink;
    {
        std::scoped_lock lock(trace_mutex_);
        sink = trace_sink_;
    }
    if (sink) {
        sink({slice, static_cast<std::uint64_t>(
                         std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count())});
    }
}

ReadStats FlashTensorProvider::stats() const noexcept {
    return {ops_.load(), bytes_.load(), ns_.load()};
}

std::string FlashTensorProvider::name() const { return "flash:" + path_.string(); }

void FlashTensorProvider::set_trace_sink(TraceSink sink) {
    std::scoped_lock lock(trace_mutex_);
    trace_sink_ = std::move(sink);
}

} // namespace h40
