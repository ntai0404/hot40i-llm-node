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
}

FlashTensorProvider::~FlashTensorProvider() {
    if (fd_ >= 0) {
        H40_CLOSE(fd_);
    }
}

void FlashTensorProvider::read(const TensorSlice& slice, std::span<std::byte> out) {
    if (slice.length != out.size()) {
        throw std::invalid_argument("TensorSlice length must match output buffer size");
    }

    const auto start = std::chrono::steady_clock::now();
    std::size_t done = 0;
    while (done < out.size()) {
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
}

ReadStats FlashTensorProvider::stats() const noexcept {
    return {ops_.load(), bytes_.load(), ns_.load()};
}

std::string FlashTensorProvider::name() const { return "flash:" + path_.string(); }

} // namespace h40
