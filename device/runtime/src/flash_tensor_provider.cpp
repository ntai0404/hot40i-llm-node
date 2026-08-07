#include "h40/flash_tensor_provider.hpp"

#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace h40 {

FlashTensorProvider::FlashTensorProvider(const std::filesystem::path& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        throw std::system_error(errno, std::generic_category(), "open " + path.string());
    }
}

FlashTensorProvider::~FlashTensorProvider() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void FlashTensorProvider::read(const TensorSlice& slice, std::span<std::byte> out) {
    if (slice.length != out.size()) {
        throw std::invalid_argument("TensorSlice length must match output buffer size");
    }

    const auto start = std::chrono::steady_clock::now();
    std::size_t done = 0;
    while (done < out.size()) {
        const auto rc = ::pread(
            fd_,
            out.data() + done,
            out.size() - done,
            static_cast<off_t>(slice.offset + done));
        if (rc < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread " + path_.string());
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
