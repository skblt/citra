// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <type_traits>

namespace Common {

struct AsyncHandle {
public:
    [[nodiscard]] bool IsBuilt() noexcept {
        return is_built.load(std::memory_order::relaxed);
    }

    void WaitBuilt() noexcept {
        std::unique_lock lock{mutex};
        condvar.wait(lock, [this] { return is_built.load(std::memory_order::relaxed); });
    }

    void MarkBuilt() noexcept {
        std::scoped_lock lock{mutex};
        is_built = true;
        condvar.notify_all();
    }

private:
    std::condition_variable condvar;
    std::mutex mutex;
    std::atomic_bool is_built{false};
};

} // namespace Common
