#pragma once

// Draft interface.

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <coroutine>
#include <mutex>
#include <vector>

namespace pulsar {

// Starts when awaited and resumes its awaiter when it completes. It runs on the event_loop of the task awaiting it, or
// the one that spawned it. Copies share the coroutine, which is destroyed with the last copy. Once cancelled, a
// completion no longer resumes the awaiter.
template <typename t> class [[nodiscard]] task {
  public:
    struct promise_type;

  private:
    std::coroutine_handle<promise_type> handle;

    explicit task(std::coroutine_handle<promise_type> handle) noexcept;

  public:
    task(const task& other) noexcept;
    task(task&& other) noexcept;
    task& operator=(const task& other) noexcept;
    task& operator=(task&& other) noexcept;
    ~task();

    void cancel() noexcept;
    bool cancelled() const noexcept;

    bool await_ready() const noexcept;
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiter) noexcept;
    t await_resume();
};

class event_loop {
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<std::coroutine_handle<>> ready;
    bool stopping = false;

  public:
    // Safe to call from any thread, including CUDA host functions.
    void schedule(std::coroutine_handle<> handle);
    // Starts root on this loop.
    void spawn(task<void> root);
    // Resumes scheduled coroutines on the calling thread until stop, or until a resumed coroutine throws, which it
    // rethrows.
    void run();
    void stop() noexcept;
};

// Completes once the work queued on stream so far has finished, resuming on the awaiting task's event_loop.
task<void> synchronize(cudaStream_t stream);

}  // namespace pulsar
