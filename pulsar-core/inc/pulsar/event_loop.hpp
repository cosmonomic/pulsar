#pragma once

// Draft interface.

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <coroutine>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace pulsar {

class event_loop;

// Starts when awaited and resumes its awaiter when it completes. Awaiting consumes the task: await a prvalue or
// std::move(t). An exception escaping the coroutine panics event_loop::current().
template <typename t> class [[nodiscard]] task {
  public:
    struct promise_type;
    class awaiter;

  private:
    std::coroutine_handle<promise_type> handle;

    explicit task(std::coroutine_handle<promise_type> handle) noexcept;

    friend class event_loop;

  public:
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    task(task&& other) noexcept;
    task& operator=(task&& other) noexcept;
    ~task();

    awaiter operator co_await() && noexcept;
};

// Once set, schedules every coroutine awaiting it on event_loop::current(); awaiting it after does not suspend. Set and
// await it on the loop thread. Awaiters link into the event, so it does not move, and it must not be destroyed while
// awaited.
class event {
  public:
    class awaiter;

  private:
    awaiter* waiters = nullptr;
    bool is_set = false;

  public:
    event() noexcept = default;
    event(const event&) = delete;
    event& operator=(const event&) = delete;
    ~event();

    void set() noexcept;
    awaiter operator co_await() noexcept;
};

// Resolved from outside the loop (a CUDA callback, a Python callback), possibly before it is awaited. Awaiting
// consumes it, as with task. Await it on the loop thread; resolving schedules the awaiter on that loop.
template <typename t> class [[nodiscard]] external_task {
    struct future;

    std::shared_ptr<future> shared;

    explicit external_task(std::shared_ptr<future> shared) noexcept;

  public:
    class resolver;
    class awaiter;

    static std::pair<external_task, resolver> make();

    external_task(const external_task&) = delete;
    external_task& operator=(const external_task&) = delete;
    external_task(external_task&& other) noexcept;
    external_task& operator=(external_task&& other) noexcept;
    ~external_task();

    awaiter operator co_await() && noexcept;
};

// Safe to call from any thread, once. Does nothing once its external_task was destroyed, so the value is dropped.
template <typename t> class external_task<t>::resolver {
    std::shared_ptr<future> shared;

    explicit resolver(std::shared_ptr<future> shared) noexcept;

    friend class external_task;

  public:
    void resolve(t value) const;
};

class event_loop {
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<std::coroutine_handle<>> ready;
    std::exception_ptr exception;
    bool stopping = false;
    std::vector<task<void>> roots;

    void panic(std::exception_ptr thrown) noexcept;

    template <typename> friend class task;

  public:
    // The loop running on the calling thread, null outside run.
    static event_loop* current() noexcept;

    // Destroys the remaining roots without resuming them.
    ~event_loop();

    // Safe to call from any thread.
    void schedule(std::coroutine_handle<> handle);
    // Safe to call from any thread.
    void spawn(task<void> root);
    // Resumes scheduled coroutines on the calling thread until stop, or until a task panics. Returns the panic's
    // exception, or null after stop.
    std::exception_ptr run();
    // Safe to call from any thread.
    void stop() noexcept;
};

// Completes once the work queued on stream so far has finished. Uses cudaStreamAddCallback, which unlike
// cudaLaunchHostFunc also runs after a CUDA error; an error panics.
task<void> synchronize(cudaStream_t stream);

}  // namespace pulsar
