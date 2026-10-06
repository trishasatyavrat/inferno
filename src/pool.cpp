#include "pool.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace inferno {
namespace {

// One unit of submitted work. It owns its own copy of the function and
// its own counters, so a worker that wakes late and finds this job
// already finished can look, see nothing left, and move on - nothing
// here dangles once parallel_for has returned.
//
// The first draft kept a single pair of counters in the pool and reset
// them for each job. That has a race: a straggler finishing job N does
// one last fetch_add on `next` *after* the caller has reset it for job
// N+1, claims task 0 of the new job, and runs it with job N's function -
// whose captures (A, B, C) no longer exist. Per-job state is the fix:
// stragglers only ever touch the job they were handed.
struct Job {
    std::function<void(size_t)> fn;
    size_t n_tasks;
    std::atomic<size_t> next{0};   // next task index to hand out
    std::atomic<size_t> done{0};   // tasks finished

    Job(const std::function<void(size_t)>& f, size_t n) : fn(f), n_tasks(n) {}

    // Pull tasks until none remain. Workers and the submitting thread
    // both run this - the caller is just another worker here.
    void work() {
        for (;;) {
            const size_t t = next.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tasks) return;
            fn(t);
            // release: the writes fn made to C happen-before the caller's
            // acquire load of `done` in Pool::run.
            done.fetch_add(1, std::memory_order_release);
        }
    }
};

class Pool {
public:
    Pool() {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        // hw-1 workers: the thread that calls parallel_for is the last one.
        for (unsigned i = 1; i < hw; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_.store(true);
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    size_t threads() const { return workers_.size() + 1; }

    void run(size_t n_tasks, const std::function<void(size_t)>& fn) {
        auto job = std::make_shared<Job>(fn, n_tasks);
        {
            std::lock_guard<std::mutex> lk(m_);
            current_ = job;
            generation_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        job->work();
        // Our own share is done; wait for the tasks other threads took.
        // Spin rather than sleep: what is left is microseconds away, and a
        // condition-variable round trip costs more than the wait itself.
        while (job->done.load(std::memory_order_acquire) < n_tasks)
            std::this_thread::yield();
    }

private:
    void worker_loop() {
        std::uint64_t seen = 0;
        for (;;) {
            // Spin briefly before sleeping. During decode, jobs arrive every
            // ~100 us; a worker that just finished one will almost certainly
            // see the next before a futex sleep + wake (tens of us each way)
            // could even complete. Past the spin budget, sleep for real, so
            // an idle process is not burning ten cores.
            bool woke = false;
            for (int i = 0; i < kSpinIters; ++i) {
                if (generation_.load(std::memory_order_acquire) != seen || stop_.load()) {
                    woke = true;
                    break;
                }
                std::this_thread::yield();
            }
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                if (!woke)
                    cv_.wait(lk, [&] {
                        return stop_.load() || generation_.load(std::memory_order_acquire) != seen;
                    });
                if (stop_.load()) return;
                seen = generation_.load(std::memory_order_acquire);
                job = current_;
            }
            // A worker that slept through a whole job lands here on a job
            // that is already complete; work() sees next >= n_tasks and
            // returns at once. That is the per-job design paying off.
            job->work();
        }
    }

    static constexpr int kSpinIters = 4000;

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_;
    std::shared_ptr<Job> current_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<bool> stop_{false};
};

Pool& pool() {
    static Pool p;  // built on first use, torn down at process exit
    return p;
}

} // namespace

void parallel_for(size_t n_tasks, const std::function<void(size_t)>& fn) {
    if (n_tasks == 0) return;
    if (n_tasks == 1) { fn(0); return; }  // nothing to share: skip the wake-up
    pool().run(n_tasks, fn);
}

size_t pool_threads() { return pool().threads(); }

} // namespace inferno
