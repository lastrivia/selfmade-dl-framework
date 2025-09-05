#pragma once

// todo multithreaded backend, unused yet

#include <thread>
#include <vector>
#include <queue>
#include <future>
#include <functional>
#include <condition_variable>
#include <mutex>

class thread_pool {
public:
    class task_token {
    public:
        explicit task_token(std::shared_future<void> fut) noexcept
            : future_(std::move(fut)) {}

        void join() noexcept {
            future_.wait();
        }

    private:
        std::shared_future<void> future_;
    };

    static task_token run(const std::function<void()> &func) noexcept {
        static thread_pool pool;
        return pool.run_(func);
    }

private:
    explicit thread_pool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() { this->worker_loop(); });
        }
    }

    ~thread_pool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &t: workers_) {
            if (t.joinable()) t.join();
        }
    }

    task_token run_(const std::function<void()> &func) noexcept {
        std::shared_ptr<std::packaged_task<void()> > task =
            std::make_shared<std::packaged_task<void()> >(func);

        std::shared_future<void> fut = task->get_future().share();

        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();

        return task_token(fut);
    }

    void worker_loop() noexcept {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty())
                    return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }

            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()> > tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};
