#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <utility>

class progress_bar {
public:
    progress_bar(uint64_t max_steps, uint64_t max_length, std::string info = "") :
        max_steps_(max_steps), max_length_(max_length),
        step_(0), length_(0), percentage_(0),
        info_(std::move(info)),
        stop_(false), started_(false) {
    }

    void start() {
        start_ = std::chrono::steady_clock::now();
        last_upd_ = std::chrono::steady_clock::now();
        started_ = true;
        drawcall();
        timer_thread_ = std::thread(&progress_bar::timer_loop, this);
    }

    void step() {
        if (stop_.load() || !started_)
            return;

        {
            std::unique_lock<std::mutex> lock(data_mutex_);

            ++step_;
            uint64_t length = step_ * max_length_ / max_steps_;
            uint64_t percentage = step_ * 100LL / max_steps_;
            auto current = std::chrono::steady_clock::now();

            if (step_ == 1 || step_ == max_steps_ || length > length_ || percentage > percentage_
                || std::chrono::duration_cast<std::chrono::milliseconds>(current - last_upd_).count() > upd_interval_ms) {

                length_ = length;
                percentage_ = percentage;
                last_upd_ = current;
                drawcall();
            }

            if (step_ == max_steps_) {
                stop_.store(true);
                cv_.notify_one();
                std::cout << std::endl;
            }
        }
    }

    ~progress_bar() {
        stop_.store(true);
        cv_.notify_one();
        if (timer_thread_.joinable())
            timer_thread_.join();
    }

private:
    static void print_time(uint64_t time_sec) {
        if (time_sec >= 3600LL)
            std::cout << time_sec / 3600LL << ':';
        std::cout << std::setw(2) << std::setfill('0') << std::right << (time_sec / 60LL) % 60LL << ':';
        std::cout << std::setw(2) << std::setfill('0') << std::right << time_sec % 60LL;
    }

    void drawcall(bool from_timer = false) {

        std::cout << "\r\033[K";
        if (!info_.empty())
            std::cout << info_ << ' ';

        std::cout << std::setw(3) << std::setfill(' ') << std::right << percentage_ << "% |";
        for (uint64_t i = 0; i < length_; ++i)
            std::cout << '=';
        if (length_ != max_length_)
            std::cout << '>';
        for (uint64_t i = length_ + 1; i < max_length_; ++i)
            std::cout << ' ';
        std::cout << "| " << step_ << '/' << max_steps_ << " elapsed: ";

        auto current = std::chrono::steady_clock::now();
        uint64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start_).count();
        print_time(from_timer ? (elapsed_ms + 5LL) / 1000LL : elapsed_ms / 1000LL);

        if (step_ > 0 && step_ < max_steps_) {
            auto eta_ms = static_cast<uint64_t>(static_cast<double>(elapsed_ms) / static_cast<double>(step_) * static_cast<double>(max_steps_ - step_));
            std::cout << " eta: ";
            print_time(eta_ms / 1000LL);
        }
        if (step_ == max_steps_) {
            std::cout << '.' << std::setw(3) << std::setfill('0') << std::right << elapsed_ms % 1000LL;
        }

        std::cout << std::flush;
    }

    void timer_loop() {
        while (!stop_.load()) {
            {
                std::unique_lock<std::mutex> lock(timer_mutex_);
                auto current = std::chrono::steady_clock::now();
                uint64_t current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start_).count();
                uint64_t wait_ms = 1000 - current_ms % 1000;
                if (cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this]() { return stop_.load(); })) {
                    break;
                }
            }
            {
                std::unique_lock<std::mutex> t_lock(data_mutex_);
                last_upd_ = std::chrono::steady_clock::now();
                drawcall(true);
            }
        }
    }

    std::chrono::time_point<std::chrono::steady_clock> start_, last_upd_;
    static constexpr uint64_t upd_interval_ms = 50;
    uint64_t max_steps_, max_length_, step_, length_, percentage_;
    std::string info_;

    std::mutex data_mutex_, timer_mutex_;
    std::condition_variable cv_;
    std::thread timer_thread_;
    std::atomic<bool> stop_;
    bool started_;
};
