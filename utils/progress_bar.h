#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>

class progress_bar {
public:
    progress_bar(uint64_t max_steps, uint64_t max_length) : max_steps_(max_steps), max_length_(max_length),
                                                            step_(0), length_(0), percentage_(0) {}

    void step() {
        ++step_;
        uint64_t length = step_ * max_length_ / max_steps_;
        uint64_t percentage = step_ * 100LL / max_steps_;
        if (step_ == 1 || length > length_ || percentage > percentage_) {
            std::cout << "\r[" << std::setw(3) << std::setfill(' ') << percentage << "%] ";
            for (uint64_t i = 0; i < length; ++i) {
                std::cout << '=';
            }
            std::cout << std::flush;
        }
        if (step_ == max_steps_)
            std::cout << std::endl;
    }

private:
    uint64_t max_steps_, max_length_, step_, length_, percentage_;
};
