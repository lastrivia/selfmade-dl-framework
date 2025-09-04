#pragma once

#include <cstdint>
#include <iostream>

class progress_bar {
public:
    progress_bar(uint64_t steps, uint64_t length): steps_(steps), length_(length), current_step_(0),
                                                   current_length_(0) {}

    void step() {
        ++current_step_;
        uint64_t new_length = current_step_ * length_ / steps_;
        while (new_length > current_length_) {
            std::cout << "=" << std::flush;
            ++current_length_;
        }
        if (current_step_ == steps_)
            std::cout << std::endl;
    }

private:
    uint64_t steps_, length_, current_step_, current_length_;
};
