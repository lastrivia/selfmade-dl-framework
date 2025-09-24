#pragma once

#include <stdexcept>
#include <string>

inline std::runtime_error nn_except(const std::string &desc, const std::string &file, int line) {
    size_t pos = file.find("nn/");
    std::string err_info = std::string("[ERROR] ") + desc
                           + " (" + (pos == std::string::npos ? file : file.substr(pos))
                           + ":" + std::to_string(line) + ")";
    return std::runtime_error(err_info);
}
