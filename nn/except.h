#pragma once

#include <stdexcept>
#include <string>

[[nodiscard]] inline std::runtime_error FatalExcept(const std::string &desc, std::string file, int line) {
    for (char &c: file)
        if (c == '\\')
            c = '/';
    size_t pos = file.find("nn/");
    std::string err_info = std::string("[ERROR] ") + desc
                           + " (" + (pos == std::string::npos ? file : file.substr(pos))
                           + ":" + std::to_string(line) + ")";
    std::cerr << '\n' << err_info << std::endl;
    return std::runtime_error(err_info);
}
