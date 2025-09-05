#pragma once

#include <unordered_map>
#include <vector>

class mem_pool {
public:
    template<typename T>
    static T *alloc(size_t size) {
        return reinterpret_cast<T *>(instance().alloc_(size * sizeof(T)));
    }

    template<typename T>
    static void recycle(T *p) {
        instance().recycle_(reinterpret_cast<char *>(p));
    }

private:
    static mem_pool &instance() {
        thread_local mem_pool instance;
        return instance;
    }

    std::unordered_map<size_t, std::vector<char *> > recycle_pool{};

    std::unordered_map<char *, size_t> allocated_pool{};

    char *alloc_(size_t size_bytes) {
        auto it = recycle_pool.find(size_bytes);
        char *ret;
        if (it == recycle_pool.end() || it->second.empty())
            ret = new char[size_bytes];
        else {
            ret = it->second.back();
            it->second.pop_back();
        }
        allocated_pool[ret] = size_bytes;
        return ret;
    }

    void recycle_(char *p) {
        auto it = allocated_pool.find(p);
        if (it == allocated_pool.end()) // exception?
            delete[] p;
        else {
            recycle_pool[it->second].push_back(p);
            allocated_pool.erase(it);
        }
    }

    ~mem_pool() {
        for (const auto& it: recycle_pool)
            for (char *p: it.second)
                delete[] p;
        for (auto it: allocated_pool)
            delete[] it.first;
    }
};
