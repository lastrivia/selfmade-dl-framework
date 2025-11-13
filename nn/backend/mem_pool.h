#pragma once

#include <unordered_map>
#include <vector>

#include "base_backend.h"
#include "../except.h"
#include "cuda/arch.cuh"

inline bool mem_pool_log = false;

template<device_type device>
class base_mem_pool {
public:
    template<typename T>
    [[nodiscard]] static T *alloc(size_t size) {
        return reinterpret_cast<T *>(instance().alloc_p(size * sizeof(T)));
    }

    template<typename T>
    static void recycle(T *p) noexcept {
        instance().recycle_p(reinterpret_cast<char *>(p));
    }

    static void recycle(void *p) noexcept {
        instance().recycle_p(static_cast<char *>(p));
    }

    static void query(const void *p) {
        auto it = instance().allocated_pool_.find(const_cast<char *>(static_cast<const char *>(p)));
        if (it == instance().allocated_pool_.end()) {
            std::cout << p << " not found" << std::endl;
        }
        else
            std::cout << p << " found, size: " << it->second << std::endl;
    }

    static void query_allocated() {
        size_t sum = 0;
        for (auto &it : instance().allocated_pool_) {
            sum += it.second;
        }
        std::cout << "Allocated sum: " << sum << std::endl;
    }

private:
    static base_mem_pool &instance() noexcept {
        thread_local base_mem_pool instance;
        return instance;
    }

    std::unordered_map<size_t, std::vector<char *> > recycle_pool_{};

    std::unordered_map<char *, size_t> allocated_pool_{};

    char *device_new(size_t size_bytes) {
        char *ret;
        if constexpr (device == device_type::cpu) {
            try {
                ret = new char[size_bytes];
            } catch (std::bad_alloc &) {
                throw nn_except("memory allocation failed", __FILE__, __LINE__);
            }
        }
        else if constexpr (device == device_type::cuda) {
            cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&ret), size_bytes, cuda_backend::default_stream());
            if (err != cudaSuccess) {
                throw nn_except("cuda memory allocation failed", __FILE__, __LINE__);
            }
        }
        return ret;
    }

    void device_delete(char *p) noexcept {
        if constexpr (device == device_type::cpu)
            delete[] p;
        else if constexpr (device == device_type::cuda)
            cudaFreeAsync(p, cuda_backend::default_stream());
        if (mem_pool_log)
            std::cout << "FREE ON " << (device == device_type::cpu ? "HOST" : "DEVICE")
                    << ": " << static_cast<void *>(p) << std::endl;
    }

    char *alloc_p(size_t size_bytes) {
        if (size_bytes == 0)
            return nullptr;
        auto it = recycle_pool_.find(size_bytes);
        char *ret;
        if (it == recycle_pool_.end() || it->second.empty()) {
            ret = device_new(size_bytes);
        }
        else {
            ret = it->second.back();
            it->second.pop_back();
        }
        allocated_pool_[ret] = size_bytes;
        if (mem_pool_log)
            std::cout << "MALLOC ON " << (device == device_type::cpu ? "HOST" : "DEVICE")
                    << ": " << static_cast<void *>(ret) << ", size: " << size_bytes << std::endl;
        return ret;
    }

    void recycle_p(char *p) noexcept {
        if (p == nullptr)
            return;
        auto it = allocated_pool_.find(p);
        if (it == allocated_pool_.end()) // unexpected
            device_delete(p);
        else {
            if (mem_pool_log)
                std::cout << "RECYCLE ON " << (device == device_type::cpu ? "HOST" : "DEVICE")
                        << ": " << static_cast<void *>(p) << ", size: " << it->second << std::endl;
            recycle_pool_[it->second].push_back(p);
            allocated_pool_.erase(it);
        }
    }

    ~base_mem_pool() {
        for (const auto &it: recycle_pool_)
            for (char *p: it.second)
                device_delete(p);
        for (auto it: allocated_pool_)
            device_delete(it.first);
    }
};

using mem_pool = base_mem_pool<device_type::cpu>;
using cuda_mem_pool = base_mem_pool<device_type::cuda>;

class workspace {
public:
    explicit workspace(device_desc device) :
        workspace_(nullptr), device_(device) {}

    void init(size_t bytes) {
        if (workspace_)
            recycle(workspace_);
        workspace_ = alloc(bytes);
    }

    explicit workspace(size_t bytes, device_desc device) :
        device_(device) {
        workspace_ = alloc(bytes);
    }

    ~workspace() {
        recycle(workspace_);
    }

    workspace(const workspace &) = delete;
    workspace &operator=(const workspace &) = delete;

    workspace(workspace &&other) noexcept {
        workspace_ = other.workspace_;
        other.workspace_ = nullptr;
        device_ = other.device_;
    }

    workspace &operator=(workspace &&other) noexcept {
        if (workspace_ == other.workspace_)
            return *this;
        if (workspace_)
            recycle(workspace_);
        workspace_ = other.workspace_;
        other.workspace_ = nullptr;
        device_ = other.device_;
        return *this;
    }

    operator void *() const { // NOLINT(*-explicit-constructor)
        return workspace_;
    }

    template<typename T>
    operator T *() const { // NOLINT(*-explicit-constructor)
        return static_cast<T *>(workspace_);
    }

private:
    void *workspace_;
    device_desc device_;

    void *alloc(size_t bytes) const {
        switch (device_.type) {
        case device_type::cpu:
            return mem_pool::alloc<char>(bytes);
        case device_type::cuda:
            return cuda_mem_pool::alloc<char>(bytes);
        }
        return nullptr;
    }

    void recycle(void *ptr) const {
        switch (device_.type) {
        case device_type::cpu:
            mem_pool::recycle(ptr);
            return;
        case device_type::cuda:
            cuda_mem_pool::recycle(ptr);
            return;
        }
    }
};
