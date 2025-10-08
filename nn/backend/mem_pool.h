#pragma once

#include <unordered_map>
#include <vector>

#include "base_kernel.h"
#include "../except.h"

template<device_type device>
class base_mem_pool {
public:
    template<typename T>
    [[nodiscard]] static T *alloc(size_t size) {
        return reinterpret_cast<T *>(instance().alloc_p(size * sizeof(T)));
    }

    template<typename T>
    static void recycle(T *p) {
        instance().recycle_p(reinterpret_cast<char *>(p));
    }

    static void recycle(void *p) {
        instance().recycle_p(static_cast<char *>(p));
    }

private:
    static base_mem_pool &instance() {
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
            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&ret), size_bytes);
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
            cudaFree(p);
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
        return ret;
    }

    void recycle_p(char *p) {
        if (p == nullptr)
            return;
        auto it = allocated_pool_.find(p);
        if (it == allocated_pool_.end()) // unexpected
            device_delete(p);
        else {
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

template<device_type device>
class base_workspace {
public:
    base_workspace() {
        workspace_ = nullptr;
    }

    void init(size_t bytes) {
        if (workspace_)
            base_mem_pool<device>::recycle(workspace_);
        workspace_ = base_mem_pool<device>::template alloc<char>(bytes);
    }

    explicit base_workspace(size_t bytes) {
        workspace_ = base_mem_pool<device>::template alloc<char>(bytes);
    }

    ~base_workspace() {
        base_mem_pool<device>::recycle(workspace_);
    }

    base_workspace(const base_workspace &) = delete;
    base_workspace &operator=(const base_workspace &) = delete;
    base_workspace(base_workspace &&other) = delete;
    base_workspace &operator=(base_workspace &&other) = delete;

    operator void *() const {
        return workspace_;
    }

    template<typename T>
    operator T *() const {
        return static_cast<T *>(workspace_);
    }

private:
    void *workspace_;
};

using workspace = base_workspace<device_type::cpu>;
using cuda_workspace = base_workspace<device_type::cuda>;
