#pragma once

#include <unordered_map>
#include <vector>

#include "base_backend.h"
#include "../except.h"
#include "cuda/arch.cuh"

inline bool mem_pool_log = false;

template<DeviceType device>
class BaseMemPool {
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
    static BaseMemPool &instance() noexcept {
        thread_local BaseMemPool instance;
        return instance;
    }

    std::unordered_map<size_t, std::vector<char *> > recycle_pool_{};

    std::unordered_map<char *, size_t> allocated_pool_{};

    char *device_new(size_t size_bytes) {
        char *ret;
        if constexpr (device == DeviceType::cpu) {
            try {
                ret = new char[size_bytes];
            } catch (std::bad_alloc &) {
                throw FatalExcept("memory allocation failed", __FILE__, __LINE__);
            }
        }
        else if constexpr (device == DeviceType::cuda) {
            cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&ret), size_bytes, cuda_backend::default_stream());
            if (err != cudaSuccess) {
                throw FatalExcept("cuda memory allocation failed", __FILE__, __LINE__);
            }
        }
        return ret;
    }

    void device_delete(char *p) noexcept {
        if constexpr (device == DeviceType::cpu)
            delete[] p;
        else if constexpr (device == DeviceType::cuda)
            cudaFreeAsync(p, cuda_backend::default_stream());
        if (mem_pool_log)
            std::cout << "FREE ON " << (device == DeviceType::cpu ? "HOST" : "DEVICE")
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
            std::cout << "MALLOC ON " << (device == DeviceType::cpu ? "HOST" : "DEVICE")
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
                std::cout << "RECYCLE ON " << (device == DeviceType::cpu ? "HOST" : "DEVICE")
                        << ": " << static_cast<void *>(p) << ", size: " << it->second << std::endl;
            recycle_pool_[it->second].push_back(p);
            allocated_pool_.erase(it);
        }
    }

    ~BaseMemPool() {
        for (const auto &it: recycle_pool_)
            for (char *p: it.second)
                device_delete(p);
        for (auto it: allocated_pool_)
            device_delete(it.first);
    }
};

using MemPool = BaseMemPool<DeviceType::cpu>;
using CudaMemPool = BaseMemPool<DeviceType::cuda>;

class Workspace {
public:
    explicit Workspace(DeviceDesc device) :
        workspace_(nullptr), device_(device) {}

    void init(size_t bytes) {
        if (workspace_)
            recycle(workspace_);
        workspace_ = alloc(bytes);
    }

    explicit Workspace(size_t bytes, DeviceDesc device) :
        device_(device) {
        workspace_ = alloc(bytes);
    }

    ~Workspace() {
        recycle(workspace_);
    }

    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;

    Workspace(Workspace &&other) noexcept {
        workspace_ = other.workspace_;
        other.workspace_ = nullptr;
        device_ = other.device_;
    }

    Workspace &operator=(Workspace &&other) noexcept {
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
    DeviceDesc device_;

    void *alloc(size_t bytes) const {
        switch (device_.type) {
        case DeviceType::cpu:
            return MemPool::alloc<char>(bytes);
        case DeviceType::cuda:
            return CudaMemPool::alloc<char>(bytes);
        }
        return nullptr;
    }

    void recycle(void *ptr) const {
        switch (device_.type) {
        case DeviceType::cpu:
            MemPool::recycle(ptr);
            return;
        case DeviceType::cuda:
            CudaMemPool::recycle(ptr);
            return;
        }
    }
};
