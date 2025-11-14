#pragma once

#include <ranges>
#include <vector>
#include <deque>
#include <unordered_map>

#include "tensor_impl.h"
#include "backend.h"

class GradEngine {
public:
    static void backward(const Tensor &root) {
        backward(root.object_);
    }

private:
    struct DfsFrame {
        TensorImpl *object;
        bool expanded;
    };

    struct DagInfo {
        std::vector<TensorImpl *> edges_out;
        size_t edges_in;
        bool is_in_path; /* (info not exist):   never visited;
                            true:               in path;
                            false:              waiting or popped */
    };

    struct FastPtrHash {
        template<typename any>
        size_t operator()(any *ptr) const noexcept {
            size_t x = reinterpret_cast<size_t>(ptr);
            return (x >> 3) ^ (x >> 19);
        }
    };

    class GradCtx {
        // manages grad_data of internal nodes
        friend class GradEngine;

        std::vector<TensorImpl *> registered;

        void register_tensor(TensorImpl *object) {
            object->grad_data_ = object->alloc_data(nullptr);
            registered.push_back(object);
            object->zero_grad();
        }

        ~GradCtx() {
            for (TensorImpl *object: registered) {
                object->release_data(object->grad_data_);
                object->grad_data_ = nullptr;
            }
        }
    };

    static void backward(TensorImpl *const root) {

        std::vector<DfsFrame> dfs_stack;
        std::unordered_map<TensorImpl *, DagInfo, FastPtrHash> dag;
        std::deque<TensorImpl *> ready;

        GradCtx ctx;

        dfs_stack.emplace_back(root, false);
        dag.emplace(root, DagInfo{{}, 0, false});

        // traverse DAG
        while (!dfs_stack.empty()) {
            TensorImpl *object = dfs_stack.back().object;
            if (!dfs_stack.back().expanded) {
                dfs_stack.back().expanded = true;
                dag[object].is_in_path = true;

                GradNode *node = object->grad_node_;
                if (node) { // internal tensor
                    std::vector<TensorImpl *> inputs = node->inputs();
                    for (TensorImpl *input: std::ranges::reverse_view(inputs)) {
                        auto it = dag.find(input);
                        if (it == dag.end()) {
                            dfs_stack.emplace_back(input, false);
                            dag.emplace(input, DagInfo{{}, 1, false});
                        }
                        else {
                            DagInfo &info = it->second;
                            if (info.is_in_path) {
                                // unexpected
                                throw FatalExcept("autograd: cycle detected in computation graph", __FILE__, __LINE__);
                            }
                            info.edges_in++;
                        }
                    }
                    dag[object].edges_out = std::move(inputs);

                    ctx.register_tensor(object);
                }
                // else: leaf tensor, pass
            }
            else {
                dfs_stack.pop_back();
                dag[object].is_in_path = false;
            }
        }

        switch (root->dtype_) {
        case ScalarType::fp32:
            dispatch_kernel(root->device_).broadcast_fp32(root->shape_.size, root->grad_data_, 1.0f);
            break;
        case ScalarType::int32:
            dispatch_kernel(root->device_).broadcast_int32(root->shape_.size, root->grad_data_, 1);
            break;
        }

        ready.push_back(root);

        // run backward
        while (!ready.empty()) {
            TensorImpl *object = ready.front();
            ready.pop_front();
            if (object->grad_node_) { // internal tensor
                object->grad_node_->backward();

                DagInfo &info = dag[object];
                for (TensorImpl *next: info.edges_out) {
                    DagInfo &next_info = dag[next];
                    next_info.edges_in--;
                    if (!next_info.edges_in)
                        ready.push_back(next);
                }
            }
        }
    }
};

inline void Tensor::backward() {
    if (!object_->requires_grad_)
        throw FatalExcept("backward tensor is not in autograd graph", __FILE__, __LINE__);
    GradEngine::backward(*this);
}
