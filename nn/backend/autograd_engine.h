#pragma once

#include <ranges>
#include <vector>
#include <deque>
#include <unordered_map>

#include "tensor.h"
#include "kernel_dispatcher.h"

class grad_engine {
public:
    static void backward(const tensor &root) {
        backward(root.object_);
    }

private:
    struct dfs_frame {
        tensor_impl *object;
        bool expanded;
    };

    struct dag_info {
        std::vector<tensor_impl *> edges_out;
        size_t edges_in;
        bool is_in_path; /* (info not exist):   never visited;
                            true:               in path;
                            false:              waiting or popped */
    };

    struct fast_ptr_hash {
        template<typename any>
        size_t operator()(any *ptr) const noexcept {
            size_t x = reinterpret_cast<size_t>(ptr);
            return (x >> 3) ^ (x >> 19);
        }
    };

    class grad_ctx {
        // manages grad_data of internal nodes
        friend class grad_engine;

        std::vector<tensor_impl *> registered;

        void register_tensor(tensor_impl *object) {
            object->grad_data_ = object->alloc_data(nullptr);
            registered.push_back(object);
            object->zero_grad();
        }

        ~grad_ctx() {
            for (tensor_impl *object: registered) {
                object->release_data(object->grad_data_);
                object->grad_data_ = nullptr;
            }
        }
    };

    static void backward(tensor_impl *const root) {

        std::vector<dfs_frame> dfs_stack;
        std::unordered_map<tensor_impl *, dag_info, fast_ptr_hash> dag;
        std::deque<tensor_impl *> ready;

        grad_ctx ctx;

        dfs_stack.emplace_back(root, false);
        dag.emplace(root, dag_info{{}, 0, false});

        // traverse DAG
        while (!dfs_stack.empty()) {
            tensor_impl *object = dfs_stack.back().object;
            if (!dfs_stack.back().expanded) {
                dfs_stack.back().expanded = true;
                dag[object].is_in_path = true;

                grad_node *node = object->grad_node_;
                if (node) { // internal tensor
                    std::vector<tensor_impl *> inputs = node->inputs();
                    for (tensor_impl *input: std::ranges::reverse_view(inputs)) {
                        auto it = dag.find(input);
                        if (it == dag.end()) {
                            dfs_stack.emplace_back(input, false);
                            dag.emplace(input, dag_info{{}, 1, false});
                        }
                        else {
                            dag_info &info = it->second;
                            if (info.is_in_path) {
                                // unexpected
                                throw nn_except("autograd: cycle detected in computation graph", __FILE__, __LINE__);
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
        case data_type::fp32:
            dispatch_kernel(root->device_).broadcast_fp32(root->shape_.size, root->grad_data_, 1.0f);
            break;
        case data_type::int32:
            dispatch_kernel(root->device_).broadcast_int32(root->shape_.size, root->grad_data_, 1);
            break;
        }

        ready.push_back(root);

        // run backward
        while (!ready.empty()) {
            tensor_impl *object = ready.front();
            ready.pop_front();
            if (object->grad_node_) { // internal tensor
                object->grad_node_->backward();

                dag_info &info = dag[object];
                for (tensor_impl *next: info.edges_out) {
                    dag_info &next_info = dag[next];
                    next_info.edges_in--;
                    if (!next_info.edges_in)
                        ready.push_back(next);
                }

                // cudaError_t err = cudaDeviceSynchronize();
                // if (err != cudaSuccess) {
                //     throw nn_except(std::string() + cudaGetErrorString(err) + " at " + std::string(object->shape_), __FILE__, __LINE__);
                // }
                // else
                //     std::cout << "backward " << std::string(object->shape_) << " executed;" << std::endl;
            }
        }
    }
};

inline void tensor::backward() {
    if (!object_->requires_grad_)
        throw nn_except("backward tensor is not in autograd graph", __FILE__, __LINE__);
    grad_engine::backward(*this);
}
