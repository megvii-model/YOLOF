#pragma once
#include <torch/extension.h>

extern std::tuple<at::Tensor, at::Tensor, at::Tensor>
    tree_filter_refine_forward(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & self_weight_tensor,
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_index_tensor, 
        const at::Tensor & sorted_child_index_tensor 
    );

extern at::Tensor tree_filter_refine_backward_feature(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & self_weight_tensor,
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor,
        const at::Tensor & feature_aggr_tensor,
        const at::Tensor & feature_aggr_up_tensor,
        const at::Tensor & grad_out_tensor
    );

extern at::Tensor tree_filter_refine_backward_edge_weight(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & self_weight_tensor,
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor,
        const at::Tensor & feature_aggr_tensor,
        const at::Tensor & feature_aggr_up_tensor,
        const at::Tensor & grad_out_tensor
    );

extern at::Tensor tree_filter_refine_backward_self_weight(
        const at::Tensor & feature_in_tensor, 
        const at::Tensor & edge_weight_tensor, 
        const at::Tensor & self_weight_tensor,
        const at::Tensor & sorted_index_tensor, 
        const at::Tensor & sorted_parent_tensor, 
        const at::Tensor & sorted_child_tensor,
        const at::Tensor & feature_aggr_tensor,
        const at::Tensor & feature_aggr_up_tensor,
        const at::Tensor & grad_out_tensor
    );

