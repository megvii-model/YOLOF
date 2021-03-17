#pragma once
#include <torch/extension.h>

extern std::tuple<at::Tensor, at::Tensor, at::Tensor>
    bfs_forward(
        const at::Tensor & edge_index_tensor,
        int max_adj_per_node
    );

