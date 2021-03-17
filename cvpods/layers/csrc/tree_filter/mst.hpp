#pragma once
#include <torch/extension.h>

extern at::Tensor mst_forward(
            const at::Tensor & edge_index_tensor,
            const at::Tensor & edge_weight_tensor,
            int vertex_count);

