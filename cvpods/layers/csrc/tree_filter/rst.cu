#include <thread>
#include <iostream>
#include <stdlib.h>
#include <fstream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "boruvka_rst.hpp"

static void forward_kernel(int * edge_index, float * edge_weight, int * edge_out, int vertex_count, int edge_count){
    struct Graph * g = create_graph(vertex_count, edge_count);
    for (int i = 0; i < edge_count; ++i){
        g->edge[i].src = edge_index[i * 2];
        g->edge[i].dest = edge_index[i * 2 + 1];
        g->edge[i].weight = edge_weight[i];
    }
    
    boruvka_rst(g, edge_out);

    delete[] g->edge;
    delete[] g;
}
    
at::Tensor rst_forward(
            const at::Tensor & edge_index_tensor,
            const at::Tensor & edge_weight_tensor,
            int vertex_count){
    unsigned batch_size = edge_index_tensor.size(0);
    unsigned edge_count = edge_index_tensor.size(1);
    
    auto edge_index_cpu   = edge_index_tensor.cpu();
    auto edge_weight_cpu  = edge_weight_tensor.cpu(); 
    auto edge_out_cpu     = at::empty({batch_size, vertex_count - 1, 2}, edge_index_cpu.options());
    
    int * edge_out      = edge_out_cpu.contiguous().data_ptr<int>();
    int * edge_index    = edge_index_cpu.contiguous().data_ptr<int>();
    float * edge_weight = edge_weight_cpu.contiguous().data_ptr<float>(); 

    // Loop for batch
    std::thread pids[batch_size];
    for (unsigned i = 0; i < batch_size; i++){
        auto edge_index_iter  = edge_index + i * edge_count * 2;
        auto edge_weight_iter = edge_weight + i * edge_count;
        auto edge_out_iter    = edge_out + i * (vertex_count - 1) * 2;
        pids[i] = std::thread(forward_kernel, edge_index_iter, edge_weight_iter, edge_out_iter, vertex_count, edge_count);
    }
    
    for (unsigned i = 0; i < batch_size; i++){
        pids[i].join();
    }
    
    auto edge_out_tensor = edge_out_cpu.to(edge_index_tensor.device());
    
    return edge_out_tensor;
}

