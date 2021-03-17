#include <math.h>
#include <thread>
#include <vector>
#include <deque>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define CUDA_NUM_THREADS 64
#define GET_CUDA_BLOCKS(N) ceil((float)N / CUDA_NUM_THREADS)

__global__ void adj_vec_kernel(
        int batch_size, 
        int * edge_index, 
        int vertex_count,
        int * adj_vec,
        int * adj_vec_len,
        int max_adj_per_node){

    const int edge_count    = vertex_count - 1;
    const int batch_idx     = blockIdx.x;
    const int thread_idx    = threadIdx.x;
    const int thread_count  = blockDim.x;

    edge_index  += batch_idx * edge_count * 2;
    adj_vec     += batch_idx * vertex_count * max_adj_per_node;
    adj_vec_len += batch_idx * vertex_count;

    for (int i = thread_idx; i < edge_count; i += thread_count){
        int source = edge_index[2 * i];
        int target = edge_index[2 * i + 1];
        int source_len = atomicAdd(&(adj_vec_len[source]), 1);
        adj_vec[source * max_adj_per_node + source_len] = target;
        int target_len = atomicAdd(&(adj_vec_len[target]), 1);
        adj_vec[target * max_adj_per_node + target_len] = source;
    }
}

__global__ void breadth_first_sort_kernel(
        int * sorted_index,
        int * sorted_parent_index,
        int * sorted_child_index,
        int * adj_vec,
        int * adj_vec_len,
        int * parent_index,
        int batch_size,
        int vertex_count,
        int max_adj_per_node){

    const int batch_idx     = blockIdx.x;
    const int thread_idx    = threadIdx.x;
    const int thread_count  = blockDim.x;

    adj_vec              += batch_idx * vertex_count * max_adj_per_node;
    adj_vec_len          += batch_idx * vertex_count;
    parent_index         += batch_idx * vertex_count;
    sorted_index         += batch_idx * vertex_count;
    sorted_parent_index  += batch_idx * vertex_count;
    sorted_child_index   += batch_idx * vertex_count * max_adj_per_node;

    __shared__ int sorted_len;
    if (thread_idx == 0) {
        sorted_len = 1;
        parent_index[0] = 0;
        sorted_index[0] = 0;
        sorted_parent_index[0] = 0;
    }
    __syncthreads();

    int i = thread_idx;
    while (i < vertex_count){
        if ((sorted_index[i] > 0) || (i == 0)){
            int child_index = 0;
            int par         = parent_index[i];
            int cur         = sorted_index[i];
            for (int j = 0; j < adj_vec_len[cur]; j++){
                int child = adj_vec[cur * max_adj_per_node + j];
                if (child != par){
                    int pos = atomicAdd(&(sorted_len), 1);
                    sorted_index[pos]        = child;
                    parent_index[pos]        = cur;
                    sorted_parent_index[pos] = i;
                    sorted_child_index[i * max_adj_per_node + child_index] = pos;
                    child_index++;
                }
            }
            i += thread_count;
        }
        __syncthreads();
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bfs_forward(
    const at::Tensor & edge_index_tensor,
    int max_adj_per_node){

    int batch_size   = edge_index_tensor.size(0);
    int vertex_count = edge_index_tensor.size(1) + 1;

    auto options = edge_index_tensor.options();
    auto sorted_index_tensor    = at::zeros({batch_size, vertex_count}, options);
    auto sorted_parent_tensor   = at::zeros({batch_size, vertex_count}, options);
    auto sorted_child_tensor    = at::zeros({batch_size, vertex_count, max_adj_per_node}, options);
    auto adj_vec_tensor         = at::zeros({batch_size, vertex_count, max_adj_per_node}, options);
    auto adj_vec_len_tensor     = at::zeros({batch_size, vertex_count}, options);
    auto parent_index_tensor    = at::zeros({batch_size, vertex_count}, options);

    int * edge_index      = edge_index_tensor.contiguous().data_ptr<int>();
    int * sorted_index    = sorted_index_tensor.contiguous().data_ptr<int>();
    int * sorted_parent   = sorted_parent_tensor.contiguous().data_ptr<int>();
    int * sorted_child    = sorted_child_tensor.contiguous().data_ptr<int>();
    int * adj_vec         = adj_vec_tensor.contiguous().data_ptr<int>();
    int * adj_vec_len     = adj_vec_len_tensor.contiguous().data_ptr<int>();
    int * parent_index    = parent_index_tensor.contiguous().data_ptr<int>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();        

    dim3 block_dims(CUDA_NUM_THREADS, 1, 1), grid_dims(batch_size, 1, 1);
    adj_vec_kernel <<< grid_dims, block_dims, 0, stream >>>(
            batch_size, edge_index, vertex_count, adj_vec, adj_vec_len, max_adj_per_node);

    breadth_first_sort_kernel <<< grid_dims, block_dims, 1, stream >>>(
            sorted_index, sorted_parent, sorted_child, adj_vec, adj_vec_len, parent_index,
            batch_size, vertex_count, max_adj_per_node);

    return std::make_tuple(sorted_index_tensor, sorted_parent_tensor, sorted_child_tensor);
}
