#include <thread>
#include <iostream>
#include <stdlib.h>
#include <fstream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

/* Switch of minimal spanning tree algorithms */
/* Note: we will migrate the cuda implementaion to PyTorch in the next version */
//#define MST_PRIM
//#define MST_KRUSKAL
#define MST_BORUVKA

#ifdef MST_PRIM
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/prim_minimum_spanning_tree.hpp>
#endif
#ifdef MST_KRUSKAL 
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/kruskal_min_spanning_tree.hpp>
#endif
#ifdef MST_BORUVKA 
    #include "boruvka.hpp"
#endif


#ifndef MST_BORUVKA
    using namespace boost;
    typedef adjacency_list <vecS, vecS, undirectedS, no_property,
            property < edge_weight_t, float > > Graph;
    typedef graph_traits < Graph >::edge_descriptor Edge;
    typedef graph_traits < Graph >::vertex_descriptor Vertex;
    typedef std::pair<int, int> E;
#endif

static void forward_kernel(int * edge_index, float * edge_weight, int * edge_out, int vertex_count, int edge_count){
#ifdef MST_BORUVKA
    struct Graph * g = createGraph(vertex_count, edge_count);
    for (int i = 0; i < edge_count; ++i){
        g->edge[i].src = edge_index[i * 2];
        g->edge[i].dest = edge_index[i * 2 + 1];
        g->edge[i].weight = edge_weight[i];
    }
#else
    Graph g(vertex_count);
    for (int i = 0; i < edge_count; ++i)
        boost::add_edge((int)edge_index[i * 2], (int)edge_index[i * 2 + 1],
                edge_weight[i], g);
#endif

#ifdef MST_PRIM
    std::vector < graph_traits < Graph >::vertex_descriptor > p(num_vertices(g));
    prim_minimum_spanning_tree(g, &(p[0]));
    int * edge_out_ptr = edge_out;
    for (std::size_t i = 0; i != p.size(); ++i)
        if (p[i] != i) {
            *(edge_out_ptr++) = i;
            *(edge_out_ptr++) = p[i];
        }
#endif
    
#ifdef MST_KRUSKAL
    std::vector < Edge > spanning_tree;
    kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
    float * edge_out_ptr = edge_out;
    for (std::vector < Edge >::iterator ei = spanning_tree.begin();
            ei != spanning_tree.end(); ++ei){
        *(edge_out_ptr++) = source(*ei, g);
        *(edge_out_ptr++) = target(*ei, g);
    }
#endif

#ifdef MST_BORUVKA
    boruvkaMST(g, edge_out);
    delete[] g->edge;
    delete[] g;
#endif

}
    
at::Tensor mst_forward(
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

