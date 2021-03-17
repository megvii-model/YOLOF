#pragma once

// a structure to represent a weighted edge in graph 
struct Edge 
{ 
    int src, dest;
    float weight; 
}; 

// a structure to represent a connected, undirected 
// and weighted graph as a collection of edges. 
struct Graph 
{ 
    // V-> Number of vertices, E-> Number of edges
    int V, E;

    // graph is represented as an array of edges.
    // Since the graph is undirected, the edge
    // from src to dest is also edge from dest
    // to src. Both are counted as 1 edge here.
    Edge* edge;
}; 

extern struct Graph* createGraph(int V, int E); 
extern void boruvkaMST(struct Graph* graph, int * edge_out);

