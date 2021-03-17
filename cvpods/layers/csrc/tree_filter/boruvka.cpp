// Boruvka's algorithm to find Minimum Spanning 
// Tree of a given connected, undirected and 
// weighted graph 
#include <stdio.h> 
#include "boruvka.hpp"

// A structure to represent a subset for union-find 
struct subset 
{ 
    int parent;
    int rank;
}; 

// Function prototypes for union-find (These functions are defined 
// after boruvkaMST() ) 
int find(struct subset subsets[], int i); 
void Union(struct subset subsets[], int x, int y); 

// The main function for MST using Boruvka's algorithm 
void boruvkaMST(struct Graph* graph, int * edge_out) 
{ 
    // Get data of given graph
    int V = graph->V, E = graph->E;
    Edge *edge = graph->edge;

    // Allocate memory for creating V subsets.
    struct subset *subsets = new subset[V];

    // An array to store index of the cheapest edge of
    // subset. The stored index for indexing array 'edge[]'
    int *cheapest = new int[V];

    // Create V subsets with single elements
    for (int v = 0; v < V; ++v)
    {
        subsets[v].parent = v;
        subsets[v].rank = 0;
        cheapest[v] = -1;
    }

    // Initially there are V different trees.
    // Finally there will be one tree that will be MST
    int numTrees = V;
    int MSTweight = 0;

    // Keep combining components (or sets) until all
    // compnentes are not combined into single MST.
    while (numTrees > 1)
    {
        // Everytime initialize cheapest array
        for (int v = 0; v < V; ++v)
        {
            cheapest[v] = -1;
        }

        // Traverse through all edges and update
        // cheapest of every component
        for (int i=0; i<E; i++)
        {
            // Find components (or sets) of two corners
            // of current edge
            int set1 = find(subsets, edge[i].src);
            int set2 = find(subsets, edge[i].dest);

            // If two corners of current edge belong to
            // same set, ignore current edge
            if (set1 == set2)
                continue;

            // Else check if current edge is closer to previous
            // cheapest edges of set1 and set2
            else
            {
            if (cheapest[set1] == -1 ||
                edge[cheapest[set1]].weight > edge[i].weight)
                cheapest[set1] = i;

            if (cheapest[set2] == -1 ||
                edge[cheapest[set2]].weight > edge[i].weight)
                cheapest[set2] = i;
            }
        }

        // Consider the above picked cheapest edges and add them
        // to MST
        for (int i=0; i<V; i++)
        {
            // Check if cheapest for current set exists
            if (cheapest[i] != -1)
            {
                int set1 = find(subsets, edge[cheapest[i]].src);
                int set2 = find(subsets, edge[cheapest[i]].dest);

                if (set1 == set2)
                    continue;
                MSTweight += edge[cheapest[i]].weight;
                *(edge_out++) = edge[cheapest[i]].src;
                *(edge_out++) = edge[cheapest[i]].dest;
                //printf("Edge %d-%d included in MST\n",
                    //edge[cheapest[i]].src, edge[cheapest[i]].dest);

                // Do a union of set1 and set2 and decrease number
                // of trees
                Union(subsets, set1, set2);
                numTrees--;
            }
        }
    }

    delete[] subsets;
    delete[] cheapest;
}

// Creates a graph with V vertices and E edges 
struct Graph* createGraph(int V, int E) 
{ 
    Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    graph->edge = new Edge[E];
    return graph;
} 

// A utility function to find set of an element i 
// (uses path compression technique) 
int find(struct subset subsets[], int i) 
{ 
    // find root and make root as parent of i
    // (path compression)
    if (subsets[i].parent != i)
    subsets[i].parent =
            find(subsets, subsets[i].parent);

    return subsets[i].parent;
} 

// A function that does union of two sets of x and y 
// (uses union by rank) 
void Union(struct subset subsets[], int x, int y) 
{ 
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    // Attach smaller rank tree under root of high
    // rank tree (Union by Rank)
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;

    // If ranks are same, then make one as root and
    // increment its rank by one
    else
    {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
} 

