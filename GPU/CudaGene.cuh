//
// Created by alexandre on 16.06.20.
//

#ifndef NEAT_GRU_CUDAGENE_CUH
#define NEAT_GRU_CUDAGENE_CUH


struct CUDAGene {
    double const input_weight;
    double const memory_weight;
    double const reset_input_weight;
    double const reset_memory_weight;
    double const update_input_weight;
    double const update_memory_weight;
    int input_pos;
    int output_pos;
};

struct CUDAConnectionCount {
    int pos;
    size_t count;
};

#endif //NEAT_GRU_CUDAGENE_CUH
