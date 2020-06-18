//
// Created by alexandre on 16.06.20.
//

#ifndef NEAT_GRU_CUDAPHENOTYPE_CUH
#define NEAT_GRU_CUDAPHENOTYPE_CUH


struct CUDAPhenotype {
    float const input_weight;
    float const memory_weight;
    float const reset_input_weight;
    float const reset_memory_weight;
    float const update_input_weight;
    float const update_memory_weight;
    int input_pos;
    int output_pos;
};

struct CUDAConnectionCount {
    int pos;
    size_t count;
};

#endif //NEAT_GRU_CUDAPHENOTYPE_CUH