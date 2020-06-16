//
// Created by alexandre on 16.06.20.
//

#ifndef NEAT_GRU_CUDAPHENOTYPE_CUH
#define NEAT_GRU_CUDAPHENOTYPE_CUH


struct CUDAPhenotype {
    double const input_weight;
    double const memory_weight;
    double const reset_input_weight;
    double const reset_memory_weight;
    double const update_input_weight;
    double const update_memory_weight;
    int input_pos[2];
    int output_pos[2];
};


#endif //NEAT_GRU_CUDAPHENOTYPE_CUH
