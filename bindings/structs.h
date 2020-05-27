//
// Created by alexandre on 20.05.20.
//

#ifndef NEAT_GRU_STRUCTS_H
#define NEAT_GRU_STRUCTS_H

extern "C" {
typedef struct NetWrapper{
    void *net;
} NetWrapper;


typedef struct Simulation{
    double *(*run_generation)(void const *);
    void (*reset_players)(void const *, NetWrapper *, unsigned);
    void const * context;
} Simulation;
}

#endif //NEAT_GRU_STRUCTS_H
