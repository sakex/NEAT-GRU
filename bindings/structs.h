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
    double *(*run_generation)(void *);
    void (*reset_players)(void *, NetWrapper *, unsigned);
    void * context;
} Simulation;
}

#endif //NEAT_GRU_STRUCTS_H
