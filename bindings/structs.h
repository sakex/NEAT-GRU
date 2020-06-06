//
// Created by alexandre on 20.05.20.
//

#ifndef NEAT_GRU_STRUCTS_H
#define NEAT_GRU_STRUCTS_H

extern "C" {
/// C binding for the Simulation
typedef struct Simulation {
    /**
     * Logic for each generation (has to be implemented or it will be undefined behaviour)
     *
     * @param cont Context to call the method on
     * @return A C array of scores
     */
    double *(*run_generation)(void *cont);

    /**
     * Reset players implementation (has to be implemented or it will be undefined behaviour)
     *
     * @param cont Context to call the method on
     * @param networks New neural networks for the next generation
     * @param size Number of networks passed
     */
    void (*reset_players)(void *cont, void *networks, unsigned ptr_size, unsigned size);

    /// Optional field, if a context has to be kept
    void *context;
} Simulation;
}

#endif //NEAT_GRU_STRUCTS_H
