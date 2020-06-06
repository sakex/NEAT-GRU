//
// Created by alexandre on 16.05.20.
//

#ifndef NEAT_BINDINGS_H
#define NEAT_BINDINGS_H

#include "../Train/Train.h"
#include "GameBinding.h"
#include "structs.h"

extern "C" {
    /**
     * Binding to allow a computation of a neural network
     *
     * @param net The network to be computed
     * @param inputs The inputs of the computation
     * @return The output of the computation
     */
double * compute_network(void * net, const double * inputs);

/**
 * Binding to call Train::fit
 *
 * @param sim Simulation to run
 * @param iterations Number of iterations
 * @param max_individuals Maximum number of individuals in a given generation
 * @param max_layers Maximum number of layers in a given network
 * @param max_per_layer Maximum number of neurons per layer in a given network
 * @param inputs Number of neurons on the input layer
 * @param outputs Number of neurons on the output layer
 */
void fit(void * sim, int iterations, int max_individuals, int max_layers, int max_per_layers, int inputs, int outputs);
}

#endif //NEAT_BINDINGS_H
