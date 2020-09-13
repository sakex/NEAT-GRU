//
// Created by alexandre on 16.05.20.
//

#ifndef NEAT_BINDINGS_H
#define NEAT_BINDINGS_H

#include "../Train/Train.h"
#include "GameBinding.h"
#include "structs.h"
#include <nlohmann/json.hpp>
#include <TopologyParser.h>
#include "NN.h"

extern "C" {

/**
 * Binding to allow a computation of a neural network
 *
 * Definined in NeuralNetwork/NN.cpp for optimisation reasons
 *
 * @param net The network to be computed
 * @param inputs The inputs of the computation
 * @return The output of the computation
 */
double *compute_network(NeuralNetwork::NN *net, const double *inputs);

/**
 * Binding to reset the hidden state of a neural network
 *
 * Definined in NeuralNetwork/NN.cpp for optimisation reasons
 *
 * @param net The network to be reset
 */
void reset_network_state(NeuralNetwork::NN *net);

/**
 * Generate a neural network from a json serialized string
 *
 * @param serialized - The string to convert
 * @return - Pointer to a neural network on the heap
 */
NN *network_from_string(char const *serialized);

/**
 * Generate a neural network from a topology pointer
 *
 * @param topology - Pointer to a topology to convert
 * @return - Pointer to a neural network on the heap
 */
NN *network_from_topology(NeuralNetwork::Topology *topology);

/**
 * Serializes a topology pointer to string
 *
 * @param topology - Topology pointer to be serialized
 * @return - A c string representing the serialized topology
 */
char * topology_to_string(NeuralNetwork::Topology * topology);

/**
 * Binding to call Train::fit
 *
 * @param sim Simulation to run
 * @param iterations Number of iterations
 * @param max_individuals Maximum number of individuals in a given generation
 * @param max_species Maximum number of species for a given generation
 * @param max_layers Maximum number of layers in a given network
 * @param max_per_layer Maximum number of neurons per layer in a given network
 * @param inputs Number of neurons on the input layer
 * @param outputs Number of neurons on the output layer
 */
void fit(void *sim, int iterations, int max_individuals, int max_species, int max_layers, int max_per_layers, int inputs, int outputs);

/**
 * Bindings call to Topology::delta_compatibility
 *
 * @param top1 First topology
 * @param top2 Second topology
 * @return delta between both topologies
 */
double topology_delta_compatibility(NeuralNetwork::Topology const * top1, NeuralNetwork::Topology const * top2);
}

#endif //NEAT_BINDINGS_H
