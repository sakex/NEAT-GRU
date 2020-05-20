//
// Created by alexandre on 16.05.20.
//

#ifndef NEAT_BINDINGS_H
#define NEAT_BINDINGS_H

#include "../Train/Train.h"
#include "GameBinding.h"
#include "structs.h"

extern "C" {
double * compute_network(NetWrapper net, const double * inputs);

void fit(Simulation sim, int iterations, int max_individuals, int inputs, int outputs);
}

#endif //NEAT_BINDINGS_H
