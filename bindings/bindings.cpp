//
// Created by alexandre on 16.05.20.
//

#include "bindings.h"

double *compute_network(void *net, const double *inputs) {
    NN *actual_net = reinterpret_cast<NN *>(net);
    double* outputs = actual_net->compute(inputs);
    return outputs;
}

void fit(void *s, int const iterations, int const max_individuals, int const max_layers, int const max_per_layer,
         int const inputs, int const outputs) {
    auto *sim = reinterpret_cast<Simulation *>(s);
    auto *binding = new GameBinding(sim);
    Train::Train t(binding, iterations, max_individuals, max_layers, max_per_layer, inputs, outputs);
    t.start();
}