//
// Created by alexandre on 16.05.20.
//

#include "bindings.h"

double * compute_network(NetWrapper net, const double * inputs) {
    NN * actual_net = static_cast<NN*>(net.net);
    std::vector<double> outputs = actual_net->compute(inputs);
    return outputs.data();
}

void fit(void * s, int const iterations, int const max_individuals, int const inputs, int const outputs) {
    auto * sim = reinterpret_cast<Simulation*>(s);
    auto *binding = new GameBinding(sim);
    Train::Train t(binding, iterations, max_individuals, inputs, outputs);
    t.start();
}