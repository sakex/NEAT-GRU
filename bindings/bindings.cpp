//
// Created by alexandre on 16.05.20.
//

#include "bindings.h"

double *compute_network(NN *net, const double *inputs) {
    double *outputs = net->compute(inputs);
    return outputs;
}

void reset_network_state(NN *net) {
    net->reset_state();
}

NN *network_from_string(char const *serialized) {
    using json = nlohmann::json;
    json j = json::parse(serialized);
    Topology_ptr topology = std::make_unique<Topology>(TopologyParser::parse(j));
    auto * net = new NN(topology);
    return net;
}

void fit(void *s, int const iterations, int const max_individuals, int const max_layers, int const max_per_layer,
         int const inputs, int const outputs) {
    auto *sim = reinterpret_cast<Simulation *>(s);
    auto *binding = new GameBinding(sim);
    Train::Train t(binding, iterations, max_individuals, max_layers, max_per_layer, inputs, outputs);
    t.start();
}