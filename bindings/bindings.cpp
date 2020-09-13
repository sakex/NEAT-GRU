//
// Created by alexandre on 16.05.20.
//

#include "bindings.h"

NN *network_from_string(char const *serialized) {
    using json = nlohmann::json;
    json j = json::parse(serialized);
    NeuralNetwork::Topology_ptr topology = std::make_unique<NeuralNetwork::Topology>(TopologyParser::parse(j));
    auto *net = new NN(*topology);
    return net;
}

NN *network_from_topology(NeuralNetwork::Topology *topology) {
    auto *net = new NN(*topology);
    return net;
}

char *topology_to_string(NeuralNetwork::Topology *topology) {
    std::string serialized = topology->parse_to_string();
    char *out = (char *) malloc(sizeof(char) * serialized.length());
    strcpy(out, serialized.c_str());
    return out;
}

void fit(void *s, int const iterations, int const max_individuals, int const max_species, int const max_layers,
         int const max_per_layer,
         int const inputs, int const outputs) {
    auto *sim = reinterpret_cast<Simulation *>(s);
    auto *binding = new GameBinding(sim);
    Train::Train t(binding, iterations, max_individuals, max_species, max_layers, max_per_layer, inputs, outputs);
    t.start();
}

double topology_delta_compatibility(NeuralNetwork::Topology const * top1, NeuralNetwork::Topology const * top2) {
    return NeuralNetwork::Topology::delta_compatibility(*top1, *top2);
}