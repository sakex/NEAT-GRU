//
// Created by alexandre on 16.05.20.
//

#include "bindings.h"

extern "C" {
EMSCRIPTEN_EXPORT
NN *network_from_string(char const *serialized) {
    using json = nlohmann::json;
    json j = json::parse(serialized);
    NeuralNetwork::Topology_ptr topology = std::make_unique<NeuralNetwork::Topology>(TopologyParser::parse(j));
    auto *net = new NN(*topology);
    return net;
}

EMSCRIPTEN_EXPORT
Topology *topology_from_string(char const *serialized) {
    using json = nlohmann::json;
    json j = json::parse(serialized);
    auto * topology = new Topology(TopologyParser::parse(j));
    return topology;
}

EMSCRIPTEN_EXPORT
NN *network_from_topology(NeuralNetwork::Topology *topology) {
    auto *net = new NN(*topology);
    return net;
}

EMSCRIPTEN_EXPORT
char *topology_to_string(NeuralNetwork::Topology *topology) {
    std::string serialized = topology->parse_to_string();
    char *out = (char *) malloc(sizeof(char) * (serialized.length() + 1));
    strcpy(out, serialized.c_str());
    return out;
}

EMSCRIPTEN_EXPORT
void fit(void *s, int const iterations, int const max_individuals, int const max_species, int const max_layers,
    int const max_per_layer,
    int const inputs, int const outputs) {
    auto *sim = reinterpret_cast<Simulation *>(s);
    auto *binding = new GameBinding(sim);
    Train::Train t(binding, iterations, max_individuals, max_species, max_layers, max_per_layer, inputs, outputs);
    t.start();
}

EMSCRIPTEN_EXPORT
double topology_delta_compatibility(NeuralNetwork::Topology const *top1, NeuralNetwork::Topology const *top2) {
    return NeuralNetwork::Topology::delta_compatibility(*top1, *top2);
}

EMSCRIPTEN_EXPORT
bool topologies_equal(NeuralNetwork::Topology const *top1, NeuralNetwork::Topology const *top2) {
    return *top1 == *top2;
}

EMSCRIPTEN_EXPORT
void delete_network(NeuralNetwork::NN *network) {
    delete network;
    network = nullptr;
}

}