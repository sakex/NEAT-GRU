//
// Created by sakex on 13.11.19.
//

#include "TopologyParser.h"

NeuralNetwork::Topology TopologyParser::parse(nlohmann::json &j) {
    using namespace NeuralNetwork;
    Topology topology;
    int max_layers = 0;
    std::vector<Phenotype *> new_phenotypes;
    for (auto &it : j) {
        Phenotype::point input = {it["input"][0], it["input"][1]};
        Phenotype::point output = {it["output"][0], it["output"][1]};
        if (output[0] > max_layers)
            max_layers = output[0];
        float const input_weight = it["input_weight"];
        float const memory_weight = it["memory_weight"];
        float const reset_input_weight = it["reset_input_weight"];
        float const reset_memory_weight = it["reset_memory_weight"];
        float const update_memory_weight = it["update_memory_weight"];
        bool disabled = it["disabled"];
        new_phenotypes.push_back(
                new Phenotype(input, output, input_weight, memory_weight, reset_input_weight, reset_memory_weight,
                              update_memory_weight, disabled, 0));
    }
    topology.set_layers(max_layers + 1);
    for (Phenotype *phen_ptr : new_phenotypes) {
        topology.add_relationship(phen_ptr, true);
    }
    return topology;
}