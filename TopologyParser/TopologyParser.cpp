//
// Created by sakex on 13.11.19.
//

#include "TopologyParser.h"

NeuralNetwork::Topology TopologyParser::parse(nlohmann::json &j) {
    using namespace NeuralNetwork;
    Topology topology;
    int max_layers = 0;

    std::vector<Phenotype *> new_phenotypes;
    for (auto &it : j["phenotypes"]) {
        Phenotype::point input = {it["input"][0], it["input"][1]};
        Phenotype::point output = {it["output"][0], it["output"][1]};
        if (output[0] > max_layers)
            max_layers = output[0];
        double const input_weight = it["input_weight"];
        double const memory_weight = it["memory_weight"];
        double const reset_input_weight = it["reset_input_weight"];
        double const update_input_weight = it["update_input_weight"];
        double const reset_memory_weight = it["reset_memory_weight"];
        double const update_memory_weight = it["update_memory_weight"];
        bool disabled = it["disabled"];
        new_phenotypes.push_back(
                new Phenotype(input, output, input_weight, memory_weight, reset_input_weight, update_input_weight,
                              reset_memory_weight,
                              update_memory_weight, disabled, 0));
    }
    topology.set_layers(max_layers + 1);
    for (Phenotype *phen_ptr : new_phenotypes) {
        topology.add_relationship(phen_ptr, true);
    }
    for (auto &it : j["biases"]) {
        auto &it_neuron = it["neuron"];
        auto &it_bias = it["bias"];
        std::array<int, 2> neuron = {it_neuron[0], it_neuron[1]};
        Bias const bias{
                it_bias["bias_input"],
                it_bias["bias_update"],
                it_bias["bias_reset"]
        };
        topology.set_bias(neuron, bias);
    }
    return topology;
}