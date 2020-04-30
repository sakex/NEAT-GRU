/*
 * NN.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "NN.h"

namespace NeuralNetwork {

    NN::NN(Topology_ptr &topology) {
        init_topology(topology);
    }

    NN::~NN() {
        for (Layer *layer : layers) {
            for (Neuron *neuron : *layer)
                delete neuron;
            delete layer;
        }
    }

    void NN::init_topology(Topology_ptr &topology) {
        size_t layers_count = topology->get_layers();
        for (size_t it = 0; it < layers_count; ++it) {
            auto *layer = new Layer();
            layers.push_back(layer);
        }
        Topology::relationships_map &relationships = topology->get_relationships();
        for (auto &it : relationships) {
            for (Phenotype *phenotype : it.second) {
                if (phenotype->is_disabled()) {
                    continue;
                }
                Phenotype::point input = phenotype->get_input();
                Phenotype::point output = phenotype->get_output();
                double const input_weight = phenotype->get_input_weight();
                double const memory_weight = phenotype->get_memory_weight();
                double const reset_input_weight = phenotype->get_reset_input_weight();
                double const reset_memory_weight = phenotype->get_reset_memory_weight();
                double const update_input_weight = phenotype->get_update_input_weight();
                double const update_memory_weight = phenotype->get_update_memory_weight();
                Neuron *input_neuron_ptr = merge_neuron(input[0], input[1]);
                Neuron *output_neuron_ptr = merge_neuron(output[0], output[1]);
                input_neuron_ptr->add_connection(output_neuron_ptr, input_weight, memory_weight, reset_input_weight,
                                                 reset_memory_weight, update_input_weight, update_memory_weight);
            }
        }
    }

    Neuron *NN::merge_neuron(size_t const layer, size_t const index) {
        Layer &_layer = *layers[layer];
        if (index < _layer.size()) {
            if (!_layer[index])
                _layer[index] = new Neuron;
            return _layer[index];
        } else {
            auto *neuron = new Neuron;
            _layer.resize(index + 1);
            _layer[index] = neuron;
            return neuron;
        }
    }

    std::vector<double> NN::compute(const double *inputs_vector) {
        set_inputs(inputs_vector);
        for (auto layer = layers.begin(); layer < layers.end() - 1; ++layer) {
            for (Neuron *neuron: **layer) {
                neuron->feed_forward();
            }
        }
        Layer *last_layer = layers.back();
        std::vector<double> values;
        values.reserve(last_layer->size());
        for (Neuron *neuron : *last_layer) {
            values.push_back(neuron->get_value());
            neuron->set_value(0);
        }
        return softmax(values);
    }


    void NN::set_inputs(const double *inputs_array) {
        Layer &first_layer = *layers[0];
        size_t length = first_layer.size();
        for (size_t i = 0; i < length; ++i) {
            Neuron *neuron = first_layer[i];
            neuron->set_input_value(inputs_array[i]);
        }
    }

} /* namespace NeuralNetwork */
