/*
 * NN.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "NN.h"

namespace NeuralNetwork {

    inline float sigmoid(float const value) {
        return value / (1 + std::abs(value));
    }

    inline void softmax(float *input, unsigned size) {
        float total = 0;
        for (unsigned i = 0; i < size; ++i) {
            if (input[i] < 0.) input[i] = 0.;
            else total += input[i];
        }
        if (total > 1) {
            for (unsigned i = 0; i < size; ++i) {
                input[i] /= total;
            }
        }
    }

    NN::NN() : neurons_count(0),
               layers{nullptr},
               input_size(0),
               output_size(0),
               layer_count(0) {
    }

    NN::NN(Topology_ptr const &topology) : neurons_count(0),
                                           layers{nullptr},
                                           input_size(0),
                                           output_size(0),
                                           layer_count(0)  {
        init_topology(topology);
    }

    NN::~NN() {
        delete_layers();
    }

    void NN::delete_layers() {
        delete[] layers;
    }

    void NN::init_topology(Topology_ptr const &topology) {
        layer_count = topology->get_layers();
        std::vector<int> const &sizes = topology->get_layers_size();
        int *layer_addresses = new int[layer_count];
        neurons_count = 0;
        int i = 0;
        for (; i < layer_count; ++i) {
            layer_addresses[i] = neurons_count;
            neurons_count += sizes[i];
        }
        input_size = sizes[0];
        output_size = sizes.back();
        layers = new Neuron[neurons_count];
        Topology::relationships_map &relationships = topology->get_relationships();
        for (auto &it : relationships) {
            for (Phenotype *phenotype : it.second) {
                if (phenotype->is_disabled()) {
                    continue;
                }
                Phenotype::point input = phenotype->get_input();
                Phenotype::point output = phenotype->get_output();
                float const input_weight = phenotype->get_input_weight();
                float const memory_weight = phenotype->get_memory_weight();
                float const reset_input_weight = phenotype->get_reset_input_weight();
                float const reset_memory_weight = phenotype->get_reset_memory_weight();
                float const update_input_weight = phenotype->get_update_input_weight();
                float const update_memory_weight = phenotype->get_update_memory_weight();
                Neuron *input_neuron_ptr = &layers[layer_addresses[input[0]] + input[1]];
                Neuron *output_neuron_ptr = &layers[layer_addresses[output[0]] + output[1]];
                input_neuron_ptr->add_connection(output_neuron_ptr, input_weight, memory_weight, reset_input_weight,
                                                 reset_memory_weight, update_input_weight, update_memory_weight);
            }
        }
        delete[] layer_addresses;
    }

    float *NN::compute(const float *inputs_vector) {
        set_inputs(inputs_vector);
        for (int it = 0; it < neurons_count - output_size; ++it) {
            layers[it].feed_forward();
        }
        float *out;
        out = new float[output_size];
        for (int it = 0; it < output_size; ++it) {
            Neuron *neuron = &layers[neurons_count - output_size + it];
            out[it] = neuron->get_value();
            neuron->set_value(0);
        }
        softmax(out, output_size);
        return out;
    }

    void NN::reset_state() {
        for (int it = 0; it < neurons_count; ++it) {
            layers[it].reset_state();
        }
    }

    void NN::set_inputs(const float *inputs_array) {
        for (int i = 0; i < input_size; ++i) {
            layers[i].set_input_value(inputs_array[i]);
        }
    }

} /* namespace NeuralNetwork */
