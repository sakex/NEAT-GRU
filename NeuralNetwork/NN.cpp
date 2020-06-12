/*
 * NN.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "NN.h"

namespace NeuralNetwork {

    NN::NN() : layers(nullptr), layer_count(0) {
    }

    NN::NN(Topology_ptr const &topology) : layers(nullptr), layer_count(0) {
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
        delete_layers();
        layers = new Layer[layer_count];
        std::vector<int> const &sizes = topology->get_layers_size();
        for (int i = 0; i < layer_count; ++i) {
            layers[i].set_size(sizes[i]);
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
                Neuron *input_neuron_ptr = layers[input[0]][input[1]];
                Neuron *output_neuron_ptr = layers[output[0]][output[1]];
                input_neuron_ptr->add_connection(output_neuron_ptr, input_weight, memory_weight, reset_input_weight,
                                                 reset_memory_weight, update_input_weight, update_memory_weight);
            }
        }
    }

    double * NN::compute(const double *inputs_vector) {
        set_inputs(inputs_vector);
        for (int it = 0; it < layer_count - 1; ++it) {
            for (size_t j = 0; j < layers[it].size(); ++j) {
                layers[it][j]->feed_forward();
            }
        }
        Layer &last_layer = layers[layer_count - 1];
        unsigned const size = last_layer.size();
        double * out = nullptr;
        out = new double[size];
        for (unsigned it = 0; it < size; ++it) {
            Neuron *neuron = last_layer[it];
            out[it] = neuron->get_value();
            neuron->set_value(0);
        }
        softmax(out, size);
        return out;
    }

    void NN::reset_state() {
        for (int it = 0; it < layer_count; ++it) {
            for (size_t j = 0; j < layers[it].size(); ++j) {
                layers[it][j]->reset_state();
            }
        }
    }

    void NN::set_inputs(const double *inputs_array) {
        Layer &first_layer = layers[0];
        size_t length = first_layer.size();
        for (size_t i = 0; i < length; ++i) {
            Neuron *neuron = first_layer[i];
            neuron->set_input_value(inputs_array[i]);
        }
    }

} /* namespace NeuralNetwork */
