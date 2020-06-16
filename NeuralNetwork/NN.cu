//
// Created by alexandre on 16.06.20.
//

#include "NN.cuh"

namespace NeuralNetwork {

    double sigmoid(double const value) {
        return 1 / (1 + std::exp(-value));
    }

    void softmax(double *input, unsigned size) {
        double total = 0;
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

    NN::NN() : layers(nullptr), layer_count(0) {
    }

    __host__ NN::NN(Topology_ptr const &topology) : layers(nullptr), layer_count(0) {
        init_topology(topology);
    }

    NN::~NN() {
        delete_layers();
    }

    void NN::delete_layers() {
        delete layers;
    }

    __host__ void NN::init_topology(Topology_ptr const &topology) {
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

    double * NN::compute(const double *inputs_array) {
        set_inputs(inputs_array);
        for (int it = 0; it < layer_count - 1; ++it) {
            layers[it].feed_forward();
        }
        Layer &last_layer = layers[layer_count - 1];
        double * output = last_layer.get_result();
        softmax(output, last_layer.size());
        return output;
    }

    void NN::reset_state() {
        /*for (int it = 0; it < layer_count; ++it) {
            for (size_t j = 0; j < layers[it].size(); ++j) {
                layers[it][j]->reset_state();
            }
        }*/
    }

    void NN::set_inputs(const double *inputs_array) {
        Layer &first_layer = layers[0];
        first_layer.set_input(inputs_array);
    }

} /* namespace NeuralNetwork */
