/*
 * NN.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */


#include "../Private/Neuron.h"
#include "../bindings/bindings.h"
#include "NN.h"

namespace NeuralNetwork {
    inline float sigmoid(float const value) {
        return value / (1.f + std::abs(value));
    }

    inline Connection::Connection(float const _input_weight, float const _memory_weight, float const riw, float const rmw,
                           float const uiw, float const umw, Neuron *output) :
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            reset_memory_weight(rmw),
            update_input_weight(uiw),
            update_memory_weight(umw),
            output{output} {
    }

    inline void Connection::activate(float const value) {
        float const prev_reset = output->get_prev_reset();
        memory = prev_input * input_weight + memory_weight * prev_reset * memory;
        prev_input = value;
        // std::cout << "FIRST: " << memory * memory_weight << " " << value * input_weight << std::endl;

        output->increment_state(memory * memory_weight,
                              value * input_weight,
                              value * reset_input_weight + memory * reset_memory_weight,
                              value * update_memory_weight + memory * update_memory_weight);
    }

    inline void Connection::reset_state() {
        memory = 0.f;
        prev_input = 0.f;
    }


} /* namespace NeuralNetwork */


namespace NeuralNetwork {

    Neuron::Neuron() : input(0.0) {
    }

    void
    Neuron::add_connection(Neuron *neuron, float const input_weight, float const memory_weight, float const riw,
                           float const rmw,
                           float const uiw, float const umw) {
        connections.emplace_back(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
    }

    /*inline void Neuron::increment_state(xmm const && new_values) {
        xmm registers = {memory, input, reset, update};
        registers.simd = _mm_add_ss(new_values.simd, registers.simd);
        memory = registers.data[0];
        input = registers.data[1];
        reset = registers.data[2];
        update = registers.data[3];
    }*/

    inline void Neuron::increment_state(float const mem, float const inp, float const res, float const upd) {
        memory += mem;
        input += inp;
        reset += res;
        update += upd;
    }


    inline void Neuron::set_value(float new_value) {
        input = new_value;
    }

    inline void Neuron::set_input_value(float new_value) {
        input = new_value;
    }

    inline float Neuron::get_value() {
        float const update_gate = sigmoid(std::abs(update));
        float const reset_gate = sigmoid(std::abs(reset));

        const float current_memory = std::tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1.f - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return std::tanh(value);
    }

    inline float Neuron::get_prev_reset() const {
        return prev_reset;
    }

    inline void Neuron::reset_value() {
        input = 0.f;
        update = 0.f;
        memory = 0.f;
    }

    inline void Neuron::feed_forward() {
        float const update_gate = sigmoid(std::abs(update));
        float const reset_gate = sigmoid(std::abs(reset));

        const float current_memory = std::tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1.f - update_gate) * current_memory;
        for (Connection &connection : connections) {
            connection.activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }


    inline void Neuron::reset_state() {
        reset_value();
        reset = 0.;
        prev_reset = 0.;
        for (Connection &connection : connections) {
            connection.reset_state();
        }
    }
}

namespace NeuralNetwork {
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
               layer_count(0),
               input_size(0),
               output_size(0),
               layers{nullptr} {
    }

    NN::NN(Topology_ptr const &topology) : neurons_count(0),
                                           layer_count(0),
                                           input_size(0),
                                           output_size(0),
                                           layers{nullptr} {
        init_topology(topology);
    }

    NN::~NN() {
        delete_layers();
    }

    inline void NN::delete_layers() {
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

    inline float *NN::compute(const float *inputs_vector) {
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

    inline void NN::set_inputs(const float *inputs_array) {
        for (int i = 0; i < input_size; ++i) {
            layers[i].set_input_value(inputs_array[i]);
        }
    }


} /* namespace NeuralNetwork */

float *compute_network(NN *net, const float *inputs) {
    float *outputs = net->compute(inputs);
    return outputs;
}

void reset_network_state(NN *net) {
    net->reset_state();
}