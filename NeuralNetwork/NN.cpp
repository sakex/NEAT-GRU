/*
 * NN.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */


#include "Neuron.h"
#include "../bindings/bindings.h"
#include "NN.h"

#include <cmath>

namespace NeuralNetwork {
    inline float fast_sigmoid(float const value) {
        return value / (1.f + std::abs(value));
    }

    inline float fast_tanh(float const x) {
        if (std::abs(x) >= 4.97) {
            float const values[2] = {-1., 1.};
            return values[x > 0.];
        }
        float const x2 = x * x;
        float const a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float const b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }

    inline void
    Connection::init(float const _input_weight, float const _memory_weight, float const riw, float const uiw,
                     float const rmw, float const umw, Neuron
                     *_output) {
        memory = 0.f;
        prev_input = 0.f;
        input_weight = _input_weight;
        memory_weight = _memory_weight;
        reset_input_weight = riw;
        update_input_weight = uiw;
        reset_memory_weight = rmw;
        update_memory_weight = umw;
        output = _output;
    }

    inline void Connection::activate(float const value) {
        float const prev_reset = output->get_prev_reset();
        memory = fast_tanh(prev_input * input_weight + memory_weight * prev_reset * memory);
        prev_input = value;

        float const update_mem = memory * memory_weight;
        output->increment_state(update_mem,
                                value * input_weight,
                                value * reset_input_weight + memory * reset_memory_weight,
                                value * update_input_weight + memory * update_memory_weight);
    }

    inline void Connection::reset_state() {
        memory = 0.f;
        prev_input = 0.f;
    }


} /* namespace NeuralNetwork */


namespace NeuralNetwork {

    Neuron::Neuron() :
            input(0.f),
            memory(0.f),
            update(0.f),
            reset(0.f),
            prev_reset(0.f),
            last_added(0),
            connections{nullptr} {
    }

    Neuron::~Neuron() {
        delete[] connections;
    }

    void
    Neuron::add_connection(Neuron *neuron, float const input_weight, float const memory_weight, float const riw,
                           float const uiw, float const rmw, float const umw) {
        connections[last_added++].init(input_weight, memory_weight, riw, uiw, rmw, umw, neuron);
    }

    inline void Neuron::increment_state(float const mem, float const inp, float const res, float const upd) {
        memory += mem;
        input += inp;
        reset += res;
        update += upd;
    }

    inline void Neuron::set_input_value(float new_value) {
        input = new_value;
        update = -10.f;
        reset = -10.f;
    }

    inline float Neuron::get_value() {
        float const update_gate = fast_sigmoid(update);
        float const reset_gate = fast_sigmoid(reset);

        const float current_memory = fast_tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1.f - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return fast_tanh(value);
    }

    inline float Neuron::get_prev_reset() const {
        return prev_reset;
    }

    inline void Neuron::reset_value() {
        input = bias_input;
        update = bias_update;
        reset = bias_reset;
        memory = 0.f;
    }

    inline void Neuron::set_bias(Bias bias) {
        bias_input = bias.bias_input;
        bias_update = bias.bias_update;
        bias_reset = bias.bias_reset;
        input = bias_input;
        update = bias_update;
        reset = bias_reset;
    }

    inline void Neuron::feed_forward() {
        float const update_gate = fast_sigmoid(update);
        float const reset_gate = fast_sigmoid(reset);

        const float current_memory = fast_tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1.f - update_gate) * current_memory;
        for (int i = 0; i < last_added; ++i) {
            connections[i].activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }


    inline void Neuron::reset_state() {
        reset_value();
        prev_reset = 0.;
        for (int i = 0; i < last_added; ++i) {
            connections[i].reset_state();
        }
    }

    inline void Neuron::set_connections_count(int count) {
        connections = new Connection[count]();
    }
}

namespace NeuralNetwork {
    /*inline void softmax(float *input, unsigned size) {
        double total = 0;
        for (unsigned i = 0; i < size; ++i) {
            input[i] = static_cast<float>(exp(static_cast<double>(input[i])));
            total += input[i];
        }
        for (unsigned i = 0; i < size; ++i) {
            input[i] /= static_cast<float>(total);
        }
    }*/

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
        layers = new Neuron[neurons_count]();
        Topology::relationships_map &relationships = topology->get_relationships();
        for (auto &it : relationships) {
            Neuron *input_neuron_ptr = &layers[layer_addresses[it.first[0]] + it.first[1]];
            input_neuron_ptr->set_bias(it.second.bias);
            input_neuron_ptr->set_connections_count(it.second.phenotypes.size());
            for (Phenotype *phenotype : it.second.phenotypes) {
                Phenotype::point output = phenotype->get_output();
                float const input_weight = phenotype->get_input_weight();
                float const update_input_weight = phenotype->get_update_input_weight();
                float const memory_weight = phenotype->get_memory_weight();
                float const reset_input_weight = phenotype->get_reset_input_weight();
                float const reset_memory_weight = phenotype->get_reset_memory_weight();
                float const update_memory_weight = phenotype->get_update_memory_weight();

                Neuron *output_neuron_ptr = &layers[layer_addresses[output[0]] + output[1]];
                input_neuron_ptr->add_connection(output_neuron_ptr, input_weight, memory_weight, reset_input_weight,
                                                 update_input_weight, reset_memory_weight, update_memory_weight);
            }
        }
        std::vector<Bias> const &output_bias_vec = topology->get_output_bias();
        for (int it = neurons_count - output_size; it < neurons_count; ++it) {
            layers[it].set_bias(output_bias_vec[it - neurons_count + output_size]);
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
        for (int it = neurons_count - output_size; it < neurons_count; ++it) {
            out[it - neurons_count + output_size] = layers[it].get_value();
        }
        // softmax(out, output_size);
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