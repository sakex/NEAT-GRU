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
    inline double fast_sigmoid(double const value) {
        return value / (1.f + std::abs(value));
    }

    inline double fast_tanh(double const x) {
        if (std::abs(x) >= 4.97) {
            double const values[2] = {-1., 1.};
            return values[x > 0.];
        }
        double const x2 = x * x;
        double const a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        double const b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }

    inline void
    Connection::init(double const _input_weight, double const _memory_weight, double const riw, double const uiw,
                     double const rmw, double const umw, Neuron
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

    inline void Connection::activate(double const value) {
        double const prev_reset = output->get_prev_reset();
        memory = prev_input * input_weight + memory_weight * prev_reset * memory;
        prev_input = value;

        double const update_mem = memory * memory_weight;
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
    Neuron::add_connection(Neuron *neuron, double const input_weight, double const memory_weight, double const riw,
                           double const uiw, double const rmw, double const umw) {
        connections[last_added++].init(input_weight, memory_weight, riw, uiw, rmw, umw, neuron);
    }

    inline void Neuron::increment_state(double const mem, double const inp, double const res, double const upd) {
        memory += mem;
        input += inp;
        reset += res;
        update += upd;
    }

    inline void Neuron::set_input_value(double new_value) {
        input = new_value;
        update = -10.f;
        reset = -10.f;
    }

    inline double Neuron::get_value() {
        double const update_gate = fast_sigmoid(update);
        double const reset_gate = fast_sigmoid(reset);

        const double current_memory = fast_tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1.f - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return fast_tanh(value);
    }

    inline double Neuron::get_prev_reset() const {
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
        double const update_gate = fast_sigmoid(update);
        double const reset_gate = fast_sigmoid(reset);

        const double current_memory = fast_tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1.f - update_gate) * current_memory;
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
    /*inline void softmax(double *input, unsigned size) {
        double total = 0;
        for (unsigned i = 0; i < size; ++i) {
            input[i] = static_cast<double>(exp(static_cast<double>(input[i])));
            total += input[i];
        }
        for (unsigned i = 0; i < size; ++i) {
            input[i] /= static_cast<double>(total);
        }
    }*/

    NN::NN() : neurons_count(0),
               layer_count(0),
               input_size(0),
               output_size(0),
               layers{nullptr} {
    }

    NN::NN(Topology &topology) : neurons_count(0),
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

    void NN::init_topology(Topology &topology) {
        layer_count = topology.get_layers();
        std::vector<int> const &sizes = topology.get_layers_size();
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
        Topology::relationships_map &relationships = topology.get_relationships();
        for (auto &it : relationships) {
            Neuron *input_neuron_ptr = &layers[layer_addresses[it.first[0]] + it.first[1]];
            input_neuron_ptr->set_bias(it.second.bias);
            input_neuron_ptr->set_connections_count(it.second.genes.size());
            for (Gene *gene : it.second.genes) {
                Gene::point output = gene->get_output();
                double const input_weight = gene->get_input_weight();
                double const update_input_weight = gene->get_update_input_weight();
                double const memory_weight = gene->get_memory_weight();
                double const reset_input_weight = gene->get_reset_input_weight();
                double const reset_memory_weight = gene->get_reset_memory_weight();
                double const update_memory_weight = gene->get_update_memory_weight();

                Neuron *output_neuron_ptr = &layers[layer_addresses[output[0]] + output[1]];
                input_neuron_ptr->add_connection(output_neuron_ptr, input_weight, memory_weight, reset_input_weight,
                                                 update_input_weight, reset_memory_weight, update_memory_weight);
            }
        }
        std::vector<Bias> const &output_bias_vec = topology.get_output_bias();
        for (int it = neurons_count - output_size; it < neurons_count; ++it) {
            layers[it].set_bias(output_bias_vec[it - neurons_count + output_size]);
        }
        delete[] layer_addresses;
    }

    inline double *NN::compute(const double *inputs_array) {
        set_inputs(inputs_array);
        for (int it = 0; it < neurons_count - output_size; ++it) {
            layers[it].feed_forward();
        }
        double *out;
        out = new double[output_size];
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

    inline void NN::set_inputs(const double *inputs_array) {
        for (int i = 0; i < input_size; ++i) {
            layers[i].set_input_value(inputs_array[i]);
        }
    }


} /* namespace NeuralNetwork */

double *compute_network(NeuralNetwork::NN *net, const double *inputs) {
    double *outputs = net->compute(inputs);
    return outputs;
}

void reset_network_state(NeuralNetwork::NN *net) {
    net->reset_state();
}