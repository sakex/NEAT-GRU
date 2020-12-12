/*
 * NN.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */


#include "Neuron.h"
#include "../bindings/bindings.h"
#include "NN.h"
#include "ConnectionSigmoid.h"

#include <cmath>

namespace NeuralNetwork {
    inline bool approx_equal(double const a, double const b) {
        double const diff = std::fabs(a - b);
        return diff <= 1e-7;
    }

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
}

namespace NeuralNetwork {
    inline void ConnectionSigmoid::init(double const w, Neuron *_output) {
        weight = w;
        output = _output;
    }

    inline void ConnectionSigmoid::activate(double const value) {
        output->increment_value(value * weight);
    }

    bool ConnectionSigmoid::operator==(ConnectionSigmoid const &other) const {
        return approx_equal(weight, other.weight);
    }

} /* namespace NeuralNetwork */

namespace NeuralNetwork {
    inline void
    ConnectionGru::init(double const _input_weight, double const _memory_weight, double const riw, double const uiw,
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

    inline void ConnectionGru::activate(double const value) {
        double const prev_reset = output->get_prev_reset();
        memory = fast_tanh(prev_input * input_weight + memory_weight * prev_reset * memory);
        prev_input = value;

        double const update_mem = memory * memory_weight;
        output->increment_state(update_mem,
                                value * input_weight,
                                value * reset_input_weight + memory * reset_memory_weight,
                                value * update_input_weight + memory * update_memory_weight);
    }

    inline void ConnectionGru::reset_state() {
        memory = 0.;
        prev_input = 0.;
    }

    bool ConnectionGru::operator==(ConnectionGru const &other) const {
        return approx_equal(memory, other.memory) &&
               approx_equal(prev_input, other.prev_input) &&
               approx_equal(input_weight, other.input_weight) &&
               approx_equal(memory_weight, other.memory_weight) &&
               approx_equal(reset_input_weight, other.reset_input_weight) &&
               approx_equal(update_input_weight, other.update_input_weight) &&
               approx_equal(reset_memory_weight, other.reset_memory_weight) &&
               approx_equal(update_memory_weight, other.update_memory_weight);
    }


} /* namespace NeuralNetwork */


namespace NeuralNetwork {

    Neuron::Neuron() :
            input(0.),
            memory(0.),
            update(0.),
            reset(0.),
            prev_reset(0.),
            last_added_gru(0.),
            last_added_sigmoid(0.),
            connections_gru{nullptr} {
    }

    Neuron::~Neuron() {
        delete[] connections_sigmoid;
        delete[] connections_gru;
    }

    void
    Neuron::add_connection_gru(Neuron *neuron, double const input_weight, double const memory_weight, double const riw,
                               double const uiw, double const rmw, double const umw) {
        connections_gru[last_added_gru++].init(input_weight, memory_weight, riw, uiw, rmw, umw, neuron);
    }

    void
    Neuron::add_connection_sigmoid(Neuron *neuron, double const weight) {
        connections_sigmoid[last_added_sigmoid++].init(weight, neuron);
    }

    inline void Neuron::increment_state(double const mem, double const inp, double const res, double const upd) {
        memory += mem;
        input += inp;
        reset += res;
        update += upd;
    }

    inline void Neuron::increment_value(double const value) {
        input += value;
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
        for (int i = 0; i < last_added_gru; ++i) {
            connections_gru[i].activate(value);
        }
        for (int i = 0; i < last_added_sigmoid; ++i) {
            connections_sigmoid[i].activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }


    inline void Neuron::reset_state() {
        reset_value();
        prev_reset = 0.;
        for (int i = 0; i < last_added_gru; ++i) {
            connections_gru[i].reset_state();
        }
    }

    inline void Neuron::set_connections_count(int sigmoid_count, int gru_count) {
        connections_sigmoid = new ConnectionSigmoid[sigmoid_count]();
        connections_gru = new ConnectionGru[gru_count]();
    }

    bool Neuron::operator==(Neuron const &other) const {
        bool const values_equal = approx_equal(input, other.input) &&
                                  approx_equal(memory, other.memory) &&
                                  approx_equal(update, other.update) &&
                                  approx_equal(reset, other.reset) &&
                                  approx_equal(prev_reset, other.prev_reset) &&
                                  approx_equal(last_added_gru, other.last_added_gru) &&
                                  approx_equal(last_added_sigmoid, other.last_added_sigmoid) &&
                                  approx_equal(bias_input, other.bias_input) &&
                                  approx_equal(bias_update, other.bias_update) &&
                                  approx_equal(bias_reset, other.bias_reset);
        if (!values_equal) return false;
        for (int i = 0; i < last_added_gru; ++i) {
            bool found = false;
            for(int j = 0; j < last_added_gru; ++j) {
                if (connections_gru[i] == other.connections_gru[j]) {
                    found = true;
                    break;
                }
            }
            if(!found) return false;
        }
        for (int i = 0; i < last_added_sigmoid; ++i) {
            bool found = false;
            for(int j = 0; j < last_added_sigmoid; ++j) {
                if (connections_sigmoid[i] == other.connections_sigmoid[j]) {
                    found = true;
                    break;
                }
            }
            if(!found) return false;
        }
        return true;
    }
}

namespace NeuralNetwork {
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

            int type_counts[2] = {0, 0};
            for (Gene *gene : it.second.genes) {
                if(!gene->is_disabled()) {
                    type_counts[gene->get_type()]++;
                }
            }
            input_neuron_ptr->set_connections_count(type_counts[0], type_counts[1]);
            for (Gene *gene : it.second.genes) {
                if(gene->is_disabled()) continue;
                Gene::point output = gene->get_output();
                double const input_weight = gene->get_input_weight();
                double const update_input_weight = gene->get_update_input_weight();
                double const memory_weight = gene->get_memory_weight();
                double const reset_input_weight = gene->get_reset_input_weight();
                double const reset_memory_weight = gene->get_reset_memory_weight();
                double const update_memory_weight = gene->get_update_memory_weight();

                Neuron *output_neuron_ptr = &layers[layer_addresses[output[0]] + output[1]];
                switch (gene->get_type()) {
                    case Sigmoid:
                        input_neuron_ptr->add_connection_sigmoid(output_neuron_ptr, input_weight);
                        break;
                    case GRU:
                        input_neuron_ptr->add_connection_gru(output_neuron_ptr, input_weight, memory_weight,
                                                             reset_input_weight, update_input_weight,
                                                             reset_memory_weight, update_memory_weight);
                        break;
                }
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

    bool NN::operator==(NN const &other) const {
        for (int it = 0; it < neurons_count; ++it) {
            if (!(layers[it] == other.layers[it])) return false;
        }
        return true;
    }

} /* namespace NeuralNetwork */

double *compute_network(NeuralNetwork::NN *net, const double *inputs) {
    double *outputs = net->compute(inputs);
    return outputs;
}

void reset_network_state(NeuralNetwork::NN *net) {
    net->reset_state();
}
