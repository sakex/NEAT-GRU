/*
 * Neuron.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "Neuron.h"

namespace NeuralNetwork {

    Neuron::Neuron() : input(0.0) {
    }

    inline void
    Neuron::add_connection(Neuron *neuron, double const input_weight, double const memory_weight, double const riw,
                           double const rmw,
                           double const uiw, double const umw) {
        connections.emplace_back(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
    }

    inline void Neuron::increment_input(const double inc_value) {
        input += inc_value;
        activated = true;
    }

    inline void Neuron::increment_update(const double inc_value) {
        update += inc_value;
    }

    inline void Neuron::increment_memory(const double inc_value) {
        memory += inc_value;
    }

    inline void Neuron::increment_reset(const double inc_value) {
        reset += inc_value;
    }

    inline void Neuron::set_value(double new_value) {
        input = new_value;
    }

    inline void Neuron::set_input_value(double new_value) {
        input = new_value;
        activated = true;
    }

    inline double Neuron::get_value() {
        if (!activated) return 0;
        const double update_gate = sigmoid(update);
        const double reset_gate = sigmoid(reset);
        const double current_memory = std::tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1. - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return std::tanh(value);
    }

    inline double Neuron::get_prev_reset() const {
        return prev_reset;
    }

    inline void Neuron::reset_value() {
        input = 0.;
        update = 0.;
        memory = 0.;
        activated = false;
    }

    void Neuron::feed_forward() {
        if (!activated) return;
        const double update_gate = sigmoid(update);
        const double reset_gate = sigmoid(reset);
        const double current_memory = std::tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1. - update_gate) * current_memory;
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
} /* namespace NeuralNetwork */
