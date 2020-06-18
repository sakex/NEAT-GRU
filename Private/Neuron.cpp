/*
 * Neuron.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "Neuron.h"

namespace NeuralNetwork {

    inline float sigmoid(float const value) {
        return value / (1 + std::abs(value));
    }

    Neuron::Neuron() : input(0.0) {
    }

    void
    Neuron::add_connection(Neuron *neuron, float const input_weight, float const memory_weight, float const riw,
                           float const rmw,
                           float const uiw, float const umw) {
        connections.emplace_back(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
    }

    void Neuron::increment_input(const float inc_value) {
        input += inc_value;
        activated = true;
    }

    void Neuron::increment_update(const float inc_value) {
        update += inc_value;
    }

    void Neuron::increment_memory(const float inc_value) {
        memory += inc_value;
    }

    void Neuron::increment_reset(const float inc_value) {
        reset += inc_value;
    }

    void Neuron::set_value(float new_value) {
        input = new_value;
    }

    void Neuron::set_input_value(float new_value) {
        input = new_value;
        activated = true;
    }

    float Neuron::get_value() {
        if (!activated) return 0;
        const float update_gate = sigmoid(update);
        const float reset_gate = sigmoid(reset);
        const float current_memory = std::tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1. - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return std::tanh(value);
    }

    float Neuron::get_prev_reset() const {
        return prev_reset;
    }

    void Neuron::reset_value() {
        input = 0.;
        update = 0.;
        memory = 0.;
        activated = false;
    }

    void Neuron::feed_forward() {
        if (!activated) return;
        const float update_gate = sigmoid(update);
        const float reset_gate = sigmoid(reset);
        const float current_memory = std::tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1. - update_gate) * current_memory;
        for (Connection &connection : connections) {
            connection.activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }


    void Neuron::reset_state() {
        reset_value();
        reset = 0.;
        prev_reset = 0.;
        for (Connection &connection : connections) {
            connection.reset_state();
        }
    }
} /* namespace NeuralNetwork */
