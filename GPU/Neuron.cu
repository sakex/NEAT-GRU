//
// Created by alexandre on 16.06.20.
//

#include "Neuron.cuh"

namespace NeuralNetwork {
    __device__ inline float sigmoid(float const value) {
        return 1 / (1 + std::exp(-value));
    }

    __device__ void
    Neuron::add_connection(Neuron *neuron, float const input_weight, float const memory_weight, float const riw,
                           float const rmw,
                           float const uiw, float const umw) {
        Connection *co = &connections[last_connection_added++];
        co->init(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
        __syncthreads();
    }

    __device__ void Neuron::init() {
        connections = nullptr;
        activated = false;
        input = 0.;
        memory = 0.;
        update = 0.;
        reset = 0.;
        prev_reset = 0.;
        last_connection_added = 0;
    }

    __device__ void Neuron::set_connections_count(size_t const value) {
        connections = new Connection[value];
        __syncthreads();
    }

    __device__ void Neuron::increment_input(const float inc_value) {
        atomicAdd(&input, inc_value);
        activated = true;
        __syncthreads();
    }

    __device__ void Neuron::increment_update(const float inc_value) {
        atomicAdd(&update, inc_value);
        __syncthreads();
    }

    __device__ void Neuron::increment_memory(const float inc_value) {
        atomicAdd(&memory, inc_value);
    }

    __device__ void Neuron::increment_reset(const float inc_value) {
        atomicAdd(&reset, inc_value);
    }

    __device__  void Neuron::set_value(float new_value) {
        input = new_value;
    }

    __device__ float Neuron::get_prev_reset() const {
        return prev_reset;
    }

    __device__ void Neuron::feed_forward() {
        if (!activated) return;
        const float update_gate = sigmoid(update);
        const float reset_gate = sigmoid(reset);
        const float current_memory = std::tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1. - update_gate) * current_memory;
        for (size_t it = 0; it < last_connection_added; ++it) {
            connections[it].activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }

    __device__ void Neuron::reset_value() {
        input = 0.;
        update = 0.;
        memory = 0.;
        activated = false;
    }

    __device__ void Neuron::set_input_value(float new_value) {
        input = new_value;
        activated = true;
        __syncthreads();
    }

    __device__ float Neuron::get_value() {
        if (!activated) return 0;
        const float update_gate = sigmoid(update);
        const float reset_gate = sigmoid(reset);
        const float current_memory = std::tanh(input + memory * reset_gate);
        const float value = update_gate * memory + (1. - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return std::tanh(value);
    }

    __device__ void Neuron::free_connections() {
        delete []connections;
        connections = nullptr;
        __syncthreads();
    }
}