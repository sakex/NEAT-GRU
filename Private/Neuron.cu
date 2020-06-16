//
// Created by alexandre on 16.06.20.
//

#include "Neuron.cuh"

namespace NeuralNetwork {
    __device__ __host__ inline double sigmoid(double const value) {
        return 1 / (1 + std::exp(-value));
    }

    __device__ void
    Neuron::add_connection(Neuron *neuron, double const input_weight, double const memory_weight, double const riw,
                           double const rmw,
                           double const uiw, double const umw) {
        connections[last_connection_added++].init(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
        __syncthreads();
    }

    __device__ void Neuron::set_connections_count(size_t const value) {
        connections = (Connection *) malloc(sizeof(Connection) * value);
        __syncthreads();
    }

    __device__ __host__ void Neuron::increment_input(const double inc_value) {
        input += inc_value;
        activated = true;
    }

    __device__ __host__ void Neuron::increment_update(const double inc_value) {
        update += inc_value;
    }

    __device__ __host__ void Neuron::increment_memory(const double inc_value) {
        memory += inc_value;
    }

    __device__ __host__ void Neuron::increment_reset(const double inc_value) {
        reset += inc_value;
    }

    __device__ __host__  void Neuron::set_value(double new_value) {
        input = new_value;
    }

    __device__ __host__ double Neuron::get_prev_reset() const {
        return prev_reset;
    }

    __device__ __host__ void Neuron::feed_forward() {
        if (!activated) return;
        const double update_gate = sigmoid(update);
        const double reset_gate = sigmoid(reset);
        const double current_memory = std::tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1. - update_gate) * current_memory;
        for (size_t it = 0; it < last_connection_added; ++it) {
            connections[it].activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }

    __device__ __host__ void Neuron::reset_value() {
        input = 0.;
        update = 0.;
        memory = 0.;
        activated = false;
    }

    __device__ void Neuron::set_input_value(double new_value) {
        input = new_value;
        activated = true;
        __syncthreads();
    }

    __device__ double Neuron::get_value() {
        if (!activated) return 0;
        const double update_gate = sigmoid(update);
        const double reset_gate = sigmoid(reset);
        const double current_memory = std::tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1. - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return std::tanh(value);
    }
}