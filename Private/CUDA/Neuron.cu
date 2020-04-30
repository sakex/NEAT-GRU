/*
 * Neuron.cu
 *
 *  Created on: November 13, 2019
 *      Author: sakex
 */

#include "Neuron.cuh"

namespace NeuralNetwork {

    Neuron::Neuron() :
    activated(new bool(true)),
    input(new double(0)),
    memory(new double(0)),
    update(new double(0)),
    reset(new double(0)),
    prev_reset(new double(0)) {
        cudaMallocManaged(&activated, 1);
        cudaMallocManaged(&input, 1);
        cudaMallocManaged(&memory, 1);
        cudaMallocManaged(&update, 1);
        cudaMallocManaged(&reset, 1);
        cudaMallocManaged(&prev_reset, 1);
        cudaDeviceSynchronize();
    }

    Neuron::~Neuron() {
        for (Connection *connection : connections) {
            delete connection;
        }
        cudaFree(activated);
        cudaFree(input);
        cudaFree(memory);
        cudaFree(update);
        cudaFree(reset);
        cudaFree(prev_reset);
        cudaDeviceSynchronize();
    }

    Connection *
    Neuron::add_connection(Neuron *neuron, double const input_weight, double const memory_weight, double const riw,
                           double const rmw,
                           double const uiw, double const umw) {
        auto *connection = new Connection(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
        connections.push_back(connection);
        connections_arr = connections.data();
        connections_count++;
        return connection;
    }

    __device__ void Neuron::increment_input(const double inc_value) {
        *input += inc_value;
        *activated = true;
    }

    __device__ void Neuron::increment_update(const double inc_value) {
        *update += inc_value;
    }

    __device__ void Neuron::increment_memory(const double inc_value) {
        *memory += inc_value;
    }

    __device__ void Neuron::increment_reset(const double inc_value) {
        *reset += inc_value;
    }

    __host__ __device__ void Neuron::set_value(double const new_value) {
        *input = new_value;
    }

    __host__ __device__ void Neuron::set_input_value(double const new_value) {
        cudaDeviceSynchronize();
        *input = new_value;
        *activated = true;
    }

    __host__ __device__ double Neuron::get_value() const {
        return *input;
    }

    __device__ double Neuron::get_prev_reset() const {
        return *prev_reset;
    }

    __device__ void Neuron::reset_value() {
        *input = 0;
        *update = 0;
        *memory = 0;
        *activated = false;
    }

    __global__ void connection_ff(NeuralNetwork::Connection *connections, double const value, size_t n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        printf("%d\n", idx);
        if (idx < n) {
            connections[idx].activate(value);
        }
    }

    __device__ void Neuron::feed_forward() {
        if (!activated) return;
        const double update_gate = sigmoid(*update);
        const double reset_gate = sigmoid(*reset);
        const double current_memory = std::tanh(*input + *memory * reset_gate);
        const double value = update_gate * *memory + (1 - update_gate) * current_memory;
        connection_ff<<<*connections_count, 32>>>(*connections_arr, value, *connections_count);
        cudaDeviceSynchronize();
        *prev_reset = reset_gate;
        reset_value();
    }

} /* namespace NeuralNetwork */
