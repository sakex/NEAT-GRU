/*
 * Connection.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "Connection.cuh"

namespace NeuralNetwork {

    Connection::Connection(double const _input_weight, double const _memory_weight, double const riw, double const rmw,
                           double const uiw, double const umw, Neuron *output) :
            input_weight{new double(_input_weight)},
            memory_weight(new double(_memory_weight)),
            reset_input_weight(new double (riw)),
            reset_memory_weight(new double (rmw)),
            update_input_weight(new double (uiw)),
            update_memory_weight(new double (umw)),
            memory(new double(0)),
            prev_input(new double(0)),
            output{output} {
        cudaMallocManaged(&memory, 1);
        cudaMallocManaged(&prev_input, 1);
        cudaMallocManaged(&input_weight, 1);
        cudaMallocManaged(&memory_weight, 1);
        cudaMallocManaged(&reset_input_weight, 1);
        cudaMallocManaged(&reset_memory_weight, 1);
        cudaMallocManaged(&update_memory_weight, 1);
        cudaDeviceSynchronize();
    }

    Connection::~Connection() {
        cudaFree(memory);
        cudaFree(prev_input);
        cudaFree(input_weight);
        cudaFree(memory_weight);
        cudaFree(reset_input_weight);
        cudaFree(reset_memory_weight);
        cudaFree(update_input_weight);
        cudaFree(update_memory_weight);
        cudaDeviceSynchronize();
    }

    __device__ void Connection::activate(double const value) {
        double const prev_reset = output->get_prev_reset();
        *memory = (*prev_input) * (*input_weight) + (*memory_weight) * (prev_reset) * (*memory);
        *prev_input = value;
        output->increment_memory((*memory) * (*memory_weight));
        output->increment_input(value * *input_weight);
        output->increment_reset(value * *reset_input_weight + *memory * *reset_memory_weight);
        output->increment_update(value * *update_memory_weight + *memory * *update_memory_weight);
    }


} /* namespace NeuralNetwork */
