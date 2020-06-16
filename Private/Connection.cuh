//
// Created by alexandre on 16.06.20.
//

#ifndef CUDA_CONNECTION_CUH
#define CUDA_CONNECTION_CUH

#include "Neuron.cuh"
#include <stdio.h>

namespace NeuralNetwork {
    class Neuron;

    class Connection {

    public:
        __device__ __host__ Connection() = default;

        __device__ __host__ Connection(double _input_weight, double _memory_weight, double riw, double rmw,
                                       double uiw, double umw, Neuron *output);

        __device__ __host__ void init(double _input_weight, double _memory_weight, double riw, double rmw,
                                      double uiw, double umw, Neuron *output);

        __host__ __device__ void activate(double value);

    private:
        double memory = 0.;
        double prev_input = 0.;
        double input_weight;
        double memory_weight;
        double reset_input_weight;
        double reset_memory_weight;
        double update_input_weight;
        double update_memory_weight;
        Neuron *output;
    };
}
#endif //CUDA_CONNECTION_CUH
