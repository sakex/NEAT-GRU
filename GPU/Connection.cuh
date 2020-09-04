//
// Created by alexandre on 16.06.20.
//

#ifndef CUDA_CONNECTION_CUH
#define CUDA_CONNECTION_CUH

#include "Neuron.cuh"
#include <stdio.h>

namespace NeuralNetworkCuda {
    class Neuron;

    class Connection {

    public:
        __device__ Connection() = default;

        __device__ void init(double _input_weight, double _memory_weight, double riw, double rmw,
                                      double uiw, double umw, Neuron *output);

        __device__ void activate(double value);

        __device__ inline void reset_state();

    private:
        double memory = 0.;
        double prev_input = 0.;
        double input_weight{};
        double memory_weight{};
        double reset_input_weight{};
        double reset_memory_weight{};
        double update_input_weight{};
        double update_memory_weight{};
        Neuron *output{};
    };
}
#endif //CUDA_CONNECTION_CUH
