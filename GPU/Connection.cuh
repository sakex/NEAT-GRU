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
        __device__ Connection() = default;

        __device__ Connection(float _input_weight, float _memory_weight, float riw, float rmw,
                                       float uiw, float umw, Neuron *output);

        __device__ void init(float _input_weight, float _memory_weight, float riw, float rmw,
                                      float uiw, float umw, Neuron *output);

        __device__ void activate(float value);

    private:
        float memory = 0.;
        float prev_input = 0.;
        float input_weight;
        float memory_weight;
        float reset_input_weight;
        float reset_memory_weight;
        float update_input_weight;
        float update_memory_weight;
        Neuron *output;
    };
}
#endif //CUDA_CONNECTION_CUH
