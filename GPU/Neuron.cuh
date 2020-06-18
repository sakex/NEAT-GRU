//
// Created by alexandre on 16.06.20.
//

#ifndef CUDA_NEURON_CUH
#define CUDA_NEURON_CUH

#include "Connection.cuh"

namespace NeuralNetwork {
    class Connection;

    class Neuron {
    public:
        __device__ void add_connection(Neuron *, float, float, float, float, float, float);


        __device__ void increment_input(float inc_value);

        __device__ void increment_update(float inc_value);

        __device__ void increment_memory(float inc_value);

        __device__ void increment_reset(float inc_value);

        __device__  void set_value(float new_value);

        __device__ float get_prev_reset() const;

        __device__ void feed_forward();

        __device__ void reset_value();

        __device__ void set_connections_count(size_t);

        __device__ void set_input_value(float new_value);

        __device__ float get_value();

        __device__ void free_connections();

        __device__ void init();

    private:
        bool activated;
        float input;
        float memory;
        float update;
        float reset;
        float prev_reset;
        size_t last_connection_added;
        Connection *connections;
    };
}

#endif //CUDA_NEURON_CUH
