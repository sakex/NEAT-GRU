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
        __device__ __host__ Neuron() = default;

        __device__ void add_connection(Neuron *, float, float, float, float, float, float);


        __device__ void increment_input(float inc_value);

        __device__ void increment_update(float inc_value);

        __device__ void increment_memory(float inc_value);

        __device__ void increment_reset(float inc_value);

        __device__  void set_value(float new_value);

        __device__ float get_prev_reset() const;

        __device__ void feed_forward();

        __device__ __host__ void reset_value();

        __device__ void set_connections_count(size_t);

        __device__ void set_input_value(float new_value);

        __device__ float get_value();

        __device__ void free_connections();

    private:
        bool activated = false;
        float input = 0.;
        float memory = 0.;
        float update = 0.;
        float reset = 0.;
        float prev_reset = 0.;
        size_t last_connection_added = 0;
        Connection *connections;
    };
}

#endif //CUDA_NEURON_CUH
