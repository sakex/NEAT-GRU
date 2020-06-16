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

        __device__ void add_connection(Neuron *, double, double, double, double, double, double);


        __device__ __host__ void increment_input(double inc_value);

        __device__ __host__ void increment_update(double inc_value);

        __device__ __host__ void increment_memory(double inc_value);

        __device__ __host__ void increment_reset(double inc_value);

        __device__ __host__  void set_value(double new_value);

        __device__ __host__ double get_prev_reset() const;

        __device__ __host__ void feed_forward();

        __device__ __host__ void reset_value();

        __device__ void set_connections_count(size_t);

        __device__ void set_input_value(double new_value);

        __device__ double get_value();

    private:
        bool activated = false;
        double input = 0.;
        double memory = 0.;
        double update = 0.;
        double reset = 0.;
        double prev_reset = 0.;
        size_t last_connection_added = 0;
        Connection *connections;
    };
}

#endif //CUDA_NEURON_CUH
