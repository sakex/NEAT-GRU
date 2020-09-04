//
// Created by alexandre on 16.06.20.
//

#ifndef CUDA_NEURON_CUH
#define CUDA_NEURON_CUH

#include "Connection.cuh"

namespace NeuralNetworkCuda {
    class Connection;

    class Neuron {
    public:
        __device__ void add_connection(Neuron *, double, double, double, double, double, double);


        __device__ void increment_state(double mem, double inp, double res, double upd);


        __device__ double get_prev_reset() const;

        __device__ void reset_value();

        __device__ void set_connections_count(size_t);

        __device__ void set_input_value(double new_value);

        __device__ double get_value();

        __device__ void free_connections();

        __device__ void init();

        __device__ void reset_state();

        __device__ void feed_forward();

    public:
        Connection *connections;
        size_t last_connection_added;

    private:
        double input;
        double memory;
        double update;
        double reset;
        double prev_reset;
    };
}

#endif //CUDA_NEURON_CUH
