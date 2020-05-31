/*

 * Neuron.cuh
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NEURON_CUH_
#define NEURALNETWORK_NEURON_CUH_

#include <vector>
#include <iostream>
#include <cmath>

#include "Connection.cuh"

namespace NeuralNetwork {

    class Connection;

    class Neuron {
    public:
        Neuron();

        virtual ~Neuron();

        Connection *add_connection(Neuron *, double, double, double, double, double, double);

        __device__ void increment_input(double);

        __device__ void increment_update(double);

        __device__ void increment_memory(double);

        __device__ void increment_reset(double);

        __host__ __device__ void set_value(double new_value);

        __host__ __device__ void set_input_value(double new_value);

        __host__ __device__ double get_value() const;

        __device__ double get_prev_reset() const;

        __device__ void feed_forward();

    private:
        bool * activated;
        double * input;
        double * memory;
        double * update;
        double * reset ;
        double *  prev_reset;
        std::vector<Connection *> connections;
        Connection ** connections_arr;
        size_t * connections_count;

    private:
        __device__ void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_CUH_ */
