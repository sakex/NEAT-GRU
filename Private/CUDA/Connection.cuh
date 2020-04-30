/*
 * Connection.cuh
 *
 *  Created on: November 13, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_CONNECTION_H_
#define NEURALNETWORK_CONNECTION_H_

#include "routines.cuh"
#include "Neuron.cuh"

namespace NeuralNetwork {

    class Neuron;

    class Connection {

    public:
        Connection(double, double, double, double, double, double, Neuron *);

        virtual ~Connection();

        __device__ void activate(double);

    private:
        double * memory;
        double * prev_input;
        double * input_weight;
        double * memory_weight;
        double * reset_input_weight;
        double * reset_memory_weight;
        double * update_input_weight;
        double * update_memory_weight;
        Neuron *output;
    };

}


#endif /* NEURALNETWORK_CONNECTION_H_ */
