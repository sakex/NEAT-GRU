/*
 * Connection.h
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_CONNECTION_H_
#define NEURALNETWORK_CONNECTION_H_

#include "routines.h"
#include "Neuron.h"

namespace NeuralNetwork {

    class Neuron;

    class Connection {

    public:
        Connection(double, double, double, double, double, double, Neuron *);

        ~Connection() = default;

        void activate(double);

    private:
        double memory = 0;
        double prev_input = 0;
        double const input_weight;
        double const memory_weight;
        double const reset_input_weight;
        double const reset_memory_weight;
        double const update_input_weight;
        double const update_memory_weight;
        Neuron *output;
    };

}


#endif /* NEURALNETWORK_CONNECTION_H_ */
