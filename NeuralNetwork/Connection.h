/*
 * Connection.h
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_CONNECTION_H_
#define NEURALNETWORK_CONNECTION_H_

#include "Neuron.h"

namespace NeuralNetwork {

    class Neuron;

    class Connection {

    public:
        Connection() = default;

        void init(double, double, double, double, double, double, Neuron *);

        ~Connection() = default;

        void activate(double);

        void reset_state();

    private:
        double memory = 0.f;
        double prev_input = 0.f;
        double input_weight = 0.f;
        double memory_weight = 0.f;
        double reset_input_weight = 0.f;
        double update_input_weight = 0.f;
        double reset_memory_weight = 0.f;
        double update_memory_weight = 0.f;
        Neuron *output;
    };

}


#endif /* NEURALNETWORK_CONNECTION_H_ */
