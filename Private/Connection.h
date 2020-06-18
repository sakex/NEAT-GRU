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
        Connection(float, float, float, float, float, float, Neuron *);

        ~Connection() = default;

        void activate(float);

        void reset_state();

    private:
        float memory = 0.;
        float prev_input = 0.;
        float const input_weight;
        float const memory_weight;
        float const reset_input_weight;
        float const reset_memory_weight;
        float const update_input_weight;
        float const update_memory_weight;
        Neuron *output;
    };

}


#endif /* NEURALNETWORK_CONNECTION_H_ */
