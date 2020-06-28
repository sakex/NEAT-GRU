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

        void init(float, float, float, float, float, float, Neuron *);

        ~Connection() = default;

        void activate(float);

        void reset_state();

    private:
        float memory = 0.f;
        float prev_input = 0.f;
        float input_weight = 0.f;
        float memory_weight = 0.f;
        float reset_input_weight = 0.f;
        float update_input_weight = 0.f;
        float reset_memory_weight = 0.f;
        float update_memory_weight = 0.f;
        Neuron *output;
    };

}


#endif /* NEURALNETWORK_CONNECTION_H_ */
