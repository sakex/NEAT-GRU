/*

 * Neuron.h
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NEURON_H_
#define NEURALNETWORK_NEURON_H_

#include <vector>
#include <iostream>
#include <cmath>

#include "Connection.h"

namespace NeuralNetwork {

    class Connection;

    class Neuron {
    public:
        Neuron();

        virtual ~Neuron() = default;

        void add_connection(Neuron *, float, float, float, float, float, float);

        void increment_input(float);

        void increment_update(float);

        void increment_memory(float);

        void increment_reset(float);

        void set_value(float new_value);

        void set_input_value(float new_value);

        float get_value();

        float get_prev_reset() const;

        void feed_forward();

        void reset_state();

    private:
        bool activated = false;
        float input = 0.;
        float memory = 0.;
        float update = 0.;
        float reset = 0.;
        float prev_reset = 0.;
        std::vector<Connection> connections;

    private:
        void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_H_ */
