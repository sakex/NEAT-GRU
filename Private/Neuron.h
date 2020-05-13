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

        virtual ~Neuron();

        Connection *add_connection(Neuron *, double, double, double, double, double, double);

        void increment_input(double);

        void increment_update(double);

        void increment_memory(double);

        void increment_reset(double);

        void set_value(double new_value);

        void set_input_value(double new_value);

        double get_value();

        double get_prev_reset() const;

        void feed_forward();

    private:
        bool activated = false;
        double input = 0;
        double memory = 0;
        double update = 0;
        double reset = 0;
        double prev_reset = 0;
        std::vector<Connection *> connections;

    private:
        void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_H_ */
