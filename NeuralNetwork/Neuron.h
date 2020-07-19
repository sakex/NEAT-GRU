/*

 * Neuron.h
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NEURON_H_
#define NEURALNETWORK_NEURON_H_

#include <iostream>
#include <cmath>

#include "Connection.h"
#include "../Private/Bias.h"


namespace NeuralNetwork {

    class Connection;

    class Neuron {
    public:
        Neuron();

        ~Neuron();

        void add_connection(Neuron *, double, double, double, double, double, double);

        void increment_state(double mem, double inp, double res, double upd);

        void set_input_value(double new_value);

        double get_value();

        double get_prev_reset() const;

        void feed_forward();

        void reset_state();

        void set_connections_count(int count);

        void set_bias(Bias);

    private:
        double input = 0.f;
        double memory = 0.f;
        double update = 0.f;
        double reset = 0.f;
        double prev_reset = 0.f;
        int last_added = 0;
        double bias_input = 0.f;
        double bias_update = 0.f;
        double bias_reset = 0.f;
        Connection *connections{nullptr};

    private:
        void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_H_ */
