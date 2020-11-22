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

#include "ConnectionGru.h"
#include "ConnectionSigmoid.h"
#include "../Private/Bias.h"


namespace NeuralNetwork {

    class ConnectionGru;
    class ConnectionSigmoid;

    class Neuron {
    public:
        Neuron();

        ~Neuron();

        void add_connection_gru(Neuron *, double, double, double, double, double, double);

        void add_connection_sigmoid(Neuron *, double);

        void increment_state(double mem, double inp, double res, double upd);

        void increment_value(double value);

        void set_input_value(double new_value);

        double get_value();

        [[nodiscard]] double get_prev_reset() const;

        void feed_forward();

        void reset_state();

        void set_connections_count(int sigmoid_count, int gru_count);

        void set_bias(Bias);

        bool operator==(Neuron const &) const;

    private:
        double input = 0.;
        double memory = 0.;
        double update = 0.;
        double reset = 0.;
        double prev_reset = 0.;
        int last_added_gru = 0;
        int last_added_sigmoid = 0;
        double bias_input = 0.;
        double bias_update = 0.;
        double bias_reset = 0.;
        ConnectionGru *connections_gru{nullptr};
        ConnectionSigmoid *connections_sigmoid{nullptr};

    private:
        void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_H_ */
