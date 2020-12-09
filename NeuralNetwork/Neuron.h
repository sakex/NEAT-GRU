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
        double input;
        double memory;
        double update;
        double reset;
        double prev_reset;
        int last_added_gru;
        int last_added_sigmoid;
        double bias_input{};
        double bias_update{};
        double bias_reset{};
        bool activated;
        ConnectionGru *connections_gru{nullptr};
        ConnectionSigmoid *connections_sigmoid{nullptr};

    private:
        void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_H_ */
