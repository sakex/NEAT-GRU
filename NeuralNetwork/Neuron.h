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

        void add_connection(Neuron *, float, float, float, float, float, float);

        void increment_state(float mem, float inp, float res, float upd);

        void set_input_value(float new_value);

        float get_value();

        float get_prev_reset() const;

        void feed_forward();

        void reset_state();

        void set_connections_count(int count);

        void set_bias(Bias);

    private:
        float input = 0.f;
        float memory = 0.f;
        float update = 0.f;
        float reset = 0.f;
        float prev_reset = 0.f;
        int last_added = 0;
        float bias_input = 0.f;
        float bias_update = 0.f;
        float bias_reset = 0.f;
        Connection *connections{nullptr};
        bool activated = false;

    private:
        void reset_value();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NEURON_H_ */
