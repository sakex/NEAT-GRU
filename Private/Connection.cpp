/*
 * Connection.cpp
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#include "Connection.h"

namespace NeuralNetwork {

    Connection::Connection(double const _input_weight, double const _memory_weight, double const riw, double const rmw,
                           double const uiw, double const umw, Neuron *output) :
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            reset_memory_weight(rmw),
            update_input_weight(uiw),
            update_memory_weight(umw),
            output{output} {
    }

    void Connection::activate(double const value) {
        double const prev_reset = output->get_prev_reset();
        memory = prev_input * input_weight + memory_weight * prev_reset * memory;
        prev_input = value;
        output->increment_memory(memory * memory_weight);
        output->increment_input(value * input_weight);
        output->increment_reset(value * reset_input_weight + memory * reset_memory_weight);
        output->increment_update(value * update_memory_weight + memory * update_memory_weight);
    }

    void Connection::reset_state() {
        memory = 0.;
        prev_input = 0.;
    }


} /* namespace NeuralNetwork */
