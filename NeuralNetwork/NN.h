/*
 * NN.h
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NN_H_
#define NEURALNETWORK_NN_H_

#include <iostream>

#include "../Private/Connection.h"
#include "Topology.h"

#include "../Private/Neuron.h"

/// Namespace containing the different classes relevant for the neural network
namespace NeuralNetwork {

    class Topology;

    /// Class Neural Network with GRU gates
    class NN {
    public:
        NN();

        /**
         * Constructor with a topology as input
         * @param topology Topology from which we create the Neural Network
         */
        explicit NN(Topology_ptr const &topology);

        virtual ~NN();

        /**
         * Compute the Neural Network with given inputs
         *
         * @param inputs_vector C float array of inputs
         * @return a vector of weights
         */
        float * compute(const float *inputs_vector);

        /**
         * Inits the Network from a topology
         * @param topology The input topology
         */
        void init_topology(Topology_ptr const &topology);

        /// Resets the hidden state to 0
        void reset_state();

    private:
        int neurons_count;
        int layer_count;
        int input_size;
        int output_size;
        Neuron *layers;

    private:
        /**
         * Sets the inputs on the first layer
         * @param inputs_vector Array of floats to initiate the inputs
         */
        void set_inputs(const float *inputs_vector);

        /// Delete data
        void delete_layers();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NN_H_ */
