/*
 * NN.h
 *
 *  Created on: May 30, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NN_H_
#define NEURALNETWORK_NN_H_

#include <iostream>
#include <vector>

#include "../Private/Connection.h"
#include "../Private/routines.h"
#include "../Private/Neuron.h"
#include "Topology.h"

namespace NeuralNetwork {

    class Neuron;

    class Topology;

    class NN {
    public:
        explicit NN(Topology_ptr &topology);

        virtual ~NN();

        std::vector<double> compute(const double *);

    private:
        std::vector<Layer *> layers;

        Neuron *merge_neuron(size_t layer, size_t index);

        void init_topology(Topology_ptr &topology);

        void set_inputs(const double *inputs_vector);
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NN_H_ */
