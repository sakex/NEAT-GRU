/*
 * NN.cuh
 *
 *  Created on: November 13, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NN_CUH_
#define NEURALNETWORK_NN_CUH_

#include <vector>
#include <cuda.h>

#include "Connection.cuh"
#include "Neuron.cuh"
#include "routines.cuh"
#include "neat/Topology/Topology.h"

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

#endif /* NEURALNETWORK_NN_CUH_ */
