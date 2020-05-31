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
#include "../Private/routines.h"
#include "Topology.h"

#include "../Private/Layer.h"
#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace NeuralNetwork {

    class Neuron;

    class Topology;

    class NN {
    public:
        NN();

        explicit NN(Topology_ptr &topology);

        virtual ~NN();

        std::vector<double> compute(const double *);

        void init_topology(Topology_ptr &topology);

        void reset_state();

    private:
        Layer *layers;
        int layer_count;

#if CUDA_ENABLED
        Layer *gpu_layers;
        cudaError_t err;
#endif

    private:
        void set_inputs(const double *inputs_vector);

        void delete_layers();
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_NN_H_ */
