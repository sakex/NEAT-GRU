/*
 * routines.cuh
 *
 *  Created on: November 13, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NEURALNETWORK_CUH_
#define NEURALNETWORK_NEURALNETWORK_CUH_

#include <cmath>
#include <exception>
#include "Neuron.cuh"

namespace NeuralNetwork {
    __device__ __host__ double sigmoid(double value);

    std::vector<double> softmax(std::vector<double> &);
}
#endif /* NEURALNETWORK_NEURALNETWORK_CUH_ */
