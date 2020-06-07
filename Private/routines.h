/*
 * routines.h
 *
 *  Created on: August 15, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_NEURALNETWORK_H_
#define NEURALNETWORK_NEURALNETWORK_H_

#include <cmath>
#include <exception>
#include <unordered_map>
#include <mutex>
#include "../NeuralNetwork/Topology.h"
#include <vector>

namespace NeuralNetwork {
    double sigmoid(double);

    /**
     * Transforms a double c array into its softmax
     * @param input The input array
     * @param size The number of doubles in the array
     */
    void softmax(double * input, unsigned size);

    struct NoLayer : public std::exception {
        const char *what() const noexcept override {
            return "Layers should be superior than 0 when calling NeuralNetwork::Topology::add_relationship\"";
        }
    };

}
#endif /* NEURALNETWORK_NEURALNETWORK_H_ */
