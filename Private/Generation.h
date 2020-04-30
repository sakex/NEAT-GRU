/*
 * Generation.h
 *
 *  Created on: Sep 1, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_GENERATION_H_
#define NEURALNETWORK_GENERATION_H_

#include <unordered_map>
#include <string>
#include <mutex>
#include "Phenotype.h"

namespace NeuralNetwork {

    class Generation {
    public:
        static long number(Phenotype::coordinate const &);

        static void reset();

    private:
        static long _counter;
        static std::unordered_map<Phenotype::coordinate, long> evolutions;
        static std::mutex mutex;
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_GENERATION_H_ */
