//
// Created by sakex on 07/10/2019.
//

#include "routines.h"

namespace NeuralNetwork {
    double sigmoid(double const value) {
        return 1 / (1 + std::exp(-value));
    }

    void softmax(double *input, unsigned size) {
        double total = 0;
        for (unsigned i = 0; i < size; ++i) {
            if (input[i] < 0.) input[i] = 0.;
            else total += input[i];
        }
        if (total > 1) {
            for (unsigned i = 0; i < size; ++i) {
                input[i] /= total;
            }
        }
    }

}