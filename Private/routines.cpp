//
// Created by sakex on 07/10/2019.
//

#include "routines.h"

namespace NeuralNetwork {
    double sigmoid(double const value) {
        return 1 / (1 + std::exp(-value));
    }

    std::vector<double> softmax(std::vector<double> &values) {
        double total = 0;
        for (double &val: values) {
            if (val < 0) val = 0;
            else {
                total += val;
            }
        }
        if (total > 1) {
            for (double &val: values) {
                val /= total;
            }
        }
        return values;
    }
}