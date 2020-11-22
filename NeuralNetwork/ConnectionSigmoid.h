//
// Created by alexandre on 10.09.20.
//

#ifndef NEAT_GRU_CONNECTIONSIGMOID_H
#define NEAT_GRU_CONNECTIONSIGMOID_H

#include "Neuron.h"

namespace NeuralNetwork {

    class Neuron;

    class ConnectionSigmoid {

    public:
        ConnectionSigmoid() = default;

        void init(double, Neuron *);

        ~ConnectionSigmoid() = default;

        void activate(double);

        bool operator==(ConnectionSigmoid const &) const;

    private:
        double weight = 0.;
        Neuron *output{nullptr};
    };

}

#endif //NEAT_GRU_CONNECTIONSIGMOID_H
